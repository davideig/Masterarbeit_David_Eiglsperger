from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import LearOperationalConfig, RunConfig, RunKind, load_config, validate_config_payload
from ..models.lear import scale_fold_point
from ..pipelines.lear import prepare_lear_operational_dataset


@dataclass(frozen=True)
class DiagnosticSample:
    label: str
    config_path: Path
    forecast_day: pd.Timestamp
    mtu: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Refit sampled LightGBM price models and save feature/group contribution "
            "diagnostics using LightGBM's built-in pred_contrib output."
        )
    )
    parser.add_argument(
        "--config",
        action="append",
        required=True,
        help="Path to a top-level lear_operational run config. Repeat for multiple models.",
    )
    parser.add_argument(
        "--label",
        action="append",
        help="Optional label for each --config. Defaults to the config file stem.",
    )
    parser.add_argument(
        "--output-dir",
        default="output/diagnostics/price_feature_importance",
        help="Directory for diagnostic CSV files.",
    )
    parser.add_argument("--start", help="Optional diagnostic start date, e.g. 2026-02-01.")
    parser.add_argument("--end", help="Optional diagnostic end date, e.g. 2026-07-31.")
    parser.add_argument(
        "--max-days",
        type=int,
        default=12,
        help="Maximum number of forecast days sampled per config.",
    )
    parser.add_argument(
        "--mtu-step",
        type=int,
        default=4,
        help="Sample every Nth MTU. 4 means hourly, 1 means all 96 MTUs.",
    )
    return parser.parse_args()


def _load_lear_config(path: Path) -> LearOperationalConfig:
    run_config = load_config(path, RunConfig)
    if run_config.kind != RunKind.LEAR_OPERATIONAL:
        raise ValueError(f"{path} has kind={run_config.kind!r}, expected lear_operational.")
    if run_config.config_path is not None:
        return load_config(run_config.config_path, LearOperationalConfig)
    return validate_config_payload(
        run_config.config,
        LearOperationalConfig,
        repo_root=run_config.repo_root,
    )


def _sample_days(
    config: LearOperationalConfig,
    X: pd.DataFrame,
    *,
    max_days: int,
    start: str | None,
    end: str | None,
) -> list[pd.Timestamp]:
    sample_start = pd.Timestamp(start, tz=config.target_tz) if start else pd.Timestamp(config.test_start).normalize()
    sample_end = pd.Timestamp(end, tz=config.target_tz) if end else pd.Timestamp(config.test_end).normalize()
    days = pd.date_range(sample_start.normalize(), sample_end.normalize(), freq="D", tz=config.target_tz)
    days = [day for day in days if day in X.index]
    if not days or len(days) <= max_days:
        return days

    indices = np.linspace(0, len(days) - 1, max_days)
    sampled = [days[int(round(idx))] for idx in indices]
    return list(dict.fromkeys(sampled))


def _feature_group(feature: str) -> str:
    name = feature.lower()
    if name.startswith("reserve_") or "_reserve_" in name:
        return "Reserve market"
    if name.startswith("exaa") or "exaa" in name:
        return "EXAA prices"
    if name.startswith("price_") or name.startswith("price_d") or "_price_" in name:
        return "Price lags"
    if name.startswith("weekday_") or name in {"is_holiday", "is_15min_market"}:
        return "Calendar"
    if any(token in name for token in ("month", "weekend", "holiday", "post_regime")):
        return "Calendar"
    if "residual_load" in name or name.startswith("rl_") or "rl_hat" in name:
        return "Residual load forecast"
    if "load" in name:
        return "Load forecast"
    if any(token in name for token in ("renewable", "solar", "wind_proxy", "wind_generation", "ren_")):
        return "Renewable forecast"
    if any(
        token in name
        for token in (
            "ssrd",
            "aswdir",
            "aswdifd",
            "sw_dir",
            "sw_dif",
            "radiation",
            "wind_speed",
            "wind_direction",
            "t2m",
            "d2m",
            "temperature",
            "precip",
            "snow",
            "weather",
            "vmax",
            "u10",
            "v10",
            "pressure",
            "humidity",
        )
    ):
        return "Raw weather"
    return "Other"


def _fit_lgbm_contribs(
    *,
    config: LearOperationalConfig,
    sample: DiagnosticSample,
    X_tr: pd.DataFrame,
    X_te: pd.DataFrame,
    y_tr: pd.Series,
) -> tuple[list[dict], dict]:
    try:
        from lightgbm import LGBMRegressor
    except ImportError as exc:  # pragma: no cover - optional environment
        raise ImportError("Price feature importance requires lightgbm in the active environment.") from exc

    finite_target_mask = np.isfinite(y_tr.to_numpy(dtype=float))
    if finite_target_mask.sum() < 5:
        raise ValueError(
            f"Not enough finite targets for {sample.label} {sample.forecast_day.date()} MTU {sample.mtu}."
        )
    if not finite_target_mask.all():
        X_tr = X_tr.loc[finite_target_mask]
        y_tr = y_tr.loc[finite_target_mask]

    X_tr_s, X_te_s, y_tr_s, _, _ = scale_fold_point(
        X_tr=X_tr,
        X_va=X_te,
        y_tr=y_tr,
        y_va=None,
        use_vst=config.use_vst,
    )

    model = LGBMRegressor(
        n_estimators=config.lgbm_n_estimators,
        learning_rate=config.lgbm_learning_rate,
        num_leaves=config.lgbm_num_leaves,
        min_child_samples=config.lgbm_min_child_samples,
        subsample=config.lgbm_subsample,
        colsample_bytree=config.lgbm_colsample_bytree,
        reg_lambda=config.lgbm_reg_lambda,
        objective="regression",
        random_state=config.random_state,
        n_jobs=1,
        verbosity=-1,
    )
    model.fit(X_tr_s.values, y_tr_s)

    contrib = np.asarray(model.predict(X_te_s.values, pred_contrib=True), dtype=float)
    feature_contrib = contrib[0, :-1]
    expected_value = float(contrib[0, -1])
    booster = model.booster_
    gain = np.asarray(booster.feature_importance(importance_type="gain"), dtype=float)
    split = np.asarray(booster.feature_importance(importance_type="split"), dtype=float)

    raw_row = X_te.iloc[0]
    records = []
    for feature, contribution, gain_value, split_count in zip(X_tr.columns, feature_contrib, gain, split, strict=True):
        group = _feature_group(str(feature))
        try:
            feature_value = float(raw_row[feature])
        except (TypeError, ValueError):
            feature_value = float("nan")
        records.append(
            {
                "model_label": sample.label,
                "config_path": str(sample.config_path),
                "forecast_day": sample.forecast_day.date().isoformat(),
                "mtu": sample.mtu,
                "feature": str(feature),
                "feature_group": group,
                "contribution": float(contribution),
                "abs_contribution": float(abs(contribution)),
                "feature_value": feature_value,
                "gain": float(gain_value),
                "split": int(split_count),
            }
        )

    fit_record = {
        "model_label": sample.label,
        "config_path": str(sample.config_path),
        "forecast_day": sample.forecast_day.date().isoformat(),
        "mtu": sample.mtu,
        "n_train_rows": int(len(X_tr)),
        "n_features": int(X_tr.shape[1]),
        "expected_value": expected_value,
    }
    return records, fit_record


def _run_one_config(
    *,
    config_path: Path,
    label: str,
    output_dir: Path,
    max_days: int,
    mtu_step: int,
    start: str | None,
    end: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = _load_lear_config(config_path)
    if config.price_model_type != "lightgbm":
        raise ValueError(
            f"{config_path} uses price_model_type={config.price_model_type!r}. "
            "This diagnostic currently supports LightGBM price configs."
        )

    print(f"[price-importance] Building dataset for {label} ...")
    dataset = prepare_lear_operational_dataset(config)
    X = dataset["X"]
    Y = dataset["Y"]
    days = _sample_days(config, X, max_days=max_days, start=start, end=end)
    mtus = list(range(0, 96, mtu_step))
    print(f"[price-importance] {label}: {len(days)} sampled days x {len(mtus)} MTUs")

    contribution_records: list[dict] = []
    fit_records: list[dict] = []
    for day in days:
        train_start = day - pd.Timedelta(days=config.train_days_rolling)
        train_end = day - pd.Timedelta(days=1)
        train_mask = (X.index >= train_start) & (X.index <= train_end)
        test_mask = X.index == day
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue

        X_tr_day = X.loc[train_mask]
        X_te_day = X.loc[test_mask]
        Y_tr_day = Y.loc[train_mask]
        for mtu in mtus:
            sample = DiagnosticSample(label=label, config_path=config_path, forecast_day=day, mtu=mtu)
            records, fit_record = _fit_lgbm_contribs(
                config=config,
                sample=sample,
                X_tr=X_tr_day,
                X_te=X_te_day,
                y_tr=Y_tr_day[mtu],
            )
            contribution_records.extend(records)
            fit_records.append(fit_record)

    contribution_df = pd.DataFrame(contribution_records)
    fit_df = pd.DataFrame(fit_records)
    output_dir.mkdir(parents=True, exist_ok=True)
    contribution_df.to_csv(output_dir / f"{label}_feature_contributions_long.csv", index=False)
    fit_df.to_csv(output_dir / f"{label}_fits.csv", index=False)
    return contribution_df, fit_df


def _summarize(contrib: pd.DataFrame, output_dir: Path) -> None:
    feature_summary = (
        contrib.groupby(["model_label", "feature_group", "feature"], as_index=False)
        .agg(
            mean_abs_contribution=("abs_contribution", "mean"),
            median_abs_contribution=("abs_contribution", "median"),
            mean_contribution=("contribution", "mean"),
            mean_gain=("gain", "mean"),
            mean_split=("split", "mean"),
            n_samples=("contribution", "size"),
        )
        .sort_values(["model_label", "mean_abs_contribution"], ascending=[True, False])
    )

    group_sample = (
        contrib.groupby(["model_label", "forecast_day", "mtu", "feature_group"], as_index=False)
        .agg(group_contribution=("contribution", "sum"))
    )
    group_summary = (
        group_sample.assign(abs_group_contribution=lambda df: df["group_contribution"].abs())
        .groupby(["model_label", "feature_group"], as_index=False)
        .agg(
            mean_abs_contribution=("abs_group_contribution", "mean"),
            median_abs_contribution=("abs_group_contribution", "median"),
            mean_contribution=("group_contribution", "mean"),
            n_samples=("group_contribution", "size"),
        )
    )
    totals = group_summary.groupby("model_label")["mean_abs_contribution"].transform("sum")
    group_summary["share_abs_contribution"] = np.where(totals > 0, group_summary["mean_abs_contribution"] / totals, 0.0)
    group_summary = group_summary.sort_values(
        ["model_label", "mean_abs_contribution"],
        ascending=[True, False],
    )

    feature_summary.to_csv(output_dir / "feature_importance.csv", index=False)
    group_summary.to_csv(output_dir / "group_importance.csv", index=False)


def main() -> None:
    args = _parse_args()
    config_paths = [Path(path) for path in args.config]
    if args.label is not None and len(args.label) != len(config_paths):
        raise ValueError("--label must be provided exactly once per --config.")
    labels = args.label if args.label is not None else [path.stem for path in config_paths]
    output_dir = Path(args.output_dir)

    all_contribs: list[pd.DataFrame] = []
    all_fits: list[pd.DataFrame] = []
    metadata = {
        "configs": [str(path) for path in config_paths],
        "labels": labels,
        "max_days": args.max_days,
        "mtu_step": args.mtu_step,
        "start": args.start,
        "end": args.end,
        "note": (
            "LightGBM pred_contrib values are computed on the fold-scaled model target. "
            "They are intended for relative diagnostic interpretation, not EUR/MWh attribution."
        ),
    }

    for config_path, label in zip(config_paths, labels, strict=True):
        contrib, fits = _run_one_config(
            config_path=config_path,
            label=label,
            output_dir=output_dir,
            max_days=args.max_days,
            mtu_step=args.mtu_step,
            start=args.start,
            end=args.end,
        )
        all_contribs.append(contrib)
        all_fits.append(fits)

    combined_contrib = pd.concat(all_contribs, ignore_index=True) if all_contribs else pd.DataFrame()
    combined_fits = pd.concat(all_fits, ignore_index=True) if all_fits else pd.DataFrame()
    combined_contrib.to_csv(output_dir / "feature_contributions_long.csv", index=False)
    combined_fits.to_csv(output_dir / "fits.csv", index=False)
    _summarize(combined_contrib, output_dir)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    group_summary = pd.read_csv(output_dir / "group_importance.csv")
    print("[price-importance] Saved diagnostics ->", output_dir)
    print(group_summary.groupby("model_label").head(8).to_string(index=False))


if __name__ == "__main__":
    main()
