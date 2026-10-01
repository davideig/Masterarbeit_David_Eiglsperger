from __future__ import annotations

import json
import re
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.stats import norm
from sklearn.linear_model import LassoCV, LassoLarsCV
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

# The per-MTU models are fit and predicted on numpy arrays, so scikit-learn emits a benign
# "X does not have valid feature names" warning at predict time. It does not affect results;
# silence only that one message so rolling-backtest logs stay readable.
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
)

Z_075 = norm.ppf(0.75)


def robust_params(x, eps: float = 1e-12) -> tuple[float, float]:
    """Compute robust centering and scaling parameters."""
    x = np.asarray(x, dtype=float)
    finite_x = x[np.isfinite(x)]
    if finite_x.size == 0:
        return 0.0, 1.0

    center = np.median(finite_x)
    mad = np.median(np.abs(finite_x - center))
    scale = mad / Z_075

    if not np.isfinite(scale) or scale <= eps:
        scale = 1.0

    return float(center), float(scale)


def inverse_vst_bias_corrected(
    y_hat_trans,
    residuals_trans,
    center: float,
    scale: float,
):
    """Map predictions from the transformed space back to the original scale."""
    y_hat_trans = np.asarray(y_hat_trans, dtype=float).reshape(-1)
    residuals_trans = np.asarray(residuals_trans, dtype=float).reshape(-1)

    if residuals_trans.size == 0:
        return center + scale * np.sinh(y_hat_trans)

    return center + scale * np.mean(
        np.sinh(y_hat_trans[:, None] + residuals_trans[None, :]),
        axis=1,
    )


def scale_fold_point(
    X_tr: pd.DataFrame,
    X_va: pd.DataFrame,
    y_tr: pd.Series,
    y_va: pd.Series | None,
    use_vst: bool = True,
    ssrd_filter_min_range: float = 20.0,
    ssrd_filter_min_pos_share: float = 0.50,
    ssrd_filter_min_iqr: float | None = None,
):
    """Apply fold-wise feature and target scaling for the LEAR point model."""
    cols = X_tr.columns.tolist()

    price_cols = [col for col in cols if col.startswith(("price_d", "exaa_d"))]
    time_cols = [col for col in cols if col.startswith("weekday_") or col in ["is_15min_market", "is_holiday"]]
    ssrd_cols = [col for col in cols if "ssrd" in col.lower()]
    sw_dir_cols = [col for col in cols if "sw_dir" in col.lower()]
    sw_dif_cols = [col for col in cols if "sw_dif" in col.lower()]
    wind_cols = [col for col in cols if "wind_speed" in col.lower()]
    load_cols = [col for col in cols if "load" in col.lower()]
    other_cols = [
        col
        for col in cols
        if col not in price_cols + time_cols + ssrd_cols + sw_dir_cols + sw_dif_cols + wind_cols + load_cols
    ]

    X_tr_work = X_tr.copy()
    X_va_work = X_va.copy()

    degenerate_ssrd_cols = []
    for col in ssrd_cols + sw_dir_cols + sw_dif_cols:
        x = X_tr_work[col].astype(float)
        col_range = x.max() - x.min()
        pos_share = (x > 0).mean()
        is_degenerate = (col_range < ssrd_filter_min_range) or (pos_share < ssrd_filter_min_pos_share)
        if ssrd_filter_min_iqr is not None:
            col_iqr = x.quantile(0.75) - x.quantile(0.25)
            is_degenerate = is_degenerate or (col_iqr < ssrd_filter_min_iqr)
        if is_degenerate:
            degenerate_ssrd_cols.append(col)

    if degenerate_ssrd_cols:
        X_tr_work.loc[:, degenerate_ssrd_cols] = 0.0
        X_va_work.loc[:, degenerate_ssrd_cols] = 0.0

    y_center, y_scale = robust_params(y_tr.values)
    y_tr_scaled = (y_tr.values - y_center) / y_scale
    y_va_scaled = (y_va.values - y_center) / y_scale if y_va is not None else None

    if use_vst:
        y_tr_scaled = np.arcsinh(y_tr_scaled)
        if y_va_scaled is not None:
            y_va_scaled = np.arcsinh(y_va_scaled)

    y_params = {
        "y_center": float(y_center),
        "y_scale": float(y_scale),
        "use_vst": bool(use_vst),
        "degenerate_ssrd_cols": degenerate_ssrd_cols,
    }

    X_tr_scaled = X_tr_work.copy()
    X_va_scaled = X_va_work.copy()

    for col in price_cols:
        center, scale = robust_params(X_tr_work[col].values)
        X_tr_scaled[col] = np.arcsinh((X_tr_work[col].values - center) / scale)
        X_va_scaled[col] = np.arcsinh((X_va_work[col].values - center) / scale)

    for col in wind_cols:
        center, scale = robust_params(X_tr_work[col].values)
        X_tr_scaled[col] = (X_tr_work[col].values - center) / scale
        X_va_scaled[col] = (X_va_work[col].values - center) / scale

    for col in load_cols:
        center, scale = robust_params(X_tr_work[col].values)
        X_tr_scaled[col] = (X_tr_work[col].values - center) / scale
        X_va_scaled[col] = (X_va_work[col].values - center) / scale

    cont_cols = ssrd_cols + sw_dir_cols + sw_dif_cols + other_cols
    if cont_cols:
        finite_cont_cols = [
            col for col in cont_cols if np.isfinite(X_tr_work[col].to_numpy(dtype=float)).any()
        ]
        empty_cont_cols = [col for col in cont_cols if col not in finite_cont_cols]
        if empty_cont_cols:
            X_tr_scaled.loc[:, empty_cont_cols] = 0.0
            X_va_scaled.loc[:, empty_cont_cols] = 0.0
        cont_cols = finite_cont_cols

    if cont_cols:
        scaler = MinMaxScaler()
        X_tr_scaled[cont_cols] = scaler.fit_transform(X_tr_work[cont_cols])
        X_va_scaled[cont_cols] = scaler.transform(X_va_work[cont_cols])

    return X_tr_scaled, X_va_scaled, y_tr_scaled, y_va_scaled, y_params


def _fit_predict_one_mtu(
    mtu: int,
    day: pd.Timestamp,
    X_tr: pd.DataFrame,
    X_te: pd.DataFrame,
    y_tr: pd.Series,
    *,
    use_vst: bool,
    price_model_type: str,
    use_lars: bool,
    hgb_max_iter: int,
    hgb_learning_rate: float,
    hgb_max_leaf_nodes: int,
    hgb_min_samples_leaf: int,
    hgb_l2_regularization: float,
    lgbm_n_estimators: int,
    lgbm_learning_rate: float,
    lgbm_num_leaves: int,
    lgbm_min_child_samples: int,
    lgbm_subsample: float,
    lgbm_colsample_bytree: float,
    lgbm_reg_lambda: float,
    random_state: int,
    lasso_cv_eps: float,
    lasso_cv_alphas,
    lasso_cv_tol: float,
    lasso_cv_max_iter: int,
    lars_max_iter: int,
    lars_max_n_alphas: int,
) -> dict:
    """Fit one MTU-specific model and return its prediction and diagnostics.

    Self-contained so the 96 MTU models for a day can run in parallel. Each call works on
    its own scaled copies and a fixed ``random_state``, so results are independent of whether
    it runs serially or across threads.
    """
    finite_target_mask = np.isfinite(y_tr.to_numpy(dtype=float))
    if finite_target_mask.sum() < 5:
        raise ValueError(
            "Not enough finite training targets for "
            f"{day.date()} MTU {mtu}: {int(finite_target_mask.sum())} available."
        )
    if not finite_target_mask.all():
        X_tr = X_tr.loc[finite_target_mask]
        y_tr = y_tr.loc[finite_target_mask]

    X_tr_s, X_te_s, y_tr_s, _, y_params = scale_fold_point(
        X_tr=X_tr,
        X_va=X_te,
        y_tr=y_tr,
        y_va=None,
        use_vst=use_vst,
    )

    if price_model_type == "lear" and use_lars:
        model = LassoLarsCV(cv=5, max_iter=lars_max_iter, max_n_alphas=lars_max_n_alphas, n_jobs=1)
    elif price_model_type == "lear":
        model = LassoCV(
            cv=5,
            eps=lasso_cv_eps,
            alphas=lasso_cv_alphas,
            tol=lasso_cv_tol,
            max_iter=lasso_cv_max_iter,
            n_jobs=1,
        )
    elif price_model_type == "hist_gradient_boosting":
        model = HistGradientBoostingRegressor(
            max_iter=hgb_max_iter,
            learning_rate=hgb_learning_rate,
            max_leaf_nodes=hgb_max_leaf_nodes,
            min_samples_leaf=hgb_min_samples_leaf,
            l2_regularization=hgb_l2_regularization,
            random_state=random_state,
        )
    else:  # lightgbm
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:  # pragma: no cover - depends on optional env
            raise ImportError("price_model_type='lightgbm' requires the optional `lightgbm` package.") from exc

        model = LGBMRegressor(
            n_estimators=lgbm_n_estimators,
            learning_rate=lgbm_learning_rate,
            num_leaves=lgbm_num_leaves,
            min_child_samples=lgbm_min_child_samples,
            subsample=lgbm_subsample,
            colsample_bytree=lgbm_colsample_bytree,
            reg_lambda=lgbm_reg_lambda,
            objective="regression",
            random_state=random_state,
            n_jobs=1,
            verbosity=-1,
        )

    model.fit(X_tr_s.values, y_tr_s)
    y_pred_s = model.predict(X_te_s.values)

    if use_vst:
        y_fit_s = model.predict(X_tr_s.values)
        residuals_s = y_tr_s - y_fit_s
        y_pred = inverse_vst_bias_corrected(
            y_hat_trans=y_pred_s,
            residuals_trans=residuals_s,
            center=y_params["y_center"],
            scale=y_params["y_scale"],
        )
    else:
        y_pred = y_params["y_center"] + y_params["y_scale"] * y_pred_s

    coef = getattr(model, "coef_", None)
    if coef is None:
        nonzero_mask = np.zeros(len(X_tr.columns), dtype=bool)
        nonzero_vals = []
        alpha = np.nan
    else:
        nonzero_mask = coef != 0
        nonzero_vals = coef[nonzero_mask].tolist()
        alpha = getattr(model, "alpha_", np.nan)

    return {
        "mtu": mtu,
        "timestamp": day + pd.Timedelta(minutes=15 * mtu),
        "y_pred": float(np.asarray(y_pred).ravel()[0]),
        "coef_record": {
            "forecast_day": day.date(),
            "mtu": mtu,
            "price_model_type": price_model_type,
            "alpha": alpha,
            "n_nonzero": int(nonzero_mask.sum()),
            "nonzero_cols": X_tr.columns[nonzero_mask].tolist(),
            "nonzero_vals": nonzero_vals,
            "use_lars": price_model_type == "lear" and use_lars,
        },
        "intercept_record": {
            "forecast_day": day.date(),
            "mtu": mtu,
            "price_model_type": price_model_type,
            "intercept": float(getattr(model, "intercept_", np.nan)),
        },
        "degenerate_record": {
            "forecast_day": day.date(),
            "mtu": mtu,
            "n_degenerate": len(y_params["degenerate_ssrd_cols"]),
            "degenerate_cols": y_params["degenerate_ssrd_cols"],
        },
    }


def rolling_point_forecast(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    forecast_days: list[pd.Timestamp] | pd.DatetimeIndex,
    train_days: int,
    lars_start_date: pd.Timestamp,
    use_vst: bool = True,
    price_model_type: str = "lear",
    hgb_max_iter: int = 600,
    hgb_learning_rate: float = 0.025,
    hgb_max_leaf_nodes: int = 15,
    hgb_min_samples_leaf: int = 20,
    hgb_l2_regularization: float = 0.0,
    lgbm_n_estimators: int = 900,
    lgbm_learning_rate: float = 0.015,
    lgbm_num_leaves: int = 15,
    lgbm_min_child_samples: int = 40,
    lgbm_subsample: float = 0.9,
    lgbm_colsample_bytree: float = 0.8,
    lgbm_reg_lambda: float = 1.0,
    random_state: int = 42,
    lasso_cv_eps: float = 1e-3,
    lasso_cv_alphas: int | list[float] = 100,
    lasso_cv_tol: float = 1e-3,
    lasso_cv_max_iter: int = 10_000,
    lars_max_iter: int = 1000,
    lars_max_n_alphas: int = 1000,
    mtu_n_jobs: int = -1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run rolling day-ahead point forecasts with MTU-specific models.

    The 96 MTU-specific models for each day are independent and are fit in parallel across
    ``mtu_n_jobs`` threads (``-1`` = all cores). LightGBM/sklearn fits release the GIL, so
    threading gives near-linear speedup while sharing the (read-only) feature matrix without
    serialization. Fixed seeds make the output identical to a serial run.
    """
    price_model_type = price_model_type.lower()
    if price_model_type not in {"lear", "hist_gradient_boosting", "lightgbm"}:
        raise ValueError(f"Unsupported price_model_type: {price_model_type!r}")

    all_preds = []
    all_trues = []
    runtime_records = []
    coef_records = []
    intercept_records = []
    degenerate_ssrd_records = []

    parallel = Parallel(n_jobs=mtu_n_jobs, prefer="threads")

    for day in forecast_days:
        day_start = time.perf_counter()
        use_lars = day >= lars_start_date

        train_start = day - pd.Timedelta(days=train_days)
        train_end = day - pd.Timedelta(days=1)
        train_mask = (X.index >= train_start) & (X.index <= train_end)
        test_mask = X.index == day

        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue

        # X slices are identical across the 96 MTU models; build them once per day.
        X_tr_day = X.loc[train_mask]
        X_te_day = X.loc[test_mask]
        Y_tr_day = Y.loc[train_mask]
        finite_target_counts = np.isfinite(Y_tr_day.to_numpy(dtype=float)).sum(axis=0)
        min_finite_targets = int(finite_target_counts.min()) if len(finite_target_counts) else 0
        if min_finite_targets < 5:
            print(
                f"  {day.date()}  skipped: only {min_finite_targets} finite "
                "training targets available for at least one MTU"
            )
            continue

        mtu_results = parallel(
            delayed(_fit_predict_one_mtu)(
                mtu,
                day,
                X_tr_day,
                X_te_day,
                Y_tr_day[mtu],
                use_vst=use_vst,
                price_model_type=price_model_type,
                use_lars=use_lars,
                hgb_max_iter=hgb_max_iter,
                hgb_learning_rate=hgb_learning_rate,
                hgb_max_leaf_nodes=hgb_max_leaf_nodes,
                hgb_min_samples_leaf=hgb_min_samples_leaf,
                hgb_l2_regularization=hgb_l2_regularization,
                lgbm_n_estimators=lgbm_n_estimators,
                lgbm_learning_rate=lgbm_learning_rate,
                lgbm_num_leaves=lgbm_num_leaves,
                lgbm_min_child_samples=lgbm_min_child_samples,
                lgbm_subsample=lgbm_subsample,
                lgbm_colsample_bytree=lgbm_colsample_bytree,
                lgbm_reg_lambda=lgbm_reg_lambda,
                random_state=random_state,
                lasso_cv_eps=lasso_cv_eps,
                lasso_cv_alphas=lasso_cv_alphas,
                lasso_cv_tol=lasso_cv_tol,
                lasso_cv_max_iter=lasso_cv_max_iter,
                lars_max_iter=lars_max_iter,
                lars_max_n_alphas=lars_max_n_alphas,
            )
            for mtu in range(96)
        )

        y_hat_day = {}
        y_true_day = {}
        for res in mtu_results:  # returned in submission order (MTU 0..95)
            y_hat_day[res["timestamp"]] = res["y_pred"]
            y_true_day[res["timestamp"]] = float(Y.loc[day, res["mtu"]])
            coef_records.append(res["coef_record"])
            intercept_records.append(res["intercept_record"])
            degenerate_ssrd_records.append(res["degenerate_record"])

        all_preds.append(pd.Series(y_hat_day, dtype=float))
        all_trues.append(pd.Series(y_true_day, dtype=float))

        day_runtime = time.perf_counter() - day_start
        runtime_records.append(
            {
                "forecast_day": day,
                "price_model_type": price_model_type,
                "train_days": train_days,
                "use_vst": use_vst,
                "use_lars": price_model_type == "lear" and use_lars,
                "lasso_cv_eps": lasso_cv_eps,
                "lasso_cv_alphas": json.dumps(lasso_cv_alphas),
                "lasso_cv_tol": lasso_cv_tol,
                "lasso_cv_max_iter": lasso_cv_max_iter,
                "lars_max_iter": lars_max_iter,
                "lars_max_n_alphas": lars_max_n_alphas,
                "runtime_seconds": day_runtime,
            }
        )
        if price_model_type == "lear":
            model_label = "LARS" if use_lars else "LassoCV"
        else:
            model_label = price_model_type
        print(f"  {day.date()}  {model_label}  {day_runtime:.1f}s")

    y_pred_all = pd.concat(all_preds).sort_index()
    y_true_all = pd.concat(all_trues).sort_index()

    forecast_df = pd.DataFrame({"y_pred": y_pred_all, "y_true": y_true_all})
    runtime_df = pd.DataFrame(runtime_records)
    coef_df = pd.DataFrame(coef_records)
    intercept_df = pd.DataFrame(intercept_records)
    degenerate_ssrd_df = pd.DataFrame(degenerate_ssrd_records)
    return forecast_df, runtime_df, coef_df, intercept_df, degenerate_ssrd_df


def compute_metrics(fc: pd.DataFrame, label: str) -> dict:
    """Compute point forecast error metrics for a given forecast slice."""
    n_inf_nan = int((~np.isfinite(fc["y_pred"])).sum())
    valid = fc[np.isfinite(fc["y_pred"]) & np.isfinite(fc["y_true"])]

    mae = mean_absolute_error(valid["y_true"], valid["y_pred"]) if len(valid) > 0 else np.nan
    rmse = np.sqrt(((valid["y_true"] - valid["y_pred"]) ** 2).mean()) if len(valid) > 0 else np.nan
    bias = (valid["y_pred"] - valid["y_true"]).mean() if len(valid) > 0 else np.nan

    return {
        "period": label,
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "n_obs": len(fc),
        "n_inf_nan": n_inf_nan,
    }


def apply_rolling_forecast_bias_correction(
    forecast_df: pd.DataFrame,
    train_days: int,
    min_train_days: int,
    by_hour: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Correct forecasts using only previously observed rolling forecast errors."""
    corrected_parts = []
    correction_records = []
    daily_index = forecast_df.index.normalize()

    for day in daily_index.unique().sort_values():
        train_start = day - pd.Timedelta(days=train_days)
        train_end = day - pd.Timedelta(minutes=15)
        train_mask = (forecast_df.index >= train_start) & (forecast_df.index <= train_end)
        test_mask = daily_index == day

        train_days_available = forecast_df.index[train_mask].normalize().nunique()
        correction = 0.0
        day_forecast = forecast_df.loc[test_mask].copy()

        if train_days_available >= min_train_days:
            past_error = forecast_df.loc[train_mask, "y_pred"] - forecast_df.loc[train_mask, "y_true"]
            correction = -float(past_error.mean())

            if by_hour:
                day_corrections = pd.Series(correction, index=day_forecast.index)
                for hour in range(24):
                    hour_train_mask = train_mask & (forecast_df.index.hour == hour)
                    hour_days_available = forecast_df.index[hour_train_mask].normalize().nunique()
                    if hour_days_available >= min_train_days:
                        hour_error = (
                            forecast_df.loc[hour_train_mask, "y_pred"]
                            - forecast_df.loc[hour_train_mask, "y_true"]
                        )
                        hour_correction = -float(hour_error.mean())
                        day_corrections.loc[day_corrections.index.hour == hour] = hour_correction
                day_forecast["y_pred"] = day_forecast["y_pred"] + day_corrections
            else:
                day_forecast["y_pred"] = day_forecast["y_pred"] + correction

        corrected_parts.append(day_forecast)
        correction_records.append(
            {
                "forecast_day": day,
                "train_days_available": int(train_days_available),
                "bias_correction": correction,
                "by_hour": bool(by_hour),
            }
        )

    return pd.concat(corrected_parts).sort_index(), pd.DataFrame(correction_records)


def save_experiment_outputs(
    name: str,
    forecast_df: pd.DataFrame,
    runtime_df: pd.DataFrame,
    config: dict,
    export_dir: Path,
):
    """Save all experiment outputs to disk, including monthly metrics."""
    export_dir.mkdir(parents=True, exist_ok=True)
    forecast_df.to_csv(export_dir / "forecast.csv", index=True)
    runtime_df.to_csv(export_dir / "runtime.csv", index=False)

    rows = [compute_metrics(forecast_df, "full")]
    months = forecast_df.index.tz_localize(None).to_period("M").unique()
    for period in months:
        mask = forecast_df.index.tz_localize(None).to_period("M") == period
        rows.append(compute_metrics(forecast_df[mask], str(period)))

    pd.DataFrame(rows).to_csv(export_dir / "metrics.csv", index=False)
    with open(export_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    print(f"✓ Saved: forecast | runtime | metrics | config  ->  {export_dir}")


def save_prediction_outputs(
    forecast_df: pd.DataFrame,
    runtime_df: pd.DataFrame,
    config: dict,
    export_dir: Path,
):
    """Save forecast-only outputs for operational submission workflows."""
    export_dir.mkdir(parents=True, exist_ok=True)
    forecast_df.to_csv(export_dir / "forecast.csv", index=True)
    runtime_df.to_csv(export_dir / "runtime.csv", index=False)
    with open(export_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    print(f"✓ Saved: forecast | runtime | config  ->  {export_dir}")


def evaluate_and_plot_forecast_from_df(
    forecasts: pd.DataFrame,
    start: str,
    end: str,
    model_label: str = "Forecast",
    title: str = "Rolling Point Forecast vs Actual",
    figsize: tuple[int, int] = (14, 5),
):
    """Evaluate and plot a rolling point forecast against actual values."""
    tz = forecasts.index.tz
    start_ts = pd.Timestamp(start, tz=tz)
    end_ts = pd.Timestamp(end, tz=tz) + pd.Timedelta(days=1) - pd.Timedelta(minutes=15)

    df = forecasts.loc[start_ts:end_ts]
    if df.empty:
        raise ValueError(f"No data found in the evaluation window {start_ts} to {end_ts}.")

    missing_cols = {"y_true", "y_pred"} - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    metrics = compute_metrics(df, label="eval")
    mae = metrics["mae"]
    rmse = metrics["rmse"]
    bias = metrics["bias"]

    print(f"Evaluation window: {start_ts} -> {end_ts}")
    print(f"MAE  : {mae:.3f}")
    print(f"RMSE : {rmse:.3f}")
    print(f"Bias : {bias:.3f}")

    plt.figure(figsize=figsize)
    plt.plot(df["y_true"].index, df["y_true"].values, label="Actual", alpha=0.8)
    plt.plot(df["y_pred"].index, df["y_pred"].values, label=model_label, alpha=0.8)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

    return {"MAE": mae, "RMSE": rmse, "Bias": bias}


def scale_fold_anc(
    X_tr: pd.DataFrame,
    X_te: pd.DataFrame,
    ssrd_filter_min_range: float = 20.0,
    ssrd_filter_min_pos_share: float = 0.50,
    ssrd_filter_min_iqr: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Apply fold-wise Min-Max scaling for ANC re-estimation."""
    cols = X_tr.columns.tolist()
    calendar_cols = [col for col in cols if col.startswith("weekday_") or col in ["is_15min_market", "is_holiday"]]
    ssrd_cols = [col for col in cols if "ssrd" in col.lower()]
    sw_dir_cols = [col for col in cols if "sw_dir" in col.lower()]
    sw_dif_cols = [col for col in cols if "sw_dif" in col.lower()]
    scale_cols = [col for col in cols if col not in calendar_cols]

    X_tr_work = X_tr.copy()
    X_te_work = X_te.copy()

    degenerate_ssrd_cols = []
    for col in ssrd_cols + sw_dir_cols + sw_dif_cols:
        x = X_tr_work[col].astype(float)
        col_range = x.max() - x.min()
        pos_share = (x > 0).mean()
        is_degenerate = (col_range < ssrd_filter_min_range) or (pos_share < ssrd_filter_min_pos_share)
        if ssrd_filter_min_iqr is not None:
            col_iqr = x.quantile(0.75) - x.quantile(0.25)
            is_degenerate = is_degenerate or (col_iqr < ssrd_filter_min_iqr)
        if is_degenerate:
            degenerate_ssrd_cols.append(col)

    if degenerate_ssrd_cols:
        X_tr_work.loc[:, degenerate_ssrd_cols] = 0.0
        X_te_work.loc[:, degenerate_ssrd_cols] = 0.0

    X_tr_scaled = X_tr_work.copy()
    X_te_scaled = X_te_work.copy()
    scaling_params = {"degenerate_ssrd_cols": degenerate_ssrd_cols}

    for col in scale_cols:
        col_min = float(X_tr_work[col].min())
        col_max = float(X_tr_work[col].max())
        col_range = col_max - col_min

        if not np.isfinite(col_range) or col_range <= 1e-12:
            X_tr_scaled[col] = 0.0
            X_te_scaled[col] = 0.0
            col_range = 1.0
        else:
            X_tr_scaled[col] = (X_tr_work[col] - col_min) / col_range
            X_te_scaled[col] = (X_te_work[col] - col_min) / col_range

        scaling_params[col] = {"min": col_min, "max": col_max, "range": float(col_range)}

    return X_tr_scaled, X_te_scaled, scaling_params


def rolling_anc_feature_importance(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    forecast_days: list[pd.Timestamp] | pd.DatetimeIndex,
    train_days: int,
    lars_start_date: pd.Timestamp,
    lasso_cv_eps: float = 1e-3,
    lasso_cv_alphas: int | list[float] = 100,
    lasso_cv_tol: float = 1e-3,
    lasso_cv_max_iter: int = 10_000,
    lars_max_iter: int = 1000,
    lars_max_n_alphas: int = 1000,
) -> pd.DataFrame:
    """Run a rolling ANC estimation with MTU-specific LEAR models."""
    anc_records = []

    for day in forecast_days:
        use_lars = day >= lars_start_date

        train_start = day - pd.Timedelta(days=train_days)
        train_end = day - pd.Timedelta(days=1)
        train_mask = (X.index >= train_start) & (X.index <= train_end)
        test_mask = X.index == day

        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue

        for mtu in range(96):
            X_tr = X.loc[train_mask]
            y_tr = Y.loc[train_mask, mtu]
            X_te = X.loc[test_mask]

            X_tr_s, X_te_s, _ = scale_fold_anc(X_tr=X_tr, X_te=X_te)

            if use_lars:
                model = LassoLarsCV(
                    cv=5,
                    max_iter=lars_max_iter,
                    max_n_alphas=lars_max_n_alphas,
                    n_jobs=1,
                )
            else:
                model = LassoCV(
                    cv=5,
                    eps=lasso_cv_eps,
                    alphas=lasso_cv_alphas,
                    tol=lasso_cv_tol,
                    max_iter=lasso_cv_max_iter,
                    n_jobs=1,
                )

            model.fit(X_tr_s.values, y_tr.values)
            beta_series = pd.Series(model.coef_, index=X_tr_s.columns, dtype=float)

            mtu_timestamp = day + pd.Timedelta(minutes=15 * mtu)
            x_row = X_te_s.iloc[0]
            for feature in X_te_s.columns:
                feature_value = float(x_row[feature])
                beta = float(beta_series[feature])
                contribution = feature_value * beta
                anc_records.append(
                    {
                        "forecast_day": day,
                        "timestamp": mtu_timestamp,
                        "mtu": mtu,
                        "train_days": train_days,
                        "use_lars": use_lars,
                        "feature": feature,
                        "feature_value": feature_value,
                        "beta": beta,
                        "contribution": contribution,
                    }
                )

        print(f"  {day.date()}  {'LARS' if use_lars else 'LassoCV'}")

    return pd.DataFrame(anc_records)


def map_feature_to_group(feature: str) -> str:
    """Map a raw regressor name to a high-level feature group."""
    feat = feature.lower()
    if "exaa" in feat:
        return "EXAA d"
    if "price_d1" in feat:
        return "Price d-1"
    if "price_d2" in feat:
        return "Price d-2"
    if "price_d7" in feat:
        return "Price d-7"
    if "load_d0" in feat:
        return "Load d"
    if "wind" in feat:
        return "Wind d"
    if "ssrd" in feat or "sw_dir" in feat or "sw_dif" in feat:
        return "Solar d"
    if feat.startswith("weekday_"):
        return "Weekday"
    if "is_holiday" in feat:
        return "Holiday"
    if "is_15min_market" in feat:
        return "15-min market dummy"
    return "Other"


def map_wind_feature_to_cluster(feature: str) -> str:
    """Map a raw feature name to its wind cluster group."""
    feat = feature.lower()
    match = re.search(r"wind_speed_cluster_(\d+)_h\d+", feat)
    if match:
        return f"Wind Cluster {match.group(1)}"
    return "Other"


def map_solar_feature_to_cluster(feature: str) -> str:
    """Map a raw feature name to its solar cluster group."""
    feat = feature.lower()
    if not any(token in feat for token in ["ssrd", "sw_dir", "sw_dif"]):
        return "Other"
    match = re.search(r"cluster_(\d+)_h\d+", feat)
    if match:
        return f"Solar Cluster {match.group(1)}"
    return "Other"


def summarize_feature_group_anc(anc_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate raw ANC records to overall feature-group ANC."""
    anc_df_analysis = anc_df.copy()
    anc_df_analysis["feature_group"] = anc_df_analysis["feature"].apply(map_feature_to_group)

    grouped_contrib_df = (
        anc_df_analysis
        .groupby(["forecast_day", "timestamp", "mtu", "train_days", "feature_group"], as_index=False)["contribution"]
        .sum()
        .rename(columns={"contribution": "group_contribution"})
    )
    grouped_contrib_df["abs_group_contribution"] = grouped_contrib_df["group_contribution"].abs()

    anc_summary_df = (
        grouped_contrib_df
        .groupby(["train_days", "feature_group"], as_index=False)["abs_group_contribution"]
        .mean()
        .rename(columns={"abs_group_contribution": "ANC"})
        .sort_values(["train_days", "ANC"], ascending=[True, False])
        .reset_index(drop=True)
    )
    return anc_df_analysis, anc_summary_df


def summarize_wind_cluster_anc(anc_df: pd.DataFrame, mtu_window: list[int] | range) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate raw ANC records to wind-cluster ANC."""
    anc_df_wind = anc_df.copy()
    anc_df_wind["cluster_group"] = anc_df_wind["feature"].apply(map_wind_feature_to_cluster)
    anc_df_wind = anc_df_wind[anc_df_wind["cluster_group"] != "Other"].copy()
    anc_df_wind = anc_df_wind[anc_df_wind["mtu"].isin(list(mtu_window))].copy()

    wind_grouped = (
        anc_df_wind
        .groupby(["forecast_day", "timestamp", "mtu", "train_days", "cluster_group"], as_index=False)["contribution"]
        .sum()
        .rename(columns={"contribution": "group_contribution"})
    )
    wind_grouped["abs_group_contribution"] = wind_grouped["group_contribution"].abs()

    wind_anc_summary = (
        wind_grouped
        .groupby(["train_days", "cluster_group"], as_index=False)["abs_group_contribution"]
        .mean()
        .rename(columns={"abs_group_contribution": "ANC"})
        .sort_values("ANC", ascending=False)
        .reset_index(drop=True)
    )

    wind_anc_export = wind_anc_summary.copy()
    wind_anc_export["cluster_id"] = wind_anc_export["cluster_group"].str.extract(r"Wind Cluster (\d+)").astype(int)
    return wind_anc_summary, wind_anc_export


def summarize_solar_cluster_anc(anc_df: pd.DataFrame, mtu_window: list[int] | range) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate raw ANC records to solar-cluster ANC."""
    anc_df_solar = anc_df.copy()
    anc_df_solar["cluster_group"] = anc_df_solar["feature"].apply(map_solar_feature_to_cluster)
    anc_df_solar = anc_df_solar[anc_df_solar["cluster_group"] != "Other"].copy()
    anc_df_solar = anc_df_solar[anc_df_solar["mtu"].isin(list(mtu_window))].copy()

    solar_grouped = (
        anc_df_solar
        .groupby(["forecast_day", "timestamp", "mtu", "train_days", "cluster_group"], as_index=False)["contribution"]
        .sum()
        .rename(columns={"contribution": "group_contribution"})
    )
    solar_grouped["abs_group_contribution"] = solar_grouped["group_contribution"].abs()

    solar_anc_summary = (
        solar_grouped
        .groupby(["train_days", "cluster_group"], as_index=False)["abs_group_contribution"]
        .mean()
        .rename(columns={"abs_group_contribution": "ANC"})
        .sort_values("ANC", ascending=False)
        .reset_index(drop=True)
    )

    solar_anc_export = solar_anc_summary.copy()
    solar_anc_export["cluster_id"] = solar_anc_export["cluster_group"].str.extract(r"Solar Cluster (\d+)").astype(int)
    return solar_anc_summary, solar_anc_export


def save_anc_outputs(
    export_dir: Path,
    anc_summary_df: pd.DataFrame,
    wind_anc_export: pd.DataFrame,
    solar_anc_export: pd.DataFrame,
    config: dict,
) -> None:
    """Save ANC outputs to disk using the notebook-compatible artifact layout."""
    export_dir.mkdir(parents=True, exist_ok=True)
    anc_summary_df.to_csv(export_dir / "anc_all_feature_results.csv", index=True)
    wind_anc_export.to_csv(export_dir / "anc_wind_feature_results.csv", index=False)
    solar_anc_export.to_csv(export_dir / "anc_solar_feature_results.csv", index=False)
    with open(export_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    print(f"✓ Saved: anc_features | anc_wind | anc_solar | config  ->  {export_dir}")
