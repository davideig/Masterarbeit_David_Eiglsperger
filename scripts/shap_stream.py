"""Memory-light, parallel rerun of price_feature_importance for one config.

Uses the repo's own _fit_lgbm_contribs (identical model, scaling, and TreeSHAP),
but keeps only a float32 contribution matrix instead of one dict per feature,
and fits the days in parallel worker processes.
"""
import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from da_price_forecasting.pipelines.lear import prepare_lear_operational_dataset
from da_price_forecasting.scripts import price_feature_importance as pfi

_G: dict = {}


def _init(config, X, Y, label, config_path):
    _G.update(config=config, X=X, Y=Y, label=label, config_path=config_path)


def _one_day(day):
    config, X, Y = _G["config"], _G["X"], _G["Y"]
    train_mask = (X.index >= day - pd.Timedelta(days=config.train_days_rolling)) & (
        X.index <= day - pd.Timedelta(days=1)
    )
    test_mask = X.index == day
    if train_mask.sum() == 0 or test_mask.sum() == 0:
        return None
    X_tr, X_te, Y_tr = X.loc[train_mask], X.loc[test_mask], Y.loc[train_mask]
    rows, expected = [], []
    for mtu in range(96):
        sample = pfi.DiagnosticSample(
            label=_G["label"], config_path=_G["config_path"], forecast_day=day, mtu=mtu
        )
        records, fit = pfi._fit_lgbm_contribs(
            config=config, sample=sample, X_tr=X_tr, X_te=X_te, y_tr=Y_tr[mtu]
        )
        rows.append(np.array([r["contribution"] for r in records], dtype=np.float32))
        expected.append(fit["expected_value"])
    return day.date().isoformat(), np.vstack(rows), np.array(expected)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--start", default="2026-02-01")
    ap.add_argument("--end", default="2026-07-31")
    ap.add_argument("--max-days", type=int, default=200)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    config_path = Path(args.config)
    config = pfi._load_lear_config(config_path)
    dataset = prepare_lear_operational_dataset(config)
    X, Y = dataset["X"], dataset["Y"]
    days = pfi._sample_days(config, X, max_days=args.max_days, start=args.start, end=args.end)
    print(f"[stream] {args.label}: {len(days)} days x 96 MTUs, {X.shape[1]} features", flush=True)

    ctx = mp.get_context("fork")
    out_days, mats, exps = [], [], []
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=ctx, initializer=_init,
        initargs=(config, X, Y, args.label, config_path),
    ) as ex:
        for i, res in enumerate(ex.map(_one_day, days), 1):
            if res is None:
                continue
            d, m, e = res
            out_days.append(d)
            mats.append(m)
            exps.append(e)
            if i % 10 == 0:
                print(f"[stream] {args.label}: {i}/{len(days)} days", flush=True)

    features = [str(c) for c in X.columns]
    groups = np.array([pfi._feature_group(f) for f in features])
    contrib = np.vstack(mats)  # (n_days*96, n_features)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / f"{args.label}_contributions.npz",
        contrib=contrib, features=np.array(features), groups=groups,
        days=np.repeat(out_days, 96), mtus=np.tile(np.arange(96), len(out_days)),
        expected=np.concatenate(exps),
    )

    # Same aggregation as price_feature_importance._summarize (group level).
    rows = []
    for g in sorted(set(groups)):
        gs = contrib[:, groups == g].astype(np.float64).sum(axis=1)
        rows.append({
            "model_label": args.label, "feature_group": g,
            "mean_abs_contribution": float(np.abs(gs).mean()),
            "median_abs_contribution": float(np.median(np.abs(gs))),
            "mean_contribution": float(gs.mean()), "n_samples": int(len(gs)),
        })
    gsum = pd.DataFrame(rows)
    gsum["share_abs_contribution"] = gsum["mean_abs_contribution"] / gsum["mean_abs_contribution"].sum()
    gsum = gsum.sort_values("mean_abs_contribution", ascending=False)
    gsum.to_csv(out / f"{args.label}_group_importance.csv", index=False)
    print(gsum.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
