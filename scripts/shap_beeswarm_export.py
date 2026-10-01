"""Export contributions and feature values of selected features for the beeswarm figure.

Reads the shap_stream npz output and rebuilds the feature matrix with the same
dataset preparation as shap_stream.py, so feature values match the fitted inputs.

    pixi run -e forecast python scripts/shap_beeswarm_export.py \
        --config configs/pricebase_sweep/oos_pgen_c2_d70_clean.yaml \
        --npz results/feature_importance/rq2_clean/pgen_contributions.npz \
        --out results/feature_importance/rq2_clean/pgen_beeswarm_long.csv \
        --skip 2026-02-07 --features price_d1_mtu_95 ...
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import da_price_forecasting.scripts.price_feature_importance as pfi
from da_price_forecasting.pipelines.lear import prepare_lear_operational_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip", nargs="*", default=[])
    ap.add_argument("--features", nargs="+", required=True)
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    features = [str(f) for f in z["features"]]
    days = z["days"].astype(str)
    keep = ~np.isin(days, args.skip)
    X = prepare_lear_operational_dataset(pfi._load_lear_config(Path(args.config)))["X"]
    xday = X.index.strftime("%Y-%m-%d")
    rows = []
    for feat in args.features:
        j = features.index(feat)
        values = pd.Series(X[feat].to_numpy(), index=xday)
        rows.append(pd.DataFrame({
            "forecast_day": days[keep], "mtu": z["mtus"][keep], "feature": feat,
            "contribution": z["contrib"][keep, j].astype(float),
            "feature_value": values.reindex(days[keep]).to_numpy(),
        }))
    out = pd.concat(rows, ignore_index=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(out.groupby("feature")[["contribution", "feature_value"]].agg(["count", "mean"]).round(3))


if __name__ == "__main__":
    main()
