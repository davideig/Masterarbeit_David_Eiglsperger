"""Subgroup SHAP shares (Table A7/A8 style) from long CSV or streaming npz output.

Share = mean_s |sum of contributions of the subgroup in sample s| divided by the
sum over groups of mean_s |sum of contributions of the group in sample s|, i.e.
the same normalization as the grouped shares.
"""
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

from da_price_forecasting.scripts.price_feature_importance import _feature_group

RULES = [
    (r"^exaa", "EXAA price"),
    (r"^reserve_", "Reserve market"),
    (r"^price_d1_", "Lagged price (day d-1)"),
    (r"^price_d2_", "Lagged price (day d-2)"),
    (r"^price_d7_", "Lagged price (day d-7)"),
    (r"^Residual_Load_Proxy_MW_ramp", "Residual load ramp"),
    (r"^Residual_Load_Proxy_MW", "Residual load"),
    (r"^wind_speed", "Wind speed"),
    (r"^sw_dif", "Shortwave diffuse radiation"),
    (r"^sw_dir", "Shortwave direct radiation"),
    (r"^Renewable_Wind", "Wind generation"),
    (r"^Renewable_Solar", "Solar generation"),
    (r"^Renewable_Total", "Total renewable generation"),
    (r"^load_d0", "Load forecast"),
]


def subgroup(feature: str) -> str:
    for pat, name in RULES:
        if re.match(pat, feature):
            return name
    return "Calendar/other"


def shares_from_matrix(contrib, features):
    groups = np.array([_feature_group(f) for f in features])
    subs = np.array([subgroup(f) for f in features])
    c = contrib.astype(np.float64)
    denom = sum(np.abs(c[:, groups == g].sum(1)).mean() for g in set(groups))
    out = {s: np.abs(c[:, subs == s].sum(1)).mean() / denom * 100 for s in set(subs)}
    return pd.Series(out).sort_values(ascending=False)


def shares_from_long_csv(path):
    sub_sum, grp_sum = defaultdict(float), defaultdict(float)
    cache = {}
    for chunk in pd.read_csv(path, usecols=["forecast_day", "mtu", "feature", "contribution"],
                             chunksize=5_000_000):
        feats = chunk["feature"].unique()
        for f in feats:
            if f not in cache:
                cache[f] = (subgroup(f), _feature_group(f))
        chunk["sub"] = chunk["feature"].map(lambda f: cache[f][0])
        chunk["grp"] = chunk["feature"].map(lambda f: cache[f][1])
        for key, val in chunk.groupby(["forecast_day", "mtu", "sub"])["contribution"].sum().items():
            sub_sum[key] += val
        for key, val in chunk.groupby(["forecast_day", "mtu", "grp"])["contribution"].sum().items():
            grp_sum[key] += val
    s = pd.Series(sub_sum).abs().groupby(level=2).mean()
    g = pd.Series(grp_sum).abs().groupby(level=2).mean()
    return (s / g.sum() * 100).sort_values(ascending=False)


if __name__ == "__main__":
    path = sys.argv[1]
    if path.endswith(".npz"):
        z = np.load(path, allow_pickle=True)
        res = shares_from_matrix(z["contrib"], [str(f) for f in z["features"]])
    else:
        res = shares_from_long_csv(path)
    print(res.round(1).to_string())
