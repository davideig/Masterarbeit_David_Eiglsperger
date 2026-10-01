"""RQ1 wind row of Table 5: OBTF vs ENTSO-E on the matched panel (2026-02-07 excluded).

    pixi run -e forecast python scripts/eval_wind_rq1.py
"""
import numpy as np
import pandas as pd

from da_price_forecasting.evaluation.metrics import gw_test_point

R = "results/renewable_generation_results/"
OBTF = R + "hybrid_dwd_mastr_wind_c100_icond2only_run06_onoff_split_wind_hub_p80_common_hgb_wind_struct_minleaf60_maxfeat08_bias30_mtu_s08_d180_cutoff1000_paper_febjul/forecast.csv"
ENTSOE = R + "entsoe_renewable_forecast_benchmark_paper_febjul/forecast.csv"
SKIP = {"2026-02-07"}


def read(path):
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True).tz_convert("Europe/Berlin")
    return df


m, e = read(OBTF), read(ENTSOE)
df = pd.DataFrame({"y_true": m["Wind_Total_Actual_MW"], "obtf": m["Wind_Total_Model_MW"],
                   "entsoe": e["Wind_Total_Model_MW"]}).loc["2026-02-01":"2026-07-31"].dropna()
day = df.index.strftime("%Y-%m-%d")
df = df[~day.isin(SKIP)]
rmse = {k: float(np.sqrt(((df[k] - df.y_true) ** 2).mean())) for k in ("obtf", "entsoe")}
mae = {k: float((df[k] - df.y_true).abs().mean()) for k in ("obtf", "entsoe")}
# GW needs complete 96-slot days
day = df.index.strftime("%Y-%m-%d")
full = df[pd.Series(day).map(pd.Series(day).value_counts()).values == 96].reset_index(drop=True)
better, worse = ("entsoe", "obtf") if rmse["entsoe"] < rmse["obtf"] else ("obtf", "entsoe")
p = gw_test_point(full, worse, better, n_periods=96, norm=2)
print(f"n={len(df)} (GW panel {len(full)})")
print(f"RMSE OBTF {rmse['obtf']:.0f}  ENTSO-E {rmse['entsoe']:.0f}  delta {100 * (rmse['obtf'] / rmse['entsoe'] - 1):+.1f}%")
print(f"MAE  OBTF {mae['obtf']:.0f}  ENTSO-E {mae['entsoe']:.0f}")
print(f"GW one-sided p ({better} better): {p:.2e}")
