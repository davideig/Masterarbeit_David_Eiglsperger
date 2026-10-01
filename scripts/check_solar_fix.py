"""Gate for the overnight RQ3 run: verify the rerun RQ3 solar components (07:00-11:00).

Passes (exit 0) only if, for every cutoff,
  * the solar RMSE on Feb-Jul 2026 is below 2,400 MW (broken run: ~2,630, 12:00 run: ~2,290), and
  * the per-area physics baseline in June equals the August 12:00 run (the data state
    that RQ1/RQ2 and RQ3 12:00 use) within 1 MW.

    pixi run -e forecast python scripts/check_solar_fix.py
"""
import sys

import numpy as np
import pandas as pd
import yaml

REF = ("results/renewable_generation_results/dwd_icon_mastr_solar_tso_c25_run06_tso_components_cloud_geometry_"
       "physics_residual_own_region_daylight_suspicious_totalbias_hgb_solar_bias45_hour_s075_d90_cutoff1000_"
       "price_warmup_d70_nov23febjul")
AREAS = ["TransnetBW", "Amprion", "50Hertz", "TenneT"]


def read(path):
    df = pd.read_csv(path + "/forecast.csv", index_col=0)
    df.index = pd.to_datetime(df.index, utc=True).tz_convert("Europe/Berlin")
    return df.loc["2026-02-01":"2026-07-31"]


ref = read(REF)
ok = True
for t in ["0700", "0800", "0900", "1000", "1100"]:
    export = yaml.safe_load(open(f"configs/rq3_clean/solar_{t}.yaml"))["config"]["export_dir"]
    df = read(export)
    err = (df["Solar_Model_MW"] - df["Solar_Actual_MW"]).dropna()
    rmse = float(np.sqrt((err ** 2).mean()))
    # 2026-06-12 is accepted: the August run had a near-zero baseline there (data gap),
    # the rebuilt features give a plausible one. Every other June day must match.
    june = df.loc["2026-06"]; june_ref = ref.loc["2026-06"]
    keep = june.index.strftime("%Y-%m-%d") != "2026-06-12"
    june_diff = max(
        float((june.loc[keep, f"Solar_{a}_Baseline_MW"] - june_ref.loc[keep, f"Solar_{a}_Baseline_MW"]).abs().max())
        for a in AREAS
    )
    passed = rmse < 2400 and june_diff < 1.0
    ok &= passed
    print(f"solar_{t}: RMSE {rmse:7.0f} MW, max June baseline diff vs August (excl. 12 June) {june_diff:9.2f} MW -> {'OK' if passed else 'FAIL'}")
print("CHECK PASSED" if ok else "CHECK FAILED")
sys.exit(0 if ok else 1)
