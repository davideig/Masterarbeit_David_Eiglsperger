"""RQ3 Table 7, Holm and reserve ablation for the harmonized grid (configs/rq3_clean).

Panel: 1 Feb-31 Jul 2026 without 2026-02-07 and 2026-06-19.

    pixi run -e forecast python scripts/rq3_clean_eval.py
"""
import numpy as np
import pandas as pd
import yaml

from da_price_forecasting.evaluation.metrics import gw_test_point

SKIP = {"2026-02-07", "2026-06-19"}
C = "configs/rq3_clean/"


def export(name):
    return yaml.safe_load(open(C + name + ".yaml"))["config"]["export_dir"]


FILES = {
    "0700": export("price_0700"),
    "0800": export("price_0800"),
    "0900": export("price_0900"),
    "1000": export("price_1000"),
    "1100": export("price_1100"),
    "1200": export("price_1200"),
    "0900_nores": export("price_0900_noreserve"),
    "1000_nores": export("price_1000_noreserve"),
    "1100_nores": export("price_1100_noreserve"),
    "1200_nores": export("price_1200_noreserve"),
    "exaa_only": "results/price_forecast_results/rq3_cutoff_grid/price_1200_exaa_only_c2_d70_lsdf",
}


def load(path):
    df = pd.read_csv(path + "/forecast.csv", index_col=0)
    df.index = pd.to_datetime(df.index, utc=True).tz_convert("Europe/Berlin")
    # trim to the test period first so all frames share the same index (the 96-slot
    # grid of the 92-MTU DST day 29 March repeats 4 timestamps in every file)
    return df.loc["2026-02-01":"2026-07-31"]


def panel(files):
    frames = {k: load(v) for k, v in files.items()}
    df = pd.DataFrame({k: f["y_pred"] for k, f in frames.items()})
    df["y_true"] = frames["1200"]["y_true"]
    df = df.loc["2026-02-01":"2026-07-31"]
    df = df[~df.index.strftime("%Y-%m-%d").isin(SKIP)].dropna()
    assert len(df) % 96 == 0, len(df)
    return df


def gw(df, worse, better):
    """One-sided p-value that `better` improves on `worse`."""
    return float(gw_test_point(df, worse, better, n_periods=96, norm=2))


def holm(pvals):
    items = sorted((p, k) for k, p in pvals.items() if p is not None)
    m, out, running = len(items), {}, 0.0
    for i, (p, k) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


df = panel(FILES)
files = FILES
print(f"===== rq3_clean grid  (n={len(df)}) =====")
rows = ["0700", "0800", "0900", "1000", "1100", "1200", "exaa_only"]
rmse = {k: float(np.sqrt(((df[k] - df.y_true) ** 2).mean())) for k in files}
mae = {k: float((df[k] - df.y_true).abs().mean()) for k in files}
p07, pprev = {}, {}
for i, k in enumerate(rows):
    if k == "0700":
        continue
    p07[k] = gw(df, "0700", k)
    prev = "1200" if k == "exaa_only" else rows[i - 1]
    pprev[k] = gw(df, prev, k)
h07, hprev = holm(p07), holm(pprev)
print(f"{'cutoff':10s} {'RMSE':>7s} {'MAE':>7s} {'p07':>8s} {'p07Holm':>8s} {'pprev':>8s} {'pprevH':>8s}")
for k in rows:
    f = lambda d: f"{d[k]:8.4f}" if k in d else "     ---"
    print(f"{k:10s} {rmse[k]:7.2f} {mae[k]:7.2f} {f(p07)} {f(h07)} {f(pprev)} {f(hprev)}")
print("reserve ablation:")
abl = {}
for c in ("0900", "1000", "1100", "1200"):
    nr = f"{c}_nores"
    d = (rmse[c] - rmse[nr]) / rmse[nr] * 100
    abl[c] = gw(df, nr, c) if c != "1200" else None
    ps = f"{abl[c]:.4f}" if abl[c] is not None else "---"
    print(f"  {c}: no-res {rmse[nr]:.2f}  with {rmse[c]:.2f}  d={d:+.1f}%  p={ps}")
ha = holm({k: v for k, v in abl.items() if v is not None})
print("  Holm:", {k: round(v, 4) for k, v in ha.items()})
