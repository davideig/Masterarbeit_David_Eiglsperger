from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

OUTPUT_DIR = Path(__file__).resolve().parent
LONG_CSV = (
    Path(__file__).resolve().parents[2]
    / "results/feature_importance/rq2_clean/pgen_beeswarm_long.csv"
)

# Curated diverse top features spanning the groups of the price forecast with
# OBTF inputs (exported with scripts/shap_beeswarm_export.py, 2026-02-07
# excluded), so the beeswarm shows one direction per driver rather than several
# near-duplicate per-MTU price lags.
FEATURES = [
    ("price_d1_mtu_95", "Lagged price, MTU 95"),
    ("price_d1_mtu_91", "Lagged price, MTU 91"),
    ("Residual_Load_Proxy_MW_mtu_28", "Residual load, MTU 28"),
    ("Residual_Load_Proxy_MW_mtu_60", "Residual load, MTU 60"),
    ("Residual_Load_Proxy_MW_mtu_92", "Residual load, MTU 92"),
    ("Renewable_Wind_Proxy_MW_mtu_92", "Wind generation, MTU 92"),
]


def main() -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.size": 9, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 9,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    df = pd.read_csv(LONG_CSV)
    cmap = plt.get_cmap("coolwarm")
    fig, ax = plt.subplots(figsize=(6.2, 3.6), constrained_layout=True)

    rng = np.random.default_rng(0)
    labels = [lbl for _, lbl in FEATURES]
    for row, (feat, _lbl) in enumerate(FEATURES):
        s = df[df.feature == feat]
        x = s["contribution"].to_numpy()
        v = s["feature_value"].to_numpy()
        # normalize feature value to [0, 1] per feature for the colour gradient,
        # clipped to the 5th--95th percentile so outliers do not dominate the scale
        lo, hi = np.percentile(v, [5, 95])
        vn = np.clip((v - lo) / (hi - lo), 0, 1) if hi > lo else np.full_like(v, 0.5)
        y = row + (rng.random(len(x)) - 0.5) * 0.34
        ax.scatter(x, y, c=vn, cmap=cmap, s=6, alpha=0.55, linewidths=0, rasterized=True)

    ax.axvline(0, color="#888888", linewidth=0.8, zorder=0)
    ax.set_yticks(range(len(FEATURES)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("SHAP contribution to the price forecast (scaled target)")
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", length=0)

    sm = ScalarMappable(norm=Normalize(0, 1), cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.045)
    cb.set_ticks([0, 1])
    cb.set_ticklabels(["low", "high"])
    cb.set_label("Feature value", rotation=90)

    for suffix in ("pdf", "png"):
        fig.savefig(OUTPUT_DIR / f"rq2_shap_beeswarm.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
