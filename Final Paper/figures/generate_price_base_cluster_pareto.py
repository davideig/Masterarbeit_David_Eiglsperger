from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).resolve().parent

# Development-period base-price-model sweep (70-day window, 100 boosting rounds).
# (weather clusters, RMSE [EUR/MWh], mean training time [s/day])
POINTS = [
    (1, 24.21, 12),
    (2, 23.70, 17),
    (5, 24.02, 31),
    (8, 23.96, 49),
    (12, 24.31, 79),
    (16, 24.33, 110),
    (25, 24.56, 189),
    (40, 25.43, 363),
    (64, 25.12, 758),
    (100, 25.49, 1712),
]


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    clusters = [c for c, _, _ in POINTS]
    rmse = [r for _, r, _ in POINTS]
    times = [t for _, _, t in POINTS]

    fig, ax = plt.subplots(figsize=(5.6, 3.5), constrained_layout=True)
    ax.plot(times, rmse, color="#8a8a8a", linewidth=0.9, zorder=1)
    ax.scatter(times, rmse, s=26, color="#275d8c", zorder=2)

    # Highlight the selected two-cluster representation.
    sel_t, sel_r = times[1], rmse[1]
    ax.scatter([sel_t], [sel_r], s=90, facecolors="none", edgecolors="#b13b2e",
               linewidths=1.6, zorder=3)

    for c, r, t in POINTS:
        dy = 0.045 if c not in (2,) else -0.075
        va = "bottom" if c not in (2,) else "top"
        ax.annotate(f"{c}", (t, r), xytext=(0, 6 if va == "bottom" else -6),
                    textcoords="offset points", ha="center", va=va, fontsize=7.5,
                    color="#333333")

    ax.set_xscale("log")
    ax.set_xlabel("Mean training time per delivery day (s, log scale)")
    ax.set_ylabel("Development-period RMSE (EUR/MWh)")
    ax.grid(True, which="both", color="#e2e2e2", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for suffix in ("pdf", "png"):
        fig.savefig(OUTPUT_DIR / f"price_base_cluster_pareto.{suffix}", dpi=300)


if __name__ == "__main__":
    main()
