from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).resolve().parent

# RQ3 cutoff-grid point-forecast results (Table: tab:rq3_cutoff_results).
# (creation-time hour, RMSE [EUR/MWh], MAE [EUR/MWh])
CUTOFFS = [
    (7, 32.63, 18.34),
    (8, 32.83, 18.31),
    (9, 32.39, 18.25),
    (10, 31.78, 17.81),
    (11, 30.81, 17.53),
    (12, 27.58, 13.72),
]
# Information added at each creation time (realized data advance by one hour at every step).
ADDED = {
    7: "Baseline",
    8: "Later data",
    9: "+ FCR",
    10: "+ aFRR",
    11: "+ mFRR\n+ ENTSO-E load",
    12: "+ EXAA",
}
# Compact EXAA-only benchmark at the 12:00 cutoff.
EXAA_ONLY_RMSE, EXAA_ONLY_MAE = 27.13, 13.62


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    hours = [h for h, _, _ in CUTOFFS]
    rmse = [r for _, r, _ in CUTOFFS]
    mae = [m for _, _, m in CUTOFFS]

    fig, ax = plt.subplots(figsize=(5.8, 3.9), constrained_layout=True)
    ax.set_ylim(12, 36)

    # Mark the 12:00 cutoff, where EXAA prices first become admissible.
    ax.axvspan(11.55, 12.6, color="#f1ebe9", alpha=0.9, linewidth=0, zorder=0)

    ax.plot(hours, rmse, color="#275d8c", marker="o", markersize=4.5,
            linewidth=1.4, label="RMSE", zorder=3)
    ax.plot(hours, mae, color="#2f6c54", marker="s", markersize=4.0,
            linewidth=1.4, label="MAE", zorder=3)

    # EXAA-only benchmark at 12:00 (hollow markers, slightly offset).
    ax.scatter([12.22], [EXAA_ONLY_RMSE], facecolors="none",
               edgecolors="#275d8c", s=50, linewidths=1.2, zorder=4)
    ax.scatter([12.22], [EXAA_ONLY_MAE], facecolors="none",
               edgecolors="#2f6c54", s=50, linewidths=1.2, zorder=4)
    ax.annotate("EXAA-only", (12.22, EXAA_ONLY_MAE), xytext=(0, -11),
                textcoords="offset points", ha="center", fontsize=7,
                color="#333333")

    ax.set_xticks(hours)
    ax.set_xticklabels([f"{h:02d}:00" for h in hours])
    # Second tick row: information added at each creation time.
    for h in hours:
        ax.annotate(ADDED[h], (h, 0), xycoords=("data", "axes fraction"),
                    xytext=(0, -17), textcoords="offset points", ha="center",
                    va="top", fontsize=7, color="#7a3b33" if h == 12 else "#555555",
                    linespacing=1.1)
    ax.set_xlim(6.6, 12.75)
    ax.set_xlabel("Forecast creation time on the day before delivery", labelpad=24)
    ax.set_ylabel("Forecast error (EUR/MWh)")
    ax.grid(axis="y", color="#e3e3e3", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="center left", frameon=False)

    for suffix in ("pdf", "png"):
        fig.savefig(OUTPUT_DIR / f"rq3_cutoff_accuracy.{suffix}", dpi=300)


if __name__ == "__main__":
    main()
