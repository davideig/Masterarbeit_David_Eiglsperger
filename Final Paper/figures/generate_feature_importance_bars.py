from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).resolve().parent

# Grouped SHAP contribution shares (percent of mean absolute grouped
# contribution) from output/diagnostics/price_feature_importance/paper_core_allmtu/
# (RQ2, 09:00, EXAA-only) and paper_core_allmtu_run06_stream/ (10:00-12:00, 06 UTC run)
# (all delivery days x all 96 market time units of the evaluation window). Each
# model's shares sum to ~100.
RQ2 = {
    r"$\hat{P}^{\mathrm{base}}$": {
        "Price lags": 35.4, "Raw weather": 43.1, "ENTSO-E load forecast": 21.4,
        "Calendar": 0.2,
    },
    r"$\hat{P}^{\mathrm{gen}}$": {
        "Price lags": 24.7, "Residual load forecast": 42.3, "Raw weather": 13.0,
        "Renewable forecast": 15.3, "Load forecast": 4.6, "Calendar": 0.1,
    },
    r"$\hat{P}^{\mathrm{gen},-\mathrm{wea}}$": {
        "Price lags": 27.9, "Residual load forecast": 45.4,
        "Renewable forecast": 21.7, "Load forecast": 4.9, "Calendar": 0.1,
    },
}

RQ3 = {
    "09:00": {
        "Residual load forecast": 41.9, "Price lags": 24.1, "Renewable forecast": 14.4,
        "Raw weather": 11.8, "Reserve market": 4.4, "Load forecast": 3.4,
        "Calendar": 0.1,
    },
    "10:00": {
        "Residual load forecast": 34.5, "Price lags": 21.6, "Reserve market": 18.5,
        "Renewable forecast": 12.4, "Raw weather": 10.3, "Load forecast": 2.7,
    },
    "11:00": {
        "Residual load forecast": 33.2, "Price lags": 20.8, "Reserve market": 20.3,
        "Renewable forecast": 12.7, "Raw weather": 9.6, "Load forecast": 3.4,
    },
    "12:00": {
        "EXAA prices": 66.9, "Residual load forecast": 9.3, "Price lags": 7.5,
        "Reserve market": 5.7, "Raw weather": 4.7,
        "Renewable forecast": 4.7, "Load forecast": 1.3,
    },
    "12:00, EXAA-only": {"EXAA prices": 99.2, "Calendar": 0.8},
}

# Fixed stacking order and color per feature group, shared across both figures.
# Groups built from OBTF forecasts are labelled as such, so they are not confused
# with the ENTSO-E load forecast used by the base model.
RENAME = {
    "Residual load forecast": "OBTF residual-load forecast",
    "Load forecast": "OBTF load forecast",
    "Renewable forecast": "OBTF renewable forecast",
}
GROUP_ORDER = [
    "Price lags", "ENTSO-E load forecast", "OBTF residual-load forecast",
    "OBTF load forecast", "OBTF renewable forecast", "Raw weather",
    "Reserve market", "EXAA prices", "Calendar",
]
COLORS = {
    "Price lags": "#275d8c",
    "ENTSO-E load forecast": "#a99bc4",
    "OBTF residual-load forecast": "#2f6c54",
    "OBTF load forecast": "#7fb0d3",
    "OBTF renewable forecast": "#8cbf9e",
    "Raw weather": "#d9a441",
    "Reserve market": "#a4682f",
    "EXAA prices": "#7a3b33",
    "Calendar": "#b3b3b3",
}
DARK_TEXT = {"OBTF load forecast", "ENTSO-E load forecast"}
LABEL_THRESHOLD = 3.0  # segments at least this wide are labeled inside (percent)
SMALL_LABEL_MIN = 0.5  # narrower segments down to this width are labeled outside
                       # (below this a segment would round to 0, so it is left bare)


def _plot(data: dict, out_stem: str, height: float) -> None:
    data = {m: {RENAME.get(g, g): v for g, v in shares.items()} for m, shares in data.items()}
    models = list(data.keys())
    groups = [g for g in GROUP_ORDER if any(g in shares for shares in data.values())]
    y = range(len(models))

    fig, ax = plt.subplots(figsize=(6.2, height), constrained_layout=True)
    left = [0.0] * len(models)
    for group in groups:
        widths = [data[m].get(group, 0.0) for m in models]
        ax.barh(y, widths, left=left, height=0.62, color=COLORS[group],
                edgecolor="white", linewidth=0.6, label=group)
        for i, (w, l) in enumerate(zip(widths, left)):
            cx = l + w / 2
            if w >= LABEL_THRESHOLD:
                txt = "#20303f" if group in DARK_TEXT else "#ffffff"
                ax.text(cx, i, f"{w:.0f}", ha="center", va="center",
                        fontsize=7.5, color=txt)
            elif w >= SMALL_LABEL_MIN:
                ax.annotate(f"{w:.0f}", xy=(cx, i - 0.32), xytext=(cx, i - 0.52),
                            ha="center", va="center", fontsize=7, color="#333333",
                            arrowprops=dict(arrowstyle="-", lw=0.5, color="#8a8a8a",
                                            shrinkA=1.0, shrinkB=1.0))
        left = [l + w for l, w in zip(left, widths)]

    ax.set_yticks(list(y))
    ax.set_yticklabels(models)
    ax.invert_yaxis()
    ax.set_ylim(len(models) - 0.45, -0.9)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of grouped feature contribution (%)")
    ax.tick_params(axis="y", length=0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18 if height < 3 else -0.14),
              ncol=3, frameon=False, fontsize=8, handlelength=1.1,
              columnspacing=1.2, handletextpad=0.5)

    for suffix in ("pdf", "png"):
        fig.savefig(OUTPUT_DIR / f"{out_stem}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.size": 9, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 9,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    _plot(RQ2, "rq2_feature_importance_bars", 2.7)
    _plot(RQ3, "rq3_feature_importance_bars", 3.4)


if __name__ == "__main__":
    main()
