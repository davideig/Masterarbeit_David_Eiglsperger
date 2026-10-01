from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv

from da_price_forecasting.data.entsoe import fetch_prices


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent
TARGET_TZ = "Europe/Berlin"
START_DAY = pd.Timestamp("2025-10-01", tz=TARGET_TZ)
END_DAY = pd.Timestamp("2026-07-31", tz=TARGET_TZ)
TEST_START = pd.Timestamp("2026-02-01", tz=TARGET_TZ)


def _market_time_unit(index: pd.DatetimeIndex) -> pd.Index:
    local = index.tz_convert(TARGET_TZ)
    return pd.Index(local.hour * 4 + local.minute // 15 + 1, name="MTU")


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    prices = fetch_prices(
        start_day=START_DAY,
        end_day=END_DAY,
        target_tz=TARGET_TZ,
        chunk_days=31,
    )
    series = prices["price_da"].dropna().sort_index()

    daily = series.groupby(series.index.normalize()).agg(["mean", "min", "max"])
    daily["ma7"] = daily["mean"].rolling(7, min_periods=1).mean()
    intraday = series.groupby(_market_time_unit(series.index)).mean()

    min_price = series.min()
    max_price = series.max()
    negative_share = (series < 0).mean() * 100
    print(f"sample_start={series.index.min()}")
    print(f"sample_end={series.index.max()}")
    print(f"observations={series.size}")
    print(f"min_price={min_price:.2f}")
    print(f"max_price={max_price:.2f}")
    print(f"negative_share={negative_share:.1f}")

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

    fig = plt.figure(figsize=(7.2, 5.7), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.15, 1])
    ax_series = fig.add_subplot(grid[0, :])
    ax_hist = fig.add_subplot(grid[1, 0])
    ax_profile = fig.add_subplot(grid[1, 1])

    ax_series.fill_between(
        daily.index,
        daily["min"].to_numpy(),
        daily["max"].to_numpy(),
        color="#c7d8ea",
        alpha=0.55,
        linewidth=0,
        label="daily min-max range",
    )
    ax_series.plot(daily.index, daily["mean"], color="#275d8c", linewidth=0.9, label="daily mean")
    ax_series.plot(daily.index, daily["ma7"], color="#b13b2e", linewidth=1.2, label="7-day moving average")
    ax_series.axvspan(TEST_START, daily.index.max(), color="#e6e6e6", alpha=0.45, linewidth=0)
    ax_series.set_ylabel("Day-ahead electricity price (EUR/MWh)")
    ax_series.set_title("(a) Daily price level and range")
    ax_series.legend(loc="upper right", frameon=False, ncol=3)
    ax_series.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax_series.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax_series.set_xlim(daily.index.min(), daily.index.max())
    ax_series.grid(axis="y", color="#d9d9d9", linewidth=0.5)

    ax_hist.hist(series, bins=70, color="#6f98bd", edgecolor="white", linewidth=0.3)
    ax_hist.axvline(0, color="#333333", linewidth=0.8)
    ax_hist.set_title("(b) Price distribution")
    ax_hist.set_xlabel("Day-ahead electricity price (EUR/MWh)")
    ax_hist.set_ylabel("MTU count (-)")
    ax_hist.grid(axis="y", color="#d9d9d9", linewidth=0.5)

    ax_profile.plot(intraday.index, intraday.to_numpy(), color="#2f6c54", linewidth=1.3)
    ax_profile.set_title("(c) Average intraday profile")
    ax_profile.set_xlabel("MTU (-)")
    ax_profile.set_ylabel("Day-ahead electricity price (EUR/MWh)")
    ax_profile.set_xlim(1, 96)
    ax_profile.set_xticks([1, 24, 48, 72, 96])
    ax_profile.grid(color="#d9d9d9", linewidth=0.5)

    for ax in (ax_series, ax_hist, ax_profile):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for suffix in ("pdf", "png"):
        fig.savefig(OUTPUT_DIR / f"target_price_characteristics.{suffix}", dpi=300)


if __name__ == "__main__":
    main()
