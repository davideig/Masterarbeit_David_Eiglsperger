from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from da_price_forecasting.pipelines import lear


def test_price_cache_refreshes_missing_tail(tmp_path: Path) -> None:
    cache_file = tmp_path / "prices.csv"
    cached_index = pd.date_range("2026-05-20T00:00:00+02:00", periods=96, freq="15min")
    fetched_index = pd.date_range("2026-05-21T00:00:00+02:00", periods=96, freq="15min")
    cached = pd.DataFrame({"price_da": np.arange(96, dtype=float)}, index=cached_index)
    fetched = pd.DataFrame({"price_da": 100.0 + np.arange(96, dtype=float)}, index=fetched_index)
    cached.to_csv(cache_file)
    calls: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    def fetch_window(fetch_start: pd.Timestamp, fetch_end: pd.Timestamp) -> pd.DataFrame:
        calls.append((fetch_start, fetch_end))
        return fetched

    result = lear._load_or_fetch_price_cache(
        path=cache_file,
        target_tz="Europe/Berlin",
        start=pd.Timestamp("2026-05-20", tz="Europe/Berlin"),
        end=pd.Timestamp("2026-05-21", tz="Europe/Berlin"),
        fetch_window=fetch_window,
        label="ENTSO-E day-ahead prices",
    )

    assert calls == [
        (
            pd.Timestamp("2026-05-21", tz="Europe/Berlin"),
            pd.Timestamp("2026-05-21", tz="Europe/Berlin"),
        )
    ]
    assert result.index.min() == cached_index[0]
    assert result.index.max() == fetched_index[-1]
    assert result["price_da"].iloc[-1] == fetched["price_da"].iloc[-1]


def test_price_cache_uses_available_cache_when_refresh_fails(tmp_path: Path) -> None:
    cache_file = tmp_path / "prices.csv"
    cached_index = pd.date_range("2026-05-20T00:00:00+02:00", periods=96, freq="15min")
    cached = pd.DataFrame({"price_da": np.arange(96, dtype=float)}, index=cached_index)
    cached.to_csv(cache_file)

    def fail_fetch(fetch_start: pd.Timestamp, fetch_end: pd.Timestamp) -> pd.DataFrame:
        raise RuntimeError(f"ENTSO-E down for {fetch_start.date()}..{fetch_end.date()}")

    with pytest.warns(RuntimeWarning, match="Failed to refresh ENTSO-E day-ahead prices"):
        result = lear._load_or_fetch_price_cache(
            path=cache_file,
            target_tz="Europe/Berlin",
            start=pd.Timestamp("2026-05-20", tz="Europe/Berlin"),
            end=pd.Timestamp("2026-05-21", tz="Europe/Berlin"),
            fetch_window=fail_fetch,
            label="ENTSO-E day-ahead prices",
        )

    assert result.index.min() == cached_index[0]
    assert result.index.max() == cached_index[-1]
    assert result["price_da"].notna().all()

