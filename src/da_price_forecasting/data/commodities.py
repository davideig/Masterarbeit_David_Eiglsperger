from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import pandas as pd
import requests


@dataclass(frozen=True)
class YFinanceCommodityInstrument:
    name: str
    ticker: str
    column: str | None = None
    price_field: str = "Close"


@dataclass(frozen=True)
class InvestinyCommodityInstrument:
    name: str
    investing_id: int
    column: str | None = None
    price_field: str = "close"


def _require_yfinance():
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError(
            "Commodity covariates with provider='yfinance' require yfinance. "
            "Install it with `pixi install -e tabpfn` after updating the environment."
        ) from exc
    return yf


def _as_target_tz(timestamp: pd.Timestamp, target_tz: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(timestamp)
    if timestamp.tz is None:
        return timestamp.tz_localize(target_tz)
    return timestamp.tz_convert(target_tz)


def _yfinance_instrument_from_mapping(mapping: Mapping[str, Any]) -> YFinanceCommodityInstrument:
    return YFinanceCommodityInstrument(
        name=str(mapping["name"]),
        ticker=str(mapping["ticker"]),
        column=mapping.get("column"),
        price_field=str(mapping.get("price_field", "Close")),
    )


def _investiny_instrument_from_mapping(mapping: Mapping[str, Any]) -> InvestinyCommodityInstrument:
    return InvestinyCommodityInstrument(
        name=str(mapping["name"]),
        investing_id=int(mapping["investing_id"]),
        column=mapping.get("column"),
        price_field=str(mapping.get("price_field", "close")).lower(),
    )


def _select_price_field(raw: pd.DataFrame, ticker: str, price_field: str) -> pd.Series:
    if raw.empty:
        return pd.Series(dtype=float)

    if isinstance(raw.columns, pd.MultiIndex):
        if price_field in raw.columns.get_level_values(0):
            selected = raw[price_field]
            if isinstance(selected, pd.DataFrame):
                if ticker in selected.columns:
                    return selected[ticker]
                return selected.iloc[:, 0]
        if price_field in raw.columns.get_level_values(-1):
            selected = raw.xs(price_field, axis=1, level=-1, drop_level=False)
            return selected.iloc[:, 0]

    if price_field in raw.columns:
        return raw[price_field]
    if "Close" in raw.columns:
        return raw["Close"]
    if "Adj Close" in raw.columns:
        return raw["Adj Close"]

    numeric = raw.select_dtypes("number")
    if numeric.empty:
        return pd.Series(dtype=float)
    return numeric.iloc[:, 0]


def _fetch_yfinance_daily_series(
    instrument: YFinanceCommodityInstrument,
    query_start: pd.Timestamp,
    query_end: pd.Timestamp,
    target_tz: str,
    auto_adjust: bool,
    progress: bool,
) -> pd.Series:
    yf = _require_yfinance()
    raw = yf.download(
        instrument.ticker,
        start=query_start.date().isoformat(),
        end=(query_end + pd.Timedelta(days=1)).date().isoformat(),
        interval="1d",
        auto_adjust=auto_adjust,
        progress=progress,
    )
    if raw.empty:
        return pd.Series(dtype=float, name=instrument.column or instrument.name)

    series = _select_price_field(raw, instrument.ticker, instrument.price_field)
    series = pd.to_numeric(series, errors="coerce").dropna()
    if series.empty:
        return pd.Series(dtype=float, name=instrument.column or instrument.name)

    index = pd.to_datetime(series.index)
    if getattr(index, "tz", None) is None:
        index = index.tz_localize(target_tz)
    else:
        index = index.tz_convert(target_tz)

    series.index = pd.DatetimeIndex(index).normalize()
    series = series[~series.index.duplicated(keep="last")].sort_index()
    series.name = instrument.column or instrument.name
    return series.astype(float)


def fetch_yfinance_commodity_prices(
    start_day: pd.Timestamp,
    end_day: pd.Timestamp,
    instruments: list[YFinanceCommodityInstrument | Mapping[str, Any]],
    lag_days: int = 2,
    target_tz: str = "Europe/Berlin",
    lookback_days: int = 14,
    auto_adjust: bool = False,
    progress: bool = False,
) -> pd.DataFrame:
    """Fetch yfinance commodity proxy prices and align them to a 15-minute feature grid."""
    if not instruments:
        return pd.DataFrame()

    start_day = _as_target_tz(start_day, target_tz).normalize()
    end_day = _as_target_tz(end_day, target_tz).normalize()
    query_start = start_day - pd.Timedelta(days=int(lag_days) + int(lookback_days))
    query_end = end_day

    daily_series = []
    for raw_instrument in instruments:
        instrument = (
            raw_instrument
            if isinstance(raw_instrument, YFinanceCommodityInstrument)
            else _yfinance_instrument_from_mapping(raw_instrument)
        )
        series = _fetch_yfinance_daily_series(
            instrument=instrument,
            query_start=query_start,
            query_end=query_end,
            target_tz=target_tz,
            auto_adjust=auto_adjust,
            progress=progress,
        )
        if not series.empty:
            shifted = series.copy()
            shifted.index = shifted.index + pd.Timedelta(days=int(lag_days))
            daily_series.append(shifted)

    if not daily_series:
        raise ValueError("No yfinance commodity price data returned for the configured instruments.")

    daily = pd.concat(daily_series, axis=1).sort_index()
    daily = daily.loc[:, ~daily.columns.duplicated()]
    full_index = pd.date_range(
        start=start_day,
        end=end_day + pd.Timedelta(days=1) - pd.Timedelta(minutes=15),
        freq="15min",
        tz=target_tz,
        name="timestamp",
    )
    return daily.reindex(full_index).ffill().astype(float)


def _fetch_investiny_daily_series(
    instrument: InvestinyCommodityInstrument,
    query_start: pd.Timestamp,
    query_end: pd.Timestamp,
    target_tz: str,
    request_timeout_seconds: float,
) -> pd.Series:
    url = f"https://tvc6.investing.com/{uuid4().hex}/0/0/0/0/history"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/104.0.5112.102 Safari/537.36"
        ),
        "Referer": "https://tvc-invdn-com.investing.com/",
        "Content-Type": "application/json",
    }
    response = requests.get(
        url,
        params={
            "symbol": instrument.investing_id,
            "from": int(query_start.timestamp()),
            "to": int((query_end + pd.Timedelta(days=1)).timestamp()),
            "resolution": "D",
        },
        headers=headers,
        timeout=request_timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    if not payload or payload.get("s") != "ok":
        return pd.Series(dtype=float, name=instrument.column or instrument.name)

    field_map = {
        "open": "o",
        "high": "h",
        "low": "l",
        "close": "c",
        "volume": "v",
    }
    value_key = field_map.get(instrument.price_field.lower(), instrument.price_field)
    if "t" not in payload or value_key not in payload:
        return pd.Series(dtype=float, name=instrument.column or instrument.name)

    index = pd.to_datetime(payload["t"], unit="s", utc=True).tz_convert(target_tz).normalize()
    values = pd.to_numeric(pd.Series(payload[value_key]), errors="coerce")
    series = pd.Series(values.to_numpy(dtype=float), index=index, name=instrument.column or instrument.name)
    series = series.dropna()
    series = series[series.index.notna()]
    if series.empty:
        return pd.Series(dtype=float, name=instrument.column or instrument.name)

    series = series[~series.index.duplicated(keep="last")].sort_index()
    return series.astype(float)


def fetch_investiny_commodity_prices(
    start_day: pd.Timestamp,
    end_day: pd.Timestamp,
    instruments: list[InvestinyCommodityInstrument | Mapping[str, Any]],
    lag_days: int = 2,
    target_tz: str = "Europe/Berlin",
    lookback_days: int = 14,
    request_timeout_seconds: float = 30.0,
) -> pd.DataFrame:
    """Fetch Investing.com commodity prices and align them to a 15-minute feature grid."""
    if not instruments:
        return pd.DataFrame()

    start_day = _as_target_tz(start_day, target_tz).normalize()
    end_day = _as_target_tz(end_day, target_tz).normalize()
    query_start = start_day - pd.Timedelta(days=int(lag_days) + int(lookback_days))
    query_end = end_day

    daily_series = []
    for raw_instrument in instruments:
        instrument = (
            raw_instrument
            if isinstance(raw_instrument, InvestinyCommodityInstrument)
            else _investiny_instrument_from_mapping(raw_instrument)
        )
        series = _fetch_investiny_daily_series(
            instrument=instrument,
            query_start=query_start,
            query_end=query_end,
            target_tz=target_tz,
            request_timeout_seconds=request_timeout_seconds,
        )
        if not series.empty:
            shifted = series.copy()
            shifted.index = shifted.index + pd.Timedelta(days=int(lag_days))
            daily_series.append(shifted)

    if not daily_series:
        raise ValueError("No investiny commodity price data returned for the configured instruments.")

    daily = pd.concat(daily_series, axis=1).sort_index()
    daily = daily.loc[:, ~daily.columns.duplicated()]
    full_index = pd.date_range(
        start=start_day,
        end=end_day + pd.Timedelta(days=1) - pd.Timedelta(minutes=15),
        freq="15min",
        tz=target_tz,
        name="timestamp",
    )
    return daily.reindex(full_index).ffill().astype(float)
