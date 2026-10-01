from __future__ import annotations

import os
import re
import time
from datetime import timedelta
from pathlib import Path

import pandas as pd
from entsoe import EntsoePandasClient, EntsoeRawClient
from entsoe.parsers import parse_prices

SOLAR_CONTROL_AREA_TARGETS = {
    "Solar_50Hertz_Actual_MW": "10YDE-VE-------2",
    "Solar_Amprion_Actual_MW": "10YDE-RWENET---I",
    "Solar_TenneT_Actual_MW": "10YDE-EON------1",
    "Solar_TransnetBW_Actual_MW": "10YDE-ENBW-----N",
}


def _require_api_key(env_var: str) -> str:
    api_key = os.getenv(env_var)
    if api_key is None:
        raise ValueError(f"Environment variable '{env_var}' is not set.")
    return api_key


def _expand_hourly_series_to_quarter_hour(
    series: pd.Series,
    target_tz: str,
    value_name: str,
    *,
    require_complete_days: bool = True,
) -> pd.DataFrame:
    if series.empty:
        raise ValueError(f"No data returned for '{value_name}'.")

    full_index = pd.date_range(
        start=_local_day_start(series.index.min(), target_tz),
        end=_next_local_day_start(series.index.max(), target_tz) - pd.Timedelta(minutes=15),
        freq="15min",
        tz=target_tz,
    )
    series_15 = series.reindex(full_index).ffill(limit=3)

    if require_complete_days:
        expected_counts = pd.Series(1, index=full_index).groupby(full_index.normalize()).transform("count")
        actual_counts = series_15.groupby(series_15.index.normalize()).transform("count")
        series_15 = series_15.where(actual_counts >= expected_counts)

    return series_15.to_frame(name=value_name).rename_axis("timestamp").sort_index()


def _restrict_calendar_window(df: pd.DataFrame, start_day: pd.Timestamp, end_day: pd.Timestamp, target_tz: str) -> pd.DataFrame:
    start_cut = _local_day_start(start_day, target_tz)
    end_cut = _next_local_day_start(end_day, target_tz) - pd.Timedelta(minutes=15)
    return df.loc[start_cut:end_cut]


def _combine_parsed_price_series(parsed) -> pd.Series | None:
    if not isinstance(parsed, dict):
        return parsed

    series_parts = [series for series in parsed.values() if series is not None and len(series) > 0]
    if not series_parts:
        return None
    return pd.concat(series_parts).sort_index()


def _as_target_tz(timestamp: pd.Timestamp, target_tz: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(timestamp)
    if timestamp.tz is None:
        return timestamp.tz_localize(target_tz)
    return timestamp.tz_convert(target_tz)


def _local_day_start(timestamp: pd.Timestamp, target_tz: str) -> pd.Timestamp:
    local_timestamp = _as_target_tz(timestamp, target_tz)
    return pd.Timestamp(local_timestamp.date(), tz=target_tz)


def _next_local_day_start(timestamp: pd.Timestamp, target_tz: str) -> pd.Timestamp:
    local_timestamp = _as_target_tz(timestamp, target_tz)
    return pd.Timestamp(local_timestamp.date() + timedelta(days=1), tz=target_tz)


def _first_numeric_series(
    obj,
    value_name: str,
    preferred_tokens: tuple[str, ...] = (),
) -> pd.Series:
    if isinstance(obj, pd.DataFrame):
        numeric_cols = obj.select_dtypes(include="number").columns.tolist()
        if not numeric_cols:
            raise ValueError(f"No numeric columns found in {value_name} response: {obj.columns.tolist()}")
        if preferred_tokens:
            for candidate in numeric_cols:
                label = " ".join(str(part).lower() for part in (candidate if isinstance(candidate, tuple) else (candidate,)))
                if all(token in label for token in preferred_tokens):
                    return obj[candidate].copy()
        return obj[numeric_cols[0]].copy()
    return obj.copy()


def _feature_suffix(value: object) -> str:
    suffix = re.sub(r"[^0-9A-Za-z]+", "_", str(value).strip())
    suffix = re.sub(r"_+", "_", suffix).strip("_")
    return suffix or "Unknown"


def _entsoe_cache_path(kind: str, country_code: str, start_day, end_day, target_tz: str) -> Path:
    """Disk-cache path keyed by the exact request range (historical data is final)."""
    base = Path(os.environ.get("ENTSOE_CACHE_DIR", "data/cache/entsoe"))
    tz_tag = str(target_tz).replace("/", "-")
    start = pd.Timestamp(start_day).date()
    end = pd.Timestamp(end_day).date()
    return base / f"{kind}_{country_code}_{start}_{end}_{tz_tag}.parquet"


def _read_entsoe_cache(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        print(f"[entsoe-cache] hit: {path.name}")
        return df
    except Exception:
        return None


def _write_entsoe_cache(df: pd.DataFrame, path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
    except Exception:
        pass


def _query_with_retry(query_fn, *, label: str, attempts: int = 5):
    """Call an ENTSO-E query, retrying transient (503/502/504/timeout) errors with backoff.

    The API is frequently overloaded and returns intermittent 503s for a query that succeeds
    on retry, so a single blip should not kill the run.
    """
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return query_fn()
        except Exception as exc:
            last_exc = exc
            message = str(exc).lower()
            transient = any(
                token in message
                for token in ("503", "502", "504", "timed out", "service unavailable", "connection")
            )
            if transient and attempt < attempts:
                wait = min(60, 5 * 2 ** (attempt - 1))
                print(f"[entsoe] transient error on {label} (attempt {attempt}/{attempts}); retrying in {wait}s")
                time.sleep(wait)
                continue
            raise
    raise last_exc  # pragma: no cover - loop always returns or raises


def fetch_prices(
    start_day: pd.Timestamp,
    end_day: pd.Timestamp,
    country_code: str = "DE_LU",
    api_key_env: str = "ENTSOE_API_KEY",
    target_tz: str = "Europe/Berlin",
    chunk_days: int | None = 30,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch DE-LU SDAC day-ahead prices as a 15-minute Europe/Berlin series.

    The result is cached to disk keyed by the exact request range, so repeated runs over the
    same window (e.g. a config sweep) reuse one fetch instead of re-hitting the ENTSO-E API.
    Operational runs whose end date advances daily naturally get a fresh key and re-fetch.
    """
    cache_path = _entsoe_cache_path("prices", country_code, start_day, end_day, target_tz)
    if use_cache:
        cached = _read_entsoe_cache(cache_path)
        if cached is not None:
            return cached

    client = EntsoePandasClient(api_key=_require_api_key(api_key_env))

    def query_chunk(chunk_start: pd.Timestamp, chunk_end: pd.Timestamp) -> list[pd.Series]:
        query_start = chunk_start.tz_convert(target_tz)
        query_end = (chunk_end + pd.Timedelta(days=1)).tz_convert(target_tz)

        # Retry transient server errors (503/502/504/timeout) with backoff before falling back
        # to bisection. The ENTSO-E API is often overloaded and returns intermittent 503s for
        # a query that succeeds on retry, so retrying avoids cascading one blip into failure.
        last_exc: Exception | None = None
        for attempt in range(1, 6):
            try:
                raw_chunk = client.query_day_ahead_prices(country_code, start=query_start, end=query_end)
                if raw_chunk.empty:
                    return []
                return [raw_chunk]
            except Exception as exc:
                last_exc = exc
                message = str(exc).lower()
                transient = any(
                    token in message
                    for token in ("503", "502", "504", "timed out", "service unavailable", "connection")
                )
                if transient and attempt < 5:
                    wait = min(60, 5 * 2 ** (attempt - 1))
                    print(
                        f"[entsoe] transient error on {chunk_start.date()}..{chunk_end.date()} "
                        f"(attempt {attempt}/5); retrying in {wait}s"
                    )
                    time.sleep(wait)
                    continue
                break

        # Persistent failure: split the range (handles too-large ranges), else give up.
        if chunk_start.normalize() >= chunk_end.normalize():
            raise last_exc
        midpoint = chunk_start + pd.Timedelta(days=(chunk_end.normalize() - chunk_start.normalize()).days // 2)
        left = query_chunk(chunk_start, midpoint)
        right = query_chunk(midpoint + pd.Timedelta(days=1), chunk_end)
        return left + right

    all_series = []
    current_start = start_day.normalize()

    while current_start <= end_day:
        current_end = end_day
        if chunk_days is not None:
            current_end = min(current_start + pd.Timedelta(days=chunk_days - 1), end_day)

        all_series.extend(query_chunk(current_start, current_end))

        if chunk_days is None:
            break
        current_start = current_end + pd.Timedelta(days=1)

    if not all_series:
        raise ValueError("No day-ahead price data returned by the ENTSO-E API.")

    raw_series = pd.concat(all_series).sort_index()
    raw_series = raw_series[~raw_series.index.duplicated(keep="last")]
    raw_series = raw_series.tz_convert(target_tz).rename("price_da")
    df_prices_15 = _expand_hourly_series_to_quarter_hour(raw_series, target_tz=target_tz, value_name="price_da")
    result = _restrict_calendar_window(df_prices_15, start_day, end_day, target_tz)
    if use_cache:
        _write_entsoe_cache(result, cache_path)
    return result


def fetch_foreign_day_ahead_prices(
    start_day: pd.Timestamp,
    end_day: pd.Timestamp,
    markets: list[str],
    api_key_env: str = "ENTSOE_API_KEY",
    target_tz: str = "Europe/Berlin",
    chunk_days: int = 90,
) -> pd.DataFrame:
    """Fetch selected foreign day-ahead price curves as 15-minute covariates."""
    if not markets:
        return pd.DataFrame()

    frames = []
    for market in markets:
        market_prices = fetch_prices(
            start_day=start_day,
            end_day=end_day,
            country_code=market,
            api_key_env=api_key_env,
            target_tz=target_tz,
            chunk_days=chunk_days,
        )
        safe_market = market.replace(" ", "_").replace("-", "_").replace("/", "_")
        frames.append(market_prices.rename(columns={"price_da": f"price_da_{safe_market}"}))

    return pd.concat(frames, axis=1).sort_index().rename_axis("timestamp")


def fetch_prices_exaa(
    start_day: pd.Timestamp,
    end_day: pd.Timestamp,
    country_code: str = "DE_LU",
    api_key_env: str = "ENTSOE_API_KEY",
    target_tz: str = "Europe/Berlin",
    chunk_days: int = 90,
) -> pd.DataFrame:
    """Fetch DE-LU EXAA day-ahead prices as a 15-minute Europe/Berlin series."""
    client = EntsoeRawClient(api_key=_require_api_key(api_key_env))

    all_series = []
    current_start = start_day.normalize()

    while current_start <= end_day:
        current_end = min(current_start + pd.Timedelta(days=chunk_days - 1), end_day)

        query_start = current_start.tz_convert(target_tz)
        query_end = (current_end + pd.Timedelta(days=1)).tz_convert(target_tz)

        xml = client.query_day_ahead_prices(
            country_code,
            start=query_start,
            end=query_end,
            sequence=2,
        )
        parsed = parse_prices(xml)

        chunk_series = _combine_parsed_price_series(parsed)

        if chunk_series is None or len(chunk_series) == 0:
            current_start = current_end + pd.Timedelta(days=1)
            continue

        if getattr(chunk_series.index, "tz", None) is None:
            chunk_series = chunk_series.tz_localize("UTC")

        chunk_series = chunk_series.tz_convert(target_tz).rename("price_exaa")
        all_series.append(chunk_series)
        current_start = current_end + pd.Timedelta(days=1)

    series = pd.concat(all_series).sort_index()
    series = series[~series.index.duplicated(keep="last")]

    df_prices_exaa_15 = _expand_hourly_series_to_quarter_hour(series, target_tz=target_tz, value_name="price_exaa")
    return _restrict_calendar_window(df_prices_exaa_15, start_day, end_day, target_tz)


def fetch_load_forecast(
    start_day: pd.Timestamp,
    end_day: pd.Timestamp,
    country_code: str = "DE_LU",
    api_key_env: str = "ENTSOE_API_KEY",
    target_tz: str = "Europe/Berlin",
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch DE-LU day-ahead load forecast as a 15-minute Europe/Berlin series.

    Disk-cached by request range like :func:`fetch_prices`, so a config sweep over one window
    reuses a single fetch instead of re-hitting the ENTSO-E API.
    """
    cache_path = _entsoe_cache_path("load_forecast", country_code, start_day, end_day, target_tz)
    if use_cache:
        cached = _read_entsoe_cache(cache_path)
        if cached is not None:
            return cached

    client = EntsoePandasClient(api_key=_require_api_key(api_key_env))

    query_start = _local_day_start(start_day, target_tz)
    query_end = _next_local_day_start(end_day, target_tz)

    obj = _query_with_retry(
        lambda: client.query_load_forecast(country_code, start=query_start, end=query_end),
        label=f"load_forecast {pd.Timestamp(start_day).date()}..{pd.Timestamp(end_day).date()}",
    )
    series = _first_numeric_series(obj, "load forecast", preferred_tokens=("forecast",))
    series = series.tz_convert(target_tz)
    series.name = "load_fc"
    df_load_forecast_15 = _expand_hourly_series_to_quarter_hour(series, target_tz=target_tz, value_name="load_fc")
    result = _restrict_calendar_window(df_load_forecast_15, start_day, end_day, target_tz)
    if use_cache:
        _write_entsoe_cache(result, cache_path)
    return result


def fetch_actual_load(
    start_day: pd.Timestamp,
    end_day: pd.Timestamp,
    country_code: str = "DE_LU",
    api_key_env: str = "ENTSOE_API_KEY",
    target_tz: str = "Europe/Berlin",
    chunk_days: int = 90,
    require_complete_days: bool = True,
) -> pd.DataFrame:
    """Fetch actual total load as a 15-minute Europe/Berlin series."""
    client = EntsoePandasClient(api_key=_require_api_key(api_key_env))
    start_day = _as_target_tz(start_day, target_tz)
    end_day = _as_target_tz(end_day, target_tz)

    all_series = []
    current_start = _local_day_start(start_day, target_tz)
    while current_start <= end_day:
        current_end = min(current_start + pd.Timedelta(days=chunk_days - 1), end_day)
        query_start = _local_day_start(current_start, target_tz)
        query_end = _next_local_day_start(current_end, target_tz)

        obj = client.query_load(country_code, start=query_start, end=query_end)
        if obj is not None and not obj.empty:
            series = _first_numeric_series(obj, "actual load", preferred_tokens=("actual",))
            if getattr(series.index, "tz", None) is None:
                series = series.tz_localize("UTC")
            all_series.append(series.tz_convert(target_tz).rename("load_actual"))

        current_start = _next_local_day_start(current_end, target_tz)

    if not all_series:
        raise ValueError("No actual load data returned by the ENTSO-E API.")

    raw_series = pd.concat(all_series).sort_index()
    raw_series = raw_series[~raw_series.index.duplicated(keep="last")]
    df_actual_load_15 = _expand_hourly_series_to_quarter_hour(
        raw_series,
        target_tz=target_tz,
        value_name="load_actual",
        require_complete_days=require_complete_days,
    )
    return _restrict_calendar_window(df_actual_load_15, start_day, end_day, target_tz)


def fetch_actual_renewable_generation(
    start_day: pd.Timestamp,
    end_day: pd.Timestamp,
    country_code: str = "DE_LU",
    api_key_env: str = "ENTSOE_API_KEY",
    target_tz: str = "Europe/Berlin",
    chunk_days: int = 90,
) -> pd.DataFrame:
    """Fetch actual wind/solar generation as 15-minute Europe/Berlin series."""
    client = EntsoePandasClient(api_key=_require_api_key(api_key_env))
    start_day = _as_target_tz(start_day, target_tz)
    end_day = _as_target_tz(end_day, target_tz)

    psr_types = {
        "B16": "Solar_Actual_MW",
        "B18": "Wind_Offshore_Actual_MW",
        "B19": "Wind_Onshore_Actual_MW",
    }
    frames = []

    for psr_type, column in psr_types.items():
        series_parts = []
        current_start = start_day.normalize()
        while current_start <= end_day:
            current_end = min(current_start + pd.Timedelta(days=chunk_days - 1), end_day)
            query_start = current_start.tz_convert(target_tz)
            query_end = (current_end + pd.Timedelta(days=1)).tz_convert(target_tz)

            obj = client.query_generation(
                country_code=country_code,
                start=query_start,
                end=query_end,
                psr_type=psr_type,
                nett=False,
            )
            if obj is not None and not obj.empty:
                series = _first_numeric_series(obj, f"{column} generation")
                if getattr(series.index, "tz", None) is None:
                    series = series.tz_localize("UTC")
                series_parts.append(series.tz_convert(target_tz).rename(column))

            current_start = current_end + pd.Timedelta(days=1)

        if not series_parts:
            raise ValueError(f"No actual generation data returned for {psr_type} ({column}).")

        raw_series = pd.concat(series_parts).sort_index()
        raw_series = raw_series[~raw_series.index.duplicated(keep="last")]
        frames.append(_expand_hourly_series_to_quarter_hour(raw_series, target_tz=target_tz, value_name=column))

    result = pd.concat(frames, axis=1).sort_index()
    result["Wind_Total_Actual_MW"] = result[["Wind_Onshore_Actual_MW", "Wind_Offshore_Actual_MW"]].sum(
        axis=1,
        min_count=2,
    )
    result["Renewable_Total_Actual_MW"] = result[["Solar_Actual_MW", "Wind_Total_Actual_MW"]].sum(
        axis=1,
        min_count=2,
    )
    return _restrict_calendar_window(result, start_day, end_day, target_tz)


def fetch_actual_solar_generation_by_control_area(
    start_day: pd.Timestamp,
    end_day: pd.Timestamp,
    control_area_targets: dict[str, str] | None = None,
    api_key_env: str = "ENTSOE_API_KEY",
    target_tz: str = "Europe/Berlin",
    chunk_days: int = 90,
) -> pd.DataFrame:
    """Fetch actual solar generation for German TSO control areas."""
    client = EntsoePandasClient(api_key=_require_api_key(api_key_env))
    start_day = _as_target_tz(start_day, target_tz)
    end_day = _as_target_tz(end_day, target_tz)
    targets = control_area_targets or SOLAR_CONTROL_AREA_TARGETS
    frames = []

    for column, country_code in targets.items():
        series_parts = []
        current_start = start_day.normalize()
        while current_start <= end_day:
            current_end = min(current_start + pd.Timedelta(days=chunk_days - 1), end_day)
            query_start = current_start.tz_convert(target_tz)
            query_end = (current_end + pd.Timedelta(days=1)).tz_convert(target_tz)

            obj = client.query_generation(
                country_code=country_code,
                start=query_start,
                end=query_end,
                psr_type="B16",
                nett=False,
            )
            if obj is not None and not obj.empty:
                series = _first_numeric_series(obj, f"{column} generation")
                if getattr(series.index, "tz", None) is None:
                    series = series.tz_localize("UTC")
                series_parts.append(series.tz_convert(target_tz).rename(column))

            current_start = current_end + pd.Timedelta(days=1)

        if not series_parts:
            raise ValueError(f"No actual solar generation data returned for {country_code} ({column}).")

        raw_series = pd.concat(series_parts).sort_index()
        raw_series = raw_series[~raw_series.index.duplicated(keep="last")]
        frames.append(_expand_hourly_series_to_quarter_hour(raw_series, target_tz=target_tz, value_name=column))

    result = pd.concat(frames, axis=1).sort_index()
    if set(targets).issubset(result.columns):
        result["Solar_Control_Area_Total_Actual_MW"] = result[list(targets)].sum(axis=1, min_count=len(targets))
    return _restrict_calendar_window(result, start_day, end_day, target_tz)


def _renewable_forecast_frame_to_series(obj, column: str) -> pd.Series:
    if isinstance(obj, pd.Series):
        return obj.rename(column)

    if not isinstance(obj, pd.DataFrame):
        raise ValueError(f"Unexpected renewable forecast response type: {type(obj).__name__}")

    numeric = obj.select_dtypes(include="number")
    if numeric.empty:
        raise ValueError(f"No numeric columns found in renewable forecast response: {obj.columns.tolist()}")

    preferred_tokens = {
        "Solar_Forecast_MW": ("solar",),
        "Wind_Offshore_Forecast_MW": ("offshore",),
        "Wind_Onshore_Forecast_MW": ("onshore",),
    }.get(column, ())
    if preferred_tokens:
        for candidate in numeric.columns:
            label = " ".join(str(part).lower() for part in (candidate if isinstance(candidate, tuple) else (candidate,)))
            if all(token in label for token in preferred_tokens):
                return numeric[candidate].rename(column)

    return numeric.iloc[:, 0].rename(column)


def fetch_renewable_generation_forecast(
    start_day: pd.Timestamp,
    end_day: pd.Timestamp,
    country_code: str = "DE_LU",
    api_key_env: str = "ENTSOE_API_KEY",
    target_tz: str = "Europe/Berlin",
    chunk_days: int = 90,
    process_type: str = "A01",
) -> pd.DataFrame:
    """Fetch ENTSO-E day-ahead wind/solar generation forecasts as 15-minute series."""
    client = EntsoePandasClient(api_key=_require_api_key(api_key_env))
    start_day = _as_target_tz(start_day, target_tz)
    end_day = _as_target_tz(end_day, target_tz)

    psr_types = {
        "B16": "Solar_Forecast_MW",
        "B18": "Wind_Offshore_Forecast_MW",
        "B19": "Wind_Onshore_Forecast_MW",
    }
    frames = []

    for psr_type, column in psr_types.items():
        series_parts = []
        current_start = start_day.normalize()
        while current_start <= end_day:
            current_end = min(current_start + pd.Timedelta(days=chunk_days - 1), end_day)
            query_start = current_start.tz_convert(target_tz)
            query_end = (current_end + pd.Timedelta(days=1)).tz_convert(target_tz)

            obj = client.query_wind_and_solar_forecast(
                country_code=country_code,
                start=query_start,
                end=query_end,
                psr_type=psr_type,
                process_type=process_type,
            )
            if obj is not None and not obj.empty:
                series = _renewable_forecast_frame_to_series(obj, column)
                if getattr(series.index, "tz", None) is None:
                    series = series.tz_localize("UTC")
                series_parts.append(series.tz_convert(target_tz).rename(column))

            current_start = current_end + pd.Timedelta(days=1)

        if not series_parts:
            raise ValueError(f"No renewable forecast data returned for {psr_type} ({column}).")

        raw_series = pd.concat(series_parts).sort_index()
        raw_series = raw_series[~raw_series.index.duplicated(keep="last")]
        frames.append(_expand_hourly_series_to_quarter_hour(raw_series, target_tz=target_tz, value_name=column))

    result = pd.concat(frames, axis=1).sort_index()
    result["Wind_Total_Forecast_MW"] = result[["Wind_Onshore_Forecast_MW", "Wind_Offshore_Forecast_MW"]].sum(
        axis=1,
        min_count=2,
    )
    result["Renewable_Total_Forecast_MW"] = result[["Solar_Forecast_MW", "Wind_Total_Forecast_MW"]].sum(
        axis=1,
        min_count=2,
    )
    return _restrict_calendar_window(result, start_day, end_day, target_tz)


def fetch_ntc_data(
    start_day: pd.Timestamp,
    end_day: pd.Timestamp,
    neighbors: list[str],
    country_code: str = "DE_LU",
    api_key_env: str = "ENTSOE_API_KEY",
    target_tz: str = "Europe/Berlin",
) -> pd.DataFrame:
    """Fetch day-ahead net transfer capacity as a 15-minute net import series."""
    client = EntsoePandasClient(api_key=_require_api_key(api_key_env))

    query_start = start_day.tz_convert(target_tz)
    query_end = (end_day + pd.Timedelta(days=1)).tz_convert(target_tz)

    series_by_direction = {}
    for neighbor in neighbors:
        directions = [
            (neighbor, country_code, f"NTC_Import_{neighbor}_MW"),
            (country_code, neighbor, f"NTC_Export_{neighbor}_MW"),
        ]
        for country_code_from, country_code_to, name in directions:
            try:
                obj = client.query_net_transfer_capacity_dayahead(
                    country_code_from=country_code_from,
                    country_code_to=country_code_to,
                    start=query_start,
                    end=query_end,
                )
            except Exception:
                continue

            try:
                series = _first_numeric_series(obj, "NTC day-ahead").rename(name)
            except ValueError:
                continue
            if series.empty:
                continue
            if getattr(series.index, "tz", None) is None:
                series = series.tz_localize("UTC")
            series_by_direction[name] = series.tz_convert(target_tz).sort_index()

    if not series_by_direction:
        raise ValueError("No NTC day-ahead data returned by the ENTSO-E API.")

    df = pd.DataFrame(series_by_direction).sort_index()
    df = df[~df.index.duplicated(keep="first")]

    import_cols = [col for col in df.columns if "_Import_" in col]
    export_cols = [col for col in df.columns if "_Export_" in col]
    df["NTC_Net_MW"] = df[import_cols].sum(axis=1, skipna=True) - df[export_cols].sum(axis=1, skipna=True)

    full_index = pd.date_range(
        start=query_start,
        end=query_end - pd.Timedelta(minutes=15),
        freq="15min",
        tz=target_tz,
        name="timestamp",
    )
    result = df[["NTC_Net_MW"]].reindex(full_index).ffill(limit=3).fillna(0.0)
    return result


def fetch_generation_unavailability(
    start_day: pd.Timestamp,
    end_day: pd.Timestamp,
    country_code: str = "DE_LU",
    api_key_env: str = "ENTSOE_API_KEY",
    target_tz: str = "Europe/Berlin",
    feature_mode: str = "total",
    planned_only: bool = False,
) -> pd.DataFrame:
    """
    Fetch generation unavailability from ENTSO-E as 15-minute MW series.

    The returned index is timezone-aware in ``target_tz`` and covers full
    inclusive calendar days, matching the other ENTSO-E fetchers.
    """
    start_day = _as_target_tz(start_day, target_tz)
    end_day = _as_target_tz(end_day, target_tz)
    earliest_available = pd.Timestamp("2019-06-01", tz=target_tz)

    query_start = max(start_day.tz_convert(target_tz).normalize(), earliest_available)
    query_end = (end_day + pd.Timedelta(days=1)).tz_convert(target_tz).normalize()

    if query_end <= query_start:
        return pd.DataFrame(
            columns=["Unavail_Total_MW"],
            index=pd.DatetimeIndex([], tz=target_tz, name="timestamp"),
        )

    client = EntsoePandasClient(api_key=_require_api_key(api_key_env))

    all_chunks = []
    failed_chunks = []
    successful_queries = 0
    chunk_start = query_start

    while chunk_start < query_end:
        chunk_end = min(chunk_start + pd.DateOffset(years=1), query_end)
        try:
            chunk = client.query_unavailability_of_generation_units(
                country_code=country_code,
                start=chunk_start,
                end=chunk_end,
            )
        except Exception as exc:
            failed_chunks.append((chunk_start, chunk_end, exc))
            chunk_start = chunk_end
            continue

        successful_queries += 1
        if not chunk.empty:
            all_chunks.append(chunk)
        chunk_start = chunk_end

    if successful_queries == 0 and failed_chunks:
        first_start, first_end, first_exc = failed_chunks[0]
        raise ValueError(
            "Generation unavailability could not be fetched from ENTSO-E. "
            f"First failed chunk: {first_start.date()} to {first_end.date()}."
        ) from first_exc

    full_index = pd.date_range(
        start=query_start,
        end=query_end - pd.Timedelta(minutes=15),
        freq="15min",
        tz=target_tz,
        name="timestamp",
    )
    result = pd.DataFrame(0.0, index=full_index, columns=["Unavail_Total_MW"])

    if all_chunks:
        raw = pd.concat(all_chunks, ignore_index=False).drop_duplicates()

        if "docstatus" in raw.columns:
            raw = raw[raw["docstatus"].astype(str).str.casefold() != "cancelled"]

        business_column = next(
            (column for column in raw.columns if str(column).casefold() in {"businesstype", "business_type"}),
            None,
        )
        if planned_only and business_column is not None:
            business_values = raw[business_column].astype(str).str.casefold()
            raw = raw[business_values.str.contains("planned", na=False) & ~business_values.str.contains("unplanned", na=False)]

        required_columns = {"start", "end", "nominal_power", "avail_qty"}
        missing_columns = required_columns.difference(raw.columns)
        if missing_columns:
            raise ValueError(
                "Generation unavailability response is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        nominal_power = pd.to_numeric(raw["nominal_power"], errors="coerce")
        available_power = pd.to_numeric(raw["avail_qty"], errors="coerce")
        raw = raw.assign(unavail_mw=nominal_power - available_power)

        plant_type_column = next(
            (column for column in raw.columns if str(column).casefold() in {"plant_type", "psrtype", "productiontype"}),
            None,
        )
        if feature_mode == "plant_type" and plant_type_column is not None:
            plant_types = raw.loc[raw["unavail_mw"].gt(0), plant_type_column].dropna().map(_feature_suffix).unique()
            for plant_type in sorted(plant_types):
                result[f"Unavail_{plant_type}_MW"] = 0.0

        for _, row in raw.iterrows():
            unavail_mw = row["unavail_mw"]
            if pd.isna(unavail_mw) or unavail_mw <= 0:
                continue

            outage_start = _as_target_tz(row["start"], target_tz)
            outage_end = _as_target_tz(row["end"], target_tz)
            mask = (result.index >= outage_start) & (result.index < outage_end)
            result.loc[mask, "Unavail_Total_MW"] += unavail_mw

            if feature_mode == "plant_type" and plant_type_column is not None and pd.notna(row[plant_type_column]):
                column = f"Unavail_{_feature_suffix(row[plant_type_column])}_MW"
                if column in result.columns:
                    result.loc[mask, column] += unavail_mw

    return result
