from __future__ import annotations

import json
import time
import warnings
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..config import (
    EntsoeLoadForecastBenchmarkConfig,
    LoadForecastEnsembleConfig,
    LoadForecastModelConfig,
    RegionalLoadForecastComponentConfig,
    RegionalLoadForecastConfig,
)
from ..data.entsoe import fetch_actual_load, fetch_load_forecast
from ..data.weather import load_dwd, load_open_meteo
from ..evaluation.metrics import pinball_score, quantile_column
from .common import (
    as_local_day as _as_local_day,
    load_timestamp_csv as _load_timestamp_csv,
    point_error_stats,
    save_timestamp_csv as _save_timestamp_csv,
)

LOAD_FORECAST_REQUIRED_COLUMNS = ("Load_Model_MW", "Load_Benchmark_MW", "Load_Actual_MW")


def _calendar_window(
    config: LoadForecastModelConfig | EntsoeLoadForecastBenchmarkConfig,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    return (
        _as_local_day(config.entsoe_start_date, config.target_tz),
        _as_local_day(config.entsoe_end_date, config.target_tz),
    )


def _current_operational_day(target_tz: str) -> pd.Timestamp:
    return pd.Timestamp.now(tz=target_tz).normalize()


def _latest_operational_actual_day(target_tz: str) -> pd.Timestamp:
    return _current_operational_day(target_tz) - pd.Timedelta(days=1)


def _partial_load_source_end_day(config: LoadForecastModelConfig) -> pd.Timestamp | None:
    if not config.include_partial_load_features or config.test_end is None:
        return None
    return _as_local_day(config.test_end, config.target_tz) - pd.Timedelta(days=config.partial_load_reference_day)


def _actual_load_fetch_window(
    config: LoadForecastModelConfig | EntsoeLoadForecastBenchmarkConfig,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    start, end = _calendar_window(config)
    if isinstance(config, LoadForecastModelConfig):
        full_actual_end = min(end, _latest_operational_actual_day(config.target_tz))
        partial_source_end = _partial_load_source_end_day(config)
        if partial_source_end is not None:
            partial_actual_end = min(partial_source_end, _current_operational_day(config.target_tz))
            full_actual_end = max(full_actual_end, partial_actual_end)
        end = min(end, full_actual_end)
    return start, end


def _partial_load_refresh_day(config: LoadForecastModelConfig, fetch_end: pd.Timestamp) -> pd.Timestamp | None:
    partial_source_end = _partial_load_source_end_day(config)
    if partial_source_end is None:
        return None
    current_day = _current_operational_day(config.target_tz)
    if partial_source_end == current_day and partial_source_end <= fetch_end:
        return partial_source_end
    return None


def _weather_present_mask(dataset: pd.DataFrame, row_mask: np.ndarray, config: LoadForecastModelConfig) -> pd.Series:
    """Training rows to keep: all rows, or only rows with weather if required."""
    index = dataset.index[row_mask]
    if not config.require_weather_for_training:
        return pd.Series(True, index=index)
    column = config.weather_presence_column
    if column not in dataset.columns:
        raise ValueError(f"require_weather_for_training is set, but column {column!r} is missing.")
    return dataset.loc[row_mask, column].notna()


def _target_availability_cutoff(day: pd.Timestamp, config: LoadForecastModelConfig) -> pd.Timestamp:
    cutoff_day = day - pd.Timedelta(days=config.target_availability_lag_days)
    if config.target_availability_cutoff_hour is None:
        return cutoff_day + pd.Timedelta(hours=23, minutes=45)
    return cutoff_day + pd.Timedelta(
        hours=config.target_availability_cutoff_hour,
        minutes=config.target_availability_cutoff_minute,
    )


def _restrict_timestamp_window(
    df: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    target_tz: str,
) -> pd.DataFrame:
    if df.empty:
        return df
    start_cut = _local_day_start(start, target_tz)
    end_cut = _next_local_day_start(end, target_tz) - pd.Timedelta(minutes=15)
    return df.loc[start_cut:end_cut]


def _combine_timestamp_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    non_empty_frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not non_empty_frames:
        return pd.DataFrame()
    df = pd.concat(non_empty_frames)
    df = df.loc[~df.index.duplicated(keep="last")]
    df = df.sort_index()
    df.index.name = "timestamp"
    return df


def _local_day_start(timestamp: pd.Timestamp, target_tz: str) -> pd.Timestamp:
    local_timestamp = pd.Timestamp(timestamp).tz_convert(target_tz)
    return pd.Timestamp(local_timestamp.date(), tz=target_tz)


def _next_local_day_start(timestamp: pd.Timestamp, target_tz: str) -> pd.Timestamp:
    local_timestamp = pd.Timestamp(timestamp).tz_convert(target_tz)
    return pd.Timestamp(local_timestamp.date() + timedelta(days=1), tz=target_tz)


def _local_days(start: pd.Timestamp, end: pd.Timestamp, target_tz: str) -> pd.DatetimeIndex:
    start_day = _local_day_start(start, target_tz)
    end_day = _local_day_start(end, target_tz)
    if start_day > end_day:
        return pd.DatetimeIndex([], tz=target_tz)
    return pd.date_range(start_day, end_day, freq="D", tz=target_tz)


def _expected_quarter_hours_for_local_day(day: pd.Timestamp, target_tz: str) -> int:
    day_start = _local_day_start(day, target_tz)
    next_day_start = _next_local_day_start(day, target_tz)
    return len(pd.date_range(day_start, next_day_start, freq="15min", inclusive="left"))


def _actual_load_days_below_count(
    actual: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    target_tz: str,
) -> list[pd.Timestamp]:
    days = _local_days(start, end, target_tz)
    if not len(days):
        return []
    if actual.empty or "load_actual" not in actual.columns:
        return list(days)

    local_index = pd.DatetimeIndex(actual.index).tz_convert(target_tz)
    counts = actual["load_actual"].notna().groupby(local_index.normalize()).sum()
    return [
        day
        for day in days
        if int(counts.get(day, 0)) < _expected_quarter_hours_for_local_day(day, target_tz)
    ]


def _group_consecutive_days(days: list[pd.Timestamp]) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if not days:
        return []
    sorted_days = sorted(pd.Timestamp(day) for day in days)
    ranges: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    start = previous = sorted_days[0]
    for day in sorted_days[1:]:
        if day.date() == previous.date() + timedelta(days=1):
            previous = day
            continue
        ranges.append((start, previous))
        start = previous = day
    ranges.append((start, previous))
    return ranges


def _load_or_fetch_windowed_cache(
    *,
    path: Path,
    target_tz: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    fetch_window,
    label: str,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    fetched_new_data = False

    if path.exists():
        cached = _load_timestamp_csv(path, target_tz)
        if not cached.empty:
            frames.append(cached)
            cached_days = pd.DatetimeIndex(cached.index).tz_convert(target_tz).normalize()
            cached_start = cached_days.min()
            cached_end = cached_days.max()
            missing_ranges: list[tuple[pd.Timestamp, pd.Timestamp]] = []
            if start < cached_start:
                missing_ranges.append((start, cached_start - pd.Timedelta(days=1)))
            if end > cached_end:
                missing_ranges.append((cached_end + pd.Timedelta(days=1), end))
        else:
            missing_ranges = [(start, end)]
    else:
        missing_ranges = [(start, end)]

    for fetch_start, fetch_end in missing_ranges:
        if fetch_start > fetch_end:
            continue
        try:
            fetched = fetch_window(fetch_start, fetch_end)
        except Exception as exc:
            if frames:
                warnings.warn(
                    f"Failed to refresh {label} for {fetch_start.date()}..{fetch_end.date()}; "
                    f"using available cache from {path}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            raise
        if not fetched.empty:
            frames.append(fetched)
            fetched_new_data = True

    if not frames:
        raise ValueError(f"No {label} data available for {start.date()}..{end.date()}.")

    df = _combine_timestamp_frames(frames)
    if fetched_new_data or not path.exists():
        _save_timestamp_csv(df, path)
    return _restrict_timestamp_window(df, start, end, target_tz)


def _load_or_fetch_actual_load(config: LoadForecastModelConfig | EntsoeLoadForecastBenchmarkConfig) -> pd.DataFrame:
    start, end = _actual_load_fetch_window(config)
    actual = _load_or_fetch_windowed_cache(
        path=config.actual_load_file,
        target_tz=config.target_tz,
        start=start,
        end=end,
        label="actual load",
        fetch_window=lambda fetch_start, fetch_end: fetch_actual_load(
            start_day=fetch_start,
            end_day=fetch_end,
            country_code=config.country_code_entsoe,
            api_key_env=config.entsoe_api_key_env,
            target_tz=config.target_tz,
            chunk_days=config.chunk_days,
        ),
    )

    full_actual_end = end
    if isinstance(config, LoadForecastModelConfig):
        full_actual_end = min(end, _latest_operational_actual_day(config.target_tz))
    repair_days = _actual_load_days_below_count(
        actual,
        start=start,
        end=full_actual_end,
        target_tz=config.target_tz,
    )
    if repair_days:
        fetched_repairs = []
        for repair_start, repair_end in _group_consecutive_days(repair_days):
            try:
                fetched_repairs.append(
                    fetch_actual_load(
                        start_day=repair_start,
                        end_day=repair_end,
                        country_code=config.country_code_entsoe,
                        api_key_env=config.entsoe_api_key_env,
                        target_tz=config.target_tz,
                        chunk_days=config.chunk_days,
                        require_complete_days=True,
                    )
                )
            except Exception as exc:
                warnings.warn(
                    f"Failed to repair cached actual load for {repair_start.date()}..{repair_end.date()}; "
                    f"using available cache from {config.actual_load_file}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
        if fetched_repairs:
            cached = _load_timestamp_csv(config.actual_load_file, config.target_tz) if config.actual_load_file.exists() else actual
            combined = _combine_timestamp_frames([cached, *fetched_repairs])
            _save_timestamp_csv(combined, config.actual_load_file)
            actual = _restrict_timestamp_window(combined, start, end, config.target_tz)

    if isinstance(config, LoadForecastModelConfig):
        refresh_day = _partial_load_refresh_day(config, end)
        if refresh_day is not None:
            try:
                fetched = fetch_actual_load(
                    start_day=refresh_day,
                    end_day=refresh_day,
                    country_code=config.country_code_entsoe,
                    api_key_env=config.entsoe_api_key_env,
                    target_tz=config.target_tz,
                    chunk_days=config.chunk_days,
                    require_complete_days=False,
                )
            except Exception as exc:
                warnings.warn(
                    f"Failed to refresh current-day partial actual load for {refresh_day.date()}; "
                    f"using available cache from {config.actual_load_file}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                fetched = pd.DataFrame()
            if not fetched.empty:
                cached = _load_timestamp_csv(config.actual_load_file, config.target_tz) if config.actual_load_file.exists() else actual
                combined = _combine_timestamp_frames([cached, fetched])
                _save_timestamp_csv(combined, config.actual_load_file)
                actual = _restrict_timestamp_window(combined, start, end, config.target_tz)
    return actual


def _load_or_fetch_entsoe_load_forecast(config: EntsoeLoadForecastBenchmarkConfig) -> pd.DataFrame:
    return _load_or_fetch_load_forecast_to_file(config, config.forecast_file)


def _load_or_fetch_model_entsoe_load_forecast(config: LoadForecastModelConfig) -> pd.DataFrame:
    return _load_or_fetch_load_forecast_to_file(config, config.entsoe_load_forecast_file)


def _load_or_fetch_load_forecast_to_file(
    config: LoadForecastModelConfig | EntsoeLoadForecastBenchmarkConfig,
    forecast_file: Path,
) -> pd.DataFrame:
    start, end = _calendar_window(config)

    def fetch_window(fetch_start: pd.Timestamp, fetch_end: pd.Timestamp) -> pd.DataFrame:
        frames = []
        current_start = fetch_start
        while current_start <= fetch_end:
            current_end = min(current_start + pd.Timedelta(days=config.chunk_days - 1), fetch_end)
            frames.append(
                fetch_load_forecast(
                    start_day=current_start,
                    end_day=current_end,
                    country_code=config.country_code_entsoe,
                    api_key_env=config.entsoe_api_key_env,
                    target_tz=config.target_tz,
                )
            )
            current_start = current_end + pd.Timedelta(days=1)
        if not frames:
            return pd.DataFrame()
        return _combine_timestamp_frames(frames)

    return _load_or_fetch_windowed_cache(
        path=forecast_file,
        target_tz=config.target_tz,
        start=start,
        end=end,
        label="ENTSO-E load forecast",
        fetch_window=fetch_window,
    )


def _build_calendar_features(
    index: pd.DatetimeIndex,
    *,
    target_tz: str,
    include_holidays: bool,
    include_bridge_days: bool,
    calendar_harmonics: int,
) -> pd.DataFrame:
    local_index = index.tz_convert(target_tz) if index.tz is not None else index.tz_localize(target_tz)
    minute_of_day = local_index.hour * 60 + local_index.minute
    mtu = local_index.hour * 4 + local_index.minute // 15
    day_of_week = local_index.dayofweek
    month = local_index.month
    day_of_year = local_index.dayofyear

    features = pd.DataFrame(index=index)
    features["mtu"] = mtu.astype(float)
    features["hour"] = local_index.hour.astype(float)
    features["dayofweek"] = day_of_week.astype(float)
    features["month"] = month.astype(float)
    features["is_weekend"] = (day_of_week >= 5).astype(float)
    features["mtu_sin"] = np.sin(2 * np.pi * mtu / 96)
    features["mtu_cos"] = np.cos(2 * np.pi * mtu / 96)
    features["minute_of_day_sin"] = np.sin(2 * np.pi * minute_of_day / 1440)
    features["minute_of_day_cos"] = np.cos(2 * np.pi * minute_of_day / 1440)
    features["dow_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    features["dow_cos"] = np.cos(2 * np.pi * day_of_week / 7)
    features["month_sin"] = np.sin(2 * np.pi * month / 12)
    features["month_cos"] = np.cos(2 * np.pi * month / 12)
    iso_week = local_index.isocalendar().week.astype(float).to_numpy()
    for harmonic in range(1, calendar_harmonics + 1):
        suffix = "" if harmonic == 1 else f"_{harmonic}"
        features[f"doy_sin{suffix}"] = np.sin(2 * np.pi * harmonic * day_of_year / 366)
        features[f"doy_cos{suffix}"] = np.cos(2 * np.pi * harmonic * day_of_year / 366)
        features[f"week_sin{suffix}"] = np.sin(2 * np.pi * harmonic * iso_week / 53)
        features[f"week_cos{suffix}"] = np.cos(2 * np.pi * harmonic * iso_week / 53)

    if include_holidays:
        try:
            import holidays
        except ImportError as exc:
            raise ImportError("Holiday load features require the optional `holidays` package.") from exc

        years = sorted(set(local_index.year))
        de_holidays = holidays.country_holidays("DE", years=years)
        lu_holidays = holidays.country_holidays("LU", years=years)
        local_dates = pd.Series(local_index.date, index=index)
        features["is_holiday_de"] = local_dates.isin(set(de_holidays.keys())).astype(float)
        features["is_holiday_lu"] = local_dates.isin(set(lu_holidays.keys())).astype(float)
        features["is_holiday"] = features[["is_holiday_de", "is_holiday_lu"]].max(axis=1)
        features["is_nonworkday"] = features[["is_weekend", "is_holiday"]].max(axis=1)
        if include_bridge_days:
            holiday_dates = set(de_holidays.keys()) | set(lu_holidays.keys())
            current_dates = pd.Series(local_index.date, index=index)
            previous_dates = current_dates - pd.Timedelta(days=1)
            next_dates = current_dates + pd.Timedelta(days=1)
            is_workday = (day_of_week < 5) & ~current_dates.isin(holiday_dates).to_numpy()
            previous_holiday = previous_dates.isin(holiday_dates).to_numpy()
            next_holiday = next_dates.isin(holiday_dates).to_numpy()
            features["is_pre_holiday_workday"] = (is_workday & next_holiday).astype(float)
            features["is_post_holiday_workday"] = (is_workday & previous_holiday).astype(float)
            features["is_bridge_day"] = (
                is_workday
                & (
                    ((day_of_week == 0) & next_holiday)
                    | ((day_of_week == 4) & previous_holiday)
                )
            ).astype(float)

    features.index.name = "timestamp"
    return features


_DE_NUTS1_TO_HOLIDAYS_SUBDIV = {
    "DE1": "BW",
    "DE2": "BY",
    "DE3": "BE",
    "DE4": "BB",
    "DE5": "HB",
    "DE6": "HH",
    "DE7": "HE",
    "DE8": "MV",
    "DE9": "NI",
    "DEA": "NW",
    "DEB": "RP",
    "DEC": "SL",
    "DED": "SN",
    "DEE": "ST",
    "DEF": "SH",
    "DEG": "TH",
}


def _load_region_weights(
    path: Path,
    *,
    region_column: str,
    weight_column: str,
) -> pd.Series:
    if not path.exists():
        raise FileNotFoundError(f"Regional holiday weight file does not exist: {path}")

    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    elif suffix in {".csv", ".txt"}:
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported regional holiday weight file format: {path.suffix}")

    missing_columns = [column for column in [region_column, weight_column] if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Regional holiday weight file is missing columns: {missing_columns}")

    weights = df[[region_column, weight_column]].dropna().copy()
    weights[region_column] = weights[region_column].astype(str).str.strip().str.upper()
    weights[weight_column] = pd.to_numeric(weights[weight_column], errors="coerce")
    weights = weights.dropna(subset=[region_column, weight_column])
    weights = weights.groupby(region_column, sort=True)[weight_column].sum()
    weights = weights[np.isfinite(weights) & (weights > 0.0)]
    if weights.empty:
        raise ValueError("Regional holiday weight file contains no positive weights.")
    return weights.sort_index()


def _region_holiday_dates(region: str, years: list[int]) -> set[Any]:
    try:
        import holidays
    except ImportError as exc:
        raise ImportError("Regional holiday load features require the optional `holidays` package.") from exc

    if region.startswith("LU"):
        return set(holidays.country_holidays("LU", years=years).keys())
    if region.startswith("DE"):
        subdiv = _DE_NUTS1_TO_HOLIDAYS_SUBDIV.get(region[:3])
        if subdiv is None:
            return set(holidays.country_holidays("DE", years=years).keys())
        return set(holidays.country_holidays("DE", subdiv=subdiv, years=years).keys())
    return set()


def _daily_region_holiday_flags(dates: pd.Series, holiday_dates: set[Any]) -> pd.DataFrame:
    date_index = pd.Index(dates)
    date_timestamps = pd.to_datetime(date_index)
    day_of_week = date_timestamps.dayofweek.to_numpy()
    current_dates = pd.Series(date_timestamps.date, index=date_index)
    previous_dates = pd.Series((date_timestamps - pd.Timedelta(days=1)).date, index=date_index)
    next_dates = pd.Series((date_timestamps + pd.Timedelta(days=1)).date, index=date_index)

    is_holiday = current_dates.isin(holiday_dates).to_numpy()
    is_weekend = day_of_week >= 5
    is_workday = (day_of_week < 5) & ~is_holiday
    previous_holiday = previous_dates.isin(holiday_dates).to_numpy()
    next_holiday = next_dates.isin(holiday_dates).to_numpy()
    is_bridge_day = (
        is_workday
        & (
            ((day_of_week == 0) & next_holiday)
            | ((day_of_week == 4) & previous_holiday)
        )
    )
    return pd.DataFrame(
        {
            "public_holiday_share": is_holiday.astype(float),
            "nonworkday_share": (is_weekend | is_holiday).astype(float),
            "pre_holiday_workday_share": (is_workday & next_holiday).astype(float),
            "post_holiday_workday_share": (is_workday & previous_holiday).astype(float),
            "bridge_day_share": is_bridge_day.astype(float),
        },
        index=date_index,
    )


def _build_regional_holiday_features(
    index: pd.DatetimeIndex,
    *,
    target_tz: str,
    weight_file: Path,
    region_column: str,
    weight_column: str,
    feature_prefix: str,
) -> pd.DataFrame:
    local_index = index.tz_convert(target_tz) if index.tz is not None else index.tz_localize(target_tz)
    local_dates = pd.Series(local_index.date, index=index)
    unique_dates = pd.Index(sorted(local_dates.unique()), name="date")
    years = sorted({date.year for date in unique_dates})
    region_weights = _load_region_weights(weight_file, region_column=region_column, weight_column=weight_column)
    total_weight = float(region_weights.sum())

    weighted_daily = pd.DataFrame(
        0.0,
        index=unique_dates,
        columns=[
            "public_holiday_share",
            "nonworkday_share",
            "pre_holiday_workday_share",
            "post_holiday_workday_share",
            "bridge_day_share",
        ],
    )
    for region, weight in region_weights.items():
        holiday_dates = _region_holiday_dates(str(region), years)
        if not holiday_dates:
            continue
        region_flags = _daily_region_holiday_flags(unique_dates, holiday_dates)
        weighted_daily = weighted_daily.add(region_flags * float(weight), fill_value=0.0)

    weighted_daily = weighted_daily / total_weight
    weighted_daily = weighted_daily.rename(columns={column: f"{feature_prefix}_{column}" for column in weighted_daily.columns})
    features = weighted_daily.reindex(local_dates.to_numpy()).set_index(index)
    features.index.name = "timestamp"
    return features.astype(float)


def _add_weather_time_interactions(
    features: pd.DataFrame,
    *,
    weighted_weather_prefix: str = "weather_weighted",
) -> pd.DataFrame:
    if features.empty or "mtu_sin" not in features.columns or "mtu_cos" not in features.columns:
        return features

    def is_weather_interaction_column(column: str) -> bool:
        if "_daily_" in column:
            return False
        if column.startswith("weather_t2m_C_cluster_"):
            return True
        if column.startswith("weather_hdd") and "_cluster_" in column:
            return True
        if column.startswith("weather_cdd") and "_cluster_" in column:
            return True
        if column.startswith(f"{weighted_weather_prefix}_t2m_C"):
            return True
        if column.startswith(f"{weighted_weather_prefix}_hdd"):
            return True
        if column.startswith(f"{weighted_weather_prefix}_cdd"):
            return True
        return False

    weather_columns = [
        column
        for column in features.columns
        if is_weather_interaction_column(column)
    ]
    if not weather_columns:
        return features

    interaction_data: dict[str, pd.Series] = {}
    for column in weather_columns:
        interaction_data[f"{column}_x_mtu_sin"] = features[column] * features["mtu_sin"]
        interaction_data[f"{column}_x_mtu_cos"] = features[column] * features["mtu_cos"]
        if "is_weekend" in features.columns:
            interaction_data[f"{column}_x_weekend"] = features[column] * features["is_weekend"]
        if "is_nonworkday" in features.columns:
            interaction_data[f"{column}_x_nonworkday"] = features[column] * features["is_nonworkday"]

    interactions = pd.DataFrame(interaction_data, index=features.index)
    return pd.concat([features, interactions], axis=1)


def _build_actual_load_lag_features(
    actual_load: pd.DataFrame,
    index: pd.DatetimeIndex,
    lag_days: list[int],
) -> pd.DataFrame:
    if "load_actual" not in actual_load.columns:
        raise ValueError("Actual load data must contain a 'load_actual' column.")

    base = actual_load["load_actual"].astype(float).sort_index()
    features = pd.DataFrame(index=index)
    for lag_day in sorted({int(day) for day in lag_days if int(day) > 0}):
        lagged = base.copy()
        lagged.index = lagged.index + pd.Timedelta(days=lag_day)
        features[f"Load_Actual_MW_lag_d{lag_day}"] = lagged.reindex(index).astype(float)

    features.index.name = "timestamp"
    return features


def _window_label(start_hour: int, end_hour_inclusive: int, end_minute_inclusive: int = 45) -> str:
    end_label = (
        f"{end_hour_inclusive:02d}"
        if end_minute_inclusive == 45
        else f"{end_hour_inclusive:02d}{end_minute_inclusive:02d}"
    )
    return f"{start_hour:02d}_{end_label}"


def _actual_load_window_stats(
    actual_load: pd.DataFrame,
    *,
    target_tz: str,
    start_hour: int,
    end_hour_inclusive: int,
    end_minute_inclusive: int = 45,
) -> pd.DataFrame:
    if "load_actual" not in actual_load.columns:
        raise ValueError("Actual load data must contain a 'load_actual' column.")

    base = actual_load[["load_actual"]].astype(float).sort_index().copy()
    local_index = base.index.tz_convert(target_tz) if base.index.tz is not None else base.index.tz_localize(target_tz)
    minute_of_day = local_index.hour * 60 + local_index.minute
    window_start = start_hour * 60
    window_end = end_hour_inclusive * 60 + end_minute_inclusive
    window_mask = (minute_of_day >= window_start) & (minute_of_day <= window_end)
    window = base.loc[window_mask].copy()
    window["_date"] = pd.Index(local_index[window_mask].date)
    grouped = window.groupby("_date")["load_actual"]
    stats = grouped.agg(["mean", "min", "max", "first", "last"]).sort_index()
    stats["range"] = stats["max"] - stats["min"]
    stats["ramp"] = stats["last"] - stats["first"]
    return stats


def _time_label(value: str) -> str:
    hour, minute = (int(part) for part in value.split(":", maxsplit=1))
    return f"{hour:02d}{minute:02d}"


def _actual_load_point_values(
    actual_load: pd.DataFrame,
    *,
    target_tz: str,
    point_times: list[str],
) -> pd.DataFrame:
    if not point_times:
        return pd.DataFrame()
    if "load_actual" not in actual_load.columns:
        raise ValueError("Actual load data must contain a 'load_actual' column.")

    base = actual_load[["load_actual"]].astype(float).sort_index().copy()
    local_index = base.index.tz_convert(target_tz) if base.index.tz is not None else base.index.tz_localize(target_tz)
    minute_of_day = local_index.hour * 60 + local_index.minute
    local_dates = pd.Index(local_index.date)
    requested_minutes = {
        _time_label(point_time): int(point_time.split(":", maxsplit=1)[0]) * 60 + int(point_time.split(":", maxsplit=1)[1])
        for point_time in point_times
    }

    columns: dict[str, pd.Series] = {}
    for label, minute in requested_minutes.items():
        mask = minute_of_day == minute
        if not mask.any():
            continue
        values = pd.Series(base.loc[mask, "load_actual"].to_numpy(dtype=float), index=local_dates[mask])
        columns[label] = values.groupby(level=0).last()

    if not columns:
        return pd.DataFrame()
    return pd.DataFrame(columns).sort_index()


def _build_partial_load_features(
    actual_load: pd.DataFrame,
    index: pd.DatetimeIndex,
    *,
    target_tz: str,
    reference_day: int,
    comparison_lag_days: int,
    morning_end_hour: int,
    morning_end_minute: int = 45,
    include_shape_features: bool = False,
    point_times: list[str] | None = None,
) -> pd.DataFrame:
    local_index = index.tz_convert(target_tz) if index.tz is not None else index.tz_localize(target_tz)
    local_dates = pd.Series(local_index.date, index=index)
    target_dates = pd.Index(sorted(local_dates.unique()), name="date")
    source_dates = pd.Index((pd.to_datetime(target_dates) - pd.Timedelta(days=reference_day)).date)
    comparison_dates = pd.Index((pd.to_datetime(source_dates) - pd.Timedelta(days=comparison_lag_days)).date)

    windows = [
        (0, morning_end_hour),
        (6, morning_end_hour),
        (9, morning_end_hour),
    ]
    daily_features = pd.DataFrame(index=target_dates)
    for start_hour, end_hour in windows:
        if start_hour * 60 > end_hour * 60 + morning_end_minute:
            continue
        label = _window_label(start_hour, end_hour, morning_end_minute)
        stats = _actual_load_window_stats(
            actual_load,
            target_tz=target_tz,
            start_hour=start_hour,
            end_hour_inclusive=end_hour,
            end_minute_inclusive=morning_end_minute,
        )
        source = stats.reindex(source_dates)
        source.index = target_dates
        for stat in ["mean", "min", "max", "last"]:
            daily_features[f"partial_load_d{reference_day}_{label}_{stat}"] = source[stat].to_numpy(dtype=float)
        if include_shape_features:
            for stat in ["first", "range", "ramp"]:
                daily_features[f"partial_load_d{reference_day}_{label}_{stat}"] = source[stat].to_numpy(dtype=float)

        comparison = stats.reindex(comparison_dates)
        comparison.index = target_dates
        daily_features[f"partial_load_d{reference_day}_{label}_mean_diff_d{comparison_lag_days}"] = (
            source["mean"] - comparison["mean"]
        ).to_numpy(dtype=float)
        daily_features[f"partial_load_d{reference_day}_{label}_last_diff_d{comparison_lag_days}"] = (
            source["last"] - comparison["last"]
        ).to_numpy(dtype=float)
        if include_shape_features:
            for stat in ["range", "ramp"]:
                daily_features[f"partial_load_d{reference_day}_{label}_{stat}_diff_d{comparison_lag_days}"] = (
                    source[stat] - comparison[stat]
                ).to_numpy(dtype=float)

    point_values = _actual_load_point_values(
        actual_load,
        target_tz=target_tz,
        point_times=point_times or [],
    )
    if not point_values.empty:
        source_points = point_values.reindex(source_dates)
        source_points.index = target_dates
        comparison_points = point_values.reindex(comparison_dates)
        comparison_points.index = target_dates
        for label in point_values.columns:
            daily_features[f"partial_load_d{reference_day}_point_{label}"] = source_points[label].to_numpy(dtype=float)
            daily_features[f"partial_load_d{reference_day}_point_{label}_diff_d{comparison_lag_days}"] = (
                source_points[label] - comparison_points[label]
            ).to_numpy(dtype=float)

    features = daily_features.reindex(local_dates.to_numpy()).set_index(index)
    features.index.name = "timestamp"
    return features.astype(float)


def _build_entsoe_forecast_features(
    config: LoadForecastModelConfig,
    index: pd.DatetimeIndex,
) -> pd.DataFrame:
    forecast = _load_or_fetch_model_entsoe_load_forecast(config)
    if "load_fc" not in forecast.columns:
        raise ValueError("ENTSO-E load forecast data must contain a 'load_fc' column.")
    features = forecast[["load_fc"]].rename(columns={"load_fc": "Load_Benchmark_MW"})
    features = features.reindex(index).astype(float)
    features.index.name = "timestamp"
    return features


def _safe_feature_name(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in str(value)).strip("_")


def _regional_load_feature_config(
    config: LoadForecastModelConfig,
    *,
    name: str,
    country_code: str,
) -> LoadForecastModelConfig:
    safe_name = _safe_feature_name(name) or "region"
    data = config.model_dump()
    data.update(
        {
            "repo_root": config.repo_root,
            "country_code_entsoe": country_code,
            "actual_load_file": config.regional_load_cache_dir / f"{safe_name}_actual_load.csv",
            "entsoe_load_forecast_file": config.regional_load_cache_dir / f"{safe_name}_entsoe_load_forecast.csv",
            "export_dir": config.export_dir / "_regional_feature_cache_unused" / safe_name,
            "include_regional_load_features": False,
            "regional_load_components": {},
        }
    )
    return LoadForecastModelConfig.model_validate(data)


def _build_regional_load_features(
    config: LoadForecastModelConfig,
    index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Build regional ENTSO-E load forecast and actual-load history features."""
    feature_data: dict[str, pd.Series] = {}
    partial_blocks: list[pd.DataFrame] = []
    forecast_columns: list[str] = []
    lag_columns_by_day: dict[int, list[str]] = {}

    for raw_name, country_code in config.regional_load_components.items():
        name = _safe_feature_name(raw_name)
        component_config = _regional_load_feature_config(config, name=name, country_code=country_code)

        if config.include_regional_load_forecast_features or config.include_regional_load_summary_features:
            forecast = _load_or_fetch_model_entsoe_load_forecast(component_config)
            if "load_fc" not in forecast.columns:
                raise ValueError(f"Regional load forecast for {raw_name!r} must contain a 'load_fc' column.")
            forecast_values = forecast["load_fc"].reindex(index).astype(float)
            if config.include_regional_load_forecast_features:
                column = f"regional_load_fc_{name}"
                feature_data[column] = forecast_values
                feature_data[f"{column}_ramp_1h"] = forecast_values.diff(4)
                feature_data[f"{column}_delta_d1"] = forecast_values - forecast_values.shift(96)
                forecast_columns.append(column)
            else:
                summary_column = f"regional_load_fc_{name}"
                feature_data[summary_column] = forecast_values
                forecast_columns.append(summary_column)

        actual: pd.DataFrame | None = None
        if config.include_regional_actual_load_lag_features or config.include_regional_partial_load_features:
            actual = _load_or_fetch_actual_load(component_config)

        if config.include_regional_actual_load_lag_features and actual is not None:
            base = actual["load_actual"].astype(float).sort_index()
            for lag_day in sorted({int(day) for day in config.actual_load_lag_days if int(day) > 0}):
                lagged = base.copy()
                lagged.index = lagged.index + pd.Timedelta(days=lag_day)
                column = f"regional_load_actual_{name}_lag_d{lag_day}"
                feature_data[column] = lagged.reindex(index).astype(float)
                lag_columns_by_day.setdefault(lag_day, []).append(column)

        if config.include_regional_partial_load_features and actual is not None:
            partial = _build_partial_load_features(
                actual_load=actual,
                index=index,
                target_tz=config.target_tz,
                reference_day=config.partial_load_reference_day,
                comparison_lag_days=config.partial_load_comparison_lag_days,
                morning_end_hour=config.partial_load_morning_end_hour,
                morning_end_minute=config.partial_load_morning_end_minute,
                include_shape_features=config.include_partial_load_shape_features,
                point_times=config.partial_load_point_times,
            )
            partial = partial.add_prefix(f"regional_{name}_")
            partial_blocks.append(partial)

    features = pd.concat(
        [pd.DataFrame(feature_data, index=index), *partial_blocks],
        axis=1,
    )
    summary_data: dict[str, pd.Series] = {}

    if config.include_regional_load_summary_features:
        if forecast_columns:
            forecast_frame = features[forecast_columns]
            regional_load_fc_sum = forecast_frame.sum(axis=1)
            summary_data["regional_load_fc_sum"] = regional_load_fc_sum
            summary_data["regional_load_fc_mean"] = forecast_frame.mean(axis=1)
            summary_data["regional_load_fc_std"] = forecast_frame.std(axis=1)
            summary_data["regional_load_fc_range"] = forecast_frame.max(axis=1) - forecast_frame.min(axis=1)
            regional_sum = regional_load_fc_sum.replace(0.0, np.nan)
            summary_data["regional_load_fc_max_share"] = forecast_frame.max(axis=1) / regional_sum
            national_forecast = _load_or_fetch_model_entsoe_load_forecast(config)["load_fc"].reindex(index).astype(float)
            summary_data["regional_load_fc_missing_to_national"] = national_forecast - regional_load_fc_sum

        for lag_day, lag_columns in sorted(lag_columns_by_day.items()):
            lag_frame = features[lag_columns]
            summary_data[f"regional_load_actual_lag_d{lag_day}_sum"] = lag_frame.sum(axis=1)
            summary_data[f"regional_load_actual_lag_d{lag_day}_std"] = lag_frame.std(axis=1)
            summary_data[f"regional_load_actual_lag_d{lag_day}_range"] = lag_frame.max(axis=1) - lag_frame.min(axis=1)

    if summary_data:
        features = pd.concat([features, pd.DataFrame(summary_data, index=index)], axis=1)

    features = features.loc[:, ~features.columns.duplicated()]
    features.index.name = "timestamp"
    return features.astype(float)


def _entsoe_error_series(actual_load: pd.DataFrame, forecast: pd.DataFrame) -> pd.Series:
    if "load_actual" not in actual_load.columns:
        raise ValueError("Actual load data must contain a 'load_actual' column.")
    if "load_fc" not in forecast.columns:
        raise ValueError("ENTSO-E load forecast data must contain a 'load_fc' column.")

    frame = actual_load[["load_actual"]].astype(float).join(forecast[["load_fc"]].astype(float), how="inner")
    error = frame["load_actual"] - frame["load_fc"]
    error.name = "entsoe_load_error"
    return error.sort_index()


def _time_group_values(index: pd.DatetimeIndex, group: str) -> pd.Series:
    if group == "global":
        values = np.zeros(len(index), dtype=int)
    elif group == "hour":
        values = index.hour
    elif group == "mtu":
        values = index.hour * 4 + index.minute // 15
    else:
        raise ValueError(f"Unsupported time group: {group!r}")
    return pd.Series(values, index=index)


def _build_entsoe_error_features(
    config: LoadForecastModelConfig,
    actual_load: pd.DataFrame,
    index: pd.DatetimeIndex,
) -> pd.DataFrame:
    forecast = _load_or_fetch_model_entsoe_load_forecast(config)
    error = _entsoe_error_series(actual_load, forecast)
    features = pd.DataFrame(index=index)

    if config.include_entsoe_error_lag_features:
        for lag_day in sorted({int(day) for day in config.entsoe_error_lag_days if int(day) > 0}):
            lagged = error.copy()
            lagged.index = lagged.index + pd.Timedelta(days=lag_day)
            features[f"Entsoe_Load_Error_MW_lag_d{lag_day}"] = lagged.reindex(index).astype(float)

    if config.include_entsoe_error_rolling_features:
        local_index = index.tz_convert(config.target_tz) if index.tz is not None else index.tz_localize(config.target_tz)
        local_days = pd.Series(local_index.normalize(), index=index)
        unique_days = pd.DatetimeIndex(local_days.unique()).sort_values()
        error_frame = error.to_frame("error")
        error_frame = error_frame.loc[~error_frame.index.duplicated(keep="last")].sort_index()

        for group in config.entsoe_error_rolling_groups:
            if group == "global":
                history_group_values = pd.Series(0, index=error_frame.index)
                target_group_values = pd.Series(0, index=index)
            else:
                history_group_values = _time_group_values(error_frame.index, group)
                target_group_values = _time_group_values(index, group)

            for window_day in sorted({int(day) for day in config.entsoe_error_rolling_windows_days if int(day) > 0}):
                column = f"Entsoe_Load_Error_MW_{group}_mean_{window_day}d"
                values = pd.Series(np.nan, index=index, dtype=float)
                window = pd.Timedelta(days=window_day)
                for day in unique_days:
                    cutoff = _target_availability_cutoff(day, config)
                    window_start = cutoff - window + pd.Timedelta(minutes=15)
                    history_mask = (
                        (error_frame.index >= window_start)
                        & (error_frame.index <= cutoff)
                        & error_frame["error"].notna()
                    )
                    day_mask = local_days == day
                    if not history_mask.any() or not day_mask.any():
                        continue

                    history = error_frame.loc[history_mask, "error"]
                    if group == "global":
                        if len(history) >= config.entsoe_error_rolling_min_observations:
                            values.loc[day_mask] = float(history.mean())
                        continue

                    grouped = history.groupby(history_group_values.loc[history_mask]).agg(["mean", "count"])
                    eligible = grouped.loc[grouped["count"] >= config.entsoe_error_rolling_min_observations, "mean"]
                    if eligible.empty:
                        continue
                    values.loc[day_mask] = target_group_values.loc[day_mask].map(eligible).to_numpy(dtype=float)

                features[column] = values

    features.index.name = "timestamp"
    return features


def _cluster_id(column: str) -> str:
    return column.rsplit("_", 1)[-1]


def _split_weather_cluster_column(column: str) -> tuple[str, int] | None:
    marker = "_cluster_"
    if marker not in column:
        return None
    base_name, cluster_id = column.rsplit(marker, 1)
    if not base_name.startswith("weather_") or not cluster_id.isdigit():
        return None
    return base_name, int(cluster_id)


def _load_cluster_weights(
    path: Path,
    *,
    cluster_id_column: str,
    weight_column: str,
) -> pd.Series:
    if not path.exists():
        raise FileNotFoundError(f"Weather cluster weight file does not exist: {path}")

    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    elif suffix in {".csv", ".txt"}:
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported weather cluster weight file format: {path.suffix}")

    missing_columns = [column for column in [cluster_id_column, weight_column] if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Weather cluster weight file is missing columns: {missing_columns}")

    weights = df[[cluster_id_column, weight_column]].dropna().copy()
    weights[cluster_id_column] = weights[cluster_id_column].astype(int)
    weights[weight_column] = weights[weight_column].astype(float)
    weights = weights.groupby(cluster_id_column, sort=True)[weight_column].sum()
    weights = weights[np.isfinite(weights) & (weights > 0.0)]
    if weights.empty:
        raise ValueError("Weather cluster weight file contains no positive weights.")

    weights.index = weights.index.astype(int)
    return weights.sort_index()


def _weighted_weather_column_name(base_name: str, prefix: str) -> str:
    if base_name.startswith("weather_"):
        return f"{prefix}_{base_name.removeprefix('weather_')}"
    return f"{prefix}_{base_name}"


def _normalise_weather_base_names(base_names: list[str] | None) -> set[str] | None:
    if not base_names:
        return None
    normalised = set()
    for base_name in base_names:
        name = str(base_name).strip()
        if not name:
            continue
        normalised.add(name if name.startswith("weather_") else f"weather_{name}")
    return normalised or None


def _add_weighted_weather_features(
    features: pd.DataFrame,
    cluster_weights: pd.Series,
    *,
    prefix: str = "weather_weighted",
    base_names: list[str] | None = None,
) -> pd.DataFrame:
    if features.empty or cluster_weights.empty:
        return features

    allowed_base_names = _normalise_weather_base_names(base_names)
    grouped_columns: dict[str, list[tuple[int, str]]] = {}
    for column in features.columns:
        parsed = _split_weather_cluster_column(column)
        if parsed is None:
            continue
        base_name, cluster_id = parsed
        if allowed_base_names is not None and base_name not in allowed_base_names:
            continue
        grouped_columns.setdefault(base_name, []).append((cluster_id, column))

    weighted_data: dict[str, pd.Series] = {}
    for base_name, cluster_columns in grouped_columns.items():
        available = [
            (cluster_id, column)
            for cluster_id, column in sorted(cluster_columns)
            if cluster_id in cluster_weights.index
        ]
        if not available:
            continue

        columns = [column for _, column in available]
        weights = pd.Series(
            {column: float(cluster_weights.loc[cluster_id]) for cluster_id, column in available},
            dtype=float,
        )
        values = features[columns].astype(float)
        denominator = values.notna().mul(weights, axis=1).sum(axis=1)
        numerator = values.mul(weights, axis=1).sum(axis=1, min_count=1)
        weighted_data[_weighted_weather_column_name(base_name, prefix)] = numerator / denominator.replace(0.0, np.nan)

    if not weighted_data:
        return features

    weighted_features = pd.DataFrame(weighted_data, index=features.index)
    result = pd.concat([features, weighted_features], axis=1)
    return result.loc[:, ~result.columns.duplicated()]


def _drop_weather_cluster_features(features: pd.DataFrame) -> pd.DataFrame:
    keep_columns = [
        column
        for column in features.columns
        if _split_weather_cluster_column(column) is None
    ]
    return features[keep_columns]


def _add_weighted_weather_daily_features(
    features: pd.DataFrame,
    *,
    prefix: str,
    base_names: list[str] | None,
    stats: list[str],
    lag_days: list[int],
) -> pd.DataFrame:
    if features.empty:
        return features

    bases = base_names or ["t2m_C", "hdd18", "cdd22"]
    columns = [
        f"{prefix}_{base_name.removeprefix('weather_').removeprefix(prefix + '_')}"
        for base_name in bases
    ]
    columns = [column for column in columns if column in features.columns]
    if not columns:
        return features

    local_days = pd.Series(features.index.normalize(), index=features.index)
    unique_days = pd.DatetimeIndex(local_days.unique()).sort_values()
    daily_features = pd.DataFrame(index=unique_days)
    for column in columns:
        grouped = features[column].groupby(local_days)
        daily_stat_frame = pd.DataFrame(index=unique_days)
        if "mean" in stats:
            daily_stat_frame["mean"] = grouped.mean().reindex(unique_days)
        if "min" in stats:
            daily_stat_frame["min"] = grouped.min().reindex(unique_days)
        if "max" in stats:
            daily_stat_frame["max"] = grouped.max().reindex(unique_days)
        if "range" in stats:
            daily_stat_frame["range"] = grouped.max().reindex(unique_days) - grouped.min().reindex(unique_days)

        for stat in daily_stat_frame.columns:
            daily_features[f"{column}_daily_{stat}"] = daily_stat_frame[stat]
            for lag_day in sorted({int(day) for day in lag_days if int(day) > 0}):
                lagged = daily_stat_frame[stat].copy()
                lagged.index = lagged.index + pd.Timedelta(days=lag_day)
                daily_features[f"{column}_daily_{stat}_diff_d{lag_day}"] = daily_stat_frame[stat] - lagged.reindex(unique_days)

    expanded = daily_features.reindex(local_days.to_numpy()).set_index(features.index)
    expanded.index.name = features.index.name
    result = pd.concat([features, expanded], axis=1)
    return result.loc[:, ~result.columns.duplicated()]


def _add_weather_cluster_spread_features(
    features: pd.DataFrame,
    *,
    prefix: str = "weather_spread",
    base_names: list[str] | None = None,
    stats: list[str] | None = None,
) -> pd.DataFrame:
    if features.empty:
        return features

    requested_stats = stats or ["min", "max", "range", "std"]
    allowed_base_names = _normalise_weather_base_names(base_names)
    grouped_columns: dict[str, list[str]] = {}
    for column in features.columns:
        parsed = _split_weather_cluster_column(column)
        if parsed is None:
            continue
        base_name, _cluster_id = parsed
        if allowed_base_names is not None and base_name not in allowed_base_names:
            continue
        grouped_columns.setdefault(base_name, []).append(column)

    spread_data: dict[str, pd.Series] = {}
    for base_name, columns in grouped_columns.items():
        values = features[sorted(columns)].astype(float)
        output_base = base_name.removeprefix("weather_")
        if "mean" in requested_stats:
            spread_data[f"{prefix}_{output_base}_mean"] = values.mean(axis=1)
        if "min" in requested_stats:
            spread_data[f"{prefix}_{output_base}_min"] = values.min(axis=1)
        if "max" in requested_stats:
            spread_data[f"{prefix}_{output_base}_max"] = values.max(axis=1)
        if "range" in requested_stats:
            spread_data[f"{prefix}_{output_base}_range"] = values.max(axis=1) - values.min(axis=1)
        if "std" in requested_stats:
            spread_data[f"{prefix}_{output_base}_std"] = values.std(axis=1)
        if "p10" in requested_stats:
            spread_data[f"{prefix}_{output_base}_p10"] = values.quantile(0.10, axis=1)
        if "p90" in requested_stats:
            spread_data[f"{prefix}_{output_base}_p90"] = values.quantile(0.90, axis=1)

    if not spread_data:
        return features
    spread_features = pd.DataFrame(spread_data, index=features.index)
    result = pd.concat([features, spread_features], axis=1)
    return result.loc[:, ~result.columns.duplicated()]


def _weighted_quantile_row(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not mask.any():
        return np.nan
    valid_values = values[mask]
    valid_weights = weights[mask]
    order = np.argsort(valid_values)
    sorted_values = valid_values[order]
    sorted_weights = valid_weights[order]
    cutoff = float(quantile) * sorted_weights.sum()
    index = int(np.searchsorted(np.cumsum(sorted_weights), cutoff, side="left"))
    return float(sorted_values[min(index, len(sorted_values) - 1)])


def _quantile_label(quantile: float) -> str:
    return f"q{int(round(float(quantile) * 100)):02d}"


def _weather_cluster_columns(
    features: pd.DataFrame,
    *,
    base_names: list[str] | None = None,
) -> dict[str, list[tuple[int, str]]]:
    allowed_base_names = _normalise_weather_base_names(base_names)
    grouped_columns: dict[str, list[tuple[int, str]]] = {}
    for column in features.columns:
        parsed = _split_weather_cluster_column(column)
        if parsed is None:
            continue
        base_name, cluster_id = parsed
        if allowed_base_names is not None and base_name not in allowed_base_names:
            continue
        grouped_columns.setdefault(base_name, []).append((cluster_id, column))
    return {
        base_name: sorted(cluster_columns)
        for base_name, cluster_columns in grouped_columns.items()
        if cluster_columns
    }


def _add_weighted_weather_quantile_features(
    features: pd.DataFrame,
    cluster_weights: pd.Series,
    *,
    base_names: list[str] | None = None,
    quantiles: list[float] | None = None,
    prefix: str = "weather_weighted_q",
) -> pd.DataFrame:
    if features.empty or cluster_weights.empty:
        return features

    requested_quantiles = sorted({float(quantile) for quantile in (quantiles or [0.1, 0.5, 0.9])})
    quantile_data: dict[str, list[float]] = {}
    for base_name, cluster_columns in _weather_cluster_columns(features, base_names=base_names).items():
        available = [
            (cluster_id, column)
            for cluster_id, column in cluster_columns
            if cluster_id in cluster_weights.index
        ]
        if not available:
            continue
        columns = [column for _, column in available]
        weights = np.array([float(cluster_weights.loc[cluster_id]) for cluster_id, _ in available], dtype=float)
        values = features[columns].to_numpy(dtype=float)
        output_base = base_name.removeprefix("weather_")
        for quantile in requested_quantiles:
            quantile_data[f"{prefix}_{output_base}_{_quantile_label(quantile)}"] = [
                _weighted_quantile_row(row, weights, quantile)
                for row in values
            ]

    if not quantile_data:
        return features
    quantile_features = pd.DataFrame(quantile_data, index=features.index)
    result = pd.concat([features, quantile_features], axis=1)
    return result.loc[:, ~result.columns.duplicated()]


def _add_weighted_weather_inertia_features(
    features: pd.DataFrame,
    *,
    prefix: str,
    base_names: list[str] | None,
    windows_hours: list[int],
    stats: list[str],
) -> pd.DataFrame:
    if features.empty:
        return features

    bases = base_names or ["t2m_C", "hdd18", "cdd22"]
    columns = [
        f"{prefix}_{base_name.removeprefix('weather_').removeprefix(prefix + '_')}"
        for base_name in bases
    ]
    columns = [column for column in columns if column in features.columns]
    if not columns:
        return features

    inertia_data: dict[str, pd.Series] = {}
    sorted_features = features.sort_index()
    for column in columns:
        values = sorted_features[column].astype(float)
        for window_hours in sorted({int(window) for window in windows_hours if int(window) > 0}):
            rolling = values.rolling(
                window=pd.Timedelta(hours=window_hours),
                min_periods=max(1, int(window_hours * 2)),
            )
            if "mean" in stats or "delta_mean" in stats:
                mean = rolling.mean()
                if "mean" in stats:
                    inertia_data[f"{column}_roll{window_hours}h_mean"] = mean
                if "delta_mean" in stats:
                    inertia_data[f"{column}_minus_roll{window_hours}h_mean"] = values - mean
            if "min" in stats:
                inertia_data[f"{column}_roll{window_hours}h_min"] = rolling.min()
            if "max" in stats:
                inertia_data[f"{column}_roll{window_hours}h_max"] = rolling.max()
            if "range" in stats:
                inertia_data[f"{column}_roll{window_hours}h_range"] = rolling.max() - rolling.min()

    if not inertia_data:
        return features
    inertia_features = pd.DataFrame(inertia_data, index=sorted_features.index).reindex(features.index)
    result = pd.concat([features, inertia_features], axis=1)
    return result.loc[:, ~result.columns.duplicated()]


def _temperature_threshold_label(value: float) -> str:
    numeric = float(value)
    if numeric.is_integer():
        label = str(int(numeric))
    else:
        label = f"{numeric:g}"
    return label.replace("-", "m").replace(".", "p")


def _add_temperature_weather_columns(
    data: dict[str, np.ndarray],
    *,
    cluster: str,
    temp_k: np.ndarray,
    hdd_thresholds: list[float],
    cdd_thresholds: list[float],
) -> None:
    temp_c = temp_k.astype(float) - 273.15
    data[f"weather_t2m_C_cluster_{cluster}"] = temp_c
    for threshold in hdd_thresholds:
        label = _temperature_threshold_label(threshold)
        data[f"weather_hdd{label}_cluster_{cluster}"] = np.clip(float(threshold) - temp_c, 0.0, None)
    for threshold in cdd_thresholds:
        label = _temperature_threshold_label(threshold)
        data[f"weather_cdd{label}_cluster_{cluster}"] = np.clip(temp_c - float(threshold), 0.0, None)


def _saturation_vapor_pressure_hpa(temp_c: np.ndarray) -> np.ndarray:
    return 6.1094 * np.exp((17.625 * temp_c) / (243.04 + temp_c))


def _add_humidity_weather_columns(
    data: dict[str, np.ndarray],
    *,
    cluster: str,
    temp_k: np.ndarray,
    dewpoint_k: np.ndarray,
) -> None:
    temp_c = temp_k.astype(float) - 273.15
    dewpoint_c = dewpoint_k.astype(float) - 273.15
    saturation = _saturation_vapor_pressure_hpa(temp_c)
    actual = _saturation_vapor_pressure_hpa(dewpoint_c)
    with np.errstate(divide="ignore", invalid="ignore"):
        relative_humidity = np.clip(100.0 * actual / saturation, 0.0, 100.0)
    data[f"weather_rel_humidity_pct_cluster_{cluster}"] = relative_humidity
    data[f"weather_vpd_hPa_cluster_{cluster}"] = np.clip(saturation - actual, 0.0, None)


def _build_load_weather_features(config: LoadForecastModelConfig) -> pd.DataFrame:
    if config.weather_source == "open_meteo":
        return _build_load_open_meteo_weather_features(config)

    if config.dwd_icon_auto_update:
        from ..preprocessing.dwd_icon_operational import ensure_dwd_icon_weather

        ensure_dwd_icon_weather(
            repo_root=config.repo_root,
            icon_dir=config.icon_dir,
            forecast_start=config.test_start,
            forecast_end=config.test_end,
            run_hour=config.required_run,
            folder_offset_date=config.dwd_folder_offset_date,
            raw_base_dir=config.dwd_icon_raw_dir,
            shapefile_path=config.dwd_icon_aggregation_shapefile_path,
            n_clusters=config.dwd_icon_aggregation_n_clusters,
            buffer_km=config.dwd_icon_aggregation_buffer_km,
            cluster_source=config.dwd_icon_aggregation_cluster_source,
            cluster_output_file=config.dwd_icon_aggregation_cluster_output_file,
            capacity_file=config.dwd_icon_aggregation_capacity_file,
            capacity_weighted_aggregation=config.dwd_icon_aggregation_capacity_weighted,
            variables=config.dwd_icon_download_variables,
            base_url=config.dwd_icon_base_url,
            timeout_seconds=config.dwd_icon_download_timeout_seconds,
            request_pause_seconds=config.dwd_icon_request_pause_seconds,
            catch_up_missing_days=config.dwd_icon_catch_up_missing_days,
            force=config.dwd_icon_force_update,
        )

    df_hourly, df_qh = load_dwd(
        icon_dir=config.icon_dir,
        start_folder_date=config.start_folder_date,
        required_run=config.required_run,
        skip_dates=set(config.skip_dates),
        folder_offset_date=config.dwd_folder_offset_date,
        target_tz=config.target_tz,
    )

    hdd_thresholds = config.weather_hdd_thresholds if config.include_rich_temperature_features else [18.0]
    cdd_thresholds = config.weather_cdd_thresholds if config.include_rich_temperature_features else [22.0]
    hourly_data: dict[str, np.ndarray] = {}
    for column in sorted(col for col in df_hourly.columns if col.startswith("t2m_cluster_")):
        cluster = _cluster_id(column)
        _add_temperature_weather_columns(
            hourly_data,
            cluster=cluster,
            temp_k=df_hourly[column].to_numpy(dtype=float),
            hdd_thresholds=hdd_thresholds,
            cdd_thresholds=cdd_thresholds,
        )

    for column in sorted(col for col in df_hourly.columns if col.startswith("td2m_cluster_")):
        cluster = _cluster_id(column)
        dewpoint_k = df_hourly[column].to_numpy(dtype=float)
        hourly_data[f"weather_td2m_C_cluster_{cluster}"] = dewpoint_k - 273.15
        temp_column = f"t2m_cluster_{cluster}"
        if temp_column in df_hourly.columns:
            _add_humidity_weather_columns(
                hourly_data,
                cluster=cluster,
                temp_k=df_hourly[temp_column].to_numpy(dtype=float),
                dewpoint_k=dewpoint_k,
            )

    for prefix, name in [
        ("sp_cluster_", "weather_sp_Pa"),
        ("tp_cluster_", "weather_precip"),
        ("sde_cluster_", "weather_snow_depth"),
        ("snow_gsp_cluster_", "weather_snowfall"),
        ("vmax10m_cluster_", "weather_vmax10m"),
    ]:
        for column in sorted(col for col in df_hourly.columns if col.startswith(prefix)):
            hourly_data[f"{name}_cluster_{_cluster_id(column)}"] = df_hourly[column].to_numpy(dtype=float)

    for u_col in sorted(col for col in df_hourly.columns if col.startswith("u10_cluster_")):
        cluster = _cluster_id(u_col)
        v_col = f"v10_cluster_{cluster}"
        if v_col not in df_hourly.columns:
            continue
        speed = np.sqrt(df_hourly[u_col].to_numpy(dtype=float) ** 2 + df_hourly[v_col].to_numpy(dtype=float) ** 2)
        hourly_data[f"weather_wind_speed_10m_cluster_{cluster}"] = speed

    hourly_features = pd.DataFrame(hourly_data, index=df_hourly.index)

    qh_data: dict[str, np.ndarray] = {}
    for dir_col in sorted(col for col in df_qh.columns if col.startswith("ASWDIR_cluster_")):
        cluster = _cluster_id(dir_col)
        dif_col = f"ASWDIFD_cluster_{cluster}"
        if dif_col not in df_qh.columns:
            continue
        direct = np.clip(df_qh[dir_col].to_numpy(dtype=float), 0.0, None)
        diffuse = np.clip(df_qh[dif_col].to_numpy(dtype=float), 0.0, None)
        qh_data[f"weather_solar_direct_cluster_{cluster}"] = direct
        qh_data[f"weather_solar_diffuse_cluster_{cluster}"] = diffuse
        qh_data[f"weather_solar_global_cluster_{cluster}"] = direct + diffuse

    qh_features = pd.DataFrame(qh_data, index=df_qh.index)

    full_index = pd.date_range(
        start=min(hourly_features.index.min(), qh_features.index.min()),
        end=max(hourly_features.index.max(), qh_features.index.max()),
        freq="15min",
        tz=config.target_tz,
        name="timestamp",
    )
    hourly_features = hourly_features.loc[~hourly_features.index.duplicated(keep="last")].reindex(full_index).ffill(limit=3)
    qh_features = qh_features.loc[~qh_features.index.duplicated(keep="last")].reindex(full_index)
    features = pd.concat([hourly_features, qh_features], axis=1).sort_index()
    features.index.name = "timestamp"
    return features


def _open_meteo_weather_to_load_features(
    weather: pd.DataFrame,
    config: LoadForecastModelConfig,
) -> pd.DataFrame:
    hdd_thresholds = config.weather_hdd_thresholds if config.include_rich_temperature_features else [18.0]
    cdd_thresholds = config.weather_cdd_thresholds if config.include_rich_temperature_features else [22.0]
    hourly_data: dict[str, np.ndarray] = {}
    for column in sorted(col for col in weather.columns if col.startswith("t2m_cluster_")):
        cluster = _cluster_id(column)
        _add_temperature_weather_columns(
            hourly_data,
            cluster=cluster,
            temp_k=weather[column].to_numpy(dtype=float),
            hdd_thresholds=hdd_thresholds,
            cdd_thresholds=cdd_thresholds,
        )

    for column in sorted(col for col in weather.columns if col.startswith("td2m_cluster_")):
        cluster = _cluster_id(column)
        dewpoint_k = weather[column].to_numpy(dtype=float)
        hourly_data[f"weather_td2m_C_cluster_{cluster}"] = dewpoint_k - 273.15
        temp_column = f"t2m_cluster_{cluster}"
        if temp_column in weather.columns:
            _add_humidity_weather_columns(
                hourly_data,
                cluster=cluster,
                temp_k=weather[temp_column].to_numpy(dtype=float),
                dewpoint_k=dewpoint_k,
            )

    for prefix, name in [
        ("sp_cluster_", "weather_sp_Pa"),
        ("tp_cluster_", "weather_precip"),
        ("sde_cluster_", "weather_snow_depth"),
        ("tcc_cluster_", "weather_cloud_cover"),
    ]:
        for column in sorted(col for col in weather.columns if col.startswith(prefix)):
            hourly_data[f"{name}_cluster_{_cluster_id(column)}"] = weather[column].to_numpy(dtype=float)

    for u_prefix, v_prefix, output_name in [
        ("u10_cluster_", "v10_cluster_", "weather_wind_speed_10m"),
        ("u100_cluster_", "v100_cluster_", "weather_wind_speed_100m"),
    ]:
        for u_col in sorted(col for col in weather.columns if col.startswith(u_prefix)):
            cluster = _cluster_id(u_col)
            v_col = f"{v_prefix}{cluster}"
            if v_col not in weather.columns:
                continue
            speed = np.sqrt(weather[u_col].to_numpy(dtype=float) ** 2 + weather[v_col].to_numpy(dtype=float) ** 2)
            hourly_data[f"{output_name}_cluster_{cluster}"] = speed

    qh_data: dict[str, np.ndarray] = {}
    for global_col, direct_col, diffuse_col in [
        ("ssrd_cluster_", "fdir_cluster_", None),
    ]:
        for column in sorted(col for col in weather.columns if col.startswith(global_col)):
            cluster = _cluster_id(column)
            qh_data[f"weather_solar_global_cluster_{cluster}"] = np.clip(weather[column].to_numpy(dtype=float), 0.0, None)
            fdir_column = f"{direct_col}{cluster}"
            if fdir_column in weather.columns:
                direct = np.clip(weather[fdir_column].to_numpy(dtype=float), 0.0, None)
                qh_data[f"weather_solar_direct_cluster_{cluster}"] = direct
                qh_data[f"weather_solar_diffuse_cluster_{cluster}"] = (
                    qh_data[f"weather_solar_global_cluster_{cluster}"] - direct
                )

    hourly_features = pd.DataFrame(hourly_data, index=weather.index)
    qh_features = pd.DataFrame(qh_data, index=weather.index)
    full_index = pd.date_range(
        start=weather.index.min(),
        end=weather.index.max() + pd.Timedelta(minutes=45),
        freq="15min",
        tz=config.target_tz,
        name="timestamp",
    )
    hourly_features = hourly_features.loc[~hourly_features.index.duplicated(keep="last")].reindex(full_index).ffill(limit=3)
    qh_features = qh_features.loc[~qh_features.index.duplicated(keep="last")].reindex(full_index).ffill(limit=3)
    features = pd.concat([hourly_features, qh_features], axis=1).sort_index()
    features.index.name = "timestamp"
    return features


def _load_cached_open_meteo_weather_features(path: Path, config: LoadForecastModelConfig) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Extra Open-Meteo weather file does not exist: {path}")
    return _open_meteo_weather_to_load_features(_load_timestamp_csv(path, config.target_tz), config)


def _load_weather_ensemble_column_name(column: str, stat: str) -> str:
    parsed = _split_weather_cluster_column(column)
    if parsed is None:
        return f"weather_ens_{stat}_{column}"
    base_name, cluster_id = parsed
    return f"{base_name}_ens_{stat}_cluster_{cluster_id}"


def _summarise_load_open_meteo_provider_features(
    provider_features: list[pd.DataFrame],
    *,
    stats: list[str],
) -> pd.DataFrame:
    frames = [frame for frame in provider_features if not frame.empty]
    if len(frames) < 2:
        return pd.DataFrame(index=frames[0].index if frames else None)

    base_index = frames[0].index
    common_columns = set(frames[0].columns)
    for frame in frames[1:]:
        common_columns &= set(frame.columns)

    weather_columns = [
        column
        for column in sorted(common_columns)
        if _split_weather_cluster_column(column) is not None
    ]
    if not weather_columns:
        return pd.DataFrame(index=base_index)

    summary: dict[str, np.ndarray] = {}
    count_feature_added = False
    for column in weather_columns:
        values = np.vstack(
            [frame[column].reindex(base_index).to_numpy(dtype=float) for frame in frames]
        )
        valid = np.isfinite(values)
        counts = valid.sum(axis=0).astype(float)
        safe_counts = np.where(counts > 0, counts, np.nan)

        if not count_feature_added:
            summary["weather_ens_available_provider_count"] = counts
            count_feature_added = True

        if "mean" in stats:
            summary[_load_weather_ensemble_column_name(column, "mean")] = np.nansum(values, axis=0) / safe_counts
        if "std" in stats:
            means = np.nansum(values, axis=0) / safe_counts
            centered = np.where(valid, values - means, 0.0)
            variance = np.sum(centered**2, axis=0) / safe_counts
            summary[_load_weather_ensemble_column_name(column, "std")] = np.where(
                counts > 1,
                np.sqrt(variance),
                0.0,
            )
        if "min" in stats:
            finite_values = np.where(valid, values, np.inf)
            summary[_load_weather_ensemble_column_name(column, "min")] = np.where(
                counts > 0,
                finite_values.min(axis=0),
                np.nan,
            )
        if "max" in stats or "range" in stats:
            finite_values = np.where(valid, values, -np.inf)
            maxima = np.where(counts > 0, finite_values.max(axis=0), np.nan)
            if "max" in stats:
                summary[_load_weather_ensemble_column_name(column, "max")] = maxima
        if "range" in stats:
            finite_min_values = np.where(valid, values, np.inf)
            minima = np.where(counts > 0, finite_min_values.min(axis=0), np.nan)
            summary[_load_weather_ensemble_column_name(column, "range")] = maxima - minima

    result = pd.DataFrame(summary, index=base_index)
    result.index.name = "timestamp"
    return result


def _build_load_open_meteo_weather_features(config: LoadForecastModelConfig) -> pd.DataFrame:
    weather = load_open_meteo(
        cluster_file=config.open_meteo_cluster_file,
        start_date=config.open_meteo_start_date,
        end_date=config.open_meteo_end_date,
        cache_file=config.open_meteo_weather_file,
        base_url=config.open_meteo_base_url,
        api_key_env=config.open_meteo_api_key_env,
        model=config.open_meteo_model,
        hourly_variables=config.open_meteo_hourly_variables,
        batch_size=config.open_meteo_batch_size,
        cell_selection=config.open_meteo_cell_selection,
        timeout_seconds=config.open_meteo_timeout_seconds,
        target_tz=config.target_tz,
        force_download=config.open_meteo_force_download,
        point_selection=config.open_meteo_point_selection,
        max_points_per_cluster=config.open_meteo_max_points_per_cluster,
        api_mode=config.open_meteo_api_mode,
        single_run_hour_utc=config.open_meteo_single_run_hour_utc,
        single_run_forecast_days=config.open_meteo_single_run_forecast_days,
        request_pause_seconds=config.open_meteo_request_pause_seconds,
        retry_attempts=config.open_meteo_retry_attempts,
        retry_backoff_seconds=config.open_meteo_retry_backoff_seconds,
        skip_unavailable_runs=config.open_meteo_skip_unavailable_runs,
        fallback_previous_runs=config.open_meteo_fallback_previous_runs,
        fallback_step_hours=config.open_meteo_fallback_step_hours,
        fallback_max_lookback_hours=config.open_meteo_fallback_max_lookback_hours,
    )
    features = _open_meteo_weather_to_load_features(weather, config)

    if config.include_open_meteo_weather_ensemble_features and config.extra_open_meteo_weather_files:
        provider_features = [features]
        provider_features.extend(
            _load_cached_open_meteo_weather_features(path, config)
            for path in config.extra_open_meteo_weather_files
        )
        ensemble = _summarise_load_open_meteo_provider_features(
            provider_features,
            stats=config.open_meteo_weather_ensemble_stats,
        )
        features = pd.concat([features, ensemble], axis=1).sort_index()
        features = features.loc[:, ~features.columns.duplicated()]

    return features


def build_load_forecast_dataset(config: LoadForecastModelConfig) -> pd.DataFrame:
    """Build timestamp-level direct-load features and actual-load targets."""
    actual = _load_or_fetch_actual_load(config)

    feature_blocks: list[pd.DataFrame] = []
    if config.include_weather_features:
        weather = _build_load_weather_features(config)
        feature_index = weather.index
        feature_blocks.append(weather)
    else:
        feature_index = actual.index

    if config.include_calendar_features:
        feature_blocks.append(
            _build_calendar_features(
                feature_index,
                target_tz=config.target_tz,
                include_holidays=config.include_holiday_features,
                include_bridge_days=config.include_bridge_day_features,
                calendar_harmonics=config.calendar_harmonics,
            )
        )

    if config.include_regional_holiday_features:
        if config.regional_holiday_weight_file is None:
            raise ValueError("regional_holiday_weight_file is required when include_regional_holiday_features is true.")
        feature_blocks.append(
            _build_regional_holiday_features(
                feature_index,
                target_tz=config.target_tz,
                weight_file=config.regional_holiday_weight_file,
                region_column=config.regional_holiday_region_column,
                weight_column=config.regional_holiday_weight_column,
                feature_prefix=config.regional_holiday_feature_prefix,
            )
        )

    if config.include_entsoe_forecast_features or config.load_target_mode == "entsoe_residual":
        feature_blocks.append(_build_entsoe_forecast_features(config, feature_index))

    if config.include_regional_load_features:
        feature_blocks.append(_build_regional_load_features(config, feature_index))

    if config.include_entsoe_error_lag_features or config.include_entsoe_error_rolling_features:
        feature_blocks.append(
            _build_entsoe_error_features(
                config=config,
                actual_load=actual,
                index=feature_index,
            )
        )

    if config.include_partial_load_features:
        feature_blocks.append(
            _build_partial_load_features(
                actual_load=actual,
                index=feature_index,
                target_tz=config.target_tz,
                reference_day=config.partial_load_reference_day,
                comparison_lag_days=config.partial_load_comparison_lag_days,
                morning_end_hour=config.partial_load_morning_end_hour,
                morning_end_minute=config.partial_load_morning_end_minute,
                include_shape_features=config.include_partial_load_shape_features,
                point_times=config.partial_load_point_times,
            )
        )

    if config.include_actual_load_lag_features:
        feature_blocks.append(
            _build_actual_load_lag_features(
                actual_load=actual,
                index=feature_index,
                lag_days=config.actual_load_lag_days,
            )
        )

    if not feature_blocks:
        raise ValueError("At least one load forecast feature block must be enabled.")

    features = pd.concat(feature_blocks, axis=1).sort_index()
    features = features.loc[:, ~features.columns.duplicated()]
    if config.include_weather_cluster_spread_features:
        features = _add_weather_cluster_spread_features(
            features,
            base_names=config.weather_spread_feature_bases or config.weather_weighted_feature_bases,
            stats=config.weather_spread_stats,
        )
    cluster_weights = None
    needs_cluster_weights = (
        config.include_weighted_weather_features
        or config.include_weighted_weather_quantile_features
    )
    if needs_cluster_weights:
        if config.weather_cluster_weight_file is None:
            raise ValueError("weather_cluster_weight_file is required for weighted weather features.")
        cluster_weights = _load_cluster_weights(
            config.weather_cluster_weight_file,
            cluster_id_column=config.weather_cluster_id_column,
            weight_column=config.weather_cluster_weight_column,
        )
    if config.include_weighted_weather_quantile_features:
        if cluster_weights is None:
            raise ValueError("Weighted weather quantile features require cluster weights.")
        features = _add_weighted_weather_quantile_features(
            features,
            cluster_weights,
            base_names=config.weighted_weather_quantile_feature_bases or config.weather_spread_feature_bases,
            quantiles=config.weighted_weather_quantiles,
        )
    if config.include_weighted_weather_features:
        if cluster_weights is None:
            raise ValueError("Weighted weather features require cluster weights.")
        features = _add_weighted_weather_features(
            features,
            cluster_weights,
            prefix=config.weather_weighted_feature_prefix,
            base_names=config.weather_weighted_feature_bases,
        )
    if config.include_weighted_weather_features:
        if config.include_weighted_weather_daily_features:
            features = _add_weighted_weather_daily_features(
                features,
                prefix=config.weather_weighted_feature_prefix,
                base_names=config.weighted_weather_daily_feature_bases or config.weather_weighted_feature_bases,
                stats=config.weighted_weather_daily_stats,
                lag_days=config.weighted_weather_daily_lag_days,
            )
        if config.include_weighted_weather_inertia_features:
            features = _add_weighted_weather_inertia_features(
                features,
                prefix=config.weather_weighted_feature_prefix,
                base_names=config.weighted_weather_inertia_feature_bases or config.weather_weighted_feature_bases,
                windows_hours=config.weighted_weather_inertia_windows_hours,
                stats=config.weighted_weather_inertia_stats,
            )
        if not config.keep_weather_cluster_features:
            features = _drop_weather_cluster_features(features)
    if config.include_weather_time_interactions:
        features = _add_weather_time_interactions(
            features,
            weighted_weather_prefix=config.weather_weighted_feature_prefix,
        )
    target = actual[["load_actual"]].rename(columns={"load_actual": "Load_Actual_MW"})
    dataset = features.join(target, how="left").sort_index()
    dataset.index.name = "timestamp"
    return dataset


def _feature_columns(dataset: pd.DataFrame) -> list[str]:
    return [
        column
        for column in dataset.select_dtypes(include="number").columns
        if column != "Load_Actual_MW"
    ]


def _make_model(config: LoadForecastModelConfig):
    if config.model_type == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            loss="squared_error",
            max_iter=config.hgb_max_iter,
            learning_rate=config.hgb_learning_rate,
            max_leaf_nodes=config.hgb_max_leaf_nodes,
            l2_regularization=config.hgb_l2_regularization,
            random_state=config.random_state,
        )
    if config.model_type == "ridge":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            Ridge(alpha=config.ridge_alpha),
        )
    if config.model_type == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise ImportError("model_type='lightgbm' requires the optional `lightgbm` package.") from exc

        return LGBMRegressor(
            objective="regression",
            n_estimators=config.lgbm_n_estimators,
            learning_rate=config.lgbm_learning_rate,
            num_leaves=config.lgbm_num_leaves,
            min_child_samples=config.lgbm_min_child_samples,
            subsample=config.lgbm_subsample,
            colsample_bytree=config.lgbm_colsample_bytree,
            reg_lambda=config.lgbm_reg_lambda,
            random_state=config.random_state,
            n_jobs=-1,
            verbosity=-1,
        )
    if config.model_type == "tabpfn":
        try:
            from tabpfn import TabPFNRegressor
        except ImportError as exc:
            raise ImportError("model_type='tabpfn' requires the optional `tabpfn` package.") from exc

        return TabPFNRegressor(
            n_estimators=config.tabpfn_n_estimators,
            device=config.tabpfn_device,
            ignore_pretraining_limits=config.tabpfn_ignore_pretraining_limits,
            inference_precision=config.tabpfn_inference_precision,
            fit_mode=config.tabpfn_fit_mode,
            n_preprocessing_jobs=config.tabpfn_n_preprocessing_jobs,
            random_state=config.random_state,
        )
    raise ValueError(f"Unsupported load forecast model_type: {config.model_type!r}")


def _select_features(X_train: pd.DataFrame, y_train: pd.Series, config: LoadForecastModelConfig) -> list[str]:
    max_features = config.max_features
    if max_features is None or max_features <= 0 or X_train.shape[1] <= max_features:
        return list(X_train.columns)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in divide")
        scores = (
            X_train.corrwith(y_train)
            .abs()
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .sort_values(ascending=False)
        )
    selected = list(scores.head(max_features).index)
    return selected or list(X_train.columns)


def _prediction_upper_bound(y_train: pd.Series, config: LoadForecastModelConfig) -> float | None:
    if not config.clip_predictions_to_training_target_range:
        return None
    valid = y_train.dropna()
    if valid.empty:
        return None
    if config.prediction_upper_quantile >= 1.0:
        return float(valid.max())
    return float(valid.quantile(config.prediction_upper_quantile))


def _prepare_tabpfn_matrices(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = X_train.replace([np.inf, -np.inf], np.nan)
    test = X_test.replace([np.inf, -np.inf], np.nan)
    medians = train.median(numeric_only=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    train = train.fillna(medians).fillna(0.0)
    test = test.fillna(medians).fillna(0.0)
    return train.astype(float), test.astype(float)


def _robust_target_params(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    center = float(np.median(finite))
    q25, q75 = np.percentile(finite, [25, 75])
    scale = float((q75 - q25) / 1.349)
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(np.std(finite))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    return center, scale


def _transform_load_target(
    y_train: pd.Series,
    config: LoadForecastModelConfig,
) -> tuple[np.ndarray, dict[str, float | str]]:
    values = y_train.to_numpy(dtype=float)
    if config.tabpfn_target_transform == "none":
        return values, {"kind": "none"}

    center, scale = _robust_target_params(values)
    return np.arcsinh((values - center) / scale), {
        "kind": "asinh_robust",
        "center": center,
        "scale": scale,
    }


def _inverse_load_target(values: np.ndarray, params: dict[str, float | str]) -> np.ndarray:
    if params.get("kind") != "asinh_robust":
        return np.asarray(values, dtype=float)
    center = float(params["center"])
    scale = float(params["scale"])
    return center + scale * np.sinh(np.asarray(values, dtype=float))


def _is_tabpfn_oom_error(exc: Exception) -> bool:
    message = str(exc).lower()
    name = exc.__class__.__name__.lower()
    return "outofmemory" in name or "out of memory" in message or "oom" in message


def _predict_tabpfn_point(
    model: Any,
    X_test: pd.DataFrame,
    config: LoadForecastModelConfig,
) -> np.ndarray:
    batch_size = config.tabpfn_predict_batch_size
    if batch_size is None or batch_size <= 0 or batch_size >= len(X_test):
        return np.asarray(model.predict(X_test, output_type=config.tabpfn_output_type), dtype=float)

    chunks = []
    for start in range(0, len(X_test), batch_size):
        chunk = X_test.iloc[start : start + batch_size]
        chunks.append(np.asarray(model.predict(chunk, output_type=config.tabpfn_output_type), dtype=float))
    return np.concatenate(chunks)


def _evenly_spaced_datetime_index(index: pd.DatetimeIndex, n_rows: int) -> pd.DatetimeIndex:
    if n_rows <= 0 or len(index) == 0:
        return pd.DatetimeIndex([], tz=index.tz)
    if len(index) <= n_rows:
        return index
    positions = np.linspace(0, len(index) - 1, n_rows).round().astype(int)
    return index[np.unique(positions)]


def _select_tabpfn_context(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    config: LoadForecastModelConfig,
) -> tuple[pd.DataFrame, pd.Series]:
    max_rows = config.tabpfn_max_train_rows
    if max_rows is None or len(X_train) <= max_rows:
        return X_train, y_train

    if config.tabpfn_context_selection == "tail":
        selected_index = X_train.tail(max_rows).index
    else:
        forecast_day = X_test.index.min()
        recent_rows = int(round(max_rows * config.tabpfn_context_recent_fraction))
        same_weekday_rows = int(round(max_rows * config.tabpfn_context_same_weekday_fraction))
        recent_rows = min(max(recent_rows, 0), max_rows)
        same_weekday_rows = min(max(same_weekday_rows, 0), max_rows - recent_rows)

        selected_parts: list[pd.DatetimeIndex] = []
        if recent_rows:
            selected_parts.append(X_train.tail(recent_rows).index)

        selected_index = selected_parts[0] if selected_parts else pd.DatetimeIndex([], tz=X_train.index.tz)
        if same_weekday_rows and isinstance(X_train.index, pd.DatetimeIndex):
            same_weekday_index = X_train.index[X_train.index.dayofweek == forecast_day.dayofweek]
            same_weekday_index = same_weekday_index.difference(selected_index).sort_values()
            selected_parts.append(same_weekday_index[-same_weekday_rows:])
            selected_index = selected_index.union(selected_parts[-1]).sort_values()

        remaining_rows = max_rows - len(selected_index)
        if remaining_rows > 0:
            older_index = X_train.index.difference(selected_index).sort_values()
            selected_parts.append(_evenly_spaced_datetime_index(older_index, remaining_rows))
            selected_index = selected_index.union(selected_parts[-1]).sort_values()

        if len(selected_index) > max_rows:
            selected_index = selected_index[-max_rows:]

    return X_train.loc[selected_index], y_train.loc[selected_index]


def _fit_predict_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    config: LoadForecastModelConfig,
) -> tuple[np.ndarray, int]:
    if config.model_type != "tabpfn":
        model.fit(X_train, y_train)
        return np.asarray(model.predict(X_test), dtype=float), len(X_train)

    X_train, y_train = _select_tabpfn_context(X_train, y_train, X_test, config)

    X_train_prepared, X_test_prepared = _prepare_tabpfn_matrices(X_train, X_test)
    y_model, target_params = _transform_load_target(y_train, config)
    try:
        model.fit(X_train_prepared, y_model)
        raw = _predict_tabpfn_point(model, X_test_prepared, config)
    except Exception as exc:
        if _is_tabpfn_oom_error(exc):
            raise RuntimeError(
                "TabPFN load forecast ran out of memory. Reduce `tabpfn_max_train_rows`, "
                "`max_features`, `tabpfn_n_estimators`, or set `tabpfn_device: cpu`."
            ) from exc
        raise
    return _inverse_load_target(raw, target_params), len(X_train_prepared)


def _model_target_series(dataset: pd.DataFrame, config: LoadForecastModelConfig) -> pd.Series:
    if config.load_target_mode == "actual_load":
        return dataset["Load_Actual_MW"]
    if config.load_target_mode == "entsoe_residual":
        if "Load_Benchmark_MW" not in dataset.columns:
            raise ValueError("load_target_mode='entsoe_residual' requires Load_Benchmark_MW in the dataset.")
        return dataset["Load_Actual_MW"] - dataset["Load_Benchmark_MW"]
    raise ValueError(f"Unsupported load_target_mode: {config.load_target_mode!r}")


def _finalise_model_predictions(
    raw_predictions: np.ndarray,
    dataset: pd.DataFrame,
    test_mask: np.ndarray,
    y_train_actual: pd.Series,
    config: LoadForecastModelConfig,
) -> np.ndarray:
    if config.load_target_mode == "entsoe_residual":
        benchmark = dataset.loc[test_mask, "Load_Benchmark_MW"].to_numpy(dtype=float)
        predictions = benchmark + raw_predictions
        upper_bound = _prediction_upper_bound(y_train_actual, config)
    else:
        predictions = raw_predictions
        upper_bound = _prediction_upper_bound(y_train_actual, config)

    return np.clip(predictions, 0.0, upper_bound)


def _hour_block_pairs(boundaries: list[int]) -> list[tuple[int, int]]:
    return [(int(start), int(end)) for start, end in zip(boundaries[:-1], boundaries[1:])]


def rolling_load_forecast(
    dataset: pd.DataFrame,
    config: LoadForecastModelConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train rolling direct load models and predict complete forecast days."""
    features = _feature_columns(dataset)
    if not features:
        raise ValueError("No load forecast feature columns were built.")
    if "Load_Actual_MW" not in dataset.columns:
        raise ValueError("Dataset is missing target column 'Load_Actual_MW'.")

    test_start = _as_local_day(config.test_start, config.target_tz)
    test_end = _as_local_day(config.test_end, config.target_tz)
    forecast_days = pd.date_range(start=test_start, end=test_end, freq="D", tz=config.target_tz)
    min_train_rows = config.min_train_days * 96
    model_target = _model_target_series(dataset, config)

    forecast_blocks: list[pd.DataFrame] = []
    runtime_rows: list[dict[str, Any]] = []
    for day in forecast_days:
        day_start = time.perf_counter()
        train_start = day - pd.Timedelta(days=config.train_days_rolling)
        train_end = _target_availability_cutoff(day, config)
        test_end_ts = day + pd.Timedelta(days=1) - pd.Timedelta(minutes=15)

        train_mask = (dataset.index >= train_start) & (dataset.index <= train_end)
        test_mask = (dataset.index >= day) & (dataset.index <= test_end_ts)
        X_test = dataset.loc[test_mask, features]
        if len(X_test) == 0:
            continue

        selected_feature_counts: list[int] = []
        if config.model_granularity == "global":
            X_train_all = dataset.loc[train_mask, features]
            y_train_all = model_target.loc[train_mask]
            y_train_actual_all = dataset.loc[train_mask, "Load_Actual_MW"]
            valid_train = y_train_all.notna() & _weather_present_mask(dataset, train_mask, config)
            X_train = X_train_all.loc[valid_train]
            y_train = y_train_all.loc[valid_train]
            y_train_actual = y_train_actual_all.loc[valid_train]
            if len(X_train) < min_train_rows:
                continue

            selected_features = _select_features(X_train, y_train, config)
            model = _make_model(config)
            raw_predictions, train_rows = _fit_predict_model(
                model,
                X_train[selected_features],
                y_train,
                X_test[selected_features],
                config,
            )

            predictions = pd.Series(
                _finalise_model_predictions(
                    raw_predictions,
                    dataset,
                    test_mask,
                    y_train_actual,
                    config,
                ),
                index=X_test.index,
                dtype=float,
            )
            selected_feature_counts.append(len(selected_features))
            n_models = 1
        elif config.model_granularity == "hour_block":
            predictions = pd.Series(np.nan, index=X_test.index, dtype=float)
            train_rows = 0
            n_models = 0
            for start_hour, end_hour in _hour_block_pairs(config.hour_block_boundaries):
                block_train_mask = (
                    train_mask
                    & (dataset.index.hour >= start_hour)
                    & (dataset.index.hour < end_hour)
                )
                block_test_mask = (
                    test_mask
                    & (dataset.index.hour >= start_hour)
                    & (dataset.index.hour < end_hour)
                )
                if not block_test_mask.any():
                    continue

                X_train_all = dataset.loc[block_train_mask, features]
                y_train_all = model_target.loc[block_train_mask]
                y_train_actual_all = dataset.loc[block_train_mask, "Load_Actual_MW"]
                valid_train = y_train_all.notna() & _weather_present_mask(dataset, block_train_mask, config)
                X_train = X_train_all.loc[valid_train]
                y_train = y_train_all.loc[valid_train]
                y_train_actual = y_train_actual_all.loc[valid_train]
                block_min_train_rows = config.min_train_days * max(end_hour - start_hour, 1) * 4
                if len(X_train) < block_min_train_rows:
                    continue

                X_test_block = dataset.loc[block_test_mask, features]
                selected_features = _select_features(X_train, y_train, config)
                model = _make_model(config)
                raw_block_predictions, effective_train_rows = _fit_predict_model(
                    model,
                    X_train[selected_features],
                    y_train,
                    X_test_block[selected_features],
                    config,
                )
                block_predictions = _finalise_model_predictions(
                    raw_block_predictions,
                    dataset,
                    block_test_mask,
                    y_train_actual,
                    config,
                )
                predictions.loc[X_test_block.index] = block_predictions
                train_rows += effective_train_rows
                selected_feature_counts.append(len(selected_features))
                n_models += 1

            if predictions.isna().any():
                missing_hours = sorted(set(int(hour) for hour in predictions.index[predictions.isna()].hour))
                raise ValueError(f"Hour-block load model did not produce predictions for hours: {missing_hours}")
        else:
            raise ValueError(f"Unsupported model_granularity: {config.model_granularity!r}")

        day_forecast = pd.DataFrame(index=X_test.index)
        day_forecast["Load_Model_MW"] = predictions.to_numpy(dtype=float)
        if "Load_Benchmark_MW" in dataset.columns:
            day_forecast["Load_Benchmark_MW"] = dataset.loc[test_mask, "Load_Benchmark_MW"]
        day_forecast["Load_Actual_MW"] = dataset.loc[test_mask, "Load_Actual_MW"]
        forecast_blocks.append(day_forecast)

        runtime_rows.append(
            {
                "date": day,
                "train_rows": train_rows,
                "test_rows": len(X_test),
                "n_features": len(features),
                "n_selected_features": max(selected_feature_counts) if selected_feature_counts else 0,
                "n_models": n_models,
                "model_type": config.model_type,
                "model_granularity": config.model_granularity,
                "runtime_seconds": time.perf_counter() - day_start,
            }
        )
        print(
            f"  {day.date()}  {config.model_type}/{config.model_granularity}  "
            f"{runtime_rows[-1]['runtime_seconds']:.1f}s"
        )

    if not forecast_blocks:
        raise ValueError("No load forecasts were produced.")

    forecast = pd.concat(forecast_blocks).sort_index()
    forecast.index.name = "timestamp"
    runtime = pd.DataFrame(runtime_rows)
    return forecast, runtime


def _bias_correction_group_values(index: pd.DatetimeIndex, group: str) -> pd.Series:
    if group == "global":
        values = np.zeros(len(index), dtype=int)
    elif group == "hour":
        values = index.hour
    elif group == "mtu":
        values = index.hour * 4 + index.minute // 15
    else:
        raise ValueError(f"Unsupported bias_correction_group: {group!r}")
    return pd.Series(values, index=index)


def apply_rolling_residual_quantiles(
    forecast: pd.DataFrame,
    config: LoadForecastModelConfig | LoadForecastEnsembleConfig,
    *,
    pred_col: str = "Load_Model_MW",
    true_col: str = "Load_Actual_MW",
) -> pd.DataFrame:
    """Add operational residual-calibrated quantile forecasts from past forecast errors."""
    if not config.include_rolling_residual_quantiles:
        return forecast
    if pred_col not in forecast.columns or true_col not in forecast.columns:
        raise ValueError(f"Forecast must contain {pred_col!r} and {true_col!r} for residual quantiles.")

    quantiles = sorted({float(quantile) for quantile in config.residual_quantiles})
    quantile_cols = [quantile_column(quantile) for quantile in quantiles]
    calibrated = forecast.copy()
    residuals = calibrated[true_col].astype(float) - calibrated[pred_col].astype(float)
    group_values = _bias_correction_group_values(calibrated.index, config.residual_quantile_group)
    window = pd.Timedelta(days=config.residual_quantile_window_days)
    min_observations = config.residual_quantile_min_observations

    quantile_values = pd.DataFrame(np.nan, index=calibrated.index, columns=quantile_cols, dtype=float)
    for day in pd.DatetimeIndex(calibrated.index.normalize().unique()).sort_values():
        cutoff = _target_availability_cutoff(day, config)
        window_start = cutoff - window + pd.Timedelta(minutes=15)
        day_mask = calibrated.index.normalize() == day
        history_mask = (calibrated.index >= window_start) & (calibrated.index <= cutoff) & residuals.notna()
        if not history_mask.any() or not day_mask.any():
            continue

        history_residuals = residuals.loc[history_mask]
        global_quantiles = (
            history_residuals.quantile(quantiles).to_dict()
            if len(history_residuals) >= min_observations
            else {}
        )
        grouped_residuals = {
            key: values
            for key, values in history_residuals.groupby(group_values.loc[history_mask])
            if len(values) >= min_observations
        }
        grouped_quantiles = {
            key: values.quantile(quantiles).to_dict()
            for key, values in grouped_residuals.items()
        }

        for position in np.flatnonzero(day_mask):
            index_value = calibrated.index[position]
            group_key = group_values.iloc[position]
            residual_quantiles = grouped_quantiles.get(group_key, global_quantiles)
            if not residual_quantiles:
                continue
            base_prediction = float(calibrated[pred_col].iloc[position])
            for quantile, column in zip(quantiles, quantile_cols, strict=True):
                if quantile in residual_quantiles:
                    quantile_values.at[index_value, column] = max(0.0, base_prediction + float(residual_quantiles[quantile]))

    if quantile_cols:
        scale = float(config.residual_quantile_spread_scale)
        median_col = quantile_column(0.5)
        if scale != 1.0 and median_col in quantile_values.columns:
            median = quantile_values[median_col]
            quantile_values.loc[:, quantile_cols] = median.to_numpy()[:, None] + scale * (
                quantile_values[quantile_cols].sub(median, axis=0)
            )
            quantile_values.loc[:, quantile_cols] = quantile_values[quantile_cols].clip(lower=0.0)
        quantile_values.loc[:, quantile_cols] = np.maximum.accumulate(
            quantile_values[quantile_cols].to_numpy(dtype=float),
            axis=1,
        )
        calibrated[quantile_cols] = quantile_values[quantile_cols]
    return calibrated


def _quantile_evaluation_frame(
    forecast: pd.DataFrame,
    config: LoadForecastModelConfig | LoadForecastEnsembleConfig,
) -> pd.DataFrame:
    start = config.quantile_evaluation_start or config.test_start
    end = config.quantile_evaluation_end or config.test_end
    start_ts = _as_local_day(start, config.target_tz)
    end_ts = _as_local_day(end, config.target_tz) + pd.Timedelta(days=1) - pd.Timedelta(minutes=15)
    return forecast.loc[(forecast.index >= start_ts) & (forecast.index <= end_ts)]


def _quantile_period_metrics(
    forecast: pd.DataFrame,
    *,
    quantiles: list[float],
    true_col: str,
    period: str,
) -> dict[str, Any]:
    quantile_cols = [quantile_column(quantile) for quantile in quantiles]
    valid = forecast[[true_col, *quantile_cols]].dropna()
    row: dict[str, Any] = {
        "target": "Load",
        "model": "RollingResidualQuantiles",
        "period": period,
        "lqs": np.nan,
        "n_obs": int(len(valid)),
    }
    if valid.empty:
        return row

    y_true = valid[true_col].to_numpy(dtype=float)
    losses = [
        pinball_score(y_true, valid[quantile_column(quantile)].to_numpy(dtype=float), quantile)
        for quantile in quantiles
    ]
    row["lqs"] = float(np.mean(np.vstack(losses), axis=0).mean())

    if 0.5 in quantiles:
        row["median_mae"] = float(np.abs(y_true - valid[quantile_column(0.5)].to_numpy(dtype=float)).mean())
    for nominal, lower, upper in [(0.50, 0.25, 0.75), (0.95, 0.025, 0.975)]:
        if lower in quantiles and upper in quantiles:
            lower_values = valid[quantile_column(lower)].to_numpy(dtype=float)
            upper_values = valid[quantile_column(upper)].to_numpy(dtype=float)
            row[f"coverage_{nominal:.2f}"] = float(((y_true >= lower_values) & (y_true <= upper_values)).mean())
    return row


def evaluate_load_quantile_forecast(
    forecast: pd.DataFrame,
    quantiles: list[float],
    *,
    true_col: str = "Load_Actual_MW",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    quantile_cols = [quantile_column(quantile) for quantile in quantiles]
    if forecast.empty or true_col not in forecast.columns or any(column not in forecast.columns for column in quantile_cols):
        return pd.DataFrame(rows)

    rows.append(_quantile_period_metrics(forecast, quantiles=quantiles, true_col=true_col, period="full"))
    month_index = forecast.index.tz_localize(None).to_period("M")
    for month, group in forecast.groupby(month_index):
        rows.append(_quantile_period_metrics(group, quantiles=quantiles, true_col=true_col, period=str(month)))
    return pd.DataFrame(rows)


def _period_metrics(
    forecast: pd.DataFrame,
    *,
    pred_col: str,
    true_col: str,
    period: str,
) -> dict[str, Any]:
    stats = point_error_stats(forecast, pred_col, true_col)
    mean_actual = float(stats["mean_actual_mw"])
    rmse = float(stats["rmse"])
    return {
        "target": "Load",
        "model": pred_col.removeprefix("Load_").removesuffix("_MW"),
        "period": period,
        "mae": stats["mae"],
        "mse": stats["mse"],
        "rmse": rmse,
        "r2": stats["r2"],
        "nrmse_mean_load_pct": float(rmse / mean_actual * 100.0) if mean_actual > 0 else np.nan,
        "bias": stats["bias"],
        "mean_actual_mw": mean_actual,
        "n_obs": stats["n_obs"],
    }


def evaluate_load_forecast(
    forecast: pd.DataFrame,
    prediction_columns: list[str] | None = None,
    true_col: str = "Load_Actual_MW",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if forecast.empty:
        return pd.DataFrame(rows)

    if prediction_columns is None:
        prediction_columns = [
            column
            for column in forecast.select_dtypes(include="number").columns
            if column.startswith("Load_") and column.endswith("_MW") and column != true_col
        ]
    if not prediction_columns:
        return pd.DataFrame(rows)

    for pred_col in prediction_columns:
        rows.append(_period_metrics(forecast, pred_col=pred_col, true_col=true_col, period="full"))

    month_index = forecast.index.tz_localize(None).to_period("M")
    for month, group in forecast.groupby(month_index):
        for pred_col in prediction_columns:
            rows.append(_period_metrics(group, pred_col=pred_col, true_col=true_col, period=str(month)))
    return pd.DataFrame(rows)


def _load_model_forecast_csv(path: Path, target_tz: str) -> pd.DataFrame:
    frame = _load_timestamp_csv(path, target_tz)
    missing = set(LOAD_FORECAST_REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Load forecast file {path} is missing required columns: {sorted(missing)}")
    return frame[list(LOAD_FORECAST_REQUIRED_COLUMNS)].sort_index()


def build_load_point_ensemble(
    source_forecasts: dict[str, pd.DataFrame],
    *,
    method: str = "mean",
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Blend saved load point forecasts on their common timestamps."""
    if len(source_forecasts) < 2:
        raise ValueError("Load point ensemble requires at least two source forecasts.")
    if method not in {"mean", "median", "fixed"}:
        raise ValueError(f"Unsupported load ensemble method: {method!r}")

    source_names = list(source_forecasts)
    joined_parts: list[pd.DataFrame] = []
    for name, forecast in source_forecasts.items():
        missing = set(LOAD_FORECAST_REQUIRED_COLUMNS) - set(forecast.columns)
        if missing:
            raise ValueError(f"Source forecast {name!r} is missing required columns: {sorted(missing)}")
        part = forecast[list(LOAD_FORECAST_REQUIRED_COLUMNS)].rename(
            columns={
                "Load_Model_MW": f"Load_Source_{name}_MW",
                "Load_Benchmark_MW": f"Load_Benchmark_MW_{name}",
                "Load_Actual_MW": f"Load_Actual_MW_{name}",
            }
        )
        joined_parts.append(part)

    joined = pd.concat(joined_parts, axis=1, join="inner").sort_index()
    source_columns = [f"Load_Source_{name}_MW" for name in source_names]
    benchmark_columns = [f"Load_Benchmark_MW_{name}" for name in source_names]
    actual_columns = [f"Load_Actual_MW_{name}" for name in source_names]

    benchmark = joined[benchmark_columns[0]].copy()
    actual = joined[actual_columns[0]].copy()
    for column in benchmark_columns[1:]:
        if not np.allclose(benchmark.to_numpy(dtype=float), joined[column].to_numpy(dtype=float), equal_nan=True):
            raise ValueError("Source forecasts have different Load_Benchmark_MW values.")
    for column in actual_columns[1:]:
        if not np.allclose(actual.to_numpy(dtype=float), joined[column].to_numpy(dtype=float), equal_nan=True):
            raise ValueError("Source forecasts have different Load_Actual_MW values.")

    source_values = joined[source_columns].to_numpy(dtype=float)
    if method == "median":
        prediction = np.nanmedian(source_values, axis=1)
    elif method == "fixed":
        if weights is None:
            raise ValueError("Fixed load ensemble requires source weights.")
        weight_values = np.array([float(weights.get(name, 0.0)) for name in source_names], dtype=float)
        if np.any(weight_values < 0) or weight_values.sum() <= 0:
            raise ValueError("Fixed load ensemble weights must be non-negative and sum to a positive value.")
        prediction = np.average(source_values, axis=1, weights=weight_values)
    else:
        prediction = np.nanmean(source_values, axis=1)

    forecast = pd.DataFrame(index=joined.index)
    forecast["Load_Model_MW"] = np.clip(prediction, 0.0, None)
    forecast["Load_Benchmark_MW"] = benchmark
    forecast["Load_Actual_MW"] = actual
    for column in source_columns:
        forecast[column] = joined[column]
    forecast.index.name = "timestamp"
    return forecast.sort_index()


def sum_regional_load_component_forecasts(
    component_forecasts: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Sum regional load component forecasts into one DE-LU-style forecast frame."""
    if not component_forecasts:
        raise ValueError("Regional load forecast requires at least one component forecast.")

    joined_parts: list[pd.DataFrame] = []
    for name, forecast in component_forecasts.items():
        missing = set(LOAD_FORECAST_REQUIRED_COLUMNS) - set(forecast.columns)
        if missing:
            raise ValueError(f"Regional load component {name!r} is missing columns: {sorted(missing)}")
        joined_parts.append(
            forecast[list(LOAD_FORECAST_REQUIRED_COLUMNS)].rename(
                columns={
                    "Load_Model_MW": f"Load_Model_MW_{name}",
                    "Load_Benchmark_MW": f"Load_Benchmark_MW_{name}",
                    "Load_Actual_MW": f"Load_Actual_MW_{name}",
                }
            )
        )

    joined = pd.concat(joined_parts, axis=1, join="inner").sort_index()
    if joined.empty:
        raise ValueError("Regional load component forecasts have no common timestamps.")

    forecast = pd.DataFrame(index=joined.index)
    model_columns = [f"Load_Model_MW_{name}" for name in component_forecasts]
    benchmark_columns = [f"Load_Benchmark_MW_{name}" for name in component_forecasts]
    actual_columns = [f"Load_Actual_MW_{name}" for name in component_forecasts]
    forecast["Load_Model_MW"] = joined[model_columns].sum(axis=1)
    forecast["Load_Benchmark_MW"] = joined[benchmark_columns].sum(axis=1)
    forecast["Load_Actual_MW"] = joined[actual_columns].sum(axis=1)
    for column in [*model_columns, *benchmark_columns, *actual_columns]:
        forecast[column] = joined[column]
    forecast.index.name = "timestamp"
    return forecast.sort_index()


def _component_cache_name(component: RegionalLoadForecastComponentConfig) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in component.name).strip("_") or "component"


def _regional_component_load_config(
    config: RegionalLoadForecastConfig,
    component: RegionalLoadForecastComponentConfig,
) -> LoadForecastModelConfig:
    component_name = _component_cache_name(component)
    data = config.base_config.model_dump()
    data.update(
        {
            "repo_root": config.repo_root,
            "target_tz": config.target_tz or config.base_config.target_tz,
            "country_code_entsoe": component.country_code_entsoe,
            "actual_load_file": component.actual_load_file
            or config.component_cache_dir / f"{component_name}_actual_load.csv",
            "entsoe_load_forecast_file": component.entsoe_load_forecast_file
            or config.component_cache_dir / f"{component_name}_entsoe_load_forecast.csv",
            "export_dir": component.export_dir or config.export_dir / "components" / component_name,
            "include_rolling_residual_quantiles": False,
        }
    )
    if component.weather_cluster_weight_file is not None:
        data["weather_cluster_weight_file"] = component.weather_cluster_weight_file
    if component.regional_holiday_weight_file is not None:
        data["regional_holiday_weight_file"] = component.regional_holiday_weight_file
    return LoadForecastModelConfig.model_validate(data)


def _load_saved_regional_component_result(
    component_config: LoadForecastModelConfig,
) -> dict[str, pd.DataFrame] | None:
    forecast_path = component_config.export_dir / "forecast.csv"
    if not forecast_path.exists():
        return None

    forecast = _load_model_forecast_csv(forecast_path, component_config.target_tz)
    runtime_path = component_config.export_dir / "runtime.csv"
    metrics_path = component_config.export_dir / "metrics.csv"
    runtime = pd.read_csv(runtime_path) if runtime_path.exists() else pd.DataFrame()
    metrics = pd.read_csv(metrics_path) if metrics_path.exists() else evaluate_load_forecast(forecast)
    return {
        "forecast": forecast,
        "runtime": runtime,
        "metrics": metrics,
    }


def _aggregate_load_reference_frame(config: RegionalLoadForecastConfig) -> pd.DataFrame:
    data = config.base_config.model_dump()
    data.update(
        {
            "repo_root": config.repo_root,
            "target_tz": config.target_tz or config.base_config.target_tz,
            "country_code_entsoe": config.aggregate_country_code_entsoe,
            "actual_load_file": config.aggregate_actual_load_file or config.base_config.actual_load_file,
            "entsoe_load_forecast_file": (
                config.aggregate_entsoe_load_forecast_file or config.base_config.entsoe_load_forecast_file
            ),
            "export_dir": config.export_dir / "_aggregate_reference_unused",
        }
    )
    aggregate_config = LoadForecastModelConfig.model_validate(data)
    actual = _load_or_fetch_actual_load(aggregate_config)
    forecast = _load_or_fetch_model_entsoe_load_forecast(aggregate_config)
    frame = forecast[["load_fc"]].rename(columns={"load_fc": "Load_Benchmark_MW"}).join(
        actual[["load_actual"]].rename(columns={"load_actual": "Load_Actual_MW"}),
        how="inner",
    )
    start = _as_local_day(aggregate_config.test_start, aggregate_config.target_tz)
    end = _as_local_day(aggregate_config.test_end, aggregate_config.target_tz) + pd.Timedelta(days=1) - pd.Timedelta(minutes=15)
    frame = frame.loc[(frame.index >= start) & (frame.index <= end)].sort_index()
    frame.index.name = "timestamp"
    return frame


def add_aggregate_residual_load_component(
    component_forecasts: dict[str, pd.DataFrame],
    aggregate_reference: pd.DataFrame,
    *,
    component_name: str,
) -> dict[str, pd.DataFrame]:
    """Add the aggregate-minus-components residual with ENTSO-E forecast as its model value."""
    if component_name in component_forecasts:
        raise ValueError(f"Regional load component name {component_name!r} already exists.")
    component_sum = sum_regional_load_component_forecasts(component_forecasts)
    joined = aggregate_reference.join(
        component_sum[["Load_Benchmark_MW", "Load_Actual_MW"]].rename(
            columns={
                "Load_Benchmark_MW": "Load_Benchmark_MW_components",
                "Load_Actual_MW": "Load_Actual_MW_components",
            }
        ),
        how="inner",
    )
    if joined.empty:
        raise ValueError("Aggregate reference and regional components have no common timestamps.")

    residual = pd.DataFrame(index=joined.index)
    residual["Load_Benchmark_MW"] = joined["Load_Benchmark_MW"] - joined["Load_Benchmark_MW_components"]
    residual["Load_Actual_MW"] = joined["Load_Actual_MW"] - joined["Load_Actual_MW_components"]
    residual["Load_Model_MW"] = residual["Load_Benchmark_MW"]
    residual.index.name = "timestamp"

    return {**component_forecasts, component_name: residual.sort_index()}


def build_load_forecast_ensemble(config: LoadForecastEnsembleConfig) -> pd.DataFrame:
    source_forecasts = {
        source.name: _load_model_forecast_csv(source.path, config.target_tz)
        for source in config.sources
    }
    weights = {source.name: source.weight for source in config.sources}
    forecast = build_load_point_ensemble(source_forecasts, method=config.method, weights=weights)
    start = _as_local_day(config.test_start, config.target_tz)
    end = _as_local_day(config.test_end, config.target_tz) + pd.Timedelta(days=1) - pd.Timedelta(minutes=15)
    return forecast.loc[(forecast.index >= start) & (forecast.index <= end)]


def build_entsoe_load_forecast_benchmark(
    config: EntsoeLoadForecastBenchmarkConfig,
) -> pd.DataFrame:
    """Build a model-style forecast frame from ENTSO-E total load forecast data."""
    actual = _load_or_fetch_actual_load(config)
    forecast = _load_or_fetch_entsoe_load_forecast(config)

    frame = forecast[["load_fc"]].rename(columns={"load_fc": "Load_Benchmark_MW"}).join(
        actual[["load_actual"]].rename(columns={"load_actual": "Load_Actual_MW"}),
        how="inner",
    )
    frame = frame.sort_index()
    if config.test_start is not None or config.test_end is not None:
        start = _as_local_day(config.test_start or frame.index.min(), config.target_tz)
        end = _as_local_day(config.test_end or frame.index.max(), config.target_tz) + pd.Timedelta(days=1) - pd.Timedelta(minutes=15)
        frame = frame.loc[(frame.index >= start) & (frame.index <= end)]
    frame.index.name = "timestamp"
    return frame


def run_entsoe_load_forecast_benchmark(
    config: EntsoeLoadForecastBenchmarkConfig,
    *,
    save_outputs: bool = True,
) -> dict[str, pd.DataFrame]:
    load_dotenv(config.repo_root / ".env")
    print("\n--- ENTSO-E Load Forecast Benchmark ---")
    forecast = build_entsoe_load_forecast_benchmark(config)
    metrics = evaluate_load_forecast(forecast, prediction_columns=["Load_Benchmark_MW"])
    print(metrics.to_string(index=False))

    if save_outputs:
        config.export_dir.mkdir(parents=True, exist_ok=True)
        forecast.to_csv(config.export_dir / "forecast.csv")
        metrics.to_csv(config.export_dir / "metrics.csv", index=False)
        with (config.export_dir / "config.json").open("w", encoding="utf-8") as handle:
            json.dump(config.model_dump(mode="json"), handle, indent=2)
        print(f"Saved ENTSO-E load forecast benchmark outputs -> {config.export_dir}")

    return {
        "forecast": forecast,
        "metrics": metrics,
    }


def run_load_forecast_ensemble_pipeline(
    config: LoadForecastEnsembleConfig,
    *,
    save_outputs: bool = True,
) -> dict[str, pd.DataFrame]:
    print("\n--- Building Load Forecast Ensemble ---")
    forecast = build_load_forecast_ensemble(config)
    forecast = apply_rolling_residual_quantiles(forecast, config)
    metrics = evaluate_load_forecast(forecast)
    quantile_metrics = (
        evaluate_load_quantile_forecast(
            _quantile_evaluation_frame(forecast, config),
            config.residual_quantiles,
        )
        if config.include_rolling_residual_quantiles
        else pd.DataFrame()
    )

    print("\n--- Load Forecast Ensemble Metrics ---")
    print(metrics.to_string(index=False))
    if not quantile_metrics.empty:
        print("\n--- Load Ensemble Quantile Forecast Metrics ---")
        print(quantile_metrics.to_string(index=False))

    if save_outputs:
        config.export_dir.mkdir(parents=True, exist_ok=True)
        forecast.to_csv(config.export_dir / "forecast.csv")
        metrics.to_csv(config.export_dir / "metrics.csv", index=False)
        if not quantile_metrics.empty:
            quantile_metrics.to_csv(config.export_dir / "quantile_metrics.csv", index=False)
        with (config.export_dir / "config.json").open("w", encoding="utf-8") as handle:
            json.dump(config.model_dump(mode="json"), handle, indent=2)
        print(f"Saved load forecast ensemble outputs -> {config.export_dir}")

    return {
        "forecast": forecast,
        "metrics": metrics,
        "quantile_metrics": quantile_metrics,
    }


def run_regional_load_forecast_pipeline(
    config: RegionalLoadForecastConfig,
    *,
    save_outputs: bool = True,
) -> dict[str, pd.DataFrame]:
    """Run regional load component models and sum them into a DE-LU forecast."""
    load_dotenv(config.repo_root / ".env")
    print("\n--- Regional Load Forecast Components ---")

    component_forecasts: dict[str, pd.DataFrame] = {}
    component_runtime: list[pd.DataFrame] = []
    component_metrics: list[pd.DataFrame] = []
    for component in config.components:
        print(f"\n=== Regional load component: {component.name} ({component.country_code_entsoe}) ===")
        component_config = _regional_component_load_config(config, component)
        result = (
            _load_saved_regional_component_result(component_config)
            if config.reuse_component_forecasts
            else None
        )
        if result is None:
            result = run_load_forecast_pipeline(component_config, save_outputs=save_outputs)
        else:
            print(f"Reusing saved component forecast -> {component_config.export_dir / 'forecast.csv'}")
        component_forecasts[component.name] = result["forecast"]
        runtime = result["runtime"].copy()
        runtime.insert(0, "component", component.name)
        runtime.insert(1, "country_code_entsoe", component.country_code_entsoe)
        component_runtime.append(runtime)
        metrics = result["metrics"].copy()
        metrics.insert(0, "component", component.name)
        metrics.insert(1, "country_code_entsoe", component.country_code_entsoe)
        component_metrics.append(metrics)

    if config.include_aggregate_residual_component:
        print(
            "\n=== Regional load component: "
            f"{config.aggregate_component_name} ({config.aggregate_country_code_entsoe} minus components) ==="
        )
        component_forecasts = add_aggregate_residual_load_component(
            component_forecasts,
            _aggregate_load_reference_frame(config),
            component_name=config.aggregate_component_name,
        )
        residual_metrics = evaluate_load_forecast(
            component_forecasts[config.aggregate_component_name],
            prediction_columns=["Load_Model_MW", "Load_Benchmark_MW"],
        )
        residual_metrics.insert(0, "component", config.aggregate_component_name)
        residual_metrics.insert(1, "country_code_entsoe", config.aggregate_country_code_entsoe)
        component_metrics.append(residual_metrics)

    forecast = sum_regional_load_component_forecasts(component_forecasts)
    metrics = evaluate_load_forecast(forecast, prediction_columns=["Load_Model_MW", "Load_Benchmark_MW"])
    runtime = pd.concat(component_runtime, ignore_index=True) if component_runtime else pd.DataFrame()
    regional_metrics = pd.concat(component_metrics, ignore_index=True) if component_metrics else pd.DataFrame()

    print("\n--- Regional Summed Load Forecast Metrics ---")
    print(metrics.to_string(index=False))

    if save_outputs:
        config.export_dir.mkdir(parents=True, exist_ok=True)
        forecast.to_csv(config.export_dir / "forecast.csv")
        metrics.to_csv(config.export_dir / "metrics.csv", index=False)
        runtime.to_csv(config.export_dir / "runtime.csv", index=False)
        regional_metrics.to_csv(config.export_dir / "component_metrics.csv", index=False)
        with (config.export_dir / "config.json").open("w", encoding="utf-8") as handle:
            json.dump(config.model_dump(mode="json"), handle, indent=2)
        print(f"Saved regional load forecast outputs -> {config.export_dir}")

    return {
        "forecast": forecast,
        "metrics": metrics,
        "runtime": runtime,
        "component_metrics": regional_metrics,
    }


def run_load_forecast_pipeline(
    config: LoadForecastModelConfig,
    *,
    save_outputs: bool = True,
) -> dict[str, pd.DataFrame]:
    load_dotenv(config.repo_root / ".env")
    if config.fetch_weather_only:
        print("\n--- Fetching Load Weather Data ---")
        weather = _build_load_weather_features(config)
        print(f"Weather: {weather.shape[0]:,} rows, {weather.shape[1]:,} columns")
        return {"weather": weather}

    print("\n--- Building Load Forecast Dataset ---")
    dataset = build_load_forecast_dataset(config)
    print(f"Dataset: {dataset.shape[0]:,} rows, {dataset.shape[1]:,} columns")

    print("\n--- Rolling Direct Load Forecast ---")
    forecast, runtime = rolling_load_forecast(dataset, config)
    forecast = apply_rolling_residual_quantiles(forecast, config)
    metrics = evaluate_load_forecast(forecast)
    quantile_metrics = (
        evaluate_load_quantile_forecast(
            _quantile_evaluation_frame(forecast, config),
            config.residual_quantiles,
        )
        if config.include_rolling_residual_quantiles
        else pd.DataFrame()
    )
    print("\n--- Load Forecast Metrics ---")
    print(metrics.to_string(index=False))
    if not quantile_metrics.empty:
        print("\n--- Load Quantile Forecast Metrics ---")
        print(quantile_metrics.to_string(index=False))

    if save_outputs:
        config.export_dir.mkdir(parents=True, exist_ok=True)
        forecast.to_csv(config.export_dir / "forecast.csv")
        runtime.to_csv(config.export_dir / "runtime.csv", index=False)
        metrics.to_csv(config.export_dir / "metrics.csv", index=False)
        if not quantile_metrics.empty:
            quantile_metrics.to_csv(config.export_dir / "quantile_metrics.csv", index=False)
        with (config.export_dir / "config.json").open("w", encoding="utf-8") as handle:
            json.dump(config.model_dump(mode="json"), handle, indent=2)
        print(f"Saved load forecast outputs -> {config.export_dir}")

    return {
        "dataset": dataset,
        "forecast": forecast,
        "runtime": runtime,
        "metrics": metrics,
        "quantile_metrics": quantile_metrics,
    }
