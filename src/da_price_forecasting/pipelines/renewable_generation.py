from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..config import (
    EntsoeRenewableForecastBenchmarkConfig,
    RenewableGenerationModelConfig,
    RenewableGenerationPostprocessConfig,
)
from ..data.entsoe import (
    fetch_actual_renewable_generation,
    fetch_actual_solar_generation_by_control_area,
    fetch_generation_unavailability,
    fetch_renewable_generation_forecast,
)
from ..data.weather import load_dwd
from ..evaluation.metrics import pinball_score, quantile_column
from .common import (
    as_local_day as _as_local_day,
    load_timestamp_csv as _load_timestamp_csv,
    point_error_stats,
    save_timestamp_csv as _save_timestamp_csv,
)


def _target_availability_cutoff(
    day: pd.Timestamp,
    *,
    lag_days: int,
    cutoff_hour: int | None,
    cutoff_minute: int,
) -> pd.Timestamp:
    """Latest timestamp whose actual generation is operationally available for ``day``.

    Mirrors the day-ahead information set: when forecasting delivery day ``day`` on the
    previous morning (before the 12:00 gate), actuals are only known up to
    ``day - lag_days`` at the configured morning cutoff, or the end of that day when no
    cutoff hour is set.
    """
    cutoff_day = day - pd.Timedelta(days=lag_days)
    if cutoff_hour is None:
        return cutoff_day - pd.Timedelta(minutes=15)
    return cutoff_day + pd.Timedelta(hours=cutoff_hour, minutes=cutoff_minute)


def _training_target_cutoff(day: pd.Timestamp, config: RenewableGenerationModelConfig) -> pd.Timestamp:
    return _target_availability_cutoff(
        day,
        lag_days=config.target_availability_lag_days,
        cutoff_hour=config.target_availability_cutoff_hour,
        cutoff_minute=config.target_availability_cutoff_minute,
    )


def _load_renewable_proxy(config: RenewableGenerationModelConfig) -> pd.DataFrame:
    if not config.renewable_proxy_file.exists():
        raise FileNotFoundError(
            f"Renewable proxy file not found: {config.renewable_proxy_file}. "
            "Run the regional renewable feature preprocessing step first."
        )

    proxy = _load_timestamp_csv(config.renewable_proxy_file, config.target_tz)
    if config.renewable_proxy_fallback_file is not None:
        if not config.renewable_proxy_fallback_file.exists():
            raise FileNotFoundError(
                f"Renewable proxy fallback file not found: {config.renewable_proxy_fallback_file}. "
                "Run the fallback regional renewable feature preprocessing step first."
            )

        fallback = _load_timestamp_csv(config.renewable_proxy_fallback_file, config.target_tz)
        if config.renewable_proxy_fallback_end_date is not None:
            fallback_end = _as_local_day(config.renewable_proxy_fallback_end_date, config.target_tz)
            fallback = fallback.loc[fallback.index < fallback_end + pd.Timedelta(days=1)]
        proxy = proxy.combine_first(fallback).sort_index()

    extra_proxies = []
    for extra_file in config.extra_renewable_proxy_files:
        if not extra_file.exists():
            raise FileNotFoundError(
                f"Extra renewable proxy file not found: {extra_file}. "
                "Run the corresponding regional renewable feature preprocessing step first."
            )
        extra = _load_timestamp_csv(extra_file, config.target_tz)
        extra_proxies.append(extra.reindex(proxy.index))

    if config.extra_renewable_proxy_ensemble_mode == "summary" and extra_proxies:
        proxy = proxy.join(_summarise_extra_renewable_proxies(extra_proxies, config), how="left")
    else:
        prefixes = config.extra_renewable_proxy_prefixes or [
            f"extra_{idx + 1}_" for idx in range(len(config.extra_renewable_proxy_files))
        ]
        for extra, prefix in zip(extra_proxies, prefixes, strict=True):
            if prefix:
                extra = extra.add_prefix(prefix)
            proxy = proxy.join(extra, how="left")

    raw_prefixes = config.raw_extra_renewable_proxy_prefixes or [
        f"raw_extra_{idx + 1}_" for idx in range(len(config.raw_extra_renewable_proxy_files))
    ]
    for raw_file, prefix in zip(config.raw_extra_renewable_proxy_files, raw_prefixes, strict=True):
        if not raw_file.exists():
            raise FileNotFoundError(
                f"Raw extra renewable proxy file not found: {raw_file}. "
                "Run the corresponding regional renewable feature preprocessing step first."
            )
        raw_extra = _load_timestamp_csv(raw_file, config.target_tz).reindex(proxy.index)
        if prefix:
            raw_extra = raw_extra.add_prefix(prefix)
        proxy = proxy.join(raw_extra, how="left")

    proxy = proxy.loc[:, ~proxy.columns.duplicated()]
    proxy.index.name = "timestamp"
    return proxy


def _summarise_extra_renewable_proxies(
    extra_proxies: list[pd.DataFrame],
    config: RenewableGenerationModelConfig,
) -> pd.DataFrame:
    """Compress provider feature tables into provider mean/spread summaries.

    The Open-Meteo provider files are constructed with matching feature names. In
    summary mode, instead of joining every provider column, we calculate the
    configured statistics across providers for each common feature. This keeps
    the model architecture unchanged while reducing the weather-ensemble feature
    dimensionality and exposing provider disagreement explicitly.
    """
    common_columns = set(extra_proxies[0].columns)
    for extra in extra_proxies[1:]:
        common_columns &= set(extra.columns)

    patterns = config.extra_renewable_proxy_ensemble_column_patterns
    columns = sorted(
        column for column in common_columns
        if not patterns or any(pattern in column for pattern in patterns)
    )
    if not columns:
        raise ValueError("No common extra renewable proxy columns available for ensemble summary mode.")

    if config.extra_renewable_proxy_ensemble_groups:
        summaries = []
        group_means = {}
        for group_name, indices in config.extra_renewable_proxy_ensemble_groups.items():
            group_proxies = [extra_proxies[index] for index in indices]
            arrays = _provider_summary_arrays(group_proxies, columns)
            group_means[group_name] = arrays["mean"]
            summaries.append(
                _provider_summary_frame(
                    arrays,
                    columns=columns,
                    index=extra_proxies[0].index,
                    stats=config.extra_renewable_proxy_ensemble_stats,
                    prefix=f"{config.extra_renewable_proxy_ensemble_prefix}{group_name}_",
                )
            )

        for left, right in config.extra_renewable_proxy_ensemble_group_diff_pairs:
            diff = group_means[left] - group_means[right]
            data = {
                f"{config.extra_renewable_proxy_ensemble_prefix}{left}_minus_{right}_mean_{column}": diff[:, idx]
                for idx, column in enumerate(columns)
            }
            summaries.append(pd.DataFrame(data, index=extra_proxies[0].index))

        result = pd.concat(summaries, axis=1).sort_index()
        result.index.name = "timestamp"
        return result

    arrays = _provider_summary_arrays(extra_proxies, columns)
    return _provider_summary_frame(
        arrays,
        columns=columns,
        index=extra_proxies[0].index,
        stats=config.extra_renewable_proxy_ensemble_stats,
        prefix=config.extra_renewable_proxy_ensemble_prefix,
    )


def _provider_summary_arrays(
    provider_frames: list[pd.DataFrame],
    columns: list[str],
) -> dict[str, np.ndarray]:
    stack = np.stack(
        [extra.loc[:, columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float) for extra in provider_frames],
        axis=0,
    )
    finite = np.isfinite(stack)
    counts = finite.sum(axis=0)
    sums = np.where(finite, stack, 0.0).sum(axis=0)
    means = np.divide(sums, counts, out=np.full_like(sums, np.nan, dtype=float), where=counts > 0)
    centered = np.where(finite, stack - means, 0.0)
    variances = np.divide(
        (centered * centered).sum(axis=0),
        counts,
        out=np.full_like(sums, np.nan, dtype=float),
        where=counts > 0,
    )
    stds = np.sqrt(variances)
    mins = np.min(np.where(finite, stack, np.inf), axis=0)
    mins = np.where(counts > 0, mins, np.nan)
    maxs = np.max(np.where(finite, stack, -np.inf), axis=0)
    maxs = np.where(counts > 0, maxs, np.nan)

    return {
        "mean": means,
        "std": stds,
        "min": mins,
        "max": maxs,
        "range": maxs - mins,
    }


def _provider_summary_frame(
    arrays: dict[str, np.ndarray],
    *,
    columns: list[str],
    index: pd.DatetimeIndex,
    stats: list[str],
    prefix: str,
) -> pd.DataFrame:
    summary = {}
    for stat in stats:
        values = arrays[stat]
        summary.update({f"{prefix}{stat}_{column}": values[:, idx] for idx, column in enumerate(columns)})

    result = pd.DataFrame(summary, index=index)
    result.index.name = "timestamp"
    return result


def _load_or_fetch_actual_generation(config: RenewableGenerationModelConfig) -> pd.DataFrame:
    if config.actual_generation_file.exists():
        actual = _load_timestamp_csv(config.actual_generation_file, config.target_tz)
        required_columns = set(config.target_columns)
        if config.actual_generation_lag_days:
            required_columns.update(config.actual_generation_lag_columns)
        if config.include_partial_actual_generation_features:
            partial_columns = config.partial_generation_columns or config.actual_generation_lag_columns
            required_columns.update(partial_columns)
        if config.include_partial_proxy_error_features:
            partial_proxy_columns = config.partial_proxy_error_columns or list(
                config.partial_proxy_error_proxy_column_patterns
            )
            required_columns.update(partial_proxy_columns or config.target_columns)
        missing_columns = [column for column in required_columns if column not in actual.columns]
        if not (config.include_solar_control_area_targets and missing_columns):
            return actual

    start = _as_local_day(config.entsoe_start_date, config.target_tz)
    end = _as_local_day(config.entsoe_end_date, config.target_tz)
    df = fetch_actual_renewable_generation(
        start_day=start,
        end_day=end,
        country_code=config.country_code_entsoe,
        api_key_env=config.entsoe_api_key_env,
        target_tz=config.target_tz,
    )
    if config.include_solar_control_area_targets:
        control_area_actual = fetch_actual_solar_generation_by_control_area(
            start_day=start,
            end_day=end,
            control_area_targets=config.solar_control_area_targets,
            api_key_env=config.entsoe_api_key_env,
            target_tz=config.target_tz,
        )
        df = df.join(control_area_actual, how="outer")
    _save_timestamp_csv(df, config.actual_generation_file)
    return df


def _load_or_fetch_benchmark_actual_generation(config: EntsoeRenewableForecastBenchmarkConfig) -> pd.DataFrame:
    if config.actual_generation_file.exists():
        return _load_timestamp_csv(config.actual_generation_file, config.target_tz)

    start = _as_local_day(config.entsoe_start_date, config.target_tz)
    end = _as_local_day(config.entsoe_end_date, config.target_tz)
    df = fetch_actual_renewable_generation(
        start_day=start,
        end_day=end,
        country_code=config.country_code_entsoe,
        api_key_env=config.entsoe_api_key_env,
        target_tz=config.target_tz,
        chunk_days=config.chunk_days,
    )
    _save_timestamp_csv(df, config.actual_generation_file)
    return df


def _load_or_fetch_entsoe_renewable_forecast(config: EntsoeRenewableForecastBenchmarkConfig) -> pd.DataFrame:
    if config.forecast_file.exists():
        return _load_timestamp_csv(config.forecast_file, config.target_tz)

    start = _as_local_day(config.entsoe_start_date, config.target_tz)
    end = _as_local_day(config.entsoe_end_date, config.target_tz)
    df = fetch_renewable_generation_forecast(
        start_day=start,
        end_day=end,
        country_code=config.country_code_entsoe,
        api_key_env=config.entsoe_api_key_env,
        target_tz=config.target_tz,
        chunk_days=config.chunk_days,
        process_type=config.process_type,
    )
    _save_timestamp_csv(df, config.forecast_file)
    return df


def _load_or_fetch_unavailability(config: RenewableGenerationModelConfig) -> pd.DataFrame:
    if config.unavailability_file.exists():
        return _load_timestamp_csv(config.unavailability_file, config.target_tz)

    start = _as_local_day(config.entsoe_start_date, config.target_tz)
    end = _as_local_day(config.entsoe_end_date, config.target_tz)
    df = fetch_generation_unavailability(
        start_day=start,
        end_day=end,
        country_code=config.country_code_entsoe,
        api_key_env=config.entsoe_api_key_env,
        target_tz=config.target_tz,
        feature_mode=config.unavailability_feature_mode,
        planned_only=config.unavailability_planned_only,
    )
    _save_timestamp_csv(df, config.unavailability_file)
    return df


def _build_time_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    minute_of_day = index.hour * 60 + index.minute
    day_of_week = index.dayofweek
    day_of_year = index.dayofyear
    mtu = index.hour * 4 + index.minute // 15

    features = pd.DataFrame(index=index)
    features["mtu"] = mtu
    features["tod_sin"] = np.sin(2 * np.pi * minute_of_day / (24 * 60))
    features["tod_cos"] = np.cos(2 * np.pi * minute_of_day / (24 * 60))
    features["dow_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    features["dow_cos"] = np.cos(2 * np.pi * day_of_week / 7)
    features["doy_sin"] = np.sin(2 * np.pi * day_of_year / 366)
    features["doy_cos"] = np.cos(2 * np.pi * day_of_year / 366)
    features["is_weekend"] = (day_of_week >= 5).astype(int)
    return features


def _region_proxy_columns(df: pd.DataFrame, prefix: str) -> list[str]:
    return [
        column
        for column in df.columns
        if column.startswith(prefix) and column.endswith("_proxy_mw") and "_cluster_" not in column
    ]


def _safe_column(df: pd.DataFrame, column: str) -> pd.Series | None:
    return df[column] if column in df.columns else None


def _series_or_zero(index: pd.DatetimeIndex, series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(0.0, index=index)
    return series.astype(float)


def _add_summary_stats(
    source: pd.DataFrame,
    output: dict[str, pd.Series],
    *,
    name: str,
    columns: list[str],
) -> None:
    if not columns:
        return
    values = source[columns].astype(float)
    output[f"{name}_sum"] = values.sum(axis=1)
    output[f"{name}_max"] = values.max(axis=1)
    output[f"{name}_min"] = values.min(axis=1)
    output[f"{name}_spread"] = values.max(axis=1) - values.min(axis=1)
    output[f"{name}_std"] = values.std(axis=1).fillna(0.0)
    output[f"{name}_top2_sum"] = values.apply(lambda row: row.nlargest(min(2, len(row))).sum(), axis=1)


def _build_regional_summary_features(proxy: pd.DataFrame) -> pd.DataFrame:
    """Derive regional spread/gradient features from existing renewable proxy columns."""
    output: dict[str, pd.Series] = {}
    index = proxy.index

    wind_cols = _region_proxy_columns(proxy, "wind_")
    onshore_cols = _region_proxy_columns(proxy, "wind_onshore_")
    offshore_cols = _region_proxy_columns(proxy, "wind_offshore_")
    solar_cols = _region_proxy_columns(proxy, "solar_")

    _add_summary_stats(proxy, output, name="wind_region_proxy", columns=wind_cols)
    _add_summary_stats(proxy, output, name="wind_onshore_region_proxy", columns=onshore_cols)
    _add_summary_stats(proxy, output, name="wind_offshore_region_proxy", columns=offshore_cols)
    _add_summary_stats(proxy, output, name="solar_region_proxy", columns=solar_cols)

    output["wind_offshore_minus_onshore_proxy"] = (
        _series_or_zero(index, proxy[offshore_cols].sum(axis=1) if offshore_cols else None)
        - _series_or_zero(index, proxy[onshore_cols].sum(axis=1) if onshore_cols else None)
    )
    for prefix, name in [("wind_onshore", "wind_onshore"), ("solar", "solar")]:
        north = _safe_column(proxy, f"{prefix}_north_proxy_mw")
        south = _safe_column(proxy, f"{prefix}_south_proxy_mw")
        east = _safe_column(proxy, f"{prefix}_east_proxy_mw")
        west = _safe_column(proxy, f"{prefix}_west_proxy_mw")
        central = _safe_column(proxy, f"{prefix}_central_proxy_mw")
        output[f"{name}_north_minus_south_proxy"] = _series_or_zero(index, north) - _series_or_zero(index, south)
        output[f"{name}_east_minus_west_proxy"] = _series_or_zero(index, east) - _series_or_zero(index, west)
        output[f"{name}_north_minus_central_proxy"] = _series_or_zero(index, north) - _series_or_zero(index, central)

    speed_cols = [column for column in proxy.columns if column.startswith("wind_") and "speed_hub" in column]
    _add_summary_stats(proxy, output, name="wind_region_speed_hub", columns=speed_cols)

    irradiance_cols = [
        column
        for column in proxy.columns
        if column.startswith("solar_") and column.endswith("irradiance_cap_weighted_W_m2")
    ]
    _add_summary_stats(proxy, output, name="solar_region_irradiance", columns=irradiance_cols)

    summary = pd.DataFrame(output, index=index)
    ramp_cols = [column for column in summary.columns if column.endswith("_proxy") or column.endswith("_sum")]
    for step, suffix in [(4, "ramp_1h"), (12, "ramp_3h")]:
        for column in ramp_cols:
            summary[f"{column}_{suffix}"] = summary[column].diff(step)

    summary.index.name = "timestamp"
    return summary


def _build_wind_direction_regime_features(proxy: pd.DataFrame) -> pd.DataFrame:
    """Derive compact wind-direction regime features from regional vector means."""
    output: dict[str, pd.Series] = {}
    weighted_components: dict[str, dict[str, list[pd.Series]]] = {}

    cos_suffix = "_direction_cos_cap_weighted"
    sin_suffix = "_direction_sin_cap_weighted"
    cos_columns = [
        column
        for column in proxy.select_dtypes(include="number").columns
        if column.endswith(cos_suffix)
    ]
    for cos_column in cos_columns:
        base = cos_column[: -len(cos_suffix)]
        sin_column = f"{base}{sin_suffix}"
        if sin_column not in proxy.columns:
            continue

        cos_values = proxy[cos_column].astype(float).clip(-1.0, 1.0)
        sin_values = proxy[sin_column].astype(float).clip(-1.0, 1.0)
        resultant = np.sqrt(cos_values**2 + sin_values**2).clip(0.0, 1.0)
        output[f"{base}_direction_resultant"] = resultant

        proxy_column = f"{base}_proxy_mw"
        if proxy_column in proxy.columns:
            proxy_values = proxy[proxy_column].astype(float).clip(lower=0.0)
            output[f"{base}_proxy_x_direction_cos"] = proxy_values * cos_values
            output[f"{base}_proxy_x_direction_sin"] = proxy_values * sin_values
            output[f"{base}_proxy_x_direction_resultant"] = proxy_values * resultant

            source = "open_meteo" if base.startswith("om_") else "dwd"
            weighted_components.setdefault(source, {"proxy": [], "cos": [], "sin": []})
            weighted_components[source]["proxy"].append(proxy_values)
            weighted_components[source]["cos"].append(proxy_values * cos_values)
            weighted_components[source]["sin"].append(proxy_values * sin_values)

        speed_column = f"{base}_speed_hub_cap_weighted_m_s"
        if speed_column in proxy.columns:
            speed_values = proxy[speed_column].astype(float).clip(lower=0.0)
            output[f"{base}_speed_x_direction_cos"] = speed_values * cos_values
            output[f"{base}_speed_x_direction_sin"] = speed_values * sin_values
            output[f"{base}_speed_x_direction_resultant"] = speed_values * resultant

    for source, components in weighted_components.items():
        proxy_sum = pd.concat(components["proxy"], axis=1).sum(axis=1)
        cos_sum = pd.concat(components["cos"], axis=1).sum(axis=1)
        sin_sum = pd.concat(components["sin"], axis=1).sum(axis=1)
        denominator = proxy_sum.replace(0.0, np.nan)
        weighted_cos = (cos_sum / denominator).fillna(0.0)
        weighted_sin = (sin_sum / denominator).fillna(0.0)
        output[f"wind_direction_{source}_proxy_weighted_cos"] = weighted_cos
        output[f"wind_direction_{source}_proxy_weighted_sin"] = weighted_sin
        output[f"wind_direction_{source}_proxy_weighted_resultant"] = (
            np.sqrt(weighted_cos**2 + weighted_sin**2).clip(0.0, 1.0)
        )

    features = pd.DataFrame(output, index=proxy.index)
    features.index.name = "timestamp"
    return features


def _build_forecast_lead_features(
    index: pd.DatetimeIndex,
    config: RenewableGenerationModelConfig,
) -> pd.DataFrame:
    """Build lead-time features for operational single-run weather forecasts."""
    run_hour_text = (
        config.forecast_run_hour_utc + ":00"
        if config.forecast_run_hour_utc.count(":") == 1
        else config.forecast_run_hour_utc
    )
    run_hour = pd.to_timedelta(run_hour_text)
    delivery_days = index.tz_localize(None).normalize()
    run_times_local = (
        delivery_days.tz_localize("UTC")
        - pd.Timedelta(days=config.forecast_run_day_offset)
        + run_hour
    ).tz_convert(config.target_tz)
    lead_hours = (index - run_times_local) / pd.Timedelta(hours=1)

    features = pd.DataFrame(index=index)
    features["forecast_lead_hours"] = lead_hours.astype(float)
    features["forecast_lead_days"] = features["forecast_lead_hours"] / 24.0
    features["forecast_lead_sin"] = np.sin(2 * np.pi * features["forecast_lead_hours"] / 24.0)
    features["forecast_lead_cos"] = np.cos(2 * np.pi * features["forecast_lead_hours"] / 24.0)
    return features


_SOLAR_GEOMETRY_POINTS = {
    "central": (51.0, 10.0),
    "east": (51.5, 13.0),
    "north": (53.5, 9.5),
    "south": (48.5, 11.5),
    "west": (51.0, 7.0),
}


def _solar_position_features(index: pd.DatetimeIndex, *, latitude: float, longitude: float) -> pd.DataFrame:
    """Approximate solar position features from NOAA-style equations."""
    output_index = index
    utc = pd.to_datetime(pd.Index(index), utc=True)
    minutes_utc = (
        utc.hour.to_numpy(dtype=float) * 60.0
        + utc.minute.to_numpy(dtype=float)
        + utc.second.to_numpy(dtype=float) / 60.0
    )
    hour_utc = minutes_utc / 60.0
    day_of_year = utc.dayofyear.to_numpy(dtype=float)

    gamma = 2.0 * np.pi / 365.0 * (day_of_year - 1.0 + (hour_utc - 12.0) / 24.0)
    equation_of_time = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2.0 * gamma)
        - 0.040849 * np.sin(2.0 * gamma)
    )
    declination = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2.0 * gamma)
        + 0.000907 * np.sin(2.0 * gamma)
        - 0.002697 * np.cos(3.0 * gamma)
        + 0.00148 * np.sin(3.0 * gamma)
    )

    true_solar_time = (minutes_utc + equation_of_time + 4.0 * longitude) % 1440.0
    hour_angle = np.deg2rad(true_solar_time / 4.0 - 180.0)
    lat_rad = np.deg2rad(latitude)

    cos_zenith = (
        np.sin(lat_rad) * np.sin(declination)
        + np.cos(lat_rad) * np.cos(declination) * np.cos(hour_angle)
    )
    cos_zenith = np.clip(cos_zenith, -1.0, 1.0)
    positive_cos_zenith = np.clip(cos_zenith, 0.0, None)
    elevation_deg = np.rad2deg(np.arcsin(cos_zenith))

    return pd.DataFrame(
        {
            "cos_zenith": cos_zenith,
            "positive_cos_zenith": positive_cos_zenith,
            "elevation_deg": elevation_deg,
            "clear_sky_proxy": positive_cos_zenith**1.25,
            "is_daylight": (positive_cos_zenith > 0.0).astype(float),
        },
        index=output_index,
    )


def _build_nwp_disagreement_features(proxy: pd.DataFrame) -> pd.DataFrame:
    """Per-TSO ECMWF/DWD proxy ratio and difference — exposes NWP ensemble disagreement."""
    frames: dict[str, pd.Series] = {}
    for col in proxy.columns:
        if not (col.startswith("solar_") and col.endswith("_proxy_mw")):
            continue
        region = col[len("solar_"):-len("_proxy_mw")]
        ecmwf_col = f"ecmwf_solar_{region}_proxy_mw"
        if ecmwf_col not in proxy.columns:
            continue
        dwd = proxy[col].astype(float)
        ecmwf = proxy[ecmwf_col].astype(float)
        ratio = np.where(dwd > 10.0, ecmwf / dwd, 1.0)
        frames[f"solar_{region}_ecmwf_dwd_proxy_ratio"] = pd.Series(
            np.clip(ratio, 0.0, 3.0), index=proxy.index
        )
        frames[f"solar_{region}_ecmwf_dwd_proxy_diff_mw"] = pd.Series(
            (ecmwf - dwd).values, index=proxy.index
        )
    return pd.DataFrame(frames, index=proxy.index)


def _build_solar_geometry_features(proxy: pd.DataFrame) -> pd.DataFrame:
    """Build deterministic solar-geometry and clear-sky-index features."""
    blocks: list[pd.DataFrame] = []
    clear_sky_columns = []
    for region, (latitude, longitude) in _SOLAR_GEOMETRY_POINTS.items():
        geometry = _solar_position_features(proxy.index, latitude=latitude, longitude=longitude)
        geometry = geometry.add_prefix(f"solar_geom_{region}_")

        irradiance_col = f"solar_{region}_irradiance_cap_weighted_W_m2"
        clear_sky_col = f"solar_geom_{region}_clear_sky_proxy"
        if irradiance_col in proxy.columns:
            denominator = 1000.0 * geometry[clear_sky_col].to_numpy(dtype=float)
            clear_sky_index = np.divide(
                proxy[irradiance_col].to_numpy(dtype=float),
                denominator,
                out=np.zeros(len(proxy), dtype=float),
                where=denominator > 1e-6,
            )
            geometry[f"solar_geom_{region}_clear_sky_index"] = np.clip(clear_sky_index, 0.0, 2.0)

        blocks.append(geometry)
        clear_sky_columns.append(clear_sky_col)

    result = pd.concat(blocks, axis=1)
    result["solar_geom_clear_sky_proxy_mean"] = result[clear_sky_columns].mean(axis=1)
    result["solar_geom_clear_sky_proxy_max"] = result[clear_sky_columns].max(axis=1)
    result["solar_geom_clear_sky_proxy_spread"] = (
        result["solar_geom_clear_sky_proxy_max"] - result[clear_sky_columns].min(axis=1)
    )
    result.index.name = "timestamp"
    return result


def _build_solar_physics_features(proxy: pd.DataFrame) -> pd.DataFrame:
    """Build compact PV-oriented features from regional solar weather proxies."""
    frames: dict[str, pd.Series] = {}
    summary_columns: dict[str, list[str]] = {
        "direct_proxy": [],
        "diffuse_proxy": [],
        "temperature_corrected_proxy": [],
        "clear_sky_index": [],
    }

    for region, (latitude, longitude) in _SOLAR_GEOMETRY_POINTS.items():
        prefix = f"solar_{region}"
        proxy_col = f"{prefix}_proxy_mw"
        irradiance_col = f"{prefix}_irradiance_cap_weighted_W_m2"
        direct_col = f"{prefix}_direct_irradiance_cap_weighted_W_m2"
        t2m_col = f"{prefix}_t2m_cap_weighted_K"

        if proxy_col not in proxy.columns or irradiance_col not in proxy.columns:
            continue

        proxy_mw = proxy[proxy_col].astype(float)
        irradiance = proxy[irradiance_col].astype(float).clip(lower=0.0)
        if direct_col in proxy.columns:
            direct = proxy[direct_col].astype(float).clip(lower=0.0)
        else:
            direct = pd.Series(0.0, index=proxy.index)
        diffuse = (irradiance - direct).clip(lower=0.0)
        direct_share = pd.Series(
            np.divide(
                direct.to_numpy(dtype=float),
                irradiance.to_numpy(dtype=float),
                out=np.zeros(len(proxy), dtype=float),
                where=irradiance.to_numpy(dtype=float) > 1e-6,
            ),
            index=proxy.index,
        ).clip(lower=0.0, upper=1.0)
        diffuse_share = (1.0 - direct_share).clip(lower=0.0, upper=1.0)

        frames[f"solar_phys_{region}_direct_share"] = direct_share
        frames[f"solar_phys_{region}_diffuse_irradiance_W_m2"] = diffuse
        frames[f"solar_phys_{region}_direct_proxy_mw"] = proxy_mw * direct_share
        frames[f"solar_phys_{region}_diffuse_proxy_mw"] = proxy_mw * diffuse_share
        summary_columns["direct_proxy"].append(f"solar_phys_{region}_direct_proxy_mw")
        summary_columns["diffuse_proxy"].append(f"solar_phys_{region}_diffuse_proxy_mw")

        geometry = _solar_position_features(proxy.index, latitude=latitude, longitude=longitude)
        clear_sky = 1000.0 * geometry["clear_sky_proxy"].astype(float)
        clear_sky_index = pd.Series(
            np.divide(
                irradiance.to_numpy(dtype=float),
                clear_sky.to_numpy(dtype=float),
                out=np.zeros(len(proxy), dtype=float),
                where=clear_sky.to_numpy(dtype=float) > 1e-6,
            ),
            index=proxy.index,
        ).clip(lower=0.0, upper=2.0)
        frames[f"solar_phys_{region}_clear_sky_index"] = clear_sky_index
        frames[f"solar_phys_{region}_cloud_attenuation_proxy"] = (1.0 - clear_sky_index).clip(lower=-1.0, upper=1.0)
        summary_columns["clear_sky_index"].append(f"solar_phys_{region}_clear_sky_index")

        if t2m_col in proxy.columns:
            t2m_c = proxy[t2m_col].astype(float) - 273.15
            module_temp_c = t2m_c + 0.025 * irradiance
            efficiency_factor = (1.0 - 0.004 * (module_temp_c - 25.0)).clip(lower=0.75, upper=1.15)
            temp_corrected_proxy = proxy_mw * efficiency_factor
            frames[f"solar_phys_{region}_module_temp_proxy_C"] = module_temp_c
            frames[f"solar_phys_{region}_temperature_efficiency_factor"] = efficiency_factor
            frames[f"solar_phys_{region}_temperature_corrected_proxy_mw"] = temp_corrected_proxy
            frames[f"solar_phys_{region}_hot_irradiance_interaction"] = (module_temp_c - 25.0).clip(lower=0.0) * irradiance
            summary_columns["temperature_corrected_proxy"].append(
                f"solar_phys_{region}_temperature_corrected_proxy_mw"
            )

    if not frames:
        return pd.DataFrame(index=proxy.index)

    result = pd.DataFrame(frames, index=proxy.index)
    for name, columns in summary_columns.items():
        available = [column for column in columns if column in result.columns]
        if not available:
            continue
        result[f"solar_phys_{name}_sum"] = result[available].sum(axis=1)
        result[f"solar_phys_{name}_mean"] = result[available].mean(axis=1)
        result[f"solar_phys_{name}_spread"] = result[available].max(axis=1) - result[available].min(axis=1)

    result.index.name = "timestamp"
    return result


def _build_solar_physics_proxy_baselines(
    proxy: pd.DataFrame,
    config: RenewableGenerationModelConfig,
) -> pd.DataFrame:
    """Build temperature-corrected physical PV baseline columns.

    The raw regional renewable proxy is already capacity weighted. This helper
    keeps that structure and applies a simple module-temperature efficiency
    correction so the model can learn a residual around a physics-informed
    baseline rather than the full solar generation level.
    """
    frames: dict[str, pd.Series] = {}
    proxy_suffix = "_proxy_mw"
    proxy_columns = [
        column
        for column in proxy.columns
        if column.startswith("solar_")
        and column.endswith(proxy_suffix)
        and "_physics_" not in column
    ]
    if not proxy_columns:
        return pd.DataFrame(index=proxy.index)

    for proxy_col in proxy_columns:
        region = proxy_col.removeprefix("solar_").removesuffix(proxy_suffix)
        region_prefix = f"solar_{region}"
        base_proxy = proxy[proxy_col].astype(float).clip(lower=0.0)
        irradiance_col = f"{region_prefix}_irradiance_cap_weighted_W_m2"
        t2m_col = f"{region_prefix}_t2m_cap_weighted_K"

        if irradiance_col in proxy.columns and t2m_col in proxy.columns:
            irradiance = proxy[irradiance_col].astype(float).clip(lower=0.0)
            t2m_c = proxy[t2m_col].astype(float) - 273.15
            module_temp_c = (
                t2m_c
                + config.solar_physics_proxy_module_temperature_irradiance_coeff * irradiance
            )
            temperature_factor = (
                1.0
                + config.solar_physics_proxy_temperature_coefficient
                * (module_temp_c - 25.0)
            ).clip(
                lower=config.solar_physics_proxy_min_temperature_factor,
                upper=config.solar_physics_proxy_max_temperature_factor,
            )
        else:
            module_temp_c = pd.Series(np.nan, index=proxy.index)
            temperature_factor = pd.Series(1.0, index=proxy.index)

        frames[f"{region_prefix}_physics_proxy_mw"] = (base_proxy * temperature_factor).clip(lower=0.0)
        frames[f"{region_prefix}_physics_temperature_factor"] = temperature_factor
        frames[f"{region_prefix}_physics_module_temp_C"] = module_temp_c

    result = pd.DataFrame(frames, index=proxy.index)
    tso_physics_columns = [
        f"solar_{region}_physics_proxy_mw"
        for region in ("50hertz", "amprion", "tennet", "transnetbw")
        if f"solar_{region}_physics_proxy_mw" in result.columns
    ]
    if tso_physics_columns:
        result["Renewable_Solar_Physics_Proxy_MW"] = result[tso_physics_columns].sum(axis=1)
        result["Solar_Physics_Proxy_MW"] = result["Renewable_Solar_Physics_Proxy_MW"]

    result.index.name = "timestamp"
    return result


def _solar_position_for_model_chain(
    index: pd.DatetimeIndex,
    *,
    latitude: float,
    longitude: float,
) -> pd.DataFrame:
    """Solar position with azimuth for the approximate PV model-chain baseline."""
    utc = pd.to_datetime(pd.Index(index), utc=True)
    minutes_utc = (
        utc.hour.to_numpy(dtype=float) * 60.0
        + utc.minute.to_numpy(dtype=float)
        + utc.second.to_numpy(dtype=float) / 60.0
    )
    hour_utc = minutes_utc / 60.0
    day_of_year = utc.dayofyear.to_numpy(dtype=float)

    gamma = 2.0 * np.pi / 365.0 * (day_of_year - 1.0 + (hour_utc - 12.0) / 24.0)
    equation_of_time = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2.0 * gamma)
        - 0.040849 * np.sin(2.0 * gamma)
    )
    declination = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2.0 * gamma)
        + 0.000907 * np.sin(2.0 * gamma)
        - 0.002697 * np.cos(3.0 * gamma)
        + 0.00148 * np.sin(3.0 * gamma)
    )

    lat_rad = np.deg2rad(latitude)
    true_solar_time = (minutes_utc + equation_of_time + 4.0 * longitude) % 1440.0
    hour_angle = np.deg2rad(true_solar_time / 4.0 - 180.0)
    cos_zenith = (
        np.sin(lat_rad) * np.sin(declination)
        + np.cos(lat_rad) * np.cos(declination) * np.cos(hour_angle)
    )
    cos_zenith = np.clip(cos_zenith, -1.0, 1.0)
    positive_cos_zenith = np.clip(cos_zenith, 0.0, None)
    zenith_rad = np.arccos(np.clip(cos_zenith, -1.0, 1.0))
    elevation_deg = np.rad2deg(np.arcsin(cos_zenith))
    # NOAA-style azimuth, clockwise from north.
    azimuth_rad = (
        np.arctan2(
            np.sin(hour_angle),
            np.cos(hour_angle) * np.sin(lat_rad) - np.tan(declination) * np.cos(lat_rad),
        )
        + np.pi
    ) % (2.0 * np.pi)
    return pd.DataFrame(
        {
            "cos_zenith": positive_cos_zenith,
            "sin_zenith": np.sin(zenith_rad).clip(0.0, None),
            "azimuth_rad": azimuth_rad,
            "elevation_deg": elevation_deg,
        },
        index=index,
    )


def _poa_irradiance_from_horizontal(
    *,
    ghi: pd.Series,
    direct_horizontal: pd.Series,
    diffuse_horizontal: pd.Series,
    position: pd.DataFrame,
    config: RenewableGenerationModelConfig,
) -> pd.Series:
    """Approximate POA irradiance using an orientation mixture and isotropic diffuse sky."""
    cos_zenith = position["cos_zenith"].to_numpy(dtype=float)
    sin_zenith = position["sin_zenith"].to_numpy(dtype=float)
    solar_azimuth = position["azimuth_rad"].to_numpy(dtype=float)
    ghi_values = ghi.to_numpy(dtype=float)
    direct_values = direct_horizontal.to_numpy(dtype=float)
    diffuse_values = diffuse_horizontal.to_numpy(dtype=float)
    dni = np.divide(
        direct_values,
        cos_zenith,
        out=np.zeros_like(direct_values, dtype=float),
        where=cos_zenith > 1e-4,
    )
    dni = np.clip(dni, 0.0, 1400.0)

    weights = np.asarray(config.solar_model_chain_orientation_weights, dtype=float)
    weights = weights / weights.sum()
    poa = np.zeros(len(ghi), dtype=float)
    for tilt_deg, azimuth_deg, weight in zip(
        config.solar_model_chain_tilt_degrees,
        config.solar_model_chain_azimuth_degrees,
        weights,
        strict=True,
    ):
        tilt = np.deg2rad(float(tilt_deg))
        surface_azimuth = np.deg2rad(float(azimuth_deg))
        cos_incidence = (
            cos_zenith * np.cos(tilt)
            + sin_zenith * np.sin(tilt) * np.cos(solar_azimuth - surface_azimuth)
        )
        cos_incidence = np.clip(cos_incidence, 0.0, None)
        beam_poa = dni * cos_incidence
        diffuse_poa = diffuse_values * (1.0 + np.cos(tilt)) / 2.0
        ground_poa = config.solar_model_chain_albedo * ghi_values * (1.0 - np.cos(tilt)) / 2.0
        poa += weight * (beam_poa + diffuse_poa + ground_poa)

    return pd.Series(np.clip(poa, 0.0, None), index=ghi.index)


def _build_solar_model_chain_proxy_baselines(
    proxy: pd.DataFrame,
    config: RenewableGenerationModelConfig,
) -> pd.DataFrame:
    """Build an approximate paper-style PV model-chain baseline by TSO region.

    The baseline converts horizontal direct/diffuse irradiance to a simple
    orientation-mixed plane-of-array irradiance estimate, applies module
    temperature derating, and rescales the existing MaStR-capacity proxy.
    """
    frames: dict[str, pd.Series] = {}
    proxy_suffix = "_proxy_mw"
    proxy_columns = [
        column
        for column in proxy.columns
        if column.startswith("solar_")
        and column.endswith(proxy_suffix)
        and "_physics_" not in column
    ]
    if not proxy_columns:
        return pd.DataFrame(index=proxy.index)

    performance_ratio_scale = (
        config.solar_model_chain_performance_ratio
        / config.solar_model_chain_base_proxy_performance_ratio
    )
    for proxy_col in proxy_columns:
        region = proxy_col.removeprefix("solar_").removesuffix(proxy_suffix)
        region_prefix = f"solar_{region}"
        geometry_region = _solar_geometry_region_key_for_tso_region(region) or region
        latitude, longitude = _SOLAR_GEOMETRY_POINTS.get(geometry_region, _SOLAR_GEOMETRY_POINTS["central"])

        irradiance_col = f"{region_prefix}_irradiance_cap_weighted_W_m2"
        direct_col = f"{region_prefix}_direct_irradiance_cap_weighted_W_m2"
        diffuse_col = f"{region_prefix}_diffuse_irradiance_cap_weighted_W_m2"
        t2m_col = f"{region_prefix}_t2m_cap_weighted_K"
        if irradiance_col not in proxy.columns:
            continue

        base_proxy = proxy[proxy_col].astype(float).clip(lower=0.0)
        ghi = proxy[irradiance_col].astype(float).clip(lower=0.0)
        if direct_col in proxy.columns:
            direct_horizontal = proxy[direct_col].astype(float).clip(lower=0.0)
        else:
            direct_horizontal = pd.Series(0.0, index=proxy.index)
        if diffuse_col in proxy.columns:
            diffuse_horizontal = proxy[diffuse_col].astype(float).clip(lower=0.0)
        else:
            diffuse_horizontal = (ghi - direct_horizontal).clip(lower=0.0)

        position = _solar_position_for_model_chain(proxy.index, latitude=latitude, longitude=longitude)
        poa = _poa_irradiance_from_horizontal(
            ghi=ghi,
            direct_horizontal=direct_horizontal,
            diffuse_horizontal=diffuse_horizontal,
            position=position,
            config=config,
        )
        poa_ratio = pd.Series(
            np.divide(
                poa.to_numpy(dtype=float),
                ghi.to_numpy(dtype=float),
                out=np.zeros(len(proxy), dtype=float),
                where=ghi.to_numpy(dtype=float) > 1e-6,
            ),
            index=proxy.index,
        ).clip(lower=0.0, upper=2.5)

        if t2m_col in proxy.columns:
            t2m_c = proxy[t2m_col].astype(float) - 273.15
            module_temp_c = (
                t2m_c
                + config.solar_physics_proxy_module_temperature_irradiance_coeff * poa
            )
            temperature_factor = (
                1.0
                + config.solar_physics_proxy_temperature_coefficient
                * (module_temp_c - 25.0)
            ).clip(
                lower=config.solar_physics_proxy_min_temperature_factor,
                upper=config.solar_physics_proxy_max_temperature_factor,
            )
        else:
            module_temp_c = pd.Series(np.nan, index=proxy.index)
            temperature_factor = pd.Series(1.0, index=proxy.index)

        model_chain = (base_proxy * poa_ratio * temperature_factor * performance_ratio_scale).clip(lower=0.0)
        frames[f"{region_prefix}_model_chain_proxy_mw"] = model_chain
        frames[f"{region_prefix}_model_chain_poa_irradiance_W_m2"] = poa
        frames[f"{region_prefix}_model_chain_poa_ratio"] = poa_ratio
        frames[f"{region_prefix}_model_chain_temperature_factor"] = temperature_factor
        frames[f"{region_prefix}_model_chain_module_temp_C"] = module_temp_c

    result = pd.DataFrame(frames, index=proxy.index)
    tso_columns = [
        f"solar_{region}_model_chain_proxy_mw"
        for region in ("50hertz", "amprion", "tennet", "transnetbw")
        if f"solar_{region}_model_chain_proxy_mw" in result.columns
    ]
    if tso_columns:
        result["Renewable_Solar_Model_Chain_Proxy_MW"] = result[tso_columns].sum(axis=1)
        result["Solar_Model_Chain_Proxy_MW"] = result["Renewable_Solar_Model_Chain_Proxy_MW"]

    result.index.name = "timestamp"
    return result


def _build_dwd_cluster_features(config: RenewableGenerationModelConfig) -> pd.DataFrame:
    df_hourly, df_qh = load_dwd(
        icon_dir=config.icon_dir,
        start_folder_date=config.start_folder_date,
        required_run=config.required_run,
        skip_dates=set(config.skip_dates),
        folder_offset_date=config.dwd_folder_offset_date,
        target_tz=config.target_tz,
    )

    hourly_data: dict[str, np.ndarray] = {}
    u_cols = sorted(column for column in df_hourly.columns if column.startswith("u10_cluster_"))
    for u_col in u_cols:
        cluster_id = u_col.rsplit("_", 1)[-1]
        v_col = f"v10_cluster_{cluster_id}"
        if v_col not in df_hourly.columns:
            continue

        speed_10m = np.sqrt(df_hourly[u_col].to_numpy(dtype=float) ** 2 + df_hourly[v_col].to_numpy(dtype=float) ** 2)
        speed_hub = speed_10m * (config.wind_hub_height_m / config.wind_reference_height_m) ** config.wind_shear_alpha
        hourly_data[f"wind_speed_10m_cluster_{cluster_id}"] = speed_10m
        hourly_data[f"wind_speed_hub_cluster_{cluster_id}"] = speed_hub
        hourly_data[f"wind_speed_hub_sq_cluster_{cluster_id}"] = speed_hub**2
        hourly_data[f"wind_speed_hub_cube_cluster_{cluster_id}"] = speed_hub**3

    for column in sorted(column for column in df_hourly.columns if column.startswith("t2m_cluster_")):
        hourly_data[column] = df_hourly[column].to_numpy(dtype=float)

    hourly_features = pd.DataFrame(hourly_data, index=df_hourly.index)

    qh_data: dict[str, np.ndarray] = {}
    dir_cols = sorted(column for column in df_qh.columns if column.startswith("ASWDIR_cluster_"))
    for dir_col in dir_cols:
        cluster_id = dir_col.rsplit("_", 1)[-1]
        dif_col = f"ASWDIFD_cluster_{cluster_id}"
        if dif_col not in df_qh.columns:
            continue

        direct = np.clip(df_qh[dir_col].to_numpy(dtype=float), 0.0, None)
        diffuse = np.clip(df_qh[dif_col].to_numpy(dtype=float), 0.0, None)
        qh_data[f"solar_direct_cluster_{cluster_id}"] = direct
        qh_data[f"solar_diffuse_cluster_{cluster_id}"] = diffuse
        qh_data[f"solar_global_cluster_{cluster_id}"] = direct + diffuse

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


def _add_lag_diff_features(
    features: pd.DataFrame,
    *,
    lag_steps: list[int],
    lead_steps: list[int] | None = None,
    diff_steps: list[int],
    include_original: bool = True,
) -> pd.DataFrame:
    """Add paper-style lagged and differenced NWP features without changing raw inputs."""
    if features.empty:
        return features

    blocks = [features] if include_original else []
    for step in lag_steps:
        if step <= 0:
            continue
        lagged = features.shift(step).add_suffix(f"_lag_{step}")
        blocks.append(lagged)
    for step in lead_steps or []:
        if step <= 0:
            continue
        lead = features.shift(-step).add_suffix(f"_lead_{step}")
        blocks.append(lead)
    for step in diff_steps:
        if step <= 0:
            continue
        differenced = features.diff(step).add_suffix(f"_diff_{step}")
        blocks.append(differenced)

    return pd.concat(blocks, axis=1)


def _select_proxy_lag_diff_columns(
    proxy: pd.DataFrame,
    patterns: list[str],
) -> pd.DataFrame:
    numeric = proxy.select_dtypes(include="number")
    if not patterns:
        return numeric

    lowered_patterns = [pattern.lower() for pattern in patterns if pattern]
    if not lowered_patterns:
        return numeric

    columns = [
        column
        for column in numeric.columns
        if any(pattern in column.lower() for pattern in lowered_patterns)
    ]
    return numeric[columns]


def _build_proxy_window_context_features(
    proxy: pd.DataFrame,
    *,
    patterns: list[str],
    window_steps: list[int],
    stats: list[str],
) -> pd.DataFrame:
    """Add centered forecast-trajectory summaries around each timestamp."""
    source = _select_proxy_lag_diff_columns(proxy, patterns)
    if source.empty:
        return pd.DataFrame(index=proxy.index)

    blocks: list[pd.DataFrame] = []
    wanted_stats = {stat.lower() for stat in stats}
    for step in sorted({int(value) for value in window_steps if int(value) > 0}):
        window = 2 * step + 1
        rolled = source.rolling(window=window, center=True, min_periods=1)
        suffix = f"window_pm{step}"
        if "mean" in wanted_stats:
            blocks.append(rolled.mean().add_suffix(f"_{suffix}_mean"))
        if "std" in wanted_stats:
            blocks.append(rolled.std().fillna(0.0).add_suffix(f"_{suffix}_std"))
        if "min" in wanted_stats:
            blocks.append(rolled.min().add_suffix(f"_{suffix}_min"))
        if "max" in wanted_stats:
            blocks.append(rolled.max().add_suffix(f"_{suffix}_max"))

    if not blocks:
        return pd.DataFrame(index=proxy.index)
    features = pd.concat(blocks, axis=1)
    features.index.name = "timestamp"
    return features


def _source_label_for_feature(column: str) -> str:
    return "open_meteo" if column.startswith("om_") else "dwd"


def _build_wind_high_regime_features(
    proxy: pd.DataFrame,
    *,
    speed_thresholds: list[float],
    proxy_thresholds: list[float],
) -> pd.DataFrame:
    """Build high-wind and possible cut-out regime indicators from forecast features."""
    numeric = proxy.select_dtypes(include="number")
    output: dict[str, pd.Series] = {}

    for source in ["dwd", "open_meteo"]:
        source_columns = [
            column
            for column in numeric.columns
            if _source_label_for_feature(column) == source
        ]
        speed_columns = [
            column
            for column in source_columns
            if "speed_hub_cap_weighted_m_s" in column
            or "speed_120m_cap_weighted_m_s" in column
            or "speed_80m_cap_weighted_m_s" in column
        ]
        if speed_columns:
            speeds = numeric[speed_columns].astype(float).clip(lower=0.0)
            output[f"wind_high_{source}_speed_max_m_s"] = speeds.max(axis=1)
            output[f"wind_high_{source}_speed_mean_m_s"] = speeds.mean(axis=1)
            output[f"wind_high_{source}_speed_std_m_s"] = speeds.std(axis=1).fillna(0.0)
            for threshold in speed_thresholds:
                threshold_value = float(threshold)
                label = str(threshold_value).rstrip("0").rstrip(".").replace(".", "p")
                exceedance = (speeds >= threshold_value).astype(float)
                output[f"wind_high_{source}_speed_share_ge_{label}"] = exceedance.mean(axis=1)
                output[f"wind_high_{source}_speed_max_excess_ge_{label}"] = (
                    speeds.sub(threshold_value).clip(lower=0.0).max(axis=1)
                )

        proxy_columns = [
            column
            for column in source_columns
            if column.endswith("Renewable_Wind_Proxy_MW")
            or column.endswith("Renewable_Wind_Proxy_MW".lower())
            or column.endswith("_proxy_mw")
        ]
        proxy_columns = [
            column
            for column in proxy_columns
            if "_ramp_" not in column.lower()
            and "_lag_" not in column.lower()
            and "_lead_" not in column.lower()
            and "_diff_" not in column.lower()
        ]
        if proxy_columns:
            proxy_values = numeric[proxy_columns].astype(float).clip(lower=0.0)
            proxy_max = proxy_values.max(axis=1)
            output[f"wind_high_{source}_proxy_max_mw"] = proxy_max
            output[f"wind_high_{source}_proxy_sum_mw"] = proxy_values.sum(axis=1)
            for threshold in proxy_thresholds:
                threshold_value = float(threshold)
                label = str(threshold_value).rstrip("0").rstrip(".").replace(".", "p")
                output[f"wind_high_{source}_proxy_max_ge_{label}"] = (proxy_max >= threshold_value).astype(float)
                output[f"wind_high_{source}_proxy_max_excess_ge_{label}"] = (
                    proxy_max - threshold_value
                ).clip(lower=0.0)

    features = pd.DataFrame(output, index=proxy.index)
    features.index.name = "timestamp"
    return features


def _summary_stat_columns(
    columns: list[str],
    *,
    stat: str,
    required_terms: tuple[str, ...],
    technology: str,
) -> list[str]:
    marker = f"_{stat}_"
    return [
        column
        for column in columns
        if marker in column.lower()
        and technology in column.lower()
        and all(term in column.lower() for term in required_terms)
        and "_ramp_" not in column.lower()
        and "_lag_" not in column.lower()
        and "_lead_" not in column.lower()
        and "_diff_" not in column.lower()
    ]


def _threshold_label(value: float) -> str:
    return str(float(value)).rstrip("0").rstrip(".").replace(".", "p")


def _build_wind_ensemble_regime_features(
    proxy: pd.DataFrame,
    *,
    speed_thresholds: list[float],
) -> pd.DataFrame:
    """Interact provider disagreement with wind regimes from ensemble-summary features."""
    numeric = proxy.select_dtypes(include="number")
    columns = list(numeric.columns)
    output: dict[str, pd.Series] = {}

    speed_terms = (
        ("speed_hub_cap_weighted_m_s",),
        ("speed_120m_cap_weighted_m_s",),
        ("speed_80m_cap_weighted_m_s",),
    )

    for technology in ["onshore", "offshore"]:
        mean_speed_cols = sorted(
            {
                column
                for terms in speed_terms
                for column in _summary_stat_columns(
                    columns,
                    stat="mean",
                    required_terms=terms,
                    technology=technology,
                )
            }
        )
        std_speed_cols = sorted(
            {
                column
                for terms in speed_terms
                for column in _summary_stat_columns(
                    columns,
                    stat="std",
                    required_terms=terms,
                    technology=technology,
                )
            }
        )
        mean_proxy_cols = _summary_stat_columns(
            columns,
            stat="mean",
            required_terms=("proxy_mw",),
            technology=technology,
        )
        std_proxy_cols = _summary_stat_columns(
            columns,
            stat="std",
            required_terms=("proxy_mw",),
            technology=technology,
        )

        if not mean_speed_cols and not mean_proxy_cols:
            continue

        prefix = f"wind_ensemble_regime_{technology}"
        if mean_speed_cols:
            mean_speed = numeric[mean_speed_cols].astype(float).clip(lower=0.0).mean(axis=1)
            output[f"{prefix}_speed_mean_m_s"] = mean_speed
        else:
            mean_speed = pd.Series(np.nan, index=proxy.index)

        if std_speed_cols:
            std_speed_values = numeric[std_speed_cols].astype(float).clip(lower=0.0)
            std_speed_mean = std_speed_values.mean(axis=1).fillna(0.0)
            std_speed_max = std_speed_values.max(axis=1).fillna(0.0)
            output[f"{prefix}_provider_speed_std_mean_m_s"] = std_speed_mean
            output[f"{prefix}_provider_speed_std_max_m_s"] = std_speed_max
        else:
            std_speed_mean = pd.Series(0.0, index=proxy.index)
            std_speed_max = pd.Series(0.0, index=proxy.index)

        if mean_proxy_cols:
            mean_proxy = numeric[mean_proxy_cols].astype(float).clip(lower=0.0).sum(axis=1)
            output[f"{prefix}_proxy_mean_sum_mw"] = mean_proxy
        if std_proxy_cols:
            std_proxy = numeric[std_proxy_cols].astype(float).clip(lower=0.0).sum(axis=1)
            output[f"{prefix}_provider_proxy_std_sum_mw"] = std_proxy
        else:
            std_proxy = pd.Series(0.0, index=proxy.index)

        for threshold in speed_thresholds:
            threshold_value = float(threshold)
            label = _threshold_label(threshold_value)
            high = (mean_speed >= threshold_value).astype(float)
            excess = (mean_speed - threshold_value).clip(lower=0.0).fillna(0.0)
            output[f"{prefix}_speed_mean_ge_{label}"] = high
            output[f"{prefix}_provider_speed_std_mean_x_ge_{label}"] = std_speed_mean * high
            output[f"{prefix}_provider_speed_std_max_x_ge_{label}"] = std_speed_max * high
            output[f"{prefix}_provider_speed_std_mean_x_excess_ge_{label}"] = std_speed_mean * excess
            output[f"{prefix}_provider_proxy_std_sum_x_excess_ge_{label}"] = std_proxy * excess

    features = pd.DataFrame(output, index=proxy.index)
    features.index.name = "timestamp"
    return features


def _build_wind_cutout_risk_features(
    proxy: pd.DataFrame,
    *,
    speed_thresholds: list[float],
    std_floor_m_s: float,
) -> pd.DataFrame:
    """Build soft high-wind risk features from provider mean/std wind speeds.

    Provider summary files expose an empirical ensemble mean and spread for the
    same regional hub-height wind features. A smooth exceedance score lets the
    tree model distinguish ordinary high production from cases where the
    ensemble is near rated/cut-out regimes and provider disagreement is high.
    """
    numeric = proxy.select_dtypes(include="number")
    columns = set(numeric.columns)
    output: dict[str, pd.Series] = {}

    speed_terms = (
        ("speed_hub_cap_weighted_m_s",),
        ("speed_80m_cap_weighted_m_s",),
        ("speed_120m_cap_weighted_m_s",),
        ("speed_180m_cap_weighted_m_s",),
    )

    for technology in ["onshore", "offshore"]:
        mean_speed_cols = sorted(
            {
                column
                for terms in speed_terms
                for column in _summary_stat_columns(
                    list(columns),
                    stat="mean",
                    required_terms=terms,
                    technology=technology,
                )
            }
        )
        if not mean_speed_cols:
            continue

        std_speed_cols = [
            column.replace("_mean_", "_std_", 1)
            if column.replace("_mean_", "_std_", 1) in columns
            else None
            for column in mean_speed_cols
        ]
        mean_speed = numeric[mean_speed_cols].astype(float).clip(lower=0.0)
        std_speed = pd.DataFrame(
            {
                mean_col: (
                    numeric[std_col].astype(float).clip(lower=0.0)
                    if std_col is not None
                    else pd.Series(0.0, index=proxy.index)
                )
                for mean_col, std_col in zip(mean_speed_cols, std_speed_cols, strict=True)
            },
            index=proxy.index,
        )
        scale = std_speed.clip(lower=float(std_floor_m_s))

        proxy_mean_cols = _summary_stat_columns(
            list(columns),
            stat="mean",
            required_terms=("proxy_mw",),
            technology=technology,
        )
        proxy_std_cols = [
            column.replace("_mean_", "_std_", 1)
            for column in proxy_mean_cols
            if column.replace("_mean_", "_std_", 1) in columns
        ]
        proxy_sum = (
            numeric[proxy_mean_cols].astype(float).clip(lower=0.0).sum(axis=1)
            if proxy_mean_cols
            else pd.Series(0.0, index=proxy.index)
        )
        proxy_spread_sum = (
            numeric[proxy_std_cols].astype(float).clip(lower=0.0).sum(axis=1)
            if proxy_std_cols
            else pd.Series(0.0, index=proxy.index)
        )

        prefix = f"wind_cutout_risk_{technology}"
        output[f"{prefix}_speed_mean_m_s"] = mean_speed.mean(axis=1)
        output[f"{prefix}_speed_max_m_s"] = mean_speed.max(axis=1)
        output[f"{prefix}_provider_speed_std_mean_m_s"] = std_speed.mean(axis=1)
        output[f"{prefix}_provider_speed_std_max_m_s"] = std_speed.max(axis=1)

        for threshold in speed_thresholds:
            threshold_value = float(threshold)
            label = _threshold_label(threshold_value)
            z = ((mean_speed - threshold_value) / scale).clip(lower=-40.0, upper=40.0)
            soft_exceedance = 1.0 / (1.0 + np.exp(-z))
            hard_exceedance = (mean_speed >= threshold_value).astype(float)
            excess = (mean_speed - threshold_value).clip(lower=0.0)

            soft_mean = soft_exceedance.mean(axis=1)
            soft_max = soft_exceedance.max(axis=1)
            output[f"{prefix}_soft_share_ge_{label}"] = soft_mean
            output[f"{prefix}_soft_max_ge_{label}"] = soft_max
            output[f"{prefix}_hard_share_ge_{label}"] = hard_exceedance.mean(axis=1)
            output[f"{prefix}_speed_excess_mean_ge_{label}_m_s"] = excess.mean(axis=1)
            output[f"{prefix}_speed_excess_max_ge_{label}_m_s"] = excess.max(axis=1)
            output[f"{prefix}_provider_std_x_soft_ge_{label}"] = (std_speed * soft_exceedance).mean(axis=1)
            output[f"{prefix}_proxy_x_soft_ge_{label}_mw"] = proxy_sum * soft_mean
            output[f"{prefix}_proxy_spread_x_soft_ge_{label}_mw"] = proxy_spread_sum * soft_mean

    features = pd.DataFrame(output, index=proxy.index)
    features.index.name = "timestamp"
    return features


def _build_actual_generation_lag_features(
    actual: pd.DataFrame,
    config: RenewableGenerationModelConfig,
) -> pd.DataFrame:
    """Build leakage-safe actual generation lags available before day-ahead gate closure."""
    lag_days = sorted({int(day) for day in config.actual_generation_lag_days if int(day) > 0})
    if not lag_days:
        return pd.DataFrame(index=actual.index)

    columns = [column for column in config.actual_generation_lag_columns if column in actual.columns]
    if not columns:
        return pd.DataFrame(index=actual.index)

    blocks = []
    for lag_day in lag_days:
        lagged = actual[columns].copy()
        lagged.index = lagged.index + pd.Timedelta(days=lag_day)
        lagged = lagged.rename(columns={column: f"{column}_lag_d{lag_day}" for column in columns})
        blocks.append(lagged)

    result = pd.concat(blocks, axis=1).sort_index()
    result = result.loc[~result.index.duplicated(keep="last")]
    result.index.name = "timestamp"
    return result


def _partial_generation_feature_label(column: str) -> str:
    return column.removesuffix("_Actual_MW").lower()


def _partial_generation_window_label(end_hour: int, end_minute: int) -> str:
    if end_minute == 0:
        return f"{end_hour:02d}00"
    return f"{end_hour:02d}{end_minute:02d}"


def _build_partial_actual_generation_features(
    actual: pd.DataFrame,
    target_index: pd.DatetimeIndex,
    *,
    columns: list[str],
    reference_day: int,
    comparison_lag_days: int,
    morning_end_hour: int,
    morning_end_minute: int,
) -> pd.DataFrame:
    """Summarise actual generation observed on the forecast creation morning.

    For a forecast day D, source values are taken from D-reference_day between
    00:00 and the configured morning cutoff. This keeps live forecasts aligned
    with the information available before day-ahead gate closure.
    """
    if target_index.empty:
        return pd.DataFrame(index=target_index)

    available_columns = [column for column in columns if column in actual.columns]
    if not available_columns:
        return pd.DataFrame(index=target_index)

    actual = actual.sort_index()
    forecast_days = pd.DatetimeIndex(target_index.normalize().unique()).sort_values()
    end_delta = pd.Timedelta(hours=morning_end_hour, minutes=morning_end_minute)
    window_label = _partial_generation_window_label(morning_end_hour, morning_end_minute)

    rows: dict[pd.Timestamp, dict[str, float]] = {}
    for day in forecast_days:
        source_day = day - pd.Timedelta(days=reference_day)
        comparison_day = source_day - pd.Timedelta(days=comparison_lag_days)
        source = actual.loc[source_day:source_day + end_delta, available_columns]
        comparison = actual.loc[comparison_day:comparison_day + end_delta, available_columns]

        row: dict[str, float] = {}
        for column in available_columns:
            label = _partial_generation_feature_label(column)
            prefix = f"partial_gen_{label}_d{reference_day}_00_{window_label}"

            source_values = source[column].dropna()
            comparison_values = comparison[column].dropna()

            if source_values.empty:
                source_stats = {
                    "mean": np.nan,
                    "max": np.nan,
                    "min": np.nan,
                    "last": np.nan,
                    "sum": np.nan,
                    "range": np.nan,
                    "ramp": np.nan,
                }
            else:
                source_stats = {
                    "mean": float(source_values.mean()),
                    "max": float(source_values.max()),
                    "min": float(source_values.min()),
                    "last": float(source_values.iloc[-1]),
                    "sum": float(source_values.sum()),
                    "range": float(source_values.max() - source_values.min()),
                    "ramp": float(source_values.iloc[-1] - source_values.iloc[0]),
                }

            if comparison_values.empty:
                comparison_stats = {
                    "mean": np.nan,
                    "last": np.nan,
                    "sum": np.nan,
                    "ramp": np.nan,
                }
            else:
                comparison_stats = {
                    "mean": float(comparison_values.mean()),
                    "last": float(comparison_values.iloc[-1]),
                    "sum": float(comparison_values.sum()),
                    "ramp": float(comparison_values.iloc[-1] - comparison_values.iloc[0]),
                }

            for stat, value in source_stats.items():
                row[f"{prefix}_{stat}"] = value
            for stat, comparison_value in comparison_stats.items():
                row[f"{prefix}_{stat}_diff_d{comparison_lag_days}"] = (
                    source_stats[stat] - comparison_value
                )

        rows[day] = row

    daily = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    result = daily.reindex(pd.DatetimeIndex(target_index.normalize()))
    result.index = target_index
    result.index.name = "timestamp"
    return result


def _select_partial_proxy_columns(
    proxy: pd.DataFrame,
    *,
    patterns: list[str],
    suffix: str,
) -> list[str]:
    columns = [
        column
        for column in proxy.columns
        if (not suffix or column.endswith(suffix))
        and all(pattern in column for pattern in patterns)
    ]
    return sorted(columns)


def _safe_ratio(numerator: float, denominator: float, *, min_denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= min_denominator:
        return np.nan
    return float(numerator / denominator)


def _partial_proxy_error_feature_label(column: str) -> str:
    return column.removesuffix("_Actual_MW").lower()


def _build_partial_proxy_error_features(
    actual: pd.DataFrame,
    proxy: pd.DataFrame,
    target_index: pd.DatetimeIndex,
    config: RenewableGenerationModelConfig,
) -> pd.DataFrame:
    """Summarise same-morning actual-vs-proxy errors available before gate closure.

    For a forecast day D, this uses actual generation and the corresponding NWP
    proxy values from D-reference_day between 00:00 and the configured morning
    cutoff. The resulting features are constant across all MTUs of D and are
    operationally available when forecasting D on the morning of D-1.
    """
    if target_index.empty:
        return pd.DataFrame(index=target_index)

    columns = config.partial_proxy_error_columns or list(config.partial_proxy_error_proxy_column_patterns)
    if not columns:
        columns = [column for column in config.target_columns if column in actual.columns]
    columns = [column for column in columns if column in actual.columns]
    if not columns:
        return pd.DataFrame(index=target_index)

    actual = actual.sort_index()
    proxy = proxy.sort_index()
    forecast_days = pd.DatetimeIndex(target_index.normalize().unique()).sort_values()
    end_delta = pd.Timedelta(
        hours=config.partial_proxy_error_morning_end_hour,
        minutes=config.partial_proxy_error_morning_end_minute,
    )
    window_label = _partial_generation_window_label(
        config.partial_proxy_error_morning_end_hour,
        config.partial_proxy_error_morning_end_minute,
    )

    rows: dict[pd.Timestamp, dict[str, float]] = {}
    for day in forecast_days:
        source_day = day - pd.Timedelta(days=config.partial_proxy_error_reference_day)
        window_start = source_day
        window_end = source_day + end_delta

        row: dict[str, float] = {}
        for column in columns:
            patterns = config.partial_proxy_error_proxy_column_patterns.get(column, [])
            proxy_columns = _select_partial_proxy_columns(
                proxy,
                patterns=patterns,
                suffix=config.partial_proxy_error_proxy_column_suffix,
            )
            if not proxy_columns:
                continue

            actual_values = actual.loc[window_start:window_end, column].astype(float)
            proxy_values = proxy.loc[window_start:window_end, proxy_columns].astype(float).sum(axis=1, min_count=1)
            aligned = pd.concat({"actual": actual_values, "proxy": proxy_values}, axis=1).dropna()

            label = _partial_proxy_error_feature_label(column)
            prefix = (
                f"partial_proxy_error_{label}_d{config.partial_proxy_error_reference_day}"
                f"_00_{window_label}"
            )

            if aligned.empty:
                row.update(
                    {
                        f"{prefix}_actual_mean": np.nan,
                        f"{prefix}_proxy_mean": np.nan,
                        f"{prefix}_error_mean": np.nan,
                        f"{prefix}_ratio_mean": np.nan,
                        f"{prefix}_actual_sum": np.nan,
                        f"{prefix}_proxy_sum": np.nan,
                        f"{prefix}_error_sum": np.nan,
                        f"{prefix}_ratio_sum": np.nan,
                        f"{prefix}_last_error": np.nan,
                        f"{prefix}_last_ratio": np.nan,
                        f"{prefix}_ramp_error": np.nan,
                    }
                )
                continue

            actual_series = aligned["actual"]
            proxy_series = aligned["proxy"]
            actual_mean = float(actual_series.mean())
            proxy_mean = float(proxy_series.mean())
            actual_sum = float(actual_series.sum())
            proxy_sum = float(proxy_series.sum())
            last_actual = float(actual_series.iloc[-1])
            last_proxy = float(proxy_series.iloc[-1])
            actual_ramp = float(actual_series.iloc[-1] - actual_series.iloc[0])
            proxy_ramp = float(proxy_series.iloc[-1] - proxy_series.iloc[0])

            row.update(
                {
                    f"{prefix}_actual_mean": actual_mean,
                    f"{prefix}_proxy_mean": proxy_mean,
                    f"{prefix}_error_mean": actual_mean - proxy_mean,
                    f"{prefix}_ratio_mean": _safe_ratio(
                        actual_mean,
                        proxy_mean,
                        min_denominator=config.partial_proxy_error_min_proxy_mw,
                    ),
                    f"{prefix}_actual_sum": actual_sum,
                    f"{prefix}_proxy_sum": proxy_sum,
                    f"{prefix}_error_sum": actual_sum - proxy_sum,
                    f"{prefix}_ratio_sum": _safe_ratio(
                        actual_sum,
                        proxy_sum,
                        min_denominator=config.partial_proxy_error_min_proxy_mw,
                    ),
                    f"{prefix}_last_error": last_actual - last_proxy,
                    f"{prefix}_last_ratio": _safe_ratio(
                        last_actual,
                        last_proxy,
                        min_denominator=config.partial_proxy_error_min_proxy_mw,
                    ),
                    f"{prefix}_ramp_error": actual_ramp - proxy_ramp,
                }
            )

        rows[day] = row

    daily = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    result = daily.reindex(pd.DatetimeIndex(target_index.normalize()))
    result.index = target_index
    result.index.name = "timestamp"
    return result


def _capacity_nowcasting_clear_sky_column(target: str, columns: pd.Index) -> str | None:
    region_key = _solar_region_key_for_target(target)
    geometry_region_key = _solar_geometry_region_key_for_tso_region(region_key)
    return _first_existing_column(
        [
            f"solar_geom_{geometry_region_key}_clear_sky_proxy" if geometry_region_key else "",
            f"solar_geom_{region_key}_clear_sky_proxy" if region_key else "",
            "solar_geom_clear_sky_proxy_mean",
        ],
        columns,
    )


def _apply_capacity_nowcasting(
    dataset: pd.DataFrame,
    config: RenewableGenerationModelConfig,
) -> pd.DataFrame:
    """Rescale each solar target's baseline proxy by a leakage-safe trailing capacity factor.

    The physics/ECMWF proxy baselines are pinned to static MaStR registry capacity.
    As real PV capacity grows, the proxy under-predicts with a multiplicative,
    irradiance-proportional shape. For each forecast day D we estimate

        factor = sum(actual) / sum(baseline)

    over a trailing window of high-sun hours, using only data available before the
    day-ahead gate closure for D (the same cutoff used for training targets), and
    multiply the baseline column for D by that factor.
    """
    if not config.enable_capacity_nowcasting:
        return dataset

    targets = config.capacity_nowcasting_targets or [
        target for target in config.target_columns if "solar" in target.lower()
    ]
    if not targets:
        return dataset

    columns = dataset.columns
    all_features = [
        column
        for column in dataset.select_dtypes(include="number").columns
        if not column.endswith("_Actual_MW")
    ]

    days = pd.DatetimeIndex(dataset.index.normalize().unique()).sort_values()
    window = pd.Timedelta(days=config.capacity_nowcasting_window_days)

    for target in targets:
        baseline_column = _target_baseline_column(target, config, all_features)
        if baseline_column is None or baseline_column not in dataset.columns:
            continue
        if target not in dataset.columns:
            continue
        clear_sky_column = _capacity_nowcasting_clear_sky_column(target, columns)
        if clear_sky_column is None:
            raise ValueError(
                "enable_capacity_nowcasting requires solar geometry features "
                f"(clear-sky proxy) for target {target!r}."
            )

        actual = dataset[target].astype(float)
        baseline = dataset[baseline_column].astype(float)
        clear_sky = dataset[clear_sky_column].astype(float)
        eligible = (
            (clear_sky >= config.capacity_nowcasting_min_clear_sky)
            & (baseline > 0.0)
            & actual.notna()
        )

        factor_by_day: dict[pd.Timestamp, float] = {}
        last_factor = 1.0
        for day in days:
            cutoff = _training_target_cutoff(day, config)
            window_start = cutoff - window
            mask = eligible & (dataset.index > window_start) & (dataset.index <= cutoff)
            n_obs = int(mask.sum())
            if n_obs >= config.capacity_nowcasting_min_observations:
                baseline_sum = float(baseline[mask].sum())
                actual_sum = float(actual[mask].sum())
                if baseline_sum > 0.0:
                    raw_factor = actual_sum / baseline_sum
                    factor = 1.0 + config.capacity_nowcasting_shrinkage * (raw_factor - 1.0)
                    factor = float(
                        np.clip(
                            factor,
                            config.capacity_nowcasting_min_factor,
                            config.capacity_nowcasting_max_factor,
                        )
                    )
                    last_factor = factor
            factor_by_day[day] = last_factor

        day_index = pd.DatetimeIndex(dataset.index.normalize())
        factors = day_index.map(factor_by_day).astype(float)
        dataset[baseline_column] = dataset[baseline_column].astype(float) * factors.to_numpy()

    return dataset


def build_renewable_generation_dataset(config: RenewableGenerationModelConfig) -> pd.DataFrame:
    """Build aligned timestamp-level features and actual renewable generation targets."""
    actual = _load_or_fetch_actual_generation(config)
    proxy = _load_renewable_proxy(config)

    feature_blocks = [proxy]
    if config.include_proxy_lag_diff_features:
        proxy_lag_diff_source = _select_proxy_lag_diff_columns(
            proxy,
            config.proxy_lag_diff_feature_patterns,
        )
        feature_blocks.append(
            _add_lag_diff_features(
                proxy_lag_diff_source,
                lag_steps=config.nwp_lag_steps,
                lead_steps=config.nwp_lead_steps,
                diff_steps=config.nwp_diff_steps,
                include_original=False,
            )
        )
    if config.include_proxy_window_context_features:
        feature_blocks.append(
            _build_proxy_window_context_features(
                proxy,
                patterns=config.proxy_window_context_feature_patterns,
                window_steps=config.proxy_window_context_steps,
                stats=config.proxy_window_context_stats,
            )
        )
    if config.include_wind_direction_regime_features:
        feature_blocks.append(_build_wind_direction_regime_features(proxy))
    if config.include_wind_high_regime_features:
        feature_blocks.append(
            _build_wind_high_regime_features(
                proxy,
                speed_thresholds=config.wind_high_regime_speed_thresholds_m_s,
                proxy_thresholds=config.wind_high_regime_proxy_thresholds_mw,
            )
        )
    if config.include_wind_ensemble_regime_features:
        feature_blocks.append(
            _build_wind_ensemble_regime_features(
                proxy,
                speed_thresholds=config.wind_ensemble_regime_speed_thresholds_m_s,
            )
        )
    if config.include_wind_cutout_risk_features:
        feature_blocks.append(
            _build_wind_cutout_risk_features(
                proxy,
                speed_thresholds=config.wind_cutout_risk_speed_thresholds_m_s,
                std_floor_m_s=config.wind_cutout_risk_std_floor_m_s,
            )
        )
    if config.include_regional_summary_features:
        feature_blocks.append(_build_regional_summary_features(proxy))
    feature_blocks.append(_build_time_features(proxy.index))
    if config.include_forecast_lead_features:
        feature_blocks.append(_build_forecast_lead_features(proxy.index, config))
    if config.include_nwp_disagreement_features:
        feature_blocks.append(_build_nwp_disagreement_features(proxy))
    if config.include_solar_geometry_features:
        feature_blocks.append(_build_solar_geometry_features(proxy))
    if config.include_solar_physics_features:
        feature_blocks.append(_build_solar_physics_features(proxy))
    if config.include_solar_physics_features or config.target_baseline_mode == "solar_physics_proxy":
        feature_blocks.append(_build_solar_physics_proxy_baselines(proxy, config))
    if config.include_solar_physics_features or config.target_baseline_mode == "solar_model_chain_proxy":
        feature_blocks.append(_build_solar_model_chain_proxy_baselines(proxy, config))
    if config.include_dwd_cluster_features:
        dwd_features = _build_dwd_cluster_features(config)
        if config.include_nwp_lag_diff_features:
            dwd_features = _add_lag_diff_features(
                dwd_features,
                lag_steps=config.nwp_lag_steps,
                lead_steps=config.nwp_lead_steps,
                diff_steps=config.nwp_diff_steps,
            )
        feature_blocks.append(dwd_features)
    if config.include_unavailability:
        feature_blocks.append(_load_or_fetch_unavailability(config))
    if config.actual_generation_lag_days:
        feature_blocks.append(_build_actual_generation_lag_features(actual, config))
    if config.include_partial_actual_generation_features:
        partial_columns = config.partial_generation_columns or config.actual_generation_lag_columns
        feature_blocks.append(
            _build_partial_actual_generation_features(
                actual,
                proxy.index,
                columns=partial_columns,
                reference_day=config.partial_generation_reference_day,
                comparison_lag_days=config.partial_generation_comparison_lag_days,
                morning_end_hour=config.partial_generation_morning_end_hour,
                morning_end_minute=config.partial_generation_morning_end_minute,
            )
        )
    if config.include_partial_proxy_error_features:
        feature_blocks.append(_build_partial_proxy_error_features(actual, proxy, proxy.index, config))

    features = pd.concat(feature_blocks, axis=1).sort_index()
    features = features.loc[:, ~features.columns.duplicated()]
    features = features.reindex(proxy.index)
    dataset = features.join(actual, how="left")
    dataset.index.name = "timestamp"
    dataset = dataset.sort_index()
    dataset = _apply_capacity_nowcasting(dataset, config)
    return dataset


def _feature_columns(dataset: pd.DataFrame, target_columns: list[str]) -> list[str]:
    target_set = set(target_columns) | {
        "Solar_Actual_MW",
        "Wind_Onshore_Actual_MW",
        "Wind_Offshore_Actual_MW",
        "Wind_Total_Actual_MW",
        "Renewable_Total_Actual_MW",
        "Solar_Control_Area_Total_Actual_MW",
    }
    return [
        column
        for column in dataset.select_dtypes(include="number").columns
        if column not in target_set and not column.endswith("_Actual_MW")
    ]


def _read_feature_allowlist(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Feature allowlist file not found: {path}")

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        if "feature" in df.columns:
            values = df["feature"]
        elif len(df.columns) >= 1:
            values = df.iloc[:, 0]
        else:
            values = pd.Series(dtype=str)
        return [str(value) for value in values.dropna().tolist() if str(value).strip()]

    with path.open(encoding="utf-8") as handle:
        return [
            line.strip()
            for line in handle
            if line.strip() and not line.lstrip().startswith("#")
        ]


def _feature_allowlist_for_target(
    target: str,
    config: RenewableGenerationModelConfig,
) -> list[str] | None:
    path = None
    for alias in _target_aliases(target):
        path = config.target_feature_allowlist_files.get(alias)
        if path is not None:
            break
    if path is None:
        path = config.feature_allowlist_file
    if path is None:
        return None
    return _read_feature_allowlist(path)


def _target_candidate_features(
    all_features: list[str],
    target: str,
    config: RenewableGenerationModelConfig,
) -> list[str]:
    if config.target_feature_mode == "all":
        return all_features

    target_lower = target.lower()
    time_features = {
        "mtu",
        "tod_sin",
        "tod_cos",
        "dow_sin",
        "dow_cos",
        "doy_sin",
        "doy_cos",
        "is_weekend",
        "forecast_lead_hours",
        "forecast_lead_days",
        "forecast_lead_sin",
        "forecast_lead_cos",
    }

    if "solar" in target_lower:
        technology_terms = ("solar", "t2m")
    elif "offshore" in target_lower:
        technology_terms = ("offshore", "t2m", "sp", "cloud")
    elif "onshore" in target_lower:
        technology_terms = ("onshore", "t2m", "sp", "cloud")
    elif "wind" in target_lower:
        technology_terms = ("wind", "t2m")
    else:
        return all_features

    selected = [
        feature
        for feature in all_features
        if (
            feature in time_features
            or any(term in feature.lower() for term in technology_terms)
            or (config.include_unavailability_in_target_features and feature.lower().startswith("unavail_"))
        )
    ]
    if not selected:
        return all_features
    if (
        "solar" in target_lower
        and config.solar_target_region_scope == "own_region_plus_global"
        and (region_key := _solar_region_key_for_target(target)) is not None
    ):
        selected = [
            feature
            for feature in selected
            if _solar_feature_matches_region_scope(feature, region_key)
        ]
        if not selected:
            return all_features
    return selected


def _target_aliases(target: str) -> tuple[str, ...]:
    return (
        target,
        target.removesuffix("_Actual_MW"),
        target.removesuffix("_Model_MW"),
        target.replace("_Actual_MW", ""),
        target.replace("_Model_MW", ""),
    )


_SOLAR_TARGET_REGION_KEYS = {
    "50hertz": "50hertz",
    "amprion": "amprion",
    "tennet": "tennet",
    "transnetbw": "transnetbw",
}


def _solar_region_key_for_target(target: str) -> str | None:
    target_lower = target.lower()
    for region_key in _SOLAR_TARGET_REGION_KEYS:
        if region_key in target_lower:
            return region_key
    return None


def _solar_geometry_region_key_for_tso_region(region_key: str | None) -> str | None:
    if region_key is None:
        return None
    return {
        "50hertz": "east",
        "amprion": "west",
        "tennet": "north",
        "transnetbw": "south",
    }.get(region_key)


def _solar_feature_matches_region_scope(feature: str, region_key: str) -> bool:
    feature_lower = feature.lower()
    region_tokens = [
        f"solar_{key}"
        for key in _SOLAR_TARGET_REGION_KEYS
    ] + [
        f"solar_geom_{key}"
        for key in _SOLAR_TARGET_REGION_KEYS
    ] + [
        f"solar_phys_{key}"
        for key in _SOLAR_TARGET_REGION_KEYS
    ]
    if not any(token in feature_lower for token in region_tokens):
        return True
    return (
        f"solar_{region_key}" in feature_lower
        or f"solar_geom_{region_key}" in feature_lower
        or f"solar_phys_{region_key}" in feature_lower
    )


def _target_model_overrides(config: RenewableGenerationModelConfig, target: str | None) -> dict[str, Any]:
    if target is None:
        return {}
    for key in _target_aliases(target):
        overrides = config.target_model_overrides.get(key)
        if overrides:
            return overrides
    return {}


def _model_param(config: RenewableGenerationModelConfig, overrides: dict[str, Any], name: str) -> Any:
    return overrides.get(name, getattr(config, name))


def _make_model(config: RenewableGenerationModelConfig, target: str | None = None):
    overrides = _target_model_overrides(config, target)
    model_type = _model_param(config, overrides, "model_type")
    use_pca = bool(_model_param(config, overrides, "use_pca"))
    pca_n_components = _model_param(config, overrides, "pca_n_components")
    pca_whiten = bool(_model_param(config, overrides, "pca_whiten"))

    if model_type == "hist_gradient_boosting":
        estimator = HistGradientBoostingRegressor(
            loss="squared_error",
            max_iter=int(_model_param(config, overrides, "hgb_max_iter")),
            learning_rate=float(_model_param(config, overrides, "hgb_learning_rate")),
            max_leaf_nodes=int(_model_param(config, overrides, "hgb_max_leaf_nodes")),
            max_depth=_model_param(config, overrides, "hgb_max_depth"),
            min_samples_leaf=int(_model_param(config, overrides, "hgb_min_samples_leaf")),
            l2_regularization=float(_model_param(config, overrides, "hgb_l2_regularization")),
            max_features=float(_model_param(config, overrides, "hgb_max_features")),
            max_bins=int(_model_param(config, overrides, "hgb_max_bins")),
            random_state=int(_model_param(config, overrides, "random_state")),
        )
        if use_pca:
            return make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                PCA(n_components=pca_n_components, whiten=pca_whiten),
                estimator,
            )
        return estimator
    if model_type == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise ImportError("model_type='lightgbm' requires the optional `lightgbm` package.") from exc

        estimator = LGBMRegressor(
            objective="regression",
            n_estimators=int(_model_param(config, overrides, "lgbm_n_estimators")),
            learning_rate=float(_model_param(config, overrides, "lgbm_learning_rate")),
            num_leaves=int(_model_param(config, overrides, "lgbm_num_leaves")),
            max_depth=int(_model_param(config, overrides, "lgbm_max_depth")),
            min_child_samples=int(_model_param(config, overrides, "lgbm_min_child_samples")),
            subsample=float(_model_param(config, overrides, "lgbm_subsample")),
            colsample_bytree=float(_model_param(config, overrides, "lgbm_colsample_bytree")),
            reg_alpha=float(_model_param(config, overrides, "lgbm_reg_alpha")),
            reg_lambda=float(_model_param(config, overrides, "lgbm_reg_lambda")),
            random_state=int(_model_param(config, overrides, "random_state")),
            n_jobs=-1,
            verbosity=-1,
        )
        if use_pca:
            return make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                PCA(n_components=pca_n_components, whiten=pca_whiten),
                estimator,
            )
        return estimator
    if model_type == "ridge":
        steps = [
            SimpleImputer(strategy="median"),
            StandardScaler(),
        ]
        if use_pca:
            steps.append(PCA(n_components=pca_n_components, whiten=pca_whiten))
        steps.append(Ridge(alpha=float(_model_param(config, overrides, "ridge_alpha"))))
        return make_pipeline(
            *steps,
        )
    raise ValueError(f"Unsupported renewable generation model_type: {model_type!r}")


def _recency_sample_weight(
    index: pd.DatetimeIndex,
    *,
    reference_time: pd.Timestamp,
    half_life_days: float | None,
) -> np.ndarray | None:
    if half_life_days is None:
        return None
    if half_life_days <= 0:
        raise ValueError("training_recency_half_life_days must be positive.")

    ages_days = (
        reference_time - pd.DatetimeIndex(index)
    ).total_seconds() / 86400.0
    ages_days = np.clip(ages_days.astype(float), 0.0, None)
    weights = np.power(0.5, ages_days / float(half_life_days))
    mean_weight = float(np.nanmean(weights))
    if not np.isfinite(mean_weight) or mean_weight <= 0.0:
        return None
    return weights / mean_weight


def _fit_model_with_optional_sample_weight(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sample_weight: np.ndarray | None,
) -> None:
    if sample_weight is None:
        model.fit(X_train, y_train)
        return

    if hasattr(model, "steps") and model.steps:
        final_step_name = model.steps[-1][0]
        model.fit(X_train, y_train, **{f"{final_step_name}__sample_weight": sample_weight})
        return

    model.fit(X_train, y_train, sample_weight=sample_weight)


def _target_output_name(target: str) -> str:
    return target.removesuffix("_Actual_MW")


def _solar_physics_proxy_baseline_candidates(target: str) -> list[str]:
    target_lower = _target_output_name(target).lower()
    if "50hertz" in target_lower:
        regions = ["50hertz"]
    elif "amprion" in target_lower:
        regions = ["amprion"]
    elif "tennet" in target_lower:
        regions = ["tennet"]
    elif "transnetbw" in target_lower:
        regions = ["transnetbw"]
    elif "solar" in target_lower:
        return [
            "Solar_Physics_Proxy_MW",
            "Renewable_Solar_Physics_Proxy_MW",
            "Renewable_Solar_Proxy_MW",
        ]
    else:
        regions = []

    candidates: list[str] = []
    for region in regions:
        candidates.extend(
            [
                f"solar_{region}_physics_proxy_mw",
                f"solar_{region}_proxy_mw",
            ]
        )
    return candidates


def _solar_model_chain_proxy_baseline_candidates(target: str) -> list[str]:
    target_lower = _target_output_name(target).lower()
    if "50hertz" in target_lower:
        regions = ["50hertz"]
    elif "amprion" in target_lower:
        regions = ["amprion"]
    elif "tennet" in target_lower:
        regions = ["tennet"]
    elif "transnetbw" in target_lower:
        regions = ["transnetbw"]
    elif "solar" in target_lower:
        return [
            "Solar_Model_Chain_Proxy_MW",
            "Renewable_Solar_Model_Chain_Proxy_MW",
        ]
    else:
        regions = []

    return [f"solar_{region}_model_chain_proxy_mw" for region in regions]


def _target_baseline_column(
    target: str,
    config: RenewableGenerationModelConfig,
    features: list[str],
) -> str | None:
    if config.target_baseline_mode == "none":
        return None

    feature_set = set(features)
    for alias in _target_aliases(target):
        configured = config.target_baseline_columns.get(alias)
        if configured is None:
            continue
        if configured not in feature_set:
            raise ValueError(
                f"Configured target baseline column {configured!r} for {target!r} "
                "is not available in the renewable generation dataset."
            )
        return configured

    if config.target_baseline_mode == "solar_physics_proxy":
        if "solar" not in target.lower():
            return None
        for candidate in _solar_physics_proxy_baseline_candidates(target):
            if candidate in feature_set:
                return candidate
        raise ValueError(
            f"target_baseline_mode='solar_physics_proxy' could not find a baseline "
            f"column for {target!r}. Expected one of "
            f"{_solar_physics_proxy_baseline_candidates(target)}."
        )

    if config.target_baseline_mode == "solar_model_chain_proxy":
        if "solar" not in target.lower():
            return None
        for candidate in _solar_model_chain_proxy_baseline_candidates(target):
            if candidate in feature_set:
                return candidate
        raise ValueError(
            f"target_baseline_mode='solar_model_chain_proxy' could not find a baseline "
            f"column for {target!r}. Expected one of "
            f"{_solar_model_chain_proxy_baseline_candidates(target)}."
        )

    raise ValueError(f"Unsupported target_baseline_mode: {config.target_baseline_mode!r}")


def _wind_proxy_baseline_feature_columns(
    features: list[str],
    config: RenewableGenerationModelConfig,
) -> list[str]:
    if config.wind_proxy_baseline_columns:
        feature_set = set(features)
        columns = [
            column
            for column in config.wind_proxy_baseline_columns
            if column in feature_set
        ]
        if columns:
            return columns

    patterns = [
        pattern.lower()
        for pattern in config.wind_proxy_baseline_column_patterns
        if pattern
    ]
    columns = [
        feature
        for feature in features
        if any(pattern in feature.lower() for pattern in patterns)
        and "wind" in feature.lower()
        and "_ramp_" not in feature.lower()
        and "_lag_" not in feature.lower()
        and "_lead_" not in feature.lower()
        and "_diff_" not in feature.lower()
        and "_window_" not in feature.lower()
    ]
    if not columns:
        raise ValueError(
            "wind_proxy_residual_baseline could not find proxy baseline columns. "
            "Set wind_proxy_baseline_columns explicitly."
        )
    return columns


def _fit_wind_proxy_residual_baseline(
    target: str,
    y_train: pd.Series,
    X_train_all: pd.DataFrame,
    X_test: pd.DataFrame,
    features: list[str],
    config: RenewableGenerationModelConfig,
    upper_bound: float | None,
) -> tuple[pd.Series, pd.Series] | tuple[None, None]:
    if not config.wind_proxy_residual_baseline or "wind" not in target.lower():
        return None, None

    baseline_columns = _wind_proxy_baseline_feature_columns(features, config)
    baseline_model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        Ridge(alpha=config.wind_proxy_baseline_ridge_alpha),
    )
    baseline_model.fit(X_train_all.loc[y_train.index, baseline_columns], y_train)
    train_values = baseline_model.predict(X_train_all.loc[y_train.index, baseline_columns])
    test_values = baseline_model.predict(X_test.loc[:, baseline_columns])
    if config.wind_proxy_baseline_clip:
        train_values = np.clip(train_values, 0.0, upper_bound)
        test_values = np.clip(test_values, 0.0, upper_bound)

    return (
        pd.Series(train_values, index=y_train.index),
        pd.Series(test_values, index=X_test.index),
    )


def _target_model_values_with_baseline(
    target: str,
    y_train: pd.Series,
    X_train_all: pd.DataFrame,
    config: RenewableGenerationModelConfig,
    installed_capacity_mw: dict[str, float],
    baseline_column: str | None,
) -> pd.Series:
    if baseline_column is None:
        return _model_target_values(target, y_train, config, installed_capacity_mw)

    baseline = X_train_all.loc[y_train.index, baseline_column].astype(float).fillna(0.0)
    capacity = _installed_capacity_for_target(target, installed_capacity_mw)
    if "solar" in target.lower() and capacity is not None:
        baseline = baseline.clip(lower=0.0, upper=capacity)
    return y_train - baseline


def _first_existing_column(candidates: list[str], columns: pd.Index | list[str]) -> str | None:
    column_set = set(columns)
    for candidate in candidates:
        if candidate in column_set:
            return candidate
    return None


def _solar_training_row_mask(
    target: str,
    y_train: pd.Series,
    X_train_all: pd.DataFrame,
    config: RenewableGenerationModelConfig,
    installed_capacity_mw: dict[str, float],
    baseline_column: str | None,
) -> pd.Series:
    mask = pd.Series(True, index=y_train.index)
    if "solar" not in target.lower():
        return mask

    region_key = _solar_region_key_for_target(target)
    geometry_region_key = _solar_geometry_region_key_for_tso_region(region_key)
    clear_sky_col = _first_existing_column(
        [
            f"solar_geom_{region_key}_clear_sky_proxy" if region_key else "",
            f"solar_geom_{geometry_region_key}_clear_sky_proxy" if geometry_region_key else "",
            "solar_geom_clear_sky_proxy_mean",
        ],
        X_train_all.columns,
    )
    elevation_col = _first_existing_column(
        [
            f"solar_geom_{region_key}_elevation_deg" if region_key else "",
            f"solar_geom_{geometry_region_key}_elevation_deg" if geometry_region_key else "",
            "solar_geom_central_elevation_deg",
        ],
        X_train_all.columns,
    )

    if config.solar_training_clear_sky_min is not None:
        if clear_sky_col is None:
            raise ValueError(
                "solar_training_clear_sky_min requires solar geometry features "
                "with a clear-sky proxy column."
            )
        clear_sky = X_train_all.loc[y_train.index, clear_sky_col].astype(float)
        mask &= clear_sky >= config.solar_training_clear_sky_min

    if config.solar_training_elevation_min_deg is not None:
        if elevation_col is None:
            raise ValueError(
                "solar_training_elevation_min_deg requires solar geometry features "
                "with regional elevation columns."
            )
        elevation = X_train_all.loc[y_train.index, elevation_col].astype(float)
        mask &= elevation >= config.solar_training_elevation_min_deg

    if config.solar_suspicious_training_filter:
        if baseline_column is None:
            raise ValueError("solar_suspicious_training_filter requires a target baseline column.")
        baseline = X_train_all.loc[y_train.index, baseline_column].astype(float).fillna(0.0)
        capacity = _installed_capacity_for_target(target, installed_capacity_mw)
        baseline_min = 0.0 if capacity is None else capacity * config.solar_suspicious_baseline_min_capacity_share
        high_potential = baseline >= baseline_min
        if clear_sky_col is not None:
            clear_sky = X_train_all.loc[y_train.index, clear_sky_col].astype(float)
            high_potential &= clear_sky >= config.solar_suspicious_clear_sky_min

        actual_to_baseline = pd.Series(
            np.divide(
                y_train.to_numpy(dtype=float),
                baseline.to_numpy(dtype=float),
                out=np.ones(len(y_train), dtype=float),
                where=baseline.to_numpy(dtype=float) > 1e-6,
            ),
            index=y_train.index,
        )
        suspicious = high_potential & (
            actual_to_baseline < config.solar_suspicious_actual_to_baseline_min
        )
        mask &= ~suspicious

    return mask.fillna(False)


def _solar_cloud_cover_column(
    target: str,
    columns: pd.Index | list[str],
) -> str | None:
    region_key = _solar_region_key_for_target(target)
    if region_key is None:
        return _first_existing_column(
            [
                "om_solar_cloud_cover_cap_weighted",
                "solar_cloud_cover_cap_weighted",
            ],
            columns,
        )

    exact_candidates = [
        f"om_solar_{region_key}_cloud_cover_cap_weighted",
        f"solar_{region_key}_cloud_cover_cap_weighted",
    ]
    exact = _first_existing_column(exact_candidates, columns)
    if exact is not None:
        return exact

    matches = [
        column
        for column in columns
        if (
            f"solar_{region_key}_" in column.lower()
            and "cloud_cover_cap_weighted" in column.lower()
            and "low_cloud" not in column.lower()
            and "mid_cloud" not in column.lower()
            and "high_cloud" not in column.lower()
            and not column.lower().endswith(("_std", "_min", "_max"))
            and "_ramp_" not in column.lower()
        )
    ]
    return matches[0] if matches else None


def _solar_irradiance_ratio(
    target: str,
    X: pd.DataFrame,
) -> pd.Series | None:
    region_key = _solar_region_key_for_target(target)
    geometry_region_key = _solar_geometry_region_key_for_tso_region(region_key)
    irradiance_col = _first_existing_column(
        [
            f"solar_{region_key}_irradiance_cap_weighted_W_m2" if region_key else "",
            "Renewable_Solar_Irradiance_Cap_Weighted_W_m2",
        ],
        X.columns,
    )
    clear_sky_col = _first_existing_column(
        [
            f"solar_geom_{region_key}_clear_sky_proxy" if region_key else "",
            f"solar_geom_{geometry_region_key}_clear_sky_proxy" if geometry_region_key else "",
            "solar_geom_clear_sky_proxy_mean",
        ],
        X.columns,
    )
    if irradiance_col is None or clear_sky_col is None:
        return None

    irradiance = X[irradiance_col].astype(float).clip(lower=0.0)
    clear_sky = 1000.0 * X[clear_sky_col].astype(float).clip(lower=0.0)
    ratio = np.divide(
        irradiance.to_numpy(dtype=float),
        clear_sky.to_numpy(dtype=float),
        out=np.full(len(X), np.nan, dtype=float),
        where=clear_sky.to_numpy(dtype=float) > 1e-6,
    )
    return pd.Series(ratio, index=X.index).clip(lower=0.0, upper=2.0)


def _solar_weather_regime_labels(
    target: str,
    X: pd.DataFrame,
    config: RenewableGenerationModelConfig,
) -> pd.Series | None:
    """Classify forecast rows into clear, mixed, and overcast solar weather regimes."""
    if not config.solar_weather_regime_split or "solar" not in target.lower():
        return None

    cloud_col = _solar_cloud_cover_column(target, X.columns)
    cloud = None if cloud_col is None else X[cloud_col].astype(float)
    irradiance_ratio = _solar_irradiance_ratio(target, X)
    if cloud is None and irradiance_ratio is None:
        return None

    labels = pd.Series("mixed", index=X.index, dtype="object")
    clear_mask = pd.Series(True, index=X.index)
    overcast_mask = pd.Series(False, index=X.index)

    if cloud is not None:
        clear_mask &= cloud <= config.solar_weather_regime_clear_cloud_max
        overcast_mask |= cloud >= config.solar_weather_regime_overcast_cloud_min
    if irradiance_ratio is not None:
        clear_mask &= irradiance_ratio >= config.solar_weather_regime_clear_irradiance_ratio_min
        overcast_mask |= (
            irradiance_ratio <= config.solar_weather_regime_overcast_irradiance_ratio_max
        )

    labels.loc[overcast_mask.fillna(False)] = "overcast"
    labels.loc[clear_mask.fillna(False)] = "clear"
    return labels


def _predict_solar_weather_regime_split(
    target: str,
    X_train: pd.DataFrame,
    X_train_all: pd.DataFrame,
    y_model: pd.Series,
    X_test: pd.DataFrame,
    target_features: list[str],
    config: RenewableGenerationModelConfig,
    train_end: pd.Timestamp,
) -> np.ndarray | None:
    train_regime = _solar_weather_regime_labels(target, X_train_all.loc[y_model.index], config)
    test_regime = _solar_weather_regime_labels(target, X_test, config)
    if train_regime is None or test_regime is None:
        return None

    predictions = pd.Series(np.nan, index=X_test.index, dtype=float)
    global_cache: tuple[list[str], Any] | None = None

    def fit_predict(train_index: pd.Index, test_index: pd.Index) -> np.ndarray:
        nonlocal global_cache
        is_global = train_index.equals(y_model.index)
        if is_global and global_cache is not None:
            selected, model = global_cache
        else:
            X_subset = X_train.loc[train_index, target_features]
            y_subset = y_model.loc[train_index]
            selected = _select_target_features(X_subset, y_subset, config)
            model = _make_model(config, target=target)
            recency_half_life_days = _model_param(
                config,
                _target_model_overrides(config, target),
                "training_recency_half_life_days",
            )
            sample_weight = _recency_sample_weight(
                y_subset.index,
                reference_time=train_end,
                half_life_days=recency_half_life_days,
            )
            _fit_model_with_optional_sample_weight(
                model,
                X_subset[selected],
                y_subset,
                sample_weight,
            )
            if is_global:
                global_cache = (selected, model)
        return model.predict(X_test.loc[test_index, selected])

    for regime in ("clear", "mixed", "overcast"):
        test_index = test_regime.index[test_regime == regime]
        if len(test_index) == 0:
            continue
        train_index = train_regime.index[train_regime == regime]
        if len(train_index) < config.solar_weather_regime_min_train_rows:
            train_index = y_model.index
        predictions.loc[test_index] = fit_predict(train_index, test_index)

    if predictions.isna().any():
        missing_index = predictions.index[predictions.isna()]
        predictions.loc[missing_index] = fit_predict(y_model.index, missing_index)
    return predictions.to_numpy(dtype=float)


def _wind_training_row_mask(
    target: str,
    y_train: pd.Series,
    X_train_all: pd.DataFrame,
    config: RenewableGenerationModelConfig,
) -> pd.Series:
    mask = pd.Series(True, index=y_train.index)
    if "wind" not in target.lower() or not config.wind_suspicious_training_filter:
        return mask

    configured_columns = [
        column
        for column in config.wind_suspicious_proxy_columns
        if column in X_train_all.columns
    ]
    if configured_columns:
        proxy_columns = configured_columns
    else:
        patterns = [
            pattern.lower()
            for pattern in config.wind_suspicious_proxy_column_patterns
            if pattern
        ]
        proxy_columns = [
            column
            for column in X_train_all.select_dtypes(include="number").columns
            if any(pattern in column.lower() for pattern in patterns)
            and "_ramp_" not in column.lower()
            and "_lag_" not in column.lower()
            and "_lead_" not in column.lower()
            and "_diff_" not in column.lower()
        ]

    if not proxy_columns:
        raise ValueError(
            "wind_suspicious_training_filter could not find wind proxy columns. "
            "Set wind_suspicious_proxy_columns explicitly."
        )

    proxies = X_train_all.loc[y_train.index, proxy_columns].astype(float).clip(lower=0.0)
    if config.wind_suspicious_proxy_aggregation == "max":
        potential = proxies.max(axis=1)
    else:
        potential = proxies.mean(axis=1)

    potential = potential.replace([np.inf, -np.inf], np.nan)
    valid_potential = potential.dropna()
    if valid_potential.empty:
        return mask

    potential_threshold = max(
        float(valid_potential.quantile(config.wind_suspicious_potential_quantile)),
        float(config.wind_suspicious_min_potential_mw),
    )
    high_potential = potential >= potential_threshold
    actual_to_proxy = pd.Series(
        np.divide(
            y_train.to_numpy(dtype=float),
            potential.to_numpy(dtype=float),
            out=np.ones(len(y_train), dtype=float),
            where=potential.to_numpy(dtype=float) > 1e-6,
        ),
        index=y_train.index,
    )
    suspicious = high_potential & (actual_to_proxy < config.wind_suspicious_actual_to_proxy_min)
    return (mask & ~suspicious).fillna(False)


def _model_predictions_with_baseline_to_mw(
    target: str,
    predictions: np.ndarray,
    X_test: pd.DataFrame,
    config: RenewableGenerationModelConfig,
    installed_capacity_mw: dict[str, float],
    baseline_column: str | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    if baseline_column is None:
        return (
            _model_predictions_to_mw(target, predictions, config, installed_capacity_mw),
            None,
        )

    baseline = X_test[baseline_column].astype(float).fillna(0.0).to_numpy(dtype=float)
    capacity = _installed_capacity_for_target(target, installed_capacity_mw)
    if "solar" in target.lower() and capacity is not None:
        baseline = np.clip(baseline, 0.0, capacity)
    return baseline + predictions, baseline


def _target_upper_bound(
    target: str,
    y_train: pd.Series,
    config: RenewableGenerationModelConfig,
) -> float | None:
    for key in (target, _target_output_name(target)):
        if key in config.target_capacity_caps_mw:
            cap = config.target_capacity_caps_mw[key]
            return float(cap) if pd.notna(cap) and cap > 0 else None

    if not config.clip_predictions_to_training_target_range:
        return None

    valid = y_train.dropna()
    if valid.empty:
        return None
    if config.prediction_upper_quantile >= 1.0:
        return float(valid.max())
    return float(valid.quantile(config.prediction_upper_quantile))


def _select_target_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: RenewableGenerationModelConfig,
) -> list[str]:
    max_features = config.max_features_per_target
    if max_features is None or max_features <= 0 or X_train.shape[1] <= max_features:
        return list(X_train.columns)

    varying_columns = [
        column
        for column in X_train.columns
        if X_train[column].nunique(dropna=True) > 1
    ]
    if not varying_columns or y_train.nunique(dropna=True) <= 1:
        return list(X_train.columns[:max_features])

    scores = (
        X_train.loc[:, varying_columns].corrwith(y_train)
        .abs()
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .sort_values(ascending=False)
    )
    selected = list(scores.head(max_features).index)
    if not selected:
        return list(X_train.columns)
    return selected


def _prediction_output_upper_bound(
    target: str,
    y_train: pd.Series,
    config: RenewableGenerationModelConfig,
    installed_capacity_mw: dict[str, float],
) -> float | None:
    capacity = _installed_capacity_for_target(target, installed_capacity_mw)
    if config.target_transform == "capacity_factor":
        if capacity is None:
            raise ValueError(
                f"target_transform='capacity_factor' requires installed capacity for {target!r}. "
                "Provide target_installed_capacity_mw or ensure the renewable proxy metadata exists."
            )
        return capacity
    if "solar" in target.lower() and capacity is not None:
        return capacity
    return _target_upper_bound(target, y_train, config)


def _model_target_values(
    target: str,
    y_train: pd.Series,
    config: RenewableGenerationModelConfig,
    installed_capacity_mw: dict[str, float],
) -> pd.Series:
    if config.target_transform == "mw":
        return y_train
    if config.target_transform != "capacity_factor":
        raise ValueError(f"Unsupported target_transform: {config.target_transform!r}")

    capacity = _installed_capacity_for_target(target, installed_capacity_mw)
    if capacity is None:
        raise ValueError(
            f"target_transform='capacity_factor' requires installed capacity for {target!r}. "
            "Provide target_installed_capacity_mw or ensure the renewable proxy metadata exists."
        )
    return (y_train / capacity).clip(0.0, 1.0)


def _model_predictions_to_mw(
    target: str,
    predictions: np.ndarray,
    config: RenewableGenerationModelConfig,
    installed_capacity_mw: dict[str, float],
) -> np.ndarray:
    if config.target_transform == "mw":
        return predictions
    capacity = _installed_capacity_for_target(target, installed_capacity_mw)
    if capacity is None:
        raise ValueError(
            f"target_transform='capacity_factor' requires installed capacity for {target!r}. "
            "Provide target_installed_capacity_mw or ensure the renewable proxy metadata exists."
        )
    return np.clip(predictions, 0.0, 1.0) * capacity


def _bias_group_values(index: pd.DatetimeIndex, group: str) -> np.ndarray:
    if group == "global":
        return np.repeat("global", len(index))
    if group == "hour":
        return index.hour
    if group == "mtu":
        return index.hour * 4 + index.minute // 15
    raise ValueError(f"Unsupported rolling_bias_correction_group: {group!r}")


def _apply_rolling_bias_correction(
    day_forecast: pd.DataFrame,
    forecast_blocks: list[pd.DataFrame],
    *,
    day: pd.Timestamp,
    pred_col: str,
    true_col: str,
    upper_bound: float | None,
    config: RenewableGenerationModelConfig,
    window_days: int | None = None,
    group: str | None = None,
    min_observations: int | None = None,
    shrinkage: float | None = None,
) -> None:
    window_days = int(config.rolling_bias_correction_window_days if window_days is None else window_days)
    group = config.rolling_bias_correction_group if group is None else group
    min_observations = (
        config.rolling_bias_correction_min_observations if min_observations is None else int(min_observations)
    )
    shrinkage = config.rolling_bias_correction_shrinkage if shrinkage is None else float(shrinkage)

    if window_days <= 0 or pred_col not in day_forecast.columns or true_col not in day_forecast.columns:
        return

    history_blocks = [
        block[[pred_col, true_col]]
        for block in forecast_blocks
        if pred_col in block.columns and true_col in block.columns
    ]
    if not history_blocks:
        return

    window_start = day - pd.Timedelta(days=window_days)
    available_until = _training_target_cutoff(day, config)
    history = pd.concat(history_blocks).sort_index()
    history = history.loc[
        (history.index >= window_start) & (history.index <= available_until),
        [pred_col, true_col],
    ].dropna()
    if len(history) < min_observations:
        return

    errors = history[pred_col] - history[true_col]
    global_bias = float(errors.mean())

    if group == "global":
        correction = np.repeat(global_bias, len(day_forecast))
    else:
        history_keys = _bias_group_values(history.index, group)
        bias_by_group = errors.groupby(history_keys).mean()
        count_by_group = errors.groupby(history_keys).count()
        bias_by_group = bias_by_group[count_by_group >= min_observations]
        current_keys = pd.Series(_bias_group_values(day_forecast.index, group), index=day_forecast.index)
        correction_series = current_keys.map(bias_by_group).fillna(global_bias)
        correction = correction_series.to_numpy(dtype=float)

    corrected = day_forecast[pred_col].to_numpy(dtype=float) - (shrinkage * correction)
    day_forecast[pred_col] = np.clip(corrected, 0.0, upper_bound)


def rolling_renewable_generation_forecast(
    dataset: pd.DataFrame,
    config: RenewableGenerationModelConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train rolling timestamp-level models and predict complete forecast days."""
    target_columns = list(config.target_columns)
    missing_targets = [target for target in target_columns if target not in dataset.columns]
    if missing_targets:
        raise ValueError(f"Dataset is missing configured target columns: {missing_targets}")

    features = _feature_columns(dataset, target_columns)
    if not features:
        raise ValueError("No renewable generation model feature columns were built.")

    features_by_target = {
        target: _target_candidate_features(features, target, config)
        for target in target_columns
    }
    baseline_columns_by_target = {
        target: _target_baseline_column(target, config, features)
        for target in target_columns
    }
    feature_allowlists_by_target = {
        target: _feature_allowlist_for_target(target, config)
        for target in target_columns
    }
    model_types_by_target = {
        target: _model_param(config, _target_model_overrides(config, target), "model_type")
        for target in target_columns
    }
    training_required = any(model_type != "baseline" for model_type in model_types_by_target.values())

    test_start = _as_local_day(config.test_start, config.target_tz)
    test_end = _as_local_day(config.test_end, config.target_tz)
    forecast_days = pd.date_range(start=test_start, end=test_end, freq="D", tz=config.target_tz)

    forecast_blocks = []
    runtime_rows: list[dict[str, Any]] = []
    min_train_rows = config.min_train_days * 96
    installed_capacity_mw = _load_installed_capacity_denominators(config)

    for day in forecast_days:
        day_start = time.perf_counter()
        train_start = day - pd.Timedelta(days=config.train_days_rolling)
        train_end = _training_target_cutoff(day, config)
        test_end_ts = day + pd.Timedelta(days=1) - pd.Timedelta(minutes=15)

        train_mask = (dataset.index >= train_start) & (dataset.index <= train_end)
        test_mask = (dataset.index >= day) & (dataset.index <= test_end_ts)
        X_train_all = dataset.loc[train_mask, features]
        X_test = dataset.loc[test_mask, features]

        if len(X_test) == 0 or (training_required and len(X_train_all) < min_train_rows):
            continue

        day_forecast = pd.DataFrame(index=X_test.index)
        for target in target_columns:
            target_features = features_by_target[target]
            baseline_column = baseline_columns_by_target[target]
            feature_allowlist = feature_allowlists_by_target[target]
            target_model_type = model_types_by_target[target]
            if feature_allowlist is not None:
                candidate_set = set(target_features)
                target_features = [
                    feature
                    for feature in feature_allowlist
                    if feature in candidate_set
                ]
                if not target_features:
                    raise ValueError(
                        f"Feature allowlist for {target!r} did not match any candidate features."
                    )
            output_name = _target_output_name(target)
            pred_col = f"{output_name}_Model_MW"
            true_col = f"{output_name}_Actual_MW"

            if target_model_type == "baseline":
                if baseline_column is None and config.wind_proxy_residual_baseline and "wind" in target.lower():
                    train_target_mask = dataset.loc[train_mask, target].notna()
                    y_train = dataset.loc[train_mask, target].loc[train_target_mask]
                    X_train_for_baseline = X_train_all.loc[y_train.index]
                    train_row_mask = _wind_training_row_mask(
                        target,
                        y_train,
                        X_train_for_baseline,
                        config,
                    )
                    y_train = y_train.loc[train_row_mask]
                    if len(y_train) < min_train_rows:
                        continue
                    upper_bound = _installed_capacity_for_target(target, installed_capacity_mw)
                    _, dynamic_baseline_test = _fit_wind_proxy_residual_baseline(
                        target,
                        y_train,
                        X_train_all,
                        X_test,
                        features,
                        config,
                        upper_bound,
                    )
                    if dynamic_baseline_test is None:
                        raise ValueError(
                            "model_type='baseline' with wind_proxy_residual_baseline=True "
                            f"could not build a wind proxy baseline for {target!r}."
                        )
                    baseline_values = dynamic_baseline_test.astype(float).fillna(0.0).to_numpy(dtype=float)
                elif baseline_column is None:
                    raise ValueError(
                        f"model_type='baseline' requires a target baseline column for {target!r}."
                    )
                else:
                    upper_bound = _installed_capacity_for_target(target, installed_capacity_mw)
                    baseline_values = X_test[baseline_column].astype(float).fillna(0.0).to_numpy(dtype=float)
                day_forecast[f"{output_name}_Baseline_MW"] = np.clip(baseline_values, 0.0, upper_bound)
                day_forecast[pred_col] = np.clip(baseline_values, 0.0, upper_bound)
                if target in dataset.columns:
                    day_forecast[true_col] = dataset.loc[test_mask, target]
                _apply_rolling_bias_correction(
                    day_forecast,
                    forecast_blocks,
                    day=day,
                    pred_col=pred_col,
                    true_col=true_col,
                    upper_bound=upper_bound,
                    config=config,
                )
                continue

            train_target_mask = dataset.loc[train_mask, target].notna()
            y_train = dataset.loc[train_mask, target].loc[train_target_mask]
            train_row_mask = _solar_training_row_mask(
                target,
                y_train,
                X_train_all,
                config,
                installed_capacity_mw,
                baseline_column,
            )
            train_row_mask &= _wind_training_row_mask(
                target,
                y_train,
                X_train_all,
                config,
            )
            y_train = y_train.loc[train_row_mask]
            X_train = X_train_all.loc[y_train.index, target_features]

            if len(X_train) < min_train_rows:
                continue

            upper_bound = _prediction_output_upper_bound(
                target,
                y_train,
                config,
                installed_capacity_mw,
            )
            dynamic_baseline_train, dynamic_baseline_test = _fit_wind_proxy_residual_baseline(
                target,
                y_train,
                X_train_all,
                X_test,
                features,
                config,
                upper_bound,
            )
            if dynamic_baseline_train is not None:
                y_model = y_train - dynamic_baseline_train
            else:
                y_model = _target_model_values_with_baseline(
                    target,
                    y_train,
                    X_train_all,
                    config,
                    installed_capacity_mw,
                    baseline_column,
                )

            output_name = _target_output_name(target)
            pred_col = f"{output_name}_Model_MW"
            true_col = f"{output_name}_Actual_MW"
            model_prediction = _predict_solar_weather_regime_split(
                target,
                X_train,
                X_train_all,
                y_model,
                X_test,
                target_features,
                config,
                train_end,
            )
            if model_prediction is None:
                selected_features = _select_target_features(X_train, y_model, config)
                model = _make_model(config, target=target)
                recency_half_life_days = _model_param(
                    config,
                    _target_model_overrides(config, target),
                    "training_recency_half_life_days",
                )
                sample_weight = _recency_sample_weight(
                    y_model.index,
                    reference_time=train_end,
                    half_life_days=recency_half_life_days,
                )
                _fit_model_with_optional_sample_weight(
                    model,
                    X_train[selected_features],
                    y_model,
                    sample_weight,
                )
                model_prediction = model.predict(X_test.loc[:, selected_features])
            if dynamic_baseline_test is not None:
                baseline_values = dynamic_baseline_test.to_numpy(dtype=float)
                prediction_mw = baseline_values + model_prediction
            else:
                prediction_mw, baseline_values = _model_predictions_with_baseline_to_mw(
                    target,
                    model_prediction,
                    X_test,
                    config,
                    installed_capacity_mw,
                    baseline_column,
                )
            if baseline_values is not None:
                day_forecast[f"{output_name}_Baseline_MW"] = np.clip(baseline_values, 0.0, upper_bound)
            day_forecast[pred_col] = np.clip(
                prediction_mw,
                0.0,
                upper_bound,
            )
            if target in dataset.columns:
                day_forecast[true_col] = dataset.loc[test_mask, target]

            _apply_rolling_bias_correction(
                day_forecast,
                forecast_blocks,
                day=day,
                pred_col=pred_col,
                true_col=true_col,
                upper_bound=upper_bound,
                config=config,
            )

        if {"Wind_Onshore_Model_MW", "Wind_Offshore_Model_MW"}.issubset(day_forecast.columns):
            day_forecast["Wind_Total_Model_MW"] = (
                day_forecast["Wind_Onshore_Model_MW"] + day_forecast["Wind_Offshore_Model_MW"]
            )
        if "Wind_Total_Model_MW" in day_forecast.columns and "Wind_Total_Actual_MW" in dataset.columns:
            day_forecast["Wind_Total_Actual_MW"] = dataset.loc[test_mask, "Wind_Total_Actual_MW"]

        solar_control_targets = [
            target
            for target in config.solar_control_area_targets
            if target in target_columns
        ]
        if (
            config.solar_twilight_clear_sky_threshold is not None
            and "solar_geom_clear_sky_proxy_mean" in X_test.columns
        ):
            twilight_mask = (
                X_test["solar_geom_clear_sky_proxy_mean"] < config.solar_twilight_clear_sky_threshold
            )
            if twilight_mask.any():
                for solar_target in solar_control_targets:
                    col = f"{_target_output_name(solar_target)}_Model_MW"
                    if col in day_forecast.columns:
                        day_forecast.loc[twilight_mask, col] = 0.0
        solar_control_model_columns = [
            f"{_target_output_name(target)}_Model_MW"
            for target in solar_control_targets
        ]
        if solar_control_model_columns and set(solar_control_model_columns).issubset(day_forecast.columns):
            day_forecast["Solar_Model_MW"] = day_forecast[solar_control_model_columns].sum(axis=1)
            if "Solar_Actual_MW" in dataset.columns:
                day_forecast["Solar_Actual_MW"] = dataset.loc[test_mask, "Solar_Actual_MW"]
            else:
                solar_control_actual_columns = [
                    f"{_target_output_name(target)}_Actual_MW"
                    for target in solar_control_targets
                ]
                if set(solar_control_actual_columns).issubset(day_forecast.columns):
                    day_forecast["Solar_Actual_MW"] = day_forecast[solar_control_actual_columns].sum(axis=1)
            _apply_rolling_bias_correction(
                day_forecast,
                forecast_blocks,
                day=day,
                pred_col="Solar_Model_MW",
                true_col="Solar_Actual_MW",
                upper_bound=_installed_capacity_for_target("Solar_Actual_MW", installed_capacity_mw),
                config=config,
                window_days=config.solar_total_bias_correction_window_days,
                group=config.solar_total_bias_correction_group,
                min_observations=config.solar_total_bias_correction_min_observations,
                shrinkage=config.solar_total_bias_correction_shrinkage,
            )

        if {"Solar_Model_MW", "Wind_Total_Model_MW"}.issubset(day_forecast.columns):
            day_forecast["Renewable_Total_Model_MW"] = (
                day_forecast["Solar_Model_MW"] + day_forecast["Wind_Total_Model_MW"]
            )
        if "Renewable_Total_Actual_MW" in dataset.columns:
            day_forecast["Renewable_Total_Actual_MW"] = dataset.loc[test_mask, "Renewable_Total_Actual_MW"]

        if not day_forecast.empty:
            forecast_blocks.append(day_forecast)
        runtime_rows.append(
            {
                "date": day,
                "train_rows": len(X_train_all),
                "test_rows": len(X_test),
                "n_features": len(features),
                "model_type": config.model_type,
                "runtime_seconds": time.perf_counter() - day_start,
            }
        )
        print(f"  {day.date()}  {config.model_type}  {runtime_rows[-1]['runtime_seconds']:.1f}s")

    if not forecast_blocks:
        raise ValueError("No renewable generation forecasts were produced.")

    forecast = pd.concat(forecast_blocks).sort_index()
    forecast.index.name = "timestamp"
    runtime = pd.DataFrame(runtime_rows)
    return forecast, runtime


def _capacity_aliases(target: str) -> tuple[str, ...]:
    return (
        *_target_aliases(target),
        target,
        target.removesuffix("_Model_MW"),
        target.removesuffix("_Actual_MW"),
        target.replace("_Model_MW", "_Actual_MW"),
        target.replace("_Actual_MW", "_Model_MW"),
    )


def _installed_capacity_for_target(target: str, installed_capacity_mw: dict[str, float]) -> float | None:
    for key in _capacity_aliases(target):
        if key in installed_capacity_mw:
            value = installed_capacity_mw[key]
            return float(value) if pd.notna(value) and value > 0 else None
    return None


def _load_installed_capacity_denominators(config: RenewableGenerationModelConfig) -> dict[str, float]:
    denominators: dict[str, float] = {
        str(key): float(value)
        for key, value in {**config.target_capacity_caps_mw, **config.target_installed_capacity_mw}.items()
        if pd.notna(value) and float(value) > 0
    }
    if denominators:
        return denominators

    metadata_path = config.renewable_proxy_file.with_suffix(".json")
    if not metadata_path.exists():
        return {}

    try:
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}

    summary = metadata.get("capacity_summary_mw")
    if not isinstance(summary, list):
        return {}

    wind = 0.0
    wind_onshore = 0.0
    wind_offshore = 0.0
    solar = 0.0
    solar_region_aliases = {
        "solar_50hertz": "Solar_50Hertz",
        "solar_amprion": "Solar_Amprion",
        "solar_tennet": "Solar_TenneT",
        "solar_transnetbw": "Solar_TransnetBW",
    }
    for row in summary:
        if not isinstance(row, dict):
            continue
        technology = str(row.get("technology_group", "")).lower()
        region = str(row.get("region", "")).lower()
        capacity = float(row.get("capacity_mw", 0.0) or 0.0)
        if technology == "wind_onshore":
            wind_onshore += capacity
            wind += capacity
        elif technology == "wind_offshore":
            wind_offshore += capacity
            wind += capacity
        elif technology == "solar":
            solar += capacity
            if region in solar_region_aliases:
                denominators[solar_region_aliases[region]] = capacity

    if wind_onshore > 0:
        denominators["Wind_Onshore"] = wind_onshore
    if wind_offshore > 0:
        denominators["Wind_Offshore"] = wind_offshore
    if wind > 0:
        denominators["Wind_Total"] = wind
    if solar > 0:
        denominators["Solar"] = solar
    if wind + solar > 0:
        denominators["Renewable_Total"] = wind + solar
    return denominators


def _period_metrics(
    forecast: pd.DataFrame,
    pred_col: str,
    true_col: str,
    period: str,
    installed_capacity_mw: dict[str, float],
) -> dict[str, Any]:
    stats = point_error_stats(forecast, pred_col, true_col)
    rmse = float(stats["rmse"])
    capacity = _installed_capacity_for_target(pred_col, installed_capacity_mw)
    nrmse = float(rmse / capacity) if capacity is not None else np.nan
    return {
        "target": pred_col.removesuffix("_Model_MW"),
        "period": period,
        "mae": stats["mae"],
        "mse": stats["mse"],
        "rmse": rmse,
        "nrmse": nrmse,
        "nrmse_pct": float(nrmse * 100.0) if pd.notna(nrmse) else np.nan,
        "r2": stats["r2"],
        "bias": stats["bias"],
        "installed_capacity_mw": float(capacity) if capacity is not None else np.nan,
        "n_obs": stats["n_obs"],
    }


def evaluate_renewable_generation_forecast(
    forecast: pd.DataFrame,
    installed_capacity_mw: dict[str, float] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    pairs = []
    installed_capacity_mw = installed_capacity_mw or {}
    for pred_col in [column for column in forecast.columns if column.endswith("_Model_MW")]:
        true_col = pred_col.replace("_Model_MW", "_Actual_MW")
        if true_col in forecast.columns:
            pairs.append((pred_col, true_col))

    for pred_col, true_col in pairs:
        rows.append(_period_metrics(forecast, pred_col, true_col, "full", installed_capacity_mw))
        month_index = forecast.index.tz_localize(None).to_period("M")
        for month, group in forecast.groupby(month_index):
            rows.append(_period_metrics(group, pred_col, true_col, str(month), installed_capacity_mw))

    return pd.DataFrame(rows)


def _load_forecast_csv(path: Path, target_tz: str) -> pd.DataFrame:
    forecast = pd.read_csv(path, index_col=0)
    raw_index = forecast.index.astype(str)
    has_tz_offsets = pd.Series(raw_index).str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$", regex=True).any()
    if has_tz_offsets:
        forecast.index = pd.to_datetime(forecast.index, utc=True).tz_convert(target_tz)
    else:
        forecast.index = pd.to_datetime(forecast.index)
        forecast.index = forecast.index.tz_localize(target_tz)
    forecast = forecast.sort_index()
    forecast = forecast.loc[~forecast.index.duplicated(keep="last")]
    forecast.index.name = "timestamp"
    return forecast


def _forecast_pairs(forecast: pd.DataFrame) -> list[tuple[str, str]]:
    return [
        (pred_col, pred_col.replace("_Model_MW", "_Actual_MW"))
        for pred_col in forecast.columns
        if pred_col.endswith("_Model_MW") and pred_col.replace("_Model_MW", "_Actual_MW") in forecast.columns
    ]


def _target_name_from_prediction(pred_col: str) -> str:
    return pred_col.removesuffix("_Model_MW")


def _normalise_stacking_target(target: str) -> str:
    return target.removesuffix("_Actual_MW").removesuffix("_Model_MW")


def _stacking_provider_prediction_col(provider: str, target: str) -> str:
    return f"{provider}__{target}_Model_MW"


def _stacking_target_prediction_col(target: str) -> str:
    return f"{target}_Model_MW"


def _stacking_target_actual_col(target: str) -> str:
    return f"{target}_Actual_MW"


def _provider_mean_prediction(frame: pd.DataFrame, provider_cols: list[str]) -> pd.Series:
    if not provider_cols:
        return pd.Series(np.nan, index=frame.index)
    return frame[provider_cols].mean(axis=1, skipna=True)


def _load_stacking_forecast_frame(config: RenewableGenerationPostprocessConfig) -> pd.DataFrame:
    """Load provider forecast files into one frame for rolling forecast stacking."""
    if not config.stacking_forecast_files:
        raise ValueError("include_forecast_stacking requires at least one stacking_forecast_files entry.")

    provider_frames: dict[str, pd.DataFrame] = {}
    index: pd.DatetimeIndex | None = None
    for provider, path in config.stacking_forecast_files.items():
        if not path.exists():
            raise FileNotFoundError(f"Stacking forecast file not found for provider {provider!r}: {path}")
        frame = _load_forecast_csv(path, config.target_tz)
        provider_frames[provider] = frame
        index = frame.index if index is None else index.union(frame.index)

    if index is None:
        raise ValueError("No provider forecast rows were available for stacking.")

    targets = [_normalise_stacking_target(target) for target in config.stacking_targets]
    combined = pd.DataFrame(index=index.sort_values())

    for target in targets:
        pred_col = _stacking_target_prediction_col(target)
        actual_col = _stacking_target_actual_col(target)
        for provider, frame in provider_frames.items():
            if pred_col in frame.columns:
                combined[_stacking_provider_prediction_col(provider, target)] = frame[pred_col].reindex(combined.index)
            if actual_col in frame.columns:
                if actual_col not in combined.columns:
                    combined[actual_col] = frame[actual_col].reindex(combined.index)
                else:
                    combined[actual_col] = combined[actual_col].combine_first(frame[actual_col].reindex(combined.index))

    for actual_col in ["Wind_Total_Actual_MW", "Renewable_Total_Actual_MW", "Solar_Actual_MW"]:
        for frame in provider_frames.values():
            if actual_col not in frame.columns:
                continue
            if actual_col not in combined.columns:
                combined[actual_col] = frame[actual_col].reindex(combined.index)
            else:
                combined[actual_col] = combined[actual_col].combine_first(frame[actual_col].reindex(combined.index))

    combined.index.name = "timestamp"
    return combined.sort_index()


def _fit_predict_stacked_target(
    history: pd.DataFrame,
    current: pd.DataFrame,
    *,
    target: str,
    provider_cols: list[str],
    config: RenewableGenerationPostprocessConfig,
) -> pd.Series:
    """Fit the provider-level meta-model for one target and one forecast day."""
    if not provider_cols:
        return pd.Series(np.nan, index=current.index)

    fallback = _provider_mean_prediction(current, provider_cols)
    if config.stacking_model_type == "simple_average":
        return fallback

    actual_col = _stacking_target_actual_col(target)
    if actual_col not in history.columns:
        return fallback

    train = history[[*provider_cols, actual_col]].replace([np.inf, -np.inf], np.nan)
    train = train.loc[train[actual_col].notna()]
    train = train.loc[train[provider_cols].notna().any(axis=1)]
    usable_provider_cols = [column for column in provider_cols if train[column].notna().any()]
    if len(train) < config.stacking_min_observations or not usable_provider_cols:
        return fallback

    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        Ridge(alpha=config.stacking_ridge_alpha),
    )
    model.fit(train[usable_provider_cols], train[actual_col].astype(float))
    predicted = pd.Series(
        model.predict(current[usable_provider_cols]),
        index=current.index,
    )
    predicted = predicted.where(current[usable_provider_cols].notna().any(axis=1), fallback)
    return predicted


def build_stacked_renewable_generation_forecast(
    config: RenewableGenerationPostprocessConfig,
) -> pd.DataFrame:
    """Build a leakage-safe rolling stack over first-stage provider forecasts."""
    base = _load_stacking_forecast_frame(config)
    targets = [_normalise_stacking_target(target) for target in config.stacking_targets]
    providers = list(config.stacking_forecast_files)
    stacked_blocks: list[pd.DataFrame] = []

    test_days = pd.date_range(
        base.index.min().normalize(),
        base.index.max().normalize(),
        freq="D",
        tz=config.target_tz,
    )

    for day in test_days:
        day_end = day + pd.Timedelta(days=1) - pd.Timedelta(minutes=15)
        current_base = base.loc[(base.index >= day) & (base.index <= day_end)].copy()
        if current_base.empty:
            continue

        if config.stacking_window_days > 0:
            window_start = day - pd.Timedelta(days=config.stacking_window_days)
        else:
            window_start = base.index.min()
        available_until = _target_availability_cutoff(
            day,
            lag_days=config.target_availability_lag_days,
            cutoff_hour=config.target_availability_cutoff_hour,
            cutoff_minute=config.target_availability_cutoff_minute,
        )
        history = base.loc[(base.index >= window_start) & (base.index <= available_until)]

        stacked_day = pd.DataFrame(index=current_base.index)
        for target in targets:
            provider_cols = [
                _stacking_provider_prediction_col(provider, target)
                for provider in providers
                if _stacking_provider_prediction_col(provider, target) in current_base.columns
            ]
            pred_col = _stacking_target_prediction_col(target)
            actual_col = _stacking_target_actual_col(target)
            stacked_day[pred_col] = _fit_predict_stacked_target(
                history,
                current_base,
                target=target,
                provider_cols=provider_cols,
                config=config,
            )
            upper_bound = _postprocess_upper_bound(pred_col, config)
            stacked_day[pred_col] = np.clip(stacked_day[pred_col].to_numpy(dtype=float), 0.0, upper_bound)
            if actual_col in current_base.columns:
                stacked_day[actual_col] = current_base[actual_col]

        if {"Wind_Onshore_Model_MW", "Wind_Offshore_Model_MW"}.issubset(stacked_day.columns):
            stacked_day["Wind_Total_Model_MW"] = (
                stacked_day["Wind_Onshore_Model_MW"] + stacked_day["Wind_Offshore_Model_MW"]
            )
        if {"Wind_Onshore_Actual_MW", "Wind_Offshore_Actual_MW"}.issubset(stacked_day.columns):
            stacked_day["Wind_Total_Actual_MW"] = (
                stacked_day["Wind_Onshore_Actual_MW"] + stacked_day["Wind_Offshore_Actual_MW"]
            )
        elif "Wind_Total_Actual_MW" in current_base.columns:
            stacked_day["Wind_Total_Actual_MW"] = current_base["Wind_Total_Actual_MW"]

        if {"Solar_Model_MW", "Wind_Total_Model_MW"}.issubset(stacked_day.columns):
            stacked_day["Renewable_Total_Model_MW"] = (
                stacked_day["Solar_Model_MW"] + stacked_day["Wind_Total_Model_MW"]
            )
        if "Renewable_Total_Actual_MW" in current_base.columns:
            stacked_day["Renewable_Total_Actual_MW"] = current_base["Renewable_Total_Actual_MW"]

        stacked_blocks.append(stacked_day)

    if not stacked_blocks:
        raise ValueError("No forecast rows were available for renewable generation stacking.")

    stacked = pd.concat(stacked_blocks).sort_index()
    stacked.index.name = "timestamp"
    return stacked


def _sequence_day_frame(
    forecast: pd.DataFrame,
    day: pd.Timestamp,
    *,
    n_steps: int,
) -> pd.DataFrame | None:
    day_end = day + pd.Timedelta(days=1) - pd.Timedelta(minutes=15)
    frame = forecast.loc[(forecast.index >= day) & (forecast.index <= day_end)].copy()
    if len(frame) != n_steps:
        return None
    return frame


def _sequence_target_name(target: str) -> str:
    return target.removesuffix("_Actual_MW").removesuffix("_Model_MW")


def _sequence_feature_vector(
    day_frame: pd.DataFrame,
    *,
    feature_targets: list[str],
    n_steps: int,
) -> dict[str, float] | None:
    vector: dict[str, float] = {}
    any_target = False
    for target in feature_targets:
        target_name = _sequence_target_name(target)
        pred_col = f"{target_name}_Model_MW"
        if pred_col not in day_frame.columns:
            continue
        values = pd.to_numeric(day_frame[pred_col], errors="coerce").to_numpy(dtype=float)
        if len(values) != n_steps:
            continue
        any_target = True
        ramps = np.diff(values, prepend=values[0])
        for step, value in enumerate(values):
            vector[f"{target_name}_pred_t{step:03d}"] = value
        for step, value in enumerate(ramps):
            vector[f"{target_name}_ramp_t{step:03d}"] = value
        finite = values[np.isfinite(values)]
        if finite.size:
            vector[f"{target_name}_pred_mean"] = float(np.mean(finite))
            vector[f"{target_name}_pred_std"] = float(np.std(finite))
            vector[f"{target_name}_pred_min"] = float(np.min(finite))
            vector[f"{target_name}_pred_max"] = float(np.max(finite))
            vector[f"{target_name}_pred_range"] = float(np.max(finite) - np.min(finite))
        else:
            vector[f"{target_name}_pred_mean"] = np.nan
            vector[f"{target_name}_pred_std"] = np.nan
            vector[f"{target_name}_pred_min"] = np.nan
            vector[f"{target_name}_pred_max"] = np.nan
            vector[f"{target_name}_pred_range"] = np.nan
        finite_ramps = ramps[np.isfinite(ramps)]
        if finite_ramps.size:
            vector[f"{target_name}_ramp_abs_mean"] = float(np.mean(np.abs(finite_ramps)))
            vector[f"{target_name}_ramp_abs_max"] = float(np.max(np.abs(finite_ramps)))
        else:
            vector[f"{target_name}_ramp_abs_mean"] = np.nan
            vector[f"{target_name}_ramp_abs_max"] = np.nan
    return vector if any_target else None


def _sequence_residual_vector(
    day_frame: pd.DataFrame,
    *,
    target: str,
    n_steps: int,
) -> np.ndarray | None:
    target_name = _sequence_target_name(target)
    pred_col = f"{target_name}_Model_MW"
    true_col = f"{target_name}_Actual_MW"
    if pred_col not in day_frame.columns or true_col not in day_frame.columns:
        return None
    pred = pd.to_numeric(day_frame[pred_col], errors="coerce").to_numpy(dtype=float)
    actual = pd.to_numeric(day_frame[true_col], errors="coerce").to_numpy(dtype=float)
    if len(pred) != n_steps or len(actual) != n_steps:
        return None
    residual = actual - pred
    if not np.isfinite(residual).all():
        return None
    return residual


def _sequence_complete_history_days(
    forecast: pd.DataFrame,
    *,
    day: pd.Timestamp,
    available_until: pd.Timestamp,
    window_days: int,
    n_steps: int,
) -> list[pd.Timestamp]:
    window_start = day - pd.Timedelta(days=window_days) if window_days > 0 else forecast.index.min().normalize()
    candidate_days = pd.DatetimeIndex(forecast.index.normalize().unique()).sort_values()
    days: list[pd.Timestamp] = []
    for candidate in candidate_days:
        if candidate >= day or candidate < window_start:
            continue
        candidate_end = candidate + pd.Timedelta(days=1) - pd.Timedelta(minutes=15)
        if candidate_end > available_until:
            continue
        day_frame = _sequence_day_frame(forecast, candidate, n_steps=n_steps)
        if day_frame is None:
            continue
        days.append(candidate)
    return days


def _fit_predict_sequence_residual(
    forecast: pd.DataFrame,
    *,
    day: pd.Timestamp,
    target: str,
    config: RenewableGenerationPostprocessConfig,
) -> np.ndarray | None:
    n_steps = config.sequence_residual_n_steps
    current_day = _sequence_day_frame(forecast, day, n_steps=n_steps)
    if current_day is None:
        return None

    available_until = _target_availability_cutoff(
        day,
        lag_days=config.target_availability_lag_days,
        cutoff_hour=config.target_availability_cutoff_hour,
        cutoff_minute=config.target_availability_cutoff_minute,
    )
    history_days = _sequence_complete_history_days(
        forecast,
        day=day,
        available_until=available_until,
        window_days=config.sequence_residual_window_days,
        n_steps=n_steps,
    )

    feature_rows: list[dict[str, float]] = []
    residual_rows: list[np.ndarray] = []
    for history_day in history_days:
        history_frame = _sequence_day_frame(forecast, history_day, n_steps=n_steps)
        if history_frame is None:
            continue
        feature_vector = _sequence_feature_vector(
            history_frame,
            feature_targets=config.sequence_residual_feature_targets,
            n_steps=n_steps,
        )
        residual_vector = _sequence_residual_vector(history_frame, target=target, n_steps=n_steps)
        if feature_vector is None or residual_vector is None:
            continue
        feature_rows.append(feature_vector)
        residual_rows.append(residual_vector)

    if len(feature_rows) < config.sequence_residual_min_train_days:
        return None

    current_features = _sequence_feature_vector(
        current_day,
        feature_targets=config.sequence_residual_feature_targets,
        n_steps=n_steps,
    )
    if current_features is None:
        return None

    X_train = pd.DataFrame(feature_rows)
    X_current = pd.DataFrame([current_features])
    X_current = X_current.reindex(columns=X_train.columns)
    y_train = np.vstack(residual_rows)

    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        Ridge(alpha=config.sequence_residual_ridge_alpha),
    )
    model.fit(X_train, y_train)
    return model.predict(X_current)[0]


def apply_renewable_generation_sequence_residual_correction(
    forecast: pd.DataFrame,
    config: RenewableGenerationPostprocessConfig,
) -> pd.DataFrame:
    """Apply a rolling day-level residual trajectory correction.

    For delivery day D this only trains on complete previous days whose actuals are
    available before the configured operational cutoff. With the usual D-1 10:00 cutoff,
    the newest complete training day is D-2.
    """
    if not config.include_sequence_residual_correction:
        return forecast

    corrected_blocks: list[pd.DataFrame] = []
    test_days = pd.DatetimeIndex(forecast.index.normalize().unique()).sort_values()
    targets = [_sequence_target_name(target) for target in config.sequence_residual_targets]

    for day in test_days:
        day_frame = _sequence_day_frame(forecast, day, n_steps=config.sequence_residual_n_steps)
        if day_frame is None:
            day_end = day + pd.Timedelta(days=1) - pd.Timedelta(minutes=15)
            partial = forecast.loc[(forecast.index >= day) & (forecast.index <= day_end)].copy()
            if not partial.empty:
                corrected_blocks.append(partial)
            continue

        corrected_day = day_frame.copy()
        for target in targets:
            pred_col = f"{target}_Model_MW"
            if pred_col not in corrected_day.columns:
                continue
            base_col = _point_base_column(target)
            if base_col not in corrected_day.columns:
                corrected_day[base_col] = corrected_day[pred_col]
            residual = _fit_predict_sequence_residual(
                forecast,
                day=day,
                target=target,
                config=config,
            )
            if residual is None:
                continue
            values = corrected_day[pred_col].to_numpy(dtype=float) + (
                config.sequence_residual_shrinkage * residual
            )
            upper_bound = _postprocess_upper_bound(pred_col, config)
            corrected_day[pred_col] = np.clip(values, 0.0, upper_bound)

        if {"Wind_Onshore_Model_MW", "Wind_Offshore_Model_MW"}.issubset(corrected_day.columns):
            if "Wind_Total_PointBase_MW" not in corrected_day.columns and "Wind_Total_Model_MW" in corrected_day.columns:
                corrected_day["Wind_Total_PointBase_MW"] = corrected_day["Wind_Total_Model_MW"]
            corrected_day["Wind_Total_Model_MW"] = (
                corrected_day["Wind_Onshore_Model_MW"] + corrected_day["Wind_Offshore_Model_MW"]
            )
        if {"Solar_Model_MW", "Wind_Total_Model_MW"}.issubset(corrected_day.columns):
            corrected_day["Renewable_Total_Model_MW"] = (
                corrected_day["Solar_Model_MW"] + corrected_day["Wind_Total_Model_MW"]
            )
        corrected_blocks.append(corrected_day)

    if not corrected_blocks:
        raise ValueError("No forecast rows were available for sequence residual correction.")
    corrected = pd.concat(corrected_blocks).sort_index()
    corrected.index.name = "timestamp"
    return corrected


def _target_quantile_column(target: str, quantile: float) -> str:
    return f"{target}_{quantile_column(quantile)}"


def _point_base_column(target: str) -> str:
    return f"{target}_PointBase_MW"


def _postprocess_upper_bound(pred_col: str, config: RenewableGenerationPostprocessConfig) -> float | None:
    target = _target_name_from_prediction(pred_col)
    for key in (pred_col, target, target.replace("_", "")):
        if key in config.target_upper_bounds_mw:
            value = float(config.target_upper_bounds_mw[key])
            return value if pd.notna(value) and value > 0 else None
    return None


def _prediction_bin_edges(values: pd.Series, n_bins: int) -> np.ndarray | None:
    valid = values.replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid) < max(2, n_bins):
        return None
    quantiles = np.linspace(0.0, 1.0, max(2, n_bins) + 1)
    edges = np.unique(valid.quantile(quantiles).to_numpy(dtype=float))
    if len(edges) < 3:
        return None
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def _prediction_bins(values: pd.Series, edges: np.ndarray | None) -> pd.Series:
    if edges is None:
        return pd.Series(np.repeat("all", len(values)), index=values.index)
    bins = pd.cut(values.astype(float), bins=edges, include_lowest=True, labels=False)
    return bins.astype("Int64").astype(str).replace("<NA>", "missing")


def _postprocess_group_values(
    index: pd.DatetimeIndex,
    predictions: pd.Series,
    *,
    group: str,
    prediction_edges: np.ndarray | None,
) -> pd.Series:
    if group == "none":
        return pd.Series(np.repeat("none", len(index)), index=index)
    if group == "global":
        return pd.Series(np.repeat("global", len(index)), index=index)

    hour = pd.Series(index.hour, index=index).astype(str)
    mtu = pd.Series(index.hour * 4 + index.minute // 15, index=index).astype(str)
    daytype = pd.Series(np.where(index.dayofweek >= 5, "weekend", "weekday"), index=index)
    pred_bin = _prediction_bins(predictions, prediction_edges)

    if group == "hour":
        return hour
    if group == "mtu":
        return mtu
    if group == "daytype_hour":
        return daytype + "_h" + hour
    if group == "daytype_mtu":
        return daytype + "_m" + mtu
    if group == "prediction_bin":
        return pred_bin
    if group == "hour_prediction_bin":
        return hour + "_b" + pred_bin
    if group == "mtu_prediction_bin":
        return mtu + "_b" + pred_bin
    if group == "daytype_prediction_bin":
        return daytype + "_b" + pred_bin
    raise ValueError(f"Unsupported correction_group: {group!r}")


def _apply_postprocess_correction_for_day(
    corrected_day: pd.DataFrame,
    history: pd.DataFrame,
    *,
    pred_col: str,
    true_col: str,
    config: RenewableGenerationPostprocessConfig,
) -> None:
    if config.correction_group == "none" or config.correction_window_days <= 0:
        return
    if pred_col not in corrected_day.columns or true_col not in corrected_day.columns:
        return
    if pred_col not in history.columns or true_col not in history.columns:
        return

    hist = history[[pred_col, true_col]].dropna()
    if len(hist) < config.correction_min_observations:
        return

    errors = hist[pred_col] - hist[true_col]
    global_bias = float(errors.mean())
    edges = _prediction_bin_edges(hist[pred_col], config.correction_prediction_bins)
    hist_groups = _postprocess_group_values(
        hist.index,
        hist[pred_col],
        group=config.correction_group,
        prediction_edges=edges,
    )
    current_groups = _postprocess_group_values(
        corrected_day.index,
        corrected_day[pred_col],
        group=config.correction_group,
        prediction_edges=edges,
    )

    bias_by_group = errors.groupby(hist_groups).mean()
    count_by_group = errors.groupby(hist_groups).count()
    bias_by_group = bias_by_group[count_by_group >= config.correction_min_observations]
    correction = current_groups.map(bias_by_group).fillna(global_bias).to_numpy(dtype=float)
    values = corrected_day[pred_col].to_numpy(dtype=float) - config.correction_shrinkage * correction
    upper_bound = _postprocess_upper_bound(pred_col, config)
    corrected_day[pred_col] = np.clip(values, 0.0, upper_bound)


def _should_calibrate_point(target: str, config: RenewableGenerationPostprocessConfig) -> bool:
    if not config.include_point_calibrator:
        return False
    if not config.point_calibrator_targets:
        return True
    return target in set(config.point_calibrator_targets)


def _postprocess_time_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    hour_angle = 2.0 * np.pi * (index.hour * 4 + index.minute // 15) / 96.0
    doy_angle = 2.0 * np.pi * index.dayofyear / 366.0
    return pd.DataFrame(
        {
            "hour_sin": np.sin(hour_angle),
            "hour_cos": np.cos(hour_angle),
            "doy_sin": np.sin(doy_angle),
            "doy_cos": np.cos(doy_angle),
            "is_weekend": (index.dayofweek >= 5).astype(float),
        },
        index=index,
    )


def _point_calibrator_feature_frame(
    frame: pd.DataFrame,
    *,
    target: str,
    pred_col: str,
) -> pd.DataFrame:
    """Compact final-output calibration features built only from forecast-time values."""
    base_col = _point_base_column(target)
    base_prediction = frame[base_col] if base_col in frame.columns else frame[pred_col]
    features = _postprocess_time_features(frame.index)
    features[f"{target}_base_prediction_mw"] = base_prediction.astype(float)
    features[f"{target}_base_prediction_log1p"] = np.log1p(base_prediction.astype(float).clip(lower=0.0))

    prefix = f"{target}_"
    component_model_cols = [
        col
        for col in frame.columns
        if col.startswith(prefix)
        and col.endswith("_Model_MW")
        and col != pred_col
    ]
    component_baseline_cols = [
        col
        for col in frame.columns
        if col.startswith(prefix) and col.endswith("_Baseline_MW")
    ]

    for col in [*component_model_cols, *component_baseline_cols]:
        features[col] = frame[col].astype(float)

    if component_model_cols:
        component_models = frame[component_model_cols].astype(float)
        features[f"{target}_component_model_spread_mw"] = component_models.max(axis=1) - component_models.min(axis=1)
        features[f"{target}_component_model_std_mw"] = component_models.std(axis=1).fillna(0.0)
    if component_baseline_cols:
        baselines = frame[component_baseline_cols].astype(float)
        baseline_sum = baselines.sum(axis=1)
        features[f"{target}_baseline_sum_mw"] = baseline_sum
        features[f"{target}_model_minus_baseline_sum_mw"] = base_prediction.astype(float) - baseline_sum
        features[f"{target}_baseline_spread_mw"] = baselines.max(axis=1) - baselines.min(axis=1)

    return features.replace([np.inf, -np.inf], np.nan)


def _make_point_calibrator(config: RenewableGenerationPostprocessConfig):
    if config.point_calibrator_model_type == "ridge":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0))
    if config.point_calibrator_model_type == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            max_iter=config.point_calibrator_hgb_max_iter,
            learning_rate=config.point_calibrator_hgb_learning_rate,
            max_leaf_nodes=config.point_calibrator_hgb_max_leaf_nodes,
            l2_regularization=config.point_calibrator_hgb_l2_regularization,
            random_state=config.point_calibrator_random_state,
        )
    raise ValueError(f"Unsupported point_calibrator_model_type: {config.point_calibrator_model_type!r}")


def _apply_point_calibrator_for_day(
    corrected_day: pd.DataFrame,
    history: pd.DataFrame,
    *,
    pred_col: str,
    true_col: str,
    config: RenewableGenerationPostprocessConfig,
) -> None:
    target = _target_name_from_prediction(pred_col)
    if not _should_calibrate_point(target, config):
        return
    if config.point_calibrator_window_days <= 0:
        return
    if pred_col not in corrected_day.columns or true_col not in corrected_day.columns:
        return
    if history.empty or pred_col not in history.columns or true_col not in history.columns:
        return

    base_col = _point_base_column(target)
    if base_col not in corrected_day.columns:
        corrected_day[base_col] = corrected_day[pred_col]
    if base_col not in history.columns:
        return

    hist = history.dropna(subset=[base_col, true_col])
    if len(hist) < config.point_calibrator_min_observations:
        return

    X_train = _point_calibrator_feature_frame(hist, target=target, pred_col=pred_col)
    y_train = hist[true_col].astype(float) - hist[base_col].astype(float)
    valid_train = X_train.notna().any(axis=1) & y_train.notna()
    X_train = X_train.loc[valid_train]
    y_train = y_train.loc[valid_train]
    if len(X_train) < config.point_calibrator_min_observations:
        return

    X_test = _point_calibrator_feature_frame(corrected_day, target=target, pred_col=pred_col)
    model = _make_point_calibrator(config)
    model.fit(X_train, y_train)
    residual_prediction = np.asarray(model.predict(X_test), dtype=float)
    base_prediction = corrected_day[base_col].astype(float).to_numpy(dtype=float)
    calibrated = base_prediction + config.point_calibrator_shrinkage * residual_prediction
    upper_bound = _postprocess_upper_bound(pred_col, config)
    corrected_day[pred_col] = np.clip(calibrated, 0.0, upper_bound)


def _should_postprocess_quantiles(target: str, config: RenewableGenerationPostprocessConfig) -> bool:
    if not config.include_residual_quantiles:
        return False
    if not config.residual_quantile_targets:
        return True
    return target in set(config.residual_quantile_targets)


def _apply_residual_quantiles_for_day(
    corrected_day: pd.DataFrame,
    history: pd.DataFrame,
    *,
    pred_col: str,
    true_col: str,
    config: RenewableGenerationPostprocessConfig,
) -> None:
    """Add leakage-safe residual quantiles around a final renewable generation point forecast."""
    target = _target_name_from_prediction(pred_col)
    if not _should_postprocess_quantiles(target, config):
        return
    if config.residual_quantile_window_days <= 0:
        return
    if pred_col not in corrected_day.columns or true_col not in corrected_day.columns:
        return
    if history.empty or pred_col not in history.columns or true_col not in history.columns:
        return

    hist = history[[pred_col, true_col]].dropna()
    if len(hist) < config.residual_quantile_min_observations:
        return

    quantiles = list(config.residual_quantiles)
    residuals = hist[true_col].astype(float) - hist[pred_col].astype(float)
    global_quantiles = residuals.quantile(quantiles).to_dict()
    edges = _prediction_bin_edges(hist[pred_col], config.residual_quantile_prediction_bins)
    hist_groups = _postprocess_group_values(
        hist.index,
        hist[pred_col],
        group=config.residual_quantile_group,
        prediction_edges=edges,
    )
    current_groups = _postprocess_group_values(
        corrected_day.index,
        corrected_day[pred_col],
        group=config.residual_quantile_group,
        prediction_edges=edges,
    )
    grouped_quantiles = {
        key: values.quantile(quantiles).to_dict()
        for key, values in residuals.groupby(hist_groups)
        if len(values) >= config.residual_quantile_min_observations
    }

    quantile_cols = [_target_quantile_column(target, quantile) for quantile in quantiles]
    quantile_values = pd.DataFrame(np.nan, index=corrected_day.index, columns=quantile_cols, dtype=float)
    base_prediction = corrected_day[pred_col].astype(float)
    upper_bound = _postprocess_upper_bound(pred_col, config)

    for index_value, group_key in current_groups.items():
        residual_quantiles = grouped_quantiles.get(group_key, global_quantiles)
        if not residual_quantiles:
            continue
        base = float(base_prediction.at[index_value])
        for quantile, column in zip(quantiles, quantile_cols, strict=True):
            if quantile in residual_quantiles:
                quantile_values.at[index_value, column] = base + float(residual_quantiles[quantile])

    median_col = _target_quantile_column(target, 0.5)
    if config.residual_quantile_spread_scale != 1.0 and median_col in quantile_values.columns:
        median = quantile_values[median_col]
        quantile_values.loc[:, quantile_cols] = median.to_numpy()[:, None] + config.residual_quantile_spread_scale * (
            quantile_values[quantile_cols].sub(median, axis=0)
        )

    quantile_values.loc[:, quantile_cols] = quantile_values[quantile_cols].clip(lower=0.0, upper=upper_bound)
    quantile_values.loc[:, quantile_cols] = np.maximum.accumulate(
        quantile_values[quantile_cols].to_numpy(dtype=float),
        axis=1,
    )
    corrected_day.loc[:, quantile_cols] = quantile_values[quantile_cols]


def apply_renewable_generation_postprocess(
    forecast: pd.DataFrame,
    config: RenewableGenerationPostprocessConfig,
) -> pd.DataFrame:
    """Apply leakage-safe rolling residual correction to an existing forecast.

    The correction history is capped at the operational target-availability cutoff
    (``target_availability_lag_days`` / ``target_availability_cutoff_hour``) so it only
    uses actuals that are known before the 12:00 gate on the forecast morning.
    """
    corrected_blocks: list[pd.DataFrame] = []
    test_days = pd.date_range(
        forecast.index.min().normalize(),
        forecast.index.max().normalize(),
        freq="D",
        tz=config.target_tz,
    )
    pairs = [
        pair
        for pair in _forecast_pairs(forecast)
        if pair[0] not in {"Renewable_Total_Model_MW"}
    ]

    for day in test_days:
        day_end = day + pd.Timedelta(days=1) - pd.Timedelta(minutes=15)
        day_forecast = forecast.loc[(forecast.index >= day) & (forecast.index <= day_end)].copy()
        if day_forecast.empty:
            continue

        history_window_days = config.correction_window_days
        if config.include_point_calibrator:
            history_window_days = max(history_window_days, config.point_calibrator_window_days)
        if config.include_residual_quantiles:
            history_window_days = max(history_window_days, config.residual_quantile_window_days)
        window_start = day - pd.Timedelta(days=history_window_days)
        available_until = _target_availability_cutoff(
            day,
            lag_days=config.target_availability_lag_days,
            cutoff_hour=config.target_availability_cutoff_hour,
            cutoff_minute=config.target_availability_cutoff_minute,
        )
        history = pd.concat(corrected_blocks).sort_index() if corrected_blocks else pd.DataFrame()
        if not history.empty:
            history = history.loc[(history.index >= window_start) & (history.index <= available_until)]

        for pred_col, true_col in pairs:
            target_name = _target_name_from_prediction(pred_col)
            if pred_col in day_forecast.columns:
                base_col = _point_base_column(target_name)
                if base_col not in day_forecast.columns:
                    day_forecast[base_col] = day_forecast[pred_col]
            if config.correction_targets and target_name not in set(config.correction_targets):
                continue
            _apply_postprocess_correction_for_day(
                day_forecast,
                history,
                pred_col=pred_col,
                true_col=true_col,
                config=config,
            )
            _apply_point_calibrator_for_day(
                day_forecast,
                history,
                pred_col=pred_col,
                true_col=true_col,
                config=config,
            )
            _apply_residual_quantiles_for_day(
                day_forecast,
                history,
                pred_col=pred_col,
                true_col=true_col,
                config=config,
            )

        if {"Wind_Onshore_Model_MW", "Wind_Offshore_Model_MW"}.issubset(day_forecast.columns):
            day_forecast["Wind_Total_Model_MW"] = (
                day_forecast["Wind_Onshore_Model_MW"] + day_forecast["Wind_Offshore_Model_MW"]
            )
        if {"Solar_Model_MW", "Wind_Total_Model_MW"}.issubset(day_forecast.columns):
            day_forecast["Renewable_Total_Model_MW"] = (
                day_forecast["Solar_Model_MW"] + day_forecast["Wind_Total_Model_MW"]
            )

        corrected_blocks.append(day_forecast)

    if not corrected_blocks:
        raise ValueError("No forecast rows were available for post-processing.")
    corrected = pd.concat(corrected_blocks).sort_index()
    corrected.index.name = "timestamp"
    return corrected


def _metrics_from_errors(
    errors: pd.Series,
    actual: pd.Series,
    *,
    target: str,
    group_name: str,
    group_value: str,
    installed_capacity_mw: dict[str, float],
) -> dict[str, Any]:
    valid = pd.concat([errors.rename("error"), actual.rename("actual")], axis=1).dropna()
    if valid.empty:
        return {
            "target": target,
            "group": group_name,
            "value": group_value,
            "mae": np.nan,
            "rmse": np.nan,
            "bias": np.nan,
            "nrmse_pct": np.nan,
            "n_obs": 0,
        }
    err = valid["error"]
    rmse = float(np.sqrt((err**2).mean()))
    capacity = _installed_capacity_for_target(target, installed_capacity_mw)
    nrmse = rmse / capacity if capacity is not None else np.nan
    return {
        "target": target,
        "group": group_name,
        "value": group_value,
        "mae": float(err.abs().mean()),
        "rmse": rmse,
        "bias": float(err.mean()),
        "nrmse_pct": float(nrmse * 100.0) if pd.notna(nrmse) else np.nan,
        "n_obs": int(len(valid)),
    }


def _diagnostic_quantile_groups(series: pd.Series, n_bins: int) -> pd.Series:
    edges = _prediction_bin_edges(series, n_bins)
    return _prediction_bins(series, edges)


def build_renewable_residual_diagnostics(
    forecast: pd.DataFrame,
    *,
    installed_capacity_mw: dict[str, float] | None = None,
    n_bins: int = 5,
) -> dict[str, pd.DataFrame]:
    """Build residual diagnostics by time, level, and ramp regimes."""
    installed_capacity_mw = installed_capacity_mw or {}
    rows_by_time: list[dict[str, Any]] = []
    rows_by_level: list[dict[str, Any]] = []
    rows_by_ramp: list[dict[str, Any]] = []

    for pred_col, true_col in _forecast_pairs(forecast):
        target = _target_name_from_prediction(pred_col)
        errors = forecast[pred_col] - forecast[true_col]
        actual = forecast[true_col]

        time_groups = {
            "hour": pd.Series(forecast.index.hour, index=forecast.index).astype(str),
            "mtu": pd.Series(forecast.index.hour * 4 + forecast.index.minute // 15, index=forecast.index).astype(str),
            "month": pd.Series(forecast.index.strftime("%Y-%m"), index=forecast.index),
            "weekday": pd.Series(forecast.index.day_name(), index=forecast.index),
            "daytype": pd.Series(np.where(forecast.index.dayofweek >= 5, "weekend", "weekday"), index=forecast.index),
        }
        for group_name, groups in time_groups.items():
            for value, idx in groups.groupby(groups).groups.items():
                rows_by_time.append(
                    _metrics_from_errors(
                        errors.loc[idx],
                        actual.loc[idx],
                        target=target,
                        group_name=group_name,
                        group_value=str(value),
                        installed_capacity_mw=installed_capacity_mw,
                    )
                )

        level_groups = {
            "actual_bin": _diagnostic_quantile_groups(actual, n_bins),
            "prediction_bin": _diagnostic_quantile_groups(forecast[pred_col], n_bins),
        }
        for group_name, groups in level_groups.items():
            for value, idx in groups.groupby(groups).groups.items():
                rows_by_level.append(
                    _metrics_from_errors(
                        errors.loc[idx],
                        actual.loc[idx],
                        target=target,
                        group_name=group_name,
                        group_value=str(value),
                        installed_capacity_mw=installed_capacity_mw,
                    )
                )

        for step, label in [(4, "1h"), (12, "3h")]:
            ramp = actual.diff(step).abs()
            groups = _diagnostic_quantile_groups(ramp, n_bins)
            for value, idx in groups.groupby(groups).groups.items():
                rows_by_ramp.append(
                    _metrics_from_errors(
                        errors.loc[idx],
                        actual.loc[idx],
                        target=target,
                        group_name=f"actual_ramp_{label}_bin",
                        group_value=str(value),
                        installed_capacity_mw=installed_capacity_mw,
                    )
                )

    return {
        "diagnostics_by_time": pd.DataFrame(rows_by_time),
        "diagnostics_by_level": pd.DataFrame(rows_by_level),
        "diagnostics_by_ramp": pd.DataFrame(rows_by_ramp),
    }


def _renewable_quantile_period_metrics(
    forecast: pd.DataFrame,
    *,
    target: str,
    quantiles: list[float],
    period: str,
) -> dict[str, Any]:
    true_col = f"{target}_Actual_MW"
    quantile_cols = [_target_quantile_column(target, quantile) for quantile in quantiles]
    row: dict[str, Any] = {
        "target": target,
        "model": "RollingResidualQuantiles",
        "period": period,
        "lqs": np.nan,
        "n_obs": 0,
    }
    if true_col not in forecast.columns or any(column not in forecast.columns for column in quantile_cols):
        return row

    valid = forecast[[true_col, *quantile_cols]].dropna()
    row["n_obs"] = int(len(valid))
    if valid.empty:
        return row

    y_true = valid[true_col].to_numpy(dtype=float)
    losses = [
        pinball_score(y_true, valid[_target_quantile_column(target, quantile)].to_numpy(dtype=float), quantile)
        for quantile in quantiles
    ]
    row["lqs"] = float(np.mean(np.vstack(losses), axis=0).mean())

    if 0.5 in quantiles:
        median = valid[_target_quantile_column(target, 0.5)].to_numpy(dtype=float)
        row["median_mae"] = float(np.abs(y_true - median).mean())

    for nominal, lower, upper in [(0.50, 0.25, 0.75), (0.80, 0.1, 0.9), (0.90, 0.05, 0.95), (0.95, 0.025, 0.975)]:
        if lower in quantiles and upper in quantiles:
            lower_values = valid[_target_quantile_column(target, lower)].to_numpy(dtype=float)
            upper_values = valid[_target_quantile_column(target, upper)].to_numpy(dtype=float)
            row[f"coverage_{nominal:.2f}"] = float(((y_true >= lower_values) & (y_true <= upper_values)).mean())
            row[f"width_{nominal:.2f}"] = float((upper_values - lower_values).mean())
    return row


def evaluate_renewable_generation_quantiles(
    forecast: pd.DataFrame,
    config: RenewableGenerationPostprocessConfig,
) -> pd.DataFrame:
    if not config.include_residual_quantiles:
        return pd.DataFrame()

    quantiles = list(config.residual_quantiles)
    targets = config.residual_quantile_targets or sorted(
        {
            column.rsplit("_q", 1)[0]
            for column in forecast.columns
            if "_q" in column and column.rsplit("_q", 1)[-1].replace(".", "", 1).isdigit()
        }
    )
    rows: list[dict[str, Any]] = []
    for target in targets:
        rows.append(_renewable_quantile_period_metrics(forecast, target=target, quantiles=quantiles, period="full"))
        month_index = forecast.index.tz_localize(None).to_period("M")
        for month, group in forecast.groupby(month_index):
            rows.append(
                _renewable_quantile_period_metrics(
                    group,
                    target=target,
                    quantiles=quantiles,
                    period=str(month),
                )
            )
    return pd.DataFrame(rows)


def run_renewable_generation_postprocess_pipeline(
    config: RenewableGenerationPostprocessConfig,
    *,
    save_outputs: bool = True,
) -> dict[str, pd.DataFrame]:
    print("\n--- Renewable Generation Postprocess ---")
    if config.include_forecast_stacking:
        forecast = build_stacked_renewable_generation_forecast(config)
    else:
        forecast = _load_forecast_csv(config.forecast_file, config.target_tz)
    forecast = apply_renewable_generation_sequence_residual_correction(forecast, config)
    corrected = apply_renewable_generation_postprocess(forecast, config)
    installed_capacity_mw = {
        str(key): float(value)
        for key, value in config.installed_capacity_mw.items()
        if pd.notna(value) and float(value) > 0
    }
    metrics = evaluate_renewable_generation_forecast(corrected, installed_capacity_mw=installed_capacity_mw)
    quantile_metrics = evaluate_renewable_generation_quantiles(corrected, config)
    diagnostics = build_renewable_residual_diagnostics(
        corrected,
        installed_capacity_mw=installed_capacity_mw,
        n_bins=config.diagnostics_bins,
    )

    print(metrics[metrics["period"].astype(str).eq("full")].to_string(index=False))
    if not quantile_metrics.empty:
        print("\n--- Renewable Generation Quantile Metrics ---")
        print(quantile_metrics[quantile_metrics["period"].astype(str).eq("full")].to_string(index=False))

    if save_outputs:
        config.export_dir.mkdir(parents=True, exist_ok=True)
        corrected.to_csv(config.export_dir / "forecast.csv")
        metrics.to_csv(config.export_dir / "metrics.csv", index=False)
        if not quantile_metrics.empty:
            quantile_metrics.to_csv(config.export_dir / "quantile_metrics.csv", index=False)
        for name, df in diagnostics.items():
            df.to_csv(config.export_dir / f"{name}.csv", index=False)
        with (config.export_dir / "config.json").open("w", encoding="utf-8") as handle:
            json.dump(config.model_dump(mode="json"), handle, indent=2)
        print(f"Saved renewable generation postprocess outputs -> {config.export_dir}")

    return {
        "forecast": corrected,
        "metrics": metrics,
        "quantile_metrics": quantile_metrics,
        **diagnostics,
    }


def build_entsoe_renewable_forecast_benchmark(
    config: EntsoeRenewableForecastBenchmarkConfig,
) -> pd.DataFrame:
    """Build a model-style forecast frame from ENTSO-E renewable forecast data."""
    actual = _load_or_fetch_benchmark_actual_generation(config)
    forecast = _load_or_fetch_entsoe_renewable_forecast(config)

    frame = pd.DataFrame(index=forecast.index)
    rename_map = {
        "Solar_Forecast_MW": "Solar_Model_MW",
        "Wind_Onshore_Forecast_MW": "Wind_Onshore_Model_MW",
        "Wind_Offshore_Forecast_MW": "Wind_Offshore_Model_MW",
        "Wind_Total_Forecast_MW": "Wind_Total_Model_MW",
        "Renewable_Total_Forecast_MW": "Renewable_Total_Model_MW",
    }
    for source, target in rename_map.items():
        if source in forecast.columns:
            frame[target] = forecast[source]

    for column in [
        "Solar_Actual_MW",
        "Wind_Onshore_Actual_MW",
        "Wind_Offshore_Actual_MW",
        "Wind_Total_Actual_MW",
        "Renewable_Total_Actual_MW",
    ]:
        if column in actual.columns:
            frame[column] = actual[column]

    frame = frame.sort_index()
    frame = frame.loc[:, ~frame.columns.duplicated()]
    frame.index.name = "timestamp"
    return frame


def run_entsoe_renewable_forecast_benchmark(
    config: EntsoeRenewableForecastBenchmarkConfig,
    *,
    save_outputs: bool = True,
) -> dict[str, pd.DataFrame]:
    load_dotenv(config.repo_root / ".env")
    print("\n--- ENTSO-E Renewable Forecast Benchmark ---")
    forecast = build_entsoe_renewable_forecast_benchmark(config)
    metrics = evaluate_renewable_generation_forecast(
        forecast,
        installed_capacity_mw=config.installed_capacity_mw,
    )
    print(metrics.to_string(index=False))

    if save_outputs:
        config.export_dir.mkdir(parents=True, exist_ok=True)
        forecast.to_csv(config.export_dir / "forecast.csv")
        metrics.to_csv(config.export_dir / "metrics.csv", index=False)
        with (config.export_dir / "config.json").open("w", encoding="utf-8") as handle:
            json.dump(config.model_dump(mode="json"), handle, indent=2)
        print(f"Saved ENTSO-E renewable forecast benchmark outputs -> {config.export_dir}")

    return {
        "forecast": forecast,
        "metrics": metrics,
    }


def run_renewable_generation_pipeline(
    config: RenewableGenerationModelConfig,
    *,
    save_outputs: bool = True,
) -> dict[str, pd.DataFrame]:
    load_dotenv(config.repo_root / ".env")
    print("\n--- Building Renewable Generation Dataset ---")
    dataset = build_renewable_generation_dataset(config)
    print(f"Dataset: {dataset.shape[0]:,} rows, {dataset.shape[1]:,} columns")

    print("\n--- Rolling Renewable Generation Forecast ---")
    forecast, runtime = rolling_renewable_generation_forecast(dataset, config)
    installed_capacity_mw = _load_installed_capacity_denominators(config)
    metrics = evaluate_renewable_generation_forecast(forecast, installed_capacity_mw=installed_capacity_mw)
    print("\n--- Renewable Generation Metrics ---")
    print(metrics.to_string(index=False))

    if save_outputs:
        config.export_dir.mkdir(parents=True, exist_ok=True)
        forecast.to_csv(config.export_dir / "forecast.csv")
        runtime.to_csv(config.export_dir / "runtime.csv", index=False)
        metrics.to_csv(config.export_dir / "metrics.csv", index=False)
        with (config.export_dir / "config.json").open("w", encoding="utf-8") as handle:
            json.dump(config.model_dump(mode="json"), handle, indent=2)
        print(f"Saved renewable generation outputs -> {config.export_dir}")

    return {
        "dataset": dataset,
        "forecast": forecast,
        "runtime": runtime,
        "metrics": metrics,
    }
