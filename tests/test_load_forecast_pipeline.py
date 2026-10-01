from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from da_price_forecasting.config import (
    EntsoeLoadForecastBenchmarkConfig,
    LoadForecastModelConfig,
    RegionalLoadForecastConfig,
)
from da_price_forecasting.pipelines import load_forecast as lf


def _config(tmp_path: Path, **overrides) -> LoadForecastModelConfig:
    data = {
        "repo_root": tmp_path,
        "actual_load_file": tmp_path / "actual_load.csv",
        "icon_dir": tmp_path / "icon",
        "export_dir": tmp_path / "out",
        "include_holiday_features": False,
        "actual_load_lag_days": [2, 7],
        "train_days_rolling": 8,
        "min_train_days": 2,
        "model_type": "ridge",
    }
    data.update(overrides)
    return LoadForecastModelConfig(**data)


def test_target_availability_cutoff_defaults_to_previous_full_day(tmp_path: Path) -> None:
    day = pd.Timestamp("2026-01-10T00:00:00+01:00")
    config = _config(tmp_path, target_availability_lag_days=1)

    assert lf._target_availability_cutoff(day, config) == pd.Timestamp("2026-01-09T23:45:00+01:00")


def test_target_availability_cutoff_can_use_morning_cutoff(tmp_path: Path) -> None:
    day = pd.Timestamp("2026-01-10T00:00:00+01:00")
    config = _config(
        tmp_path,
        target_availability_lag_days=1,
        target_availability_cutoff_hour=10,
        target_availability_cutoff_minute=15,
    )

    assert lf._target_availability_cutoff(day, config) == pd.Timestamp("2026-01-09T10:15:00+01:00")


def test_windowed_cache_uses_available_cache_when_refresh_fails(tmp_path: Path) -> None:
    cache_file = tmp_path / "actual_load.csv"
    cached_index = pd.date_range("2026-05-20T00:00:00+02:00", periods=96, freq="15min")
    cached = pd.DataFrame({"load_actual": np.arange(len(cached_index), dtype=float)}, index=cached_index)
    cached.to_csv(cache_file)

    def fail_fetch(fetch_start: pd.Timestamp, fetch_end: pd.Timestamp) -> pd.DataFrame:
        raise RuntimeError(f"ENTSO-E down for {fetch_start.date()}..{fetch_end.date()}")

    with pytest.warns(RuntimeWarning, match="Failed to refresh actual load"):
        result = lf._load_or_fetch_windowed_cache(
            path=cache_file,
            target_tz="Europe/Berlin",
            start=pd.Timestamp("2026-05-20", tz="Europe/Berlin"),
            end=pd.Timestamp("2026-05-21", tz="Europe/Berlin"),
            fetch_window=fail_fetch,
            label="actual load",
        )

    assert result.index.min() == cached_index[0]
    assert result.index.max() == cached_index[-1]
    assert result["load_actual"].notna().all()


def test_actual_load_partial_refresh_uses_cache_when_fetch_fails(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        lf,
        "_current_operational_day",
        lambda target_tz: pd.Timestamp("2026-05-22", tz=target_tz),
    )

    actual_file = tmp_path / "actual_load.csv"
    cached_index = pd.date_range("2026-05-20T00:00:00+02:00", periods=3 * 96, freq="15min")
    cached = pd.DataFrame({"load_actual": np.arange(len(cached_index), dtype=float)}, index=cached_index)
    cached.to_csv(actual_file)

    def fail_fetch_actual_load(**kwargs):
        raise RuntimeError("ENTSO-E 503")

    monkeypatch.setattr(lf, "fetch_actual_load", fail_fetch_actual_load)

    config = _config(
        tmp_path,
        actual_load_file=actual_file,
        entsoe_start_date=date(2026, 5, 20),
        entsoe_end_date=date(2026, 5, 23),
        test_start=date(2026, 5, 23),
        test_end=date(2026, 5, 23),
        include_partial_load_features=True,
        partial_load_reference_day=1,
    )

    with pytest.warns(RuntimeWarning, match="Failed to refresh current-day partial actual load"):
        result = lf._load_or_fetch_actual_load(config)

    assert result.index.max() == cached_index[-1]
    assert result["load_actual"].notna().all()


def test_actual_load_fetch_window_includes_partial_current_day(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        lf,
        "_current_operational_day",
        lambda target_tz: pd.Timestamp("2026-05-22", tz=target_tz),
    )

    config = _config(
        tmp_path,
        entsoe_start_date=date(2026, 5, 1),
        entsoe_end_date=date(2026, 5, 23),
        test_start=date(2026, 5, 23),
        test_end=date(2026, 5, 23),
        include_partial_load_features=True,
        partial_load_reference_day=1,
    )

    _, end = lf._actual_load_fetch_window(config)

    assert end == pd.Timestamp("2026-05-22", tz="Europe/Berlin")


def test_actual_load_fetch_window_without_partial_stops_at_latest_complete_day(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        lf,
        "_current_operational_day",
        lambda target_tz: pd.Timestamp("2026-05-22", tz=target_tz),
    )

    config = _config(
        tmp_path,
        entsoe_start_date=date(2026, 5, 1),
        entsoe_end_date=date(2026, 5, 23),
        test_start=date(2026, 5, 23),
        test_end=date(2026, 5, 23),
        include_partial_load_features=False,
    )

    _, end = lf._actual_load_fetch_window(config)

    assert end == pd.Timestamp("2026-05-21", tz="Europe/Berlin")


def test_actual_load_fetch_refreshes_current_partial_day(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        lf,
        "_current_operational_day",
        lambda target_tz: pd.Timestamp("2026-05-22", tz=target_tz),
    )

    actual_file = tmp_path / "actual_load.csv"
    cached_index = pd.date_range("2026-05-20T00:00:00+02:00", periods=2 * 96, freq="15min")
    cached = pd.DataFrame({"load_actual": np.arange(len(cached_index), dtype=float)}, index=cached_index)
    cached.loc[pd.Timestamp("2026-05-22T00:00:00+02:00"), "load_actual"] = 3.0
    cached.to_csv(actual_file)

    fetched_index = pd.to_datetime(["2026-05-22T10:15:00+02:00"], utc=True).tz_convert("Europe/Berlin")
    fetched = pd.DataFrame({"load_actual": [99.0]}, index=fetched_index)
    calls = []

    def fake_fetch_actual_load(**kwargs):
        calls.append(kwargs)
        return fetched

    monkeypatch.setattr(lf, "fetch_actual_load", fake_fetch_actual_load)

    config = _config(
        tmp_path,
        actual_load_file=actual_file,
        entsoe_start_date=date(2026, 5, 20),
        entsoe_end_date=date(2026, 5, 23),
        test_start=date(2026, 5, 23),
        test_end=date(2026, 5, 23),
        include_partial_load_features=True,
        partial_load_reference_day=1,
    )

    result = lf._load_or_fetch_actual_load(config)

    assert len(calls) == 1
    assert calls[0]["start_day"] == pd.Timestamp("2026-05-22", tz="Europe/Berlin")
    assert calls[0]["require_complete_days"] is False
    assert result.loc[fetched_index[0], "load_actual"] == 99.0


def test_actual_load_fetch_repairs_cached_nan_days(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        lf,
        "_current_operational_day",
        lambda target_tz: pd.Timestamp("2026-05-22", tz=target_tz),
    )

    actual_file = tmp_path / "actual_load.csv"
    cached_index = pd.date_range("2026-05-20T00:00:00+02:00", periods=2 * 96, freq="15min")
    cached = pd.DataFrame({"load_actual": np.nan}, index=cached_index)
    cached.loc[cached_index.normalize() == pd.Timestamp("2026-05-21", tz="Europe/Berlin"), "load_actual"] = 42.0
    cached.to_csv(actual_file)

    repaired_index = pd.date_range("2026-05-20T00:00:00+02:00", periods=96, freq="15min")
    repaired = pd.DataFrame({"load_actual": np.arange(96, dtype=float)}, index=repaired_index)
    calls = []

    def fake_fetch_actual_load(**kwargs):
        calls.append(kwargs)
        return repaired

    monkeypatch.setattr(lf, "fetch_actual_load", fake_fetch_actual_load)

    config = _config(
        tmp_path,
        actual_load_file=actual_file,
        entsoe_start_date=date(2026, 5, 20),
        entsoe_end_date=date(2026, 5, 21),
        include_partial_load_features=False,
    )

    result = lf._load_or_fetch_actual_load(config)

    assert len(calls) == 1
    assert calls[0]["start_day"] == pd.Timestamp("2026-05-20", tz="Europe/Berlin")
    assert calls[0]["end_day"] == pd.Timestamp("2026-05-20", tz="Europe/Berlin")
    assert calls[0]["require_complete_days"] is True
    assert result.loc[repaired_index[0], "load_actual"] == 0.0
    assert result.loc[repaired_index[-1], "load_actual"] == 95.0


def test_actual_load_repair_accepts_complete_dst_spring_day() -> None:
    index = pd.date_range(
        pd.Timestamp("2026-03-29", tz="Europe/Berlin"),
        pd.Timestamp("2026-03-30", tz="Europe/Berlin"),
        freq="15min",
        inclusive="left",
    )
    actual = pd.DataFrame({"load_actual": np.arange(len(index), dtype=float)}, index=index)

    repair_days = lf._actual_load_days_below_count(
        actual,
        start=pd.Timestamp("2026-03-29", tz="Europe/Berlin"),
        end=pd.Timestamp("2026-03-29", tz="Europe/Berlin"),
        target_tz="Europe/Berlin",
    )

    assert len(index) == 92
    assert repair_days == []

    actual.iloc[-1, 0] = np.nan
    repair_days = lf._actual_load_days_below_count(
        actual,
        start=pd.Timestamp("2026-03-29", tz="Europe/Berlin"),
        end=pd.Timestamp("2026-03-29", tz="Europe/Berlin"),
        target_tz="Europe/Berlin",
    )

    assert repair_days == [pd.Timestamp("2026-03-29", tz="Europe/Berlin")]


def test_build_load_forecast_dataset_adds_calendar_lags_and_weather(monkeypatch, tmp_path: Path) -> None:
    index = pd.date_range("2026-01-01T00:00:00+01:00", periods=10 * 96, freq="15min")
    actual = pd.DataFrame({"load_actual": 50_000.0 + np.arange(len(index), dtype=float)}, index=index)
    weather = pd.DataFrame({"weather_t2m_C_cluster_0": np.linspace(0.0, 5.0, len(index))}, index=index)

    monkeypatch.setattr(lf, "_load_or_fetch_actual_load", lambda config: actual)
    monkeypatch.setattr(lf, "_build_load_weather_features", lambda config: weather)

    dataset = lf.build_load_forecast_dataset(_config(tmp_path))
    timestamp = index[2 * 96 + 5]

    assert "Load_Actual_MW" in dataset.columns
    assert "Load_Actual_MW_lag_d2" in dataset.columns
    assert "Load_Actual_MW_lag_d7" in dataset.columns
    assert "mtu_sin" in dataset.columns
    assert "weather_t2m_C_cluster_0" in dataset.columns
    assert dataset.loc[timestamp, "Load_Actual_MW_lag_d2"] == actual.loc[timestamp - pd.Timedelta(days=2), "load_actual"]


def test_build_load_forecast_dataset_keeps_future_feature_rows_without_actuals(monkeypatch, tmp_path: Path) -> None:
    actual_index = pd.date_range("2026-01-01T00:00:00+01:00", periods=3 * 96, freq="15min")
    weather_index = pd.date_range("2026-01-01T00:00:00+01:00", periods=4 * 96, freq="15min")
    actual = pd.DataFrame({"load_actual": 50_000.0 + np.arange(len(actual_index), dtype=float)}, index=actual_index)
    weather = pd.DataFrame({"weather_t2m_C_cluster_0": np.linspace(0.0, 5.0, len(weather_index))}, index=weather_index)

    monkeypatch.setattr(lf, "_load_or_fetch_actual_load", lambda config: actual)
    monkeypatch.setattr(lf, "_build_load_weather_features", lambda config: weather)

    dataset = lf.build_load_forecast_dataset(_config(tmp_path))
    future_timestamp = pd.Timestamp("2026-01-04T12:00:00+01:00")

    assert future_timestamp in dataset.index
    assert np.isnan(dataset.loc[future_timestamp, "Load_Actual_MW"])
    assert np.isfinite(dataset.loc[future_timestamp, "weather_t2m_C_cluster_0"])


def test_calendar_features_include_bridge_days_and_extra_harmonics() -> None:
    index = pd.date_range("2026-05-15T00:00:00+02:00", periods=1, freq="15min")

    features = lf._build_calendar_features(
        index,
        target_tz="Europe/Berlin",
        include_holidays=True,
        include_bridge_days=True,
        calendar_harmonics=3,
    )

    assert features.loc[index[0], "is_bridge_day"] == 1.0
    assert "doy_sin_3" in features.columns
    assert "week_cos_3" in features.columns


def test_regional_holiday_features_are_population_weighted(tmp_path: Path) -> None:
    index = pd.date_range("2026-01-06T00:00:00+01:00", periods=1, freq="15min")
    weight_file = tmp_path / "region_weights.csv"
    pd.DataFrame({"region": ["DE1", "LU0"], "weight": [2.0, 1.0]}).to_csv(weight_file, index=False)

    features = lf._build_regional_holiday_features(
        index,
        target_tz="Europe/Berlin",
        weight_file=weight_file,
        region_column="region",
        weight_column="weight",
        feature_prefix="regional_holiday",
    )

    assert np.isclose(features.loc[index[0], "regional_holiday_public_holiday_share"], 2.0 / 3.0)
    assert np.isclose(features.loc[index[0], "regional_holiday_nonworkday_share"], 2.0 / 3.0)


def test_weather_time_interactions_add_temperature_interactions() -> None:
    index = pd.date_range("2026-01-01T00:00:00+01:00", periods=2, freq="15min")
    features = pd.DataFrame(
        {
            "weather_t2m_C_cluster_0": [5.0, 6.0],
            "weather_hdd18_cluster_0": [13.0, 12.0],
            "mtu_sin": [0.0, 0.1],
            "mtu_cos": [1.0, 0.9],
            "is_weekend": [0.0, 0.0],
        },
        index=index,
    )

    result = lf._add_weather_time_interactions(features)

    assert result.loc[index[0], "weather_t2m_C_cluster_0_x_mtu_cos"] == 5.0
    assert "weather_hdd18_cluster_0_x_mtu_sin" in result.columns


def test_rich_temperature_weather_features_add_configured_thresholds(monkeypatch, tmp_path: Path) -> None:
    hourly_index = pd.date_range("2026-01-01T00:00:00", periods=2, freq="h", tz="Europe/Berlin")
    df_hourly = pd.DataFrame({"t2m_cluster_0": [283.15, 298.15]}, index=hourly_index)
    df_qh = pd.DataFrame(index=pd.date_range("2026-01-01T00:00:00", periods=8, freq="15min", tz="Europe/Berlin"))
    monkeypatch.setattr(lf, "load_dwd", lambda **kwargs: (df_hourly, df_qh))

    features = lf._build_load_weather_features(
        _config(
            tmp_path,
            include_rich_temperature_features=True,
            weather_hdd_thresholds=[12, 18],
            weather_cdd_thresholds=[20, 24],
        )
    )

    assert "weather_hdd12_cluster_0" in features.columns
    assert "weather_hdd18_cluster_0" in features.columns
    assert "weather_cdd20_cluster_0" in features.columns
    assert "weather_cdd24_cluster_0" in features.columns
    assert features.loc[hourly_index[0], "weather_hdd12_cluster_0"] == 2.0
    assert features.loc[hourly_index[1], "weather_cdd24_cluster_0"] == 1.0


def test_dwd_weather_auto_update_runs_before_loading_archive(monkeypatch, tmp_path: Path) -> None:
    from da_price_forecasting.preprocessing import dwd_icon_operational

    hourly_index = pd.date_range("2026-05-16T00:00:00", periods=2, freq="h", tz="Europe/Berlin")
    qh_index = pd.date_range("2026-05-16T00:00:00", periods=8, freq="15min", tz="Europe/Berlin")
    df_hourly = pd.DataFrame({"t2m_cluster_0": [283.15, 284.15]}, index=hourly_index)
    df_qh = pd.DataFrame({"ASWDIR_cluster_0": [1.0] * 8, "ASWDIFD_cluster_0": [2.0] * 8}, index=qh_index)
    calls = {}

    def fake_ensure_dwd_icon_weather(**kwargs):
        calls.update(kwargs)
        assert not calls.get("load_dwd_called", False)
        return [date(2026, 5, 15)]

    def fake_load_dwd(**kwargs):
        calls["load_dwd_called"] = True
        return df_hourly, df_qh

    monkeypatch.setattr(dwd_icon_operational, "ensure_dwd_icon_weather", fake_ensure_dwd_icon_weather)
    monkeypatch.setattr(lf, "load_dwd", fake_load_dwd)

    config = _config(
        tmp_path,
        required_run="06",
        test_start=date(2026, 5, 16),
        test_end=date(2026, 5, 16),
        dwd_folder_offset_date=date(2025, 10, 26),
        dwd_icon_auto_update=True,
        dwd_icon_raw_dir=tmp_path / "raw_dwd",
        dwd_icon_aggregation_shapefile_path=tmp_path / "countries.shp",
        dwd_icon_aggregation_n_clusters=25,
    )

    result = lf._build_load_weather_features(config)

    assert calls["forecast_start"] == date(2026, 5, 16)
    assert calls["forecast_end"] == date(2026, 5, 16)
    assert calls["run_hour"] == "06"
    assert calls["raw_base_dir"] == tmp_path / "raw_dwd"
    assert calls["n_clusters"] == 25
    assert calls["load_dwd_called"] is True
    assert "weather_t2m_C_cluster_0" in result.columns


def test_open_meteo_weather_features_use_existing_loader(monkeypatch, tmp_path: Path) -> None:
    weather_index = pd.date_range("2026-01-01T00:00:00", periods=2, freq="h", tz="Europe/Berlin")
    weather = pd.DataFrame(
        {
            "t2m_cluster_0": [283.15, 298.15],
            "td2m_cluster_0": [280.15, 290.15],
            "u10_cluster_0": [3.0, 4.0],
            "v10_cluster_0": [4.0, 3.0],
            "ssrd_cluster_0": [0.0, 100.0],
            "fdir_cluster_0": [0.0, 60.0],
        },
        index=weather_index,
    )
    monkeypatch.setattr(lf, "load_open_meteo", lambda **kwargs: weather)

    features = lf._build_load_weather_features(
        _config(
            tmp_path,
            weather_source="open_meteo",
            open_meteo_cluster_file=tmp_path / "clusters.parquet",
            open_meteo_weather_file=tmp_path / "open_meteo.csv",
            include_rich_temperature_features=True,
            weather_hdd_thresholds=[12, 18],
            weather_cdd_thresholds=[20, 24],
        )
    )

    assert "weather_hdd12_cluster_0" in features.columns
    assert "weather_cdd24_cluster_0" in features.columns
    assert "weather_wind_speed_10m_cluster_0" in features.columns
    assert "weather_solar_global_cluster_0" in features.columns
    assert "weather_rel_humidity_pct_cluster_0" in features.columns
    assert "weather_vpd_hPa_cluster_0" in features.columns
    assert np.isclose(features.loc[weather_index[0], "weather_wind_speed_10m_cluster_0"], 5.0)
    assert features.loc[weather_index[1], "weather_solar_diffuse_cluster_0"] == 40.0
    assert 0.0 <= features.loc[weather_index[0], "weather_rel_humidity_pct_cluster_0"] <= 100.0
    assert features.loc[weather_index[1], "weather_vpd_hPa_cluster_0"] > 0.0


def test_open_meteo_weather_features_extend_final_hour_to_full_quarter_day(
    monkeypatch,
    tmp_path: Path,
) -> None:
    weather_index = pd.date_range("2026-05-24", periods=24, freq="h", tz="Europe/Berlin")
    weather = pd.DataFrame(
        {
            "t2m_cluster_0": np.full(len(weather_index), 293.15),
            "ssrd_cluster_0": np.arange(len(weather_index), dtype=float),
        },
        index=weather_index,
    )
    monkeypatch.setattr(lf, "load_open_meteo", lambda **kwargs: weather)

    features = lf._build_load_weather_features(
        _config(
            tmp_path,
            weather_source="open_meteo",
            open_meteo_cluster_file=tmp_path / "clusters.parquet",
            open_meteo_weather_file=tmp_path / "open_meteo.csv",
        )
    )

    day = pd.Timestamp("2026-05-24", tz="Europe/Berlin")
    day_features = features.loc[day: day + pd.Timedelta(days=1) - pd.Timedelta(minutes=15)]

    assert len(day_features) == 96
    assert day_features.index[-1] == pd.Timestamp("2026-05-24T23:45:00+02:00")
    assert day_features["weather_t2m_C_cluster_0"].notna().all()


def test_open_meteo_weather_features_add_provider_ensemble_summary(
    monkeypatch,
    tmp_path: Path,
) -> None:
    weather_index = pd.date_range("2026-05-24", periods=2, freq="h", tz="Europe/Berlin")
    primary = pd.DataFrame(
        {
            "t2m_cluster_0": [293.15, 294.15],
            "ssrd_cluster_0": [100.0, 200.0],
        },
        index=weather_index,
    )
    extra = pd.DataFrame(
        {
            "t2m_cluster_0": [295.15, 296.15],
            "ssrd_cluster_0": [300.0, 400.0],
        },
        index=weather_index,
    )
    extra_file = tmp_path / "open_meteo_extra.csv"
    extra.tz_convert("UTC").to_csv(extra_file)
    monkeypatch.setattr(lf, "load_open_meteo", lambda **kwargs: primary)

    features = lf._build_load_weather_features(
        _config(
            tmp_path,
            weather_source="open_meteo",
            open_meteo_cluster_file=tmp_path / "clusters.parquet",
            open_meteo_weather_file=tmp_path / "open_meteo.csv",
            extra_open_meteo_weather_files=[extra_file],
            include_open_meteo_weather_ensemble_features=True,
        )
    )

    assert np.isclose(features.loc[weather_index[0], "weather_t2m_C_ens_mean_cluster_0"], 21.0)
    assert np.isclose(features.loc[weather_index[0], "weather_t2m_C_ens_std_cluster_0"], 1.0)
    assert np.isclose(features.loc[weather_index[0], "weather_solar_global_ens_mean_cluster_0"], 200.0)
    assert lf._split_weather_cluster_column("weather_t2m_C_ens_mean_cluster_0") == (
        "weather_t2m_C_ens_mean",
        0,
    )


def test_weighted_weather_features_use_cluster_weights() -> None:
    index = pd.date_range("2026-01-01T00:00:00+01:00", periods=2, freq="15min")
    features = pd.DataFrame(
        {
            "weather_t2m_C_cluster_0": [10.0, 20.0],
            "weather_t2m_C_cluster_1": [20.0, 40.0],
            "weather_solar_global_cluster_0": [1.0, 3.0],
            "weather_solar_global_cluster_1": [5.0, 7.0],
        },
        index=index,
    )
    weights = pd.Series({0: 1.0, 1: 3.0})

    result = lf._add_weighted_weather_features(features, weights)

    assert np.isclose(result.loc[index[0], "weather_weighted_t2m_C"], 17.5)
    assert np.isclose(result.loc[index[1], "weather_weighted_solar_global"], 6.0)


def test_build_load_forecast_dataset_adds_weighted_weather_interactions(monkeypatch, tmp_path: Path) -> None:
    index = pd.date_range("2026-01-01T00:00:00+01:00", periods=10 * 96, freq="15min")
    actual = pd.DataFrame({"load_actual": 50_000.0 + np.arange(len(index), dtype=float)}, index=index)
    weather = pd.DataFrame(
        {
            "weather_t2m_C_cluster_0": np.full(len(index), 10.0),
            "weather_t2m_C_cluster_1": np.full(len(index), 20.0),
            "weather_hdd18_cluster_0": np.full(len(index), 8.0),
            "weather_hdd18_cluster_1": np.full(len(index), 0.0),
        },
        index=index,
    )
    weight_file = tmp_path / "weather_cluster_weights.csv"
    pd.DataFrame({"cluster_id": [0, 1], "weight": [1.0, 3.0]}).to_csv(weight_file, index=False)

    monkeypatch.setattr(lf, "_load_or_fetch_actual_load", lambda config: actual)
    monkeypatch.setattr(lf, "_build_load_weather_features", lambda config: weather)

    dataset = lf.build_load_forecast_dataset(
        _config(
            tmp_path,
            include_weighted_weather_features=True,
            include_weather_time_interactions=True,
            weather_cluster_weight_file=weight_file,
        )
    )

    assert np.isclose(dataset.loc[index[0], "weather_weighted_t2m_C"], 17.5)
    assert "weather_weighted_t2m_C_x_mtu_cos" in dataset.columns
    assert "weather_weighted_hdd18_x_mtu_sin" in dataset.columns


def test_build_load_forecast_dataset_can_use_only_selected_weighted_weather(monkeypatch, tmp_path: Path) -> None:
    index = pd.date_range("2026-01-01T00:00:00+01:00", periods=10 * 96, freq="15min")
    actual = pd.DataFrame({"load_actual": 50_000.0 + np.arange(len(index), dtype=float)}, index=index)
    weather = pd.DataFrame(
        {
            "weather_t2m_C_cluster_0": np.full(len(index), 10.0),
            "weather_t2m_C_cluster_1": np.full(len(index), 20.0),
            "weather_solar_global_cluster_0": np.full(len(index), 100.0),
            "weather_solar_global_cluster_1": np.full(len(index), 200.0),
        },
        index=index,
    )
    weight_file = tmp_path / "weather_cluster_weights.csv"
    pd.DataFrame({"cluster_id": [0, 1], "weight": [1.0, 3.0]}).to_csv(weight_file, index=False)

    monkeypatch.setattr(lf, "_load_or_fetch_actual_load", lambda config: actual)
    monkeypatch.setattr(lf, "_build_load_weather_features", lambda config: weather)

    dataset = lf.build_load_forecast_dataset(
        _config(
            tmp_path,
            include_weighted_weather_features=True,
            include_weather_time_interactions=True,
            weather_cluster_weight_file=weight_file,
            weather_weighted_feature_bases=["t2m_C"],
            keep_weather_cluster_features=False,
        )
    )

    assert "weather_weighted_t2m_C" in dataset.columns
    assert "weather_weighted_t2m_C_x_mtu_cos" in dataset.columns
    assert "weather_weighted_solar_global" not in dataset.columns
    assert "weather_t2m_C_cluster_0" not in dataset.columns
    assert "weather_solar_global_cluster_0" not in dataset.columns


def test_partial_load_features_use_previous_morning_and_week_comparison() -> None:
    index = pd.date_range("2026-01-01T00:00:00+01:00", periods=12 * 96, freq="15min")
    actual = pd.DataFrame({"load_actual": np.arange(len(index), dtype=float)}, index=index)
    features = lf._build_partial_load_features(
        actual,
        pd.date_range("2026-01-10T00:00:00+01:00", periods=96, freq="15min"),
        target_tz="Europe/Berlin",
        reference_day=1,
        comparison_lag_days=7,
        morning_end_hour=11,
    )

    d1 = actual.loc["2026-01-09T00:00:00+01:00":"2026-01-09T11:45:00+01:00", "load_actual"]
    d8 = actual.loc["2026-01-02T00:00:00+01:00":"2026-01-02T11:45:00+01:00", "load_actual"]
    timestamp = pd.Timestamp("2026-01-10T12:00:00+01:00")

    assert np.isclose(features.loc[timestamp, "partial_load_d1_00_11_mean"], d1.mean())
    assert np.isclose(features.loc[timestamp, "partial_load_d1_00_11_mean_diff_d7"], d1.mean() - d8.mean())


def test_partial_load_features_support_quarter_hour_cutoff() -> None:
    index = pd.date_range("2026-01-01T00:00:00+01:00", periods=12 * 96, freq="15min")
    actual = pd.DataFrame({"load_actual": np.arange(len(index), dtype=float)}, index=index)
    features = lf._build_partial_load_features(
        actual,
        pd.date_range("2026-01-10T00:00:00+01:00", periods=96, freq="15min"),
        target_tz="Europe/Berlin",
        reference_day=1,
        comparison_lag_days=7,
        morning_end_hour=10,
        morning_end_minute=15,
    )

    d1 = actual.loc["2026-01-09T00:00:00+01:00":"2026-01-09T10:15:00+01:00", "load_actual"]
    timestamp = pd.Timestamp("2026-01-10T12:00:00+01:00")

    assert np.isclose(features.loc[timestamp, "partial_load_d1_00_1015_mean"], d1.mean())
    assert features.loc[timestamp, "partial_load_d1_00_1015_last"] == actual.loc[
        pd.Timestamp("2026-01-09T10:15:00+01:00"),
        "load_actual",
    ]


def test_partial_load_features_can_add_shape_and_point_values() -> None:
    index = pd.date_range("2026-01-01T00:00:00+01:00", periods=12 * 96, freq="15min")
    actual = pd.DataFrame({"load_actual": np.arange(len(index), dtype=float)}, index=index)
    features = lf._build_partial_load_features(
        actual,
        pd.date_range("2026-01-10T00:00:00+01:00", periods=96, freq="15min"),
        target_tz="Europe/Berlin",
        reference_day=1,
        comparison_lag_days=7,
        morning_end_hour=10,
        morning_end_minute=15,
        include_shape_features=True,
        point_times=["06:00", "10:15"],
    )

    timestamp = pd.Timestamp("2026-01-10T12:00:00+01:00")
    d1 = actual.loc["2026-01-09T00:00:00+01:00":"2026-01-09T10:15:00+01:00", "load_actual"]
    d8 = actual.loc["2026-01-02T00:00:00+01:00":"2026-01-02T10:15:00+01:00", "load_actual"]

    assert np.isclose(features.loc[timestamp, "partial_load_d1_00_1015_range"], d1.max() - d1.min())
    assert np.isclose(
        features.loc[timestamp, "partial_load_d1_00_1015_ramp_diff_d7"],
        (d1.iloc[-1] - d1.iloc[0]) - (d8.iloc[-1] - d8.iloc[0]),
    )
    assert features.loc[timestamp, "partial_load_d1_point_0600"] == actual.loc[
        pd.Timestamp("2026-01-09T06:00:00+01:00"),
        "load_actual",
    ]
    assert features.loc[timestamp, "partial_load_d1_point_1015"] == actual.loc[
        pd.Timestamp("2026-01-09T10:15:00+01:00"),
        "load_actual",
    ]


def test_weighted_weather_daily_features_add_daily_stats_and_diffs() -> None:
    index = pd.date_range("2026-01-01T00:00:00+01:00", periods=3 * 96, freq="15min")
    values = np.concatenate([np.full(96, 10.0), np.full(96, 12.0), np.linspace(15.0, 18.0, 96)])
    features = pd.DataFrame({"weather_weighted_t2m_C": values}, index=index)

    result = lf._add_weighted_weather_daily_features(
        features,
        prefix="weather_weighted",
        base_names=["t2m_C"],
        stats=["mean", "min", "max", "range"],
        lag_days=[1],
    )

    timestamp = pd.Timestamp("2026-01-03T12:00:00+01:00")
    assert np.isclose(result.loc[timestamp, "weather_weighted_t2m_C_daily_mean"], 16.5)
    assert np.isclose(result.loc[timestamp, "weather_weighted_t2m_C_daily_mean_diff_d1"], 4.5)
    assert np.isclose(result.loc[timestamp, "weather_weighted_t2m_C_daily_range"], 3.0)


def test_weather_cluster_spread_features_add_cross_cluster_stats() -> None:
    index = pd.date_range("2026-01-01T00:00:00+01:00", periods=2, freq="15min")
    features = pd.DataFrame(
        {
            "weather_t2m_C_cluster_0": [10.0, 20.0],
            "weather_t2m_C_cluster_1": [20.0, 40.0],
            "weather_hdd18_cluster_0": [8.0, 0.0],
            "weather_hdd18_cluster_1": [0.0, 0.0],
        },
        index=index,
    )

    result = lf._add_weather_cluster_spread_features(
        features,
        base_names=["t2m_C"],
        stats=["min", "max", "range", "std", "p10", "p90"],
    )

    assert "weather_spread_t2m_C_range" in result.columns
    assert "weather_spread_hdd18_range" not in result.columns
    assert np.isclose(result.loc[index[0], "weather_spread_t2m_C_range"], 10.0)
    assert np.isclose(result.loc[index[1], "weather_spread_t2m_C_p90"], 38.0)


def test_weighted_weather_quantile_features_use_population_weights() -> None:
    index = pd.date_range("2026-01-01T00:00:00+01:00", periods=2, freq="15min")
    features = pd.DataFrame(
        {
            "weather_t2m_C_cluster_0": [10.0, 20.0],
            "weather_t2m_C_cluster_1": [20.0, 40.0],
            "weather_t2m_C_cluster_2": [30.0, 60.0],
        },
        index=index,
    )
    weights = pd.Series({0: 1.0, 1: 1.0, 2: 8.0})

    result = lf._add_weighted_weather_quantile_features(
        features,
        weights,
        base_names=["t2m_C"],
        quantiles=[0.1, 0.5, 0.9],
    )

    assert np.isclose(result.loc[index[0], "weather_weighted_q_t2m_C_q10"], 10.0)
    assert np.isclose(result.loc[index[0], "weather_weighted_q_t2m_C_q50"], 30.0)
    assert np.isclose(result.loc[index[1], "weather_weighted_q_t2m_C_q90"], 60.0)


def test_weighted_weather_inertia_features_add_rolling_history() -> None:
    index = pd.date_range("2026-01-01T00:00:00+01:00", periods=12, freq="15min")
    features = pd.DataFrame({"weather_weighted_t2m_C": np.arange(len(index), dtype=float)}, index=index)

    result = lf._add_weighted_weather_inertia_features(
        features,
        prefix="weather_weighted",
        base_names=["t2m_C"],
        windows_hours=[1],
        stats=["mean", "range", "delta_mean"],
    )

    timestamp = index[4]
    assert np.isclose(result.loc[timestamp, "weather_weighted_t2m_C_roll1h_mean"], 2.5)
    assert np.isclose(result.loc[timestamp, "weather_weighted_t2m_C_roll1h_range"], 3.0)
    assert np.isclose(result.loc[timestamp, "weather_weighted_t2m_C_minus_roll1h_mean"], 1.5)


def test_rolling_load_forecast_can_model_entsoe_residual(tmp_path: Path) -> None:
    index = pd.date_range("2026-01-01T00:00:00+01:00", periods=20 * 96, freq="15min")
    mtu = index.hour * 4 + index.minute // 15
    benchmark = 45_000.0 + 900.0 * np.sin(2 * np.pi * mtu / 96)
    residual = 100.0 + np.arange(len(index), dtype=float) * 0.05
    dataset = pd.DataFrame(index=index)
    dataset["Load_Benchmark_MW"] = benchmark
    dataset["feature_residual_trend"] = np.arange(len(index), dtype=float)
    dataset["Load_Actual_MW"] = benchmark + residual

    config = _config(
        tmp_path,
        load_target_mode="entsoe_residual",
        test_start=pd.Timestamp("2026-01-12").date(),
        test_end=pd.Timestamp("2026-01-12").date(),
    )
    forecast, _ = lf.rolling_load_forecast(dataset, config)

    assert "Load_Benchmark_MW" in forecast.columns
    assert forecast["Load_Model_MW"].notna().all()
    assert forecast["Load_Model_MW"].mean() > forecast["Load_Benchmark_MW"].mean()


def test_evaluate_load_quantile_forecast_reports_lqs_and_coverage() -> None:
    index = pd.date_range("2026-01-01T00:00:00+01:00", periods=4, freq="15min")
    forecast = pd.DataFrame(
        {
            "Load_Actual_MW": [10.0, 20.0, 30.0, 40.0],
            "q0.025": [0.0, 10.0, 20.0, 30.0],
            "q0.250": [5.0, 15.0, 25.0, 35.0],
            "q0.500": [10.0, 20.0, 30.0, 40.0],
            "q0.750": [15.0, 25.0, 35.0, 45.0],
            "q0.975": [20.0, 30.0, 40.0, 50.0],
        },
        index=index,
    )

    metrics = lf.evaluate_load_quantile_forecast(forecast, [0.025, 0.25, 0.5, 0.75, 0.975])
    full = metrics.loc[metrics["period"].eq("full")].iloc[0]

    assert full["lqs"] >= 0.0
    assert full["coverage_0.50"] == 1.0
    assert full["coverage_0.95"] == 1.0
    assert full["median_mae"] == 0.0


def test_rolling_load_forecast_outputs_complete_day(tmp_path: Path) -> None:
    index = pd.date_range("2026-01-01T00:00:00+01:00", periods=20 * 96, freq="15min")
    mtu = index.hour * 4 + index.minute // 15
    load = 45_000.0 + 1_000.0 * np.sin(2 * np.pi * mtu / 96) + np.arange(len(index), dtype=float) * 0.5
    dataset = pd.DataFrame(index=index)
    dataset["feature_trend"] = np.arange(len(index), dtype=float)
    dataset["feature_mtu_sin"] = np.sin(2 * np.pi * mtu / 96)
    dataset["Load_Actual_MW"] = load

    config = _config(
        tmp_path,
        test_start=pd.Timestamp("2026-01-12").date(),
        test_end=pd.Timestamp("2026-01-12").date(),
    )
    forecast, runtime = lf.rolling_load_forecast(dataset, config)

    assert len(forecast) == 96
    assert forecast.columns.tolist() == ["Load_Model_MW", "Load_Actual_MW"]
    assert forecast["Load_Model_MW"].notna().all()
    assert runtime.loc[0, "train_rows"] >= 2 * 96


def test_build_load_point_ensemble_averages_sources() -> None:
    index = pd.date_range("2026-01-01T00:00:00+01:00", periods=4, freq="15min")
    actual = pd.Series([10.0, 20.0, 30.0, 40.0], index=index)
    benchmark = actual + 1.0
    source_a = pd.DataFrame(
        {
            "Load_Model_MW": actual + 2.0,
            "Load_Benchmark_MW": benchmark,
            "Load_Actual_MW": actual,
        },
        index=index,
    )
    source_b = pd.DataFrame(
        {
            "Load_Model_MW": actual + 6.0,
            "Load_Benchmark_MW": benchmark,
            "Load_Actual_MW": actual,
        },
        index=index,
    )

    forecast = lf.build_load_point_ensemble({"a": source_a, "b": source_b}, method="mean")

    assert forecast["Load_Model_MW"].tolist() == [14.0, 24.0, 34.0, 44.0]
    assert forecast["Load_Source_a_MW"].tolist() == [12.0, 22.0, 32.0, 42.0]
    assert forecast["Load_Source_b_MW"].tolist() == [16.0, 26.0, 36.0, 46.0]


def test_build_regional_load_features_adds_forecasts_lags_and_summaries(monkeypatch, tmp_path: Path) -> None:
    index = pd.date_range("2026-01-03T00:00:00+01:00", periods=2, freq="15min")
    history_index = index - pd.Timedelta(days=2)
    forecasts = {
        "DE_LU": pd.DataFrame({"load_fc": [100.0, 110.0]}, index=index),
        "A": pd.DataFrame({"load_fc": [10.0, 11.0]}, index=index),
        "B": pd.DataFrame({"load_fc": [20.0, 22.0]}, index=index),
    }
    actuals = {
        "A": pd.DataFrame({"load_actual": [12.0, 13.0]}, index=history_index),
        "B": pd.DataFrame({"load_actual": [21.0, 23.0]}, index=history_index),
    }

    monkeypatch.setattr(lf, "_load_or_fetch_model_entsoe_load_forecast", lambda config: forecasts[config.country_code_entsoe])
    monkeypatch.setattr(lf, "_load_or_fetch_actual_load", lambda config: actuals[config.country_code_entsoe])

    config = _config(
        tmp_path,
        country_code_entsoe="DE_LU",
        include_regional_load_features=True,
        include_regional_load_forecast_features=True,
        include_regional_actual_load_lag_features=True,
        include_regional_partial_load_features=False,
        regional_load_components={"north": "A", "south": "B"},
        actual_load_lag_days=[2],
    )

    features = lf._build_regional_load_features(config, index)

    assert features["regional_load_fc_north"].tolist() == [10.0, 11.0]
    assert features["regional_load_fc_south"].tolist() == [20.0, 22.0]
    assert features["regional_load_fc_sum"].tolist() == [30.0, 33.0]
    assert features["regional_load_fc_missing_to_national"].tolist() == [70.0, 77.0]
    assert features["regional_load_actual_north_lag_d2"].tolist() == [12.0, 13.0]
    assert features["regional_load_actual_lag_d2_sum"].tolist() == [33.0, 36.0]


def test_sum_regional_load_component_forecasts() -> None:
    index = pd.date_range("2026-01-01T00:00:00+01:00", periods=2, freq="15min")
    north = pd.DataFrame(
        {
            "Load_Model_MW": [10.0, 11.0],
            "Load_Benchmark_MW": [9.0, 10.0],
            "Load_Actual_MW": [12.0, 13.0],
        },
        index=index,
    )
    south = pd.DataFrame(
        {
            "Load_Model_MW": [20.0, 21.0],
            "Load_Benchmark_MW": [19.0, 20.0],
            "Load_Actual_MW": [22.0, 23.0],
        },
        index=index,
    )

    forecast = lf.sum_regional_load_component_forecasts({"north": north, "south": south})

    assert forecast["Load_Model_MW"].tolist() == [30.0, 32.0]
    assert forecast["Load_Benchmark_MW"].tolist() == [28.0, 30.0]
    assert forecast["Load_Actual_MW"].tolist() == [34.0, 36.0]
    assert forecast["Load_Model_MW_north"].tolist() == [10.0, 11.0]
    assert forecast["Load_Model_MW_south"].tolist() == [20.0, 21.0]


def test_add_aggregate_residual_load_component_uses_benchmark_residual() -> None:
    index = pd.date_range("2026-01-01T00:00:00+01:00", periods=2, freq="15min")
    north = pd.DataFrame(
        {
            "Load_Model_MW": [10.0, 11.0],
            "Load_Benchmark_MW": [9.0, 10.0],
            "Load_Actual_MW": [12.0, 13.0],
        },
        index=index,
    )
    south = pd.DataFrame(
        {
            "Load_Model_MW": [20.0, 21.0],
            "Load_Benchmark_MW": [19.0, 20.0],
            "Load_Actual_MW": [22.0, 23.0],
        },
        index=index,
    )
    aggregate = pd.DataFrame(
        {
            "Load_Benchmark_MW": [35.0, 37.0],
            "Load_Actual_MW": [40.0, 42.0],
        },
        index=index,
    )

    components = lf.add_aggregate_residual_load_component(
        {"north": north, "south": south},
        aggregate,
        component_name="residual",
    )

    residual = components["residual"]
    assert residual["Load_Benchmark_MW"].tolist() == [7.0, 7.0]
    assert residual["Load_Model_MW"].tolist() == [7.0, 7.0]
    assert residual["Load_Actual_MW"].tolist() == [6.0, 6.0]
    summed = lf.sum_regional_load_component_forecasts(components)
    assert summed["Load_Benchmark_MW"].tolist() == [35.0, 37.0]
    assert summed["Load_Actual_MW"].tolist() == [40.0, 42.0]


def test_regional_load_forecast_pipeline_runs_component_configs(monkeypatch, tmp_path: Path) -> None:
    index = pd.date_range("2026-01-01T00:00:00+01:00", periods=2, freq="15min")
    calls = []

    def fake_run_load_forecast_pipeline(config, save_outputs):
        calls.append(config)
        multiplier = 1.0 if config.country_code_entsoe == "A" else 2.0
        forecast = pd.DataFrame(
            {
                "Load_Model_MW": [10.0, 11.0],
                "Load_Benchmark_MW": [9.0, 10.0],
                "Load_Actual_MW": [12.0, 13.0],
            },
            index=index,
        ) * multiplier
        runtime = pd.DataFrame({"date": [index[0]], "runtime_seconds": [1.0]})
        metrics = pd.DataFrame({"period": ["full"], "rmse": [1.0]})
        return {"forecast": forecast, "runtime": runtime, "metrics": metrics}

    monkeypatch.setattr(lf, "run_load_forecast_pipeline", fake_run_load_forecast_pipeline)
    config = RegionalLoadForecastConfig(
        repo_root=tmp_path,
        export_dir=tmp_path / "regional",
        component_cache_dir=tmp_path / "cache",
        base_config={
            "repo_root": tmp_path,
            "actual_load_file": tmp_path / "actual.csv",
            "entsoe_load_forecast_file": tmp_path / "forecast.csv",
            "icon_dir": tmp_path / "icon",
            "export_dir": tmp_path / "base",
            "include_holiday_features": False,
            "model_type": "ridge",
            "load_target_mode": "entsoe_residual",
            "include_entsoe_forecast_features": True,
            "actual_load_lag_days": [2],
        },
        components=[
            {"name": "a", "country_code_entsoe": "A"},
            {"name": "b", "country_code_entsoe": "B"},
        ],
    )

    result = lf.run_regional_load_forecast_pipeline(config, save_outputs=False)

    assert [call.country_code_entsoe for call in calls] == ["A", "B"]
    assert calls[0].actual_load_file == tmp_path / "cache" / "a_actual_load.csv"
    assert calls[1].entsoe_load_forecast_file == tmp_path / "cache" / "b_entsoe_load_forecast.csv"
    assert result["forecast"]["Load_Model_MW"].tolist() == [30.0, 33.0]
    assert result["runtime"]["component"].tolist() == ["a", "b"]


def test_rolling_load_forecast_predicts_future_day_without_actual_target(tmp_path: Path) -> None:
    index = pd.date_range("2026-01-01T00:00:00+01:00", periods=20 * 96, freq="15min")
    mtu = index.hour * 4 + index.minute // 15
    load = 45_000.0 + 1_000.0 * np.sin(2 * np.pi * mtu / 96) + np.arange(len(index), dtype=float) * 0.5
    dataset = pd.DataFrame(index=index)
    dataset["feature_trend"] = np.arange(len(index), dtype=float)
    dataset["feature_mtu_sin"] = np.sin(2 * np.pi * mtu / 96)
    dataset["Load_Actual_MW"] = load
    target_day = pd.Timestamp("2026-01-12T00:00:00+01:00")
    dataset.loc[dataset.index.normalize() == target_day, "Load_Actual_MW"] = np.nan

    config = _config(
        tmp_path,
        test_start=target_day.date(),
        test_end=target_day.date(),
    )
    forecast, _ = lf.rolling_load_forecast(dataset, config)

    assert len(forecast) == 96
    assert forecast["Load_Model_MW"].notna().all()
    assert forecast["Load_Actual_MW"].isna().all()


def test_rolling_load_forecast_supports_hour_block_models(tmp_path: Path) -> None:
    index = pd.date_range("2026-01-01T00:00:00+01:00", periods=20 * 96, freq="15min")
    mtu = index.hour * 4 + index.minute // 15
    load = 45_000.0 + 1_000.0 * np.sin(2 * np.pi * mtu / 96) + np.arange(len(index), dtype=float) * 0.5
    dataset = pd.DataFrame(index=index)
    dataset["feature_trend"] = np.arange(len(index), dtype=float)
    dataset["feature_mtu_sin"] = np.sin(2 * np.pi * mtu / 96)
    dataset["Load_Actual_MW"] = load

    config = _config(
        tmp_path,
        test_start=pd.Timestamp("2026-01-12").date(),
        test_end=pd.Timestamp("2026-01-12").date(),
        model_granularity="hour_block",
        hour_block_boundaries=[0, 12, 24],
    )
    forecast, runtime = lf.rolling_load_forecast(dataset, config)

    assert len(forecast) == 96
    assert forecast["Load_Model_MW"].notna().all()
    assert runtime.loc[0, "model_granularity"] == "hour_block"
    assert runtime.loc[0, "n_models"] == 2


def test_evaluate_load_forecast_includes_r2() -> None:
    index = pd.date_range("2026-01-01T00:00:00+01:00", periods=4, freq="15min")
    forecast = pd.DataFrame(
        {
            "Load_Model_MW": [1.0, 2.0, 3.0, 4.0],
            "Load_Actual_MW": [1.0, 2.0, 3.0, 5.0],
        },
        index=index,
    )

    metrics = lf.evaluate_load_forecast(forecast)

    assert "r2" in metrics.columns
    assert np.isclose(metrics.loc[metrics["period"].eq("full"), "r2"].iloc[0], 0.8857142857142857)


def test_entsoe_load_forecast_benchmark_frame_and_metrics(monkeypatch, tmp_path: Path) -> None:
    index = pd.date_range("2026-01-01T00:00:00+01:00", periods=2, freq="15min")
    actual = pd.DataFrame({"load_actual": [10.0, 20.0]}, index=index)
    forecast = pd.DataFrame({"load_fc": [11.0, 19.0]}, index=index)

    monkeypatch.setattr(lf, "_load_or_fetch_actual_load", lambda config: actual)
    monkeypatch.setattr(lf, "_load_or_fetch_entsoe_load_forecast", lambda config: forecast)

    config = EntsoeLoadForecastBenchmarkConfig(
        repo_root=tmp_path,
        actual_load_file=tmp_path / "actual.csv",
        forecast_file=tmp_path / "forecast.csv",
        export_dir=tmp_path / "out",
        test_start=None,
        test_end=None,
    )

    result = lf.build_entsoe_load_forecast_benchmark(config)
    metrics = lf.evaluate_load_forecast(result)

    assert result.columns.tolist() == ["Load_Benchmark_MW", "Load_Actual_MW"]
    assert result.loc[index[0], "Load_Benchmark_MW"] == 11.0
    assert metrics.loc[metrics["period"].eq("full"), "model"].iloc[0] == "Benchmark"
    assert metrics.loc[metrics["period"].eq("full"), "mae"].iloc[0] == 1.0


def test_entsoe_load_forecast_benchmark_respects_test_window(monkeypatch, tmp_path: Path) -> None:
    index = pd.date_range("2026-01-01T00:00:00+01:00", periods=3 * 96, freq="15min")
    actual = pd.DataFrame({"load_actual": np.arange(len(index), dtype=float)}, index=index)
    forecast = pd.DataFrame({"load_fc": np.arange(len(index), dtype=float)}, index=index)

    monkeypatch.setattr(lf, "_load_or_fetch_actual_load", lambda config: actual)
    monkeypatch.setattr(lf, "_load_or_fetch_entsoe_load_forecast", lambda config: forecast)

    config = EntsoeLoadForecastBenchmarkConfig(
        repo_root=tmp_path,
        actual_load_file=tmp_path / "actual.csv",
        forecast_file=tmp_path / "forecast.csv",
        export_dir=tmp_path / "out",
        test_start=pd.Timestamp("2026-01-02").date(),
        test_end=pd.Timestamp("2026-01-02").date(),
    )

    result = lf.build_entsoe_load_forecast_benchmark(config)

    assert len(result) == 96
    assert result.index.min() == pd.Timestamp("2026-01-02T00:00:00+01:00")
    assert result.index.max() == pd.Timestamp("2026-01-02T23:45:00+01:00")
