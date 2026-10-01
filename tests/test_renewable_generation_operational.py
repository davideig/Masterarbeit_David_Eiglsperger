from __future__ import annotations

import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from da_price_forecasting.config import RenewableGenerationModelConfig, RenewableGenerationPostprocessConfig
from da_price_forecasting.pipelines.common import save_timestamp_csv
from da_price_forecasting.pipelines.renewable_generation import (
    _apply_rolling_bias_correction,
    _build_partial_actual_generation_features,
    _build_partial_proxy_error_features,
    _build_solar_physics_proxy_baselines,
    _build_wind_cutout_risk_features,
    _build_wind_ensemble_regime_features,
    _load_renewable_proxy,
    _solar_weather_regime_labels,
    _solar_training_row_mask,
    _select_target_features,
    _target_candidate_features,
    _target_baseline_column,
    apply_renewable_generation_sequence_residual_correction,
    build_stacked_renewable_generation_forecast,
    build_renewable_generation_dataset,
)


def test_renewable_generation_dataset_keeps_future_proxy_rows_without_actuals(tmp_path: Path) -> None:
    target_tz = "Europe/Berlin"
    proxy_index = pd.date_range("2026-05-29T00:00:00+02:00", periods=3 * 96, freq="15min")
    actual_index = proxy_index[proxy_index < pd.Timestamp("2026-05-31T00:00:00+02:00")]

    proxy_file = tmp_path / "renewable_proxy.csv"
    actual_file = tmp_path / "actual_generation.csv"
    save_timestamp_csv(
        pd.DataFrame(
            {
                "solar_proxy_mw": np.linspace(0.0, 100.0, len(proxy_index)),
                "wind_proxy_mw": np.linspace(100.0, 200.0, len(proxy_index)),
            },
            index=proxy_index,
        ),
        proxy_file,
    )
    save_timestamp_csv(
        pd.DataFrame(
            {
                "Solar_Actual_MW": np.linspace(0.0, 100.0, len(actual_index)),
                "Wind_Total_Actual_MW": np.linspace(100.0, 200.0, len(actual_index)),
                "Renewable_Total_Actual_MW": np.linspace(100.0, 300.0, len(actual_index)),
            },
            index=actual_index,
        ),
        actual_file,
    )

    config = RenewableGenerationModelConfig(
        repo_root=tmp_path,
        target_tz=target_tz,
        entsoe_start_date=date(2026, 5, 1),
        entsoe_end_date=date(2026, 5, 30),
        actual_generation_file=actual_file,
        renewable_proxy_file=proxy_file,
        unavailability_file=tmp_path / "unavailability.csv",
        icon_dir=tmp_path / "icon",
        export_dir=tmp_path / "export",
        include_dwd_cluster_features=False,
        include_unavailability=False,
        target_columns=["Solar_Actual_MW", "Wind_Total_Actual_MW"],
        test_start=date(2026, 5, 31),
        test_end=date(2026, 5, 31),
    )

    dataset = build_renewable_generation_dataset(config)
    forecast_day = pd.Timestamp("2026-05-31T00:00:00+02:00")

    assert dataset.index.max() == pd.Timestamp("2026-05-31T23:45:00+02:00")
    assert len(dataset.loc[forecast_day:]) == 96
    assert dataset.loc[forecast_day:, "Solar_Actual_MW"].isna().all()
    assert dataset.loc[forecast_day:, "solar_proxy_mw"].notna().all()


def test_renewable_proxy_supports_summary_ensemble_plus_raw_extra(tmp_path: Path) -> None:
    target_tz = "Europe/Berlin"
    index = pd.date_range("2026-05-01T00:00:00+02:00", periods=2, freq="15min")

    base_file = tmp_path / "base.csv"
    provider_a_file = tmp_path / "provider_a.csv"
    provider_b_file = tmp_path / "provider_b.csv"
    raw_extra_file = tmp_path / "eps.csv"

    save_timestamp_csv(pd.DataFrame({"base_proxy_mw": [1.0, 2.0]}, index=index), base_file)
    save_timestamp_csv(pd.DataFrame({"wind_speed_hub_m_s": [4.0, 6.0]}, index=index), provider_a_file)
    save_timestamp_csv(pd.DataFrame({"wind_speed_hub_m_s": [8.0, 10.0]}, index=index), provider_b_file)
    save_timestamp_csv(pd.DataFrame({"u10_ens_std_m_s": [0.3, 0.4]}, index=index), raw_extra_file)

    config = RenewableGenerationModelConfig(
        repo_root=tmp_path,
        target_tz=target_tz,
        renewable_proxy_file=base_file,
        extra_renewable_proxy_files=[provider_a_file, provider_b_file],
        extra_renewable_proxy_ensemble_mode="summary",
        extra_renewable_proxy_ensemble_prefix="ens_",
        raw_extra_renewable_proxy_files=[raw_extra_file],
        raw_extra_renewable_proxy_prefixes=["eps_"],
        actual_generation_file=tmp_path / "actual.csv",
        unavailability_file=tmp_path / "unavailable.csv",
        icon_dir=tmp_path / "icon",
        export_dir=tmp_path / "export",
    )

    proxy = _load_renewable_proxy(config)

    assert "base_proxy_mw" in proxy.columns
    assert "ens_mean_wind_speed_hub_m_s" in proxy.columns
    assert "ens_std_wind_speed_hub_m_s" in proxy.columns
    assert "eps_u10_ens_std_m_s" in proxy.columns
    assert np.allclose(proxy["ens_mean_wind_speed_hub_m_s"], [6.0, 8.0])
    assert np.allclose(proxy["ens_std_wind_speed_hub_m_s"], [2.0, 2.0])
    assert np.allclose(proxy["eps_u10_ens_std_m_s"], [0.3, 0.4])


def test_partial_actual_generation_features_use_previous_morning() -> None:
    target_tz = "Europe/Berlin"
    actual_index = pd.date_range("2026-04-20T00:00:00+02:00", "2026-04-30T23:45:00+02:00", freq="15min")
    actual = pd.DataFrame(
        {
            "Solar_50Hertz_Actual_MW": np.arange(len(actual_index), dtype=float),
        },
        index=actual_index,
    )
    target_index = pd.date_range("2026-04-29T00:00:00+02:00", periods=96, freq="15min")

    features = _build_partial_actual_generation_features(
        actual,
        target_index,
        columns=["Solar_50Hertz_Actual_MW"],
        reference_day=1,
        comparison_lag_days=7,
        morning_end_hour=10,
        morning_end_minute=0,
    )

    source = actual.loc[
        pd.Timestamp("2026-04-28T00:00:00+02:00"):pd.Timestamp("2026-04-28T10:00:00+02:00"),
        "Solar_50Hertz_Actual_MW",
    ]
    comparison = actual.loc[
        pd.Timestamp("2026-04-21T00:00:00+02:00"):pd.Timestamp("2026-04-21T10:00:00+02:00"),
        "Solar_50Hertz_Actual_MW",
    ]
    timestamp = pd.Timestamp("2026-04-29T12:00:00+02:00")

    assert np.isclose(features.loc[timestamp, "partial_gen_solar_50hertz_d1_00_1000_mean"], source.mean())
    assert np.isclose(features.loc[timestamp, "partial_gen_solar_50hertz_d1_00_1000_last"], source.iloc[-1])
    assert np.isclose(
        features.loc[timestamp, "partial_gen_solar_50hertz_d1_00_1000_mean_diff_d7"],
        source.mean() - comparison.mean(),
    )


def test_wind_ensemble_regime_features_interact_provider_spread_with_speed() -> None:
    index = pd.date_range("2026-04-01T00:00:00+02:00", periods=2, freq="15min")
    proxy = pd.DataFrame(
        {
            "ens_mean_wind_onshore_north_speed_120m_cap_weighted_m_s": [10.0, 16.0],
            "ens_std_wind_onshore_north_speed_120m_cap_weighted_m_s": [0.5, 2.0],
            "ens_mean_wind_onshore_north_proxy_mw": [1000.0, 3000.0],
            "ens_std_wind_onshore_north_proxy_mw": [100.0, 400.0],
            "ens_mean_wind_offshore_north_speed_120m_cap_weighted_m_s": [8.0, 14.0],
            "ens_std_wind_offshore_north_speed_120m_cap_weighted_m_s": [0.2, 1.5],
        },
        index=index,
    )

    features = _build_wind_ensemble_regime_features(proxy, speed_thresholds=[12.0])

    assert "wind_ensemble_regime_onshore_provider_speed_std_mean_x_ge_12" in features.columns
    assert np.isclose(features.iloc[0]["wind_ensemble_regime_onshore_provider_speed_std_mean_x_ge_12"], 0.0)
    assert np.isclose(features.iloc[1]["wind_ensemble_regime_onshore_provider_speed_std_mean_x_ge_12"], 2.0)
    assert np.isclose(features.iloc[1]["wind_ensemble_regime_onshore_provider_proxy_std_sum_x_excess_ge_12"], 1600.0)
    assert "wind_ensemble_regime_offshore_speed_mean_ge_12" in features.columns


def test_wind_cutout_risk_features_softly_score_rated_regime() -> None:
    index = pd.date_range("2026-04-01T00:00:00+02:00", periods=2, freq="15min")
    proxy = pd.DataFrame(
        {
            "ens_mean_wind_onshore_north_speed_120m_cap_weighted_m_s": [8.0, 14.0],
            "ens_std_wind_onshore_north_speed_120m_cap_weighted_m_s": [0.5, 2.0],
            "ens_mean_wind_onshore_south_speed_120m_cap_weighted_m_s": [7.0, 10.0],
            "ens_std_wind_onshore_south_speed_120m_cap_weighted_m_s": [0.4, 1.0],
            "ens_mean_wind_onshore_north_proxy_mw": [1000.0, 5000.0],
            "ens_std_wind_onshore_north_proxy_mw": [50.0, 500.0],
            "ens_mean_wind_offshore_north_speed_hub_cap_weighted_m_s": [11.0, 18.0],
            "ens_std_wind_offshore_north_speed_hub_cap_weighted_m_s": [1.0, 3.0],
        },
        index=index,
    )

    features = _build_wind_cutout_risk_features(
        proxy,
        speed_thresholds=[12.0, 15.0],
        std_floor_m_s=0.35,
    )

    assert "wind_cutout_risk_onshore_soft_share_ge_12" in features.columns
    assert "wind_cutout_risk_offshore_soft_share_ge_15" in features.columns
    assert features.iloc[1]["wind_cutout_risk_onshore_soft_share_ge_12"] > features.iloc[0][
        "wind_cutout_risk_onshore_soft_share_ge_12"
    ]
    assert features.iloc[1]["wind_cutout_risk_onshore_proxy_x_soft_ge_12_mw"] > features.iloc[0][
        "wind_cutout_risk_onshore_proxy_x_soft_ge_12_mw"
    ]


def test_partial_proxy_error_features_use_previous_morning_actual_vs_proxy(tmp_path: Path) -> None:
    actual_index = pd.date_range("2026-04-20T00:00:00+02:00", "2026-04-30T23:45:00+02:00", freq="15min")
    actual = pd.DataFrame(
        {
            "Wind_Onshore_Actual_MW": np.full(len(actual_index), 100.0),
        },
        index=actual_index,
    )
    proxy = pd.DataFrame(
        {
            "ens_mean_wind_onshore_north_proxy_mw": np.full(len(actual_index), 40.0),
            "ens_mean_wind_onshore_south_proxy_mw": np.full(len(actual_index), 50.0),
            "ens_std_wind_onshore_north_proxy_mw": np.full(len(actual_index), 5.0),
        },
        index=actual_index,
    )
    target_index = pd.date_range("2026-04-29T00:00:00+02:00", periods=96, freq="15min")
    config = RenewableGenerationModelConfig(
        repo_root=tmp_path,
        renewable_proxy_file=tmp_path / "proxy.csv",
        actual_generation_file=tmp_path / "actual.csv",
        unavailability_file=tmp_path / "unavailable.csv",
        icon_dir=tmp_path / "icon",
        export_dir=tmp_path / "export",
        include_partial_proxy_error_features=True,
        partial_proxy_error_columns=["Wind_Onshore_Actual_MW"],
        partial_proxy_error_proxy_column_patterns={
            "Wind_Onshore_Actual_MW": ["ens_mean_wind_onshore"],
        },
    )

    features = _build_partial_proxy_error_features(actual, proxy, target_index, config)
    timestamp = pd.Timestamp("2026-04-29T12:00:00+02:00")

    assert np.isclose(
        features.loc[timestamp, "partial_proxy_error_wind_onshore_d1_00_1000_actual_mean"],
        100.0,
    )
    assert np.isclose(
        features.loc[timestamp, "partial_proxy_error_wind_onshore_d1_00_1000_proxy_mean"],
        90.0,
    )
    assert np.isclose(
        features.loc[timestamp, "partial_proxy_error_wind_onshore_d1_00_1000_error_mean"],
        10.0,
    )
    assert np.isclose(
        features.loc[timestamp, "partial_proxy_error_wind_onshore_d1_00_1000_ratio_mean"],
        100.0 / 90.0,
    )


def test_renewable_forecast_stacking_builds_component_and_total_forecasts(tmp_path: Path) -> None:
    index = pd.date_range("2026-04-01T00:00:00+02:00", periods=3 * 96, freq="15min")
    provider_a_file = tmp_path / "provider_a.csv"
    provider_b_file = tmp_path / "provider_b.csv"
    actual_onshore = np.linspace(100.0, 300.0, len(index))
    actual_offshore = np.linspace(20.0, 80.0, len(index))

    save_timestamp_csv(
        pd.DataFrame(
            {
                "Wind_Onshore_Model_MW": actual_onshore + 5.0,
                "Wind_Onshore_Actual_MW": actual_onshore,
                "Wind_Offshore_Model_MW": actual_offshore + 2.0,
                "Wind_Offshore_Actual_MW": actual_offshore,
            },
            index=index,
        ),
        provider_a_file,
    )
    save_timestamp_csv(
        pd.DataFrame(
            {
                "Wind_Onshore_Model_MW": actual_onshore - 15.0,
                "Wind_Onshore_Actual_MW": actual_onshore,
                "Wind_Offshore_Model_MW": actual_offshore - 4.0,
                "Wind_Offshore_Actual_MW": actual_offshore,
            },
            index=index,
        ),
        provider_b_file,
    )

    config = RenewableGenerationPostprocessConfig(
        repo_root=tmp_path,
        target_tz="Europe/Berlin",
        include_forecast_stacking=True,
        stacking_forecast_files={
            "provider_a": provider_a_file,
            "provider_b": provider_b_file,
        },
        stacking_model_type="simple_average",
        export_dir=tmp_path / "export",
    )

    stacked = build_stacked_renewable_generation_forecast(config)

    assert "Wind_Onshore_Model_MW" in stacked.columns
    assert "Wind_Offshore_Model_MW" in stacked.columns
    assert "Wind_Total_Model_MW" in stacked.columns
    assert np.allclose(
        stacked["Wind_Total_Model_MW"],
        stacked["Wind_Onshore_Model_MW"] + stacked["Wind_Offshore_Model_MW"],
    )
    assert np.allclose(stacked["Wind_Onshore_Model_MW"], actual_onshore - 5.0)


def test_sequence_residual_correction_uses_past_complete_days(tmp_path: Path) -> None:
    index_parts: list[pd.DatetimeIndex] = []
    for day in pd.date_range("2026-04-01T00:00:00+02:00", periods=5, freq="D"):
        index_parts.append(pd.date_range(day, periods=4, freq="15min"))
    index = index_parts[0].append(index_parts[1:])

    onshore_residual = np.array([10.0, 20.0, 30.0, 40.0])
    offshore_residual = np.array([2.0, 4.0, 6.0, 8.0])
    onshore_actual: list[float] = []
    offshore_actual: list[float] = []
    for day_number in range(5):
        onshore_actual.extend((100.0 + day_number * 5.0 + np.array([0.0, 20.0, 10.0, 5.0])).tolist())
        offshore_actual.extend((20.0 + day_number * 2.0 + np.array([0.0, 4.0, 2.0, 1.0])).tolist())
    onshore_actual_array = np.array(onshore_actual)
    offshore_actual_array = np.array(offshore_actual)
    onshore_model = onshore_actual_array - np.tile(onshore_residual, 5)
    offshore_model = offshore_actual_array - np.tile(offshore_residual, 5)
    forecast = pd.DataFrame(
        {
            "Wind_Onshore_Model_MW": onshore_model,
            "Wind_Onshore_Actual_MW": onshore_actual_array,
            "Wind_Offshore_Model_MW": offshore_model,
            "Wind_Offshore_Actual_MW": offshore_actual_array,
        },
        index=index,
    )
    forecast["Wind_Total_Model_MW"] = forecast["Wind_Onshore_Model_MW"] + forecast["Wind_Offshore_Model_MW"]
    forecast["Wind_Total_Actual_MW"] = (
        forecast["Wind_Onshore_Actual_MW"] + forecast["Wind_Offshore_Actual_MW"]
    )

    config = RenewableGenerationPostprocessConfig(
        repo_root=tmp_path,
        target_tz="Europe/Berlin",
        forecast_file=tmp_path / "forecast.csv",
        export_dir=tmp_path / "export",
        include_sequence_residual_correction=True,
        sequence_residual_n_steps=4,
        sequence_residual_min_train_days=2,
        sequence_residual_window_days=10,
        sequence_residual_ridge_alpha=10.0,
        sequence_residual_shrinkage=1.0,
        target_availability_lag_days=0,
        target_upper_bounds_mw={
            "Wind_Onshore": 1000.0,
            "Wind_Offshore": 1000.0,
            "Wind_Total": 2000.0,
        },
    )

    corrected = apply_renewable_generation_sequence_residual_correction(forecast, config)
    last_day = corrected.loc[pd.Timestamp("2026-04-05T00:00:00+02:00") :]

    base_error = np.abs(
        forecast.loc[last_day.index, "Wind_Total_Model_MW"]
        - forecast.loc[last_day.index, "Wind_Total_Actual_MW"]
    ).mean()
    corrected_error = np.abs(last_day["Wind_Total_Model_MW"] - last_day["Wind_Total_Actual_MW"]).mean()

    assert "Wind_Onshore_PointBase_MW" in corrected.columns
    assert "Wind_Offshore_PointBase_MW" in corrected.columns
    assert corrected_error < base_error


def test_rolling_bias_correction_supports_explicit_overrides(tmp_path: Path) -> None:
    index_history = pd.date_range("2026-04-20T00:00:00+02:00", periods=96, freq="15min")
    index_current = pd.date_range("2026-04-21T00:00:00+02:00", periods=96, freq="15min")
    history = pd.DataFrame(
        {
            "Solar_Model_MW": np.full(len(index_history), 110.0),
            "Solar_Actual_MW": np.full(len(index_history), 100.0),
        },
        index=index_history,
    )
    current = pd.DataFrame(
        {
            "Solar_Model_MW": np.full(len(index_current), 120.0),
            "Solar_Actual_MW": np.full(len(index_current), 100.0),
        },
        index=index_current,
    )
    config = RenewableGenerationModelConfig(
        repo_root=tmp_path,
        renewable_proxy_file=tmp_path / "proxy.csv",
        actual_generation_file=tmp_path / "actual.csv",
        unavailability_file=tmp_path / "unavailable.csv",
        icon_dir=tmp_path / "icon",
        export_dir=tmp_path / "export",
    )

    _apply_rolling_bias_correction(
        current,
        [history],
        day=pd.Timestamp("2026-04-21T00:00:00+02:00"),
        pred_col="Solar_Model_MW",
        true_col="Solar_Actual_MW",
        upper_bound=None,
        config=config,
        window_days=7,
        group="global",
        min_observations=24,
        shrinkage=0.5,
    )

    assert np.allclose(current["Solar_Model_MW"], 115.0)


def test_rolling_bias_correction_respects_target_availability_cutoff(tmp_path: Path) -> None:
    # Forecasting delivery day D on the morning of D-1: only D-1 actuals up to the 10:00
    # cutoff are operationally available, so the rolling bias correction must ignore the
    # afternoon/evening of the previous day even though those rows exist in the history.
    index_history = pd.date_range("2026-04-20T00:00:00+02:00", periods=96, freq="15min")
    morning = index_history <= pd.Timestamp("2026-04-20T10:00:00+02:00")
    history = pd.DataFrame(
        {
            "Wind_Total_Model_MW": np.where(morning, 110.0, 2100.0),
            "Wind_Total_Actual_MW": np.full(len(index_history), 100.0),
        },
        index=index_history,
    )
    index_current = pd.date_range("2026-04-21T00:00:00+02:00", periods=96, freq="15min")
    current = pd.DataFrame(
        {
            "Wind_Total_Model_MW": np.full(len(index_current), 120.0),
            "Wind_Total_Actual_MW": np.full(len(index_current), 100.0),
        },
        index=index_current,
    )
    config = RenewableGenerationModelConfig(
        repo_root=tmp_path,
        renewable_proxy_file=tmp_path / "proxy.csv",
        actual_generation_file=tmp_path / "actual.csv",
        unavailability_file=tmp_path / "unavailable.csv",
        icon_dir=tmp_path / "icon",
        export_dir=tmp_path / "export",
        target_availability_lag_days=1,
        target_availability_cutoff_hour=10,
    )

    _apply_rolling_bias_correction(
        current,
        [history],
        day=pd.Timestamp("2026-04-21T00:00:00+02:00"),
        pred_col="Wind_Total_Model_MW",
        true_col="Wind_Total_Actual_MW",
        upper_bound=None,
        config=config,
        window_days=7,
        group="global",
        min_observations=24,
        shrinkage=1.0,
    )

    # Only the morning bias (+10) is available. The leaked full-day mean (~1150) would
    # instead drive the corrected prediction to its 0 floor, so 110 proves the cutoff holds.
    assert np.allclose(current["Wind_Total_Model_MW"], 110.0)


def test_solar_physics_proxy_baselines_follow_tso_regions(tmp_path: Path) -> None:
    index = pd.date_range("2026-05-01T10:00:00+02:00", periods=2, freq="15min")
    proxy = pd.DataFrame(
        {
            "solar_50hertz_proxy_mw": [1000.0, 2000.0],
            "solar_50hertz_irradiance_cap_weighted_W_m2": [800.0, 900.0],
            "solar_50hertz_t2m_cap_weighted_K": [293.15, 303.15],
            "solar_amprion_proxy_mw": [500.0, 600.0],
            "solar_amprion_irradiance_cap_weighted_W_m2": [700.0, 800.0],
            "solar_amprion_t2m_cap_weighted_K": [288.15, 293.15],
        },
        index=index,
    )
    config = RenewableGenerationModelConfig(
        repo_root=tmp_path,
        renewable_proxy_file=tmp_path / "proxy.csv",
        actual_generation_file=tmp_path / "actual.csv",
        unavailability_file=tmp_path / "unavailable.csv",
        icon_dir=tmp_path / "icon",
        export_dir=tmp_path / "export",
    )

    features = _build_solar_physics_proxy_baselines(proxy, config)

    assert "solar_50hertz_physics_proxy_mw" in features.columns
    assert "solar_amprion_physics_proxy_mw" in features.columns
    assert np.allclose(
        features["Renewable_Solar_Physics_Proxy_MW"],
        features["solar_50hertz_physics_proxy_mw"] + features["solar_amprion_physics_proxy_mw"],
    )
    assert not np.allclose(
        features["solar_50hertz_physics_proxy_mw"],
        proxy["solar_50hertz_proxy_mw"],
    )


def test_solar_physics_baseline_column_maps_control_area_target(tmp_path: Path) -> None:
    config = RenewableGenerationModelConfig(
        repo_root=tmp_path,
        renewable_proxy_file=tmp_path / "proxy.csv",
        actual_generation_file=tmp_path / "actual.csv",
        unavailability_file=tmp_path / "unavailable.csv",
        icon_dir=tmp_path / "icon",
        export_dir=tmp_path / "export",
        target_baseline_mode="solar_physics_proxy",
    )

    baseline = _target_baseline_column(
        "Solar_TenneT_Actual_MW",
        config,
        [
            "solar_50hertz_physics_proxy_mw",
            "solar_tennet_physics_proxy_mw",
            "wind_onshore_proxy_mw",
        ],
    )

    assert baseline == "solar_tennet_physics_proxy_mw"
    assert _target_baseline_column("Wind_Total_Actual_MW", config, ["wind_onshore_proxy_mw"]) is None


def test_target_feature_selection_skips_constant_columns(tmp_path: Path) -> None:
    index = pd.RangeIndex(6)
    X_train = pd.DataFrame(
        {
            "constant": np.ones(len(index)),
            "all_nan": np.full(len(index), np.nan),
            "signal": np.arange(len(index), dtype=float),
            "inverse_signal": -np.arange(len(index), dtype=float),
        },
        index=index,
    )
    y_train = pd.Series(np.arange(len(index), dtype=float), index=index)
    config = RenewableGenerationModelConfig(
        repo_root=tmp_path,
        renewable_proxy_file=tmp_path / "proxy.csv",
        actual_generation_file=tmp_path / "actual.csv",
        unavailability_file=tmp_path / "unavailable.csv",
        icon_dir=tmp_path / "icon",
        export_dir=tmp_path / "export",
        max_features_per_target=2,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        selected = _select_target_features(X_train, y_train, config)

    assert selected == ["signal", "inverse_signal"]
    assert not [warning for warning in caught if issubclass(warning.category, RuntimeWarning)]


def test_solar_own_region_feature_scope_keeps_global_context(tmp_path: Path) -> None:
    config = RenewableGenerationModelConfig(
        repo_root=tmp_path,
        renewable_proxy_file=tmp_path / "proxy.csv",
        actual_generation_file=tmp_path / "actual.csv",
        unavailability_file=tmp_path / "unavailable.csv",
        icon_dir=tmp_path / "icon",
        export_dir=tmp_path / "export",
        target_feature_mode="technology_specific",
        solar_target_region_scope="own_region_plus_global",
    )
    features = [
        "tod_sin",
        "solar_50hertz_proxy_mw",
        "solar_amprion_proxy_mw",
        "solar_geom_50hertz_elevation_deg",
        "solar_geom_tennet_elevation_deg",
        "solar_phys_clear_sky_index_mean",
        "Renewable_Solar_Proxy_MW",
    ]

    selected = _target_candidate_features(features, "Solar_50Hertz_Actual_MW", config)

    assert "tod_sin" in selected
    assert "solar_50hertz_proxy_mw" in selected
    assert "solar_geom_50hertz_elevation_deg" in selected
    assert "solar_phys_clear_sky_index_mean" in selected
    assert "Renewable_Solar_Proxy_MW" in selected
    assert "solar_amprion_proxy_mw" not in selected
    assert "solar_geom_tennet_elevation_deg" not in selected


def test_solar_training_row_mask_filters_daylight_and_suspicious_rows(tmp_path: Path) -> None:
    index = pd.RangeIndex(4)
    y_train = pd.Series([10.0, 30.0, 5.0, 80.0], index=index)
    X_train_all = pd.DataFrame(
        {
            "solar_geom_50hertz_clear_sky_proxy": [0.0, 0.4, 0.5, 0.6],
            "solar_geom_50hertz_elevation_deg": [-5.0, 12.0, 25.0, 35.0],
            "solar_50hertz_physics_proxy_mw": [0.0, 50.0, 100.0, 100.0],
        },
        index=index,
    )
    config = RenewableGenerationModelConfig(
        repo_root=tmp_path,
        renewable_proxy_file=tmp_path / "proxy.csv",
        actual_generation_file=tmp_path / "actual.csv",
        unavailability_file=tmp_path / "unavailable.csv",
        icon_dir=tmp_path / "icon",
        export_dir=tmp_path / "export",
        solar_training_clear_sky_min=0.2,
        solar_training_elevation_min_deg=5.0,
        solar_suspicious_training_filter=True,
        solar_suspicious_actual_to_baseline_min=0.35,
        solar_suspicious_baseline_min_capacity_share=0.0,
    )

    mask = _solar_training_row_mask(
        "Solar_50Hertz_Actual_MW",
        y_train,
        X_train_all,
        config,
        installed_capacity_mw={"Solar_50Hertz": 1000.0},
        baseline_column="solar_50hertz_physics_proxy_mw",
    )

    assert mask.tolist() == [False, True, False, True]


def test_solar_weather_regime_labels_use_tso_cloud_and_irradiance(tmp_path: Path) -> None:
    index = pd.RangeIndex(3)
    X = pd.DataFrame(
        {
            "om_solar_50hertz_cloud_cover_cap_weighted": [10.0, 55.0, 90.0],
            "solar_50hertz_irradiance_cap_weighted_W_m2": [720.0, 450.0, 120.0],
            "solar_geom_east_clear_sky_proxy": [0.8, 0.8, 0.8],
        },
        index=index,
    )
    config = RenewableGenerationModelConfig(
        repo_root=tmp_path,
        renewable_proxy_file=tmp_path / "proxy.csv",
        actual_generation_file=tmp_path / "actual.csv",
        unavailability_file=tmp_path / "unavailable.csv",
        icon_dir=tmp_path / "icon",
        export_dir=tmp_path / "export",
        solar_weather_regime_split=True,
        solar_weather_regime_clear_cloud_max=35.0,
        solar_weather_regime_overcast_cloud_min=75.0,
        solar_weather_regime_clear_irradiance_ratio_min=0.7,
        solar_weather_regime_overcast_irradiance_ratio_max=0.35,
    )

    labels = _solar_weather_regime_labels("Solar_50Hertz_Actual_MW", X, config)

    assert labels is not None
    assert labels.tolist() == ["clear", "mixed", "overcast"]
