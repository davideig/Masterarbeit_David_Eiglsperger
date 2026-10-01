from __future__ import annotations

import pandas as pd

from da_price_forecasting.config import CovariateConfig, LearOperationalConfig
from da_price_forecasting.features.covariates import (
    _build_foreign_price_spreads,
    build_timestamp_covariates,
    build_daily_scalar_covariate_features,
    build_daily_summary_covariate_features,
    build_daily_vector_covariate_features,
    load_load_forecast_covariates,
)
from da_price_forecasting.pipelines import lear


def _config(**overrides) -> LearOperationalConfig:
    payload = {
        "repo_root": ".",
        "target_tz": "Europe/Berlin",
        "entsoe_start_date": "2025-01-01T00:00:00+01:00",
        "entsoe_end_date": "2025-01-03T00:00:00+01:00",
        "test_start": "2025-01-02T00:00:00+01:00",
        "test_end": "2025-01-03T00:00:00+01:00",
    } | overrides
    return LearOperationalConfig.model_validate(payload)


def test_build_daily_scalar_covariate_features_uses_last_value_per_day() -> None:
    index = pd.date_range("2025-01-01T00:00:00+01:00", periods=2 * 96, freq="15min")
    covariates = pd.DataFrame(
        {
            "gas_ttf": range(len(index)),
            "co2_eua": range(1000, 1000 + len(index)),
        },
        index=index,
    )
    daily_index = pd.date_range("2025-01-01T00:00:00+01:00", periods=2, freq="D")

    features = build_daily_scalar_covariate_features(covariates, daily_index)

    assert features.index.equals(daily_index.rename("date"))
    assert features.loc[daily_index[0], "gas_ttf"] == 95.0
    assert features.loc[daily_index[1], "co2_eua"] == 1191.0


def test_build_daily_vector_covariate_features_keeps_mtu_shape() -> None:
    index = pd.date_range("2025-01-01T00:00:00+01:00", periods=2 * 96, freq="15min")
    covariates = pd.DataFrame(
        {
            "NTC_Net_MW": range(len(index)),
            "Unavail_Total_MW": range(1000, 1000 + len(index)),
        },
        index=index,
    )
    daily_index = pd.date_range("2025-01-01T00:00:00+01:00", periods=2, freq="D")

    features = build_daily_vector_covariate_features(covariates, daily_index)

    assert features.index.equals(daily_index.rename("date"))
    assert features.shape == (2, 192)
    assert features.loc[daily_index[0], "NTC_Net_MW_mtu_00"] == 0.0
    assert features.loc[daily_index[0], "NTC_Net_MW_mtu_95"] == 95.0
    assert features.loc[daily_index[1], "Unavail_Total_MW_mtu_00"] == 1096.0
    assert features.loc[daily_index[1], "Unavail_Total_MW_mtu_95"] == 1191.0


def test_build_foreign_price_spreads_subtracts_exaa() -> None:
    index = pd.date_range("2025-01-01T00:00:00+01:00", periods=4, freq="15min")
    foreign_prices = pd.DataFrame({"price_da_CH": [70.0, 72.0, 75.0, 80.0]}, index=index)
    exaa_prices = pd.DataFrame({"price_exaa": [68.0, 70.0, 74.0, 79.0]}, index=index)

    spreads = _build_foreign_price_spreads(foreign_prices, exaa_prices)

    assert spreads.columns.tolist() == ["price_spread_CH_minus_exaa"]
    assert spreads["price_spread_CH_minus_exaa"].tolist() == [2.0, 2.0, 1.0, 1.0]


def test_custom_load_forecast_file_is_loaded_as_load_fc(tmp_path) -> None:
    index = pd.date_range("2025-01-01T00:00:00+01:00", periods=4, freq="15min")
    path = tmp_path / "load_forecast.csv"
    pd.DataFrame({"timestamp": index.astype(str), "Load_Model_MW": [10.0, 11.0, 12.0, 13.0]}).to_csv(
        path,
        index=False,
    )
    config = CovariateConfig(load_forecast_file=str(path))

    loaded = load_load_forecast_covariates(
        config,
        start_day=pd.Timestamp("2025-01-01", tz="Europe/Berlin"),
        end_day=pd.Timestamp("2025-01-01", tz="Europe/Berlin"),
    )

    assert loaded.columns.tolist() == ["load_fc"]
    assert loaded["load_fc"].tolist() == [10.0, 11.0, 12.0, 13.0]


def test_renewable_forecast_file_aliases_model_columns_and_drops_actuals(tmp_path) -> None:
    index = pd.date_range("2025-01-01T00:00:00+01:00", periods=4, freq="15min", name="timestamp")
    path = tmp_path / "renewable_forecast.csv"
    pd.DataFrame(
        {
            "Solar_Model_MW": [1.0, 2.0, 3.0, 4.0],
            "Solar_Actual_MW": [100.0, 100.0, 100.0, 100.0],
            "Wind_Total_Model_MW": [5.0, 6.0, 7.0, 8.0],
            "Wind_Total_Actual_MW": [200.0, 200.0, 200.0, 200.0],
            "Renewable_Total_Model_MW": [6.0, 8.0, 10.0, 12.0],
            "Renewable_Total_Actual_MW": [300.0, 300.0, 300.0, 300.0],
        },
        index=index,
    ).to_csv(path)
    config = CovariateConfig(
        covariates=["renewable_generation_proxy"],
        renewable_proxy_file=str(path),
        renewable_proxy_columns=[
            "Renewable_Solar_Proxy_MW",
            "Renewable_Wind_Proxy_MW",
            "Renewable_Total_Proxy_MW",
        ],
    )

    covariates = build_timestamp_covariates(
        config,
        start_day=pd.Timestamp("2025-01-01", tz="Europe/Berlin"),
        end_day=pd.Timestamp("2025-01-01", tz="Europe/Berlin"),
    )

    assert covariates.columns.tolist() == [
        "Renewable_Solar_Proxy_MW",
        "Renewable_Wind_Proxy_MW",
        "Renewable_Total_Proxy_MW",
    ]
    assert covariates["Renewable_Total_Proxy_MW"].tolist() == [6.0, 8.0, 10.0, 12.0]


def test_multiple_renewable_forecast_files_are_merged_and_total_is_derived(tmp_path) -> None:
    index = pd.date_range("2025-01-01T00:00:00+01:00", periods=4, freq="15min", name="timestamp")
    solar_path = tmp_path / "solar_forecast.csv"
    wind_path = tmp_path / "wind_forecast.csv"
    pd.DataFrame(
        {
            "Solar_Model_MW": [1.0, 2.0, 3.0, 4.0],
            "Solar_Actual_MW": [100.0, 100.0, 100.0, 100.0],
        },
        index=index,
    ).to_csv(solar_path)
    pd.DataFrame(
        {
            "Wind_Total_Model_MW": [5.0, 6.0, 7.0, 8.0],
            "Wind_Total_Actual_MW": [200.0, 200.0, 200.0, 200.0],
        },
        index=index,
    ).to_csv(wind_path)
    config = CovariateConfig(
        covariates=["renewable_generation_proxy"],
        renewable_proxy_files=[str(solar_path), str(wind_path)],
        renewable_proxy_columns=[
            "Renewable_Solar_Proxy_MW",
            "Renewable_Wind_Proxy_MW",
            "Renewable_Total_Proxy_MW",
        ],
    )

    covariates = build_timestamp_covariates(
        config,
        start_day=pd.Timestamp("2025-01-01", tz="Europe/Berlin"),
        end_day=pd.Timestamp("2025-01-01", tz="Europe/Berlin"),
    )

    assert covariates.columns.tolist() == [
        "Renewable_Solar_Proxy_MW",
        "Renewable_Wind_Proxy_MW",
        "Renewable_Total_Proxy_MW",
    ]
    assert covariates["Renewable_Total_Proxy_MW"].tolist() == [6.0, 8.0, 10.0, 12.0]


def test_reserve_market_file_is_loaded_with_configured_columns(tmp_path) -> None:
    index = pd.date_range("2025-01-01T00:00:00+01:00", periods=4, freq="15min", name="timestamp")
    path = tmp_path / "reserve_market.csv"
    pd.DataFrame(
        {
            "timestamp": index.astype(str),
            "reserve_fcr_germany_demand_mw": [580.0, 580.0, 580.0, 580.0],
            "reserve_afrr_pos_avg_capacity_price_eur_mw_h": [18.0, 18.0, 18.0, 18.0],
            "ignored_metadata": ["cached", "cached", "cached", "cached"],
        }
    ).to_csv(path, index=False)
    config = CovariateConfig(
        covariates=["reserve_market"],
        reserve_market_file=str(path),
        reserve_market_columns=[
            "reserve_fcr_germany_demand_mw",
            "reserve_afrr_pos_avg_capacity_price_eur_mw_h",
        ],
    )

    covariates = build_timestamp_covariates(
        config,
        start_day=pd.Timestamp("2025-01-01", tz="Europe/Berlin"),
        end_day=pd.Timestamp("2025-01-01", tz="Europe/Berlin"),
    )

    assert covariates.columns.tolist() == [
        "reserve_fcr_germany_demand_mw",
        "reserve_afrr_pos_avg_capacity_price_eur_mw_h",
    ]
    assert covariates["reserve_fcr_germany_demand_mw"].tolist() == [580.0, 580.0, 580.0, 580.0]


def test_build_daily_summary_covariate_features_keeps_compact_scarcity_stats() -> None:
    index = pd.date_range("2025-01-01T00:00:00+01:00", periods=96, freq="15min")
    covariates = pd.DataFrame({"load_fc": range(96)}, index=index)
    daily_index = pd.date_range("2025-01-01T00:00:00+01:00", periods=1, freq="D")

    features = build_daily_summary_covariate_features(covariates, daily_index, prefix="scarcity_")

    day = daily_index[0]
    assert features.loc[day, "scarcity_load_fc_mean"] == 47.5
    assert features.loc[day, "scarcity_load_fc_min"] == 0.0
    assert features.loc[day, "scarcity_load_fc_max"] == 95.0
    assert features.loc[day, "scarcity_load_fc_range"] == 95.0
    assert features.loc[day, "scarcity_load_fc_ramp_1h_abs_max"] == 4.0
    assert features.loc[day, "scarcity_load_fc_morning_mean"] == 35.5
    assert features.loc[day, "scarcity_load_fc_evening_max"] == 75.0


def test_lear_extra_covariates_filter_legacy_exaa_and_load(monkeypatch) -> None:
    calls = []
    daily_index = pd.date_range("2025-01-01T00:00:00+01:00", periods=2, freq="D")
    timestamp_index = pd.date_range("2025-01-01T00:00:00+01:00", periods=2 * 96, freq="15min")

    def fake_build_timestamp_covariates(covariate_config, **kwargs):
        calls.append((covariate_config, kwargs))
        return pd.DataFrame(
            {
                "gas_ttf": 40.0,
                "co2_eua": 70.0,
            },
            index=timestamp_index,
        )

    monkeypatch.setattr(lear, "build_timestamp_covariates", fake_build_timestamp_covariates)
    config = _config(
        features={
            "covariates": ["exaa", "load_forecast", "commodities"],
            "commodities": {
                "provider": "investiny",
                "instruments": [{"name": "gas_ttf", "investing_id": 1178699}],
            },
        }
    )

    features = lear._build_lear_extra_covariate_features(
        config=config,
        start_day=pd.Timestamp("2025-01-01", tz="Europe/Berlin"),
        end_day=pd.Timestamp("2025-01-02", tz="Europe/Berlin"),
        daily_index=daily_index,
    )

    assert calls[0][0].covariates == ["commodities"]
    assert features.columns.tolist() == ["gas_ttf", "co2_eua"]
    assert features.loc[daily_index[0], "gas_ttf"] == 40.0


def test_lear_extra_covariates_use_vectors_for_intraday_sources(monkeypatch) -> None:
    calls = []
    daily_index = pd.date_range("2025-01-01T00:00:00+01:00", periods=2, freq="D")
    timestamp_index = pd.date_range("2025-01-01T00:00:00+01:00", periods=2 * 96, freq="15min")

    def fake_build_timestamp_covariates(covariate_config, **kwargs):
        calls.append((covariate_config, kwargs))
        if covariate_config.covariates == ["commodities"]:
            return pd.DataFrame({"gas_ttf": 40.0}, index=timestamp_index)
        return pd.DataFrame(
            {
                "NTC_Net_MW": range(len(timestamp_index)),
                "Unavail_Total_MW": range(1000, 1000 + len(timestamp_index)),
                "price_spread_CH_minus_exaa": range(2000, 2000 + len(timestamp_index)),
                "reserve_fcr_germany_demand_mw": 580.0,
            },
            index=timestamp_index,
        )

    monkeypatch.setattr(lear, "build_timestamp_covariates", fake_build_timestamp_covariates)
    config = _config(
        features={
            "covariates": [
                "exaa",
                "load_forecast",
                "ntc",
                "generation_unavailability",
                "foreign_day_ahead_price_spreads",
                "reserve_market",
                "commodities",
            ],
            "foreign_price_markets": ["CH"],
            "commodities": {
                "provider": "investiny",
                "instruments": [{"name": "gas_ttf", "investing_id": 1178699}],
            },
        }
    )

    features = lear._build_lear_extra_covariate_features(
        config=config,
        start_day=pd.Timestamp("2025-01-01", tz="Europe/Berlin"),
        end_day=pd.Timestamp("2025-01-02", tz="Europe/Berlin"),
        daily_index=daily_index,
    )

    assert [call[0].covariates for call in calls] == [
        ["ntc", "generation_unavailability", "foreign_day_ahead_price_spreads", "reserve_market"],
        ["commodities"],
    ]
    assert "NTC_Net_MW_mtu_00" in features.columns
    assert "NTC_Net_MW_mtu_95" in features.columns
    assert "Unavail_Total_MW_mtu_00" in features.columns
    assert "price_spread_CH_minus_exaa_mtu_00" in features.columns
    assert "reserve_fcr_germany_demand_mw_mtu_00" in features.columns
    assert "gas_ttf" in features.columns
    assert features.loc[daily_index[0], "NTC_Net_MW_mtu_95"] == 95.0
    assert features.loc[daily_index[0], "price_spread_CH_minus_exaa_mtu_95"] == 2095.0
    assert features.loc[daily_index[0], "gas_ttf"] == 40.0


def test_lear_scarcity_features_build_compact_summaries(monkeypatch) -> None:
    calls = []
    daily_index = pd.date_range("2025-01-01T00:00:00+01:00", periods=1, freq="D")
    timestamp_index = pd.date_range("2025-01-01T00:00:00+01:00", periods=96, freq="15min")

    def fake_build_timestamp_covariates(covariate_config, **kwargs):
        calls.append((covariate_config, kwargs))
        return pd.DataFrame(
            {
                "load_fc": range(96),
                "NTC_Net_MW": 100.0,
                "Unavail_Total_MW": 20.0,
            },
            index=timestamp_index,
        )

    monkeypatch.setattr(lear, "build_timestamp_covariates", fake_build_timestamp_covariates)
    config = _config(add_scarcity_features=True)

    features = lear._build_lear_scarcity_features(
        config=config,
        start_day=pd.Timestamp("2025-01-01", tz="Europe/Berlin"),
        end_day=pd.Timestamp("2025-01-01", tz="Europe/Berlin"),
        daily_index=daily_index,
    )

    assert calls[0][0].covariates == ["load_forecast", "ntc", "generation_unavailability"]
    assert "scarcity_load_fc_max" in features.columns
    assert "scarcity_NTC_Net_MW_min" in features.columns
    assert "scarcity_Unavail_Total_MW_max" in features.columns
    assert features.loc[daily_index[0], "scarcity_load_fc_max"] == 95.0
