from __future__ import annotations

import pandas as pd

from da_price_forecasting.config import EntsoeRenewableForecastBenchmarkConfig, RenewableGenerationModelConfig
from da_price_forecasting.pipelines import renewable_generation as rg


def test_entsoe_renewable_benchmark_frame_builds_model_columns(monkeypatch, tmp_path):
    index = pd.date_range("2026-01-01 00:00", periods=2, freq="15min", tz="Europe/Berlin")
    actual = pd.DataFrame(
        {
            "Solar_Actual_MW": [1.0, 2.0],
            "Wind_Onshore_Actual_MW": [3.0, 4.0],
            "Wind_Offshore_Actual_MW": [5.0, 6.0],
            "Wind_Total_Actual_MW": [8.0, 10.0],
            "Renewable_Total_Actual_MW": [9.0, 12.0],
        },
        index=index,
    )
    forecast = pd.DataFrame(
        {
            "Solar_Forecast_MW": [1.5, 2.5],
            "Wind_Onshore_Forecast_MW": [3.5, 4.5],
            "Wind_Offshore_Forecast_MW": [5.5, 6.5],
            "Wind_Total_Forecast_MW": [9.0, 11.0],
            "Renewable_Total_Forecast_MW": [10.5, 13.5],
        },
        index=index,
    )

    monkeypatch.setattr(rg, "_load_or_fetch_benchmark_actual_generation", lambda config: actual)
    monkeypatch.setattr(rg, "_load_or_fetch_entsoe_renewable_forecast", lambda config: forecast)

    config = EntsoeRenewableForecastBenchmarkConfig(
        repo_root=tmp_path,
        actual_generation_file=tmp_path / "actual.csv",
        forecast_file=tmp_path / "forecast.csv",
        export_dir=tmp_path / "out",
    )

    result = rg.build_entsoe_renewable_forecast_benchmark(config)

    assert result.loc[index[0], "Solar_Model_MW"] == 1.5
    assert result.loc[index[0], "Wind_Total_Model_MW"] == 9.0
    assert result.loc[index[0], "Renewable_Total_Model_MW"] == 10.5
    assert result.loc[index[0], "Solar_Actual_MW"] == 1.0
    assert result.loc[index[0], "Renewable_Total_Actual_MW"] == 9.0


def test_renewable_training_cutoff_defaults_to_previous_complete_day(tmp_path):
    config = RenewableGenerationModelConfig(
        repo_root=tmp_path,
        actual_generation_file=tmp_path / "actual.csv",
        renewable_proxy_file=tmp_path / "proxy.csv",
        unavailability_file=tmp_path / "unavailability.csv",
        export_dir=tmp_path / "out",
    )
    day = pd.Timestamp("2026-01-10", tz="Europe/Berlin")

    assert rg._training_target_cutoff(day, config) == pd.Timestamp("2026-01-09 23:45", tz="Europe/Berlin")


def test_renewable_training_cutoff_can_use_issue_time_morning_data(tmp_path):
    config = RenewableGenerationModelConfig(
        repo_root=tmp_path,
        actual_generation_file=tmp_path / "actual.csv",
        renewable_proxy_file=tmp_path / "proxy.csv",
        unavailability_file=tmp_path / "unavailability.csv",
        export_dir=tmp_path / "out",
        target_availability_lag_days=1,
        target_availability_cutoff_hour=10,
        target_availability_cutoff_minute=0,
    )
    day = pd.Timestamp("2026-01-10", tz="Europe/Berlin")

    assert rg._training_target_cutoff(day, config) == pd.Timestamp("2026-01-09 10:00", tz="Europe/Berlin")
