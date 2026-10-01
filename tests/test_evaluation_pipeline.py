from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from da_price_forecasting.config import EvaluationConfig, EvaluationForecastConfig
from da_price_forecasting.evaluation.pipeline import run_evaluation


def _write_forecast_csv(path: Path, *, non_monotone_quantiles: bool = False) -> pd.DataFrame:
    index = pd.date_range("2026-03-01T00:00:00+01:00", periods=6, freq="12h")
    y_true = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0, 60.0], index=index)
    frame = pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred": y_true + pd.Series([1.0, -2.0, 1.0, -2.0, 1.0, -2.0], index=index),
            "q0.100": y_true - 5.0,
            "q0.250": y_true - 2.0,
            "q0.500": y_true,
            "q0.750": y_true + 2.0,
            "q0.900": y_true + 5.0,
        },
        index=index,
    )
    if non_monotone_quantiles:
        frame["q0.500"] = frame["q0.250"] - 1.0
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path)
    return frame


def _evaluation_config(tmp_path: Path, forecast_path: Path) -> EvaluationConfig:
    return EvaluationConfig(
        repo_root=Path.cwd(),
        target_tz="Europe/Berlin",
        test_start="2026-03-01T00:00:00+01:00",
        test_end="2026-03-03T12:00:00+01:00",
        frequency="12h",
        n_periods=2,
        output_dir=tmp_path / "evaluation",
        run_gw_tests=False,
        point_forecasts=[
            EvaluationForecastConfig(name="toy_model", kind="point", path=forecast_path),
        ],
        quantile_forecasts=[
            EvaluationForecastConfig(name="toy_model", kind="quantile", path=forecast_path),
        ],
    )


def test_run_evaluation_writes_point_and_quantile_outputs(tmp_path: Path) -> None:
    forecast_path = tmp_path / "forecast.csv"
    _write_forecast_csv(forecast_path)

    output = run_evaluation(_evaluation_config(tmp_path, forecast_path))

    output_dir = tmp_path / "evaluation"
    assert sorted(output) == ["point_metrics", "quantile_coverage", "quantile_metrics"]
    assert (output_dir / "point_panel.csv").exists()
    assert (output_dir / "point_metrics.csv").exists()
    assert (output_dir / "quantile_panel.csv").exists()
    assert (output_dir / "quantile_metrics.csv").exists()
    assert (output_dir / "quantile_coverage.csv").exists()
    assert (output_dir / "config.json").exists()

    point_metrics = pd.read_csv(output_dir / "point_metrics.csv")
    assert point_metrics.loc[0, "model"] == "toy_model"
    assert point_metrics.loc[0, "mae"] == pytest.approx(1.5)
    assert point_metrics.loc[0, "n_obs"] == 6

    quantile_panel = pd.read_csv(output_dir / "quantile_panel.csv")
    assert {"timestamp", "y_true", "toy_model_q0.100", "toy_model_q0.900"}.issubset(
        quantile_panel.columns
    )


def test_run_evaluation_rejects_non_monotone_quantiles(tmp_path: Path) -> None:
    forecast_path = tmp_path / "forecast.csv"
    _write_forecast_csv(forecast_path, non_monotone_quantiles=True)

    with pytest.raises(ValueError, match="not monotone"):
        run_evaluation(_evaluation_config(tmp_path, forecast_path))
