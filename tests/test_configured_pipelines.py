from __future__ import annotations

from pathlib import Path

import pandas as pd

from da_price_forecasting.config import RunConfig, RunKind
from da_price_forecasting.scripts.run import main, run_from_config


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _write_synthetic_forecast(path: Path) -> None:
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
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path)


def test_evaluation_runs_through_dispatcher(tmp_path: Path) -> None:
    repo_root = _repo_root()
    forecast_path = tmp_path / "forecast.csv"
    evaluation_dir = tmp_path / "evaluation"
    _write_synthetic_forecast(forecast_path)

    run_from_config(
        RunConfig(
            repo_root=repo_root,
            kind=RunKind.EVALUATION,
            config={
                "target_tz": "Europe/Berlin",
                "test_start": "2026-03-01T00:00:00+01:00",
                "test_end": "2026-03-03T12:00:00+01:00",
                "frequency": "12h",
                "n_periods": 2,
                "output_dir": str(evaluation_dir),
                "run_gw_tests": False,
                "point_forecasts": [
                    {"name": "toy_model", "kind": "point", "path": str(forecast_path)},
                ],
                "quantile_forecasts": [
                    {"name": "toy_model", "kind": "quantile", "path": str(forecast_path)},
                ],
            },
        )
    )

    assert (evaluation_dir / "point_metrics.csv").exists()
    assert (evaluation_dir / "quantile_coverage.csv").exists()


def test_cli_main_runs_evaluation_config_file(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    forecast_path = tmp_path / "forecast.csv"
    evaluation_dir = tmp_path / "evaluation"
    config_path = tmp_path / "run_evaluation.yaml"
    _write_synthetic_forecast(forecast_path)

    config_path.write_text(
        f"""
kind: evaluation
config:
  target_tz: Europe/Berlin
  test_start: "2026-03-01T00:00:00+01:00"
  test_end: "2026-03-03T12:00:00+01:00"
  frequency: 12h
  n_periods: 2
  output_dir: "{evaluation_dir}"
  run_gw_tests: false
  point_forecasts:
    - name: toy_model
      kind: point
      path: "{forecast_path}"
  quantile_forecasts:
    - name: toy_model
      kind: quantile
      path: "{forecast_path}"
""".lstrip(),
        encoding="utf-8",
    )

    main(["--config", str(config_path)])

    assert (evaluation_dir / "point_metrics.csv").exists()
    assert (evaluation_dir / "quantile_metrics.csv").exists()
