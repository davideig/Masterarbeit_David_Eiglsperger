from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from ..paths import resolve_path
from .base import RepoConfigModel, _default_datetime, _parse_timestamp


class EvaluationForecastConfig(BaseModel):
    name: str
    path: Path
    kind: Literal["point", "quantile"]
    value_column: str = "y_pred"
    y_true_column: str = "y_true"
    quantile_columns: list[str] | None = None


class EvaluationConfig(RepoConfigModel):
    target_tz: str = "Europe/Berlin"
    test_start: datetime = Field(
        default_factory=lambda: _default_datetime("2025-12-01T00:00:00+01:00")
    )
    test_end: datetime = Field(
        default_factory=lambda: _default_datetime("2026-02-28T23:45:00+01:00")
    )
    skip_dates: list[date] = Field(default_factory=list)
    frequency: str = "15min"
    n_periods: int = 96
    y_true_column: str = "y_true"
    quantiles: list[float] = Field(default_factory=lambda: [0.10, 0.25, 0.50, 0.75, 0.90])
    prediction_intervals: list[tuple[float, float]] = Field(
        default_factory=lambda: [(0.10, 0.90), (0.25, 0.75)]
    )
    significance_level: float = 0.05
    point_forecasts: list[EvaluationForecastConfig] = Field(default_factory=list)
    quantile_forecasts: list[EvaluationForecastConfig] = Field(default_factory=list)
    output_dir: Path = Path("results/evaluation")
    validate_inputs: bool = True
    run_gw_tests: bool = True

    @field_validator("test_start", "test_end", mode="before")
    @classmethod
    def _coerce_datetimes(cls, value: Any, info) -> datetime:
        target_tz = info.data.get("target_tz", "Europe/Berlin")
        return _parse_timestamp(value, target_tz)

    @model_validator(mode="after")
    def _resolve_paths(self) -> "EvaluationConfig":
        for forecast in [*self.point_forecasts, *self.quantile_forecasts]:
            forecast.path = resolve_path(forecast.path, self.repo_root)
        self.output_dir = resolve_path(self.output_dir, self.repo_root)
        return self
