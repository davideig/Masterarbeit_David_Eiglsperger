from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import Field, model_validator

from ..paths import resolve_path
from .base import RepoConfigModel


class RunKind(str, Enum):
    ENTSOE_LOAD_FORECAST_BENCHMARK = "entsoe_load_forecast_benchmark"
    ENTSOE_RENEWABLE_FORECAST_BENCHMARK = "entsoe_renewable_forecast_benchmark"
    LEAR_OPERATIONAL = "lear_operational"
    LEAR_ANC = "lear_anc"
    LOAD_FORECAST_MODEL = "load_forecast_model"
    LOAD_FORECAST_ENSEMBLE = "load_forecast_ensemble"
    REGIONAL_LOAD_FORECAST = "regional_load_forecast"
    ICON_AGGREGATE = "icon_aggregate"
    MASTR_CAPACITY = "mastr_capacity"
    POPULATION_CLUSTER_WEIGHTS = "population_cluster_weights"
    REGIONAL_RENEWABLE_FEATURES = "regional_renewable_features"
    RENEWABLE_GENERATION_MODEL = "renewable_generation_model"
    RENEWABLE_GENERATION_POSTPROCESS = "renewable_generation_postprocess"
    RENEWABLE_PROXY = "renewable_proxy"
    RESERVE_MARKET = "reserve_market"
    EVALUATION = "evaluation"


class RunConfig(RepoConfigModel):
    """Top-level runner config used by the single repo entry point."""

    kind: RunKind
    config_path: Path | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    plot: bool = False
    submit: bool = False
    save_outputs: bool = True

    @model_validator(mode="after")
    def _resolve_paths(self) -> "RunConfig":
        if self.config_path is None and not self.config:
            raise ValueError("Run config requires either config_path or an embedded config mapping.")
        if self.config_path is not None:
            self.config_path = resolve_path(self.config_path, self.repo_root)
        return self
