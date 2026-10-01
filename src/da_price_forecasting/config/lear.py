from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from ..paths import resolve_path
from .base import ForecastVariant, RepoConfigModel, WeatherSource, _default_datetime, _parse_timestamp
from .features import CovariateConfig


class LearOperationalConfig(RepoConfigModel):
    target_tz: str = "Europe/Berlin"
    post_regime_start: datetime = Field(
        default_factory=lambda: _default_datetime("2025-10-01T00:00:00+02:00")
    )
    country_code_entsoe: str = "DE_LU"
    entsoe_api_key_env: str = "ENTSOE_API_KEY"
    entsoe_start_date: datetime = Field(
        default_factory=lambda: _default_datetime("2024-01-01T00:00:00+01:00")
    )
    entsoe_end_date: datetime = Field(
        default_factory=lambda: _default_datetime("2026-04-23T00:00:00+02:00")
    )
    entsoe_price_file: Path = Path("data/processed/lear/entsoe_day_ahead_prices.csv")
    entsoe_exaa_price_file: Path = Path("data/processed/lear/entsoe_exaa_prices.csv")
    era5_dirs: list[Path] = Field(
        default_factory=lambda: [
            Path("data/processed/era5_aggregated"),
        ]
    )
    icon_dir: Path = Path("data/processed/icon_aggregated_c5")
    dwd_folder_offset_date: date = date(2025, 10, 26)
    start_folder_date: date = date(2025, 8, 1)
    required_run: str = "09"
    skip_dates: list[date] = Field(
        default_factory=lambda: [
            date(2025, 10, 26),
            date(2025, 10, 27),
            date(2025, 10, 28),
        ]
    )
    use_vst: bool = True
    weather_source: WeatherSource = WeatherSource.DWD
    variant: ForecastVariant = ForecastVariant.EXAA_ONLY
    lars_start_date: datetime = Field(
        default_factory=lambda: _default_datetime("2025-12-01T00:00:00+01:00")
    )
    test_start: datetime = Field(
        default_factory=lambda: _default_datetime("2026-04-16T00:00:00+02:00")
    )
    test_end: datetime = Field(
        default_factory=lambda: _default_datetime("2026-04-23T23:45:00+02:00")
    )
    train_days_rolling: int = 56
    n_clusters: int = 5
    add_calendar_features: bool = False
    add_scarcity_features: bool = False
    include_raw_weather_features: bool = True
    include_load_forecast_features: bool = True
    price_model_type: Literal["lear", "hist_gradient_boosting", "lightgbm"] = "lear"
    hgb_max_iter: int = 600
    hgb_learning_rate: float = 0.025
    hgb_max_leaf_nodes: int = 15
    hgb_min_samples_leaf: int = 20
    hgb_l2_regularization: float = 0.0
    lgbm_n_estimators: int = 900
    lgbm_learning_rate: float = 0.015
    lgbm_num_leaves: int = 15
    lgbm_min_child_samples: int = 40
    lgbm_subsample: float = 0.9
    lgbm_colsample_bytree: float = 0.8
    lgbm_reg_lambda: float = 1.0
    random_state: int = 42
    lasso_cv_eps: float = 1e-3
    lasso_cv_alphas: int | list[float] = 100
    lasso_cv_tol: float = 1e-3
    lasso_cv_max_iter: int = 10_000
    lars_max_iter: int = 1000
    lars_max_n_alphas: int = 1000
    forecast_bias_correction: Literal["none", "rolling_mean_error", "rolling_hour_mean_error"] = "none"
    forecast_bias_train_days: int = 28
    forecast_bias_min_train_days: int = 7
    features: CovariateConfig = Field(default_factory=CovariateConfig)
    export_dir: Path | None = None

    @field_validator(
        "post_regime_start",
        "entsoe_start_date",
        "entsoe_end_date",
        "lars_start_date",
        "test_start",
        "test_end",
        mode="before",
    )
    @classmethod
    def _coerce_datetimes(cls, value: Any, info) -> datetime:
        target_tz = info.data.get("target_tz", "Europe/Berlin")
        return _parse_timestamp(value, target_tz)

    @model_validator(mode="after")
    def _resolve_paths(self) -> "LearOperationalConfig":
        self.era5_dirs = [resolve_path(path, self.repo_root) for path in self.era5_dirs]
        self.icon_dir = resolve_path(self.icon_dir, self.repo_root)
        self.entsoe_price_file = resolve_path(self.entsoe_price_file, self.repo_root)
        self.entsoe_exaa_price_file = resolve_path(self.entsoe_exaa_price_file, self.repo_root)
        if self.export_dir is not None:
            self.export_dir = resolve_path(self.export_dir, self.repo_root)
        return self

    @property
    def use_exaa(self) -> bool:
        return self.variant == ForecastVariant.EXAA

    @property
    def use_exaa_only(self) -> bool:
        return self.variant == ForecastVariant.EXAA_ONLY

    @property
    def experiment_name(self) -> str:
        weather_tag = self.weather_source.value.lower() if not self.use_exaa_only else "exaa_only"
        variant_tag = ""
        if self.variant == ForecastVariant.EXAA and not self.use_exaa_only:
            variant_tag = "_exaa"
        elif self.variant == ForecastVariant.FUNDAMENTAL and not self.use_exaa_only:
            variant_tag = "_fundamental"

        cluster_tag = f"_c{self.n_clusters}" if not self.use_exaa_only else ""
        covariate_tag = _covariate_tag(self.features.covariates)
        return f"lear_{weather_tag}{variant_tag}{cluster_tag}{covariate_tag}_d{self.train_days_rolling}"

    @property
    def resolved_export_dir(self) -> Path:
        if self.export_dir is not None:
            return self.export_dir

        base = self.repo_root / "results" / "lear_op_results"
        if self.use_exaa_only:
            return base / "exaa_only" / f"d{self.train_days_rolling}"
        if self.use_exaa:
            return base / self.weather_source.value.lower() / f"d{self.train_days_rolling}" / f"c{self.n_clusters}" / "exaa"
        return base / self.weather_source.value.lower() / f"d{self.train_days_rolling}" / f"c{self.n_clusters}" / "fundamental"


class LearAncConfig(RepoConfigModel):
    target_tz: str = "Europe/Berlin"
    country_code_entsoe: str = "DE_LU"
    entsoe_api_key_env: str = "ENTSOE_API_KEY"
    entsoe_start_date: datetime = Field(
        default_factory=lambda: _default_datetime("2024-01-01T00:00:00+01:00")
    )
    entsoe_end_date: datetime = Field(
        default_factory=lambda: _default_datetime("2026-04-23T00:00:00+02:00")
    )
    era5_dirs: list[Path] = Field(
        default_factory=lambda: [
            Path("data/processed/era5_aggregated"),
        ]
    )
    icon_dir: Path = Path("data/processed/icon_aggregated_c5")
    dwd_folder_offset_date: date = date(2025, 10, 26)
    start_folder_date: date = date(2025, 8, 1)
    required_run: str = "09"
    skip_dates: list[date] = Field(
        default_factory=lambda: [
            date(2025, 10, 26),
            date(2025, 10, 27),
            date(2025, 10, 28),
        ]
    )
    weather_source: WeatherSource = WeatherSource.ERA5
    variant: ForecastVariant = ForecastVariant.EXAA
    lars_start_date: datetime = Field(
        default_factory=lambda: _default_datetime("2025-12-01T00:00:00+01:00")
    )
    test_start: datetime = Field(
        default_factory=lambda: _default_datetime("2025-12-01T00:00:00+01:00")
    )
    test_end: datetime = Field(
        default_factory=lambda: _default_datetime("2026-02-28T23:45:00+01:00")
    )
    train_days_rolling: int = 112
    n_clusters: int = 5
    include_raw_weather_features: bool = True
    lasso_cv_eps: float = 1e-3
    lasso_cv_alphas: int | list[float] = 100
    lasso_cv_tol: float = 1e-3
    lasso_cv_max_iter: int = 10_000
    lars_max_iter: int = 1000
    lars_max_n_alphas: int = 1000
    mtu_window_wind: list[int] = Field(default_factory=lambda: list(range(96)))
    mtu_window_solar: list[int] = Field(default_factory=lambda: list(range(96)))
    features: CovariateConfig = Field(default_factory=CovariateConfig)
    export_dir: Path | None = None

    @field_validator("entsoe_start_date", "entsoe_end_date", "lars_start_date", "test_start", "test_end", mode="before")
    @classmethod
    def _coerce_datetimes(cls, value: Any, info) -> datetime:
        target_tz = info.data.get("target_tz", "Europe/Berlin")
        return _parse_timestamp(value, target_tz)

    @model_validator(mode="after")
    def _resolve_paths(self) -> "LearAncConfig":
        self.era5_dirs = [resolve_path(path, self.repo_root) for path in self.era5_dirs]
        self.icon_dir = resolve_path(self.icon_dir, self.repo_root)
        if self.export_dir is not None:
            self.export_dir = resolve_path(self.export_dir, self.repo_root)
        return self

    @property
    def use_exaa(self) -> bool:
        return self.variant == ForecastVariant.EXAA

    @property
    def use_exaa_only(self) -> bool:
        return self.variant == ForecastVariant.EXAA_ONLY

    @property
    def experiment_name(self) -> str:
        weather_tag = self.weather_source.value.lower() if not self.use_exaa_only else "exaa_only"
        variant_tag = ""
        if self.variant == ForecastVariant.EXAA and not self.use_exaa_only:
            variant_tag = "_exaa"
        elif self.variant == ForecastVariant.FUNDAMENTAL and not self.use_exaa_only:
            variant_tag = "_fundamental"

        cluster_tag = f"_c{self.n_clusters}" if not self.use_exaa_only else ""
        covariate_tag = _covariate_tag(self.features.covariates)
        return f"anc_{weather_tag}{variant_tag}{cluster_tag}{covariate_tag}_d{self.train_days_rolling}"

    @property
    def resolved_export_dir(self) -> Path:
        if self.export_dir is not None:
            return self.export_dir

        base = self.repo_root / "results" / "lear_anc_results"
        if self.use_exaa_only:
            return base / "exaa_only" / f"d{self.train_days_rolling}"
        if self.use_exaa:
            return base / self.weather_source.value.lower() / f"c{self.n_clusters}" / f"d{self.train_days_rolling}" / "exaa"
        return base / self.weather_source.value.lower() / f"c{self.n_clusters}" / f"d{self.train_days_rolling}" / "fundamental"


def _covariate_tag(covariates: list[str]) -> str:
    if not covariates:
        return ""
    tags = {
        "exaa": "exaa",
        "load_forecast": "load",
        "ntc": "ntc",
        "generation_unavailability": "unavail",
        "renewable_generation_proxy": "renewproxy",
        "commodities": "commodities",
    }
    return "_" + "_".join(tags.get(covariate, covariate) for covariate in covariates)
