from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from ..paths import resolve_path
from .base import RepoConfigModel


class LoadForecastModelConfig(RepoConfigModel):
    """Train/evaluate a direct actual-load forecasting model."""

    target_tz: str = "Europe/Berlin"
    country_code_entsoe: str = "DE_LU"
    entsoe_api_key_env: str = "ENTSOE_API_KEY"
    entsoe_start_date: date = date(2025, 8, 1)
    entsoe_end_date: date = date(2026, 4, 23)
    chunk_days: int = 90

    actual_load_file: Path = Path("data/processed/load_forecast/actual_load.csv")
    icon_dir: Path = Path("data/processed/icon_aggregated_c5")
    weather_source: Literal["dwd_icon", "open_meteo"] = "dwd_icon"
    open_meteo_weather_file: Path = Path("data/processed/open_meteo/open_meteo_icon_d2_c25_load_weather.csv")
    open_meteo_cluster_file: Path = Path("data/clustering/icon_d2_clustering_c25.parquet")
    open_meteo_base_url: str = "https://customer-historical-forecast-api.open-meteo.com/v1/forecast"
    open_meteo_api_key_env: str | None = "OPEN_METEO_API_KEY"
    open_meteo_model: str | None = "icon_d2"
    open_meteo_start_date: date = date(2025, 8, 1)
    open_meteo_end_date: date = date(2026, 4, 23)
    open_meteo_hourly_variables: list[str] = Field(
        default_factory=lambda: [
            "temperature_2m",
            "dew_point_2m",
            "surface_pressure",
            "wind_speed_10m",
            "wind_direction_10m",
            "shortwave_radiation",
            "direct_radiation",
            "diffuse_radiation",
            "cloud_cover",
            "precipitation",
            "snow_depth",
        ]
    )
    open_meteo_batch_size: int = 10
    open_meteo_cell_selection: str = "nearest"
    open_meteo_timeout_seconds: int = 60
    open_meteo_force_download: bool = False
    open_meteo_api_mode: Literal["historical_forecast", "single_run"] = "historical_forecast"
    open_meteo_single_run_hour_utc: str = "06:00"
    open_meteo_single_run_forecast_days: int = 2
    open_meteo_request_pause_seconds: float = 0.0
    open_meteo_retry_attempts: int = 5
    open_meteo_retry_backoff_seconds: float = 30.0
    open_meteo_skip_unavailable_runs: bool = False
    open_meteo_fallback_previous_runs: bool = False
    open_meteo_fallback_step_hours: int = Field(default=2, gt=0)
    open_meteo_fallback_max_lookback_hours: int = Field(default=24, ge=0)
    open_meteo_point_selection: Literal["centroid", "grid_mean"] = "centroid"
    open_meteo_max_points_per_cluster: int | None = None
    extra_open_meteo_weather_files: list[Path] = Field(default_factory=list)
    include_open_meteo_weather_ensemble_features: bool = False
    open_meteo_weather_ensemble_stats: list[str] = Field(default_factory=lambda: ["mean", "std"])
    export_dir: Path = Path("results/load_forecast_results/direct_dwd_icon_c5_hgb")

    include_calendar_features: bool = True
    include_holiday_features: bool = True
    include_bridge_day_features: bool = True
    include_actual_load_lag_features: bool = True
    include_weather_features: bool = True
    include_weighted_weather_features: bool = False
    include_weighted_weather_daily_features: bool = False
    include_weighted_weather_inertia_features: bool = False
    include_weather_cluster_spread_features: bool = False
    include_weighted_weather_quantile_features: bool = False
    include_weather_time_interactions: bool = False
    include_regional_holiday_features: bool = False
    include_partial_load_features: bool = False
    include_partial_load_shape_features: bool = False
    include_entsoe_forecast_features: bool = False
    include_entsoe_error_lag_features: bool = False
    include_entsoe_error_rolling_features: bool = False
    include_regional_load_features: bool = False
    include_rich_temperature_features: bool = False
    fetch_weather_only: bool = False
    calendar_harmonics: int = 1

    entsoe_load_forecast_file: Path = Path("data/processed/load_forecast/entsoe_load_forecast.csv")
    regional_load_cache_dir: Path = Path("data/processed/load_forecast/regional_features")
    regional_load_components: dict[str, str] = Field(default_factory=dict)
    include_regional_load_forecast_features: bool = True
    include_regional_actual_load_lag_features: bool = True
    include_regional_partial_load_features: bool = True
    include_regional_load_summary_features: bool = True
    weather_cluster_weight_file: Path | None = None
    weather_cluster_id_column: str = "cluster_id"
    weather_cluster_weight_column: str = "weight"
    weather_weighted_feature_prefix: str = "weather_weighted"
    weather_weighted_feature_bases: list[str] = Field(default_factory=list)
    weighted_weather_daily_feature_bases: list[str] = Field(default_factory=list)
    weighted_weather_daily_stats: list[str] = Field(default_factory=lambda: ["mean", "min", "max"])
    weighted_weather_daily_lag_days: list[int] = Field(default_factory=lambda: [1, 7])
    weighted_weather_inertia_feature_bases: list[str] = Field(default_factory=list)
    weighted_weather_inertia_windows_hours: list[int] = Field(default_factory=lambda: [24, 48, 72])
    weighted_weather_inertia_stats: list[str] = Field(default_factory=lambda: ["mean", "delta_mean"])
    weather_spread_feature_bases: list[str] = Field(default_factory=list)
    weather_spread_stats: list[str] = Field(default_factory=lambda: ["min", "max", "range", "std"])
    weighted_weather_quantile_feature_bases: list[str] = Field(default_factory=list)
    weighted_weather_quantiles: list[float] = Field(default_factory=lambda: [0.1, 0.25, 0.5, 0.75, 0.9])
    weather_hdd_thresholds: list[float] = Field(default_factory=lambda: [18.0])
    weather_cdd_thresholds: list[float] = Field(default_factory=lambda: [22.0])
    keep_weather_cluster_features: bool = True

    regional_holiday_weight_file: Path | None = None
    regional_holiday_region_column: str = "region"
    regional_holiday_weight_column: str = "weight"
    regional_holiday_feature_prefix: str = "regional_holiday"

    partial_load_reference_day: int = 1
    partial_load_comparison_lag_days: int = 7
    partial_load_morning_end_hour: int = 11
    partial_load_morning_end_minute: int = 45
    partial_load_point_times: list[str] = Field(default_factory=list)

    actual_load_lag_days: list[int] = Field(default_factory=lambda: [2, 3, 7, 14])
    entsoe_error_lag_days: list[int] = Field(default_factory=lambda: [2, 3, 7, 14, 21])
    entsoe_error_rolling_windows_days: list[int] = Field(default_factory=lambda: [7, 28])
    entsoe_error_rolling_groups: list[Literal["global", "hour", "mtu"]] = Field(default_factory=lambda: ["global", "mtu"])
    entsoe_error_rolling_min_observations: int = 24
    target_availability_lag_days: int = 1
    target_availability_cutoff_hour: int | None = None
    require_weather_for_training: bool = False
    weather_presence_column: str = "weather_weighted_t2m_C"
    target_availability_cutoff_minute: int = 0

    start_folder_date: date = date(2025, 8, 1)
    required_run: str = "09"
    dwd_folder_offset_date: date = date(2025, 10, 26)
    skip_dates: list[date] = Field(
        default_factory=lambda: [
            date(2025, 10, 26),
            date(2025, 10, 27),
            date(2025, 10, 28),
        ]
    )
    dwd_icon_auto_update: bool = False
    dwd_icon_raw_dir: Path = Path("data/raw/dwd_icon_daily")
    dwd_icon_base_url: str = "https://opendata.dwd.de/weather/nwp/icon-d2/grib/"
    dwd_icon_download_variables: list[str] = Field(
        default_factory=lambda: [
            "t_2m",
            "td_2m",
            "p",
            "u_10m",
            "v_10m",
            "vmax_10m",
            "aswdir_s",
            "aswdifd_s",
            "tot_prec",
            "h_snow",
            "snow_gsp",
        ]
    )
    dwd_icon_download_timeout_seconds: int = 60
    dwd_icon_request_pause_seconds: float = 0.0
    dwd_icon_force_update: bool = False
    dwd_icon_catch_up_missing_days: bool = True
    dwd_icon_aggregation_shapefile_path: Path = Path("data/shapefile/ne_10m_admin_0_countries.shp")
    dwd_icon_aggregation_n_clusters: int | None = None
    dwd_icon_aggregation_buffer_km: int = 50
    dwd_icon_aggregation_cluster_source: Literal["grid", "mastr_solar", "mastr_solar_tso", "mastr_wind"] = "grid"
    dwd_icon_aggregation_cluster_output_file: Path | None = None
    dwd_icon_aggregation_capacity_file: Path = Path("data/raw/renewable_capacity/installed_capacity.csv")
    dwd_icon_aggregation_capacity_weighted: bool = True

    model_type: Literal["hist_gradient_boosting", "lightgbm", "ridge", "tabpfn"] = "hist_gradient_boosting"
    model_granularity: Literal["global", "hour_block"] = "global"
    hour_block_boundaries: list[int] = Field(default_factory=lambda: [0, 6, 11, 16, 21, 24])
    train_days_rolling: int = 112
    min_train_days: int = 14
    test_start: date = date(2025, 12, 1)
    test_end: date = date(2026, 2, 28)

    hgb_max_iter: int = 300
    hgb_learning_rate: float = 0.04
    hgb_max_leaf_nodes: int = 31
    hgb_l2_regularization: float = 0.1
    lgbm_n_estimators: int = 600
    lgbm_learning_rate: float = 0.02
    lgbm_num_leaves: int = 31
    lgbm_min_child_samples: int = 40
    lgbm_subsample: float = 1.0
    lgbm_colsample_bytree: float = 1.0
    lgbm_reg_lambda: float = 0.3
    ridge_alpha: float = 100.0
    tabpfn_max_train_rows: int | None = 1000
    tabpfn_context_selection: Literal["tail", "recent_weekday_even"] = "recent_weekday_even"
    tabpfn_context_recent_fraction: float = 0.50
    tabpfn_context_same_weekday_fraction: float = 0.30
    tabpfn_n_estimators: int = 4
    tabpfn_device: str = "cpu"
    tabpfn_fit_mode: Literal["low_memory", "fit_preprocessors", "fit_with_cache", "batched"] = "low_memory"
    tabpfn_n_preprocessing_jobs: int = 1
    tabpfn_inference_precision: str = "auto"
    tabpfn_ignore_pretraining_limits: bool = False
    tabpfn_output_type: Literal["mean", "median", "mode"] = "mean"
    tabpfn_predict_batch_size: int | None = 16
    tabpfn_target_transform: Literal["none", "asinh_robust"] = "none"
    random_state: int = 42
    max_features: int | None = None
    load_target_mode: Literal["actual_load", "entsoe_residual"] = "actual_load"
    clip_predictions_to_training_target_range: bool = False
    prediction_upper_quantile: float = Field(default=1.0, ge=0.0, le=1.0)
    include_rolling_residual_quantiles: bool = False
    residual_quantiles: list[float] = Field(default_factory=lambda: [0.025, 0.25, 0.50, 0.75, 0.975])
    residual_quantile_window_days: int = 28
    residual_quantile_min_observations: int = 20
    residual_quantile_group: Literal["global", "hour", "mtu"] = "mtu"
    residual_quantile_spread_scale: float = Field(default=1.0, gt=0.0)
    quantile_evaluation_start: date | None = None
    quantile_evaluation_end: date | None = None

    @field_validator("actual_load_lag_days", mode="before")
    @classmethod
    def _coerce_lag_days(cls, value: Any) -> list[int]:
        if value is None:
            return []
        if isinstance(value, int):
            return [value]
        return [int(item) for item in value]

    @field_validator("entsoe_error_lag_days", mode="before")
    @classmethod
    def _coerce_entsoe_error_lag_days(cls, value: Any) -> list[int]:
        if value is None:
            return []
        if isinstance(value, int):
            return [value]
        return [int(item) for item in value]

    @field_validator("entsoe_error_rolling_windows_days", mode="before")
    @classmethod
    def _coerce_entsoe_error_rolling_windows_days(cls, value: Any) -> list[int]:
        if value is None:
            return []
        if isinstance(value, int):
            return [value]
        return [int(item) for item in value]

    @field_validator("entsoe_error_rolling_groups", mode="before")
    @classmethod
    def _coerce_entsoe_error_rolling_groups(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return [str(item) for item in value]

    @field_validator("weather_weighted_feature_bases", mode="before")
    @classmethod
    def _coerce_weighted_feature_bases(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return [str(item) for item in value]

    @field_validator("weighted_weather_daily_feature_bases", mode="before")
    @classmethod
    def _coerce_weighted_daily_feature_bases(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return [str(item) for item in value]

    @field_validator("weighted_weather_inertia_feature_bases", mode="before")
    @classmethod
    def _coerce_weighted_inertia_feature_bases(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return [str(item) for item in value]

    @field_validator("weather_spread_feature_bases", mode="before")
    @classmethod
    def _coerce_weather_spread_feature_bases(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return [str(item) for item in value]

    @field_validator("weighted_weather_quantile_feature_bases", mode="before")
    @classmethod
    def _coerce_weighted_weather_quantile_feature_bases(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return [str(item) for item in value]

    @field_validator("weighted_weather_daily_stats", mode="before")
    @classmethod
    def _coerce_weighted_daily_stats(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return [str(item) for item in value]

    @field_validator("weighted_weather_inertia_stats", "weather_spread_stats", mode="before")
    @classmethod
    def _coerce_weather_stats(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return [str(item) for item in value]

    @field_validator("weighted_weather_daily_lag_days", mode="before")
    @classmethod
    def _coerce_weighted_daily_lag_days(cls, value: Any) -> list[int]:
        if value is None:
            return []
        if isinstance(value, int):
            return [value]
        return [int(item) for item in value]

    @field_validator("weighted_weather_inertia_windows_hours", mode="before")
    @classmethod
    def _coerce_weighted_inertia_windows_hours(cls, value: Any) -> list[int]:
        if value is None:
            return []
        if isinstance(value, int):
            return [value]
        return [int(item) for item in value]

    @field_validator("partial_load_point_times", mode="before")
    @classmethod
    def _coerce_partial_load_point_times(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return [str(item) for item in value]

    @field_validator("weather_hdd_thresholds", "weather_cdd_thresholds", mode="before")
    @classmethod
    def _coerce_weather_thresholds(cls, value: Any) -> list[float]:
        if value is None:
            return []
        if isinstance(value, (int, float)):
            return [float(value)]
        return [float(item) for item in value]

    @field_validator("hour_block_boundaries", mode="before")
    @classmethod
    def _coerce_hour_block_boundaries(cls, value: Any) -> list[int]:
        if value is None:
            return []
        return [int(item) for item in value]

    @field_validator("dwd_icon_download_variables", mode="before")
    @classmethod
    def _coerce_dwd_icon_download_variables(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return [str(item) for item in value]

    @field_validator("residual_quantiles", mode="before")
    @classmethod
    def _coerce_residual_quantiles(cls, value: Any) -> list[float]:
        if value is None:
            return []
        if isinstance(value, (int, float)):
            return [float(value)]
        return [float(item) for item in value]

    @field_validator("weighted_weather_quantiles", mode="before")
    @classmethod
    def _coerce_weighted_weather_quantiles(cls, value: Any) -> list[float]:
        if value is None:
            return []
        if isinstance(value, (int, float)):
            return [float(value)]
        return [float(item) for item in value]

    @field_validator("extra_open_meteo_weather_files", mode="before")
    @classmethod
    def _coerce_extra_open_meteo_weather_files(cls, value: Any) -> list[Path]:
        if value is None:
            return []
        if isinstance(value, (str, Path)):
            return [Path(value)]
        return [Path(item) for item in value]

    @field_validator("open_meteo_weather_ensemble_stats", mode="before")
    @classmethod
    def _coerce_open_meteo_weather_ensemble_stats(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return [str(item) for item in value]

    @field_validator("regional_load_components", mode="before")
    @classmethod
    def _coerce_regional_load_components(cls, value: Any) -> dict[str, str]:
        if value is None:
            return {}
        if isinstance(value, dict):
            return {str(name): str(country_code) for name, country_code in value.items()}
        raise TypeError("regional_load_components must be a mapping of feature name to ENTSO-E country code.")

    @model_validator(mode="after")
    def _resolve_paths(self) -> "LoadForecastModelConfig":
        if self.chunk_days < 1:
            raise ValueError("chunk_days must be positive.")
        if self.train_days_rolling < 1:
            raise ValueError("train_days_rolling must be positive.")
        if self.min_train_days < 1:
            raise ValueError("min_train_days must be positive.")
        if self.tabpfn_max_train_rows is not None and self.tabpfn_max_train_rows < 1:
            raise ValueError("tabpfn_max_train_rows must be positive when provided.")
        if self.tabpfn_context_recent_fraction < 0:
            raise ValueError("tabpfn_context_recent_fraction must be non-negative.")
        if self.tabpfn_context_same_weekday_fraction < 0:
            raise ValueError("tabpfn_context_same_weekday_fraction must be non-negative.")
        if self.tabpfn_n_estimators < 1:
            raise ValueError("tabpfn_n_estimators must be positive.")
        if self.tabpfn_n_preprocessing_jobs < 1:
            raise ValueError("tabpfn_n_preprocessing_jobs must be positive.")
        if self.tabpfn_predict_batch_size is not None and self.tabpfn_predict_batch_size < 1:
            raise ValueError("tabpfn_predict_batch_size must be positive when provided.")
        if self.target_availability_lag_days < 0:
            raise ValueError("target_availability_lag_days must be non-negative.")
        if self.target_availability_cutoff_hour is not None and not (0 <= self.target_availability_cutoff_hour <= 23):
            raise ValueError("target_availability_cutoff_hour must be between 0 and 23.")
        if self.target_availability_cutoff_minute not in {0, 15, 30, 45}:
            raise ValueError("target_availability_cutoff_minute must be one of 0, 15, 30, or 45.")
        if self.dwd_icon_download_timeout_seconds < 1:
            raise ValueError("dwd_icon_download_timeout_seconds must be positive.")
        if self.dwd_icon_request_pause_seconds < 0:
            raise ValueError("dwd_icon_request_pause_seconds must be non-negative.")
        if self.dwd_icon_aggregation_n_clusters is not None and self.dwd_icon_aggregation_n_clusters < 1:
            raise ValueError("dwd_icon_aggregation_n_clusters must be positive when provided.")
        if self.dwd_icon_aggregation_buffer_km < 0:
            raise ValueError("dwd_icon_aggregation_buffer_km must be non-negative.")
        if any(lag_day <= 0 for lag_day in self.actual_load_lag_days):
            raise ValueError("actual_load_lag_days must contain positive integers.")
        if any(lag_day <= 0 for lag_day in self.entsoe_error_lag_days):
            raise ValueError("entsoe_error_lag_days must contain positive integers.")
        if any(lag_day <= self.target_availability_lag_days for lag_day in self.entsoe_error_lag_days):
            raise ValueError("entsoe_error_lag_days must be greater than target_availability_lag_days.")
        if any(window_day <= 0 for window_day in self.entsoe_error_rolling_windows_days):
            raise ValueError("entsoe_error_rolling_windows_days must contain positive integers.")
        unknown_error_groups = set(self.entsoe_error_rolling_groups) - {"global", "hour", "mtu"}
        if unknown_error_groups:
            raise ValueError(f"entsoe_error_rolling_groups contains unsupported groups: {sorted(unknown_error_groups)}")
        if self.entsoe_error_rolling_min_observations < 1:
            raise ValueError("entsoe_error_rolling_min_observations must be positive.")
        if self.calendar_harmonics < 1:
            raise ValueError("calendar_harmonics must be at least 1.")
        if self.include_weighted_weather_features and not self.include_weather_features:
            raise ValueError("include_weighted_weather_features requires include_weather_features.")
        if self.weather_source == "open_meteo":
            if self.open_meteo_batch_size < 1:
                raise ValueError("open_meteo_batch_size must be positive.")
            if self.open_meteo_timeout_seconds < 1:
                raise ValueError("open_meteo_timeout_seconds must be positive.")
            if self.open_meteo_single_run_forecast_days < 1:
                raise ValueError("open_meteo_single_run_forecast_days must be positive.")
            if self.open_meteo_request_pause_seconds < 0:
                raise ValueError("open_meteo_request_pause_seconds must be non-negative.")
            if self.open_meteo_retry_attempts < 0:
                raise ValueError("open_meteo_retry_attempts must be non-negative.")
            if self.open_meteo_retry_backoff_seconds < 0:
                raise ValueError("open_meteo_retry_backoff_seconds must be non-negative.")
            if self.open_meteo_max_points_per_cluster is not None and self.open_meteo_max_points_per_cluster < 1:
                raise ValueError("open_meteo_max_points_per_cluster must be positive when provided.")
            valid_ensemble_stats = {"mean", "std", "min", "max", "range"}
            unknown_ensemble_stats = set(self.open_meteo_weather_ensemble_stats) - valid_ensemble_stats
            if unknown_ensemble_stats:
                raise ValueError(
                    "open_meteo_weather_ensemble_stats contains unsupported stats: "
                    f"{sorted(unknown_ensemble_stats)}"
                )
            if self.include_open_meteo_weather_ensemble_features and not self.open_meteo_weather_ensemble_stats:
                raise ValueError(
                    "open_meteo_weather_ensemble_stats must not be empty when ensemble weather features are enabled."
                )
        if self.include_weighted_weather_features and self.weather_cluster_weight_file is None:
            raise ValueError("weather_cluster_weight_file is required when include_weighted_weather_features is true.")
        if self.include_weighted_weather_daily_features and not self.include_weighted_weather_features:
            raise ValueError("include_weighted_weather_daily_features requires include_weighted_weather_features.")
        if self.include_weighted_weather_inertia_features and not self.include_weighted_weather_features:
            raise ValueError("include_weighted_weather_inertia_features requires include_weighted_weather_features.")
        if self.include_weather_cluster_spread_features and not self.include_weather_features:
            raise ValueError("include_weather_cluster_spread_features requires include_weather_features.")
        if self.include_weighted_weather_quantile_features and not self.include_weather_features:
            raise ValueError("include_weighted_weather_quantile_features requires include_weather_features.")
        if self.include_weighted_weather_quantile_features and self.weather_cluster_weight_file is None:
            raise ValueError("weather_cluster_weight_file is required for weighted weather quantiles.")
        if self.include_regional_holiday_features and self.regional_holiday_weight_file is None:
            raise ValueError("regional_holiday_weight_file is required when include_regional_holiday_features is true.")
        if self.include_regional_load_features and not self.regional_load_components:
            raise ValueError("regional_load_components is required when include_regional_load_features is true.")
        if self.include_regional_load_features and not any(
            [
                self.include_regional_load_forecast_features,
                self.include_regional_actual_load_lag_features,
                self.include_regional_partial_load_features,
                self.include_regional_load_summary_features,
            ]
        ):
            raise ValueError("At least one regional load feature subgroup must be enabled.")
        if not self.weather_cluster_id_column:
            raise ValueError("weather_cluster_id_column must not be empty.")
        if not self.weather_cluster_weight_column:
            raise ValueError("weather_cluster_weight_column must not be empty.")
        if not self.weather_weighted_feature_prefix:
            raise ValueError("weather_weighted_feature_prefix must not be empty.")
        if not self.regional_holiday_region_column:
            raise ValueError("regional_holiday_region_column must not be empty.")
        if not self.regional_holiday_weight_column:
            raise ValueError("regional_holiday_weight_column must not be empty.")
        if not self.regional_holiday_feature_prefix:
            raise ValueError("regional_holiday_feature_prefix must not be empty.")
        if self.partial_load_reference_day < 1:
            raise ValueError("partial_load_reference_day must be positive.")
        if self.partial_load_comparison_lag_days < 1:
            raise ValueError("partial_load_comparison_lag_days must be positive.")
        if not (0 <= self.partial_load_morning_end_hour <= 23):
            raise ValueError("partial_load_morning_end_hour must be between 0 and 23.")
        if self.partial_load_morning_end_minute not in {0, 15, 30, 45}:
            raise ValueError("partial_load_morning_end_minute must be one of 0, 15, 30, or 45.")
        valid_daily_stats = {"mean", "min", "max", "range"}
        unknown_daily_stats = set(self.weighted_weather_daily_stats) - valid_daily_stats
        if unknown_daily_stats:
            raise ValueError(f"weighted_weather_daily_stats contains unsupported stats: {sorted(unknown_daily_stats)}")
        if any(lag_day <= 0 for lag_day in self.weighted_weather_daily_lag_days):
            raise ValueError("weighted_weather_daily_lag_days must contain positive integers.")
        valid_inertia_stats = {"mean", "min", "max", "range", "delta_mean"}
        unknown_inertia_stats = set(self.weighted_weather_inertia_stats) - valid_inertia_stats
        if unknown_inertia_stats:
            raise ValueError(f"weighted_weather_inertia_stats contains unsupported stats: {sorted(unknown_inertia_stats)}")
        if any(window <= 0 for window in self.weighted_weather_inertia_windows_hours):
            raise ValueError("weighted_weather_inertia_windows_hours must contain positive integers.")
        valid_spread_stats = {"mean", "min", "max", "range", "std", "p10", "p90"}
        unknown_spread_stats = set(self.weather_spread_stats) - valid_spread_stats
        if unknown_spread_stats:
            raise ValueError(f"weather_spread_stats contains unsupported stats: {sorted(unknown_spread_stats)}")
        if any(quantile <= 0.0 or quantile >= 1.0 for quantile in self.weighted_weather_quantiles):
            raise ValueError("weighted_weather_quantiles must be between 0 and 1.")
        for point_time in self.partial_load_point_times:
            parts = point_time.split(":")
            if len(parts) != 2:
                raise ValueError("partial_load_point_times entries must use HH:MM format.")
            hour, minute = int(parts[0]), int(parts[1])
            if not (0 <= hour <= 23) or minute not in {0, 15, 30, 45}:
                raise ValueError("partial_load_point_times must be quarter-hour times between 00:00 and 23:45.")
        if self.include_rich_temperature_features:
            self.weather_hdd_thresholds = sorted({float(threshold) for threshold in self.weather_hdd_thresholds} | {18.0})
            self.weather_cdd_thresholds = sorted({float(threshold) for threshold in self.weather_cdd_thresholds} | {22.0})
        if any(threshold < -50 or threshold > 60 for threshold in self.weather_hdd_thresholds + self.weather_cdd_thresholds):
            raise ValueError("Weather degree thresholds must be plausible Celsius values.")
        if self.model_granularity == "hour_block":
            if len(self.hour_block_boundaries) < 2:
                raise ValueError("hour_block_boundaries must contain at least two boundaries.")
            if self.hour_block_boundaries[0] != 0 or self.hour_block_boundaries[-1] != 24:
                raise ValueError("hour_block_boundaries must start at 0 and end at 24.")
            if sorted(set(self.hour_block_boundaries)) != self.hour_block_boundaries:
                raise ValueError("hour_block_boundaries must be strictly increasing.")
            if any(boundary < 0 or boundary > 24 for boundary in self.hour_block_boundaries):
                raise ValueError("hour_block_boundaries must be between 0 and 24.")
        if self.lgbm_n_estimators < 1:
            raise ValueError("lgbm_n_estimators must be positive.")
        if self.lgbm_learning_rate <= 0:
            raise ValueError("lgbm_learning_rate must be positive.")
        if self.lgbm_num_leaves < 2:
            raise ValueError("lgbm_num_leaves must be at least 2.")
        if self.lgbm_min_child_samples < 1:
            raise ValueError("lgbm_min_child_samples must be positive.")
        if not (0 < self.lgbm_subsample <= 1.0):
            raise ValueError("lgbm_subsample must be in (0, 1].")
        if not (0 < self.lgbm_colsample_bytree <= 1.0):
            raise ValueError("lgbm_colsample_bytree must be in (0, 1].")
        if self.lgbm_reg_lambda < 0:
            raise ValueError("lgbm_reg_lambda must be non-negative.")
        if self.residual_quantile_window_days < 1:
            raise ValueError("residual_quantile_window_days must be positive.")
        if self.residual_quantile_min_observations < 1:
            raise ValueError("residual_quantile_min_observations must be positive.")
        if self.include_rolling_residual_quantiles:
            if not self.residual_quantiles:
                raise ValueError("residual_quantiles must not be empty when rolling residual quantiles are enabled.")
            invalid_quantiles = [quantile for quantile in self.residual_quantiles if not 0.0 < quantile < 1.0]
            if invalid_quantiles:
                raise ValueError(f"residual_quantiles must be between 0 and 1: {invalid_quantiles}")
            self.residual_quantiles = sorted({float(quantile) for quantile in self.residual_quantiles})

        self.actual_load_file = resolve_path(self.actual_load_file, self.repo_root)
        self.entsoe_load_forecast_file = resolve_path(self.entsoe_load_forecast_file, self.repo_root)
        self.regional_load_cache_dir = resolve_path(self.regional_load_cache_dir, self.repo_root)
        self.icon_dir = resolve_path(self.icon_dir, self.repo_root)
        self.dwd_icon_raw_dir = resolve_path(self.dwd_icon_raw_dir, self.repo_root)
        self.dwd_icon_aggregation_shapefile_path = resolve_path(self.dwd_icon_aggregation_shapefile_path, self.repo_root)
        self.dwd_icon_aggregation_capacity_file = resolve_path(self.dwd_icon_aggregation_capacity_file, self.repo_root)
        if self.dwd_icon_aggregation_cluster_output_file is not None:
            self.dwd_icon_aggregation_cluster_output_file = resolve_path(
                self.dwd_icon_aggregation_cluster_output_file,
                self.repo_root,
            )
        self.open_meteo_weather_file = resolve_path(self.open_meteo_weather_file, self.repo_root)
        self.extra_open_meteo_weather_files = [
            resolve_path(path, self.repo_root) for path in self.extra_open_meteo_weather_files
        ]
        self.open_meteo_cluster_file = resolve_path(self.open_meteo_cluster_file, self.repo_root)
        self.export_dir = resolve_path(self.export_dir, self.repo_root)
        if self.weather_cluster_weight_file is not None:
            self.weather_cluster_weight_file = resolve_path(self.weather_cluster_weight_file, self.repo_root)
        if self.regional_holiday_weight_file is not None:
            self.regional_holiday_weight_file = resolve_path(self.regional_holiday_weight_file, self.repo_root)
        return self


class LoadForecastEnsembleSourceConfig(RepoConfigModel):
    name: str
    path: Path
    weight: float = 1.0

    @model_validator(mode="after")
    def _resolve_path(self) -> "LoadForecastEnsembleSourceConfig":
        self.path = resolve_path(self.path, self.repo_root)
        return self


class LoadForecastEnsembleConfig(RepoConfigModel):
    """Blend saved load forecast files and optionally calibrate residual quantiles."""

    target_tz: str = "Europe/Berlin"
    sources: list[LoadForecastEnsembleSourceConfig] = Field(default_factory=list)
    export_dir: Path = Path("results/load_forecast_results/load_forecast_ensemble")
    method: Literal["mean", "median", "fixed"] = "mean"
    test_start: date = date(2025, 12, 1)
    test_end: date = date(2026, 2, 28)
    target_availability_lag_days: int = 1

    include_rolling_residual_quantiles: bool = False
    residual_quantiles: list[float] = Field(default_factory=lambda: [0.025, 0.25, 0.50, 0.75, 0.975])
    residual_quantile_window_days: int = 28
    residual_quantile_min_observations: int = 20
    residual_quantile_group: Literal["global", "hour", "mtu"] = "mtu"
    residual_quantile_spread_scale: float = Field(default=1.0, gt=0.0)
    quantile_evaluation_start: date | None = None
    quantile_evaluation_end: date | None = None

    @field_validator("residual_quantiles", mode="before")
    @classmethod
    def _coerce_residual_quantiles(cls, value: Any) -> list[float]:
        if value is None:
            return []
        if isinstance(value, (int, float)):
            return [float(value)]
        return [float(item) for item in value]

    @model_validator(mode="after")
    def _resolve_paths(self) -> "LoadForecastEnsembleConfig":
        if len(self.sources) < 2:
            raise ValueError("Load forecast ensemble requires at least two sources.")
        source_names = [source.name for source in self.sources]
        if len(set(source_names)) != len(source_names):
            raise ValueError("Load forecast ensemble source names must be unique.")
        if self.target_availability_lag_days < 0:
            raise ValueError("target_availability_lag_days must be non-negative.")
        if self.residual_quantile_window_days < 1:
            raise ValueError("residual_quantile_window_days must be positive.")
        if self.residual_quantile_min_observations < 1:
            raise ValueError("residual_quantile_min_observations must be positive.")
        if self.method == "fixed":
            weights = [source.weight for source in self.sources]
            if any(weight < 0 for weight in weights) or sum(weights) <= 0:
                raise ValueError("Fixed load ensemble weights must be non-negative and sum to a positive value.")
        if self.include_rolling_residual_quantiles:
            if not self.residual_quantiles:
                raise ValueError("residual_quantiles must not be empty when rolling residual quantiles are enabled.")
            invalid_quantiles = [quantile for quantile in self.residual_quantiles if not 0.0 < quantile < 1.0]
            if invalid_quantiles:
                raise ValueError(f"residual_quantiles must be between 0 and 1: {invalid_quantiles}")
            self.residual_quantiles = sorted({float(quantile) for quantile in self.residual_quantiles})
        for source in self.sources:
            source.path = resolve_path(source.path, self.repo_root)
        self.export_dir = resolve_path(self.export_dir, self.repo_root)
        return self


class RegionalLoadForecastComponentConfig(RepoConfigModel):
    """One regional ENTSO-E load component used in a summed DE-LU forecast."""

    name: str
    country_code_entsoe: str
    actual_load_file: Path | None = None
    entsoe_load_forecast_file: Path | None = None
    export_dir: Path | None = None
    weather_cluster_weight_file: Path | None = None
    regional_holiday_weight_file: Path | None = None

    @model_validator(mode="after")
    def _resolve_paths(self) -> "RegionalLoadForecastComponentConfig":
        if not self.name:
            raise ValueError("Regional load component name must not be empty.")
        if not self.country_code_entsoe:
            raise ValueError("Regional load component country_code_entsoe must not be empty.")
        if self.actual_load_file is not None:
            self.actual_load_file = resolve_path(self.actual_load_file, self.repo_root)
        if self.entsoe_load_forecast_file is not None:
            self.entsoe_load_forecast_file = resolve_path(self.entsoe_load_forecast_file, self.repo_root)
        if self.export_dir is not None:
            self.export_dir = resolve_path(self.export_dir, self.repo_root)
        if self.weather_cluster_weight_file is not None:
            self.weather_cluster_weight_file = resolve_path(self.weather_cluster_weight_file, self.repo_root)
        if self.regional_holiday_weight_file is not None:
            self.regional_holiday_weight_file = resolve_path(self.regional_holiday_weight_file, self.repo_root)
        return self


class RegionalLoadForecastConfig(RepoConfigModel):
    """Run one load model per ENTSO-E region and sum the component forecasts."""

    base_config: LoadForecastModelConfig
    components: list[RegionalLoadForecastComponentConfig] = Field(default_factory=list)
    export_dir: Path = Path("results/load_forecast_results/regional_load_forecast")
    component_cache_dir: Path = Path("data/processed/load_forecast/regional_components")
    target_tz: str | None = None
    reuse_component_forecasts: bool = True
    include_aggregate_residual_component: bool = False
    aggregate_component_name: str = "aggregate_residual"
    aggregate_country_code_entsoe: str = "DE_LU"
    aggregate_actual_load_file: Path | None = None
    aggregate_entsoe_load_forecast_file: Path | None = None

    @model_validator(mode="before")
    @classmethod
    def _propagate_repo_root(cls, value: Any) -> Any:
        if not isinstance(value, dict) or "repo_root" not in value:
            return value

        repo_root = value["repo_root"]
        data = dict(value)
        base_config = data.get("base_config")
        if isinstance(base_config, dict) and "repo_root" not in base_config:
            data["base_config"] = {**base_config, "repo_root": repo_root}

        components = data.get("components")
        if isinstance(components, list):
            data["components"] = [
                {**component, "repo_root": repo_root}
                if isinstance(component, dict) and "repo_root" not in component
                else component
                for component in components
            ]
        return data

    @model_validator(mode="after")
    def _resolve_paths(self) -> "RegionalLoadForecastConfig":
        if not self.components:
            raise ValueError("Regional load forecast requires at least one component.")
        component_names = [component.name for component in self.components]
        if len(set(component_names)) != len(component_names):
            raise ValueError("Regional load component names must be unique.")
        self.export_dir = resolve_path(self.export_dir, self.repo_root)
        self.component_cache_dir = resolve_path(self.component_cache_dir, self.repo_root)
        if self.aggregate_actual_load_file is not None:
            self.aggregate_actual_load_file = resolve_path(self.aggregate_actual_load_file, self.repo_root)
        if self.aggregate_entsoe_load_forecast_file is not None:
            self.aggregate_entsoe_load_forecast_file = resolve_path(self.aggregate_entsoe_load_forecast_file, self.repo_root)
        if self.target_tz is None:
            self.target_tz = self.base_config.target_tz
        if self.include_aggregate_residual_component and not self.aggregate_component_name:
            raise ValueError("aggregate_component_name must not be empty when aggregate residual is enabled.")
        if self.include_aggregate_residual_component and not self.aggregate_country_code_entsoe:
            raise ValueError("aggregate_country_code_entsoe must not be empty when aggregate residual is enabled.")
        return self


class EntsoeLoadForecastBenchmarkConfig(RepoConfigModel):
    """Benchmark ENTSO-E total load forecasts against actual load."""

    target_tz: str = "Europe/Berlin"
    country_code_entsoe: str = "DE_LU"
    entsoe_api_key_env: str = "ENTSOE_API_KEY"
    entsoe_start_date: date = date(2025, 8, 1)
    entsoe_end_date: date = date(2026, 4, 23)
    test_start: date | None = date(2025, 12, 1)
    test_end: date | None = date(2026, 2, 28)
    chunk_days: int = 90

    actual_load_file: Path = Path("data/processed/load_forecast/actual_load.csv")
    forecast_file: Path = Path("data/processed/load_forecast/entsoe_load_forecast.csv")
    export_dir: Path = Path("results/load_forecast_results/entsoe_load_forecast_benchmark")

    @model_validator(mode="after")
    def _resolve_paths(self) -> "EntsoeLoadForecastBenchmarkConfig":
        if self.chunk_days < 1:
            raise ValueError("chunk_days must be positive.")
        self.actual_load_file = resolve_path(self.actual_load_file, self.repo_root)
        self.forecast_file = resolve_path(self.forecast_file, self.repo_root)
        self.export_dir = resolve_path(self.export_dir, self.repo_root)
        return self
