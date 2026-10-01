from __future__ import annotations

from datetime import date
from pathlib import Path

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from ..paths import resolve_path
from .base import RepoConfigModel


class IconAggregationTargetConfig(BaseModel):
    """One output target for a shared ICON aggregation pass."""

    name: str | None = None
    output_parent: Path
    cluster_source: Literal["grid", "mastr_solar", "mastr_solar_tso", "mastr_wind"] = "grid"
    n_clusters: int = 5
    cluster_output_file: Path | None = None
    capacity_file: Path | None = None
    capacity_weighted_aggregation: bool | None = None
    variables: list[str] | None = None
    only_run_hour: str | None = None
    run_hours: list[str] = Field(default_factory=list)


class IconAggregationConfig(RepoConfigModel):
    lsdf_base: Path = Path("/Volumes/iip-projects/energy/climateData/icon_by_Max_Kleinebrahm")
    dwd_model: str = "icon-d2"
    dwd_grid_type: str = "regular-lat-lon"
    icon_grid_file: Path | None = None
    output_parent: Path = Path("data/processed/icon_aggregated_c5")
    shapefile_path: Path = Path("data/shapefile/ne_10m_admin_0_countries.shp")
    cluster_source: Literal["grid", "mastr_solar", "mastr_solar_tso", "mastr_wind"] = "grid"
    cluster_output_file: Path | None = None
    capacity_file: Path = Path("data/raw/renewable_capacity/installed_capacity.csv")
    capacity_weighted_aggregation: bool = True
    n_clusters: int = 5
    # Single-pass sweep over several grid cluster counts: read each GRIB file once and
    # aggregate it to all counts. Only supported for cluster_source == "grid". The two
    # templates must contain a "{n_clusters}" placeholder (e.g.
    # "data/processed/icon_aggregated_c{n_clusters}_run06").
    grid_cluster_counts: list[int] = Field(default_factory=list)
    output_parent_template: str | None = None
    cluster_output_file_template: str | None = None
    buffer_km: int = 50
    plot_clusters: bool = False
    skip_existing_output: bool = True
    only_day: str | None = None
    only_run_hour: str | None = "09"
    start_date: date = date(2026, 1, 3)
    variables: list[str] = Field(
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
    model_level_variables: list[str] = Field(default_factory=list)
    model_levels: list[int] = Field(default_factory=list)
    ensemble_statistics: list[Literal["mean", "std"]] = Field(default_factory=list)
    technology_column: str = "technology"
    capacity_column: str = "capacity_mw"
    latitude_column: str = "lat"
    longitude_column: str = "lon"
    federal_state_column: str = "federal_state_code"
    operating_status_column: str = "operating_status"
    active_status_codes: list[str] = Field(default_factory=lambda: ["35"])
    solar_values: list[str] = Field(default_factory=lambda: ["pv", "solar", "photovoltaic"])
    wind_onshore_values: list[str] = Field(default_factory=lambda: ["wind_onshore"])
    wind_offshore_values: list[str] = Field(default_factory=lambda: ["wind_offshore"])
    # Optional multi-output mode: stream each GRIB variable once per day/run and write
    # the same decoded field to several aggregation layouts.
    aggregation_targets: list[IconAggregationTargetConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def _resolve_paths(self) -> "IconAggregationConfig":
        self.shapefile_path = resolve_path(self.shapefile_path, self.repo_root)
        self.output_parent = resolve_path(self.output_parent, self.repo_root)
        self.capacity_file = resolve_path(self.capacity_file, self.repo_root)
        if self.icon_grid_file is not None:
            self.icon_grid_file = resolve_path(self.icon_grid_file, self.repo_root)
        if self.cluster_output_file is not None:
            self.cluster_output_file = resolve_path(self.cluster_output_file, self.repo_root)
        if self.output_parent_template is not None:
            self.output_parent_template = str(resolve_path(Path(self.output_parent_template), self.repo_root))
        if self.cluster_output_file_template is not None:
            self.cluster_output_file_template = str(
                resolve_path(Path(self.cluster_output_file_template), self.repo_root)
            )
        if self.n_clusters < 1:
            raise ValueError("n_clusters must be positive.")
        if self.grid_cluster_counts:
            if self.cluster_source != "grid":
                raise ValueError("grid_cluster_counts is only supported for cluster_source='grid'.")
            if any(count < 1 for count in self.grid_cluster_counts):
                raise ValueError("grid_cluster_counts must contain positive cluster counts.")
            if self.output_parent_template is None or "{n_clusters}" not in self.output_parent_template:
                raise ValueError(
                    "output_parent_template must be set and contain '{n_clusters}' when "
                    "grid_cluster_counts is used."
                )
            if self.cluster_output_file_template is not None and "{n_clusters}" not in self.cluster_output_file_template:
                raise ValueError("cluster_output_file_template must contain '{n_clusters}'.")
        if any(level < 1 for level in self.model_levels):
            raise ValueError("model_levels must contain positive ICON model-level ids.")
        if self.model_level_variables and not self.model_levels:
            raise ValueError("model_levels must be set when model_level_variables are requested.")
        for target in self.aggregation_targets:
            target.output_parent = resolve_path(target.output_parent, self.repo_root)
            if target.cluster_output_file is not None:
                target.cluster_output_file = resolve_path(target.cluster_output_file, self.repo_root)
            if target.capacity_file is not None:
                target.capacity_file = resolve_path(target.capacity_file, self.repo_root)
            if target.n_clusters < 1:
                raise ValueError("aggregation target n_clusters must be positive.")
            if target.only_run_hour is not None and target.run_hours:
                raise ValueError("Set either only_run_hour or run_hours for an aggregation target, not both.")
            if target.only_run_hour is not None:
                target.run_hours = [target.only_run_hour]
            if target.variables is not None and not target.variables:
                raise ValueError("aggregation target variables must not be empty when set.")
        return self

    @property
    def root_dir_pattern(self) -> str:
        return str(self.lsdf_base / "dwd_icon_daily_*")


class RenewableProxyConfig(RepoConfigModel):
    """Build DWD ICON-D2 capacity-weighted renewable generation proxy features."""

    icon_dir: Path = Path("data/processed/icon_aggregated_c5")
    capacity_file: Path = Path("data/raw/renewable_capacity/installed_capacity.csv")
    cluster_file: Path | None = None
    output_file: Path = Path("data/processed/renewable_proxy/dwd_icon_c5_renewable_proxy.csv")
    target_tz: str = "Europe/Berlin"

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

    capacity_format: Literal["plant_locations", "cluster_capacity"] = "plant_locations"
    technology_column: str = "technology"
    capacity_column: str = "capacity_mw"
    capacity_unit: Literal["MW", "kW"] = "MW"
    latitude_column: str = "lat"
    longitude_column: str = "lon"
    cluster_id_column: str = "cluster_id"
    wind_technology_values: list[str] = Field(default_factory=lambda: ["wind", "wind_onshore", "wind_offshore"])
    solar_technology_values: list[str] = Field(default_factory=lambda: ["solar", "pv", "photovoltaic"])

    wind_hub_height_m: float = 100.0
    wind_reference_height_m: float = 10.0
    wind_shear_alpha: float = 0.14
    wind_cut_in_m_s: float = 3.0
    wind_rated_m_s: float = 12.0
    wind_cut_out_m_s: float = 25.0

    solar_reference_irradiance_w_m2: float = 1000.0
    solar_performance_ratio: float = 0.85

    @model_validator(mode="after")
    def _resolve_paths(self) -> "RenewableProxyConfig":
        self.icon_dir = resolve_path(self.icon_dir, self.repo_root)
        self.capacity_file = resolve_path(self.capacity_file, self.repo_root)
        self.output_file = resolve_path(self.output_file, self.repo_root)
        if self.cluster_file is not None:
            self.cluster_file = resolve_path(self.cluster_file, self.repo_root)
        return self


class PopulationClusterWeightsConfig(RepoConfigModel):
    """Build population weights for weather clusters from a gridded population table."""

    source_file: Path | None = None
    source_url: str | None = "https://gisco-services.ec.europa.eu/grid/grid_1km.parquet"
    raw_file: Path = Path("data/raw/population/gisco_grid_1km.parquet")
    cluster_file: Path = Path("data/clustering/icon_d2_clustering_c25.parquet")
    output_file: Path = Path("data/processed/load_forecast/weather_cluster_population_weights_c25.csv")
    region_output_file: Path | None = Path("data/processed/load_forecast/weather_cluster_population_region_weights_c25.csv")
    summary_file: Path | None = None

    country_codes: list[str] = Field(default_factory=lambda: ["DE", "LU"])
    country_column: str = "CNTR_ID"
    population_column: str = "TOT_P_2021"
    x_column: str = "X_LLC"
    y_column: str = "Y_LLC"
    cell_size_m: float = 1000.0
    region_column: str | None = "NUTS2021_1"
    cluster_lon_column: str = "lon"
    cluster_lat_column: str = "lat"
    cluster_id_column: str = "cluster_id"
    cluster_crs_epsg: int = 4326
    population_crs_epsg: int = 3035
    matching_mode: Literal["nearest_grid_point", "nearest_centroid"] = "nearest_grid_point"
    force_download: bool = False
    timeout_seconds: int = 120

    @model_validator(mode="after")
    def _resolve_paths(self) -> "PopulationClusterWeightsConfig":
        if self.source_file is not None:
            self.source_file = resolve_path(self.source_file, self.repo_root)
        self.raw_file = resolve_path(self.raw_file, self.repo_root)
        self.cluster_file = resolve_path(self.cluster_file, self.repo_root)
        self.output_file = resolve_path(self.output_file, self.repo_root)
        if self.region_output_file is not None:
            self.region_output_file = resolve_path(self.region_output_file, self.repo_root)
        if self.summary_file is not None:
            self.summary_file = resolve_path(self.summary_file, self.repo_root)
        if self.source_file is None and not self.source_url:
            raise ValueError("Either source_file or source_url is required.")
        if self.cell_size_m <= 0:
            raise ValueError("cell_size_m must be positive.")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")
        if not self.country_codes:
            raise ValueError("country_codes must contain at least one country code.")
        return self


class ReserveMarketConfig(RepoConfigModel):
    """Download public regelleistung.net capacity-market results and build timestamp features."""

    target_tz: str = "Europe/Berlin"
    start_date: date = date(2025, 10, 1)
    end_date: date = date(2026, 7, 31)
    raw_dir: Path = Path("data/raw/reserve_market/regelleistung")
    output_file: Path = Path("data/processed/reserve_market/regelleistung_capacity_features.csv")
    metadata_file: Path | None = Path("data/processed/reserve_market/regelleistung_capacity_downloads.csv")
    base_url: str = "https://www.regelleistung.net/apps/crds/api/v2"
    product_types: list[Literal["FCR", "aFRR", "mFRR"]] = Field(
        default_factory=lambda: ["FCR", "aFRR", "mFRR"]
    )
    request_timeout_seconds: float = 60.0
    force_download: bool = False
    fail_on_missing: bool = False
    publication_times: dict[str, str] = Field(
        default_factory=lambda: {"FCR": "08:30", "aFRR": "09:30", "mFRR": "10:30"}
    )

    @model_validator(mode="after")
    def _resolve_paths(self) -> "ReserveMarketConfig":
        self.raw_dir = resolve_path(self.raw_dir, self.repo_root)
        self.output_file = resolve_path(self.output_file, self.repo_root)
        if self.metadata_file is not None:
            self.metadata_file = resolve_path(self.metadata_file, self.repo_root)
        if self.end_date < self.start_date:
            raise ValueError("end_date must be on or after start_date.")
        if not self.product_types:
            raise ValueError("product_types must contain at least one reserve product.")
        for product_type, time_value in self.publication_times.items():
            if product_type not in {"FCR", "aFRR", "mFRR"}:
                raise ValueError(f"Unsupported publication time product: {product_type!r}.")
            parts = time_value.split(":")
            if len(parts) != 2:
                raise ValueError(f"Invalid publication time for {product_type}: {time_value!r}.")
            hour, minute = int(parts[0]), int(parts[1])
            if hour < 0 or hour > 23 or minute < 0 or minute > 59:
                raise ValueError(f"Invalid publication time for {product_type}: {time_value!r}.")
        return self


class RegionalRenewableFeatureConfig(RepoConfigModel):
    """Build regional capacity-weighted renewable weather features from MaStR and weather data."""

    weather_source: Literal["era5", "dwd_icon", "open_meteo"] = "era5"
    feature_mode: Literal["full", "cloud_cover"] = "full"
    era5_dirs: list[Path] = Field(default_factory=lambda: [Path("data/processed/era5_aggregated")])
    icon_dir: Path = Path("data/processed/icon_aggregated_c25")
    open_meteo_weather_file: Path = Path("data/processed/open_meteo/open_meteo_icon_d2_c25_weather.csv")
    open_meteo_base_url: str = "https://customer-historical-forecast-api.open-meteo.com/v1/forecast"
    open_meteo_api_key_env: str | None = "OPEN_METEO_API_KEY"
    open_meteo_model: str | None = "icon_d2"
    open_meteo_start_date: date = date(2025, 8, 1)
    open_meteo_end_date: date = date(2026, 2, 28)
    open_meteo_hourly_variables: list[str] = Field(
        default_factory=lambda: [
            "wind_speed_80m",
            "wind_direction_80m",
            "wind_speed_120m",
            "wind_direction_120m",
            "temperature_2m",
            "dew_point_2m",
            "surface_pressure",
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
    open_meteo_point_source: Literal["clusters", "capacity"] = "clusters"
    capacity_weather_point_strategy: Literal["kmeans", "kmeans_metadata", "capacity_cells"] = "kmeans"
    capacity_weather_point_file: Path = Path("data/processed/renewable_proxy/open_meteo_capacity_weather_points.csv")
    capacity_weather_cell_size_degrees: float = Field(default=0.05, gt=0.0)
    capacity_weather_min_cell_capacity_mw: float = Field(default=0.0, ge=0.0)
    capacity_weather_metadata_columns: list[str] = Field(
        default_factory=lambda: ["hub_height_m", "rotor_diameter_m"]
    )
    capacity_weather_solar_points_per_region: int = 15
    capacity_weather_onshore_points_per_region: int = 15
    capacity_weather_offshore_points: int = 25
    capacity_weather_random_state: int = 42
    capacity_weather_technologies: list[Literal["wind_offshore", "wind_onshore", "solar"]] = Field(
        default_factory=lambda: ["wind_offshore", "wind_onshore", "solar"]
    )
    capacity_file: Path = Path("data/raw/renewable_capacity/installed_capacity.csv")
    cluster_file: Path = Path("data/clustering/icon_d2_clustering_c25.parquet")
    capacity_map_file: Path = Path("data/processed/renewable_proxy/era5_regional_capacity_map.csv")
    capacity_timeseries_file: Path | None = None
    weather_weights_file: Path | None = None
    capacity_artifact_technologies: list[Literal["wind_offshore", "wind_onshore", "solar"]] = Field(
        default_factory=lambda: ["solar"]
    )
    output_file: Path = Path("data/processed/renewable_proxy/era5_regional_renewable_features.csv")
    target_tz: str = "Europe/Berlin"
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

    technology_column: str = "technology"
    capacity_column: str = "capacity_mw"
    latitude_column: str = "lat"
    longitude_column: str = "lon"
    federal_state_column: str = "federal_state_code"
    commissioning_date_column: str = "commissioning_date"
    decommissioning_date_column: str = "decommissioning_date"
    operating_status_column: str = "operating_status"
    active_status_codes: list[str] = Field(default_factory=lambda: ["35"])
    wind_onshore_values: list[str] = Field(default_factory=lambda: ["wind_onshore"])
    wind_offshore_values: list[str] = Field(default_factory=lambda: ["wind_offshore"])
    solar_values: list[str] = Field(default_factory=lambda: ["pv", "solar", "photovoltaic"])

    north_latitude: float = 52.0
    south_latitude: float = 49.5
    west_longitude: float = 8.0
    east_longitude: float = 11.0
    solar_region_strategy: Literal["geographic", "state_tso_proxy"] = "geographic"
    wind_hub_height_m: float = 100.0
    wind_reference_height_m: float = 10.0
    wind_shear_alpha: float = 0.14
    wind_hub_height_column: str = "hub_height_m"
    wind_rotor_diameter_column: str = "rotor_diameter_m"
    wind_swept_area_column: str = "swept_area_m2"
    wind_manufacturer_column: str = "manufacturer_code"
    wind_turbine_type_column: str = "turbine_type"
    include_wind_turbine_static_features: bool = False
    include_wind_hub_height_interpolation_features: bool = False
    # Fleet-smearing of the wind power curve. The single-turbine curve is too
    # steep for an aggregated fleet (thousands of turbines spread over a region
    # hit rated/cut-out at different local wind speeds), which over-predicts at
    # high wind. >0 convolves the curve with a Gaussian of this std (m/s) over
    # wind speed to approximate the smoother aggregate response. 0 = sharp curve.
    wind_power_curve_smoothing_sigma_m_s: float = Field(default=0.0, ge=0.0)
    wind_weather_weight_mode: Literal["capacity", "rotor_area", "capacity_x_rotor_area"] = "capacity"
    solar_performance_ratio: float = 0.85
    include_ramps: bool = True
    include_cluster_features: bool = False
    include_spatial_spread_features: bool = False
    include_capacity_static_features: bool = False
    include_geographic_exposure_features: bool = False

    @model_validator(mode="after")
    def _resolve_paths(self) -> "RegionalRenewableFeatureConfig":
        self.era5_dirs = [resolve_path(path, self.repo_root) for path in self.era5_dirs]
        self.icon_dir = resolve_path(self.icon_dir, self.repo_root)
        self.open_meteo_weather_file = resolve_path(self.open_meteo_weather_file, self.repo_root)
        self.capacity_weather_point_file = resolve_path(self.capacity_weather_point_file, self.repo_root)
        self.capacity_file = resolve_path(self.capacity_file, self.repo_root)
        self.cluster_file = resolve_path(self.cluster_file, self.repo_root)
        self.capacity_map_file = resolve_path(self.capacity_map_file, self.repo_root)
        if self.capacity_timeseries_file is not None:
            self.capacity_timeseries_file = resolve_path(self.capacity_timeseries_file, self.repo_root)
        if self.weather_weights_file is not None:
            self.weather_weights_file = resolve_path(self.weather_weights_file, self.repo_root)
        self.output_file = resolve_path(self.output_file, self.repo_root)
        if not self.capacity_artifact_technologies:
            raise ValueError("capacity_artifact_technologies must contain at least one technology.")
        if self.capacity_weather_solar_points_per_region < 1:
            raise ValueError("capacity_weather_solar_points_per_region must be positive.")
        if self.capacity_weather_onshore_points_per_region < 1:
            raise ValueError("capacity_weather_onshore_points_per_region must be positive.")
        if self.capacity_weather_offshore_points < 1:
            raise ValueError("capacity_weather_offshore_points must be positive.")
        return self


class MastrCapacityConfig(RepoConfigModel):
    """Convert a MaStR full XML export into the renewable capacity CSV."""

    mastr_dir: Path = Path("data/mastr/Gesamtdatenexport_20260506_26.1")
    output_file: Path = Path("data/raw/renewable_capacity/installed_capacity.csv")

    country_code: str = "84"
    active_status_code: str = "35"
    active_only: bool = True
    max_commissioning_date: date | None = None

    capacity_source: Literal["net", "gross"] = "net"
    capacity_unit: Literal["kW"] = "kW"
    include_wind_turbine_metadata: bool = True

    @model_validator(mode="after")
    def _resolve_paths(self) -> "MastrCapacityConfig":
        self.mastr_dir = resolve_path(self.mastr_dir, self.repo_root)
        self.output_file = resolve_path(self.output_file, self.repo_root)
        return self


class RenewableGenerationModelConfig(RepoConfigModel):
    """Train/evaluate a first-stage actual renewable generation forecasting model."""

    target_tz: str = "Europe/Berlin"
    country_code_entsoe: str = "DE_LU"
    entsoe_api_key_env: str = "ENTSOE_API_KEY"
    entsoe_start_date: date = date(2025, 11, 14)
    entsoe_end_date: date = date(2026, 2, 28)
    include_solar_control_area_targets: bool = False
    solar_control_area_targets: dict[str, str] = Field(
        default_factory=lambda: {
            "Solar_50Hertz_Actual_MW": "10YDE-VE-------2",
            "Solar_Amprion_Actual_MW": "10YDE-RWENET---I",
            "Solar_TenneT_Actual_MW": "10YDE-EON------1",
            "Solar_TransnetBW_Actual_MW": "10YDE-ENBW-----N",
        }
    )

    actual_generation_file: Path = Path("data/processed/renewable_generation/actual_generation.csv")
    renewable_proxy_file: Path = Path("data/processed/renewable_proxy/dwd_icon_c5_renewable_proxy.csv")
    renewable_proxy_fallback_file: Path | None = None
    renewable_proxy_fallback_end_date: date | None = None
    extra_renewable_proxy_files: list[Path] = Field(default_factory=list)
    extra_renewable_proxy_prefixes: list[str] = Field(default_factory=list)
    extra_renewable_proxy_ensemble_mode: Literal["raw", "summary"] = "raw"
    extra_renewable_proxy_ensemble_prefix: str = "ens_"
    extra_renewable_proxy_ensemble_stats: list[Literal["mean", "std", "min", "max", "range"]] = Field(
        default_factory=lambda: ["mean", "std"]
    )
    extra_renewable_proxy_ensemble_column_patterns: list[str] = Field(default_factory=list)
    extra_renewable_proxy_ensemble_groups: dict[str, list[int]] = Field(default_factory=dict)
    extra_renewable_proxy_ensemble_group_diff_pairs: list[list[str]] = Field(default_factory=list)
    raw_extra_renewable_proxy_files: list[Path] = Field(default_factory=list)
    raw_extra_renewable_proxy_prefixes: list[str] = Field(default_factory=list)
    unavailability_file: Path = Path("data/processed/renewable_generation/generation_unavailability.csv")
    icon_dir: Path = Path("data/processed/icon_aggregated_c5")
    export_dir: Path = Path("results/renewable_generation_results/dwd_icon_c5_hgb_decfeb")

    include_unavailability: bool = True
    unavailability_feature_mode: Literal["total", "plant_type"] = "total"
    unavailability_planned_only: bool = False
    include_unavailability_in_target_features: bool = False
    include_dwd_cluster_features: bool = True
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
    wind_hub_height_m: float = 100.0
    wind_reference_height_m: float = 10.0
    wind_shear_alpha: float = 0.14
    include_nwp_lag_diff_features: bool = False
    nwp_lag_steps: list[int] = Field(default_factory=lambda: [4, 12, 96])
    nwp_lead_steps: list[int] = Field(default_factory=list)
    nwp_diff_steps: list[int] = Field(default_factory=lambda: [4, 12])
    include_proxy_lag_diff_features: bool = False
    proxy_lag_diff_feature_patterns: list[str] = Field(default_factory=list)
    include_proxy_window_context_features: bool = False
    proxy_window_context_feature_patterns: list[str] = Field(default_factory=list)
    proxy_window_context_steps: list[int] = Field(default_factory=lambda: [12, 24])
    proxy_window_context_stats: list[Literal["mean", "std", "min", "max"]] = Field(
        default_factory=lambda: ["mean", "std", "max"]
    )
    actual_generation_lag_days: list[int] = Field(default_factory=list)
    actual_generation_lag_columns: list[str] = Field(
        default_factory=lambda: [
            "Solar_Actual_MW",
            "Wind_Total_Actual_MW",
            "Renewable_Total_Actual_MW",
        ]
    )
    include_partial_actual_generation_features: bool = False
    partial_generation_columns: list[str] = Field(default_factory=list)
    partial_generation_reference_day: int = 1
    partial_generation_comparison_lag_days: int = 7
    partial_generation_morning_end_hour: int = 10
    partial_generation_morning_end_minute: int = 0
    include_partial_proxy_error_features: bool = False
    partial_proxy_error_columns: list[str] = Field(default_factory=list)
    partial_proxy_error_proxy_column_patterns: dict[str, list[str]] = Field(default_factory=dict)
    partial_proxy_error_proxy_column_suffix: str = "_proxy_mw"
    partial_proxy_error_reference_day: int = 1
    partial_proxy_error_morning_end_hour: int = 10
    partial_proxy_error_morning_end_minute: int = 0
    partial_proxy_error_min_proxy_mw: float = Field(default=1.0, ge=0.0)
    include_forecast_lead_features: bool = False
    forecast_run_hour_utc: str = "06:00"
    forecast_run_day_offset: int = 1
    include_solar_geometry_features: bool = False
    include_solar_physics_features: bool = False
    include_nwp_disagreement_features: bool = False
    include_regional_summary_features: bool = False
    include_wind_direction_regime_features: bool = False
    include_wind_high_regime_features: bool = False
    include_wind_ensemble_regime_features: bool = False
    include_wind_cutout_risk_features: bool = False
    wind_high_regime_speed_thresholds_m_s: list[float] = Field(default_factory=lambda: [15.0, 20.0, 25.0])
    wind_high_regime_proxy_thresholds_mw: list[float] = Field(default_factory=lambda: [40000.0, 55000.0, 65000.0])
    wind_ensemble_regime_speed_thresholds_m_s: list[float] = Field(default_factory=lambda: [8.0, 12.0, 15.0, 20.0])
    wind_cutout_risk_speed_thresholds_m_s: list[float] = Field(
        default_factory=lambda: [10.0, 12.0, 15.0, 18.0, 20.0, 25.0]
    )
    wind_cutout_risk_std_floor_m_s: float = Field(default=0.35, gt=0.0)
    wind_proxy_residual_baseline: bool = False
    wind_proxy_baseline_columns: list[str] = Field(default_factory=list)
    wind_proxy_baseline_column_patterns: list[str] = Field(default_factory=lambda: ["proxy_mw"])
    wind_proxy_baseline_ridge_alpha: float = Field(default=100.0, ge=0.0)
    wind_proxy_baseline_clip: bool = True
    target_feature_mode: Literal["all", "technology_specific"] = "all"
    solar_target_region_scope: Literal["all_regions", "own_region_plus_global"] = "all_regions"
    target_transform: Literal["mw", "capacity_factor"] = "mw"
    target_baseline_mode: Literal["none", "solar_physics_proxy", "solar_model_chain_proxy"] = "none"
    target_baseline_columns: dict[str, str] = Field(default_factory=dict)
    solar_physics_proxy_temperature_coefficient: float = -0.004
    solar_physics_proxy_module_temperature_irradiance_coeff: float = 0.025
    solar_physics_proxy_min_temperature_factor: float = Field(default=0.75, ge=0.0)
    solar_physics_proxy_max_temperature_factor: float = Field(default=1.15, ge=0.0)
    solar_model_chain_performance_ratio: float = Field(default=0.86, ge=0.0)
    solar_model_chain_base_proxy_performance_ratio: float = Field(default=0.85, gt=0.0)
    solar_model_chain_albedo: float = Field(default=0.2, ge=0.0)
    solar_model_chain_tilt_degrees: list[float] = Field(default_factory=lambda: [30.0, 20.0, 20.0])
    solar_model_chain_azimuth_degrees: list[float] = Field(default_factory=lambda: [180.0, 90.0, 270.0])
    solar_model_chain_orientation_weights: list[float] = Field(default_factory=lambda: [0.65, 0.175, 0.175])
    rolling_bias_correction_window_days: int = 0
    rolling_bias_correction_group: Literal["global", "hour", "mtu"] = "hour"
    rolling_bias_correction_min_observations: int = 24
    rolling_bias_correction_shrinkage: float = Field(default=1.0, ge=0.0, le=1.0)
    solar_total_bias_correction_window_days: int = 0
    solar_total_bias_correction_group: Literal["global", "hour", "mtu"] = "hour"
    solar_total_bias_correction_min_observations: int = 24
    solar_total_bias_correction_shrinkage: float = Field(default=1.0, ge=0.0, le=1.0)
    enable_capacity_nowcasting: bool = False
    capacity_nowcasting_window_days: int = 45
    capacity_nowcasting_min_clear_sky: float = Field(default=0.5, ge=0.0)
    capacity_nowcasting_min_observations: int = 50
    capacity_nowcasting_shrinkage: float = Field(default=1.0, ge=0.0, le=1.0)
    capacity_nowcasting_min_factor: float = Field(default=0.8, gt=0.0)
    capacity_nowcasting_max_factor: float = Field(default=1.5, gt=0.0)
    capacity_nowcasting_targets: list[str] = Field(default_factory=list)
    max_features_per_target: int | None = None
    feature_allowlist_file: Path | None = None
    target_feature_allowlist_files: dict[str, Path] = Field(default_factory=dict)
    model_type: Literal["hist_gradient_boosting", "lightgbm", "ridge", "baseline"] = "hist_gradient_boosting"
    use_pca: bool = False
    pca_n_components: int | float = 0.99
    pca_whiten: bool = False
    target_columns: list[str] = Field(default_factory=lambda: ["Solar_Actual_MW", "Wind_Total_Actual_MW"])
    train_days_rolling: int = 112
    training_recency_half_life_days: float | None = Field(default=None, gt=0.0)
    min_train_days: int = 14
    target_availability_lag_days: int = 0
    target_availability_cutoff_hour: int | None = None
    target_availability_cutoff_minute: int = 0
    test_start: date = date(2025, 12, 1)
    test_end: date = date(2026, 2, 28)

    hgb_max_iter: int = 300
    hgb_learning_rate: float = 0.04
    hgb_max_leaf_nodes: int = 31
    hgb_max_depth: int | None = None
    hgb_min_samples_leaf: int = 20
    hgb_l2_regularization: float = 0.1
    hgb_max_features: float = Field(default=1.0, gt=0.0, le=1.0)
    hgb_max_bins: int = Field(default=255, ge=2, le=255)
    lgbm_n_estimators: int = Field(default=800, ge=1)
    lgbm_learning_rate: float = Field(default=0.02, gt=0.0)
    lgbm_num_leaves: int = Field(default=31, ge=2)
    lgbm_max_depth: int = -1
    lgbm_min_child_samples: int = Field(default=60, ge=1)
    lgbm_subsample: float = Field(default=0.9, gt=0.0, le=1.0)
    lgbm_colsample_bytree: float = Field(default=0.8, gt=0.0, le=1.0)
    lgbm_reg_alpha: float = Field(default=0.0, ge=0.0)
    lgbm_reg_lambda: float = Field(default=1.0, ge=0.0)
    ridge_alpha: float = 100.0
    target_model_overrides: dict[str, dict[str, Any]] = Field(default_factory=dict)
    random_state: int = 42
    clip_predictions_to_training_target_range: bool = False
    prediction_upper_quantile: float = Field(default=1.0, ge=0.0, le=1.0)
    solar_twilight_clear_sky_threshold: float | None = None
    solar_training_clear_sky_min: float | None = None
    solar_training_elevation_min_deg: float | None = None
    solar_suspicious_training_filter: bool = False
    solar_suspicious_clear_sky_min: float = Field(default=0.2, ge=0.0)
    solar_suspicious_actual_to_baseline_min: float = Field(default=0.35, ge=0.0)
    solar_suspicious_baseline_min_capacity_share: float = Field(default=0.05, ge=0.0)
    solar_weather_regime_split: bool = False
    solar_weather_regime_min_train_rows: int = Field(default=672, ge=1)
    solar_weather_regime_clear_cloud_max: float = Field(default=35.0, ge=0.0, le=100.0)
    solar_weather_regime_overcast_cloud_min: float = Field(default=75.0, ge=0.0, le=100.0)
    solar_weather_regime_clear_irradiance_ratio_min: float = Field(default=0.70, ge=0.0)
    solar_weather_regime_overcast_irradiance_ratio_max: float = Field(default=0.35, ge=0.0)
    wind_suspicious_training_filter: bool = False
    wind_suspicious_proxy_columns: list[str] = Field(default_factory=list)
    wind_suspicious_proxy_column_patterns: list[str] = Field(default_factory=lambda: ["renewable_wind_proxy_mw"])
    wind_suspicious_potential_quantile: float = Field(default=0.75, ge=0.0, le=1.0)
    wind_suspicious_min_potential_mw: float = Field(default=5000.0, ge=0.0)
    wind_suspicious_actual_to_proxy_min: float = Field(default=0.35, ge=0.0)
    wind_suspicious_proxy_aggregation: Literal["mean", "max"] = "mean"
    target_capacity_caps_mw: dict[str, float] = Field(default_factory=dict)
    target_installed_capacity_mw: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _resolve_paths(self) -> "RenewableGenerationModelConfig":
        if self.train_days_rolling < 1:
            raise ValueError("train_days_rolling must be positive.")
        if self.min_train_days < 1:
            raise ValueError("min_train_days must be positive.")
        if self.target_availability_lag_days < 0:
            raise ValueError("target_availability_lag_days must be non-negative.")
        if self.target_availability_cutoff_hour is not None and not (0 <= self.target_availability_cutoff_hour <= 23):
            raise ValueError("target_availability_cutoff_hour must be between 0 and 23.")
        if self.target_availability_cutoff_minute not in {0, 15, 30, 45}:
            raise ValueError("target_availability_cutoff_minute must be one of 0, 15, 30, or 45.")
        if self.partial_generation_reference_day < 1:
            raise ValueError("partial_generation_reference_day must be positive.")
        if self.partial_generation_comparison_lag_days < 1:
            raise ValueError("partial_generation_comparison_lag_days must be positive.")
        if not (0 <= self.partial_generation_morning_end_hour <= 23):
            raise ValueError("partial_generation_morning_end_hour must be between 0 and 23.")
        if self.partial_generation_morning_end_minute not in {0, 15, 30, 45}:
            raise ValueError("partial_generation_morning_end_minute must be one of 0, 15, 30, or 45.")
        if self.partial_proxy_error_reference_day < 1:
            raise ValueError("partial_proxy_error_reference_day must be positive.")
        if not (0 <= self.partial_proxy_error_morning_end_hour <= 23):
            raise ValueError("partial_proxy_error_morning_end_hour must be between 0 and 23.")
        if self.partial_proxy_error_morning_end_minute not in {0, 15, 30, 45}:
            raise ValueError("partial_proxy_error_morning_end_minute must be one of 0, 15, 30, or 45.")
        if self.solar_total_bias_correction_window_days < 0:
            raise ValueError("solar_total_bias_correction_window_days must be non-negative.")
        if self.enable_capacity_nowcasting:
            if self.capacity_nowcasting_window_days < 1:
                raise ValueError("capacity_nowcasting_window_days must be positive.")
            if self.capacity_nowcasting_min_factor > self.capacity_nowcasting_max_factor:
                raise ValueError(
                    "capacity_nowcasting_min_factor must be <= capacity_nowcasting_max_factor."
                )
        if (
            self.solar_physics_proxy_min_temperature_factor
            > self.solar_physics_proxy_max_temperature_factor
        ):
            raise ValueError(
                "solar_physics_proxy_min_temperature_factor must be <= "
                "solar_physics_proxy_max_temperature_factor."
            )
        if not (
            len(self.solar_model_chain_tilt_degrees)
            == len(self.solar_model_chain_azimuth_degrees)
            == len(self.solar_model_chain_orientation_weights)
        ):
            raise ValueError(
                "solar_model_chain_tilt_degrees, solar_model_chain_azimuth_degrees, "
                "and solar_model_chain_orientation_weights must have the same length."
            )
        if any(weight < 0.0 for weight in self.solar_model_chain_orientation_weights):
            raise ValueError("solar_model_chain_orientation_weights must be non-negative.")
        if sum(self.solar_model_chain_orientation_weights) <= 0.0:
            raise ValueError("solar_model_chain_orientation_weights must sum to a positive value.")
        if self.target_baseline_mode != "none" and self.target_transform != "mw":
            raise ValueError("target_baseline_mode currently requires target_transform='mw'.")
        if self.solar_weather_regime_clear_cloud_max > self.solar_weather_regime_overcast_cloud_min:
            raise ValueError(
                "solar_weather_regime_clear_cloud_max must be <= "
                "solar_weather_regime_overcast_cloud_min."
            )
        if self.solar_weather_regime_overcast_irradiance_ratio_max > self.solar_weather_regime_clear_irradiance_ratio_min:
            raise ValueError(
                "solar_weather_regime_overcast_irradiance_ratio_max must be <= "
                "solar_weather_regime_clear_irradiance_ratio_min."
            )
        self.actual_generation_file = resolve_path(self.actual_generation_file, self.repo_root)
        self.renewable_proxy_file = resolve_path(self.renewable_proxy_file, self.repo_root)
        if self.renewable_proxy_fallback_file is not None:
            self.renewable_proxy_fallback_file = resolve_path(self.renewable_proxy_fallback_file, self.repo_root)
        self.extra_renewable_proxy_files = [
            resolve_path(path, self.repo_root)
            for path in self.extra_renewable_proxy_files
        ]
        if self.extra_renewable_proxy_prefixes and len(self.extra_renewable_proxy_prefixes) != len(
            self.extra_renewable_proxy_files
        ):
            raise ValueError("extra_renewable_proxy_prefixes must be empty or match extra_renewable_proxy_files length.")
        self.raw_extra_renewable_proxy_files = [
            resolve_path(path, self.repo_root)
            for path in self.raw_extra_renewable_proxy_files
        ]
        if self.raw_extra_renewable_proxy_prefixes and len(self.raw_extra_renewable_proxy_prefixes) != len(
            self.raw_extra_renewable_proxy_files
        ):
            raise ValueError(
                "raw_extra_renewable_proxy_prefixes must be empty or match raw_extra_renewable_proxy_files length."
            )
        for group_name, indices in self.extra_renewable_proxy_ensemble_groups.items():
            if not indices:
                raise ValueError(f"extra_renewable_proxy_ensemble_groups[{group_name!r}] must not be empty.")
            for index in indices:
                if index < 0 or index >= len(self.extra_renewable_proxy_files):
                    raise ValueError(
                        f"extra_renewable_proxy_ensemble_groups[{group_name!r}] contains index {index}, "
                        "which is outside extra_renewable_proxy_files."
                    )
        for pair in self.extra_renewable_proxy_ensemble_group_diff_pairs:
            if len(pair) != 2:
                raise ValueError("extra_renewable_proxy_ensemble_group_diff_pairs entries must contain two group names.")
            missing = [name for name in pair if name not in self.extra_renewable_proxy_ensemble_groups]
            if missing:
                raise ValueError(
                    "extra_renewable_proxy_ensemble_group_diff_pairs references unknown group(s): "
                    f"{missing}"
                )
        self.unavailability_file = resolve_path(self.unavailability_file, self.repo_root)
        self.icon_dir = resolve_path(self.icon_dir, self.repo_root)
        self.export_dir = resolve_path(self.export_dir, self.repo_root)
        return self


class RenewableGenerationPostprocessConfig(RepoConfigModel):
    """Diagnose and leakage-safely post-process renewable generation forecasts."""

    target_tz: str = "Europe/Berlin"
    forecast_file: Path = Path("results/renewable_generation_results/run/forecast.csv")
    export_dir: Path = Path("results/renewable_generation_postprocess/run")

    include_forecast_stacking: bool = False
    stacking_forecast_files: dict[str, Path] = Field(default_factory=dict)
    stacking_targets: list[str] = Field(default_factory=lambda: ["Wind_Onshore", "Wind_Offshore"])
    stacking_window_days: int = 90
    stacking_min_observations: int = 336
    stacking_model_type: Literal["ridge", "simple_average"] = "ridge"
    stacking_ridge_alpha: float = Field(default=100.0, ge=0.0)

    correction_window_days: int = 0
    correction_group: Literal[
        "none",
        "global",
        "hour",
        "mtu",
        "daytype_hour",
        "daytype_mtu",
        "prediction_bin",
        "hour_prediction_bin",
        "mtu_prediction_bin",
        "daytype_prediction_bin",
    ] = "none"
    correction_min_observations: int = 24
    correction_shrinkage: float = Field(default=1.0, ge=0.0, le=1.0)
    correction_prediction_bins: int = 4
    correction_targets: list[str] = Field(default_factory=list)
    # Operational target-availability cutoff for the rolling correction history. With the
    # defaults (lag 0, no cutoff hour) the history ends at the previous day, matching the
    # legacy behaviour. Set lag/cutoff to match the model config (e.g. lag 1, hour 10) so
    # the correction only uses actuals known before the 12:00 gate.
    target_availability_lag_days: int = 0
    target_availability_cutoff_hour: int | None = None
    target_availability_cutoff_minute: int = 0
    target_upper_bounds_mw: dict[str, float] = Field(default_factory=dict)
    installed_capacity_mw: dict[str, float] = Field(default_factory=dict)
    diagnostics_bins: int = 5
    include_point_calibrator: bool = False
    point_calibrator_targets: list[str] = Field(default_factory=list)
    point_calibrator_window_days: int = 45
    point_calibrator_min_observations: int = 96
    point_calibrator_shrinkage: float = Field(default=1.0, ge=0.0, le=1.0)
    point_calibrator_model_type: Literal["hist_gradient_boosting", "ridge"] = "hist_gradient_boosting"
    point_calibrator_hgb_max_iter: int = 200
    point_calibrator_hgb_learning_rate: float = 0.05
    point_calibrator_hgb_max_leaf_nodes: int = 7
    point_calibrator_hgb_l2_regularization: float = 1.0
    point_calibrator_random_state: int = 42
    include_residual_quantiles: bool = False
    residual_quantile_targets: list[str] = Field(default_factory=list)
    residual_quantiles: list[float] = Field(default_factory=lambda: [0.1, 0.25, 0.5, 0.75, 0.9])
    residual_quantile_window_days: int = 30
    residual_quantile_group: Literal[
        "global",
        "hour",
        "mtu",
        "daytype_hour",
        "daytype_mtu",
        "prediction_bin",
        "hour_prediction_bin",
        "mtu_prediction_bin",
        "daytype_prediction_bin",
    ] = "hour_prediction_bin"
    residual_quantile_min_observations: int = 8
    residual_quantile_prediction_bins: int = 4
    residual_quantile_spread_scale: float = Field(default=1.0, ge=0.0)
    include_sequence_residual_correction: bool = False
    sequence_residual_targets: list[str] = Field(default_factory=lambda: ["Wind_Onshore", "Wind_Offshore"])
    sequence_residual_feature_targets: list[str] = Field(
        default_factory=lambda: ["Wind_Onshore", "Wind_Offshore", "Wind_Total"]
    )
    sequence_residual_window_days: int = 60
    sequence_residual_min_train_days: int = 21
    sequence_residual_n_steps: int = 96
    sequence_residual_ridge_alpha: float = Field(default=1000.0, ge=0.0)
    sequence_residual_shrinkage: float = Field(default=0.5, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _resolve_paths(self) -> "RenewableGenerationPostprocessConfig":
        if self.stacking_window_days < 0:
            raise ValueError("stacking_window_days must be non-negative.")
        if self.stacking_min_observations < 1:
            raise ValueError("stacking_min_observations must be positive.")
        if self.target_availability_lag_days < 0:
            raise ValueError("target_availability_lag_days must be non-negative.")
        if self.target_availability_cutoff_hour is not None and not (
            0 <= self.target_availability_cutoff_hour <= 23
        ):
            raise ValueError("target_availability_cutoff_hour must be between 0 and 23.")
        if self.target_availability_cutoff_minute not in {0, 15, 30, 45}:
            raise ValueError("target_availability_cutoff_minute must be one of 0, 15, 30, or 45.")
        if self.point_calibrator_window_days < 0:
            raise ValueError("point_calibrator_window_days must be non-negative.")
        if self.point_calibrator_min_observations < 1:
            raise ValueError("point_calibrator_min_observations must be positive.")
        if self.point_calibrator_hgb_max_iter < 1:
            raise ValueError("point_calibrator_hgb_max_iter must be positive.")
        if self.residual_quantile_window_days < 0:
            raise ValueError("residual_quantile_window_days must be non-negative.")
        if self.residual_quantile_min_observations < 1:
            raise ValueError("residual_quantile_min_observations must be positive.")
        if self.residual_quantile_prediction_bins < 1:
            raise ValueError("residual_quantile_prediction_bins must be positive.")
        if self.sequence_residual_window_days < 0:
            raise ValueError("sequence_residual_window_days must be non-negative.")
        if self.sequence_residual_min_train_days < 1:
            raise ValueError("sequence_residual_min_train_days must be positive.")
        if self.sequence_residual_n_steps < 1:
            raise ValueError("sequence_residual_n_steps must be positive.")
        self.residual_quantiles = sorted({float(quantile) for quantile in self.residual_quantiles})
        if any(quantile <= 0.0 or quantile >= 1.0 for quantile in self.residual_quantiles):
            raise ValueError("residual_quantiles must be between 0 and 1.")
        self.forecast_file = resolve_path(self.forecast_file, self.repo_root)
        self.stacking_forecast_files = {
            str(name): resolve_path(path, self.repo_root)
            for name, path in self.stacking_forecast_files.items()
        }
        self.export_dir = resolve_path(self.export_dir, self.repo_root)
        return self


class EntsoeRenewableForecastBenchmarkConfig(RepoConfigModel):
    """Benchmark ENTSO-E renewable generation forecasts against actual generation."""

    target_tz: str = "Europe/Berlin"
    country_code_entsoe: str = "DE_LU"
    entsoe_api_key_env: str = "ENTSOE_API_KEY"
    entsoe_start_date: date = date(2025, 12, 1)
    entsoe_end_date: date = date(2026, 2, 8)
    process_type: str = "A01"
    chunk_days: int = 30
    actual_generation_file: Path = Path("data/processed/renewable_generation/actual_generation_2025_2026.csv")
    forecast_file: Path = Path("data/processed/renewable_generation/entsoe_renewable_forecast.csv")
    export_dir: Path = Path("results/renewable_generation_results/entsoe_renewable_forecast_benchmark")
    installed_capacity_mw: dict[str, float] = Field(
        default_factory=lambda: {
            "Solar": 66119.530485,
            "Wind_Total": 79629.724430,
            "Renewable_Total": 145749.254915,
        }
    )

    @model_validator(mode="after")
    def _resolve_paths(self) -> "EntsoeRenewableForecastBenchmarkConfig":
        self.actual_generation_file = resolve_path(self.actual_generation_file, self.repo_root)
        self.forecast_file = resolve_path(self.forecast_file, self.repo_root)
        self.export_dir = resolve_path(self.export_dir, self.repo_root)
        return self
