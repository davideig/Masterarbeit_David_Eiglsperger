from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.spatial import cKDTree
from sklearn.cluster import MiniBatchKMeans

from ..config import RegionalRenewableFeatureConfig
from ..data.weather import load_dwd, load_era5, load_open_meteo, load_open_meteo_points


def _normalise_label(value: object) -> str:
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


def _load_clusters(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        clusters = pd.read_parquet(path)
    else:
        clusters = pd.read_csv(path)

    required = {"lon", "lat", "cluster_id"}
    missing = required - set(clusters.columns)
    if missing:
        raise ValueError(f"Cluster file is missing required columns: {sorted(missing)}")

    columns = ["lon", "lat", "cluster_id"]
    if "cluster_region" in clusters.columns:
        columns.append("cluster_region")
    clusters = clusters[columns].dropna(subset=["lon", "lat", "cluster_id"]).copy()
    clusters["cluster_id"] = clusters["cluster_id"].astype(int)
    if "cluster_region" in clusters.columns:
        clusters["cluster_region"] = clusters["cluster_region"].astype(str)
    return clusters


def _technology_group(raw_technology: object, config: RegionalRenewableFeatureConfig) -> str | None:
    technology = _normalise_label(raw_technology)
    if technology in {_normalise_label(value) for value in config.wind_offshore_values}:
        return "wind_offshore"
    if technology in {_normalise_label(value) for value in config.wind_onshore_values}:
        return "wind_onshore"
    if technology in {_normalise_label(value) for value in config.solar_values}:
        return "solar"
    return None


_SOLAR_STATE_TSO_PROXY_REGIONS = {
    # MaStR federal state codes. This is a state-level proxy for the four
    # German TSO solar reporting areas, not an exact control-area polygon.
    "1400": "solar_50hertz",  # Brandenburg
    "1401": "solar_50hertz",  # Berlin
    "1406": "solar_50hertz",  # Hamburg
    "1407": "solar_50hertz",  # Mecklenburg-Vorpommern
    "1413": "solar_50hertz",  # Sachsen
    "1414": "solar_50hertz",  # Sachsen-Anhalt
    "1415": "solar_50hertz",  # Thueringen
    "1409": "solar_amprion",  # Nordrhein-Westfalen
    "1410": "solar_amprion",  # Rheinland-Pfalz
    "1412": "solar_amprion",  # Saarland
    "1403": "solar_tennet",  # Bayern
    "1404": "solar_tennet",  # Bremen
    "1405": "solar_tennet",  # Hessen
    "1408": "solar_tennet",  # Niedersachsen
    "1411": "solar_tennet",  # Schleswig-Holstein
    "1402": "solar_transnetbw",  # Baden-Wuerttemberg
}


def _solar_tso_proxy_region_for_row(row: pd.Series, config: RegionalRenewableFeatureConfig) -> str | None:
    if config.federal_state_column not in row.index:
        return None
    value = row[config.federal_state_column]
    if pd.isna(value):
        return None
    try:
        state_code = str(int(float(value)))
    except (TypeError, ValueError):
        state_code = str(value).strip()
    return _SOLAR_STATE_TSO_PROXY_REGIONS.get(state_code)


def _region_for_row(row: pd.Series, config: RegionalRenewableFeatureConfig) -> str:
    technology = row["technology_group"]
    lat = float(row[config.latitude_column])
    lon = float(row[config.longitude_column])

    if technology == "wind_offshore":
        return "offshore_north"

    if technology == "solar" and config.solar_region_strategy == "state_tso_proxy":
        tso_region = _solar_tso_proxy_region_for_row(row, config)
        if tso_region is not None:
            return tso_region

    if technology == "wind_onshore":
        prefix = "onshore"
    else:
        prefix = "solar"

    if lat >= config.north_latitude:
        return f"{prefix}_north"
    if lat < config.south_latitude:
        return f"{prefix}_south"
    if lon < config.west_longitude:
        return f"{prefix}_west"
    if lon >= config.east_longitude:
        return f"{prefix}_east"
    return f"{prefix}_central"


def build_capacity_map(config: RegionalRenewableFeatureConfig) -> pd.DataFrame:
    """Assign active MaStR renewable units to weather clusters and coarse regions."""
    capacity = pd.read_csv(config.capacity_file)
    required = {
        config.technology_column,
        config.capacity_column,
        config.latitude_column,
        config.longitude_column,
        config.commissioning_date_column,
    }
    missing = required - set(capacity.columns)
    if missing:
        raise ValueError(f"Capacity file is missing required columns: {sorted(missing)}")

    capacity = capacity.copy()
    capacity["technology_group"] = capacity[config.technology_column].map(lambda value: _technology_group(value, config))
    capacity["capacity_mw"] = pd.to_numeric(capacity[config.capacity_column], errors="coerce")
    capacity["commissioning_date"] = pd.to_datetime(
        capacity[config.commissioning_date_column],
        errors="coerce",
    ).dt.date
    if config.decommissioning_date_column in capacity.columns:
        capacity["decommissioning_date"] = pd.to_datetime(
            capacity[config.decommissioning_date_column],
            errors="coerce",
        ).dt.date
    else:
        capacity["decommissioning_date"] = pd.NaT

    capacity = capacity.dropna(
        subset=[
            "technology_group",
            "capacity_mw",
            config.latitude_column,
            config.longitude_column,
            "commissioning_date",
        ]
    )
    capacity = capacity.loc[capacity["capacity_mw"] > 0].copy()

    if config.operating_status_column in capacity.columns and config.active_status_codes:
        active_codes = {str(code) for code in config.active_status_codes}
        capacity = capacity.loc[capacity[config.operating_status_column].astype(str).isin(active_codes)].copy()

    capacity["region"] = capacity.apply(lambda row: _region_for_row(row, config), axis=1)
    clusters = _load_clusters(config.cluster_file)
    capacity["cluster_id"] = -1

    if "cluster_region" in clusters.columns:
        for region, region_capacity in capacity.groupby("region", sort=False):
            region_clusters = clusters.loc[clusters["cluster_region"] == region].copy()
            if region_clusters.empty:
                continue
            tree = cKDTree(region_clusters[["lon", "lat"]].to_numpy(dtype=float))
            points = region_capacity[[config.longitude_column, config.latitude_column]].to_numpy(dtype=float)
            _, nearest_idx = tree.query(points)
            capacity.loc[region_capacity.index, "cluster_id"] = (
                region_clusters.iloc[nearest_idx]["cluster_id"].to_numpy(dtype=int)
            )

    missing_cluster = capacity["cluster_id"] < 0
    if missing_cluster.any():
        tree = cKDTree(clusters[["lon", "lat"]].to_numpy(dtype=float))
        points = capacity.loc[missing_cluster, [config.longitude_column, config.latitude_column]].to_numpy(dtype=float)
        _, nearest_idx = tree.query(points)
        capacity.loc[missing_cluster, "cluster_id"] = clusters.iloc[nearest_idx]["cluster_id"].to_numpy(dtype=int)

    columns = [
        "technology_group",
        "region",
        "cluster_id",
        "capacity_mw",
        "commissioning_date",
        "decommissioning_date",
        config.latitude_column,
        config.longitude_column,
    ]
    if config.federal_state_column in capacity.columns:
        columns.append(config.federal_state_column)
    optional_columns = [
        config.wind_hub_height_column,
        config.wind_rotor_diameter_column,
        config.wind_swept_area_column,
        config.wind_manufacturer_column,
        config.wind_turbine_type_column,
        "specific_power_mw_per_m2",
        "wind_location_code",
        "wind_park_name",
        "rotor_blade_deicing",
        "curtailment_limitation_flag",
        "remote_control_grid_operator",
        "remote_control_direct_marketer",
    ]
    columns.extend(column for column in optional_columns if column in capacity.columns and column not in columns)
    return capacity[columns].sort_values(["technology_group", "region", "cluster_id"]).reset_index(drop=True)


def _capacity_weather_point_count(
    technology: str,
    region: str,
    config: RegionalRenewableFeatureConfig,
) -> int:
    if technology == "solar":
        return config.capacity_weather_solar_points_per_region
    if technology == "wind_onshore":
        return config.capacity_weather_onshore_points_per_region
    if technology == "wind_offshore":
        return config.capacity_weather_offshore_points
    raise ValueError(f"Unsupported technology group for capacity weather points: {technology!r}")


def _representative_capacity_points(
    subset: pd.DataFrame,
    *,
    n_points: int,
    latitude_column: str,
    longitude_column: str,
    random_state: int,
    metadata_columns: list[str] | None = None,
) -> pd.DataFrame:
    metadata_columns = [column for column in metadata_columns or [] if column in subset.columns]
    aggregations = {"capacity_mw": ("capacity_mw", "sum"), "source_unit_count": ("capacity_mw", "size")}
    for column in metadata_columns:
        aggregations[column] = (column, "mean")
    grouped = (
        subset
        .groupby([latitude_column, longitude_column], as_index=False)
        .agg(**aggregations)
    )
    grouped = grouped.loc[grouped["capacity_mw"] > 0].copy()
    if grouped.empty:
        return grouped

    n_points = min(int(n_points), len(grouped))
    if len(grouped) <= n_points:
        return grouped.rename(columns={latitude_column: "lat", longitude_column: "lon"})

    if metadata_columns:
        feature_frame = grouped[[longitude_column, latitude_column, *metadata_columns]].copy()
        for column in metadata_columns:
            feature_frame[column] = pd.to_numeric(feature_frame[column], errors="coerce")
            if feature_frame[column].notna().any():
                feature_frame[column] = feature_frame[column].fillna(feature_frame[column].median())
            else:
                feature_frame[column] = 0.0
        feature_values = feature_frame.to_numpy(dtype=float)
        feature_std = np.nanstd(feature_values, axis=0)
        feature_std = np.where(feature_std > 0, feature_std, 1.0)
        coords = (feature_values - np.nanmean(feature_values, axis=0)) / feature_std
    else:
        coords = grouped[[longitude_column, latitude_column]].to_numpy(dtype=float)
    lon_lat = grouped[[longitude_column, latitude_column]].to_numpy(dtype=float)
    weights = grouped["capacity_mw"].to_numpy(dtype=float)
    model = MiniBatchKMeans(
        n_clusters=n_points,
        random_state=random_state,
        n_init=5,
        batch_size=max(1024, n_points * 20),
        reassignment_ratio=0.0,
    )
    model.fit(coords, sample_weight=weights)
    labels = model.predict(coords)

    rows = []
    for label in range(n_points):
        mask = labels == label
        if not mask.any():
            continue
        label_weights = weights[mask]
        label_coords = lon_lat[mask]
        row = {
            "lat": float(np.average(label_coords[:, 1], weights=label_weights)),
            "lon": float(np.average(label_coords[:, 0], weights=label_weights)),
            "capacity_mw": float(label_weights.sum()),
            "source_unit_count": int(grouped.loc[mask, "source_unit_count"].sum()),
        }
        for column in metadata_columns:
            values = pd.to_numeric(grouped.loc[mask, column], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(values)
            if valid.any():
                row[column] = float(np.average(values[valid], weights=label_weights[valid]))
        rows.append(
            row
        )

    return pd.DataFrame(rows).sort_values(["lon", "lat"]).reset_index(drop=True)


def _representative_capacity_cells(
    subset: pd.DataFrame,
    *,
    n_points: int,
    latitude_column: str,
    longitude_column: str,
    cell_size_degrees: float,
    min_cell_capacity_mw: float,
) -> pd.DataFrame:
    grouped_units = (
        subset
        .groupby([latitude_column, longitude_column], as_index=False)
        .agg(capacity_mw=("capacity_mw", "sum"), source_unit_count=("capacity_mw", "size"))
    )
    grouped_units = grouped_units.loc[grouped_units["capacity_mw"] > 0].copy()
    if grouped_units.empty:
        return grouped_units.rename(columns={latitude_column: "lat", longitude_column: "lon"})

    grouped_units["_cell_lat"] = np.round(grouped_units[latitude_column] / cell_size_degrees) * cell_size_degrees
    grouped_units["_cell_lon"] = np.round(grouped_units[longitude_column] / cell_size_degrees) * cell_size_degrees

    rows = []
    for _, cell in grouped_units.groupby(["_cell_lat", "_cell_lon"], sort=True):
        weights = cell["capacity_mw"].to_numpy(dtype=float)
        rows.append(
            {
                "lat": float(np.average(cell[latitude_column].to_numpy(dtype=float), weights=weights)),
                "lon": float(np.average(cell[longitude_column].to_numpy(dtype=float), weights=weights)),
                "capacity_mw": float(weights.sum()),
                "source_unit_count": int(cell["source_unit_count"].sum()),
            }
        )

    cells = pd.DataFrame(rows)
    if cells.empty:
        return cells

    eligible = cells.loc[cells["capacity_mw"] >= min_cell_capacity_mw].copy()
    if eligible.empty:
        eligible = cells

    n_points = min(int(n_points), len(eligible))
    return (
        eligible
        .sort_values(["capacity_mw", "source_unit_count", "lon", "lat"], ascending=[False, False, True, True])
        .head(n_points)
        .sort_values(["lon", "lat"])
        .reset_index(drop=True)
    )


def build_capacity_weather_points(
    capacity_map: pd.DataFrame,
    config: RegionalRenewableFeatureConfig,
) -> pd.DataFrame:
    """Build technology/region-specific representative weather points weighted by installed capacity."""
    rows = []
    point_id = 0
    technologies = [technology for technology in config.capacity_weather_technologies if technology]
    if not technologies:
        raise ValueError("capacity_weather_technologies must contain at least one technology.")

    for technology in technologies:
        technology_map = capacity_map.loc[capacity_map["technology_group"] == technology]
        for region in sorted(technology_map["region"].unique()):
            subset = technology_map.loc[technology_map["region"] == region].copy()
            if subset.empty:
                continue
            n_points = _capacity_weather_point_count(technology, region, config)
            if config.capacity_weather_point_strategy == "kmeans":
                representatives = _representative_capacity_points(
                    subset,
                    n_points=n_points,
                    latitude_column=config.latitude_column,
                    longitude_column=config.longitude_column,
                    random_state=config.capacity_weather_random_state + point_id,
                )
            elif config.capacity_weather_point_strategy == "kmeans_metadata":
                representatives = _representative_capacity_points(
                    subset,
                    n_points=n_points,
                    latitude_column=config.latitude_column,
                    longitude_column=config.longitude_column,
                    random_state=config.capacity_weather_random_state + point_id,
                    metadata_columns=config.capacity_weather_metadata_columns,
                )
            elif config.capacity_weather_point_strategy == "capacity_cells":
                representatives = _representative_capacity_cells(
                    subset,
                    n_points=n_points,
                    latitude_column=config.latitude_column,
                    longitude_column=config.longitude_column,
                    cell_size_degrees=config.capacity_weather_cell_size_degrees,
                    min_cell_capacity_mw=config.capacity_weather_min_cell_capacity_mw,
                )
            else:
                raise ValueError(f"Unsupported capacity weather point strategy: {config.capacity_weather_point_strategy!r}")
            for _, representative in representatives.iterrows():
                row = {
                    "weather_point_id": point_id,
                    "technology_group": technology,
                    "region": region,
                    "lat": float(representative["lat"]),
                    "lon": float(representative["lon"]),
                    "capacity_mw": float(representative["capacity_mw"]),
                    "source_unit_count": int(representative["source_unit_count"]),
                }
                for column in config.capacity_weather_metadata_columns:
                    if column in representative and pd.notna(representative[column]):
                        row[column] = float(representative[column])
                rows.append(row)
                point_id += 1

    if not rows:
        raise ValueError("No capacity-weighted Open-Meteo weather points could be built.")
    return pd.DataFrame(rows).sort_values("weather_point_id").reset_index(drop=True)


def _sharp_wind_power_curve(speed_m_s: np.ndarray) -> np.ndarray:
    cut_in = 3.0
    rated = 12.0
    cut_out = 25.0
    values = np.zeros_like(speed_m_s, dtype=float)
    ramp = (speed_m_s >= cut_in) & (speed_m_s < rated)
    rated_mask = (speed_m_s >= rated) & (speed_m_s <= cut_out)
    values[ramp] = ((speed_m_s[ramp] - cut_in) / (rated - cut_in)) ** 3
    values[rated_mask] = 1.0
    return np.clip(values, 0.0, 1.0)


def _normalised_wind_power_curve(
    speed_m_s: np.ndarray,
    smoothing_sigma_m_s: float = 0.0,
) -> np.ndarray:
    """Single-turbine power curve, optionally fleet-smeared.

    With ``smoothing_sigma_m_s > 0`` the sharp curve is convolved with a Gaussian
    of that standard deviation over wind speed, approximating the smoother
    aggregate response of a spatially dispersed fleet whose turbines reach
    rated/cut-out at different local speeds. ``0`` reproduces the sharp curve
    exactly. The expectation is evaluated with Gauss-Hermite quadrature.
    """
    if not smoothing_sigma_m_s or smoothing_sigma_m_s <= 0.0:
        return _sharp_wind_power_curve(speed_m_s)

    nodes, weights = np.polynomial.hermite_e.hermegauss(7)
    weights = weights / weights.sum()
    smeared = np.zeros_like(speed_m_s, dtype=float)
    for node, weight in zip(nodes, weights):
        shifted = np.clip(speed_m_s + smoothing_sigma_m_s * node, 0.0, None)
        smeared += weight * _sharp_wind_power_curve(shifted)
    return np.clip(smeared, 0.0, 1.0)


def _active_capacity_by_cluster(
    capacity_map: pd.DataFrame,
    timestamps: pd.DatetimeIndex,
    *,
    technology: str,
    region: str | None,
) -> pd.DataFrame:
    subset = capacity_map.loc[capacity_map["technology_group"] == technology].copy()
    if region is not None:
        subset = subset.loc[subset["region"] == region].copy()
    clusters = sorted(capacity_map["cluster_id"].unique())
    days = pd.DatetimeIndex(timestamps.normalize().unique()).sort_values()
    result = pd.DataFrame(0.0, index=days, columns=clusters)
    if subset.empty:
        return result

    first_day = days.min().tz_localize(None).normalize()
    last_day = days.max().tz_localize(None).normalize()
    subset["_commissioning_ts"] = pd.to_datetime(subset["commissioning_date"], errors="coerce").dt.normalize()
    subset["_decommissioning_ts"] = pd.to_datetime(subset["decommissioning_date"], errors="coerce").dt.normalize()
    commissioned = subset["_commissioning_ts"]
    decommissioned = subset["_decommissioning_ts"]

    initial_mask = (commissioned <= first_day) & (decommissioned.isna() | (decommissioned > first_day))
    initial = subset.loc[initial_mask].groupby("cluster_id")["capacity_mw"].sum()
    result.loc[:, initial.index] = initial.to_numpy(dtype=float)

    event_rows = []
    commissioning_mask = (commissioned > first_day) & (commissioned <= last_day)
    for _, row in subset.loc[commissioning_mask].iterrows():
        event_rows.append((
            pd.Timestamp(row["_commissioning_ts"].date(), tz=timestamps.tz),
            row["cluster_id"],
            row["capacity_mw"],
        ))

    decommissioning_mask = decommissioned.notna() & (decommissioned > first_day) & (decommissioned <= last_day)
    for _, row in subset.loc[decommissioning_mask].iterrows():
        event_rows.append((
            pd.Timestamp(row["_decommissioning_ts"].date(), tz=timestamps.tz),
            row["cluster_id"],
            -row["capacity_mw"],
        ))

    if event_rows:
        events = pd.DataFrame(event_rows, columns=["date", "cluster_id", "capacity_delta_mw"])
        event_pivot = events.pivot_table(
            index="date",
            columns="cluster_id",
            values="capacity_delta_mw",
            aggfunc="sum",
            fill_value=0.0,
        )
        event_pivot = event_pivot.reindex(days, fill_value=0.0)
        for column in event_pivot.columns:
            if column in result.columns:
                result[column] = result[column] + event_pivot[column].cumsum()

    return result.clip(lower=0.0)


def _assign_capacity_to_weather_points(
    capacity_map: pd.DataFrame,
    points: pd.DataFrame,
    config: RegionalRenewableFeatureConfig,
) -> pd.DataFrame:
    """Assign every unit to its nearest representative point within the same technology/region."""
    mapped = capacity_map.copy()
    mapped["weather_point_id"] = -1
    for (technology, region), subset in mapped.groupby(["technology_group", "region"], sort=True):
        group_points = points.loc[
            (points["technology_group"] == technology) & (points["region"] == region)
        ].copy()
        if group_points.empty:
            continue
        tree = cKDTree(group_points[["lon", "lat"]].to_numpy(dtype=float))
        coordinates = subset[[config.longitude_column, config.latitude_column]].to_numpy(dtype=float)
        _, nearest_idx = tree.query(coordinates)
        mapped.loc[subset.index, "weather_point_id"] = (
            group_points.iloc[nearest_idx]["weather_point_id"].to_numpy(dtype=int)
        )

    mapped = mapped.loc[mapped["weather_point_id"] >= 0].copy()
    mapped["weather_point_id"] = mapped["weather_point_id"].astype(int)
    return mapped


def _active_capacity_by_weather_point(
    capacity_map: pd.DataFrame,
    timestamps: pd.DatetimeIndex,
    *,
    technology: str,
    region: str,
) -> pd.DataFrame:
    subset = capacity_map.loc[
        (capacity_map["technology_group"] == technology) & (capacity_map["region"] == region)
    ].copy()
    point_ids = sorted(subset["weather_point_id"].dropna().astype(int).unique())
    days = pd.DatetimeIndex(timestamps.normalize().unique()).sort_values()
    result = pd.DataFrame(0.0, index=days, columns=point_ids)
    if subset.empty or not point_ids:
        return result

    first_day = days.min().tz_localize(None).normalize()
    last_day = days.max().tz_localize(None).normalize()
    subset["_commissioning_ts"] = pd.to_datetime(subset["commissioning_date"], errors="coerce").dt.normalize()
    subset["_decommissioning_ts"] = pd.to_datetime(subset["decommissioning_date"], errors="coerce").dt.normalize()
    commissioned = subset["_commissioning_ts"]
    decommissioned = subset["_decommissioning_ts"]

    initial_mask = (commissioned <= first_day) & (decommissioned.isna() | (decommissioned > first_day))
    initial = subset.loc[initial_mask].groupby("weather_point_id")["capacity_mw"].sum()
    result.loc[:, initial.index] = initial.to_numpy(dtype=float)

    event_rows = []
    commissioning_mask = (commissioned > first_day) & (commissioned <= last_day)
    for _, row in subset.loc[commissioning_mask].iterrows():
        event_rows.append((
            pd.Timestamp(row["_commissioning_ts"].date(), tz=timestamps.tz),
            row["weather_point_id"],
            row["capacity_mw"],
        ))

    decommissioning_mask = decommissioned.notna() & (decommissioned > first_day) & (decommissioned <= last_day)
    for _, row in subset.loc[decommissioning_mask].iterrows():
        event_rows.append((
            pd.Timestamp(row["_decommissioning_ts"].date(), tz=timestamps.tz),
            row["weather_point_id"],
            -row["capacity_mw"],
        ))

    if event_rows:
        events = pd.DataFrame(event_rows, columns=["date", "weather_point_id", "capacity_delta_mw"])
        event_pivot = events.pivot_table(
            index="date",
            columns="weather_point_id",
            values="capacity_delta_mw",
            aggfunc="sum",
            fill_value=0.0,
        )
        event_pivot = event_pivot.reindex(days, fill_value=0.0)
        for column in event_pivot.columns:
            if column in result.columns:
                result[column] = result[column] + event_pivot[column].cumsum()

    return result.clip(lower=0.0)


def _capacity_artifact_months(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if index.empty:
        raise ValueError("Cannot build capacity artifacts for an empty feature index.")
    start_ts = index.min()
    end_ts = index.max()
    start = pd.Timestamp(year=start_ts.year, month=start_ts.month, day=1, tz=index.tz)
    end = pd.Timestamp(year=end_ts.year, month=end_ts.month, day=1, tz=index.tz)
    return pd.date_range(start=start, end=end, freq="MS", tz=index.tz, name="month")


def _capacity_artifact_days(months: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return pd.date_range(start=months.min(), end=months.max(), freq="D", tz=months.tz, name="date")


def build_capacity_timeseries_artifact(
    capacity_map: pd.DataFrame,
    timestamps: pd.DatetimeIndex,
    config: RegionalRenewableFeatureConfig,
) -> pd.DataFrame:
    """Build monthly installed capacity by technology/region from MaStR commissioning dates."""
    months = _capacity_artifact_months(timestamps)
    days = _capacity_artifact_days(months)
    result = pd.DataFrame(index=months)
    for technology in config.capacity_artifact_technologies:
        technology_regions = sorted(
            capacity_map.loc[capacity_map["technology_group"] == technology, "region"].dropna().unique()
        )
        technology_total = pd.Series(0.0, index=months)
        for region in technology_regions:
            capacity_by_cluster = _active_capacity_by_cluster(
                capacity_map,
                days,
                technology=technology,
                region=region,
            )
            installed_capacity = capacity_by_cluster.sum(axis=1).reindex(months).fillna(0.0)
            result[f"{region}_installed_capacity_mw"] = installed_capacity.to_numpy(dtype=float)
            technology_total = technology_total + installed_capacity
        if technology_regions:
            result[f"{technology}_total_installed_capacity_mw"] = technology_total.to_numpy(dtype=float)
    result.index.name = "month"
    return result.reset_index()


def build_weather_weights_artifact(
    capacity_map: pd.DataFrame,
    timestamps: pd.DatetimeIndex,
    config: RegionalRenewableFeatureConfig,
) -> pd.DataFrame:
    """Build monthly capacity weights for aggregating cluster weather by technology/region."""
    months = _capacity_artifact_months(timestamps)
    days = _capacity_artifact_days(months)
    clusters = _load_clusters(config.cluster_file).drop_duplicates("cluster_id").set_index("cluster_id")
    rows: list[dict[str, object]] = []
    for technology in config.capacity_artifact_technologies:
        technology_regions = sorted(
            capacity_map.loc[capacity_map["technology_group"] == technology, "region"].dropna().unique()
        )
        for region in technology_regions:
            capacity_by_cluster = _active_capacity_by_cluster(
                capacity_map,
                days,
                technology=technology,
                region=region,
            ).reindex(months)
            for month, active_capacity in capacity_by_cluster.iterrows():
                region_capacity = float(active_capacity.sum())
                if region_capacity <= 0:
                    continue
                for cluster_id, capacity_mw in active_capacity.items():
                    capacity_mw = float(capacity_mw)
                    if capacity_mw <= 0:
                        continue
                    cluster = clusters.loc[int(cluster_id)] if int(cluster_id) in clusters.index else None
                    rows.append(
                        {
                            "month": month,
                            "technology_group": technology,
                            "region": region,
                            "cluster_id": int(cluster_id),
                            "cluster_lat": float(cluster["lat"]) if cluster is not None else np.nan,
                            "cluster_lon": float(cluster["lon"]) if cluster is not None else np.nan,
                            "active_capacity_mw": capacity_mw,
                            "region_capacity_mw": region_capacity,
                            "weight": capacity_mw / region_capacity,
                        }
                    )
    return pd.DataFrame(
        rows,
        columns=[
            "month",
            "technology_group",
            "region",
            "cluster_id",
            "cluster_lat",
            "cluster_lon",
            "active_capacity_mw",
            "region_capacity_mw",
            "weight",
        ],
    )


_NORTH_SEA_COASTLINE_POINTS = np.array(
    [
        (6.70, 53.55),
        (7.20, 53.72),
        (7.95, 53.70),
        (8.45, 53.86),
        (8.70, 54.32),
        (8.90, 54.78),
    ],
    dtype=float,
)
_BALTIC_SEA_COASTLINE_POINTS = np.array(
    [
        (9.35, 54.80),
        (10.10, 54.38),
        (10.88, 54.37),
        (11.45, 54.35),
        (12.15, 54.18),
        (12.95, 54.43),
        (13.70, 54.10),
        (14.25, 53.92),
    ],
    dtype=float,
)


def _distance_to_polyline_km(lon: np.ndarray, lat: np.ndarray, polyline_lon_lat: np.ndarray) -> np.ndarray:
    """Approximate point-to-polyline distance in km using a local equirectangular projection."""
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    ref_lat = np.deg2rad(np.nanmean(np.concatenate([lat, polyline_lon_lat[:, 1]])))
    point_x = lon * 111.32 * np.cos(ref_lat)
    point_y = lat * 111.32
    line_x = polyline_lon_lat[:, 0] * 111.32 * np.cos(ref_lat)
    line_y = polyline_lon_lat[:, 1] * 111.32

    distances = np.full(len(lon), np.inf, dtype=float)
    for idx in range(len(polyline_lon_lat) - 1):
        start_x = line_x[idx]
        start_y = line_y[idx]
        end_x = line_x[idx + 1]
        end_y = line_y[idx + 1]
        segment_x = end_x - start_x
        segment_y = end_y - start_y
        segment_len_sq = segment_x**2 + segment_y**2
        if segment_len_sq <= 0:
            candidate = np.sqrt((point_x - start_x) ** 2 + (point_y - start_y) ** 2)
        else:
            projection = ((point_x - start_x) * segment_x + (point_y - start_y) * segment_y) / segment_len_sq
            projection = np.clip(projection, 0.0, 1.0)
            nearest_x = start_x + projection * segment_x
            nearest_y = start_y + projection * segment_y
            candidate = np.sqrt((point_x - nearest_x) ** 2 + (point_y - nearest_y) ** 2)
        distances = np.minimum(distances, candidate)
    return distances


def _weighted_share(mask: np.ndarray, weights: np.ndarray) -> float:
    total = float(np.nansum(weights))
    if total <= 0:
        return 0.0
    return float(np.nansum(weights[mask]) / total)


def _weighted_hhi(values: pd.Series, weights: np.ndarray) -> float:
    if values.empty or float(np.nansum(weights)) <= 0:
        return np.nan
    frame = pd.DataFrame({"value": values.astype(str).to_numpy(), "weight": weights})
    shares = frame.groupby("value")["weight"].sum().to_numpy(dtype=float) / float(np.nansum(weights))
    return float(np.sum(shares**2))


def _geographic_exposure_features(active: pd.DataFrame) -> dict[str, float]:
    capacity = active["capacity_mw"].to_numpy(dtype=float)
    lat = active["_lat"].to_numpy(dtype=float)
    lon = active["_lon"].to_numpy(dtype=float)
    north_sea_distance = _distance_to_polyline_km(lon, lat, _NORTH_SEA_COASTLINE_POINTS)
    baltic_sea_distance = _distance_to_polyline_km(lon, lat, _BALTIC_SEA_COASTLINE_POINTS)
    coast_distance = np.minimum(north_sea_distance, baltic_sea_distance)
    nearest_is_north_sea = north_sea_distance <= baltic_sea_distance
    exposure_score = np.exp(-coast_distance / 100.0)

    return {
        "distance_to_coast_cap_weighted_km": float(np.average(coast_distance, weights=capacity)),
        "distance_to_coast_cap_weighted_std_km": float(_weighted_std(coast_distance[None, :], capacity[None, :])[0]),
        "distance_to_north_sea_cap_weighted_km": float(np.average(north_sea_distance, weights=capacity)),
        "distance_to_baltic_sea_cap_weighted_km": float(np.average(baltic_sea_distance, weights=capacity)),
        "north_sea_nearest_capacity_share": _weighted_share(nearest_is_north_sea, capacity),
        "baltic_sea_nearest_capacity_share": _weighted_share(~nearest_is_north_sea, capacity),
        "coastal_capacity_share_50km": _weighted_share(coast_distance <= 50.0, capacity),
        "coastal_capacity_share_100km": _weighted_share(coast_distance <= 100.0, capacity),
        "coastal_capacity_share_150km": _weighted_share(coast_distance <= 150.0, capacity),
        "coastal_exposure_score_cap_weighted": float(np.average(exposure_score, weights=capacity)),
    }


def _add_capacity_static_features(
    frames: dict[str, pd.Series],
    capacity_map: pd.DataFrame,
    timestamps: pd.DatetimeIndex,
    config: RegionalRenewableFeatureConfig,
    *,
    technology: str,
    region: str,
    prefix: str,
    group_column: str,
) -> None:
    """Add operationally available static MaStR fleet descriptors by region."""
    subset = capacity_map.loc[
        (capacity_map["technology_group"] == technology) & (capacity_map["region"] == region)
    ].copy()
    if subset.empty:
        return

    required_columns = {
        "capacity_mw",
        "commissioning_date",
        "decommissioning_date",
        config.latitude_column,
        config.longitude_column,
    }
    if group_column in subset.columns:
        required_columns.add(group_column)
    missing = required_columns - set(subset.columns)
    if missing:
        raise ValueError(f"Capacity map is missing columns required for static features: {sorted(missing)}")

    subset["capacity_mw"] = pd.to_numeric(subset["capacity_mw"], errors="coerce")
    subset["_commissioning_ts"] = pd.to_datetime(subset["commissioning_date"], errors="coerce").dt.normalize()
    subset["_decommissioning_ts"] = pd.to_datetime(subset["decommissioning_date"], errors="coerce").dt.normalize()
    subset["_lat"] = pd.to_numeric(subset[config.latitude_column], errors="coerce")
    subset["_lon"] = pd.to_numeric(subset[config.longitude_column], errors="coerce")
    subset = subset.dropna(subset=["capacity_mw", "_commissioning_ts", "_lat", "_lon"])
    subset = subset.loc[subset["capacity_mw"] > 0].copy()
    if subset.empty:
        return

    days = pd.DatetimeIndex(timestamps.normalize().unique()).sort_values()
    rows: list[dict[str, float]] = []
    for day in days:
        day_naive = pd.Timestamp(day).tz_localize(None).normalize()
        active = subset.loc[
            (subset["_commissioning_ts"] <= day_naive)
            & (subset["_decommissioning_ts"].isna() | (subset["_decommissioning_ts"] > day_naive))
        ]
        if active.empty:
            row = {"timestamp": day}
            if config.include_capacity_static_features:
                row.update(
                    {
                    "installed_capacity_mw": 0.0,
                    "unit_count": 0.0,
                    "mean_unit_capacity_mw": 0.0,
                    "max_unit_capacity_mw": 0.0,
                    "capacity_weighted_lat": np.nan,
                    "capacity_weighted_lon": np.nan,
                    "capacity_spread_km": np.nan,
                    "capacity_concentration_hhi": np.nan,
                    "capacity_weighted_age_years": np.nan,
                    "capacity_new_share_365d": 0.0,
                    }
                )
            if config.include_geographic_exposure_features:
                row.update(
                    {
                        "distance_to_coast_cap_weighted_km": np.nan,
                        "distance_to_coast_cap_weighted_std_km": np.nan,
                        "distance_to_north_sea_cap_weighted_km": np.nan,
                        "distance_to_baltic_sea_cap_weighted_km": np.nan,
                        "north_sea_nearest_capacity_share": 0.0,
                        "baltic_sea_nearest_capacity_share": 0.0,
                        "coastal_capacity_share_50km": 0.0,
                        "coastal_capacity_share_100km": 0.0,
                        "coastal_capacity_share_150km": 0.0,
                        "coastal_exposure_score_cap_weighted": np.nan,
                    }
                )
            rows.append(row)
            continue

        capacity = active["capacity_mw"].to_numpy(dtype=float)
        total_capacity = float(capacity.sum())
        lat = active["_lat"].to_numpy(dtype=float)
        lon = active["_lon"].to_numpy(dtype=float)

        row = {"timestamp": day}
        if config.include_capacity_static_features:
            lat_mean = float(np.average(lat, weights=capacity))
            lon_mean = float(np.average(lon, weights=capacity))
            lat_scale = 111.32
            lon_scale = 111.32 * np.cos(np.deg2rad(lat_mean))
            spread_km = float(
                np.sqrt(
                    np.average(
                        ((lat - lat_mean) * lat_scale) ** 2 + ((lon - lon_mean) * lon_scale) ** 2,
                        weights=capacity,
                    )
                )
            )

            if group_column in active.columns:
                group_capacity = active.groupby(group_column)["capacity_mw"].sum()
                shares = group_capacity.to_numpy(dtype=float) / total_capacity
                hhi = float(np.sum(shares**2))
            else:
                unit_shares = capacity / total_capacity
                hhi = float(np.sum(unit_shares**2))

            age_days = (day_naive - active["_commissioning_ts"]).dt.days.clip(lower=0).to_numpy(dtype=float)
            recent_cutoff = day_naive - pd.Timedelta(days=365)
            recent_capacity = active.loc[active["_commissioning_ts"] >= recent_cutoff, "capacity_mw"].sum()
            row.update(
                {
                "installed_capacity_mw": total_capacity,
                "unit_count": float(len(active)),
                "mean_unit_capacity_mw": float(total_capacity / len(active)),
                "max_unit_capacity_mw": float(np.nanmax(capacity)),
                "capacity_weighted_lat": lat_mean,
                "capacity_weighted_lon": lon_mean,
                "capacity_spread_km": spread_km,
                "capacity_concentration_hhi": hhi,
                "capacity_weighted_age_years": float(np.average(age_days / 365.25, weights=capacity)),
                "capacity_new_share_365d": float(recent_capacity / total_capacity),
                }
            )

        if config.include_wind_turbine_static_features and technology.startswith("wind_"):
            metadata_updates: dict[str, float] = {}
            for column, suffix in (
                (config.wind_hub_height_column, "hub_height_m"),
                (config.wind_rotor_diameter_column, "rotor_diameter_m"),
                (config.wind_swept_area_column, "swept_area_m2"),
                ("specific_power_mw_per_m2", "specific_power_mw_per_m2"),
            ):
                if column not in active.columns:
                    continue
                values = pd.to_numeric(active[column], errors="coerce").to_numpy(dtype=float)
                valid = np.isfinite(values)
                if not valid.any():
                    continue
                metadata_updates[f"{suffix}_cap_weighted"] = float(np.average(values[valid], weights=capacity[valid]))
                metadata_updates[f"{suffix}_cap_weighted_std"] = float(
                    _weighted_std(values[valid][None, :], capacity[valid][None, :])[0]
                )

            if config.wind_manufacturer_column in active.columns:
                metadata_updates["manufacturer_capacity_hhi"] = _weighted_hhi(
                    active[config.wind_manufacturer_column],
                    capacity,
                )
                metadata_updates["manufacturer_count"] = float(active[config.wind_manufacturer_column].nunique(dropna=True))
            if config.wind_turbine_type_column in active.columns:
                metadata_updates["turbine_type_capacity_hhi"] = _weighted_hhi(
                    active[config.wind_turbine_type_column],
                    capacity,
                )
                metadata_updates["turbine_type_count"] = float(active[config.wind_turbine_type_column].nunique(dropna=True))
            if "curtailment_limitation_flag" in active.columns:
                flags = active["curtailment_limitation_flag"].astype(str).isin({"1", "true", "True"})
                metadata_updates["curtailment_limited_capacity_share"] = _weighted_share(flags.to_numpy(), capacity)

            row.update(metadata_updates)

        if config.include_geographic_exposure_features:
            row.update(_geographic_exposure_features(active))

        rows.append(row)

    daily = pd.DataFrame(rows).set_index("timestamp").sort_index()
    daily_for_timestamps = daily.reindex(timestamps.normalize())
    daily_for_timestamps.index = timestamps
    for column in daily_for_timestamps.columns:
        frames[f"{prefix}_{column}"] = daily_for_timestamps[column].astype(float)


def _columns_for_clusters(df: pd.DataFrame, prefix: str, clusters: list[int]) -> list[str]:
    return [f"{prefix}_cluster_{cluster_id}" for cluster_id in clusters if f"{prefix}_cluster_{cluster_id}" in df.columns]


def _columns_for_points(df: pd.DataFrame, prefix: str, point_ids: list[int]) -> list[str]:
    return [f"{prefix}_point_{point_id}" for point_id in point_ids if f"{prefix}_point_{point_id}" in df.columns]


def _capacity_for_timestamps(capacity_daily: pd.DataFrame, timestamps: pd.DatetimeIndex, clusters: list[int]) -> np.ndarray:
    capacity = capacity_daily.reindex(timestamps.normalize()).loc[:, clusters].fillna(0.0)
    capacity.index = timestamps
    return capacity.to_numpy(dtype=float)


def _static_point_metadata(
    assigned_capacity: pd.DataFrame,
    *,
    point_ids: list[int],
    value_column: str,
    weight_column: str = "capacity_mw",
) -> np.ndarray:
    values = []
    for point_id in point_ids:
        subset = assigned_capacity.loc[assigned_capacity["weather_point_id"] == point_id]
        if subset.empty or value_column not in subset.columns:
            values.append(np.nan)
            continue
        value = pd.to_numeric(subset[value_column], errors="coerce")
        weight = pd.to_numeric(subset[weight_column], errors="coerce") if weight_column in subset.columns else subset["capacity_mw"]
        valid = value.notna() & weight.notna() & (weight > 0)
        if not valid.any():
            values.append(np.nan)
            continue
        values.append(float(np.average(value.loc[valid], weights=weight.loc[valid])))
    return np.asarray(values, dtype=float)


def _static_point_sum(
    assigned_capacity: pd.DataFrame,
    *,
    point_ids: list[int],
    value_column: str,
) -> np.ndarray:
    values = []
    for point_id in point_ids:
        subset = assigned_capacity.loc[assigned_capacity["weather_point_id"] == point_id]
        if subset.empty or value_column not in subset.columns:
            values.append(np.nan)
            continue
        value = pd.to_numeric(subset[value_column], errors="coerce")
        values.append(float(value.dropna().sum()) if value.notna().any() else np.nan)
    return np.asarray(values, dtype=float)


def _wind_weight_matrix(
    assigned_capacity: pd.DataFrame,
    capacity: np.ndarray,
    *,
    point_ids: list[int],
    config: RegionalRenewableFeatureConfig,
) -> np.ndarray:
    if config.wind_weather_weight_mode == "capacity":
        return capacity

    rotor_area = _static_point_sum(
        assigned_capacity,
        point_ids=point_ids,
        value_column=config.wind_swept_area_column,
    )
    rotor_area = np.where(np.isfinite(rotor_area) & (rotor_area > 0), rotor_area, 1.0)

    if config.wind_weather_weight_mode == "rotor_area":
        return np.broadcast_to(rotor_area, capacity.shape)
    if config.wind_weather_weight_mode == "capacity_x_rotor_area":
        return capacity * rotor_area[None, :]
    raise ValueError(f"Unsupported wind_weather_weight_mode: {config.wind_weather_weight_mode!r}")


def _interpolate_speed_to_hub_height(
    speed80: np.ndarray,
    speed120: np.ndarray,
    hub_height_m: np.ndarray,
) -> np.ndarray:
    hub = np.where(np.isfinite(hub_height_m), hub_height_m, 100.0)
    weight = (hub - 80.0) / 40.0
    weight = np.clip(weight, -1.0, 2.5)
    return np.clip(speed80 + weight[None, :] * (speed120 - speed80), 0.0, None)


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    denominator = weights.sum(axis=1)
    numerator = (values * weights).sum(axis=1)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=denominator > 0)


def _weighted_std(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    mean = _weighted_mean(values, weights)
    centered = values - mean[:, None]
    variance = _weighted_mean(centered**2, weights)
    return np.sqrt(np.clip(variance, 0.0, None))


def _air_density_kg_m3(pressure_pa: np.ndarray, temperature_k: np.ndarray) -> np.ndarray:
    return np.divide(
        pressure_pa,
        287.05 * temperature_k,
        out=np.full_like(pressure_pa, np.nan, dtype=float),
        where=temperature_k > 0,
    )


def _weighted_wind_direction_components(u: np.ndarray, v: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    speed = np.sqrt(u**2 + v**2)
    unit_u = np.divide(u, speed, out=np.zeros_like(u, dtype=float), where=speed > 0)
    unit_v = np.divide(v, speed, out=np.zeros_like(v, dtype=float), where=speed > 0)
    return _weighted_mean(unit_u, weights), _weighted_mean(unit_v, weights)


def _add_ramps(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if not columns:
        return df
    blocks = [df]
    for step, suffix in ((4, "ramp_1h"), (12, "ramp_3h")):
        ramp = df[columns].diff(step).rename(columns={column: f"{column}_{suffix}" for column in columns})
        blocks.append(ramp)
    return pd.concat(blocks, axis=1)


def _add_wind_cluster_features(
    frames: dict[str, pd.Series],
    ramp_columns: list[str],
    capacity_map: pd.DataFrame,
    weather: pd.DataFrame,
    config: RegionalRenewableFeatureConfig,
    *,
    technology: str,
    u_prefix: str,
    v_prefix: str,
    wind_factor: float,
) -> None:
    capacity_daily = _active_capacity_by_cluster(capacity_map, weather.index, technology=technology, region=None)
    for cluster_id in sorted(capacity_daily.columns):
        u_col = f"{u_prefix}_cluster_{cluster_id}"
        v_col = f"{v_prefix}_cluster_{cluster_id}"
        if u_col not in weather.columns or v_col not in weather.columns:
            continue

        capacity = capacity_daily.reindex(weather.index.normalize())[cluster_id].fillna(0.0).to_numpy(dtype=float)
        if np.nanmax(capacity) <= 0:
            continue

        u = weather[u_col].to_numpy(dtype=float)
        v = weather[v_col].to_numpy(dtype=float)
        speed = np.sqrt(u**2 + v**2) * wind_factor
        proxy = capacity * _normalised_wind_power_curve(speed, config.wind_power_curve_smoothing_sigma_m_s)
        proxy_v3 = capacity * speed**3

        prefix = f"{technology}_cluster_{cluster_id}"
        frames[f"{prefix}_capacity_mw"] = pd.Series(capacity, index=weather.index)
        frames[f"{prefix}_proxy_mw"] = pd.Series(proxy, index=weather.index)
        frames[f"{prefix}_proxy_v3"] = pd.Series(proxy_v3, index=weather.index)
        frames[f"{prefix}_speed_hub_m_s"] = pd.Series(speed, index=weather.index)
        frames[f"{prefix}_u_m_s"] = pd.Series(u, index=weather.index)
        frames[f"{prefix}_v_m_s"] = pd.Series(v, index=weather.index)
        speed_safe = np.divide(1.0, speed, out=np.zeros_like(speed, dtype=float), where=speed > 0)
        frames[f"{prefix}_direction_cos"] = pd.Series(u * speed_safe, index=weather.index)
        frames[f"{prefix}_direction_sin"] = pd.Series(v * speed_safe, index=weather.index)
        ramp_columns.append(f"{prefix}_proxy_mw")

        for variable, suffix in (("t2m", "t2m_K"), ("sp", "sp_Pa"), ("tcc", "cloud_cover")):
            column = f"{variable}_cluster_{cluster_id}"
            if column in weather.columns:
                frames[f"{prefix}_{suffix}"] = pd.Series(weather[column].to_numpy(dtype=float), index=weather.index)


def _available_wind_model_levels(weather: pd.DataFrame, clusters: list[int]) -> list[int]:
    levels: set[int] = set()
    for column in weather.columns:
        if not column.startswith("u_ml") or "_cluster_" not in column:
            continue
        level_token = column.removeprefix("u_ml").split("_", 1)[0]
        if not level_token.isdigit():
            continue
        level = int(level_token)
        if all(
            f"u_ml{level}_cluster_{cluster_id}" in weather.columns
            and f"v_ml{level}_cluster_{cluster_id}" in weather.columns
            for cluster_id in clusters
        ):
            levels.add(level)
    return sorted(levels)


def _add_wind_model_level_features(
    frames: dict[str, pd.Series],
    ramp_columns: list[str],
    *,
    prefix: str,
    weather: pd.DataFrame,
    capacity: np.ndarray,
    clusters: list[int],
    smoothing_sigma_m_s: float = 0.0,
) -> None:
    levels = _available_wind_model_levels(weather, clusters)
    if not levels:
        return

    speed_by_level: dict[int, np.ndarray] = {}
    for level in levels:
        u = weather[_columns_for_clusters(weather, f"u_ml{level}", clusters)].to_numpy(dtype=float)
        v = weather[_columns_for_clusters(weather, f"v_ml{level}", clusters)].to_numpy(dtype=float)
        speed = np.sqrt(u**2 + v**2)
        speed_by_level[level] = speed
        proxy_curve = (capacity * _normalised_wind_power_curve(speed, smoothing_sigma_m_s)).sum(axis=1)
        proxy_v3 = (capacity * speed**3).sum(axis=1)

        frames[f"{prefix}_proxy_ml{level}_mw"] = pd.Series(proxy_curve, index=weather.index)
        frames[f"{prefix}_proxy_v3_ml{level}"] = pd.Series(proxy_v3, index=weather.index)
        frames[f"{prefix}_speed_ml{level}_cap_weighted_m_s"] = pd.Series(
            _weighted_mean(speed, capacity),
            index=weather.index,
        )
        frames[f"{prefix}_speed_ml{level}_cap_weighted_std_m_s"] = pd.Series(
            _weighted_std(speed, capacity),
            index=weather.index,
        )
        frames[f"{prefix}_u_ml{level}_cap_weighted_m_s"] = pd.Series(_weighted_mean(u, capacity), index=weather.index)
        frames[f"{prefix}_v_ml{level}_cap_weighted_m_s"] = pd.Series(_weighted_mean(v, capacity), index=weather.index)
        ramp_columns.append(f"{prefix}_proxy_ml{level}_mw")

    for lower_level, upper_level in zip(levels, levels[1:], strict=False):
        lower_speed = speed_by_level[lower_level]
        upper_speed = speed_by_level[upper_level]
        frames[f"{prefix}_speed_ml{upper_level}_minus_ml{lower_level}_cap_weighted_m_s"] = pd.Series(
            _weighted_mean(upper_speed - lower_speed, capacity),
            index=weather.index,
        )
        ratio = np.divide(
            upper_speed,
            lower_speed,
            out=np.ones_like(upper_speed, dtype=float),
            where=lower_speed > 1e-6,
        )
        frames[f"{prefix}_speed_ml{upper_level}_to_ml{lower_level}_ratio_cap_weighted"] = pd.Series(
            _weighted_mean(ratio, capacity),
            index=weather.index,
        )


def _add_solar_cluster_features(
    frames: dict[str, pd.Series],
    ramp_columns: list[str],
    capacity_map: pd.DataFrame,
    weather: pd.DataFrame,
    config: RegionalRenewableFeatureConfig,
) -> None:
    capacity_daily = _active_capacity_by_cluster(capacity_map, weather.index, technology="solar", region=None)
    for cluster_id in sorted(capacity_daily.columns):
        ssrd_col = f"ssrd_cluster_{cluster_id}"
        if ssrd_col not in weather.columns:
            continue

        capacity = capacity_daily.reindex(weather.index.normalize())[cluster_id].fillna(0.0).to_numpy(dtype=float)
        if np.nanmax(capacity) <= 0:
            continue

        ssrd = np.clip(weather[ssrd_col].to_numpy(dtype=float), 0.0, None)
        proxy = capacity * ssrd / 1000.0 * config.solar_performance_ratio

        prefix = f"solar_cluster_{cluster_id}"
        frames[f"{prefix}_capacity_mw"] = pd.Series(capacity, index=weather.index)
        frames[f"{prefix}_proxy_mw"] = pd.Series(proxy, index=weather.index)
        frames[f"{prefix}_irradiance_W_m2"] = pd.Series(ssrd, index=weather.index)
        ramp_columns.append(f"{prefix}_proxy_mw")

        for variable, suffix in (
            ("fdir", "direct_irradiance_W_m2"),
            ("t2m", "t2m_K"),
            ("tcc", "cloud_cover"),
        ):
            column = f"{variable}_cluster_{cluster_id}"
            if column in weather.columns:
                values = weather[column].to_numpy(dtype=float)
                if variable == "fdir":
                    values = np.clip(values, 0.0, None)
                frames[f"{prefix}_{suffix}"] = pd.Series(values, index=weather.index)


def _deduplicate_mean(df: pd.DataFrame) -> pd.DataFrame:
    if df.index.has_duplicates:
        return df.groupby(level=0).mean()
    return df.sort_index()


def _load_dwd_icon_weather(config: RegionalRenewableFeatureConfig) -> pd.DataFrame:
    df_hourly, df_qh = load_dwd(
        icon_dir=config.icon_dir,
        start_folder_date=config.start_folder_date,
        required_run=config.required_run,
        skip_dates=set(config.skip_dates),
        folder_offset_date=config.dwd_folder_offset_date,
        target_tz=config.target_tz,
    )
    df_hourly = _deduplicate_mean(df_hourly)
    df_qh = _deduplicate_mean(df_qh)

    full_index = pd.date_range(
        start=min(df_hourly.index.min(), df_qh.index.min()),
        end=max(df_hourly.index.max(), df_qh.index.max()),
        freq="15min",
        tz=config.target_tz,
        name="timestamp",
    )

    hourly = df_hourly.reindex(full_index).ffill(limit=3)
    qh = df_qh.reindex(full_index)
    weather = pd.concat([hourly, qh], axis=1).sort_index()

    direct_cols = sorted(column for column in weather.columns if column.startswith("ASWDIR_cluster_"))
    solar_columns: dict[str, np.ndarray] = {}
    for direct_col in direct_cols:
        cluster_id = direct_col.rsplit("_", 1)[-1]
        diffuse_col = f"ASWDIFD_cluster_{cluster_id}"
        if diffuse_col not in weather.columns:
            continue
        direct = np.clip(weather[direct_col].to_numpy(dtype=float), 0.0, None)
        diffuse = np.clip(weather[diffuse_col].to_numpy(dtype=float), 0.0, None)
        solar_columns[f"fdir_cluster_{cluster_id}"] = direct
        solar_columns[f"ssrd_cluster_{cluster_id}"] = direct + diffuse
    if solar_columns:
        weather = pd.concat([weather, pd.DataFrame(solar_columns, index=weather.index)], axis=1)

    weather.index.name = "timestamp"
    return weather


def _load_weather(config: RegionalRenewableFeatureConfig) -> pd.DataFrame:
    if config.weather_source == "era5":
        return load_era5(config.era5_dirs, target_tz=config.target_tz)
    if config.weather_source == "dwd_icon":
        return _load_dwd_icon_weather(config)
    if config.weather_source == "open_meteo":
        return load_open_meteo(
            cluster_file=config.cluster_file,
            start_date=config.open_meteo_start_date,
            end_date=config.open_meteo_end_date,
            cache_file=config.open_meteo_weather_file,
            base_url=config.open_meteo_base_url,
            api_key_env=config.open_meteo_api_key_env,
            model=config.open_meteo_model,
            hourly_variables=config.open_meteo_hourly_variables,
            batch_size=config.open_meteo_batch_size,
            cell_selection=config.open_meteo_cell_selection,
            timeout_seconds=config.open_meteo_timeout_seconds,
            target_tz=config.target_tz,
            force_download=config.open_meteo_force_download,
            point_selection=config.open_meteo_point_selection,
            max_points_per_cluster=config.open_meteo_max_points_per_cluster,
            api_mode=config.open_meteo_api_mode,
            single_run_hour_utc=config.open_meteo_single_run_hour_utc,
            single_run_forecast_days=config.open_meteo_single_run_forecast_days,
            request_pause_seconds=config.open_meteo_request_pause_seconds,
            retry_attempts=config.open_meteo_retry_attempts,
            retry_backoff_seconds=config.open_meteo_retry_backoff_seconds,
            skip_unavailable_runs=config.open_meteo_skip_unavailable_runs,
            fallback_previous_runs=config.open_meteo_fallback_previous_runs,
            fallback_step_hours=config.open_meteo_fallback_step_hours,
            fallback_max_lookback_hours=config.open_meteo_fallback_max_lookback_hours,
        )
    raise ValueError(f"Unsupported regional renewable weather_source: {config.weather_source!r}")


def _load_open_meteo_capacity_point_weather(
    config: RegionalRenewableFeatureConfig,
    points: pd.DataFrame,
) -> pd.DataFrame:
    return load_open_meteo_points(
        points=points,
        start_date=config.open_meteo_start_date,
        end_date=config.open_meteo_end_date,
        cache_file=config.open_meteo_weather_file,
        base_url=config.open_meteo_base_url,
        api_key_env=config.open_meteo_api_key_env,
        model=config.open_meteo_model,
        hourly_variables=config.open_meteo_hourly_variables,
        batch_size=config.open_meteo_batch_size,
        cell_selection=config.open_meteo_cell_selection,
        timeout_seconds=config.open_meteo_timeout_seconds,
        target_tz=config.target_tz,
        force_download=config.open_meteo_force_download,
        api_mode=config.open_meteo_api_mode,
        single_run_hour_utc=config.open_meteo_single_run_hour_utc,
        single_run_forecast_days=config.open_meteo_single_run_forecast_days,
        request_pause_seconds=config.open_meteo_request_pause_seconds,
        retry_attempts=config.open_meteo_retry_attempts,
        retry_backoff_seconds=config.open_meteo_retry_backoff_seconds,
        skip_unavailable_runs=config.open_meteo_skip_unavailable_runs,
        fallback_previous_runs=config.open_meteo_fallback_previous_runs,
        fallback_step_hours=config.open_meteo_fallback_step_hours,
        fallback_max_lookback_hours=config.open_meteo_fallback_max_lookback_hours,
    )


def _build_capacity_point_renewable_features(
    capacity_map: pd.DataFrame,
    points: pd.DataFrame,
    weather: pd.DataFrame,
    config: RegionalRenewableFeatureConfig,
) -> pd.DataFrame:
    assigned_capacity = _assign_capacity_to_weather_points(capacity_map, points, config)
    u_prefix = "u100" if any(column.startswith("u100_point_") for column in weather.columns) else "u10"
    v_prefix = "v100" if u_prefix == "u100" else "v10"
    wind_factor = 1.0
    if u_prefix == "u10":
        wind_factor = (config.wind_hub_height_m / config.wind_reference_height_m) ** config.wind_shear_alpha

    frames: dict[str, pd.Series] = {}
    ramp_columns: list[str] = []
    wind_total = pd.Series(0.0, index=weather.index)
    solar_total = pd.Series(0.0, index=weather.index)

    for technology in ["wind_offshore", "wind_onshore"]:
        regions = sorted(points.loc[points["technology_group"] == technology, "region"].unique())
        for region in regions:
            capacity_daily = _active_capacity_by_weather_point(
                assigned_capacity,
                weather.index,
                technology=technology,
                region=region,
            )
            point_ids = [
                int(point_id)
                for point_id in capacity_daily.columns
                if f"{u_prefix}_point_{point_id}" in weather.columns and f"{v_prefix}_point_{point_id}" in weather.columns
            ]
            if not point_ids:
                continue

            capacity = _capacity_for_timestamps(capacity_daily, weather.index, point_ids)
            wind_feature_weights = _wind_weight_matrix(
                assigned_capacity,
                capacity,
                point_ids=point_ids,
                config=config,
            )
            u = weather[_columns_for_points(weather, u_prefix, point_ids)].to_numpy(dtype=float)
            v = weather[_columns_for_points(weather, v_prefix, point_ids)].to_numpy(dtype=float)
            speed = np.sqrt(u**2 + v**2) * wind_factor
            proxy_curve = (capacity * _normalised_wind_power_curve(speed, config.wind_power_curve_smoothing_sigma_m_s)).sum(axis=1)
            proxy_v3 = (capacity * speed**3).sum(axis=1)

            prefix = f"wind_{region}"
            if config.include_capacity_static_features or config.include_geographic_exposure_features:
                _add_capacity_static_features(
                    frames,
                    assigned_capacity,
                    weather.index,
                    config,
                    technology=technology,
                    region=region,
                    prefix=prefix,
                    group_column="weather_point_id",
                )
            frames[f"{prefix}_proxy_mw"] = pd.Series(proxy_curve, index=weather.index)
            frames[f"{prefix}_proxy_v3"] = pd.Series(proxy_v3, index=weather.index)
            frames[f"{prefix}_speed_hub_cap_weighted_m_s"] = pd.Series(_weighted_mean(speed, capacity), index=weather.index)
            frames[f"{prefix}_speed_hub_cap_weighted_std_m_s"] = pd.Series(_weighted_std(speed, capacity), index=weather.index)
            if config.wind_weather_weight_mode != "capacity":
                frames[f"{prefix}_speed_hub_{config.wind_weather_weight_mode}_weighted_m_s"] = pd.Series(
                    _weighted_mean(speed, wind_feature_weights),
                    index=weather.index,
                )
                frames[f"{prefix}_proxy_v3_{config.wind_weather_weight_mode}_weighted"] = pd.Series(
                    (wind_feature_weights * speed**3).sum(axis=1),
                    index=weather.index,
                )
            frames[f"{prefix}_u_cap_weighted_m_s"] = pd.Series(_weighted_mean(u, capacity), index=weather.index)
            frames[f"{prefix}_v_cap_weighted_m_s"] = pd.Series(_weighted_mean(v, capacity), index=weather.index)
            direction_cos, direction_sin = _weighted_wind_direction_components(u, v, capacity)
            frames[f"{prefix}_direction_cos_cap_weighted"] = pd.Series(direction_cos, index=weather.index)
            frames[f"{prefix}_direction_sin_cap_weighted"] = pd.Series(direction_sin, index=weather.index)
            wind_total = wind_total + frames[f"{prefix}_proxy_mw"]
            ramp_columns.append(f"{prefix}_proxy_mw")

            if f"vmax10m_point_{point_ids[0]}" in weather.columns:
                vmax10 = np.clip(weather[_columns_for_points(weather, "vmax10m", point_ids)].to_numpy(dtype=float), 0.0, None)
                vmax_hub = vmax10 * (config.wind_hub_height_m / config.wind_reference_height_m) ** config.wind_shear_alpha
                gust_ratio = np.divide(
                    vmax_hub,
                    speed,
                    out=np.ones_like(vmax_hub, dtype=float),
                    where=speed > 1e-6,
                )
                frames[f"{prefix}_vmax10m_cap_weighted_m_s"] = pd.Series(
                    _weighted_mean(vmax10, capacity),
                    index=weather.index,
                )
                frames[f"{prefix}_vmax_hub_cap_weighted_m_s"] = pd.Series(
                    _weighted_mean(vmax_hub, capacity),
                    index=weather.index,
                )
                frames[f"{prefix}_gust_to_hub_speed_ratio_cap_weighted"] = pd.Series(
                    _weighted_mean(gust_ratio, capacity),
                    index=weather.index,
                )
                frames[f"{prefix}_proxy_vmax_v3"] = pd.Series((capacity * vmax_hub**3).sum(axis=1), index=weather.index)

            if all(f"{wind_column}_point_{point_ids[0]}" in weather.columns for wind_column in ["u80", "v80", "u120", "v120"]):
                u80 = weather[_columns_for_points(weather, "u80", point_ids)].to_numpy(dtype=float)
                v80 = weather[_columns_for_points(weather, "v80", point_ids)].to_numpy(dtype=float)
                u120 = weather[_columns_for_points(weather, "u120", point_ids)].to_numpy(dtype=float)
                v120 = weather[_columns_for_points(weather, "v120", point_ids)].to_numpy(dtype=float)
                speed80 = np.sqrt(u80**2 + v80**2)
                speed120 = np.sqrt(u120**2 + v120**2)
                speed_ratio = np.divide(
                    speed120,
                    speed80,
                    out=np.ones_like(speed120, dtype=float),
                    where=speed80 > 1e-6,
                )
                frames[f"{prefix}_speed_80m_cap_weighted_m_s"] = pd.Series(
                    _weighted_mean(speed80, capacity),
                    index=weather.index,
                )
                frames[f"{prefix}_speed_120m_cap_weighted_m_s"] = pd.Series(
                    _weighted_mean(speed120, capacity),
                    index=weather.index,
                )
                frames[f"{prefix}_speed_120m_minus_80m_cap_weighted_m_s"] = pd.Series(
                    _weighted_mean(speed120 - speed80, capacity),
                    index=weather.index,
                )
                frames[f"{prefix}_speed_120m_to_80m_ratio_cap_weighted"] = pd.Series(
                    _weighted_mean(speed_ratio, capacity),
                    index=weather.index,
                )
                frames[f"{prefix}_proxy_v3_80m"] = pd.Series((capacity * speed80**3).sum(axis=1), index=weather.index)
                frames[f"{prefix}_proxy_v3_120m"] = pd.Series((capacity * speed120**3).sum(axis=1), index=weather.index)

                if config.include_wind_hub_height_interpolation_features:
                    hub_height = _static_point_metadata(
                        assigned_capacity,
                        point_ids=point_ids,
                        value_column=config.wind_hub_height_column,
                    )
                    hub_height = np.where(
                        np.isfinite(hub_height) & (hub_height > 0),
                        hub_height,
                        config.wind_hub_height_m,
                    )
                    speed_hub_actual = _interpolate_speed_to_hub_height(speed80, speed120, hub_height)
                    frames[f"{prefix}_hub_height_cap_weighted_m"] = pd.Series(
                        np.full(len(weather.index), _weighted_mean(hub_height[None, :], capacity[:1, :])[0]),
                        index=weather.index,
                    )
                    frames[f"{prefix}_speed_hub_actual_cap_weighted_m_s"] = pd.Series(
                        _weighted_mean(speed_hub_actual, capacity),
                        index=weather.index,
                    )
                    frames[f"{prefix}_speed_hub_actual_cap_weighted_std_m_s"] = pd.Series(
                        _weighted_std(speed_hub_actual, capacity),
                        index=weather.index,
                    )
                    frames[f"{prefix}_proxy_hub_actual_mw"] = pd.Series(
                        (capacity * _normalised_wind_power_curve(speed_hub_actual, config.wind_power_curve_smoothing_sigma_m_s)).sum(axis=1),
                        index=weather.index,
                    )
                    frames[f"{prefix}_proxy_v3_hub_actual"] = pd.Series(
                        (capacity * speed_hub_actual**3).sum(axis=1),
                        index=weather.index,
                    )
                    if config.wind_weather_weight_mode != "capacity":
                        frames[f"{prefix}_speed_hub_actual_{config.wind_weather_weight_mode}_weighted_m_s"] = pd.Series(
                            _weighted_mean(speed_hub_actual, wind_feature_weights),
                            index=weather.index,
                        )
                        frames[f"{prefix}_proxy_v3_hub_actual_{config.wind_weather_weight_mode}_weighted"] = pd.Series(
                            (wind_feature_weights * speed_hub_actual**3).sum(axis=1),
                            index=weather.index,
                        )

                if all(f"{wind_column}_point_{point_ids[0]}" in weather.columns for wind_column in ["u180", "v180"]):
                    u180 = weather[_columns_for_points(weather, "u180", point_ids)].to_numpy(dtype=float)
                    v180 = weather[_columns_for_points(weather, "v180", point_ids)].to_numpy(dtype=float)
                    speed180 = np.sqrt(u180**2 + v180**2)
                    speed180_ratio = np.divide(
                        speed180,
                        speed120,
                        out=np.ones_like(speed180, dtype=float),
                        where=speed120 > 1e-6,
                    )
                    frames[f"{prefix}_speed_180m_cap_weighted_m_s"] = pd.Series(
                        _weighted_mean(speed180, capacity),
                        index=weather.index,
                    )
                    frames[f"{prefix}_speed_180m_cap_weighted_std_m_s"] = pd.Series(
                        _weighted_std(speed180, capacity),
                        index=weather.index,
                    )
                    frames[f"{prefix}_speed_180m_minus_120m_cap_weighted_m_s"] = pd.Series(
                        _weighted_mean(speed180 - speed120, capacity),
                        index=weather.index,
                    )
                    frames[f"{prefix}_speed_180m_to_120m_ratio_cap_weighted"] = pd.Series(
                        _weighted_mean(speed180_ratio, capacity),
                        index=weather.index,
                    )
                    frames[f"{prefix}_proxy_v3_180m"] = pd.Series(
                        (capacity * speed180**3).sum(axis=1),
                        index=weather.index,
                    )

            t2m = None
            sp = None
            if f"t2m_point_{point_ids[0]}" in weather.columns:
                t2m = weather[_columns_for_points(weather, "t2m", point_ids)].to_numpy(dtype=float)
                frames[f"{prefix}_t2m_cap_weighted_K"] = pd.Series(_weighted_mean(t2m, capacity), index=weather.index)
                frames[f"{prefix}_t2m_cap_weighted_std_K"] = pd.Series(_weighted_std(t2m, capacity), index=weather.index)
            if f"td2m_point_{point_ids[0]}" in weather.columns:
                td2m = weather[_columns_for_points(weather, "td2m", point_ids)].to_numpy(dtype=float)
                frames[f"{prefix}_td2m_cap_weighted_K"] = pd.Series(_weighted_mean(td2m, capacity), index=weather.index)
            temperature_by_height: dict[int, np.ndarray] = {}
            for height in [80, 100, 120, 180]:
                temp_prefix = f"t{height}m"
                if f"{temp_prefix}_point_{point_ids[0]}" not in weather.columns:
                    continue
                temp = weather[_columns_for_points(weather, temp_prefix, point_ids)].to_numpy(dtype=float)
                temperature_by_height[height] = temp
                frames[f"{prefix}_t{height}m_cap_weighted_K"] = pd.Series(
                    _weighted_mean(temp, capacity),
                    index=weather.index,
                )
                frames[f"{prefix}_t{height}m_cap_weighted_std_K"] = pd.Series(
                    _weighted_std(temp, capacity),
                    index=weather.index,
                )
            for lower_height, upper_height in [(80, 120), (100, 180), (120, 180)]:
                if lower_height not in temperature_by_height or upper_height not in temperature_by_height:
                    continue
                delta = temperature_by_height[upper_height] - temperature_by_height[lower_height]
                frames[f"{prefix}_t{upper_height}m_minus_t{lower_height}m_cap_weighted_K"] = pd.Series(
                    _weighted_mean(delta, capacity),
                    index=weather.index,
                )
            if f"r2m_point_{point_ids[0]}" in weather.columns:
                r2m = weather[_columns_for_points(weather, "r2m", point_ids)].to_numpy(dtype=float)
                frames[f"{prefix}_relative_humidity_2m_cap_weighted_pct"] = pd.Series(
                    _weighted_mean(r2m, capacity),
                    index=weather.index,
                )
            if f"vpd_point_{point_ids[0]}" in weather.columns:
                vpd = weather[_columns_for_points(weather, "vpd", point_ids)].to_numpy(dtype=float)
                frames[f"{prefix}_vapour_pressure_deficit_cap_weighted_kpa"] = pd.Series(
                    _weighted_mean(vpd, capacity),
                    index=weather.index,
                )
            if f"sp_point_{point_ids[0]}" in weather.columns:
                sp = weather[_columns_for_points(weather, "sp", point_ids)].to_numpy(dtype=float)
                frames[f"{prefix}_sp_cap_weighted_Pa"] = pd.Series(_weighted_mean(sp, capacity), index=weather.index)
            if f"msl_point_{point_ids[0]}" in weather.columns:
                msl = weather[_columns_for_points(weather, "msl", point_ids)].to_numpy(dtype=float)
                frames[f"{prefix}_msl_cap_weighted_Pa"] = pd.Series(_weighted_mean(msl, capacity), index=weather.index)
            if t2m is not None and sp is not None:
                density = _air_density_kg_m3(sp, t2m)
                density_factor = density / 1.225
                frames[f"{prefix}_air_density_cap_weighted_kg_m3"] = pd.Series(
                    _weighted_mean(density, capacity),
                    index=weather.index,
                )
                frames[f"{prefix}_proxy_v3_air_density"] = pd.Series(
                    (capacity * speed**3 * density_factor).sum(axis=1),
                    index=weather.index,
                )
            if f"tp_point_{point_ids[0]}" in weather.columns:
                tp = weather[_columns_for_points(weather, "tp", point_ids)].to_numpy(dtype=float)
                frames[f"{prefix}_tot_prec_cap_weighted"] = pd.Series(_weighted_mean(tp, capacity), index=weather.index)
            if f"sde_point_{point_ids[0]}" in weather.columns:
                sde = weather[_columns_for_points(weather, "sde", point_ids)].to_numpy(dtype=float)
                frames[f"{prefix}_snow_depth_cap_weighted_m"] = pd.Series(_weighted_mean(sde, capacity), index=weather.index)
            if f"pblh_point_{point_ids[0]}" in weather.columns:
                pblh = weather[_columns_for_points(weather, "pblh", point_ids)].to_numpy(dtype=float)
                frames[f"{prefix}_boundary_layer_height_cap_weighted_m"] = pd.Series(
                    _weighted_mean(pblh, capacity),
                    index=weather.index,
                )
                frames[f"{prefix}_boundary_layer_height_cap_weighted_std_m"] = pd.Series(
                    _weighted_std(pblh, capacity),
                    index=weather.index,
                )
            if f"tcc_point_{point_ids[0]}" in weather.columns:
                tcc = weather[_columns_for_points(weather, "tcc", point_ids)].to_numpy(dtype=float)
                frames[f"{prefix}_cloud_cover_cap_weighted"] = pd.Series(_weighted_mean(tcc, capacity), index=weather.index)
            for cloud_prefix, feature_suffix in [
                ("lcc", "low_cloud_cover"),
                ("mcc", "mid_cloud_cover"),
                ("hcc", "high_cloud_cover"),
            ]:
                if f"{cloud_prefix}_point_{point_ids[0]}" not in weather.columns:
                    continue
                cloud = weather[_columns_for_points(weather, cloud_prefix, point_ids)].to_numpy(dtype=float)
                frames[f"{prefix}_{feature_suffix}_cap_weighted"] = pd.Series(
                    _weighted_mean(cloud, capacity),
                    index=weather.index,
                )

    for region in sorted(points.loc[points["technology_group"] == "solar", "region"].unique()):
        capacity_daily = _active_capacity_by_weather_point(
            assigned_capacity,
            weather.index,
            technology="solar",
            region=region,
        )
        point_ids = [
            int(point_id)
            for point_id in capacity_daily.columns
            if f"ssrd_point_{point_id}" in weather.columns
        ]
        if not point_ids:
            continue

        capacity = _capacity_for_timestamps(capacity_daily, weather.index, point_ids)
        ssrd = np.clip(weather[_columns_for_points(weather, "ssrd", point_ids)].to_numpy(dtype=float), 0.0, None)
        proxy = (capacity * ssrd / 1000.0 * config.solar_performance_ratio).sum(axis=1)

        prefix = f"solar_{region.removeprefix('solar_')}"
        if config.include_capacity_static_features or config.include_geographic_exposure_features:
            _add_capacity_static_features(
                frames,
                assigned_capacity,
                weather.index,
                config,
                technology="solar",
                region=region,
                prefix=prefix,
                group_column="weather_point_id",
            )
        frames[f"{prefix}_proxy_mw"] = pd.Series(proxy, index=weather.index)
        frames[f"{prefix}_irradiance_cap_weighted_W_m2"] = pd.Series(_weighted_mean(ssrd, capacity), index=weather.index)
        frames[f"{prefix}_irradiance_cap_weighted_std_W_m2"] = pd.Series(_weighted_std(ssrd, capacity), index=weather.index)
        solar_total = solar_total + frames[f"{prefix}_proxy_mw"]
        ramp_columns.append(f"{prefix}_proxy_mw")

        fdir = None
        if f"fdir_point_{point_ids[0]}" in weather.columns:
            fdir = np.clip(weather[_columns_for_points(weather, "fdir", point_ids)].to_numpy(dtype=float), 0.0, None)
            frames[f"{prefix}_direct_irradiance_cap_weighted_W_m2"] = pd.Series(_weighted_mean(fdir, capacity), index=weather.index)
        if f"dni_point_{point_ids[0]}" in weather.columns:
            dni = np.clip(weather[_columns_for_points(weather, "dni", point_ids)].to_numpy(dtype=float), 0.0, None)
            frames[f"{prefix}_direct_normal_irradiance_cap_weighted_W_m2"] = pd.Series(
                _weighted_mean(dni, capacity),
                index=weather.index,
            )
            frames[f"{prefix}_direct_normal_irradiance_cap_weighted_std_W_m2"] = pd.Series(
                _weighted_std(dni, capacity),
                index=weather.index,
            )
        if f"diffuse_point_{point_ids[0]}" in weather.columns:
            diffuse = np.clip(weather[_columns_for_points(weather, "diffuse", point_ids)].to_numpy(dtype=float), 0.0, None)
            frames[f"{prefix}_diffuse_irradiance_cap_weighted_W_m2"] = pd.Series(
                _weighted_mean(diffuse, capacity),
                index=weather.index,
            )
        elif fdir is not None:
            diffuse = np.clip(ssrd - fdir, 0.0, None)
        else:
            diffuse = None
        if diffuse is not None:
            diffuse_share = np.divide(
                diffuse,
                ssrd,
                out=np.zeros_like(diffuse, dtype=float),
                where=ssrd > 0,
            )
            frames[f"{prefix}_diffuse_share_cap_weighted"] = pd.Series(
                _weighted_mean(diffuse_share, capacity),
                index=weather.index,
            )
        if f"t2m_point_{point_ids[0]}" in weather.columns:
            t2m = weather[_columns_for_points(weather, "t2m", point_ids)].to_numpy(dtype=float)
            frames[f"{prefix}_t2m_cap_weighted_K"] = pd.Series(_weighted_mean(t2m, capacity), index=weather.index)
            frames[f"{prefix}_t2m_cap_weighted_std_K"] = pd.Series(_weighted_std(t2m, capacity), index=weather.index)
        if f"td2m_point_{point_ids[0]}" in weather.columns:
            td2m = weather[_columns_for_points(weather, "td2m", point_ids)].to_numpy(dtype=float)
            frames[f"{prefix}_td2m_cap_weighted_K"] = pd.Series(_weighted_mean(td2m, capacity), index=weather.index)
        if f"r2m_point_{point_ids[0]}" in weather.columns:
            r2m = weather[_columns_for_points(weather, "r2m", point_ids)].to_numpy(dtype=float)
            frames[f"{prefix}_relative_humidity_2m_cap_weighted_pct"] = pd.Series(
                _weighted_mean(r2m, capacity),
                index=weather.index,
            )
        if f"vpd_point_{point_ids[0]}" in weather.columns:
            vpd = weather[_columns_for_points(weather, "vpd", point_ids)].to_numpy(dtype=float)
            frames[f"{prefix}_vapour_pressure_deficit_cap_weighted_kpa"] = pd.Series(
                _weighted_mean(vpd, capacity),
                index=weather.index,
            )
        if f"sp_point_{point_ids[0]}" in weather.columns:
            sp = weather[_columns_for_points(weather, "sp", point_ids)].to_numpy(dtype=float)
            frames[f"{prefix}_sp_cap_weighted_Pa"] = pd.Series(_weighted_mean(sp, capacity), index=weather.index)
        if f"msl_point_{point_ids[0]}" in weather.columns:
            msl = weather[_columns_for_points(weather, "msl", point_ids)].to_numpy(dtype=float)
            frames[f"{prefix}_msl_cap_weighted_Pa"] = pd.Series(_weighted_mean(msl, capacity), index=weather.index)
        if f"tp_point_{point_ids[0]}" in weather.columns:
            tp = weather[_columns_for_points(weather, "tp", point_ids)].to_numpy(dtype=float)
            frames[f"{prefix}_tot_prec_cap_weighted"] = pd.Series(_weighted_mean(tp, capacity), index=weather.index)
        if f"sde_point_{point_ids[0]}" in weather.columns:
            sde = weather[_columns_for_points(weather, "sde", point_ids)].to_numpy(dtype=float)
            frames[f"{prefix}_snow_depth_cap_weighted_m"] = pd.Series(_weighted_mean(sde, capacity), index=weather.index)
        if f"tcc_point_{point_ids[0]}" in weather.columns:
            tcc = weather[_columns_for_points(weather, "tcc", point_ids)].to_numpy(dtype=float)
            frames[f"{prefix}_cloud_cover_cap_weighted"] = pd.Series(_weighted_mean(tcc, capacity), index=weather.index)
        for cloud_prefix, feature_suffix in [
            ("lcc", "low_cloud_cover"),
            ("mcc", "mid_cloud_cover"),
            ("hcc", "high_cloud_cover"),
        ]:
            if f"{cloud_prefix}_point_{point_ids[0]}" not in weather.columns:
                continue
            cloud = weather[_columns_for_points(weather, cloud_prefix, point_ids)].to_numpy(dtype=float)
            frames[f"{prefix}_{feature_suffix}_cap_weighted"] = pd.Series(
                _weighted_mean(cloud, capacity),
                index=weather.index,
            )

    result = pd.DataFrame(frames, index=weather.index)
    result["Renewable_Wind_Proxy_MW"] = wind_total
    result["Renewable_Solar_Proxy_MW"] = solar_total
    result["Renewable_Total_Proxy_MW"] = wind_total + solar_total
    ramp_columns.extend(["Renewable_Wind_Proxy_MW", "Renewable_Solar_Proxy_MW", "Renewable_Total_Proxy_MW"])
    if config.include_ramps:
        result = _add_ramps(result, ramp_columns)

    end_time = result.index.max()
    if end_time.minute == 0 and len(result.index) > 1:
        observed_steps = result.index.to_series().diff().dropna()
        if not observed_steps.empty and observed_steps.min() >= pd.Timedelta(hours=1):
            end_time = end_time + pd.Timedelta(minutes=45)

    full_index = pd.date_range(
        start=result.index.min(),
        end=end_time,
        freq="15min",
        tz=result.index.tz,
        name="timestamp",
    )
    result = result.loc[~result.index.duplicated(keep="last")].reindex(full_index).ffill(limit=3)
    result.index.name = "timestamp"
    return result.astype(float)


def _build_cluster_cloud_cover_features(
    capacity_map: pd.DataFrame,
    weather: pd.DataFrame,
    config: RegionalRenewableFeatureConfig,
) -> pd.DataFrame:
    """Build compact capacity-weighted regional cloud-cover features from cluster weather."""
    cloud_prefixes = [
        ("tcc", "cloud_cover"),
        ("lcc", "low_cloud_cover"),
        ("mcc", "mid_cloud_cover"),
        ("hcc", "high_cloud_cover"),
    ]
    available_cloud_prefixes = [
        (cloud_prefix, feature_name)
        for cloud_prefix, feature_name in cloud_prefixes
        if any(column.startswith(f"{cloud_prefix}_cluster_") for column in weather.columns)
    ]
    if not available_cloud_prefixes:
        raise ValueError("Cloud-cover feature mode requires at least one *_cluster_* cloud-cover weather column.")

    frames: dict[str, pd.Series] = {}
    ramp_columns: list[str] = []
    for technology in sorted(capacity_map["technology_group"].unique()):
        if technology not in {"solar", "wind_onshore", "wind_offshore"}:
            continue
        for region in sorted(capacity_map.loc[capacity_map["technology_group"] == technology, "region"].unique()):
            capacity_daily = _active_capacity_by_cluster(capacity_map, weather.index, technology=technology, region=region)
            clusters = [
                cluster_id
                for cluster_id in capacity_daily.columns
                if any(f"{cloud_prefix}_cluster_{cluster_id}" in weather.columns for cloud_prefix, _ in available_cloud_prefixes)
            ]
            if not clusters:
                continue

            capacity = _capacity_for_timestamps(capacity_daily, weather.index, clusters)
            prefix = f"{technology}_{region.removeprefix('solar_').removeprefix('onshore_').removeprefix('offshore_')}"
            if technology == "wind_offshore":
                prefix = f"wind_{region}"
            elif technology == "wind_onshore":
                prefix = f"wind_{region}"

            for cloud_prefix, feature_name in available_cloud_prefixes:
                cloud_clusters = [cluster_id for cluster_id in clusters if f"{cloud_prefix}_cluster_{cluster_id}" in weather.columns]
                if not cloud_clusters:
                    continue
                cloud_capacity = _capacity_for_timestamps(capacity_daily, weather.index, cloud_clusters)
                cloud = weather[_columns_for_clusters(weather, cloud_prefix, cloud_clusters)].to_numpy(dtype=float)
                mean_name = f"{prefix}_{feature_name}_cap_weighted"
                frames[mean_name] = pd.Series(_weighted_mean(cloud, cloud_capacity), index=weather.index)
                frames[f"{prefix}_{feature_name}_cap_weighted_std"] = pd.Series(
                    _weighted_std(cloud, cloud_capacity),
                    index=weather.index,
                )
                frames[f"{prefix}_{feature_name}_cap_weighted_min"] = pd.Series(np.nanmin(cloud, axis=1), index=weather.index)
                frames[f"{prefix}_{feature_name}_cap_weighted_max"] = pd.Series(np.nanmax(cloud, axis=1), index=weather.index)
                ramp_columns.append(mean_name)

    if not frames:
        raise ValueError("No cloud-cover regional features could be built.")

    result = pd.DataFrame(frames, index=weather.index).sort_index()
    if config.include_ramps:
        result = _add_ramps(result, ramp_columns)
    if len(result.index) > 1:
        observed_steps = result.index.to_series().diff().dropna()
        if not observed_steps.empty and observed_steps.min() >= pd.Timedelta(hours=1):
            full_index = pd.date_range(
                start=result.index.min(),
                end=result.index.max() + pd.Timedelta(minutes=45),
                freq="15min",
                tz=result.index.tz,
                name="timestamp",
            )
            result = result.loc[~result.index.duplicated(keep="last")].reindex(full_index).ffill(limit=3)
    result.index.name = "timestamp"
    return result.astype(float)


def build_regional_renewable_features(
    config: RegionalRenewableFeatureConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    capacity_map = build_capacity_map(config)
    if config.weather_source == "open_meteo" and config.open_meteo_point_source == "capacity":
        capacity_points = build_capacity_weather_points(capacity_map, config)
        weather = _load_open_meteo_capacity_point_weather(config, capacity_points)
        features = _build_capacity_point_renewable_features(capacity_map, capacity_points, weather, config)
        return features, capacity_map, capacity_points

    weather = _load_weather(config)
    if config.feature_mode == "cloud_cover":
        features = _build_cluster_cloud_cover_features(capacity_map, weather, config)
        return features, capacity_map, None

    if any(column.startswith("u100_cluster_") for column in weather.columns):
        u_prefix = "u100"
        v_prefix = "v100"
    elif any(column.startswith("u10_cluster_") for column in weather.columns):
        u_prefix = "u10"
        v_prefix = "v10"
    elif any(column.startswith("u10_ensmean_cluster_") for column in weather.columns):
        u_prefix = "u10_ensmean"
        v_prefix = "v10_ensmean"
    else:
        u_prefix = "u10"
        v_prefix = "v10"
    wind_factor = 1.0
    if u_prefix.startswith("u10"):
        wind_factor = (config.wind_hub_height_m / config.wind_reference_height_m) ** config.wind_shear_alpha

    frames: dict[str, pd.Series] = {}
    ramp_columns: list[str] = []
    wind_total = pd.Series(0.0, index=weather.index)
    solar_total = pd.Series(0.0, index=weather.index)

    for technology in ["wind_offshore", "wind_onshore"]:
        for region in sorted(capacity_map.loc[capacity_map["technology_group"] == technology, "region"].unique()):
            capacity_daily = _active_capacity_by_cluster(capacity_map, weather.index, technology=technology, region=region)
            clusters = [
                cluster_id
                for cluster_id in capacity_daily.columns
                if f"{u_prefix}_cluster_{cluster_id}" in weather.columns and f"{v_prefix}_cluster_{cluster_id}" in weather.columns
            ]
            if not clusters:
                continue

            capacity = _capacity_for_timestamps(capacity_daily, weather.index, clusters)
            u = weather[_columns_for_clusters(weather, u_prefix, clusters)].to_numpy(dtype=float)
            v = weather[_columns_for_clusters(weather, v_prefix, clusters)].to_numpy(dtype=float)
            speed = np.sqrt(u**2 + v**2) * wind_factor
            proxy_curve = (capacity * _normalised_wind_power_curve(speed, config.wind_power_curve_smoothing_sigma_m_s)).sum(axis=1)
            proxy_v3 = (capacity * speed**3).sum(axis=1)

            prefix = f"wind_{region}"
            if config.include_capacity_static_features or config.include_geographic_exposure_features:
                _add_capacity_static_features(
                    frames,
                    capacity_map,
                    weather.index,
                    config,
                    technology=technology,
                    region=region,
                    prefix=prefix,
                    group_column="cluster_id",
                )
            frames[f"{prefix}_proxy_mw"] = pd.Series(proxy_curve, index=weather.index)
            frames[f"{prefix}_proxy_v3"] = pd.Series(proxy_v3, index=weather.index)
            frames[f"{prefix}_speed_hub_cap_weighted_m_s"] = pd.Series(_weighted_mean(speed, capacity), index=weather.index)
            frames[f"{prefix}_u_cap_weighted_m_s"] = pd.Series(_weighted_mean(u, capacity), index=weather.index)
            frames[f"{prefix}_v_cap_weighted_m_s"] = pd.Series(_weighted_mean(v, capacity), index=weather.index)
            direction_cos, direction_sin = _weighted_wind_direction_components(u, v, capacity)
            frames[f"{prefix}_direction_cos_cap_weighted"] = pd.Series(direction_cos, index=weather.index)
            frames[f"{prefix}_direction_sin_cap_weighted"] = pd.Series(direction_sin, index=weather.index)
            wind_total = wind_total + frames[f"{prefix}_proxy_mw"]
            ramp_columns.append(f"{prefix}_proxy_mw")

            if u_prefix.endswith("_ensmean"):
                u_std_prefix = u_prefix.replace("_ensmean", "_ensstd")
                v_std_prefix = v_prefix.replace("_ensmean", "_ensstd")
                if all(
                    f"{u_std_prefix}_cluster_{cluster_id}" in weather.columns
                    and f"{v_std_prefix}_cluster_{cluster_id}" in weather.columns
                    for cluster_id in clusters
                ):
                    u_std = weather[_columns_for_clusters(weather, u_std_prefix, clusters)].to_numpy(dtype=float)
                    v_std = weather[_columns_for_clusters(weather, v_std_prefix, clusters)].to_numpy(dtype=float)
                    component_std = np.sqrt(np.clip(u_std, 0.0, None) ** 2 + np.clip(v_std, 0.0, None) ** 2) * wind_factor
                    frames[f"{prefix}_eps_component_std_cap_weighted_m_s"] = pd.Series(
                        _weighted_mean(component_std, capacity),
                        index=weather.index,
                    )
                    frames[f"{prefix}_eps_component_std_cap_weighted_std_m_s"] = pd.Series(
                        _weighted_std(component_std, capacity),
                        index=weather.index,
                    )
                    frames[f"{prefix}_eps_proxy_v3_uncertainty"] = pd.Series(
                        (capacity * component_std**3).sum(axis=1),
                        index=weather.index,
                    )

            _add_wind_model_level_features(
                frames,
                ramp_columns,
                prefix=prefix,
                weather=weather,
                capacity=capacity,
                clusters=clusters,
                smoothing_sigma_m_s=config.wind_power_curve_smoothing_sigma_m_s,
            )

            t2m = None
            sp = None
            if "t2m_cluster_0" in weather.columns:
                t2m = weather[_columns_for_clusters(weather, "t2m", clusters)].to_numpy(dtype=float)
                frames[f"{prefix}_t2m_cap_weighted_K"] = pd.Series(_weighted_mean(t2m, capacity), index=weather.index)
            if "td2m_cluster_0" in weather.columns:
                td2m = weather[_columns_for_clusters(weather, "td2m", clusters)].to_numpy(dtype=float)
                frames[f"{prefix}_td2m_cap_weighted_K"] = pd.Series(_weighted_mean(td2m, capacity), index=weather.index)
            if "r2m_cluster_0" in weather.columns:
                r2m = weather[_columns_for_clusters(weather, "r2m", clusters)].to_numpy(dtype=float)
                frames[f"{prefix}_relative_humidity_2m_cap_weighted_pct"] = pd.Series(
                    _weighted_mean(r2m, capacity),
                    index=weather.index,
                )
            if "vpd_cluster_0" in weather.columns:
                vpd = weather[_columns_for_clusters(weather, "vpd", clusters)].to_numpy(dtype=float)
                frames[f"{prefix}_vapour_pressure_deficit_cap_weighted_kpa"] = pd.Series(
                    _weighted_mean(vpd, capacity),
                    index=weather.index,
                )
            if "sp_cluster_0" in weather.columns:
                sp = weather[_columns_for_clusters(weather, "sp", clusters)].to_numpy(dtype=float)
                frames[f"{prefix}_sp_cap_weighted_Pa"] = pd.Series(_weighted_mean(sp, capacity), index=weather.index)
            if "msl_cluster_0" in weather.columns:
                msl = weather[_columns_for_clusters(weather, "msl", clusters)].to_numpy(dtype=float)
                frames[f"{prefix}_msl_cap_weighted_Pa"] = pd.Series(_weighted_mean(msl, capacity), index=weather.index)
            if t2m is not None and sp is not None:
                density = _air_density_kg_m3(sp, t2m)
                density_factor = density / 1.225
                frames[f"{prefix}_air_density_cap_weighted_kg_m3"] = pd.Series(
                    _weighted_mean(density, capacity),
                    index=weather.index,
                )
                frames[f"{prefix}_proxy_v3_air_density"] = pd.Series(
                    (capacity * speed**3 * density_factor).sum(axis=1),
                    index=weather.index,
                )
            if "vmax10m_cluster_0" in weather.columns:
                vmax10 = np.clip(weather[_columns_for_clusters(weather, "vmax10m", clusters)].to_numpy(dtype=float), 0.0, None)
                vmax_hub = vmax10 * (config.wind_hub_height_m / config.wind_reference_height_m) ** config.wind_shear_alpha
                frames[f"{prefix}_vmax10m_cap_weighted_m_s"] = pd.Series(
                    _weighted_mean(vmax10, capacity),
                    index=weather.index,
                )
                frames[f"{prefix}_vmax_hub_cap_weighted_m_s"] = pd.Series(
                    _weighted_mean(vmax_hub, capacity),
                    index=weather.index,
                )
                frames[f"{prefix}_proxy_vmax_v3"] = pd.Series((capacity * vmax_hub**3).sum(axis=1), index=weather.index)
            if "tp_cluster_0" in weather.columns:
                tp = weather[_columns_for_clusters(weather, "tp", clusters)].to_numpy(dtype=float)
                frames[f"{prefix}_tot_prec_cap_weighted"] = pd.Series(_weighted_mean(tp, capacity), index=weather.index)
            if "sde_cluster_0" in weather.columns:
                sde = weather[_columns_for_clusters(weather, "sde", clusters)].to_numpy(dtype=float)
                frames[f"{prefix}_snow_depth_cap_weighted_m"] = pd.Series(_weighted_mean(sde, capacity), index=weather.index)
            if "snow_gsp_cluster_0" in weather.columns:
                snow_gsp = weather[_columns_for_clusters(weather, "snow_gsp", clusters)].to_numpy(dtype=float)
                frames[f"{prefix}_snow_gsp_cap_weighted"] = pd.Series(_weighted_mean(snow_gsp, capacity), index=weather.index)
            if "tcc_cluster_0" in weather.columns:
                tcc = weather[_columns_for_clusters(weather, "tcc", clusters)].to_numpy(dtype=float)
                frames[f"{prefix}_cloud_cover_cap_weighted"] = pd.Series(_weighted_mean(tcc, capacity), index=weather.index)
            for cloud_prefix, feature_suffix in [
                ("lcc", "low_cloud_cover"),
                ("mcc", "mid_cloud_cover"),
                ("hcc", "high_cloud_cover"),
            ]:
                if f"{cloud_prefix}_cluster_0" not in weather.columns:
                    continue
                cloud = weather[_columns_for_clusters(weather, cloud_prefix, clusters)].to_numpy(dtype=float)
                frames[f"{prefix}_{feature_suffix}_cap_weighted"] = pd.Series(
                    _weighted_mean(cloud, capacity),
                    index=weather.index,
                )

    if config.include_cluster_features:
        for technology in ["wind_offshore", "wind_onshore"]:
            _add_wind_cluster_features(
                frames,
                ramp_columns,
                capacity_map,
                weather,
                config,
                technology=technology,
                u_prefix=u_prefix,
                v_prefix=v_prefix,
                wind_factor=wind_factor,
            )

    for region in sorted(capacity_map.loc[capacity_map["technology_group"] == "solar", "region"].unique()):
        capacity_daily = _active_capacity_by_cluster(capacity_map, weather.index, technology="solar", region=region)
        clusters = [cluster_id for cluster_id in capacity_daily.columns if f"ssrd_cluster_{cluster_id}" in weather.columns]
        if not clusters:
            continue

        capacity = _capacity_for_timestamps(capacity_daily, weather.index, clusters)
        ssrd = np.clip(weather[_columns_for_clusters(weather, "ssrd", clusters)].to_numpy(dtype=float), 0.0, None)
        proxy = (capacity * ssrd / 1000.0 * config.solar_performance_ratio).sum(axis=1)

        prefix = f"solar_{region.removeprefix('solar_')}"
        if config.include_capacity_static_features or config.include_geographic_exposure_features:
            _add_capacity_static_features(
                frames,
                capacity_map,
                weather.index,
                config,
                technology="solar",
                region=region,
                prefix=prefix,
                group_column="cluster_id",
            )
        frames[f"{prefix}_proxy_mw"] = pd.Series(proxy, index=weather.index)
        frames[f"{prefix}_irradiance_cap_weighted_W_m2"] = pd.Series(_weighted_mean(ssrd, capacity), index=weather.index)
        if config.include_spatial_spread_features:
            frames[f"{prefix}_irradiance_cap_weighted_std_W_m2"] = pd.Series(
                _weighted_std(ssrd, capacity),
                index=weather.index,
            )
        solar_total = solar_total + frames[f"{prefix}_proxy_mw"]
        ramp_columns.append(f"{prefix}_proxy_mw")

        if "fdir_cluster_0" in weather.columns:
            fdir = np.clip(weather[_columns_for_clusters(weather, "fdir", clusters)].to_numpy(dtype=float), 0.0, None)
            frames[f"{prefix}_direct_irradiance_cap_weighted_W_m2"] = pd.Series(_weighted_mean(fdir, capacity), index=weather.index)
            diffuse = np.clip(ssrd - fdir, 0.0, None)
            if config.include_spatial_spread_features:
                frames[f"{prefix}_direct_irradiance_cap_weighted_std_W_m2"] = pd.Series(
                    _weighted_std(fdir, capacity),
                    index=weather.index,
                )
                frames[f"{prefix}_diffuse_irradiance_cap_weighted_W_m2"] = pd.Series(
                    _weighted_mean(diffuse, capacity),
                    index=weather.index,
                )
                frames[f"{prefix}_diffuse_irradiance_cap_weighted_std_W_m2"] = pd.Series(
                    _weighted_std(diffuse, capacity),
                    index=weather.index,
                )
            diffuse_share = np.divide(
                diffuse,
                ssrd,
                out=np.zeros_like(diffuse, dtype=float),
                where=ssrd > 0,
            )
            frames[f"{prefix}_diffuse_share_cap_weighted"] = pd.Series(
                _weighted_mean(diffuse_share, capacity),
                index=weather.index,
            )
            if config.include_spatial_spread_features:
                frames[f"{prefix}_diffuse_share_cap_weighted_std"] = pd.Series(
                    _weighted_std(diffuse_share, capacity),
                    index=weather.index,
                )
        if "dni_cluster_0" in weather.columns:
            dni = np.clip(weather[_columns_for_clusters(weather, "dni", clusters)].to_numpy(dtype=float), 0.0, None)
            frames[f"{prefix}_direct_normal_irradiance_cap_weighted_W_m2"] = pd.Series(
                _weighted_mean(dni, capacity),
                index=weather.index,
            )
            if config.include_spatial_spread_features:
                frames[f"{prefix}_direct_normal_irradiance_cap_weighted_std_W_m2"] = pd.Series(
                    _weighted_std(dni, capacity),
                    index=weather.index,
                )
        if "t2m_cluster_0" in weather.columns:
            t2m = weather[_columns_for_clusters(weather, "t2m", clusters)].to_numpy(dtype=float)
            frames[f"{prefix}_t2m_cap_weighted_K"] = pd.Series(_weighted_mean(t2m, capacity), index=weather.index)
            if config.include_spatial_spread_features:
                frames[f"{prefix}_t2m_cap_weighted_std_K"] = pd.Series(_weighted_std(t2m, capacity), index=weather.index)
        if "td2m_cluster_0" in weather.columns:
            td2m = weather[_columns_for_clusters(weather, "td2m", clusters)].to_numpy(dtype=float)
            frames[f"{prefix}_td2m_cap_weighted_K"] = pd.Series(_weighted_mean(td2m, capacity), index=weather.index)
            if config.include_spatial_spread_features:
                frames[f"{prefix}_td2m_cap_weighted_std_K"] = pd.Series(_weighted_std(td2m, capacity), index=weather.index)
        if "r2m_cluster_0" in weather.columns:
            r2m = weather[_columns_for_clusters(weather, "r2m", clusters)].to_numpy(dtype=float)
            frames[f"{prefix}_relative_humidity_2m_cap_weighted_pct"] = pd.Series(
                _weighted_mean(r2m, capacity),
                index=weather.index,
            )
        if "vpd_cluster_0" in weather.columns:
            vpd = weather[_columns_for_clusters(weather, "vpd", clusters)].to_numpy(dtype=float)
            frames[f"{prefix}_vapour_pressure_deficit_cap_weighted_kpa"] = pd.Series(
                _weighted_mean(vpd, capacity),
                index=weather.index,
            )
        if "sp_cluster_0" in weather.columns:
            sp = weather[_columns_for_clusters(weather, "sp", clusters)].to_numpy(dtype=float)
            frames[f"{prefix}_sp_cap_weighted_Pa"] = pd.Series(_weighted_mean(sp, capacity), index=weather.index)
        if "msl_cluster_0" in weather.columns:
            msl = weather[_columns_for_clusters(weather, "msl", clusters)].to_numpy(dtype=float)
            frames[f"{prefix}_msl_cap_weighted_Pa"] = pd.Series(_weighted_mean(msl, capacity), index=weather.index)
        if "tp_cluster_0" in weather.columns:
            tp = weather[_columns_for_clusters(weather, "tp", clusters)].to_numpy(dtype=float)
            frames[f"{prefix}_tot_prec_cap_weighted"] = pd.Series(_weighted_mean(tp, capacity), index=weather.index)
            if config.include_spatial_spread_features:
                frames[f"{prefix}_tot_prec_cap_weighted_std"] = pd.Series(_weighted_std(tp, capacity), index=weather.index)
        if "sde_cluster_0" in weather.columns:
            sde = weather[_columns_for_clusters(weather, "sde", clusters)].to_numpy(dtype=float)
            frames[f"{prefix}_snow_depth_cap_weighted_m"] = pd.Series(_weighted_mean(sde, capacity), index=weather.index)
            if config.include_spatial_spread_features:
                frames[f"{prefix}_snow_depth_cap_weighted_std_m"] = pd.Series(_weighted_std(sde, capacity), index=weather.index)
        if "snow_gsp_cluster_0" in weather.columns:
            snow_gsp = weather[_columns_for_clusters(weather, "snow_gsp", clusters)].to_numpy(dtype=float)
            frames[f"{prefix}_snow_gsp_cap_weighted"] = pd.Series(_weighted_mean(snow_gsp, capacity), index=weather.index)
            if config.include_spatial_spread_features:
                frames[f"{prefix}_snow_gsp_cap_weighted_std"] = pd.Series(
                    _weighted_std(snow_gsp, capacity),
                    index=weather.index,
                )
        if "tcc_cluster_0" in weather.columns:
            tcc = weather[_columns_for_clusters(weather, "tcc", clusters)].to_numpy(dtype=float)
            frames[f"{prefix}_cloud_cover_cap_weighted"] = pd.Series(_weighted_mean(tcc, capacity), index=weather.index)
            if config.include_spatial_spread_features:
                frames[f"{prefix}_cloud_cover_cap_weighted_std"] = pd.Series(_weighted_std(tcc, capacity), index=weather.index)
        for cloud_prefix, feature_suffix in [
            ("lcc", "low_cloud_cover"),
            ("mcc", "mid_cloud_cover"),
            ("hcc", "high_cloud_cover"),
        ]:
            if f"{cloud_prefix}_cluster_0" not in weather.columns:
                continue
            cloud = weather[_columns_for_clusters(weather, cloud_prefix, clusters)].to_numpy(dtype=float)
            frames[f"{prefix}_{feature_suffix}_cap_weighted"] = pd.Series(_weighted_mean(cloud, capacity), index=weather.index)
            if config.include_spatial_spread_features:
                frames[f"{prefix}_{feature_suffix}_cap_weighted_std"] = pd.Series(
                    _weighted_std(cloud, capacity),
                    index=weather.index,
                )

    if config.include_cluster_features:
        _add_solar_cluster_features(frames, ramp_columns, capacity_map, weather, config)

    result = pd.DataFrame(frames, index=weather.index)
    result["Renewable_Wind_Proxy_MW"] = wind_total
    result["Renewable_Solar_Proxy_MW"] = solar_total
    result["Renewable_Total_Proxy_MW"] = wind_total + solar_total
    ramp_columns.extend(["Renewable_Wind_Proxy_MW", "Renewable_Solar_Proxy_MW", "Renewable_Total_Proxy_MW"])
    if config.include_ramps:
        result = _add_ramps(result, ramp_columns)

    end_time = result.index.max()
    if end_time.minute == 0 and len(result.index) > 1:
        observed_steps = result.index.to_series().diff().dropna()
        if not observed_steps.empty and observed_steps.min() >= pd.Timedelta(hours=1):
            end_time = end_time + pd.Timedelta(minutes=45)

    full_index = pd.date_range(
        start=result.index.min(),
        end=end_time,
        freq="15min",
        tz=result.index.tz,
        name="timestamp",
    )
    result = result.loc[~result.index.duplicated(keep="last")].reindex(full_index).ffill(limit=3)
    result.index.name = "timestamp"
    return result.astype(float), capacity_map, None


def run_regional_renewable_features(config: RegionalRenewableFeatureConfig) -> pd.DataFrame:
    load_dotenv(config.repo_root / ".env")
    features, capacity_map, capacity_points = build_regional_renewable_features(config)
    config.output_file.parent.mkdir(parents=True, exist_ok=True)
    config.capacity_map_file.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(config.output_file)
    capacity_map.to_csv(config.capacity_map_file, index=False)
    if capacity_points is not None:
        config.capacity_weather_point_file.parent.mkdir(parents=True, exist_ok=True)
        capacity_points.to_csv(config.capacity_weather_point_file, index=False)
    if config.capacity_timeseries_file is not None:
        capacity_timeseries = build_capacity_timeseries_artifact(capacity_map, features.index, config)
        config.capacity_timeseries_file.parent.mkdir(parents=True, exist_ok=True)
        capacity_timeseries.to_csv(config.capacity_timeseries_file, index=False)
    if config.weather_weights_file is not None:
        weather_weights = build_weather_weights_artifact(capacity_map, features.index, config)
        config.weather_weights_file.parent.mkdir(parents=True, exist_ok=True)
        weather_weights.to_csv(config.weather_weights_file, index=False)

    summary = (
        capacity_map
        .groupby(["technology_group", "region"], as_index=False)["capacity_mw"]
        .sum()
        .sort_values(["technology_group", "region"])
    )
    metadata = {
        "era5_dirs": [str(path) for path in config.era5_dirs],
        "weather_source": config.weather_source,
        "icon_dir": str(config.icon_dir) if config.weather_source == "dwd_icon" else None,
        "required_run": config.required_run if config.weather_source == "dwd_icon" else None,
        "open_meteo_weather_file": str(config.open_meteo_weather_file) if config.weather_source == "open_meteo" else None,
        "open_meteo_base_url": config.open_meteo_base_url if config.weather_source == "open_meteo" else None,
        "open_meteo_model": config.open_meteo_model if config.weather_source == "open_meteo" else None,
        "open_meteo_api_mode": config.open_meteo_api_mode if config.weather_source == "open_meteo" else None,
        "open_meteo_single_run_hour_utc": (
            config.open_meteo_single_run_hour_utc if config.weather_source == "open_meteo" else None
        ),
        "open_meteo_single_run_forecast_days": (
            config.open_meteo_single_run_forecast_days if config.weather_source == "open_meteo" else None
        ),
        "open_meteo_request_pause_seconds": (
            config.open_meteo_request_pause_seconds if config.weather_source == "open_meteo" else None
        ),
        "open_meteo_point_selection": config.open_meteo_point_selection if config.weather_source == "open_meteo" else None,
        "open_meteo_point_source": config.open_meteo_point_source if config.weather_source == "open_meteo" else None,
        "capacity_weather_point_strategy": (
            config.capacity_weather_point_strategy
            if config.weather_source == "open_meteo" and config.open_meteo_point_source == "capacity"
            else None
        ),
        "capacity_weather_cell_size_degrees": (
            config.capacity_weather_cell_size_degrees
            if config.weather_source == "open_meteo" and config.open_meteo_point_source == "capacity"
            else None
        ),
        "capacity_weather_min_cell_capacity_mw": (
            config.capacity_weather_min_cell_capacity_mw
            if config.weather_source == "open_meteo" and config.open_meteo_point_source == "capacity"
            else None
        ),
        "open_meteo_max_points_per_cluster": (
            config.open_meteo_max_points_per_cluster if config.weather_source == "open_meteo" else None
        ),
        "capacity_weather_point_file": (
            str(config.capacity_weather_point_file)
            if config.weather_source == "open_meteo" and config.open_meteo_point_source == "capacity"
            else None
        ),
        "capacity_weather_point_count": int(len(capacity_points)) if capacity_points is not None else None,
        "open_meteo_start_date": (
            config.open_meteo_start_date.isoformat() if config.weather_source == "open_meteo" else None
        ),
        "open_meteo_end_date": (
            config.open_meteo_end_date.isoformat() if config.weather_source == "open_meteo" else None
        ),
        "capacity_file": str(config.capacity_file),
        "cluster_file": str(config.cluster_file),
        "solar_region_strategy": config.solar_region_strategy,
        "federal_state_column": config.federal_state_column,
        "include_cluster_features": config.include_cluster_features,
        "include_capacity_static_features": config.include_capacity_static_features,
        "include_geographic_exposure_features": config.include_geographic_exposure_features,
        "capacity_map_file": str(config.capacity_map_file),
        "capacity_timeseries_file": str(config.capacity_timeseries_file) if config.capacity_timeseries_file else None,
        "weather_weights_file": str(config.weather_weights_file) if config.weather_weights_file else None,
        "capacity_artifact_technologies": config.capacity_artifact_technologies,
        "output_file": str(config.output_file),
        "capacity_summary_mw": summary.to_dict(orient="records"),
    }
    with open(config.output_file.with_suffix(".json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(summary.to_string(index=False))
    print(f"Saved regional renewable features: {config.output_file}")
    print(f"Saved regional capacity map: {config.capacity_map_file}")
    if config.capacity_timeseries_file is not None:
        print(f"Saved capacity time series: {config.capacity_timeseries_file}")
    if config.weather_weights_file is not None:
        print(f"Saved weather weights: {config.weather_weights_file}")
    if capacity_points is not None:
        print(f"Saved capacity-weighted weather points: {config.capacity_weather_point_file}")
    return features
