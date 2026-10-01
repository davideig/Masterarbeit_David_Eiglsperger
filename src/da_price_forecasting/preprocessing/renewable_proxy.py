from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from ..config import RenewableProxyConfig
from ..data.weather import load_dwd


def _normalise_label(value: object) -> str:
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


def _load_cluster_assignments(cluster_file: Path) -> pd.DataFrame:
    if cluster_file.suffix.lower() == ".parquet":
        try:
            clusters = pd.read_parquet(cluster_file)
        except ImportError as exc:
            raise ImportError(
                "Reading Parquet cluster files requires pyarrow or fastparquet. "
                "Run this preprocessing step in the `core` environment or provide a CSV cluster file."
            ) from exc
    else:
        clusters = pd.read_csv(cluster_file)

    required = {"lon", "lat", "cluster_id"}
    missing = required - set(clusters.columns)
    if missing:
        raise ValueError(f"Cluster file is missing required columns: {sorted(missing)}")

    clusters = clusters[["lon", "lat", "cluster_id"]].copy()
    clusters["cluster_id"] = clusters["cluster_id"].astype(int)
    return clusters.dropna()


def _capacity_unit_factor(unit: str) -> float:
    if unit == "MW":
        return 1.0
    if unit == "kW":
        return 1.0 / 1000.0
    raise ValueError(f"Unsupported capacity unit: {unit!r}")


def _infer_cluster_file(config: RenewableProxyConfig) -> Path:
    if config.cluster_file is not None:
        return config.cluster_file

    folder_name = config.icon_dir.name
    if "_c" in folder_name:
        cluster_suffix = folder_name.rsplit("_c", 1)[-1]
        if cluster_suffix.isdigit():
            candidate = config.repo_root / "data" / "clustering" / f"icon_d2_clustering_c{cluster_suffix}.parquet"
            if candidate.exists():
                return candidate

    raise ValueError(
        "`cluster_file` is required unless it can be inferred from icon_dir like "
        "`data/processed/icon_aggregated_c5`."
    )


def _load_capacity_by_cluster(config: RenewableProxyConfig) -> pd.DataFrame:
    capacity = pd.read_csv(config.capacity_file)
    required = {config.technology_column, config.capacity_column}
    if config.capacity_format == "plant_locations":
        required |= {config.latitude_column, config.longitude_column}
    else:
        required |= {config.cluster_id_column}

    missing = required - set(capacity.columns)
    if missing:
        raise ValueError(f"Capacity file is missing required columns: {sorted(missing)}")

    capacity = capacity.copy()
    capacity["_technology"] = capacity[config.technology_column].map(_normalise_label)
    capacity["_capacity_mw"] = (
        pd.to_numeric(capacity[config.capacity_column], errors="coerce") * _capacity_unit_factor(config.capacity_unit)
    )
    capacity = capacity.dropna(subset=["_capacity_mw"])
    capacity = capacity.loc[capacity["_capacity_mw"] > 0]

    wind_values = {_normalise_label(value) for value in config.wind_technology_values}
    solar_values = {_normalise_label(value) for value in config.solar_technology_values}
    capacity = capacity.loc[capacity["_technology"].isin(wind_values | solar_values)].copy()
    if capacity.empty:
        raise ValueError("No configured wind/solar capacity rows found in the capacity file.")

    if config.capacity_format == "plant_locations":
        capacity = capacity.dropna(subset=[config.longitude_column, config.latitude_column])
        clusters = _load_cluster_assignments(_infer_cluster_file(config))
        tree = cKDTree(clusters[["lon", "lat"]].to_numpy(dtype=float))
        points = capacity[[config.longitude_column, config.latitude_column]].to_numpy(dtype=float)
        _, nearest_idx = tree.query(points)
        capacity["_cluster_id"] = clusters.iloc[nearest_idx]["cluster_id"].to_numpy(dtype=int)
    else:
        capacity["_cluster_id"] = pd.to_numeric(capacity[config.cluster_id_column], errors="coerce").astype("Int64")
        capacity = capacity.dropna(subset=["_cluster_id"])
        capacity["_cluster_id"] = capacity["_cluster_id"].astype(int)

    capacity["_technology_group"] = np.where(capacity["_technology"].isin(wind_values), "wind", "solar")
    grouped = (
        capacity
        .groupby(["_cluster_id", "_technology_group"])["_capacity_mw"]
        .sum()
        .unstack(fill_value=0.0)
        .rename_axis("cluster_id")
    )
    for column in ["wind", "solar"]:
        if column not in grouped.columns:
            grouped[column] = 0.0
    return grouped[["wind", "solar"]].sort_index()


def _cluster_capacity_vector(capacity_by_cluster: pd.DataFrame, technology: str, columns: list[str]) -> np.ndarray:
    technology_capacity = capacity_by_cluster.get(technology, pd.Series(dtype=float))
    capacities = []
    for column in columns:
        cluster_id = int(column.rsplit("_", 1)[-1])
        capacities.append(float(technology_capacity.get(cluster_id, 0.0)))
    return np.asarray(capacities, dtype=float)


def _normalised_wind_power_curve(
    wind_speed_m_s: pd.DataFrame,
    *,
    cut_in: float,
    rated: float,
    cut_out: float,
) -> pd.DataFrame:
    values = wind_speed_m_s.to_numpy(dtype=float)
    normalised = np.zeros_like(values, dtype=float)

    ramp_mask = (values >= cut_in) & (values < rated)
    rated_mask = (values >= rated) & (values <= cut_out)
    normalised[ramp_mask] = ((values[ramp_mask] - cut_in) / (rated - cut_in)) ** 3
    normalised[rated_mask] = 1.0
    return pd.DataFrame(np.clip(normalised, 0.0, 1.0), index=wind_speed_m_s.index, columns=wind_speed_m_s.columns)


def _add_ramps(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        out[f"{column}_ramp_1h"] = out[column].diff(4)
        out[f"{column}_ramp_3h"] = out[column].diff(12)
    return out


def _deduplicate_index(series: pd.Series) -> pd.Series:
    if series.index.has_duplicates:
        series = series.groupby(level=0).mean()
    return series.sort_index()


def build_renewable_proxy(config: RenewableProxyConfig) -> pd.DataFrame:
    capacity_by_cluster = _load_capacity_by_cluster(config)
    df_hourly, df_qh = load_dwd(
        icon_dir=config.icon_dir,
        start_folder_date=config.start_folder_date,
        required_run=config.required_run,
        skip_dates=set(config.skip_dates),
        folder_offset_date=config.dwd_folder_offset_date,
        target_tz=config.target_tz,
    )

    u_cols = sorted(col for col in df_hourly.columns if col.startswith("u10_cluster_"))
    wind_speed = {}
    for u_col in u_cols:
        cluster_id = u_col.rsplit("_", 1)[-1]
        v_col = f"v10_cluster_{cluster_id}"
        if v_col in df_hourly.columns:
            speed_10m = np.sqrt(df_hourly[u_col].to_numpy(dtype=float) ** 2 + df_hourly[v_col].to_numpy(dtype=float) ** 2)
            speed_hub = speed_10m * (config.wind_hub_height_m / config.wind_reference_height_m) ** config.wind_shear_alpha
            wind_speed[f"cluster_{cluster_id}"] = speed_hub

    if not wind_speed:
        raise ValueError("No matching u10/v10 DWD wind cluster columns found.")

    wind_speed_df = pd.DataFrame(wind_speed, index=df_hourly.index).sort_index()
    wind_power_norm = _normalised_wind_power_curve(
        wind_speed_df,
        cut_in=config.wind_cut_in_m_s,
        rated=config.wind_rated_m_s,
        cut_out=config.wind_cut_out_m_s,
    )
    wind_capacity = _cluster_capacity_vector(capacity_by_cluster, "wind", list(wind_power_norm.columns))
    wind_proxy_hourly = _deduplicate_index(wind_power_norm.mul(wind_capacity, axis=1).sum(axis=1))

    dir_cols = sorted(col for col in df_qh.columns if col.startswith("ASWDIR_cluster_"))
    solar_irradiance = {}
    for dir_col in dir_cols:
        cluster_id = dir_col.rsplit("_", 1)[-1]
        dif_col = f"ASWDIFD_cluster_{cluster_id}"
        if dif_col in df_qh.columns:
            irradiance = df_qh[dir_col].to_numpy(dtype=float) + df_qh[dif_col].to_numpy(dtype=float)
            solar_irradiance[f"cluster_{cluster_id}"] = np.clip(irradiance, 0.0, None)

    if not solar_irradiance:
        raise ValueError("No matching ASWDIR/ASWDIFD DWD solar cluster columns found.")

    solar_df = pd.DataFrame(solar_irradiance, index=df_qh.index).sort_index()
    solar_capacity = _cluster_capacity_vector(capacity_by_cluster, "solar", list(solar_df.columns))
    solar_norm = np.clip(solar_df / config.solar_reference_irradiance_w_m2, 0.0, 1.0)
    solar_proxy = _deduplicate_index(solar_norm.mul(solar_capacity * config.solar_performance_ratio, axis=1).sum(axis=1))

    full_index = pd.date_range(
        start=min(wind_proxy_hourly.index.min(), solar_proxy.index.min()),
        end=max(wind_proxy_hourly.index.max(), solar_proxy.index.max()),
        freq="15min",
        tz=config.target_tz,
        name="timestamp",
    )
    result = pd.DataFrame(index=full_index)
    result["Renewable_Wind_Proxy_MW"] = wind_proxy_hourly.reindex(full_index).ffill(limit=3)
    result["Renewable_Solar_Proxy_MW"] = solar_proxy.reindex(full_index)
    result["Renewable_Total_Proxy_MW"] = (
        result["Renewable_Wind_Proxy_MW"].fillna(0.0) + result["Renewable_Solar_Proxy_MW"].fillna(0.0)
    )
    result = _add_ramps(
        result,
        ["Renewable_Wind_Proxy_MW", "Renewable_Solar_Proxy_MW", "Renewable_Total_Proxy_MW"],
    )
    result.index.name = "timestamp"
    return result


def run_renewable_proxy(config: RenewableProxyConfig) -> pd.DataFrame:
    result = build_renewable_proxy(config)
    config.output_file.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(config.output_file)

    metadata = {
        "capacity_file": str(config.capacity_file),
        "icon_dir": str(config.icon_dir),
        "cluster_file": str(_infer_cluster_file(config)) if config.capacity_format == "plant_locations" else None,
        "capacity_format": config.capacity_format,
        "required_run": config.required_run,
        "start_folder_date": str(config.start_folder_date),
        "output_file": str(config.output_file),
    }
    with open(config.output_file.with_suffix(".json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Saved renewable proxy: {config.output_file}")
    return result
