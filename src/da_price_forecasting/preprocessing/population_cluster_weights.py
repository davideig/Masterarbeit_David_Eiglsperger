from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from pyproj import Transformer
from scipy.spatial import cKDTree

from ..config import PopulationClusterWeightsConfig


def _download_file(url: str, target: Path, *, force: bool, timeout_seconds: int) -> Path:
    if target.exists() and not force:
        print(f"Using cached population grid: {target}")
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_target = target.with_suffix(target.suffix + ".tmp")
    print(f"Downloading population grid from {url}")
    with requests.get(url, stream=True, timeout=timeout_seconds) as response:
        response.raise_for_status()
        with tmp_target.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    tmp_target.replace(target)
    return target


def _population_source_path(config: PopulationClusterWeightsConfig) -> Path:
    if config.source_file is not None:
        return config.source_file
    if config.source_url is None:
        raise ValueError("source_url is required when source_file is not set.")
    return _download_file(
        config.source_url,
        config.raw_file,
        force=config.force_download,
        timeout_seconds=config.timeout_seconds,
    )


def _list_tabular_columns(path: Path) -> list[str]:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        import pyarrow.parquet as pq

        return list(pq.read_schema(path).names)
    if suffix in {".csv", ".txt"}:
        return list(pd.read_csv(path, nrows=0).columns)
    raise ValueError(f"Unsupported population grid format: {path.suffix}. Use parquet or CSV.")


def _read_population_grid(path: Path, config: PopulationClusterWeightsConfig) -> pd.DataFrame:
    available_columns = set(_list_tabular_columns(path))
    required_columns = {config.x_column, config.y_column, config.population_column}
    missing = sorted(required_columns - available_columns)
    if missing:
        raise ValueError(f"Population grid is missing required columns: {missing}")

    optional_columns = {config.country_column}
    if config.region_column is not None:
        optional_columns.add(config.region_column)
    use_columns = sorted(required_columns | (optional_columns & available_columns))

    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path, columns=use_columns)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path, usecols=use_columns)
    raise ValueError(f"Unsupported population grid format: {path.suffix}. Use parquet or CSV.")


def _country_tokens(value: object) -> set[str]:
    return {token.upper() for token in re.split(r"[^A-Za-z]+", str(value)) if token}


def _region_tokens(value: object, country_codes: set[str]) -> list[str]:
    tokens = [token.upper() for token in re.split(r"[^A-Za-z0-9]+", str(value)) if token]
    return [token for token in tokens if any(token.startswith(country) for country in country_codes)]


def _filter_population_grid(df: pd.DataFrame, config: PopulationClusterWeightsConfig) -> pd.DataFrame:
    result = df.copy()
    if config.country_column in result.columns:
        countries = {country.upper() for country in config.country_codes}
        country_mask = result[config.country_column].map(lambda value: bool(_country_tokens(value) & countries))
        result = result.loc[country_mask].copy()

    result[config.population_column] = pd.to_numeric(result[config.population_column], errors="coerce")
    result[config.x_column] = pd.to_numeric(result[config.x_column], errors="coerce")
    result[config.y_column] = pd.to_numeric(result[config.y_column], errors="coerce")
    result = result.dropna(subset=[config.x_column, config.y_column, config.population_column])
    result = result.loc[result[config.population_column] > 0.0].copy()
    if result.empty:
        raise ValueError("No positive population cells remain after filtering.")
    return result


def _load_cluster_points(config: PopulationClusterWeightsConfig) -> pd.DataFrame:
    if config.cluster_file.suffix.lower() in {".parquet", ".pq"}:
        clusters = pd.read_parquet(config.cluster_file)
    else:
        clusters = pd.read_csv(config.cluster_file)

    required = {config.cluster_lon_column, config.cluster_lat_column, config.cluster_id_column}
    missing = sorted(required - set(clusters.columns))
    if missing:
        raise ValueError(f"Cluster file is missing required columns: {missing}")

    clusters = clusters[
        [config.cluster_lon_column, config.cluster_lat_column, config.cluster_id_column]
    ].dropna().copy()
    clusters[config.cluster_id_column] = clusters[config.cluster_id_column].astype(int)
    if clusters.empty:
        raise ValueError(f"No cluster points found in {config.cluster_file}")

    if config.matching_mode == "nearest_centroid":
        clusters = (
            clusters.groupby(config.cluster_id_column, as_index=False)[
                [config.cluster_lon_column, config.cluster_lat_column]
            ]
            .mean()
            .copy()
        )
    return clusters


def _project_cluster_points(clusters: pd.DataFrame, config: PopulationClusterWeightsConfig) -> np.ndarray:
    transformer = Transformer.from_crs(
        f"EPSG:{config.cluster_crs_epsg}",
        f"EPSG:{config.population_crs_epsg}",
        always_xy=True,
    )
    x, y = transformer.transform(
        clusters[config.cluster_lon_column].to_numpy(dtype=float),
        clusters[config.cluster_lat_column].to_numpy(dtype=float),
    )
    return np.column_stack([x, y])


def _assign_population_cells_to_clusters(
    population: pd.DataFrame,
    clusters: pd.DataFrame,
    config: PopulationClusterWeightsConfig,
) -> pd.DataFrame:
    cluster_xy = _project_cluster_points(clusters, config)
    cluster_ids = clusters[config.cluster_id_column].to_numpy(dtype=int)
    tree = cKDTree(cluster_xy)

    population_xy = np.column_stack(
        [
            population[config.x_column].to_numpy(dtype=float) + config.cell_size_m / 2.0,
            population[config.y_column].to_numpy(dtype=float) + config.cell_size_m / 2.0,
        ]
    )
    distance_m, nearest_idx = tree.query(population_xy)

    assigned = population.copy()
    assigned["cluster_id"] = cluster_ids[nearest_idx]
    assigned["nearest_cluster_distance_m"] = distance_m
    return assigned


def _build_cluster_weights(assigned: pd.DataFrame, config: PopulationClusterWeightsConfig) -> pd.DataFrame:
    grouped = assigned.groupby("cluster_id", sort=True).agg(
        weight=(config.population_column, "sum"),
        n_population_cells=(config.population_column, "size"),
        mean_nearest_cluster_distance_m=("nearest_cluster_distance_m", "mean"),
        max_nearest_cluster_distance_m=("nearest_cluster_distance_m", "max"),
    )
    grouped["population_share"] = grouped["weight"] / grouped["weight"].sum()
    output = grouped.reset_index()
    return output[
        [
            "cluster_id",
            "weight",
            "population_share",
            "n_population_cells",
            "mean_nearest_cluster_distance_m",
            "max_nearest_cluster_distance_m",
        ]
    ]


def _build_region_weights(assigned: pd.DataFrame, config: PopulationClusterWeightsConfig) -> pd.DataFrame:
    if config.region_column is None or config.region_column not in assigned.columns:
        return pd.DataFrame()

    region_data = assigned.dropna(subset=[config.region_column]).copy()
    if region_data.empty:
        return pd.DataFrame()

    country_codes = {country.upper() for country in config.country_codes}
    region_data["_region"] = region_data[config.region_column].map(lambda value: _region_tokens(value, country_codes))
    region_data = region_data.loc[region_data["_region"].map(bool)].copy()
    if region_data.empty:
        return pd.DataFrame()

    region_data["_region_weight"] = region_data[config.population_column] / region_data["_region"].map(len)
    region_data = region_data.explode("_region")
    grouped = (
        region_data.groupby(["cluster_id", "_region"], sort=True)["_region_weight"]
        .sum()
        .rename("weight")
        .reset_index()
        .rename(columns={"_region": "region"})
    )
    cluster_totals = grouped.groupby("cluster_id")["weight"].transform("sum")
    grouped["cluster_population_share"] = grouped["weight"] / cluster_totals
    grouped["total_population_share"] = grouped["weight"] / grouped["weight"].sum()
    return grouped[["cluster_id", "region", "weight", "cluster_population_share", "total_population_share"]]


def build_population_cluster_weights(config: PopulationClusterWeightsConfig) -> dict[str, pd.DataFrame | dict[str, Any]]:
    source = _population_source_path(config)
    print(f"Reading population grid: {source}")
    population_raw = _read_population_grid(source, config)
    population = _filter_population_grid(population_raw, config)
    print(f"Population cells after filtering: {len(population):,}")

    clusters = _load_cluster_points(config)
    print(f"Cluster matching points: {len(clusters):,} ({config.matching_mode})")
    assigned = _assign_population_cells_to_clusters(population, clusters, config)

    cluster_weights = _build_cluster_weights(assigned, config)
    region_weights = _build_region_weights(assigned, config)
    summary: dict[str, Any] = {
        "source_file": str(source),
        "cluster_file": str(config.cluster_file),
        "output_file": str(config.output_file),
        "region_output_file": str(config.region_output_file) if config.region_output_file is not None else None,
        "country_codes": config.country_codes,
        "population_column": config.population_column,
        "region_column": config.region_column,
        "matching_mode": config.matching_mode,
        "n_population_cells": int(len(assigned)),
        "n_clusters_with_population": int(len(cluster_weights)),
        "total_population": float(cluster_weights["weight"].sum()),
        "mean_nearest_cluster_distance_m": float(assigned["nearest_cluster_distance_m"].mean()),
        "max_nearest_cluster_distance_m": float(assigned["nearest_cluster_distance_m"].max()),
    }

    return {
        "cluster_weights": cluster_weights,
        "region_weights": region_weights,
        "summary": summary,
    }


def run_population_cluster_weights(config: PopulationClusterWeightsConfig) -> dict[str, pd.DataFrame | dict[str, Any]]:
    result = build_population_cluster_weights(config)
    cluster_weights = result["cluster_weights"]
    region_weights = result["region_weights"]
    summary = result["summary"]

    if not isinstance(cluster_weights, pd.DataFrame) or not isinstance(region_weights, pd.DataFrame):
        raise TypeError("Unexpected population cluster weight result.")
    if not isinstance(summary, dict):
        raise TypeError("Unexpected population cluster weight summary.")

    config.output_file.parent.mkdir(parents=True, exist_ok=True)
    cluster_weights.to_csv(config.output_file, index=False)
    print(f"Saved cluster population weights -> {config.output_file}")

    if config.region_output_file is not None and not region_weights.empty:
        config.region_output_file.parent.mkdir(parents=True, exist_ok=True)
        region_weights.to_csv(config.region_output_file, index=False)
        print(f"Saved cluster-region population weights -> {config.region_output_file}")
    elif config.region_output_file is not None:
        print("No region weights written; region column was missing or empty.")

    summary_path = config.summary_file or config.output_file.with_suffix(".json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Saved population weight summary -> {summary_path}")
    print(
        "Population matched: "
        f"{summary['total_population']:,.0f} across {summary['n_clusters_with_population']} clusters"
    )
    return result
