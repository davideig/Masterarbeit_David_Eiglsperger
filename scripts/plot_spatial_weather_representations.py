from pathlib import Path
import os
import warnings

os.environ.setdefault("MPLCONFIGDIR", "/tmp/da_price_forecasting_matplotlib")

import matplotlib

matplotlib.use("Agg")

import geopandas as gpd
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from shapely.geometry import Point, box


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "Final Paper" / "figures"
SHAPEFILE = ROOT / "data" / "shapefile" / "ne_10m_admin_0_countries.shp"


def _read_cluster_points(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path, columns=["lon", "lat", "cluster_id"])
    columns = pd.read_csv(path, nrows=0).columns
    usecols = ["lon", "lat", "cluster_id"]
    if "capacity_weight_mw" in columns:
        usecols.append("capacity_weight_mw")
    return pd.read_csv(path, usecols=usecols)


def _assign_to_weighted_centroids(points: pd.DataFrame) -> pd.DataFrame:
    centroids = []
    for cluster_id, group in points.groupby("cluster_id", sort=True):
        weights = group.get("capacity_weight_mw")
        if weights is not None and float(weights.sum()) > 0:
            active = group.loc[weights > 0]
            active_weights = active["capacity_weight_mw"].to_numpy(dtype=float)
            lon = float(np.average(active["lon"].to_numpy(dtype=float), weights=active_weights))
            lat = float(np.average(active["lat"].to_numpy(dtype=float), weights=active_weights))
        else:
            lon = float(group["lon"].mean())
            lat = float(group["lat"].mean())
        centroids.append((int(cluster_id), lon, lat))

    centroid_frame = pd.DataFrame(centroids, columns=["cluster_id", "lon", "lat"])
    point_coords = points[["lon", "lat"]].to_numpy(dtype=float)
    centroid_coords = centroid_frame[["lon", "lat"]].to_numpy(dtype=float)
    distances = ((point_coords[:, None, :] - centroid_coords[None, :, :]) ** 2).sum(axis=2)

    display_points = points[["lon", "lat"]].copy()
    display_points["cluster_id"] = centroid_frame["cluster_id"].to_numpy()[distances.argmin(axis=1)]
    return display_points


def _build_cluster_polygons(path: Path, *, clean_capacity_assignment: bool = False) -> gpd.GeoDataFrame:
    points = _read_cluster_points(path)
    if clean_capacity_assignment and "capacity_weight_mw" in points.columns:
        points = _assign_to_weighted_centroids(points)

    lon_values = np.sort(points["lon"].unique())
    lat_values = np.sort(points["lat"].unique())
    half_dlon = float(np.median(np.diff(lon_values))) / 2
    half_dlat = float(np.median(np.diff(lat_values))) / 2

    geometries = [
        box(row.lon - half_dlon, row.lat - half_dlat, row.lon + half_dlon, row.lat + half_dlat)
        for row in points.itertuples(index=False)
    ]
    cells = gpd.GeoDataFrame(
        points[["cluster_id"]],
        geometry=geometries,
        crs="EPSG:4326",
    )
    clusters = cells.dissolve(by="cluster_id", as_index=False)[["cluster_id", "geometry"]]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Geometry is in a geographic CRS.*")
        clusters["geometry"] = clusters.geometry.buffer(0.001).buffer(-0.001)
    clusters = gpd.GeoDataFrame(clusters, geometry="geometry", crs="EPSG:4326").to_crs(epsg=3035)
    clusters["geometry"] = clusters.geometry.buffer(0)
    return clusters


def _load_germany(target_crs: str) -> gpd.GeoDataFrame:
    world = gpd.read_file(SHAPEFILE)
    if world.crs is None:
        world = world.set_crs(epsg=4326)

    if "NAME" in world.columns:
        germany = world[world["NAME"].fillna("").str.lower() == "germany"].copy()
        if not germany.empty:
            return germany.to_crs(target_crs)

    germany_probe = Point(10.4515, 51.1657)
    germany_bbox = box(5.5, 47.0, 15.5, 55.5)
    germany = world[world.geometry.contains(germany_probe)].copy()
    if germany.empty:
        germany = world[world.geometry.intersects(germany_probe.buffer(0.25))].copy()
    if germany.empty:
        overlaps = world.geometry.intersection(germany_bbox).area.fillna(0)
        germany = world.loc[[overlaps.idxmax()]].copy() if (overlaps > 0).any() else world.iloc[0:0].copy()
    if germany.empty:
        raise RuntimeError("Germany not found in shapefile.")
    if len(germany) > 1:
        overlaps = germany.geometry.intersection(germany_bbox).area.fillna(0)
        germany = germany.loc[[overlaps.idxmax()]].copy()
    return germany.to_crs(target_crs)


def _plot_cluster_map(
    cluster_file: Path,
    output_file: Path,
    *,
    clean_capacity_assignment: bool = False,
) -> None:
    clusters = _build_cluster_polygons(
        cluster_file,
        clean_capacity_assignment=clean_capacity_assignment,
    )
    germany = _load_germany(clusters.crs)

    plt.rcParams["lines.solid_joinstyle"] = "round"
    plt.rcParams["lines.solid_capstyle"] = "round"

    fig, ax = plt.subplots(figsize=(4.0, 5.1))
    clusters.plot(
        ax=ax,
        color="#f2f2f2",
        edgecolor="none",
        linewidth=0,
        zorder=1,
    )
    clusters.explode(index_parts=False).dissolve(by="cluster_id").boundary.plot(
        ax=ax,
        edgecolor="#555555",
        linewidth=0.42,
        zorder=2,
    )
    germany.boundary.plot(
        ax=ax,
        linewidth=1.05,
        edgecolor="black",
        zorder=3,
    )

    ax.set_axis_off()
    ax.set_aspect("equal")
    fig.savefig(output_file, bbox_inches="tight", pad_inches=0.01, dpi=300)
    plt.close(fig)


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plots = [
        (
            ROOT / "data" / "clustering" / "icon_d2_clustering_c25.parquet",
            FIGURE_DIR / "spatial_weather_load_clusters.pdf",
            False,
        ),
        (
            ROOT / "data" / "clustering" / "icon_d2_mastr_solar_tso_c25.csv",
            FIGURE_DIR / "spatial_weather_solar_clusters.pdf",
            True,
        ),
        (
            ROOT / "data" / "clustering" / "icon_d2_mastr_wind_c100.csv",
            FIGURE_DIR / "spatial_weather_wind_clusters.pdf",
            False,
        ),
    ]
    for cluster_file, output_file, clean_capacity_assignment in plots:
        _plot_cluster_map(
            cluster_file,
            output_file,
            clean_capacity_assignment=clean_capacity_assignment,
        )
        print(f"Saved {output_file}")


if __name__ == "__main__":
    main()
