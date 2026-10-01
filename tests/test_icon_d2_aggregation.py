from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from da_price_forecasting.config import IconAggregationConfig
from da_price_forecasting.data.weather import _canonical_dwd_variable_name
from da_price_forecasting.preprocessing.icon_d2_aggregation import (
    aggregate_cluster_values,
    cluster_mastr_solar_capacity_coordinates,
    cluster_mastr_solar_tso_capacity_coordinates,
    _build_cluster_specs,
    _is_unstructured_grid,
    _lat_lon_from_icon_grid_file,
    _masked_lon_lat_coordinates,
    _model_level_from_filename,
    _reduce_ensemble,
)


def test_aggregate_cluster_values_uses_capacity_weights() -> None:
    values = np.array([10.0, 20.0, 100.0])
    labels = np.array([0, 0, 1])
    weights = np.array([1.0, 3.0, 0.0])

    aggregated = aggregate_cluster_values(values, labels, n_clusters=2, cluster_weights=weights)

    assert aggregated[0] == pytest.approx(17.5)
    assert aggregated[1] == pytest.approx(100.0)


def test_reduce_ensemble_collapses_number_dimension() -> None:
    data = xr.DataArray(
        np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]),
        dims=["number", "cell"],
    )

    mean = _reduce_ensemble(data, "mean")
    std = _reduce_ensemble(data, "std")

    assert "number" not in mean.dims
    assert mean.values.tolist() == [2.0, 3.0, 4.0]
    assert std.values == pytest.approx([1.0, 1.0, 1.0])


def test_reduce_ensemble_passes_through_deterministic_fields() -> None:
    # reduction=None and fields without a 'number' dim must be untouched, so the same
    # streaming code serves deterministic and EPS data.
    deterministic = xr.DataArray(np.array([1.0, 2.0, 3.0]), dims=["cell"])
    assert _reduce_ensemble(deterministic, "mean") is deterministic
    ensemble = xr.DataArray(np.zeros((2, 3)), dims=["number", "cell"])
    assert _reduce_ensemble(ensemble, None) is ensemble


def test_unstructured_grid_detection_and_masked_coordinates() -> None:
    icosahedral_lat = np.array([51.0, 48.0, 50.0])
    icosahedral_lon = np.array([10.0, 11.0, 12.0])
    assert _is_unstructured_grid(icosahedral_lat, icosahedral_lon)

    regular_lat = np.array([51.0, 48.0, 50.0])
    regular_lon = np.array([10.0, 11.0])
    assert not _is_unstructured_grid(regular_lat, regular_lon)

    mask = np.array([True, False, True])
    coords = _masked_lon_lat_coordinates(icosahedral_lat, icosahedral_lon, mask)
    assert coords.tolist() == [[10.0, 51.0], [12.0, 50.0]]


def test_lat_lon_from_icon_grid_file_converts_radians(tmp_path: Path) -> None:
    grid_file = tmp_path / "icon_grid.nc"
    # ICON grid files store clat/clon in RADIANS; 349.5 deg exercises the [-180, 180] wrap.
    xr.Dataset(
        {
            "clat": ("cell", np.deg2rad([51.0, 48.4])),
            "clon": ("cell", np.deg2rad([10.0, 349.5])),
        }
    ).to_netcdf(grid_file)

    lat, lon = _lat_lon_from_icon_grid_file(grid_file)

    assert lat == pytest.approx([51.0, 48.4], abs=1e-4)
    assert lon == pytest.approx([10.0, -10.5], abs=1e-4)


def test_canonical_name_preserves_ensemble_suffix() -> None:
    # The aggregator writes 'u10_ensmean'/'v10_ensstd'; load_dwd must keep the suffix so the
    # wind feature builder can detect u10_ensmean_cluster_* and its ensstd spread partner.
    assert _canonical_dwd_variable_name("u10_ensmean") == "u10_ensmean"
    assert _canonical_dwd_variable_name("v10_ensstd") == "v10_ensstd"
    assert _canonical_dwd_variable_name("10u") == "u10"


def test_model_level_from_dwd_filename() -> None:
    filename = (
        "icon-d2_germany_regular-lat-lon_model-level_"
        "2026061506_011_65_u.grib2.bz2"
    )

    assert _model_level_from_filename(filename) == 65


def test_multi_target_cluster_specs_keep_run_and_variable_filters(tmp_path: Path) -> None:
    config = IconAggregationConfig(
        repo_root=tmp_path,
        shapefile_path=tmp_path / "unused.shp",
        aggregation_targets=[
            {
                "name": "load_c1_run06",
                "only_run_hour": "06",
                "output_parent": "out/load_c1_run06",
                "cluster_source": "grid",
                "n_clusters": 1,
                "variables": ["t_2m"],
            },
            {
                "name": "price_c2_run09",
                "only_run_hour": "09",
                "output_parent": "out/price_c2_run09",
                "cluster_source": "grid",
                "n_clusters": 2,
                "variables": ["u_10m", "v_10m"],
            },
        ],
    )
    latitudes = np.array([49.0, 50.0, 51.0])
    longitudes = np.array([8.0, 9.0])
    mask = np.ones((len(latitudes), len(longitudes)), dtype=bool)

    specs = _build_cluster_specs(latitudes, longitudes, mask, config)

    assert len(specs) == 2
    assert specs[0].name == "load_c1_run06"
    assert specs[0].output_parent == str(tmp_path / "out/load_c1_run06")
    assert specs[0].applies_to_run("06")
    assert not specs[0].applies_to_run("09")
    assert specs[0].applies_to_variable("t_2m")
    assert not specs[0].applies_to_variable("u_10m")
    assert specs[1].name == "price_c2_run09"
    assert specs[1].applies_to_run("09")
    assert specs[1].applies_to_variable("v_10m")


def test_mastr_solar_clustering_maps_capacity_to_dwd_grid(tmp_path: Path) -> None:
    capacity_file = tmp_path / "installed_capacity.csv"
    pd.DataFrame(
        {
            "technology": ["pv", "pv", "wind_onshore"],
            "capacity_mw": [10.0, 30.0, 999.0],
            "lat": [50.0, 51.0, 52.0],
            "lon": [8.0, 9.0, 10.0],
            "operating_status": ["35", "35", "35"],
        }
    ).to_csv(capacity_file, index=False)

    config = IconAggregationConfig(
        repo_root=tmp_path,
        capacity_file=capacity_file,
        cluster_source="mastr_solar",
        n_clusters=2,
    )
    latitudes = np.array([49.5, 50.0, 51.0, 51.5])
    longitudes = np.array([7.5, 8.0, 9.0, 9.5])
    mask = np.ones((len(latitudes), len(longitudes)), dtype=bool)

    coords, labels, centroids, weights = cluster_mastr_solar_capacity_coordinates(
        latitudes,
        longitudes,
        mask,
        config,
    )

    assert len(coords) == mask.sum()
    assert set(labels) == {0, 1}
    assert centroids.shape == (2, 2)
    assert weights.sum() == pytest.approx(40.0)
    assert np.count_nonzero(weights) == 2


def test_mastr_solar_tso_clustering_builds_clusters_per_proxy_region(tmp_path: Path) -> None:
    capacity_file = tmp_path / "installed_capacity.csv"
    pd.DataFrame(
        {
            "technology": ["pv", "pv", "pv", "pv"],
            "capacity_mw": [10.0, 20.0, 30.0, 40.0],
            "lat": [52.0, 51.0, 49.0, 48.5],
            "lon": [13.0, 7.0, 11.0, 9.0],
            "operating_status": ["35"] * 4,
            "federal_state_code": [1400, 1409, 1403, 1402],
        }
    ).to_csv(capacity_file, index=False)

    config = IconAggregationConfig(
        repo_root=tmp_path,
        capacity_file=capacity_file,
        cluster_source="mastr_solar_tso",
        n_clusters=1,
    )
    latitudes = np.array([48.5, 49.0, 51.0, 52.0])
    longitudes = np.array([7.0, 9.0, 11.0, 13.0])
    mask = np.ones((len(latitudes), len(longitudes)), dtype=bool)

    coords, labels, centroids, weights, regions = cluster_mastr_solar_tso_capacity_coordinates(
        latitudes,
        longitudes,
        mask,
        config,
    )

    assert len(coords) == mask.sum()
    assert set(labels) == {0, 1, 2, 3}
    assert centroids.shape == (4, 2)
    assert regions.tolist() == [
        "solar_50hertz",
        "solar_amprion",
        "solar_tennet",
        "solar_transnetbw",
    ]
    assert weights.sum() == pytest.approx(100.0)
