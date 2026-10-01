from __future__ import annotations
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from da_price_forecasting.config import RegionalRenewableFeatureConfig
from da_price_forecasting.preprocessing import regional_renewable_features as rrf


def test_solar_state_tso_proxy_region_assignment(tmp_path: Path) -> None:
    capacity_file = tmp_path / "capacity.csv"
    pd.DataFrame(
        {
            "technology": ["pv", "pv", "pv", "pv"],
            "capacity_mw": [10.0, 20.0, 30.0, 40.0],
            "lat": [52.0, 51.0, 49.0, 48.5],
            "lon": [13.0, 7.0, 11.0, 9.0],
            "commissioning_date": ["2020-01-01"] * 4,
            "decommissioning_date": [None] * 4,
            "operating_status": ["35"] * 4,
            "federal_state_code": [1400, 1409, 1403, 1402],
        }
    ).to_csv(capacity_file, index=False)

    cluster_file = tmp_path / "clusters.csv"
    pd.DataFrame(
        {
            "cluster_id": [0, 1, 2, 3],
            "lat": [52.0, 51.0, 49.0, 48.5],
            "lon": [13.0, 7.0, 11.0, 9.0],
        }
    ).to_csv(cluster_file, index=False)

    config = RegionalRenewableFeatureConfig(
        repo_root=tmp_path,
        capacity_file=capacity_file,
        cluster_file=cluster_file,
        capacity_map_file=tmp_path / "capacity_map.csv",
        output_file=tmp_path / "features.csv",
        solar_region_strategy="state_tso_proxy",
    )

    capacity_map = rrf.build_capacity_map(config)

    assert capacity_map.set_index("capacity_mw")["region"].to_dict() == {
        10.0: "solar_50hertz",
        20.0: "solar_amprion",
        30.0: "solar_tennet",
        40.0: "solar_transnetbw",
    }


def test_capacity_map_prefers_cluster_from_matching_region(tmp_path: Path) -> None:
    capacity_file = tmp_path / "capacity.csv"
    pd.DataFrame(
        {
            "technology": ["pv"],
            "capacity_mw": [10.0],
            "lat": [52.0],
            "lon": [7.1],
            "commissioning_date": ["2020-01-01"],
            "decommissioning_date": [None],
            "operating_status": ["35"],
            "federal_state_code": [1400],
        }
    ).to_csv(capacity_file, index=False)

    cluster_file = tmp_path / "clusters.csv"
    pd.DataFrame(
        {
            "cluster_id": [0, 1],
            "lat": [52.0, 52.0],
            "lon": [13.0, 7.0],
            "cluster_region": ["solar_50hertz", "solar_amprion"],
        }
    ).to_csv(cluster_file, index=False)

    config = RegionalRenewableFeatureConfig(
        repo_root=tmp_path,
        capacity_file=capacity_file,
        cluster_file=cluster_file,
        capacity_map_file=tmp_path / "capacity_map.csv",
        output_file=tmp_path / "features.csv",
        solar_region_strategy="state_tso_proxy",
    )

    capacity_map = rrf.build_capacity_map(config)

    assert capacity_map["region"].iloc[0] == "solar_50hertz"
    assert capacity_map["cluster_id"].iloc[0] == 0


def test_solar_tso_capacity_artifacts_track_monthly_capacity_and_weights(tmp_path: Path) -> None:
    capacity_file = tmp_path / "capacity.csv"
    pd.DataFrame(
        {
            "technology": ["pv", "pv", "pv"],
            "capacity_mw": [10.0, 20.0, 30.0],
            "lat": [52.0, 52.2, 51.0],
            "lon": [13.0, 13.2, 7.0],
            "commissioning_date": ["2026-01-01", "2026-02-15", "2026-01-01"],
            "decommissioning_date": [None, None, None],
            "operating_status": ["35", "35", "35"],
            "federal_state_code": [1400, 1400, 1409],
        }
    ).to_csv(capacity_file, index=False)

    cluster_file = tmp_path / "clusters.csv"
    pd.DataFrame(
        {
            "cluster_id": [0, 1, 2],
            "lat": [52.0, 52.2, 51.0],
            "lon": [13.0, 13.2, 7.0],
            "cluster_region": ["solar_50hertz", "solar_50hertz", "solar_amprion"],
        }
    ).to_csv(cluster_file, index=False)

    config = RegionalRenewableFeatureConfig(
        repo_root=tmp_path,
        capacity_file=capacity_file,
        cluster_file=cluster_file,
        capacity_map_file=tmp_path / "capacity_map.csv",
        output_file=tmp_path / "features.csv",
        solar_region_strategy="state_tso_proxy",
        capacity_artifact_technologies=["solar"],
    )
    capacity_map = rrf.build_capacity_map(config)
    timestamps = pd.date_range("2026-01-01T00:00:00+01:00", "2026-03-01T00:00:00+01:00", freq="MS")

    capacity_ts = rrf.build_capacity_timeseries_artifact(capacity_map, timestamps, config)
    weights = rrf.build_weather_weights_artifact(capacity_map, timestamps, config)

    march = capacity_ts.loc[capacity_ts["month"].dt.month == 3].iloc[0]
    assert np.isclose(march["solar_50hertz_installed_capacity_mw"], 30.0)
    assert np.isclose(march["solar_amprion_installed_capacity_mw"], 30.0)
    assert np.isclose(march["solar_total_installed_capacity_mw"], 60.0)

    march_50hertz = weights.loc[
        (weights["month"].dt.month == 3) & (weights["region"] == "solar_50hertz")
    ]
    assert np.isclose(march_50hertz["weight"].sum(), 1.0)
    assert set(march_50hertz["cluster_id"]) == {0, 1}
    assert np.isclose(
        march_50hertz.set_index("cluster_id").loc[1, "weight"],
        20.0 / 30.0,
    )


def test_capacity_weighted_open_meteo_points_build_regional_features(
    monkeypatch,
    tmp_path: Path,
) -> None:
    capacity_file = tmp_path / "capacity.csv"
    pd.DataFrame(
        {
            "technology": [
                "wind_offshore",
                "wind_onshore",
                "wind_onshore",
                "pv",
                "pv",
            ],
            "capacity_mw": [100.0, 50.0, 150.0, 20.0, 80.0],
            "lat": [54.0, 53.0, 50.0, 52.5, 48.8],
            "lon": [7.0, 8.5, 11.5, 9.0, 10.0],
            "commissioning_date": ["2020-01-01"] * 5,
            "decommissioning_date": [None] * 5,
            "operating_status": ["35"] * 5,
            "hub_height_m": [120.0, 135.0, 100.0, np.nan, np.nan],
            "rotor_diameter_m": [150.0, 120.0, 90.0, np.nan, np.nan],
            "swept_area_m2": [
                np.pi * (150.0 / 2) ** 2,
                np.pi * (120.0 / 2) ** 2,
                np.pi * (90.0 / 2) ** 2,
                np.nan,
                np.nan,
            ],
            "manufacturer_code": ["offshore_maker", "maker_a", "maker_b", None, None],
            "turbine_type": ["offshore_type", "type_a", "type_b", None, None],
        }
    ).to_csv(capacity_file, index=False)

    cluster_file = tmp_path / "clusters.csv"
    pd.DataFrame(
        {
            "cluster_id": [0, 1, 2],
            "lat": [54.0, 51.0, 49.0],
            "lon": [7.0, 9.0, 11.0],
        }
    ).to_csv(cluster_file, index=False)

    def fake_load_open_meteo_points(**kwargs):  # noqa: ANN003
        points = kwargs["points"]
        index = pd.date_range("2026-01-01T00:00:00+01:00", periods=2, freq="h")
        data = {}
        for point_id in points["weather_point_id"].astype(int):
            data[f"u100_point_{point_id}"] = np.full(len(index), 6.0 + point_id)
            data[f"v100_point_{point_id}"] = np.full(len(index), 2.0)
            data[f"u80_point_{point_id}"] = np.full(len(index), 5.5 + point_id)
            data[f"v80_point_{point_id}"] = np.full(len(index), 2.0)
            data[f"u120_point_{point_id}"] = np.full(len(index), 6.5 + point_id)
            data[f"v120_point_{point_id}"] = np.full(len(index), 2.0)
            data[f"t2m_point_{point_id}"] = np.full(len(index), 283.15)
            data[f"t80m_point_{point_id}"] = np.full(len(index), 282.15)
            data[f"t120m_point_{point_id}"] = np.full(len(index), 281.15)
            data[f"t180m_point_{point_id}"] = np.full(len(index), 280.15)
            data[f"sp_point_{point_id}"] = np.full(len(index), 100000.0)
            data[f"msl_point_{point_id}"] = np.full(len(index), 101000.0)
            data[f"r2m_point_{point_id}"] = np.full(len(index), 75.0)
            data[f"vpd_point_{point_id}"] = np.full(len(index), 0.6)
            data[f"vmax10m_point_{point_id}"] = np.full(len(index), 12.0)
            data[f"ssrd_point_{point_id}"] = np.full(len(index), 500.0)
            data[f"fdir_point_{point_id}"] = np.full(len(index), 300.0)
            data[f"dni_point_{point_id}"] = np.full(len(index), 650.0)
            data[f"diffuse_point_{point_id}"] = np.full(len(index), 200.0)
            data[f"tcc_point_{point_id}"] = np.full(len(index), 20.0)
            data[f"lcc_point_{point_id}"] = np.full(len(index), 10.0)
            data[f"mcc_point_{point_id}"] = np.full(len(index), 15.0)
            data[f"hcc_point_{point_id}"] = np.full(len(index), 25.0)
        return pd.DataFrame(data, index=index)

    monkeypatch.setattr(rrf, "load_open_meteo_points", fake_load_open_meteo_points)

    config = RegionalRenewableFeatureConfig(
        repo_root=tmp_path,
        weather_source="open_meteo",
        open_meteo_point_source="capacity",
        capacity_file=capacity_file,
        cluster_file=cluster_file,
        capacity_map_file=tmp_path / "capacity_map.csv",
        capacity_weather_point_file=tmp_path / "capacity_points.csv",
        output_file=tmp_path / "features.csv",
        open_meteo_weather_file=tmp_path / "point_weather.csv",
        open_meteo_start_date=date(2026, 1, 1),
        open_meteo_end_date=date(2026, 1, 1),
        capacity_weather_solar_points_per_region=1,
        capacity_weather_onshore_points_per_region=1,
        capacity_weather_offshore_points=1,
        include_capacity_static_features=True,
        include_geographic_exposure_features=True,
        include_wind_turbine_static_features=True,
        include_wind_hub_height_interpolation_features=True,
        wind_weather_weight_mode="capacity_x_rotor_area",
    )

    features, capacity_map, points = rrf.build_regional_renewable_features(config)

    assert points is not None
    assert set(points["technology_group"]) == {"solar", "wind_offshore", "wind_onshore"}
    assert np.isclose(capacity_map["capacity_mw"].sum(), 400.0)
    assert "wind_offshore_north_proxy_mw" in features.columns
    assert "wind_offshore_north_installed_capacity_mw" in features.columns
    assert "wind_offshore_north_capacity_weighted_age_years" in features.columns
    assert "wind_offshore_north_distance_to_coast_cap_weighted_km" in features.columns
    assert "wind_onshore_north_coastal_exposure_score_cap_weighted" in features.columns
    assert "wind_onshore_north_capacity_spread_km" in features.columns
    assert "wind_onshore_north_hub_height_m_cap_weighted" in features.columns
    assert "wind_onshore_north_manufacturer_capacity_hhi" in features.columns
    assert "wind_onshore_north_speed_hub_actual_cap_weighted_m_s" in features.columns
    assert "wind_onshore_north_proxy_v3_hub_actual_capacity_x_rotor_area_weighted" in features.columns
    assert "wind_onshore_north_vmax_hub_cap_weighted_m_s" in features.columns
    assert "wind_onshore_north_t120m_minus_t80m_cap_weighted_K" in features.columns
    assert "wind_onshore_north_relative_humidity_2m_cap_weighted_pct" in features.columns
    assert "wind_onshore_north_low_cloud_cover_cap_weighted" in features.columns
    assert "wind_onshore_north_proxy_mw" in features.columns
    assert "solar_north_proxy_mw" in features.columns
    assert "solar_north_direct_normal_irradiance_cap_weighted_W_m2" in features.columns
    assert "solar_north_vapour_pressure_deficit_cap_weighted_kpa" in features.columns
    assert "solar_north_high_cloud_cover_cap_weighted" in features.columns
    assert "solar_north_diffuse_irradiance_cap_weighted_W_m2" in features.columns
    assert "Renewable_Total_Proxy_MW" in features.columns
    assert len(features) == 8


def test_capacity_weather_points_can_use_high_capacity_cells(tmp_path: Path) -> None:
    capacity_file = tmp_path / "capacity.csv"
    pd.DataFrame(
        {
            "technology": ["wind_onshore"] * 4,
            "capacity_mw": [80.0, 10.0, 60.0, 5.0],
            "lat": [53.01, 53.02, 51.01, 49.01],
            "lon": [8.01, 8.02, 10.01, 12.01],
            "commissioning_date": ["2020-01-01"] * 4,
            "decommissioning_date": [None] * 4,
            "operating_status": ["35"] * 4,
            "federal_state_code": [1408, 1408, 1414, 1403],
        }
    ).to_csv(capacity_file, index=False)

    cluster_file = tmp_path / "clusters.csv"
    pd.DataFrame(
        {
            "cluster_id": [0, 1, 2],
            "lat": [53.0, 51.0, 49.0],
            "lon": [8.0, 10.0, 12.0],
        }
    ).to_csv(cluster_file, index=False)

    config = RegionalRenewableFeatureConfig(
        repo_root=tmp_path,
        capacity_file=capacity_file,
        cluster_file=cluster_file,
        capacity_map_file=tmp_path / "capacity_map.csv",
        output_file=tmp_path / "features.csv",
        capacity_weather_technologies=["wind_onshore"],
        capacity_weather_point_strategy="capacity_cells",
        capacity_weather_cell_size_degrees=0.1,
        capacity_weather_onshore_points_per_region=1,
    )

    capacity_map = rrf.build_capacity_map(config)
    points = rrf.build_capacity_weather_points(capacity_map, config)

    assert set(points["region"]) == {"onshore_central", "onshore_north", "onshore_south"}
    north = points.loc[points["region"] == "onshore_north"].iloc[0]
    assert np.isclose(north["capacity_mw"], 90.0)
    assert np.isclose(north["lat"], (53.01 * 80.0 + 53.02 * 10.0) / 90.0)
