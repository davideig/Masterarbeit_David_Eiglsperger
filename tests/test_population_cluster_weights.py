from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

from da_price_forecasting.config import PopulationClusterWeightsConfig, RunConfig, RunKind
from da_price_forecasting.preprocessing.population_cluster_weights import build_population_cluster_weights
from da_price_forecasting.scripts.run import run_from_config


def _project(lon: list[float], lat: list[float]) -> tuple[np.ndarray, np.ndarray]:
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    x, y = transformer.transform(lon, lat)
    return np.asarray(x), np.asarray(y)


def test_population_cluster_weights_match_population_cells(tmp_path: Path) -> None:
    cluster_file = tmp_path / "clusters.csv"
    population_file = tmp_path / "population.csv"
    output_file = tmp_path / "weights.csv"
    region_output_file = tmp_path / "region_weights.csv"

    clusters = pd.DataFrame(
        {
            "lon": [10.0, 11.0],
            "lat": [50.0, 50.0],
            "cluster_id": [0, 1],
        }
    )
    clusters.to_csv(cluster_file, index=False)

    x, y = _project([10.0, 10.02, 11.0, 11.01], [50.0, 50.01, 50.0, 50.01])
    population = pd.DataFrame(
        {
            "X_LLC": x - 500.0,
            "Y_LLC": y - 500.0,
            "TOT_P_2021": [100.0, 50.0, 200.0, 25.0],
            "CNTR_ID": ["DE", "DE", "LU", "FR"],
            "NUTS2021_1": ["DE1", "DE1", "LU0-BE3", "FR1"],
        }
    )
    population.to_csv(population_file, index=False)

    config = PopulationClusterWeightsConfig(
        repo_root=tmp_path,
        source_file=population_file,
        source_url=None,
        cluster_file=cluster_file,
        output_file=output_file,
        region_output_file=region_output_file,
        country_codes=["DE", "LU"],
    )

    result = build_population_cluster_weights(config)
    weights = result["cluster_weights"]
    region_weights = result["region_weights"]

    assert isinstance(weights, pd.DataFrame)
    assert isinstance(region_weights, pd.DataFrame)
    assert weights.set_index("cluster_id").loc[0, "weight"] == 150.0
    assert weights.set_index("cluster_id").loc[1, "weight"] == 200.0
    assert region_weights.set_index(["cluster_id", "region"]).loc[(0, "DE1"), "weight"] == 150.0
    assert region_weights.set_index(["cluster_id", "region"]).loc[(1, "LU0"), "weight"] == 200.0
    assert "BE3" not in set(region_weights["region"])


def test_population_cluster_weights_dispatcher_writes_outputs(tmp_path: Path) -> None:
    cluster_file = tmp_path / "clusters.csv"
    population_file = tmp_path / "population.csv"
    output_file = tmp_path / "weights.csv"

    pd.DataFrame({"lon": [10.0], "lat": [50.0], "cluster_id": [0]}).to_csv(cluster_file, index=False)
    x, y = _project([10.0], [50.0])
    pd.DataFrame(
        {
            "X_LLC": x - 500.0,
            "Y_LLC": y - 500.0,
            "TOT_P_2021": [123.0],
            "CNTR_ID": ["DE"],
        }
    ).to_csv(population_file, index=False)

    run_from_config(
        RunConfig(
            repo_root=tmp_path,
            kind=RunKind.POPULATION_CLUSTER_WEIGHTS,
            config={
                "source_file": str(population_file),
                "source_url": None,
                "cluster_file": str(cluster_file),
                "output_file": str(output_file),
                "region_output_file": None,
            },
        )
    )

    weights = pd.read_csv(output_file)
    assert weights.loc[0, "cluster_id"] == 0
    assert weights.loc[0, "weight"] == 123.0
    assert output_file.with_suffix(".json").exists()
