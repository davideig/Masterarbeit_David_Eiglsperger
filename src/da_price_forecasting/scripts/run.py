from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ..config import (
    EntsoeLoadForecastBenchmarkConfig,
    EntsoeRenewableForecastBenchmarkConfig,
    EvaluationConfig,
    IconAggregationConfig,
    LearAncConfig,
    LearOperationalConfig,
    LoadForecastEnsembleConfig,
    LoadForecastModelConfig,
    MastrCapacityConfig,
    PopulationClusterWeightsConfig,
    RegionalLoadForecastConfig,
    RegionalRenewableFeatureConfig,
    RenewableGenerationModelConfig,
    RenewableGenerationPostprocessConfig,
    RenewableProxyConfig,
    ReserveMarketConfig,
    RunConfig,
    RunKind,
    load_config,
    validate_config_payload,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one configured forecasting pipeline.")
    parser.add_argument("--config", type=Path, required=True, help="Path to a top-level run YAML config.")
    parser.add_argument("--submit", action="store_true", help="Force API submission for energy_arena_submit runs.")
    return parser


def _load_nested_config(run_config: RunConfig, model_cls: type[Any]) -> Any:
    if run_config.config_path is not None:
        return load_config(run_config.config_path, model_cls)
    return validate_config_payload(run_config.config, model_cls, repo_root=run_config.repo_root)


def run_from_config(run_config: RunConfig, submit_override: bool = False) -> None:
    """Dispatch a typed run config to the selected pipeline."""
    kind = run_config.kind


    if kind == RunKind.ENTSOE_LOAD_FORECAST_BENCHMARK:
        from ..pipelines.load_forecast import run_entsoe_load_forecast_benchmark

        config = _load_nested_config(run_config, EntsoeLoadForecastBenchmarkConfig)
        run_entsoe_load_forecast_benchmark(config, save_outputs=run_config.save_outputs)
        return

    if kind == RunKind.ENTSOE_RENEWABLE_FORECAST_BENCHMARK:
        from ..pipelines.renewable_generation import run_entsoe_renewable_forecast_benchmark

        config = _load_nested_config(run_config, EntsoeRenewableForecastBenchmarkConfig)
        run_entsoe_renewable_forecast_benchmark(config, save_outputs=run_config.save_outputs)
        return



    if kind == RunKind.LEAR_OPERATIONAL:
        from ..pipelines.lear import run_lear_operational_pipeline

        config = _load_nested_config(run_config, LearOperationalConfig)
        run_lear_operational_pipeline(config=config, save_outputs=run_config.save_outputs, plot=run_config.plot)
        return

    if kind == RunKind.LEAR_ANC:
        from ..pipelines.lear import run_lear_anc_pipeline

        config = _load_nested_config(run_config, LearAncConfig)
        run_lear_anc_pipeline(config=config, save_outputs=run_config.save_outputs)
        return

    if kind == RunKind.LOAD_FORECAST_MODEL:
        from ..pipelines.load_forecast import run_load_forecast_pipeline

        config = _load_nested_config(run_config, LoadForecastModelConfig)
        run_load_forecast_pipeline(config=config, save_outputs=run_config.save_outputs)
        return

    if kind == RunKind.LOAD_FORECAST_ENSEMBLE:
        from ..pipelines.load_forecast import run_load_forecast_ensemble_pipeline

        config = _load_nested_config(run_config, LoadForecastEnsembleConfig)
        run_load_forecast_ensemble_pipeline(config=config, save_outputs=run_config.save_outputs)
        return

    if kind == RunKind.REGIONAL_LOAD_FORECAST:
        from ..pipelines.load_forecast import run_regional_load_forecast_pipeline

        config = _load_nested_config(run_config, RegionalLoadForecastConfig)
        run_regional_load_forecast_pipeline(config=config, save_outputs=run_config.save_outputs)
        return






    if kind == RunKind.ICON_AGGREGATE:
        from ..preprocessing.icon_d2_aggregation import run_aggregation

        config = _load_nested_config(run_config, IconAggregationConfig)
        run_aggregation(config)
        return

    if kind == RunKind.MASTR_CAPACITY:
        from ..preprocessing.mastr_capacity import run_mastr_capacity

        config = _load_nested_config(run_config, MastrCapacityConfig)
        run_mastr_capacity(config)
        return

    if kind == RunKind.POPULATION_CLUSTER_WEIGHTS:
        from ..preprocessing.population_cluster_weights import run_population_cluster_weights

        config = _load_nested_config(run_config, PopulationClusterWeightsConfig)
        run_population_cluster_weights(config)
        return

    if kind == RunKind.REGIONAL_RENEWABLE_FEATURES:
        from ..preprocessing.regional_renewable_features import run_regional_renewable_features

        config = _load_nested_config(run_config, RegionalRenewableFeatureConfig)
        run_regional_renewable_features(config)
        return

    if kind == RunKind.RENEWABLE_GENERATION_MODEL:
        from ..pipelines.renewable_generation import run_renewable_generation_pipeline

        config = _load_nested_config(run_config, RenewableGenerationModelConfig)
        run_renewable_generation_pipeline(config, save_outputs=run_config.save_outputs)
        return

    if kind == RunKind.RENEWABLE_GENERATION_POSTPROCESS:
        from ..pipelines.renewable_generation import run_renewable_generation_postprocess_pipeline

        config = _load_nested_config(run_config, RenewableGenerationPostprocessConfig)
        run_renewable_generation_postprocess_pipeline(config, save_outputs=run_config.save_outputs)
        return

    if kind == RunKind.RENEWABLE_PROXY:
        from ..preprocessing.renewable_proxy import run_renewable_proxy

        config = _load_nested_config(run_config, RenewableProxyConfig)
        run_renewable_proxy(config)
        return

    if kind == RunKind.RESERVE_MARKET:
        from ..preprocessing.reserve_market import run_reserve_market

        config = _load_nested_config(run_config, ReserveMarketConfig)
        run_reserve_market(config)
        return



    if kind == RunKind.EVALUATION:
        from ..evaluation import run_evaluation

        config = _load_nested_config(run_config, EvaluationConfig)
        run_evaluation(config)
        return

    raise ValueError(f"Unsupported run kind: {kind}")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_config = load_config(args.config, RunConfig)
    run_from_config(run_config, submit_override=args.submit)


if __name__ == "__main__":
    main()
