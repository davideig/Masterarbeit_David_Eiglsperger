from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import EvaluationConfig, EvaluationForecastConfig
from .metrics import (
    aggregated_pinball_score,
    aps_loss_matrix,
    bias_point,
    empirical_coverage,
    gw_loss_test,
    gw_test_point,
    mae_for_median,
    mae_point,
    mtu_kupiec_test,
    quantile_column,
    rmse_point,
)


def read_forecast_csv(path: Path, target_tz: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(target_tz)
    df.index.name = "timestamp"
    return df


def enforce_eval_window(
    df: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    skip_dates: set[pd.Timestamp],
) -> pd.DataFrame:
    out = df.loc[start:end].copy()
    if skip_dates:
        out = out.loc[~out.index.normalize().isin(skip_dates)]
    return out


def _expected_index(config: EvaluationConfig) -> pd.DatetimeIndex:
    start = pd.Timestamp(config.test_start)
    end = pd.Timestamp(config.test_end)
    expected = pd.date_range(start=start, end=end, freq=config.frequency, tz=config.target_tz)
    skip_dates = {pd.Timestamp(day, tz=config.target_tz).normalize() for day in config.skip_dates}
    if skip_dates:
        expected = expected[~expected.normalize().isin(skip_dates)]
    return expected


def _validate_index(df: pd.DataFrame, name: str, expected: pd.DatetimeIndex, n_periods: int) -> None:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(f"[{name}] Index must be a DatetimeIndex.")
    missing = expected.difference(df.index)
    extra = df.index.difference(expected)
    if len(missing) or len(extra):
        raise ValueError(
            f"[{name}] Index does not match expected grid. "
            f"Missing={len(missing)}, extra={len(extra)}."
        )
    if len(df) % n_periods != 0:
        raise ValueError(f"[{name}] len(df)={len(df)} is not divisible by n_periods={n_periods}.")


def _quantile_columns(config: EvaluationConfig, forecast: EvaluationForecastConfig) -> list[str]:
    return forecast.quantile_columns or [quantile_column(q) for q in config.quantiles]


def _load_point_panel(config: EvaluationConfig) -> pd.DataFrame:
    expected = _expected_index(config)
    skip_dates = {pd.Timestamp(day, tz=config.target_tz).normalize() for day in config.skip_dates}
    y_true = None
    frames = []

    for forecast in config.point_forecasts:
        df = enforce_eval_window(
            read_forecast_csv(forecast.path, config.target_tz),
            pd.Timestamp(config.test_start),
            pd.Timestamp(config.test_end),
            skip_dates,
        )
        if config.validate_inputs:
            _validate_index(df, forecast.name, expected, config.n_periods)
        if forecast.value_column not in df.columns:
            raise ValueError(f"[{forecast.name}] Missing value column '{forecast.value_column}'.")
        if y_true is None and forecast.y_true_column in df.columns:
            y_true = df[[forecast.y_true_column]].rename(columns={forecast.y_true_column: config.y_true_column})
        frames.append(df[[forecast.value_column]].rename(columns={forecast.value_column: forecast.name}))

    if not frames:
        return pd.DataFrame()
    if y_true is None:
        raise ValueError("Could not find y_true in any point forecast file.")
    panel = pd.concat([y_true, *frames], axis=1).sort_index()
    if config.validate_inputs and panel.drop(columns=[config.y_true_column]).isna().any().any():
        missing_cols = panel.columns[panel.isna().any()].tolist()
        raise ValueError(f"Point forecast panel contains missing values: {missing_cols}")
    return panel


def _load_quantile_panels(config: EvaluationConfig) -> dict[str, pd.DataFrame]:
    expected = _expected_index(config)
    skip_dates = {pd.Timestamp(day, tz=config.target_tz).normalize() for day in config.skip_dates}
    panels: dict[str, pd.DataFrame] = {}

    for forecast in config.quantile_forecasts:
        df = enforce_eval_window(
            read_forecast_csv(forecast.path, config.target_tz),
            pd.Timestamp(config.test_start),
            pd.Timestamp(config.test_end),
            skip_dates,
        )
        if config.validate_inputs:
            _validate_index(df, forecast.name, expected, config.n_periods)
        qcols = _quantile_columns(config, forecast)
        missing = [col for col in qcols if col not in df.columns]
        if missing:
            raise ValueError(f"[{forecast.name}] Missing quantile columns: {missing}")
        if forecast.y_true_column not in df.columns:
            raise ValueError(f"[{forecast.name}] Missing y_true column '{forecast.y_true_column}'.")
        panel = pd.concat(
            [
                df[[forecast.y_true_column]].rename(columns={forecast.y_true_column: config.y_true_column}),
                df[qcols].rename(columns={src: quantile_column(q) for src, q in zip(qcols, config.quantiles, strict=True)}),
            ],
            axis=1,
        )
        if config.validate_inputs:
            if panel.isna().any().any():
                missing_cols = panel.columns[panel.isna().any()].tolist()
                raise ValueError(f"[{forecast.name}] Quantile panel contains missing values: {missing_cols}")
            qmat = panel[[quantile_column(q) for q in config.quantiles]].to_numpy()
            if not np.all(np.diff(qmat, axis=1) >= -1e-12):
                raise ValueError(f"[{forecast.name}] Quantile columns are not monotone.")
        panels[forecast.name] = panel

    return panels


def _point_metrics(panel: pd.DataFrame, y_true_col: str) -> pd.DataFrame:
    rows = []
    for model in [col for col in panel.columns if col != y_true_col]:
        valid = panel[[y_true_col, model]].dropna()
        rows.append(
            {
                "model": model,
                "mae": mae_point(panel, model, y_true_col),
                "rmse": rmse_point(panel, model, y_true_col),
                "bias": bias_point(panel, model, y_true_col),
                "n_obs": int(len(valid)),
            }
        )
    return pd.DataFrame(rows).sort_values("mae")


def _pairwise_point_gw(panel: pd.DataFrame, config: EvaluationConfig, norm: int = 1) -> pd.DataFrame:
    models = [col for col in panel.columns if col != config.y_true_column]
    rows = []
    for baseline in models:
        for challenger in models:
            if baseline == challenger:
                p_value = np.nan
            else:
                p_value = gw_test_point(
                    panel,
                    baseline,
                    challenger,
                    y_true_col=config.y_true_column,
                    n_periods=config.n_periods,
                    norm=norm,
                    version="multivariate",
                )
            rows.append({"baseline": baseline, "challenger": challenger, "p_value": p_value, "norm": norm})
    return pd.DataFrame(rows)


def _quantile_metrics(panels: dict[str, pd.DataFrame], config: EvaluationConfig) -> pd.DataFrame:
    rows = []
    for model, panel in panels.items():
        rows.append(
            {
                "model": model,
                "mae_median": mae_for_median(panel, config.y_true_column),
                "aps": aggregated_pinball_score(panel, config.quantiles, config.y_true_column),
                "n_obs": int(len(panel.dropna())),
            }
        )
    return pd.DataFrame(rows).sort_values("aps")


def _coverage_metrics(panels: dict[str, pd.DataFrame], config: EvaluationConfig) -> pd.DataFrame:
    rows = []
    for model, panel in panels.items():
        for q_low, q_high in config.prediction_intervals:
            nominal = q_high - q_low
            rows.append(
                {
                    "model": model,
                    "interval": f"PI_{int(q_low * 100)}_{int(q_high * 100)}",
                    "q_low": q_low,
                    "q_high": q_high,
                    "nominal_coverage": nominal,
                    "empirical_coverage": empirical_coverage(panel, q_low, q_high, config.y_true_column),
                    "kupiec_mtu_passed": mtu_kupiec_test(
                        panel,
                        q_low,
                        q_high,
                        alpha=nominal,
                        y_true_col=config.y_true_column,
                        significance_level=config.significance_level,
                    ),
                }
            )
    return pd.DataFrame(rows)


def _pairwise_quantile_gw(panels: dict[str, pd.DataFrame], config: EvaluationConfig) -> pd.DataFrame:
    losses = {
        model: aps_loss_matrix(panel, config.quantiles, config.y_true_column, config.n_periods)
        for model, panel in panels.items()
    }
    rows = []
    for baseline in losses:
        for challenger in losses:
            if baseline == challenger:
                p_value = np.nan
            else:
                p_value = gw_loss_test(losses[baseline], losses[challenger], version="multivariate")
            rows.append({"baseline": baseline, "challenger": challenger, "p_value": p_value, "loss": "aps"})
    return pd.DataFrame(rows)


def run_evaluation(config: EvaluationConfig) -> dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}

    point_panel = _load_point_panel(config)
    if not point_panel.empty:
        point_panel.to_csv(config.output_dir / "point_panel.csv")
        point_metrics = _point_metrics(point_panel, config.y_true_column)
        point_metrics.to_csv(config.output_dir / "point_metrics.csv", index=False)
        results["point_metrics"] = point_metrics
        if config.run_gw_tests and len(point_metrics) > 1:
            point_gw = _pairwise_point_gw(point_panel, config, norm=1)
            point_gw.to_csv(config.output_dir / "point_gw_mae.csv", index=False)
            results["point_gw_mae"] = point_gw

    quantile_panels = _load_quantile_panels(config)
    if quantile_panels:
        quantile_panel = pd.concat(
            {
                model: panel.drop(columns=[config.y_true_column])
                for model, panel in quantile_panels.items()
            },
            axis=1,
        )
        quantile_panel.columns = [f"{model}_{col}" for model, col in quantile_panel.columns]
        y_true = next(iter(quantile_panels.values()))[[config.y_true_column]]
        pd.concat([y_true, quantile_panel], axis=1).to_csv(config.output_dir / "quantile_panel.csv")

        quantile_metrics = _quantile_metrics(quantile_panels, config)
        coverage_metrics = _coverage_metrics(quantile_panels, config)
        quantile_metrics.to_csv(config.output_dir / "quantile_metrics.csv", index=False)
        coverage_metrics.to_csv(config.output_dir / "quantile_coverage.csv", index=False)
        results["quantile_metrics"] = quantile_metrics
        results["quantile_coverage"] = coverage_metrics
        if config.run_gw_tests and len(quantile_panels) > 1:
            quantile_gw = _pairwise_quantile_gw(quantile_panels, config)
            quantile_gw.to_csv(config.output_dir / "quantile_gw_aps.csv", index=False)
            results["quantile_gw_aps"] = quantile_gw

    with open(config.output_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(config.model_dump(mode="json"), handle, indent=2)

    print(f"✓ Saved evaluation outputs -> {config.output_dir}")
    return results
