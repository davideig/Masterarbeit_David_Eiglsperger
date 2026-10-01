from __future__ import annotations

import warnings
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from ..config import LearAncConfig, LearOperationalConfig, WeatherSource
from ..data.entsoe import fetch_load_forecast, fetch_prices, fetch_prices_exaa
from ..features.covariates import (
    build_daily_scalar_covariate_features,
    build_daily_summary_covariate_features,
    build_daily_vector_covariate_features,
    build_timestamp_covariates,
    load_load_forecast_covariates,
)
from ..data.weather import load_dwd, load_era5
from ..features.engineering import (
    build_dwd_features,
    build_era5_features,
    build_load_features,
    build_price_features,
    build_temporal_features,
    build_y_matrix,
    merge_all_features,
)
from ..models.lear import (
    apply_rolling_forecast_bias_correction,
    evaluate_and_plot_forecast_from_df,
    rolling_anc_feature_importance,
    rolling_point_forecast,
    save_anc_outputs,
    save_experiment_outputs,
    save_prediction_outputs,
    summarize_feature_group_anc,
    summarize_solar_cluster_anc,
    summarize_wind_cluster_anc,
)
from .common import (
    as_local_day as _as_local_day,
    load_timestamp_csv as _load_timestamp_csv,
    save_timestamp_csv as _save_timestamp_csv,
)


def _load_env(repo_root) -> None:
    load_dotenv(repo_root / ".env")


def _normalise_timestamp_index(df: pd.DataFrame, target_tz: str) -> pd.DataFrame:
    if df.empty:
        out = df.copy()
        out.index.name = "timestamp"
        return out
    out = df.copy()
    out.index = pd.to_datetime(out.index, utc=True).tz_convert(target_tz)
    out = out.sort_index()
    out = out.loc[~out.index.duplicated(keep="last")]
    out.index.name = "timestamp"
    return out


def _combine_timestamp_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    non_empty_frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not non_empty_frames:
        return pd.DataFrame()
    out = pd.concat(non_empty_frames)
    out = out.loc[~out.index.duplicated(keep="last")]
    out = out.sort_index()
    out.index.name = "timestamp"
    return out


def _feature_availability_mask(X: pd.DataFrame, price_model_type: str) -> pd.Series:
    """Return rows usable by the selected price model family."""
    if price_model_type == "lear":
        return X.notna().all(axis=1)
    return X.notna().any(axis=1)


def _restrict_timestamp_window(
    df: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    target_tz: str,
) -> pd.DataFrame:
    if df.empty:
        return df
    start_cut = _as_local_day(start, target_tz)
    end_cut = _as_local_day(end, target_tz) + pd.Timedelta(days=1) - pd.Timedelta(minutes=15)
    return df.loc[start_cut:end_cut]


def _load_or_fetch_price_cache(
    *,
    path,
    target_tz: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    fetch_window,
    label: str,
) -> pd.DataFrame:
    start = _as_local_day(start, target_tz)
    end = _as_local_day(end, target_tz)
    frames: list[pd.DataFrame] = []
    fetched_new_data = False

    if path.exists():
        cached = _load_timestamp_csv(path, target_tz)
        if not cached.empty:
            frames.append(cached)
            cached_days = pd.DatetimeIndex(cached.index).tz_convert(target_tz).normalize()
            cached_start = cached_days.min()
            cached_end = cached_days.max()
            missing_ranges: list[tuple[pd.Timestamp, pd.Timestamp]] = []
            if start < cached_start:
                missing_ranges.append((start, cached_start - pd.Timedelta(days=1)))
            if end > cached_end:
                missing_ranges.append((cached_end + pd.Timedelta(days=1), end))
        else:
            missing_ranges = [(start, end)]
    else:
        missing_ranges = [(start, end)]

    for fetch_start, fetch_end in missing_ranges:
        if fetch_start > fetch_end:
            continue
        try:
            fetched = fetch_window(fetch_start, fetch_end)
        except Exception as exc:
            if frames:
                warnings.warn(
                    f"Failed to refresh {label} for {fetch_start.date()}..{fetch_end.date()}; "
                    f"using available cache from {path}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            raise
        if not fetched.empty:
            frames.append(_normalise_timestamp_index(fetched, target_tz))
            fetched_new_data = True

    if not frames:
        raise ValueError(f"No {label} data available for {start.date()}..{end.date()}.")

    df = _combine_timestamp_frames(frames)
    if fetched_new_data or not path.exists():
        _save_timestamp_csv(df, path)
    return _restrict_timestamp_window(df, start, end, target_tz)


def _load_weather_features(
    weather_source: WeatherSource,
    era5_dirs,
    icon_dir,
    start_folder_date,
    required_run,
    skip_dates,
    folder_offset_date,
    target_tz: str,
):
    if weather_source == WeatherSource.ERA5:
        df_era5 = load_era5(dirs=era5_dirs, target_tz=target_tz)
        return build_era5_features(df_era5=df_era5)

    df_dwd_hourly, df_dwd_qh = load_dwd(
        icon_dir=icon_dir,
        start_folder_date=start_folder_date,
        required_run=required_run,
        skip_dates=set(skip_dates),
        folder_offset_date=folder_offset_date,
        target_tz=target_tz,
    )
    return build_dwd_features(df_hourly=df_dwd_hourly, df_qh=df_dwd_qh, tz_local=target_tz)


def _load_or_skip_weather_features(
    config: LearOperationalConfig | LearAncConfig,
    daily_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    if not config.include_raw_weather_features:
        return pd.DataFrame(index=daily_index).rename_axis("date")

    return _load_weather_features(
        weather_source=config.weather_source,
        era5_dirs=config.era5_dirs,
        icon_dir=config.icon_dir,
        start_folder_date=config.start_folder_date,
        required_run=config.required_run,
        skip_dates=config.skip_dates,
        folder_offset_date=config.dwd_folder_offset_date,
        target_tz=config.target_tz,
    )


def _build_lear_extra_covariate_features(
    config: LearOperationalConfig | LearAncConfig,
    start_day: pd.Timestamp,
    end_day: pd.Timestamp,
    daily_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Build LEAR-compatible daily features from shared timestamp covariates."""
    extra_covariates = [
        covariate
        for covariate in config.features.covariates
        if covariate not in {"exaa", "load_forecast"}
    ]
    if not extra_covariates:
        return pd.DataFrame(index=daily_index).rename_axis("date")

    vector_covariates = [
        covariate
        for covariate in extra_covariates
        if covariate in {
            "ntc",
            "generation_unavailability",
            "renewable_generation_proxy",
            "reserve_market",
            "foreign_day_ahead_prices",
            "foreign_day_ahead_price_spreads",
        }
    ]
    scalar_covariates = [covariate for covariate in extra_covariates if covariate not in set(vector_covariates)]

    feature_blocks = []
    if vector_covariates:
        fetch_covariates = list(vector_covariates)
        if "renewable_generation_proxy" in fetch_covariates and "load_forecast" in config.features.covariates:
            fetch_covariates.append("load_forecast")
        covariate_config = config.features.model_copy(update={"covariates": fetch_covariates})
        timestamp_covariates = build_timestamp_covariates(
            covariate_config,
            start_day=start_day,
            end_day=end_day,
            country_code_entsoe=config.country_code_entsoe,
            entsoe_api_key_env=config.entsoe_api_key_env,
            target_tz=config.target_tz,
        )
        vector_columns = [
            column
            for column in timestamp_covariates.columns
            if column != "load_fc" and not column.startswith("price_exaa")
        ]
        feature_blocks.append(
            build_daily_vector_covariate_features(timestamp_covariates, daily_index, columns=vector_columns)
        )

    if scalar_covariates:
        covariate_config = config.features.model_copy(update={"covariates": scalar_covariates})
        timestamp_covariates = build_timestamp_covariates(
            covariate_config,
            start_day=start_day,
            end_day=end_day,
            country_code_entsoe=config.country_code_entsoe,
            entsoe_api_key_env=config.entsoe_api_key_env,
            target_tz=config.target_tz,
        )
        feature_blocks.append(build_daily_scalar_covariate_features(timestamp_covariates, daily_index))

    if not feature_blocks:
        return pd.DataFrame(index=daily_index).rename_axis("date")

    features = pd.concat(feature_blocks, axis=1, sort=False)
    features = features.loc[:, ~features.columns.duplicated()]
    features.index.name = "date"
    return features


def _build_lear_scarcity_features(
    config: LearOperationalConfig | LearAncConfig,
    start_day: pd.Timestamp,
    end_day: pd.Timestamp,
    daily_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Build compact load/NTC/unavailability summaries for spike/scarcity regimes."""
    covariate_config = config.features.model_copy(
        update={"covariates": ["load_forecast", "ntc", "generation_unavailability"]}
    )
    timestamp_covariates = build_timestamp_covariates(
        covariate_config,
        start_day=start_day,
        end_day=end_day,
        country_code_entsoe=config.country_code_entsoe,
        entsoe_api_key_env=config.entsoe_api_key_env,
        target_tz=config.target_tz,
    )
    return build_daily_summary_covariate_features(
        timestamp_covariates,
        daily_index,
        prefix="scarcity_",
    )


def _join_extra_feature_blocks(X: pd.DataFrame, extra_feature_blocks: list[pd.DataFrame]) -> pd.DataFrame:
    out = X
    for feature_block in extra_feature_blocks:
        if feature_block is not None and not feature_block.empty:
            out = out.join(feature_block, how="inner").sort_index()
    out.index.name = "date"
    return out


def _build_core_load_features(
    config: LearOperationalConfig,
    start_day: pd.Timestamp,
    end_day: pd.Timestamp,
    daily_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    if not config.include_load_forecast_features:
        return pd.DataFrame(index=daily_index).rename_axis("date")

    load_forecast = load_load_forecast_covariates(
        config.features,
        start_day=start_day,
        end_day=end_day,
        country_code_entsoe=config.country_code_entsoe,
        entsoe_api_key_env=config.entsoe_api_key_env,
        target_tz=config.target_tz,
    )
    return build_load_features(load_forecast).reindex(daily_index)


def prepare_lear_operational_dataset(config: LearOperationalConfig) -> dict[str, Any]:
    """Load data and construct X/Y for the operational LEAR run."""
    _load_env(config.repo_root)

    entsoe_start = pd.Timestamp(config.entsoe_start_date)
    entsoe_end = pd.Timestamp(config.entsoe_end_date)

    df_prices_15 = _load_or_fetch_price_cache(
        path=config.entsoe_price_file,
        target_tz=config.target_tz,
        start=entsoe_start,
        end=entsoe_end,
        label="ENTSO-E day-ahead prices",
        fetch_window=lambda fetch_start, fetch_end: fetch_prices(
            start_day=fetch_start,
            end_day=fetch_end,
            country_code=config.country_code_entsoe,
            api_key_env=config.entsoe_api_key_env,
            target_tz=config.target_tz,
        ),
    )

    df_prices_exaa_15 = None
    if config.use_exaa or config.use_exaa_only:
        df_prices_exaa_15 = _load_or_fetch_price_cache(
            path=config.entsoe_exaa_price_file,
            target_tz=config.target_tz,
            start=entsoe_start,
            end=entsoe_end,
            label="ENTSO-E EXAA prices",
            fetch_window=lambda fetch_start, fetch_end: fetch_prices_exaa(
                start_day=fetch_start,
                end_day=fetch_end,
                country_code=config.country_code_entsoe,
                api_key_env=config.entsoe_api_key_env,
                target_tz=config.target_tz,
            ),
        )

    daily_index = df_prices_15.index.normalize().unique().sort_values()
    extra_feature_blocks = [
        _build_lear_extra_covariate_features(
            config=config,
            start_day=entsoe_start,
            end_day=entsoe_end,
            daily_index=daily_index,
        )
    ]

    if config.use_exaa_only:
        X_lear_op = build_price_features(
            df_prices=df_prices_15,
            exaa_only=True,
            df_prices_exaa_15=df_prices_exaa_15,
        )
        if config.add_calendar_features:
            extra_feature_blocks.append(
                build_temporal_features(
                    daily_index=daily_index,
                    post_regime_start=pd.Timestamp(config.post_regime_start),
                )
            )
        if config.add_scarcity_features:
            extra_feature_blocks.append(
                _build_lear_scarcity_features(
                    config=config,
                    start_day=entsoe_start,
                    end_day=entsoe_end,
                    daily_index=daily_index,
                )
            )
        X_lear_op = _join_extra_feature_blocks(X_lear_op, extra_feature_blocks)
        dropped_X_lear_op = pd.DataFrame()
    else:
        df_weather_features = _load_or_skip_weather_features(config, daily_index)
        X_lear_op, dropped_X_lear_op = merge_all_features(
            df_weather_features=df_weather_features,
            df_price_features=build_price_features(
                df_prices=df_prices_15,
                exaa_vector=config.use_exaa,
                df_prices_exaa_15=df_prices_exaa_15 if config.use_exaa else None,
            ),
            df_load_features=_build_core_load_features(config, entsoe_start, entsoe_end, daily_index),
            df_time_features=build_temporal_features(
                daily_index=daily_index,
                post_regime_start=pd.Timestamp(config.post_regime_start),
            ),
            extra_feature_blocks=extra_feature_blocks,
            dropna=config.price_model_type == "lear",
        )

    Y_lear_op = build_y_matrix(df_prices_15, X_lear_op.index)
    valid_mask = _feature_availability_mask(X_lear_op, config.price_model_type)
    X_lear_op = X_lear_op.loc[valid_mask]
    Y_lear_op = Y_lear_op.loc[valid_mask]

    return {
        "X": X_lear_op,
        "Y": Y_lear_op,
        "dropped_rows": dropped_X_lear_op,
        "prices": df_prices_15,
        "prices_exaa": df_prices_exaa_15,
    }


def run_lear_operational_pipeline(
    config: LearOperationalConfig,
    save_outputs: bool = True,
    plot: bool = False,
) -> dict[str, Any]:
    """Run the operational LEAR pipeline outside of the notebook."""
    dataset = prepare_lear_operational_dataset(config)
    forecast_days = pd.date_range(
        start=pd.Timestamp(config.test_start).normalize(),
        end=pd.Timestamp(config.test_end).normalize(),
        freq="D",
        tz=config.target_tz,
    )

    forecast_df, runtime_df, coef_df, intercept_df, degenerate_df = rolling_point_forecast(
        X=dataset["X"],
        Y=dataset["Y"],
        forecast_days=forecast_days,
        train_days=config.train_days_rolling,
        lars_start_date=pd.Timestamp(config.lars_start_date),
        use_vst=config.use_vst,
        price_model_type=config.price_model_type,
        hgb_max_iter=config.hgb_max_iter,
        hgb_learning_rate=config.hgb_learning_rate,
        hgb_max_leaf_nodes=config.hgb_max_leaf_nodes,
        hgb_min_samples_leaf=config.hgb_min_samples_leaf,
        hgb_l2_regularization=config.hgb_l2_regularization,
        lgbm_n_estimators=config.lgbm_n_estimators,
        lgbm_learning_rate=config.lgbm_learning_rate,
        lgbm_num_leaves=config.lgbm_num_leaves,
        lgbm_min_child_samples=config.lgbm_min_child_samples,
        lgbm_subsample=config.lgbm_subsample,
        lgbm_colsample_bytree=config.lgbm_colsample_bytree,
        lgbm_reg_lambda=config.lgbm_reg_lambda,
        random_state=config.random_state,
        lasso_cv_eps=config.lasso_cv_eps,
        lasso_cv_alphas=config.lasso_cv_alphas,
        lasso_cv_tol=config.lasso_cv_tol,
        lasso_cv_max_iter=config.lasso_cv_max_iter,
        lars_max_iter=config.lars_max_iter,
        lars_max_n_alphas=config.lars_max_n_alphas,
    )
    bias_corrections_df = pd.DataFrame()
    if config.forecast_bias_correction in {"rolling_mean_error", "rolling_hour_mean_error"}:
        forecast_df, bias_corrections_df = apply_rolling_forecast_bias_correction(
            forecast_df=forecast_df,
            train_days=config.forecast_bias_train_days,
            min_train_days=config.forecast_bias_min_train_days,
            by_hour=config.forecast_bias_correction == "rolling_hour_mean_error",
        )
    elif config.forecast_bias_correction != "none":
        raise ValueError(f"Unsupported forecast bias correction: {config.forecast_bias_correction!r}")

    export_payload = {
        "experiment_name": config.experiment_name,
        "use_exaa": config.use_exaa,
        "use_exaa_only": config.use_exaa_only,
        "n_clusters": None if config.use_exaa_only else config.n_clusters,
        "weather_source": None if config.use_exaa_only else config.weather_source.value,
        "covariates": ",".join(config.features.covariates),
        "train_days": config.train_days_rolling,
        "use_vst": config.use_vst,
        "price_model_type": config.price_model_type,
        "hgb_max_iter": config.hgb_max_iter,
        "hgb_learning_rate": config.hgb_learning_rate,
        "hgb_max_leaf_nodes": config.hgb_max_leaf_nodes,
        "hgb_min_samples_leaf": config.hgb_min_samples_leaf,
        "hgb_l2_regularization": config.hgb_l2_regularization,
        "lgbm_n_estimators": config.lgbm_n_estimators,
        "lgbm_learning_rate": config.lgbm_learning_rate,
        "lgbm_num_leaves": config.lgbm_num_leaves,
        "lgbm_min_child_samples": config.lgbm_min_child_samples,
        "lgbm_subsample": config.lgbm_subsample,
        "lgbm_colsample_bytree": config.lgbm_colsample_bytree,
        "lgbm_reg_lambda": config.lgbm_reg_lambda,
        "random_state": config.random_state,
        "include_load_forecast_features": config.include_load_forecast_features,
        "load_forecast_file": config.features.load_forecast_file,
        "lars_start_date": str(pd.Timestamp(config.lars_start_date).date()),
        "lasso_cv_eps": config.lasso_cv_eps,
        "lasso_cv_alphas": config.lasso_cv_alphas,
        "lasso_cv_tol": config.lasso_cv_tol,
        "lasso_cv_max_iter": config.lasso_cv_max_iter,
        "lars_max_iter": config.lars_max_iter,
        "lars_max_n_alphas": config.lars_max_n_alphas,
        "forecast_bias_correction": config.forecast_bias_correction,
        "forecast_bias_train_days": config.forecast_bias_train_days,
        "forecast_bias_min_train_days": config.forecast_bias_min_train_days,
        "test_start": str(pd.Timestamp(config.test_start).date()),
        "test_end": str(pd.Timestamp(config.test_end).date()),
    }

    if save_outputs:
        save_experiment_outputs(
            name=config.experiment_name,
            forecast_df=forecast_df,
            runtime_df=runtime_df,
            config=export_payload,
            export_dir=config.resolved_export_dir,
        )
        if not bias_corrections_df.empty:
            bias_corrections_df.to_csv(config.resolved_export_dir / "bias_corrections.csv", index=False)

    if plot:
        evaluate_and_plot_forecast_from_df(
            forecasts=forecast_df,
            start=str(pd.Timestamp(config.test_start).date()),
            end=str(pd.Timestamp(config.test_end).date()),
            model_label=config.experiment_name,
            title=f"Rolling Point Forecast vs Actual Prices - {config.experiment_name}",
        )

    return {
        "forecast": forecast_df,
        "runtime": runtime_df,
        "coefficients": coef_df,
        "intercepts": intercept_df,
        "degenerate_ssrd": degenerate_df,
        "bias_corrections": bias_corrections_df,
        "dataset": dataset,
    }


def prepare_lear_operational_prediction_dataset(
    config: LearOperationalConfig,
    forecast_date: pd.Timestamp,
) -> dict[str, Any]:
    """Prepare a predict-only operational LEAR dataset for a future day."""
    _load_env(config.repo_root)

    entsoe_start = pd.Timestamp(config.entsoe_start_date)
    history_end = forecast_date - pd.Timedelta(days=1)
    feature_start = entsoe_start
    if feature_start.tz is None:
        feature_start = feature_start.tz_localize(config.target_tz)
    else:
        feature_start = feature_start.tz_convert(config.target_tz)
    daily_index = pd.date_range(
        start=feature_start.normalize(),
        end=forecast_date.normalize(),
        freq="D",
    )

    df_prices_15 = _load_or_fetch_price_cache(
        path=config.entsoe_price_file,
        target_tz=config.target_tz,
        start=entsoe_start,
        end=history_end,
        label="ENTSO-E day-ahead prices",
        fetch_window=lambda fetch_start, fetch_end: fetch_prices(
            start_day=fetch_start,
            end_day=fetch_end,
            country_code=config.country_code_entsoe,
            api_key_env=config.entsoe_api_key_env,
            target_tz=config.target_tz,
        ),
    )

    df_prices_exaa_15 = None
    if config.use_exaa or config.use_exaa_only:
        df_prices_exaa_15 = _load_or_fetch_price_cache(
            path=config.entsoe_exaa_price_file,
            target_tz=config.target_tz,
            start=entsoe_start,
            end=forecast_date,
            label="ENTSO-E EXAA prices",
            fetch_window=lambda fetch_start, fetch_end: fetch_prices_exaa(
                start_day=fetch_start,
                end_day=fetch_end,
                country_code=config.country_code_entsoe,
                api_key_env=config.entsoe_api_key_env,
                target_tz=config.target_tz,
            ),
        )

    extra_feature_blocks = [
        _build_lear_extra_covariate_features(
            config=config,
            start_day=feature_start,
            end_day=forecast_date,
            daily_index=daily_index,
        )
    ]

    if config.use_exaa_only:
        X_lear_op = build_price_features(
            df_prices=df_prices_15,
            exaa_only=True,
            df_prices_exaa_15=df_prices_exaa_15,
            daily_index=daily_index,
        )
        X_lear_op = _join_extra_feature_blocks(X_lear_op, extra_feature_blocks)
        dropped_X_lear_op = pd.DataFrame()
    else:
        df_weather_features = _load_or_skip_weather_features(config, daily_index)
        X_lear_op, dropped_X_lear_op = merge_all_features(
            df_weather_features=df_weather_features,
            df_price_features=build_price_features(
                df_prices=df_prices_15,
                exaa_vector=config.use_exaa,
                df_prices_exaa_15=df_prices_exaa_15 if config.use_exaa else None,
                daily_index=daily_index,
            ),
            df_load_features=_build_core_load_features(config, entsoe_start, forecast_date, daily_index),
            df_time_features=build_temporal_features(
                daily_index=daily_index,
                post_regime_start=pd.Timestamp(config.post_regime_start),
            ),
            extra_feature_blocks=extra_feature_blocks,
            dropna=config.price_model_type == "lear",
        )

    Y_lear_op = build_y_matrix(df_prices_15, X_lear_op.index)

    valid_mask = _feature_availability_mask(X_lear_op, config.price_model_type)
    X_lear_op = X_lear_op.loc[valid_mask]
    Y_lear_op = Y_lear_op.loc[valid_mask]

    return {
        "X": X_lear_op,
        "Y": Y_lear_op,
        "dropped_rows": dropped_X_lear_op,
        "prices": df_prices_15,
        "prices_exaa": df_prices_exaa_15,
    }


def run_lear_operational_prediction_pipeline(
    config: LearOperationalConfig,
    forecast_date: pd.Timestamp,
    save_outputs: bool = True,
    export_dir=None,
) -> dict[str, Any]:
    """Run a forecast-only LEAR pipeline for a single operational target day."""
    forecast_day = pd.Timestamp(forecast_date)
    if forecast_day.tz is None:
        forecast_day = forecast_day.tz_localize(config.target_tz)
    else:
        forecast_day = forecast_day.tz_convert(config.target_tz)
    forecast_day = forecast_day.normalize()
    dataset = prepare_lear_operational_prediction_dataset(config=config, forecast_date=forecast_day)

    forecast_df, runtime_df, coef_df, intercept_df, degenerate_df = rolling_point_forecast(
        X=dataset["X"],
        Y=dataset["Y"],
        forecast_days=[forecast_day],
        train_days=config.train_days_rolling,
        lars_start_date=pd.Timestamp(config.lars_start_date),
        use_vst=config.use_vst,
        price_model_type=config.price_model_type,
        hgb_max_iter=config.hgb_max_iter,
        hgb_learning_rate=config.hgb_learning_rate,
        hgb_max_leaf_nodes=config.hgb_max_leaf_nodes,
        hgb_min_samples_leaf=config.hgb_min_samples_leaf,
        hgb_l2_regularization=config.hgb_l2_regularization,
        lgbm_n_estimators=config.lgbm_n_estimators,
        lgbm_learning_rate=config.lgbm_learning_rate,
        lgbm_num_leaves=config.lgbm_num_leaves,
        lgbm_min_child_samples=config.lgbm_min_child_samples,
        lgbm_subsample=config.lgbm_subsample,
        lgbm_colsample_bytree=config.lgbm_colsample_bytree,
        lgbm_reg_lambda=config.lgbm_reg_lambda,
        random_state=config.random_state,
        lasso_cv_eps=config.lasso_cv_eps,
        lasso_cv_alphas=config.lasso_cv_alphas,
        lasso_cv_tol=config.lasso_cv_tol,
        lasso_cv_max_iter=config.lasso_cv_max_iter,
        lars_max_iter=config.lars_max_iter,
        lars_max_n_alphas=config.lars_max_n_alphas,
    )

    forecast_payload = {
        "experiment_name": config.experiment_name,
        "mode": "predict_only",
        "forecast_date": str(forecast_day.date()),
        "use_exaa": config.use_exaa,
        "use_exaa_only": config.use_exaa_only,
        "n_clusters": None if config.use_exaa_only else config.n_clusters,
        "weather_source": None if config.use_exaa_only else config.weather_source.value,
        "covariates": ",".join(config.features.covariates),
        "train_days": config.train_days_rolling,
        "use_vst": config.use_vst,
        "price_model_type": config.price_model_type,
        "include_load_forecast_features": config.include_load_forecast_features,
        "load_forecast_file": config.features.load_forecast_file,
        "lars_start_date": str(pd.Timestamp(config.lars_start_date).date()),
        "lasso_cv_eps": config.lasso_cv_eps,
        "lasso_cv_alphas": config.lasso_cv_alphas,
        "lasso_cv_tol": config.lasso_cv_tol,
        "lasso_cv_max_iter": config.lasso_cv_max_iter,
        "lars_max_iter": config.lars_max_iter,
        "lars_max_n_alphas": config.lars_max_n_alphas,
    }

    if save_outputs:
        save_prediction_outputs(
            forecast_df=forecast_df,
            runtime_df=runtime_df,
            config=forecast_payload,
            export_dir=export_dir or config.resolved_export_dir,
        )

    return {
        "forecast": forecast_df,
        "runtime": runtime_df,
        "coefficients": coef_df,
        "intercepts": intercept_df,
        "degenerate_ssrd": degenerate_df,
        "dataset": dataset,
    }


def prepare_lear_anc_dataset(config: LearAncConfig) -> dict[str, Any]:
    """Load data and construct X/Y for the ANC LEAR run."""
    _load_env(config.repo_root)

    entsoe_start = pd.Timestamp(config.entsoe_start_date)
    entsoe_end = pd.Timestamp(config.entsoe_end_date)

    df_prices_15 = fetch_prices(
        start_day=entsoe_start,
        end_day=entsoe_end,
        country_code=config.country_code_entsoe,
        api_key_env=config.entsoe_api_key_env,
        target_tz=config.target_tz,
    )

    df_prices_exaa_15 = None
    if config.use_exaa or config.use_exaa_only:
        df_prices_exaa_15 = fetch_prices_exaa(
            start_day=entsoe_start,
            end_day=entsoe_end,
            country_code=config.country_code_entsoe,
            api_key_env=config.entsoe_api_key_env,
            target_tz=config.target_tz,
        )

    daily_index = df_prices_15.index.normalize().unique().sort_values()
    extra_feature_blocks = [
        _build_lear_extra_covariate_features(
            config=config,
            start_day=entsoe_start,
            end_day=entsoe_end,
            daily_index=daily_index,
        )
    ]

    if config.use_exaa_only:
        X_lear_anc = build_price_features(
            df_prices=df_prices_15,
            exaa_only=True,
            df_prices_exaa_15=df_prices_exaa_15,
        )
        X_lear_anc = _join_extra_feature_blocks(X_lear_anc, extra_feature_blocks)
        dropped_X_lear_anc = pd.DataFrame()
    else:
        df_load_fc_entso_15 = fetch_load_forecast(
            start_day=entsoe_start,
            end_day=entsoe_end,
            country_code=config.country_code_entsoe,
            api_key_env=config.entsoe_api_key_env,
            target_tz=config.target_tz,
        )
        df_weather_features_anc = _load_or_skip_weather_features(config, daily_index)
        X_lear_anc, dropped_X_lear_anc = merge_all_features(
            df_weather_features=df_weather_features_anc,
            df_price_features=build_price_features(
                df_prices=df_prices_15,
                exaa_vector=config.use_exaa,
                df_prices_exaa_15=df_prices_exaa_15 if config.use_exaa else None,
            ),
            df_load_features=build_load_features(df_load_fc_entso_15),
            df_time_features=build_temporal_features(daily_index=daily_index),
            extra_feature_blocks=extra_feature_blocks,
        )

    Y_lear_anc = build_y_matrix(df_prices_15, X_lear_anc.index)
    valid_mask = X_lear_anc.notna().all(axis=1)
    X_lear_anc = X_lear_anc.loc[valid_mask]
    Y_lear_anc = Y_lear_anc.loc[valid_mask]

    return {
        "X": X_lear_anc,
        "Y": Y_lear_anc,
        "dropped_rows": dropped_X_lear_anc,
        "prices": df_prices_15,
        "prices_exaa": df_prices_exaa_15,
    }


def run_lear_anc_pipeline(config: LearAncConfig, save_outputs: bool = True) -> dict[str, Any]:
    """Run the ANC LEAR pipeline outside of the notebook."""
    dataset = prepare_lear_anc_dataset(config)
    forecast_days = pd.date_range(
        start=pd.Timestamp(config.test_start).normalize(),
        end=pd.Timestamp(config.test_end).normalize(),
        freq="D",
        tz=config.target_tz,
    )

    anc_df = rolling_anc_feature_importance(
        X=dataset["X"],
        Y=dataset["Y"],
        forecast_days=forecast_days,
        train_days=config.train_days_rolling,
        lars_start_date=pd.Timestamp(config.lars_start_date),
        lasso_cv_eps=config.lasso_cv_eps,
        lasso_cv_alphas=config.lasso_cv_alphas,
        lasso_cv_tol=config.lasso_cv_tol,
        lasso_cv_max_iter=config.lasso_cv_max_iter,
        lars_max_iter=config.lars_max_iter,
        lars_max_n_alphas=config.lars_max_n_alphas,
    )

    _, anc_summary_df = summarize_feature_group_anc(anc_df)
    wind_anc_summary, wind_anc_export = summarize_wind_cluster_anc(anc_df, config.mtu_window_wind)
    solar_anc_summary, solar_anc_export = summarize_solar_cluster_anc(anc_df, config.mtu_window_solar)

    export_payload = {
        "experiment_name": config.experiment_name,
        "use_exaa": config.use_exaa,
        "use_exaa_only": config.use_exaa_only,
        "n_clusters": None if config.use_exaa_only else config.n_clusters,
        "weather_source": None if config.use_exaa_only else config.weather_source.value,
        "covariates": ",".join(config.features.covariates),
        "train_days": config.train_days_rolling,
        "lars_start_date": str(pd.Timestamp(config.lars_start_date).date()),
        "lasso_cv_eps": config.lasso_cv_eps,
        "lasso_cv_alphas": config.lasso_cv_alphas,
        "lasso_cv_tol": config.lasso_cv_tol,
        "lasso_cv_max_iter": config.lasso_cv_max_iter,
        "lars_max_iter": config.lars_max_iter,
        "lars_max_n_alphas": config.lars_max_n_alphas,
        "test_start": str(pd.Timestamp(config.test_start).date()),
        "test_end": str(pd.Timestamp(config.test_end).date()),
    }

    if save_outputs:
        save_anc_outputs(
            export_dir=config.resolved_export_dir,
            anc_summary_df=anc_summary_df,
            wind_anc_export=wind_anc_export,
            solar_anc_export=solar_anc_export,
            config=export_payload,
        )

    return {
        "anc_raw": anc_df,
        "anc_summary": anc_summary_df,
        "wind_anc_summary": wind_anc_summary,
        "wind_anc_export": wind_anc_export,
        "solar_anc_summary": solar_anc_summary,
        "solar_anc_export": solar_anc_export,
        "dataset": dataset,
    }
