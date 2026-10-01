from __future__ import annotations

import holidays
import numpy as np
import pandas as pd


def build_era5_features(df_era5: pd.DataFrame) -> pd.DataFrame:
    """Build daily vector features from hourly ERA5 data."""
    df = df_era5.copy()

    ts_local = df.index.tz_convert("Europe/Berlin")
    df["date"] = ts_local.date
    df["hour"] = ts_local.hour

    u_cols = [col for col in df.columns if col.startswith("u10_cluster_")]
    for u_col in u_cols:
        cluster_id = u_col.split("_")[-1]
        v_col = f"v10_cluster_{cluster_id}"
        if v_col in df.columns:
            df[f"wind_speed_cluster_{cluster_id}"] = np.sqrt(df[u_col] ** 2 + df[v_col] ** 2)

    feature_cols = [col for col in df.columns if col.startswith("ssrd_cluster_")] + [
        col for col in df.columns if col.startswith("wind_speed_cluster_")
    ]

    df_features = df.pivot_table(
        index="date",
        columns="hour",
        values=feature_cols,
        aggfunc="mean",
    )
    df_features.columns = [f"{col_name}_h{int(hour):02d}" for col_name, hour in df_features.columns]
    df_features = df_features.interpolate(axis=1, limit=1, limit_area="inside")
    df_features.index = pd.to_datetime(df_features.index).tz_localize("Europe/Berlin")
    df_features.index.name = "date"
    return df_features


def build_dwd_features(
    df_hourly: pd.DataFrame,
    df_qh: pd.DataFrame,
    tz_local: str = "Europe/Berlin",
) -> pd.DataFrame:
    """Build DWD ICON-D2 daily vector features."""
    df_w = df_hourly.copy()
    ts_loc_w = df_w.index.tz_convert(tz_local)
    df_w = df_w.assign(date=ts_loc_w.date, hour=ts_loc_w.hour)

    u_cols = [col for col in df_w.columns if col.startswith("u10_cluster_")]
    wind_data: dict[str, np.ndarray] = {}
    for u_col in u_cols:
        cluster_id = u_col.split("_")[-1]
        v_col = f"v10_cluster_{cluster_id}"
        if v_col in df_w.columns:
            speed_col = f"wind_speed_cluster_{cluster_id}"
            wind_data[speed_col] = np.sqrt(df_w[u_col].to_numpy() ** 2 + df_w[v_col].to_numpy() ** 2)

    if not wind_data:
        raise ValueError("Function build_dwd_features: no matching u10/v10 cluster pairs found.")

    df_w_base = df_w[["date", "hour"]].copy()
    df_w_speed = pd.DataFrame(wind_data, index=df_w.index)
    df_w_full = pd.concat([df_w_base, df_w_speed], axis=1)
    wind_cols = list(df_w_speed.columns)

    df_wind_pivot = df_w_full.pivot_table(
        index="date",
        columns="hour",
        values=wind_cols,
        aggfunc="mean",
    )
    df_wind_pivot.columns = [f"{col[0]}_h{int(col[1]):02d}" for col in df_wind_pivot.columns]
    df_wind_pivot = df_wind_pivot.interpolate(axis=1, limit=1, limit_area="inside")

    df_s = df_qh.copy()
    ts_loc_s = df_s.index.tz_convert(tz_local)
    df_s = df_s.assign(date=ts_loc_s.date, mtu=ts_loc_s.hour * 4 + ts_loc_s.minute // 15)

    dir_cols = [col for col in df_s.columns if col.startswith("ASWDIR_cluster_")]
    solar_data: dict[str, np.ndarray] = {}
    for dir_col in dir_cols:
        cluster_id = dir_col.split("_")[-1]
        dif_col = f"ASWDIFD_cluster_{cluster_id}"
        if dif_col in df_s.columns:
            solar_data[f"sw_dir_cluster_{cluster_id}"] = df_s[dir_col].to_numpy()
            solar_data[f"sw_dif_cluster_{cluster_id}"] = df_s[dif_col].to_numpy()

    if not solar_data:
        raise ValueError("Function build_dwd_features: no matching ASWDIR/ASWDIFD cluster pairs found.")

    df_s_base = df_s[["date", "mtu"]].copy()
    df_solar_vals = pd.DataFrame(solar_data, index=df_s.index)
    df_s_full = pd.concat([df_s_base, df_solar_vals], axis=1)
    solar_cols = list(df_solar_vals.columns)

    df_solar_pivot = df_s_full.pivot_table(
        index="date",
        columns="mtu",
        values=solar_cols,
        aggfunc="mean",
    )
    df_solar_pivot.columns = [f"{col[0]}_mtu{int(col[1]):02d}" for col in df_solar_pivot.columns]
    df_solar_pivot = df_solar_pivot.interpolate(axis=1, limit=4, limit_area="inside")

    df_pivot = pd.concat([df_wind_pivot, df_solar_pivot], axis=1)
    full_index = pd.date_range(start=df_pivot.index.min(), end=df_pivot.index.max(), freq="D")
    df_pivot = df_pivot.reindex(full_index)
    df_pivot.index = pd.to_datetime(df_pivot.index).tz_localize(tz_local)
    df_pivot.index.name = "date"
    return df_pivot


def build_price_features(
    df_prices: pd.DataFrame,
    df_prices_exaa_15: pd.DataFrame | None = None,
    exaa_vector: bool = False,
    exaa_only: bool = False,
    daily_index: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Build price-based feature blocks for the LEAR model."""
    if exaa_only:
        exaa_vector = True

    if exaa_vector and df_prices_exaa_15 is None:
        raise ValueError("df_prices_exaa_15 is required when exaa_vector=True or exaa_only=True.")

    df = df_prices.copy().sort_index()
    df_utc = df.tz_convert("UTC")

    df_utc["price_lag1d"] = df_utc["price_da"].shift(freq="1D")
    df_utc["price_lag2d"] = df_utc["price_da"].shift(freq="2D")
    df_utc["price_lag7d"] = df_utc["price_da"].shift(freq="7D")

    df["price_lag1d"] = df_utc["price_lag1d"].values
    df["price_lag2d"] = df_utc["price_lag2d"].values
    df["price_lag7d"] = df_utc["price_lag7d"].values

    def _build_daily_matrix(series: pd.Series, col_name: str) -> pd.DataFrame:
        df_tmp = series.to_frame(col_name).copy()
        df_tmp["date_local"] = df_tmp.index.floor("D")
        df_tmp["mtu"] = df_tmp.index.hour * 4 + df_tmp.index.minute // 15
        return (
            df_tmp
            .pivot_table(
                index="date_local",
                columns="mtu",
                values=col_name,
                aggfunc="mean",
            )
            .reindex(columns=range(96))
            .interpolate(axis=1, limit=4, limit_area="inside")
        )

    daily_vector_dfs = []

    if not exaa_only:
        daily_matrix_da = _build_daily_matrix(df["price_da"], "price_da")
        if daily_index is not None:
            daily_matrix_da = daily_matrix_da.reindex(daily_index)
        for lag, prefix in [(1, "price_d1"), (2, "price_d2"), (7, "price_d7")]:
            dm = daily_matrix_da.shift(lag)
            dm.columns = [f"{prefix}_mtu_{int(col):02d}" for col in dm.columns]
            daily_vector_dfs.append(dm)

    if exaa_vector:
        daily_matrix_exaa = _build_daily_matrix(df_prices_exaa_15["price_exaa"], "price_exaa")
        if daily_index is not None:
            daily_matrix_exaa = daily_matrix_exaa.reindex(daily_index)
        dm = daily_matrix_exaa.copy()
        dm.columns = [f"exaa_d0_mtu_{int(col):02d}" for col in dm.columns]
        daily_vector_dfs.append(dm)

    result = pd.concat(daily_vector_dfs, axis=1, sort=False)
    result.index.name = "date"
    return result


def build_load_features(df_load_fc: pd.DataFrame) -> pd.DataFrame:
    """Build load forecast feature block for the LEAR model."""
    df = df_load_fc.copy().sort_index()
    df["date_local"] = df.index.floor("D")
    df["mtu"] = df.index.hour * 4 + df.index.minute // 15

    daily_matrix = (
        df
        .pivot_table(
            index="date_local",
            columns="mtu",
            values="load_fc",
            aggfunc="mean",
        )
        .reindex(columns=range(96))
        .interpolate(axis=1, limit=4, limit_area="inside")
    )
    daily_matrix.columns = [f"load_d0_mtu_{int(col):02d}" for col in daily_matrix.columns]
    daily_matrix.index.name = "date"
    return daily_matrix


def build_temporal_features(
    daily_index: pd.DatetimeIndex,
    post_regime_start: pd.Timestamp = pd.Timestamp("2025-10-01 00:00", tz="Europe/Berlin"),
) -> pd.DataFrame:
    """Build temporal features for the LEAR model."""
    if not isinstance(daily_index, pd.DatetimeIndex):
        raise TypeError("daily_index must be a DatetimeIndex.")

    regime_ts = post_regime_start.tz_convert(daily_index.tz)
    df_out = pd.DataFrame(index=daily_index)
    df_out["is_15min_market"] = (daily_index >= regime_ts).astype(int)

    weekday_oh = pd.get_dummies(daily_index.weekday, prefix="weekday", dtype=int)
    weekday_oh.index = daily_index
    df_out = pd.concat([df_out, weekday_oh], axis=1)

    de_holidays = holidays.Germany(years=daily_index.year.unique())
    df_out["is_holiday"] = pd.Series(daily_index.date, index=daily_index).isin(set(de_holidays.keys())).astype(int)
    df_out.index.name = "date"
    return df_out


def merge_all_features(
    df_weather_features: pd.DataFrame,
    df_price_features: pd.DataFrame,
    df_load_features: pd.DataFrame,
    df_time_features: pd.DataFrame,
    extra_feature_blocks: list[pd.DataFrame] | None = None,
    dropna: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge all engineered feature blocks into one daily dataset."""
    X = (
        df_weather_features
        .join(df_price_features, how="inner")
        .join(df_load_features, how="inner")
        .join(df_time_features, how="inner")
        .sort_index()
    )
    for feature_block in extra_feature_blocks or []:
        if feature_block is not None and not feature_block.empty:
            X = X.join(feature_block, how="inner").sort_index()

    nan_mask = X.isna().any(axis=1)
    dropped_info = pd.DataFrame(
        {
            "date": X.index[nan_mask],
            "nan_columns": X.loc[nan_mask].isna().apply(lambda row: row.index[row].tolist(), axis=1).values,
        }
    )

    if dropna:
        X = X.loc[~nan_mask]

    X.index.name = "date"
    return X, dropped_info


def build_y_matrix(
    df_prices_15: pd.DataFrame,
    daily_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Build the target matrix Y with shape (N_days, 96)."""
    df = df_prices_15.copy().sort_index()
    df["date_local"] = df.index.floor("D")
    df["mtu"] = df.index.hour * 4 + df.index.minute // 15

    Y = (
        df.pivot_table(
            index="date_local",
            columns="mtu",
            values="price_da",
            aggfunc="mean",
        )
        .reindex(columns=range(96))
        .interpolate(axis=1, limit=4, limit_area="inside")
    )
    Y.columns = range(96)
    Y = Y.reindex(daily_index)
    return Y
