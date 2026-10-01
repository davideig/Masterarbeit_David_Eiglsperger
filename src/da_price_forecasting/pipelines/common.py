from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def as_local_day(value: Any, target_tz: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tz is None:
        timestamp = timestamp.tz_localize(target_tz)
    else:
        timestamp = timestamp.tz_convert(target_tz)
    return timestamp.normalize()


def load_timestamp_csv(path: Path, target_tz: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(target_tz)
    df = df.sort_index()
    df = df.loc[~df.index.duplicated(keep="last")]
    df.index.name = "timestamp"
    return df


def save_timestamp_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)


def point_error_stats(forecast: pd.DataFrame, pred_col: str, true_col: str) -> dict[str, float | int]:
    valid = forecast[[pred_col, true_col]].dropna()
    errors = valid[pred_col] - valid[true_col]
    mse = float((errors**2).mean())
    rmse = float(np.sqrt(mse))
    ss_res = float((errors**2).sum())
    ss_tot = float(((valid[true_col] - valid[true_col].mean()) ** 2).sum())
    mean_actual = float(valid[true_col].mean()) if not valid.empty else np.nan
    return {
        "mae": float(errors.abs().mean()),
        "mse": mse,
        "rmse": rmse,
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
        "bias": float(errors.mean()),
        "mean_actual_mw": mean_actual,
        "n_obs": int(len(valid)),
    }
