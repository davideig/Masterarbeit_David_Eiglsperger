from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _valid_pair(df: pd.DataFrame, model: str, y_true_col: str) -> pd.DataFrame:
    if model not in df.columns:
        raise ValueError(f"Column '{model}' not found in DataFrame.")
    if y_true_col not in df.columns:
        raise ValueError(f"Column '{y_true_col}' not found in DataFrame.")
    return df[[y_true_col, model]].dropna()


def mae_point(df: pd.DataFrame, model: str, y_true_col: str = "y_true") -> float:
    valid = _valid_pair(df, model, y_true_col)
    return float(np.mean(np.abs(valid[y_true_col].to_numpy() - valid[model].to_numpy())))


def rmse_point(df: pd.DataFrame, model: str, y_true_col: str = "y_true") -> float:
    valid = _valid_pair(df, model, y_true_col)
    return float(np.sqrt(np.mean((valid[y_true_col].to_numpy() - valid[model].to_numpy()) ** 2)))


def bias_point(df: pd.DataFrame, model: str, y_true_col: str = "y_true") -> float:
    valid = _valid_pair(df, model, y_true_col)
    return float(np.mean(valid[model].to_numpy() - valid[y_true_col].to_numpy()))


def _as_daily_matrix(series: pd.Series, n_periods: int) -> np.ndarray:
    values = series.to_numpy()
    if len(values) % n_periods != 0:
        raise ValueError(f"Series length {len(values)} is not divisible by n_periods={n_periods}.")
    return values.reshape(len(values) // n_periods, n_periods)


def gw_test(
    p_real: np.ndarray,
    p_pred_1: np.ndarray,
    p_pred_2: np.ndarray,
    norm: int = 1,
    version: str = "multivariate",
) -> float | np.ndarray:
    """One-sided Giacomini-White conditional predictive ability test."""
    if p_real.shape != p_pred_1.shape or p_real.shape != p_pred_2.shape:
        raise ValueError("All three arrays must have the same shape.")
    if p_real.ndim != 2 or p_real.shape[1] <= 1:
        raise ValueError("Arrays must have shape (n_days, n_periods) with n_periods > 1.")
    if norm not in {1, 2}:
        raise ValueError("norm must be 1 or 2.")

    loss1 = p_real - p_pred_1
    loss2 = p_real - p_pred_2
    d = np.abs(loss1) - np.abs(loss2) if norm == 1 else loss1**2 - loss2**2
    return _gw_from_loss_diff(d, version=version)


def gw_test_point(
    df: pd.DataFrame,
    model_1: str,
    model_2: str,
    y_true_col: str = "y_true",
    n_periods: int = 96,
    norm: int = 1,
    version: str = "multivariate",
) -> float | np.ndarray:
    valid = df[[y_true_col, model_1, model_2]].dropna()
    return gw_test(
        p_real=_as_daily_matrix(valid[y_true_col], n_periods),
        p_pred_1=_as_daily_matrix(valid[model_1], n_periods),
        p_pred_2=_as_daily_matrix(valid[model_2], n_periods),
        norm=norm,
        version=version,
    )


def pinball_score(y: np.ndarray, q: np.ndarray, tau: float) -> np.ndarray:
    diff = y - q
    return np.maximum(tau * diff, (tau - 1) * diff)


def quantile_column(tau: float) -> str:
    return f"q{tau:.3f}"


def mae_for_median(df: pd.DataFrame, y_true_col: str = "y_true", q_median: float = 0.5) -> float:
    qcol = quantile_column(q_median)
    valid = df[[y_true_col, qcol]].dropna()
    return float(np.mean(np.abs(valid[y_true_col].to_numpy() - valid[qcol].to_numpy())))


def aps_loss_per_timestamp(
    df: pd.DataFrame,
    quantiles: list[float],
    y_true_col: str = "y_true",
) -> pd.Series:
    cols = [quantile_column(q) for q in quantiles]
    valid = df[[y_true_col, *cols]].dropna()
    y = valid[y_true_col].to_numpy()
    aps = sum(pinball_score(y, valid[quantile_column(q)].to_numpy(), q) for q in quantiles)
    return pd.Series(aps / len(quantiles), index=valid.index, name="aps")


def aggregated_pinball_score(
    df: pd.DataFrame,
    quantiles: list[float],
    y_true_col: str = "y_true",
) -> float:
    return float(aps_loss_per_timestamp(df, quantiles, y_true_col).mean())


def aps_loss_matrix(
    df: pd.DataFrame,
    quantiles: list[float],
    y_true_col: str = "y_true",
    n_periods: int = 96,
) -> np.ndarray:
    aps = aps_loss_per_timestamp(df, quantiles, y_true_col)
    return _as_daily_matrix(aps, n_periods)


def coverage_indicator(
    df: pd.DataFrame,
    lower_q: float,
    upper_q: float,
    y_true_col: str = "y_true",
) -> pd.Series:
    lcol = quantile_column(lower_q)
    ucol = quantile_column(upper_q)
    valid = df[[y_true_col, lcol, ucol]].dropna()
    return ((valid[y_true_col] >= valid[lcol]) & (valid[y_true_col] <= valid[ucol])).astype(int)


def empirical_coverage(
    df: pd.DataFrame,
    lower_q: float,
    upper_q: float,
    y_true_col: str = "y_true",
) -> float:
    return float(coverage_indicator(df, lower_q, upper_q, y_true_col).mean())


def kupiec_test(indicators: np.ndarray, alpha: float) -> float:
    n = len(indicators)
    k = int(np.asarray(indicators).sum())
    if n == 0:
        return np.nan
    if k == 0 or k == n:
        return 0.0
    pi_hat = k / n
    lr_stat = -2 * (
        (n - k) * np.log((1 - alpha) / (1 - pi_hat))
        + k * np.log(alpha / pi_hat)
    )
    return float(1 - stats.chi2.cdf(lr_stat, df=1))


def mtu_kupiec_test(
    df: pd.DataFrame,
    q_low: float,
    q_high: float,
    alpha: float,
    y_true_col: str = "y_true",
    significance_level: float = 0.05,
) -> int:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df.index must be a DatetimeIndex.")
    indicators = coverage_indicator(df, q_low, q_high, y_true_col)
    mtu_index = indicators.index.hour * 4 + indicators.index.minute // 15 + 1
    return int(
        sum(
            kupiec_test(indicators.to_numpy()[mtu_index == mtu], alpha) >= significance_level
            for mtu in range(1, 97)
            if (mtu_index == mtu).any()
        )
    )


def _gw_from_loss_diff(d: np.ndarray, version: str = "multivariate") -> float | np.ndarray:
    tau = 1
    n_days, n_periods = d.shape
    if n_days <= tau:
        raise ValueError("GW test requires at least two evaluation days.")
    t = n_days - tau

    if version == "univariate":
        gw_stat = np.full(n_periods, np.nan)
        for h in range(n_periods):
            dh = d[tau:, h]
            instruments = np.vstack([np.ones(t), d[:-tau, h]])
            reg = instruments * dh
            betas = np.linalg.lstsq(reg.T, np.ones(t), rcond=None)[0]
            err = np.ones(t) - reg.T @ betas
            gw_stat[h] = t * (1.0 - np.mean(err**2))
        gw_stat *= np.sign(np.mean(d[tau:], axis=0))
        return 1.0 - stats.chi2.cdf(gw_stat, df=2)

    if version == "multivariate":
        d_bar = d.mean(axis=1)
        dh = d_bar[tau:]
        instruments = np.vstack([np.ones(t), d_bar[:-tau]])
        reg = instruments * dh
        betas = np.linalg.lstsq(reg.T, np.ones(t), rcond=None)[0]
        err = np.ones(t) - reg.T @ betas
        gw_stat = t * (1.0 - np.mean(err**2))
        gw_stat *= np.sign(dh.mean())
        return float(1.0 - stats.chi2.cdf(gw_stat, df=2))

    raise ValueError("version must be 'univariate' or 'multivariate'.")


def gw_loss_test(loss_1: np.ndarray, loss_2: np.ndarray, version: str = "multivariate") -> float | np.ndarray:
    if loss_1.shape != loss_2.shape:
        raise ValueError("loss_1 and loss_2 must have the same shape.")
    if loss_1.ndim != 2:
        raise ValueError("Loss matrices must have shape (n_days, n_periods).")
    return _gw_from_loss_diff(loss_1 - loss_2, version=version)
