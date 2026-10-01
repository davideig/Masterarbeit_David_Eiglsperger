from __future__ import annotations

import numpy as np
import pandas as pd

from da_price_forecasting.models import lear


class _RecordingRegressor:
    fit_lengths: list[int] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def fit(self, X, y):
        assert np.isfinite(y).all()
        self.fit_lengths.append(len(y))
        self.coef_ = np.zeros(X.shape[1], dtype=float)
        self.intercept_ = float(np.mean(y))
        self.alpha_ = 0.0
        return self

    def predict(self, X):
        return np.full(X.shape[0], self.intercept_, dtype=float)


def test_rolling_point_forecast_drops_nan_training_targets(monkeypatch) -> None:
    index = pd.date_range("2026-01-01", periods=9, freq="D", tz="Europe/Berlin")
    X = pd.DataFrame(
        {
            "exaa_d0_mtu_00": np.linspace(10.0, 18.0, len(index)),
            "weekday_0": (index.weekday == 0).astype(float),
        },
        index=index,
    )
    Y = pd.DataFrame(
        {
            mtu: np.linspace(40.0 + mtu, 48.0 + mtu, len(index))
            for mtu in range(96)
        },
        index=index,
    )
    Y.loc[index[3], :] = np.nan
    Y.loc[index[-1], :] = np.nan
    _RecordingRegressor.fit_lengths = []

    monkeypatch.setattr(lear, "LassoLarsCV", _RecordingRegressor)

    forecast, *_ = lear.rolling_point_forecast(
        X=X,
        Y=Y,
        forecast_days=[index[-1]],
        train_days=8,
        lars_start_date=pd.Timestamp("2025-01-01", tz="Europe/Berlin"),
    )

    assert len(forecast) == 96
    assert forecast["y_pred"].notna().all()
    assert set(_RecordingRegressor.fit_lengths) == {7}


def test_rolling_point_forecast_skips_days_with_too_little_history(monkeypatch) -> None:
    index = pd.date_range("2026-01-01", periods=7, freq="D", tz="Europe/Berlin")
    X = pd.DataFrame(
        {
            "price_d1_mtu_00": np.linspace(20.0, 26.0, len(index)),
            "weekday_0": (index.weekday == 0).astype(float),
        },
        index=index,
    )
    Y = pd.DataFrame(
        {
            mtu: np.linspace(40.0 + mtu, 46.0 + mtu, len(index))
            for mtu in range(96)
        },
        index=index,
    )
    _RecordingRegressor.fit_lengths = []

    monkeypatch.setattr(lear, "LassoLarsCV", _RecordingRegressor)

    forecast, *_ = lear.rolling_point_forecast(
        X=X,
        Y=Y,
        forecast_days=[index[4], index[6]],
        train_days=6,
        lars_start_date=pd.Timestamp("2025-01-01", tz="Europe/Berlin"),
    )

    assert len(forecast) == 96
    assert forecast.index.min().date() == index[6].date()
    assert set(_RecordingRegressor.fit_lengths) == {6}


def test_rolling_point_forecast_supports_boosted_tree_price_model() -> None:
    index = pd.date_range("2026-01-01", periods=12, freq="D", tz="Europe/Berlin")
    X = pd.DataFrame(
        {
            "price_d1_mtu_00": np.linspace(20.0, 31.0, len(index)),
            "load_forecast": np.linspace(50_000.0, 53_000.0, len(index)),
            "weekday_0": (index.weekday == 0).astype(float),
        },
        index=index,
    )
    Y = pd.DataFrame(
        {
            mtu: np.linspace(40.0 + mtu, 51.0 + mtu, len(index))
            for mtu in range(96)
        },
        index=index,
    )

    forecast, runtime, coef, intercept, _ = lear.rolling_point_forecast(
        X=X,
        Y=Y,
        forecast_days=[index[-1]],
        train_days=11,
        lars_start_date=pd.Timestamp("2027-01-01", tz="Europe/Berlin"),
        price_model_type="hist_gradient_boosting",
        hgb_max_iter=5,
        hgb_min_samples_leaf=2,
    )

    assert len(forecast) == 96
    assert forecast["y_pred"].notna().all()
    assert set(runtime["price_model_type"]) == {"hist_gradient_boosting"}
    assert set(coef["price_model_type"]) == {"hist_gradient_boosting"}
    assert coef["n_nonzero"].eq(0).all()
    assert set(intercept["price_model_type"]) == {"hist_gradient_boosting"}


def test_scale_fold_point_sets_all_nan_continuous_columns_to_zero() -> None:
    index = pd.date_range("2026-01-01", periods=6, freq="D", tz="Europe/Berlin")
    X_tr = pd.DataFrame(
        {
            "reserve_mfrr_pos_import_export_mw_mtu_00": np.nan,
            "reserve_afrr_pos_allocated_mw_mtu_00": np.linspace(100.0, 150.0, len(index)),
        },
        index=index,
    )
    X_va = pd.DataFrame(
        {
            "reserve_mfrr_pos_import_export_mw_mtu_00": [10.0],
            "reserve_afrr_pos_allocated_mw_mtu_00": [175.0],
        },
        index=[index[-1] + pd.Timedelta(days=1)],
    )
    y_tr = pd.Series(np.linspace(40.0, 45.0, len(index)), index=index)

    X_tr_scaled, X_va_scaled, *_ = lear.scale_fold_point(
        X_tr=X_tr,
        X_va=X_va,
        y_tr=y_tr,
        y_va=None,
    )

    assert X_tr_scaled["reserve_mfrr_pos_import_export_mw_mtu_00"].eq(0.0).all()
    assert X_va_scaled["reserve_mfrr_pos_import_export_mw_mtu_00"].eq(0.0).all()
