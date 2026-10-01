from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from da_price_forecasting.evaluation.metrics import (
    aggregated_pinball_score,
    bias_point,
    empirical_coverage,
    gw_test,
    mae_for_median,
    mae_point,
    pinball_score,
    rmse_point,
)


def test_point_metrics_ignore_missing_observations() -> None:
    df = pd.DataFrame(
        {
            "y_true": [10.0, 20.0, 30.0, np.nan],
            "model": [12.0, 18.0, 33.0, 40.0],
        }
    )

    assert mae_point(df, "model") == pytest.approx((2.0 + 2.0 + 3.0) / 3.0)
    assert rmse_point(df, "model") == pytest.approx(np.sqrt((4.0 + 4.0 + 9.0) / 3.0))
    assert bias_point(df, "model") == pytest.approx((2.0 - 2.0 + 3.0) / 3.0)


def test_pinball_and_quantile_metrics() -> None:
    index = pd.date_range("2026-03-01", periods=4, freq="15min", tz="Europe/Berlin")
    df = pd.DataFrame(
        {
            "y_true": [10.0, 20.0, 30.0, 40.0],
            "q0.100": [8.0, 18.0, 28.0, 38.0],
            "q0.500": [11.0, 19.0, 29.0, 41.0],
            "q0.900": [12.0, 22.0, 32.0, 42.0],
        },
        index=index,
    )

    assert pinball_score(np.array([10.0]), np.array([8.0]), 0.1)[0] == pytest.approx(0.2)
    assert pinball_score(np.array([10.0]), np.array([12.0]), 0.9)[0] == pytest.approx(0.2)
    assert mae_for_median(df) == pytest.approx(1.0)
    assert aggregated_pinball_score(df, [0.1, 0.5, 0.9]) == pytest.approx(0.3)
    assert empirical_coverage(df, 0.1, 0.9) == pytest.approx(1.0)


def test_gw_test_validates_input_shape() -> None:
    p_real = np.array([[1.0, 2.0]])
    p_pred_1 = np.array([[1.0, 2.0]])
    p_pred_2 = np.array([[1.0, 2.0]])

    with pytest.raises(ValueError, match="at least two evaluation days"):
        gw_test(p_real, p_pred_1, p_pred_2)
