from __future__ import annotations

from datetime import date

import pandas as pd

from da_price_forecasting.preprocessing.reserve_market import build_reserve_capacity_features


def test_build_fcr_capacity_features_expands_four_hour_blocks() -> None:
    raw = pd.DataFrame(
        {
            "PRODUCTNAME": ["NEGPOS_00_04", "NEGPOS_04_08"],
            "GERMANY_DEMAND_[MW]": [580.0, 590.0],
            "GERMANY_SETTLEMENTCAPACITY_PRICE_[EUR/MW]": [12.5, 13.0],
            "CROSSBORDER_SETTLEMENTCAPACITY_PRICE_[EUR/MW]": [11.0, 11.5],
            "GERMANY_DEFICIT(-)_SURPLUS(+)_[MW]": [25.0, -5.0],
        }
    )

    features = build_reserve_capacity_features(raw, product_type="FCR", delivery_date=date(2026, 7, 1))

    assert features.loc["2026-07-01 00:00:00+02:00", "reserve_fcr_germany_demand_mw"] == 580.0
    assert features.loc["2026-07-01 03:45:00+02:00", "reserve_fcr_germany_capacity_price_eur_mw"] == 12.5
    assert features.loc["2026-07-01 04:00:00+02:00", "reserve_fcr_germany_demand_mw"] == 590.0
    assert features.loc["2026-07-01 07:45:00+02:00", "reserve_fcr_germany_surplus_mw"] == -5.0


def test_build_afrr_capacity_features_keeps_positive_and_negative_products_separate() -> None:
    raw = pd.DataFrame(
        {
            "TYPE_OF_RESERVES": ["POS", "NEG"],
            "PRODUCT": ["POS_00_04", "NEG_00_04"],
            "GERMANY_AVERAGE_CAPACITY_PRICE_[(EUR/MW)/h]": [18.0, 9.0],
            "GERMANY_MARGINAL_CAPACITY_PRICE_[(EUR/MW)/h]": [20.0, 12.0],
            "GERMANY_ALLOCATED_VOLUME_[MW]": [1_000.0, 800.0],
            "GERMANY_SUM_OF_OFFERED_CAPACITY_[MW]": [1_900.0, 1_500.0],
            "GERMANY_IMPORT(-)_EXPORT(+)_[MW]": [100.0, -50.0],
        }
    )

    features = build_reserve_capacity_features(raw, product_type="aFRR", delivery_date=date(2026, 7, 1))

    assert features.loc["2026-07-01 00:00:00+02:00", "reserve_afrr_pos_avg_capacity_price_eur_mw_h"] == 18.0
    assert features.loc["2026-07-01 00:00:00+02:00", "reserve_afrr_neg_avg_capacity_price_eur_mw_h"] == 9.0
    assert features.loc["2026-07-01 03:45:00+02:00", "reserve_afrr_pos_allocated_mw"] == 1_000.0
    assert features.loc["2026-07-01 03:45:00+02:00", "reserve_afrr_neg_import_export_mw"] == -50.0


def test_build_mfrr_capacity_features_accepts_missing_allocated_volume() -> None:
    raw = pd.DataFrame(
        {
            "TYPE_OF_RESERVES": ["POS"],
            "PRODUCT": ["POS_20_24"],
            "GERMANY_AVERAGE_CAPACITY_PRICE_[(EUR/MW)/h]": [6.0],
            "GERMANY_MARGINAL_CAPACITY_PRICE_[(EUR/MW)/h]": [7.0],
            "GERMANY_SUM_OF_OFFERED_CAPACITY_[MW]": [2_500.0],
            "GERMANY_IMPORT(-)_EXPORT(+)_[MW]": [0.0],
        }
    )

    features = build_reserve_capacity_features(raw, product_type="mFRR", delivery_date=date(2026, 7, 1))

    assert features.loc["2026-07-01 19:45:00+02:00"].isna().all()
    assert features.loc["2026-07-01 20:00:00+02:00", "reserve_mfrr_pos_avg_capacity_price_eur_mw_h"] == 6.0
    assert "reserve_mfrr_pos_allocated_mw" not in features.columns


def test_build_capacity_features_treats_dash_placeholders_as_missing() -> None:
    raw = pd.DataFrame(
        {
            "PRODUCTNAME": ["NEGPOS_00_04"],
            "GERMANY_DEMAND_[MW]": ["-"],
            "GERMANY_SETTLEMENTCAPACITY_PRICE_[EUR/MW]": ["12,5"],
            "CROSSBORDER_SETTLEMENTCAPACITY_PRICE_[EUR/MW]": ["-"],
            "GERMANY_DEFICIT(-)_SURPLUS(+)_[MW]": ["1.234,5"],
        }
    )

    features = build_reserve_capacity_features(raw, product_type="FCR", delivery_date=date(2026, 7, 1))

    assert pd.isna(features.loc["2026-07-01 00:00:00+02:00", "reserve_fcr_germany_demand_mw"])
    assert features.loc["2026-07-01 00:00:00+02:00", "reserve_fcr_germany_capacity_price_eur_mw"] == 12.5
    assert pd.isna(features.loc["2026-07-01 00:00:00+02:00", "reserve_fcr_crossborder_capacity_price_eur_mw"])
    assert features.loc["2026-07-01 00:00:00+02:00", "reserve_fcr_germany_surplus_mw"] == 1234.5
