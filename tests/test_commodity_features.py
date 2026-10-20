"""Unit tests for Phase-1 commodity feature engineering."""

from __future__ import annotations

import pandas as pd
import pytest

from src.commodity_features import (
    enrich_commodity_features,
    inventory_change,
    inventory_change_rate,
    price_change_rate,
)


def test_price_change_rate_no_lookahead_and_div_zero() -> None:
    s = pd.Series([100.0, 110.0, 0.0, 50.0])
    out = price_change_rate(s, periods=1)
    assert pd.isna(out.iloc[0])
    assert out.iloc[1] == pytest.approx(10.0)
    # 0 vs prior 110 is a valid return
    assert out.iloc[2] == pytest.approx(-100.0)
    # prior price is 0 → NaN (no inf)
    assert pd.isna(out.iloc[3])
    assert pd.isna(price_change_rate(pd.Series([0.0, 50.0])).iloc[1])


def test_inventory_change_and_rate() -> None:
    s = pd.Series([100.0, 90.0, 90.0])
    ch = inventory_change(s)
    assert pd.isna(ch.iloc[0])
    assert ch.iloc[1] == pytest.approx(-10.0)
    rate = inventory_change_rate(s)
    assert rate.iloc[1] == pytest.approx(-10.0)
    assert rate.iloc[2] == pytest.approx(0.0)


def test_enrich_sorted_by_commodity_no_future_leak() -> None:
    df = pd.DataFrame(
        {
            "date": ["2024-01-12", "2024-01-05", "2024-01-12", "2024-01-05"],
            "commodity_id": ["WTI_CRUDE", "WTI_CRUDE", "BRENT_CRUDE", "BRENT_CRUDE"],
            "price_usd_bbl": [80.0, 70.0, 82.0, 75.0],
            "inventory_mmbbl": [400.0, 410.0, 300.0, 310.0],
            "frequency": ["weekly"] * 4,
            "data_source": ["synthetic_demo"] * 4,
            "data_type": ["synthetic"] * 4,
        }
    )
    out = enrich_commodity_features(df)
    wti = out[out["commodity_id"] == "WTI_CRUDE"].reset_index(drop=True)
    assert wti.loc[0, "date"] == "2024-01-05"
    assert pd.isna(wti.loc[0, "price_change_pct"])
    assert wti.loc[1, "price_change_pct"] == pytest.approx((80 / 70 - 1) * 100)
    assert wti.loc[1, "inventory_change"] == pytest.approx(-10.0)


def test_negative_prior_inventory_rate_is_nan() -> None:
    s = pd.Series([-5.0, 10.0])
    assert pd.isna(inventory_change_rate(s).iloc[1])
