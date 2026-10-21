"""Hand-checked growth rates for the pharmaceutical panel."""

from __future__ import annotations

import pandas as pd
import pytest

from src.pharma_features import add_growth_features, cagr_percent, cagr_table


def _row(month: str, product: str, amount: float, volume: float = 1.0, region: str = "East") -> dict:
    return {
        "month": month,
        "product": product,
        "region": region,
        "sales_amount": amount,
        "sales_volume": volume,
        "data_source": "synthetic_demo",
        "data_type": "synthetic_demo",
    }


def test_mom_does_not_jump_a_missing_month_and_yoy_uses_same_month() -> None:
    frame = pd.DataFrame(
        [
            _row("2021-01-01", "PX-A", 100),
            _row("2021-02-01", "PX-A", 110),
            _row("2021-04-01", "PX-A", 120),
            _row("2022-01-01", "PX-A", 200),
            _row("2021-02-01", "PX-A", 0, region="West"),
            _row("2021-03-01", "PX-A", 50, region="West"),
        ]
    )
    out = add_growth_features(frame)
    east = out[out["region"] == "East"].set_index("month")
    assert east.loc["2021-02-01", "mom_sales_amount_pct"] == pytest.approx(10.0)
    assert pd.isna(east.loc["2021-04-01", "mom_sales_amount_pct"])
    assert east.loc["2022-01-01", "yoy_sales_amount_pct"] == pytest.approx(100.0)
    west = out[out["region"] == "West"].set_index("month")
    assert pd.isna(west.loc["2021-03-01", "mom_sales_amount_pct"])


def test_share_and_cagr_match_hand_calculation() -> None:
    frame = pd.DataFrame(
        [
            _row("2021-01-01", "PX-A", 30),
            _row("2021-01-01", "PX-B", 70),
            _row("2022-01-01", "PX-A", 121),
            _row("2022-01-01", "PX-B", 70),
        ]
    )
    out = add_growth_features(frame)
    january = out[out["month"] == "2021-01-01"].set_index("product")
    assert january.loc["PX-A", "sales_share_pct"] == pytest.approx(30.0)
    assert january.loc["PX-B", "sales_share_pct"] == pytest.approx(70.0)
    assert cagr_percent(100, 121, 12) == pytest.approx(21.0)
    table = cagr_table(frame[frame["product"] == "PX-A"])
    assert table.loc[0, "months_between"] == 12
    assert table.loc[0, "cagr_pct"] == pytest.approx((121 / 30 - 1.0) * 100.0)


def test_equal_months_are_zero_growth_and_a_gap_is_not_zero() -> None:
    frame = pd.DataFrame(
        [
            _row("2021-03-01", "PX-A", 100),
            _row("2021-01-01", "PX-A", 100),
            _row("2021-02-01", "PX-A", 100),
            _row("2021-04-01", "PX-A", 100),
        ]
    )
    # Remove March so April has no previous calendar month.
    gapped = frame[frame["month"] != "2021-03-01"]
    out = add_growth_features(gapped)
    assert list(out["month"]) == ["2021-01-01", "2021-02-01", "2021-04-01"]
    indexed = out.set_index("month")
    assert indexed.loc["2021-02-01", "mom_sales_amount_pct"] == pytest.approx(0.0)
    assert pd.isna(indexed.loc["2021-04-01", "mom_sales_amount_pct"])


def test_region_sums_and_shares_use_the_same_amounts() -> None:
    frame = pd.DataFrame(
        [
            _row("2021-01-01", "PX-A", 10, volume=1, region="East"),
            _row("2021-01-01", "PX-A", 15, volume=2, region="West"),
            _row("2021-01-01", "PX-B", 25, volume=3, region="East"),
        ]
    )
    out = add_growth_features(frame)
    east = out[out["region"] == "East"]
    assert east["sales_amount"].sum() == pytest.approx(35)
    assert east["sales_share_pct"].sum() == pytest.approx(100)
    assert out["sales_volume"].sum() == pytest.approx(6)

