"""Hand-checked quality policy for the pharmaceutical panel."""

from __future__ import annotations

import pandas as pd

from src.pharma_quality import prepare_sales_panel


def _base_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "month": "2021-01-01",
        "product": "PX-A",
        "region": "East",
        "sales_amount": 100.0,
        "sales_volume": 10.0,
        "data_source": "synthetic_demo",
        "data_type": "synthetic_demo",
    }
    row.update(overrides)
    return row


def test_exclusions_are_reported_and_values_are_not_rewritten() -> None:
    raw = pd.DataFrame(
        [
            _base_row(month="2021-01-01", sales_amount=100.0),
            _base_row(month="2021-01-01", sales_amount=999.0),  # duplicate key
            _base_row(month="2021-03-01", sales_amount=90.0),
            _base_row(month="2021-04-01", sales_amount=-5.0, sales_volume=3.0),
            _base_row(month="2021-04-15", sales_amount=50.0),  # not month start
            _base_row(month="2021-05-01", sales_amount=95.0),
            _base_row(month="2021-06-01", sales_amount=10_000.0),  # retained outlier
        ]
    )
    before = raw.copy(deep=True)
    analytical, report = prepare_sales_panel(raw)
    pd.testing.assert_frame_equal(raw, before)

    assert report["policy"]["negative_sales"] == "exclude_row"
    assert report["policy"]["missing_months"] == "report_only"
    assert report["policy"]["outliers"] == "flag_only"
    assert report["raw"]["duplicate_key_excess"] == 1
    assert report["raw"]["negative_sales_amount_rows"] == 1
    assert report["raw"]["invalid_month_rows"] >= 1
    assert report["analytical"]["negative_sales_amount_rows"] == 0
    assert report["analytical"]["duplicate_key_excess"] == 0
    assert report["analytical"]["missing_month_count"] == 2
    assert report["analytical"]["missing_months"] == [
        {"product": "PX-A", "region": "East", "month": "2021-02-01"},
        {"product": "PX-A", "region": "East", "month": "2021-04-01"},
    ]
    assert set(analytical["month"]) == {"2021-01-01", "2021-03-01", "2021-05-01", "2021-06-01"}
    january = analytical.loc[analytical["month"] == "2021-01-01", "sales_amount"].iloc[0]
    assert january == 100.0
    assert 10_000.0 in set(analytical["sales_amount"])
    assert -5.0 not in set(analytical["sales_amount"])
    assert "2021-02-01" not in set(analytical["month"])
    assert report["analytical"]["outlier_flags"]["sales_amount"] >= 1
    assert "nulls_by_column" in report["raw"]
    assert "non_numeric_sales_amount_rows" in report["raw"]


def test_non_numeric_amount_and_illegal_date_are_excluded() -> None:
    raw = pd.DataFrame(
        [
            _base_row(month="2021-01-01", sales_amount=100.0),
            _base_row(month="2021-02-01", sales_amount="not-a-number"),
            _base_row(month="2021-02-15", sales_amount=40.0),
        ]
    )
    analytical, report = prepare_sales_panel(raw)
    assert report["raw"]["non_numeric_sales_amount_rows"] == 1
    assert report["raw"]["invalid_month_rows"] >= 1
    assert report["rows_excluded"] == 2
    assert list(analytical["month"]) == ["2021-01-01"]
    assert list(analytical["sales_amount"]) == [100.0]
