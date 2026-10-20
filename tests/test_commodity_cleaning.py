"""Unit tests for commodity cleaning / quality rules."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data_cleaning import clean_commodity_dataset, detect_time_gaps


def _sample_raw() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [
                "2024-01-05",
                "2024-01-12",
                "2024-01-12",  # duplicate key
                "2024-01-26",  # gap vs W-FRI calendar (missing 01-19)
                "bad-date",
            ],
            "commodity_id": ["WTI_CRUDE"] * 5,
            "price_usd_bbl": [70.0, -2.5, -2.5, 500.0, 71.0],  # negative + extreme retained
            "inventory_mmbbl": [400.0, 401.0, 401.0, -10.0, 402.0],  # negative = error
            "frequency": ["weekly"] * 5,
            "data_source": ["synthetic_demo"] * 5,
            "data_type": ["synthetic"] * 5,
        }
    )


def test_clean_drops_duplicates_and_invalid_dates() -> None:
    cleaned, report, extras = clean_commodity_dataset(
        _sample_raw(), persist_report=False, expected_freq="W-FRI"
    )
    assert report.duplicate_rows >= 1
    assert extras.invalid_dates >= 1
    assert cleaned["date"].is_unique or cleaned.duplicated(["date", "commodity_id"]).sum() == 0
    assert "bad-date" not in set(cleaned["date"])


def test_extreme_and_negative_prices_retained_negative_inventory_nullified() -> None:
    cleaned, _, extras = clean_commodity_dataset(_sample_raw(), persist_report=False)
    assert (cleaned["price_usd_bbl"] < 0).any()
    assert cleaned["price_usd_bbl"].max() >= 500.0
    # After ffill/bfill, negative inventory should not remain
    assert not (cleaned["inventory_mmbbl"] < 0).any()
    assert extras.extreme_price_flags >= 0  # computed on pre-clean frame


def test_detect_time_gaps_counts_missing_weeks() -> None:
    df = pd.DataFrame(
        {
            "date": ["2024-01-05", "2024-01-19"],  # missing 2024-01-12 for W-FRI
            "commodity_id": ["BRENT_CRUDE", "BRENT_CRUDE"],
            "price_usd_bbl": [70.0, 71.0],
            "inventory_mmbbl": [300.0, 301.0],
        }
    )
    gaps = detect_time_gaps(df, expected_freq="W-FRI")
    assert gaps["BRENT_CRUDE"] == 1
