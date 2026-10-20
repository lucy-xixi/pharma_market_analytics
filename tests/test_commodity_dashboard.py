"""Tests for commodity dashboard analytics helpers (no Streamlit runtime)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.commodity_analytics import (
    calendar_gap_plot_frame,
    commodity_quality_summary,
    filter_commodity,
    list_commodity_ids,
    price_inventory_change_association,
)


def _sample_store() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [
                "2024-01-05",
                "2024-01-12",
                "2024-01-26",  # gap at 2024-01-19
                "2024-01-05",
                "2024-01-12",
            ],
            "commodity_id": [
                "WTI_CRUDE",
                "WTI_CRUDE",
                "WTI_CRUDE",
                "BRENT_CRUDE",
                "BRENT_CRUDE",
            ],
            "price_usd_bbl": [70.0, 77.0, 74.0, 80.0, 82.0],
            "inventory_mmbbl": [400.0, 390.0, 395.0, 300.0, 305.0],
            "price_change_pct": [None, 10.0, -3.896, None, 2.5],
            "inventory_change": [None, -10.0, 5.0, None, 5.0],
            "inventory_change_pct": [None, -2.5, 1.282, None, 1.666],
            "frequency": ["weekly"] * 5,
            "data_source": ["synthetic_demo"] * 5,
            "data_type": ["synthetic"] * 5,
        }
    )


def test_empty_panel_handlers() -> None:
    empty = pd.DataFrame()
    assert list_commodity_ids(empty) == []
    assert filter_commodity(empty, "WTI_CRUDE").empty
    qa = commodity_quality_summary(empty)
    assert qa["n_observations"] == 0
    assert qa["missing_week_count"] == 0
    assoc = price_inventory_change_association(empty)
    assert assoc["n_paired"] == 0
    assert assoc["correlation"] is None
    assert calendar_gap_plot_frame(empty).empty


def test_commodity_filter_and_ids() -> None:
    df = _sample_store()
    assert list_commodity_ids(df) == ["BRENT_CRUDE", "WTI_CRUDE"]
    wti = filter_commodity(df, "WTI_CRUDE")
    assert set(wti["commodity_id"]) == {"WTI_CRUDE"}
    assert len(wti) == 3
    assert list(wti["date"].astype(str).str[:10]) == [
        "2024-01-05",
        "2024-01-12",
        "2024-01-26",
    ]


def test_quality_summary_gaps_and_provenance() -> None:
    wti = filter_commodity(_sample_store(), "WTI_CRUDE")
    qa = commodity_quality_summary(wti, expected_freq="W-FRI")
    assert qa["n_observations"] == 3
    assert qa["date_min"] == "2024-01-05"
    assert qa["date_max"] == "2024-01-26"
    assert qa["duplicate_excess_rows"] == 0
    assert qa["missing_week_count"] == 1
    assert qa["missing_week_dates"] == ["2024-01-19"]
    assert qa["data_type"] == "synthetic"
    assert qa["data_source"] == "synthetic_demo"
    assert "not live" in qa["disclaimer"].lower() or "demonstration" in qa["disclaimer"].lower()


def test_calendar_gap_plot_frame_marks_placeholder_not_observation() -> None:
    wti = filter_commodity(_sample_store(), "WTI_CRUDE")
    plot_df = calendar_gap_plot_frame(wti, expected_freq="W-FRI")
    assert "2024-01-19" in set(plot_df["date"].dt.strftime("%Y-%m-%d"))
    gap_row = plot_df[plot_df["date"] == "2024-01-19"].iloc[0]
    assert bool(gap_row["is_calendar_placeholder"]) is True
    assert pd.isna(gap_row["price_usd_bbl"])


def test_association_handles_missing_pairs_and_disclaimer() -> None:
    wti = filter_commodity(_sample_store(), "WTI_CRUDE")
    assoc = price_inventory_change_association(wti)
    assert assoc["n_paired"] == 2  # first row NaN changes dropped
    # With only 2 pairs, correlation stays None (<3)
    assert assoc["correlation"] is None
    assert "not imply" in assoc["disclaimer"].lower() or "causation" in assoc["disclaimer"].lower()

    # Enough points for a coefficient
    big = pd.DataFrame(
        {
            "inventory_change_pct": [1.0, 2.0, 3.0, 4.0],
            "price_change_pct": [2.0, 4.0, 6.0, 8.0],
        }
    )
    assoc2 = price_inventory_change_association(big)
    assert assoc2["n_paired"] == 4
    assert assoc2["correlation"] == pytest.approx(1.0)
