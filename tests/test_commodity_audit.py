"""
Audit-focused tests: hand-checkable features, gap dates, SQLite uniqueness, idempotent writes.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from src.commodity_features import enrich_commodity_features
from src.data_cleaning import clean_commodity_dataset, detect_time_gap_dates
from src.data_loader import save_commodity_processed


def _hand_panel() -> pd.DataFrame:
    """Tiny synthetic panel with known arithmetic (not market data)."""
    return pd.DataFrame(
        {
            "date": [
                "2024-01-12",  # deliberately unsorted
                "2024-01-05",
                "2024-01-12",
                "2024-01-05",
            ],
            "commodity_id": ["WTI_CRUDE", "WTI_CRUDE", "BRENT_CRUDE", "BRENT_CRUDE"],
            "price_usd_bbl": [105.0, 100.0, 180.0, 200.0],
            "inventory_mmbbl": [40.0, 50.0, 12.0, 10.0],
            "frequency": ["weekly"] * 4,
            "data_source": ["synthetic_demo"] * 4,
            "data_type": ["synthetic"] * 4,
        }
    )


def test_hand_calculated_features_and_no_cross_commodity_leak() -> None:
    out = enrich_commodity_features(_hand_panel())
    wti = out[out["commodity_id"] == "WTI_CRUDE"].reset_index(drop=True)
    brent = out[out["commodity_id"] == "BRENT_CRUDE"].reset_index(drop=True)

    assert list(wti["date"]) == ["2024-01-05", "2024-01-12"]
    assert pd.isna(wti.loc[0, "price_change_pct"])
    assert wti.loc[1, "price_change_pct"] == pytest.approx(5.0)  # 105/100 - 1
    assert wti.loc[1, "inventory_change"] == pytest.approx(-10.0)
    assert wti.loc[1, "inventory_change_pct"] == pytest.approx(-20.0)

    assert brent.loc[1, "price_change_pct"] == pytest.approx(-10.0)  # 180/200 - 1
    assert brent.loc[1, "inventory_change"] == pytest.approx(2.0)
    assert brent.loc[1, "inventory_change_pct"] == pytest.approx(20.0)

    # Cross-check: WTI second price change must not use Brent's 200
    assert wti.loc[1, "price_change_pct"] != pytest.approx((105 / 200 - 1) * 100)


def test_zero_denominator_price_and_nonpositive_inventory_rate() -> None:
    df = pd.DataFrame(
        {
            "date": ["2024-01-05", "2024-01-12", "2024-01-05", "2024-01-12"],
            "commodity_id": ["A", "A", "B", "B"],
            "price_usd_bbl": [0.0, 10.0, 50.0, 55.0],
            "inventory_mmbbl": [0.0, 5.0, -1.0, 8.0],
            "frequency": ["weekly"] * 4,
            "data_source": ["synthetic_demo"] * 4,
            "data_type": ["synthetic"] * 4,
        }
    )
    # After clean, negative inventory becomes NaN then ffill/bfill — test features directly
    out = enrich_commodity_features(df)
    a = out[out["commodity_id"] == "A"].reset_index(drop=True)
    assert pd.isna(a.loc[1, "price_change_pct"])  # prior price 0
    assert pd.isna(a.loc[1, "inventory_change_pct"])  # prior inventory 0


def test_brent_gap_is_missing_weekly_friday_not_filled() -> None:
    # Mirrors generator intentional drop of 2021-06-18 for BRENT
    dates = pd.date_range("2021-06-04", "2021-06-25", freq="W-FRI")
    rows = []
    for d in dates:
        if d.strftime("%Y-%m-%d") == "2021-06-18":
            continue
        rows.append(
            {
                "date": d.strftime("%Y-%m-%d"),
                "commodity_id": "BRENT_CRUDE",
                "price_usd_bbl": 70.0,
                "inventory_mmbbl": 300.0,
                "frequency": "weekly",
                "data_source": "synthetic_demo",
                "data_type": "synthetic",
            }
        )
    raw = pd.DataFrame(rows)
    gap_dates = detect_time_gap_dates(raw, expected_freq="W-FRI")
    assert gap_dates["BRENT_CRUDE"] == ["2021-06-18"]

    cleaned, _, extras = clean_commodity_dataset(raw, persist_report=False)
    # Cleaning must NOT invent the missing week
    assert "2021-06-18" not in set(cleaned["date"])
    assert extras.time_gap_dates_by_commodity["BRENT_CRUDE"] == ["2021-06-18"]


def test_sqlite_unique_and_replace_idempotent(tmp_path: Path) -> None:
    df = enrich_commodity_features(_hand_panel())
    db = tmp_path / "audit.db"
    # Point save through custom db by writing CSV in tmp then using save API
    # save_commodity_processed uses project processed dir for CSV; only DB path is overridden
    from src import data_loader as dl

    out1, n1 = dl.save_commodity_processed(
        df,
        filename="audit_commodity.csv",
        table="commodity_market_series",
        db_path=db,
    )
    out2, n2 = dl.save_commodity_processed(
        df,
        filename="audit_commodity.csv",
        table="commodity_market_series",
        db_path=db,
    )
    assert n1 == n2 == len(df)
    assert out1.exists()

    con = sqlite3.connect(db)
    count = con.execute("SELECT COUNT(*) FROM commodity_market_series").fetchone()[0]
    indexes = con.execute(
        "SELECT name, sql FROM sqlite_master WHERE type='index' AND tbl_name='commodity_market_series'"
    ).fetchall()
    con.close()
    assert count == len(df)
    assert any("UNIQUE" in (sql or "") or "ux_commodity" in name for name, sql in indexes)

    # Unique index rejects duplicate insert
    con = sqlite3.connect(db)
    with pytest.raises(sqlite3.IntegrityError):
        con.execute(
            "INSERT INTO commodity_market_series(date, commodity_id, price_usd_bbl, inventory_mmbbl) "
            "VALUES ('2024-01-05', 'WTI_CRUDE', 1.0, 1.0)"
        )
        con.commit()
    con.close()


def test_duplicate_excess_row_count_semantics() -> None:
    raw = _hand_panel()
    # Add one exact duplicate of first row
    raw = pd.concat([raw, raw.iloc[[1]]], ignore_index=True)
    _, report, extras = clean_commodity_dataset(raw, persist_report=False)
    assert extras.duplicate_key == ("date", "commodity_id")
    assert report.duplicate_rows == 1  # one excess row
