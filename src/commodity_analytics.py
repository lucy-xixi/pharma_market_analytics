"""
Commodity market analytics helpers for the Streamlit page (Phase 1).

Reads enriched rows from SQLite `commodity_market_series` (via callers).
Does not regenerate synthetic raw data or claim live market feeds.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import stats

from src.data_cleaning import detect_time_gap_dates

EXPECTED_FREQ_DEFAULT = "W-FRI"


def filter_commodity(
    df: pd.DataFrame,
    commodity_id: Optional[str] = None,
) -> pd.DataFrame:
    """Return rows for one commodity_id (or a copy of all rows if None)."""
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df.copy()
    out = df.copy()
    if commodity_id:
        out = out.loc[out["commodity_id"] == commodity_id].copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out = out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return out


def list_commodity_ids(df: pd.DataFrame) -> list[str]:
    if df is None or df.empty or "commodity_id" not in df.columns:
        return []
    return sorted(df["commodity_id"].dropna().astype(str).unique().tolist())


def commodity_quality_summary(
    df: pd.DataFrame,
    expected_freq: str = EXPECTED_FREQ_DEFAULT,
) -> dict[str, Any]:
    """
    Compact QA snapshot for the selected commodity series.

    duplicate_excess_rows uses key (date, commodity_id).
    missing_weeks lists calendar gaps vs expected_freq (not filled).
    """
    empty = {
        "n_observations": 0,
        "date_min": None,
        "date_max": None,
        "duplicate_excess_rows": 0,
        "missing_week_count": 0,
        "missing_week_dates": [],
        "data_type": None,
        "data_source": None,
        "disclaimer": (
            "Series provenance comes from stored data_type/data_source fields. "
            "synthetic / synthetic_demo rows are demonstration data, not live market feeds."
        ),
    }
    if df is None or df.empty:
        return empty

    work = df.copy()
    if "date" in work.columns:
        work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work = work.dropna(subset=["date"]) if "date" in work.columns else work

    dup = 0
    if {"date", "commodity_id"}.issubset(work.columns):
        dup = int(work.duplicated(subset=["date", "commodity_id"]).sum())

    gap_map = detect_time_gap_dates(work, expected_freq=expected_freq)
    # If single commodity filtered, flatten; else sum counts
    if work["commodity_id"].nunique() == 1:
        cid = str(work["commodity_id"].iloc[0])
        gap_dates = gap_map.get(cid, [])
    else:
        gap_dates = sorted({d for dates in gap_map.values() for d in dates})

    data_types = (
        sorted(work["data_type"].dropna().astype(str).unique().tolist())
        if "data_type" in work.columns
        else []
    )
    data_sources = (
        sorted(work["data_source"].dropna().astype(str).unique().tolist())
        if "data_source" in work.columns
        else []
    )

    return {
        "n_observations": int(len(work)),
        "date_min": work["date"].min().strftime("%Y-%m-%d") if len(work) else None,
        "date_max": work["date"].max().strftime("%Y-%m-%d") if len(work) else None,
        "duplicate_excess_rows": dup,
        "missing_week_count": int(len(gap_dates)),
        "missing_week_dates": gap_dates,
        "data_type": ", ".join(data_types) if data_types else None,
        "data_source": ", ".join(data_sources) if data_sources else None,
        "disclaimer": empty["disclaimer"],
    }


def price_inventory_change_association(
    df: pd.DataFrame,
    x_col: str = "inventory_change_pct",
    y_col: str = "price_change_pct",
) -> dict[str, Any]:
    """
    Pearson association between inventory and price change rates.

    Correlation is not causation — callers must surface that disclaimer.
    """
    result = {
        "n_paired": 0,
        "correlation": None,
        "p_value": None,
        "disclaimer": (
            "Correlation measures co-movement only; it does not imply that "
            "inventory changes cause price changes (or vice versa)."
        ),
    }
    if df is None or df.empty or x_col not in df.columns or y_col not in df.columns:
        return result

    pair = df[[x_col, y_col]].apply(pd.to_numeric, errors="coerce").dropna()
    result["n_paired"] = int(len(pair))
    if len(pair) < 3:
        return result

    r, p = stats.pearsonr(pair[x_col], pair[y_col])
    result["correlation"] = float(round(r, 4))
    result["p_value"] = float(round(p, 6))
    return result


def calendar_gap_plot_frame(
    df: pd.DataFrame,
    value_cols: Optional[list[str]] = None,
    expected_freq: str = EXPECTED_FREQ_DEFAULT,
) -> pd.DataFrame:
    """
    Build a plot frame reindexed to the expected calendar.

    Missing weeks appear as NaN value rows so charts can show visual gaps
    (connectgaps=False). These NaN rows are display-only placeholders —
    not stored observations and must not be treated as real market prints.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    work = df.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work = work.dropna(subset=["date"]).sort_values("date")
    if work.empty:
        return work

    value_cols = value_cols or [
        c
        for c in [
            "price_usd_bbl",
            "inventory_mmbbl",
            "price_change_pct",
            "inventory_change",
            "inventory_change_pct",
        ]
        if c in work.columns
    ]

    # One commodity expected for gap reindex; if multiple, return sorted as-is
    if "commodity_id" in work.columns and work["commodity_id"].nunique() > 1:
        return work

    idx = pd.date_range(work["date"].min(), work["date"].max(), freq=expected_freq)
    meta = {}
    for col in ["commodity_id", "frequency", "data_source", "data_type"]:
        if col in work.columns and work[col].notna().any():
            meta[col] = work[col].dropna().iloc[0]

    base = work.set_index("date")[value_cols]
    # Drop duplicate index if any before reindex
    base = base[~base.index.duplicated(keep="first")]
    aligned = base.reindex(idx)
    aligned.index.name = "date"
    out = aligned.reset_index()
    for k, v in meta.items():
        out[k] = v
    out["is_calendar_placeholder"] = ~out["date"].isin(work["date"])
    return out
