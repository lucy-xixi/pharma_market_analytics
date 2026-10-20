"""
Commodity market feature engineering (Phase 1: crude oil panel).

Computes look-back time-series features by commodity without using future data.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

DATE_COL = "date"
ID_COL = "commodity_id"
PRICE_COL = "price_usd_bbl"
INVENTORY_COL = "inventory_mmbbl"


def _ensure_sorted(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by commodity then date; requires parseable dates."""
    out = df.copy()
    out[DATE_COL] = pd.to_datetime(out[DATE_COL], errors="coerce")
    if out[DATE_COL].isna().any():
        raise ValueError("Invalid dates found; cannot compute commodity features")
    return out.sort_values([ID_COL, DATE_COL]).reset_index(drop=True)


def price_change_rate(
    series: pd.Series,
    periods: int = 1,
) -> pd.Series:
    """
    Period-over-period price return (%).

    Uses prior observed price only (no forward fill of the current value).
    Zero / missing prior → NaN (no div-by-zero explosion).
    """
    s = pd.to_numeric(series, errors="coerce")
    prev = s.shift(periods)
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = (s / prev - 1.0) * 100.0
    rate = rate.where(prev.notna() & (prev != 0))
    return rate.replace([np.inf, -np.inf], np.nan)


def inventory_change(
    series: pd.Series,
    periods: int = 1,
) -> pd.Series:
    """Absolute inventory change vs prior period (same frequency as panel)."""
    s = pd.to_numeric(series, errors="coerce")
    return s - s.shift(periods)


def inventory_change_rate(
    series: pd.Series,
    periods: int = 1,
) -> pd.Series:
    """
    Inventory percentage change vs prior period.

    Prior ≤ 0 or missing → NaN.
    """
    s = pd.to_numeric(series, errors="coerce")
    prev = s.shift(periods)
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = (s / prev - 1.0) * 100.0
    rate = rate.where(prev.notna() & (prev > 0))
    return rate.replace([np.inf, -np.inf], np.nan)


def enrich_commodity_features(
    df: pd.DataFrame,
    periods: int = 1,
    price_col: str = PRICE_COL,
    inventory_col: str = INVENTORY_COL,
) -> pd.DataFrame:
    """
    Add lag-based features within each commodity_id series.

    Features:
      - price_change_pct = (price_t / price_{t-periods} - 1) * 100
      - inventory_change = inventory_t - inventory_{t-periods}
      - inventory_change_pct = (inventory_t / inventory_{t-periods} - 1) * 100

    Sorting: commodity_id, then date (ascending) before any lag.
    Grouping: transforms are groupby(commodity_id) so lags never cross commodities.

    Lag basis: observation order within each commodity (not calendar-week fill).
    If a weekly date is missing, the next observed row diffs against the previous
    observed row (multi-week calendar span, one observation step).

    First `periods` observations per commodity remain NaN (no look-ahead fill).
    Prior price == 0 or prior inventory <= 0 → rate features are NaN.
    """
    required = {DATE_COL, ID_COL, price_col, inventory_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns for commodity features: {sorted(missing)}")

    out = _ensure_sorted(df)

    out["price_change_pct"] = out.groupby(ID_COL, group_keys=False)[price_col].transform(
        lambda s: price_change_rate(s, periods=periods)
    )
    out["inventory_change"] = out.groupby(ID_COL, group_keys=False)[inventory_col].transform(
        lambda s: inventory_change(s, periods=periods)
    )
    out["inventory_change_pct"] = out.groupby(ID_COL, group_keys=False)[
        inventory_col
    ].transform(lambda s: inventory_change_rate(s, periods=periods))

    # Preserve date as ISO string for CSV/SQLite consistency
    out[DATE_COL] = out[DATE_COL].dt.strftime("%Y-%m-%d")
    return out


def feature_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compact per-commodity feature coverage summary (for QA / reports)."""
    feature_cols = [
        c
        for c in ["price_change_pct", "inventory_change", "inventory_change_pct"]
        if c in df.columns
    ]
    if not feature_cols or ID_COL not in df.columns:
        return pd.DataFrame()

    rows = []
    for cid, g in df.groupby(ID_COL):
        row = {"commodity_id": cid, "n_rows": len(g)}
        for col in feature_cols:
            row[f"{col}_non_null"] = int(g[col].notna().sum())
        rows.append(row)
    return pd.DataFrame(rows)
