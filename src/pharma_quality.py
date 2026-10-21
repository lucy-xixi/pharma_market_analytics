"""
Quality checks for the pharmaceutical sales panel.

Policies, also recorded on the report:

- Negative sales_amount or sales_volume: exclude the row. Do not clip to zero.
- Duplicate month + product + region: keep the first row, count the excess.
- Missing months: list them. Do not insert rows.
- IQR outliers: count them. Do not change the stored values.
- Invalid dates or non-numeric sales: exclude the row.

The raw section describes the input. The analytical section describes the
table after those exclusions.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.pharma_schema import REQUIRED_COLUMNS, UNIQUE_KEY

POLICY: dict[str, str] = {
    "negative_sales": "exclude_row",
    "duplicate_keys": "keep_first",
    "missing_months": "report_only",
    "outliers": "flag_only",
    "invalid_dates": "exclude_row",
    "non_numeric_sales": "exclude_row",
}


def _iqr_flags(series: pd.Series) -> int:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if len(numeric) < 4:
        return 0
    q1 = float(numeric.quantile(0.25))
    q3 = float(numeric.quantile(0.75))
    iqr = q3 - q1
    if iqr == 0:
        return 0
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    values = pd.to_numeric(series, errors="coerce")
    return int(((values < low) | (values > high)).fillna(False).sum())


def _missing_months(df: pd.DataFrame) -> list[dict[str, str]]:
    """Months absent inside each product-region span. Does not invent rows."""
    if df.empty or not {"month", "product", "region"}.issubset(df.columns):
        return []
    work = df.copy()
    work["month"] = pd.to_datetime(work["month"], errors="coerce")
    work = work.dropna(subset=["month", "product", "region"])
    gaps: list[dict[str, str]] = []
    for (product, region), group in work.groupby(["product", "region"], sort=True):
        observed = pd.DatetimeIndex(sorted(pd.to_datetime(group["month"].unique())))
        if len(observed) < 2:
            continue
        expected = pd.date_range(observed.min(), observed.max(), freq="MS")
        for missing in expected.difference(observed):
            gaps.append(
                {
                    "product": str(product),
                    "region": str(region),
                    "month": pd.Timestamp(missing).strftime("%Y-%m-%d"),
                }
            )
    return gaps


def _profile(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {
            "n_rows": 0,
            "nulls_by_column": {},
            "duplicate_key_excess": 0,
            "invalid_month_rows": 0,
            "negative_sales_amount_rows": 0,
            "negative_sales_volume_rows": 0,
            "non_numeric_sales_amount_rows": 0,
            "non_numeric_sales_volume_rows": 0,
            "outlier_flags": {"sales_amount": 0, "sales_volume": 0},
            "missing_month_count": 0,
            "missing_months": [],
        }
    nulls = {
        column: int(df[column].isna().sum())
        for column in REQUIRED_COLUMNS
        if column in df.columns and int(df[column].isna().sum()) > 0
    }
    months = (
        pd.to_datetime(df["month"], errors="coerce")
        if "month" in df.columns
        else pd.Series(pd.NaT, index=df.index)
    )
    invalid_months = int(months.isna().sum())
    parsed = months.dropna()
    if len(parsed):
        invalid_months += int((~parsed.dt.is_month_start).sum())

    duplicate_excess = 0
    if set(UNIQUE_KEY).issubset(df.columns):
        key = pd.DataFrame(
            {
                "month": months.dt.strftime("%Y-%m-%d"),
                "product": df["product"].astype(str),
                "region": df["region"].astype(str),
            }
        )
        duplicate_excess = int(key.duplicated(UNIQUE_KEY).sum())

    def _negative(column: str) -> int:
        if column not in df.columns:
            return 0
        numeric = pd.to_numeric(df[column], errors="coerce")
        return int((numeric < 0).fillna(False).sum())

    def _non_numeric(column: str) -> int:
        if column not in df.columns:
            return 0
        numeric = pd.to_numeric(df[column], errors="coerce")
        return int((numeric.isna() & df[column].notna()).sum())

    gaps = _missing_months(df.loc[months.notna() & months.dt.is_month_start].copy())
    return {
        "n_rows": int(len(df)),
        "nulls_by_column": nulls,
        "duplicate_key_excess": duplicate_excess,
        "invalid_month_rows": invalid_months,
        "negative_sales_amount_rows": _negative("sales_amount"),
        "negative_sales_volume_rows": _negative("sales_volume"),
        "non_numeric_sales_amount_rows": _non_numeric("sales_amount"),
        "non_numeric_sales_volume_rows": _non_numeric("sales_volume"),
        "outlier_flags": {
            "sales_amount": _iqr_flags(df["sales_amount"]) if "sales_amount" in df.columns else 0,
            "sales_volume": _iqr_flags(df["sales_volume"]) if "sales_volume" in df.columns else 0,
        },
        "missing_month_count": len(gaps),
        "missing_months": gaps,
    }


def prepare_sales_panel(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Apply the explicit exclusion policy and return the analytical table.

    The input frame is not modified. Missing months are not added.
    Outlier values that remain in the analytical table are unchanged.
    """
    raw = df.copy()
    raw_profile = _profile(raw)
    work = raw.copy()
    for column in REQUIRED_COLUMNS:
        if column not in work.columns:
            work[column] = np.nan

    keep = pd.Series(True, index=work.index)
    key_missing = work[["product", "region", "data_source", "data_type"]].isna().any(axis=1)
    blank_text = pd.Series(False, index=work.index)
    for column in ("product", "region", "data_source", "data_type"):
        blank_text = blank_text | work[column].astype(str).str.strip().eq("") | work[column].astype(str).eq("nan")
    keep &= ~key_missing & ~blank_text

    months = pd.to_datetime(work["month"], errors="coerce")
    keep &= months.notna() & months.dt.is_month_start.fillna(False)

    amount = pd.to_numeric(work["sales_amount"], errors="coerce")
    volume = pd.to_numeric(work["sales_volume"], errors="coerce")
    keep &= amount.notna() & volume.notna()
    keep &= (amount >= 0) & (volume >= 0)

    excluded_before_dedupe = int((~keep).sum())
    analytical = work.loc[keep].copy()
    analytical["month"] = months.loc[keep].dt.strftime("%Y-%m-%d")
    analytical["product"] = analytical["product"].astype(str).str.strip()
    analytical["region"] = analytical["region"].astype(str).str.strip()
    analytical["data_source"] = analytical["data_source"].astype(str).str.strip()
    analytical["data_type"] = analytical["data_type"].astype(str).str.strip()
    analytical["sales_amount"] = amount.loc[keep].astype(float)
    analytical["sales_volume"] = volume.loc[keep].astype(float)
    before = len(analytical)
    analytical = analytical.drop_duplicates(UNIQUE_KEY, keep="first")
    duplicate_removed = before - len(analytical)
    analytical = analytical[REQUIRED_COLUMNS].sort_values(UNIQUE_KEY).reset_index(drop=True)

    report = {
        "policy": dict(POLICY),
        "policy_notes": [
            "These rules are conventions for this demo project, not universal rules for real pharmaceutical sales data.",
            "Negative sales rows are removed and are not clipped to zero.",
            "Duplicate business keys keep the first row in the input order.",
            "Missing months are listed and are not filled.",
            "IQR outlier flags do not change sales values.",
        ],
        "raw": raw_profile,
        "rows_excluded": int(len(raw) - len(analytical)),
        "exclusion_counts": {
            "invalid_or_negative_or_non_numeric": excluded_before_dedupe,
            "duplicate_keys": int(duplicate_removed),
        },
        "analytical": _profile(analytical),
    }
    return analytical, report
