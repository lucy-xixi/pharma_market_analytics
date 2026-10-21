"""
Calendar growth features for the pharmaceutical sales panel.

Month-over-month uses the previous calendar month only.
Year-over-year uses the same month one year earlier.
If that month is absent, the rate is missing. An earlier row is not substituted.
A zero denominator also leaves the rate missing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _rate(current: float, previous: float) -> float:
    if pd.isna(current) or pd.isna(previous) or previous == 0:
        return float("nan")
    return float((current / previous - 1.0) * 100.0)


def _calendar_rates(frame: pd.DataFrame, value_col: str, months_back: int) -> list[float]:
    indexed = frame.set_index("month")[value_col]
    rates: list[float] = []
    for timestamp, value in indexed.items():
        previous_month = pd.Timestamp(timestamp) - pd.DateOffset(months=months_back)
        if previous_month in indexed.index:
            rates.append(_rate(float(value), float(indexed.loc[previous_month])))
        else:
            rates.append(float("nan"))
    return rates


def add_growth_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add MoM, YoY, and within region-month sales share. Does not fill gaps."""
    if df.empty:
        out = df.copy()
        for column in (
            "mom_sales_amount_pct",
            "yoy_sales_amount_pct",
            "mom_sales_volume_pct",
            "yoy_sales_volume_pct",
            "sales_share_pct",
        ):
            out[column] = pd.Series(dtype="float")
        return out

    work = df.copy()
    work["month"] = pd.to_datetime(work["month"], errors="coerce")
    work = work.dropna(subset=["month"]).sort_values(["product", "region", "month"])
    parts: list[pd.DataFrame] = []
    for (_, _), group in work.groupby(["product", "region"], sort=False):
        group = group.copy()
        group["mom_sales_amount_pct"] = _calendar_rates(group, "sales_amount", 1)
        group["yoy_sales_amount_pct"] = _calendar_rates(group, "sales_amount", 12)
        group["mom_sales_volume_pct"] = _calendar_rates(group, "sales_volume", 1)
        group["yoy_sales_volume_pct"] = _calendar_rates(group, "sales_volume", 12)
        parts.append(group)
    out = pd.concat(parts, ignore_index=True)
    region_month_total = out.groupby(["month", "region"])["sales_amount"].transform("sum")
    out["sales_share_pct"] = np.where(
        region_month_total > 0,
        out["sales_amount"] / region_month_total * 100.0,
        np.nan,
    )
    out["month"] = out["month"].dt.strftime("%Y-%m-%d")
    return out.sort_values(["month", "product", "region"]).reset_index(drop=True)


def cagr_percent(start_value: float, end_value: float, months_between: int) -> float:
    """Annualized growth between two month starts. months_between is the span."""
    if months_between <= 0 or start_value <= 0 or end_value <= 0:
        return float("nan")
    years = months_between / 12.0
    return float(((end_value / start_value) ** (1.0 / years) - 1.0) * 100.0)


def cagr_table(
    df: pd.DataFrame,
    group_cols: list[str] | None = None,
    value_col: str = "sales_amount",
) -> pd.DataFrame:
    """CAGR from the first observed month to the last observed month in each group."""
    group_cols = group_cols or ["product", "region"]
    work = df.copy()
    work["month"] = pd.to_datetime(work["month"], errors="coerce")
    rows: list[dict] = []
    for keys, group in work.groupby(group_cols, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        series = group.groupby("month", as_index=True)[value_col].sum().sort_index()
        if len(series) < 2:
            cagr = float("nan")
            months_between = 0
            start_month = end_month = None
            start_value = end_value = float("nan")
        else:
            start_month = pd.Timestamp(series.index[0])
            end_month = pd.Timestamp(series.index[-1])
            months_between = (end_month.year - start_month.year) * 12 + (
                end_month.month - start_month.month
            )
            start_value = float(series.iloc[0])
            end_value = float(series.iloc[-1])
            cagr = cagr_percent(start_value, end_value, months_between)
        row = dict(zip(group_cols, keys))
        row.update(
            {
                "start_month": None if start_month is None else start_month.strftime("%Y-%m-%d"),
                "end_month": None if end_month is None else end_month.strftime("%Y-%m-%d"),
                "months_between": int(months_between),
                "start_value": start_value,
                "end_value": end_value,
                "cagr_pct": cagr,
            }
        )
        rows.append(row)
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    return table.sort_values("cagr_pct", ascending=False, na_position="last").reset_index(drop=True)
