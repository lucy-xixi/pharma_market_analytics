"""
Indicator calculation engine.

Reusable functions for normalization, growth rates, CAGR, and composite indices.
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import pandas as pd

from src.config import load_settings

NormalizeMethod = Literal["minmax", "zscore"]


def normalize_series(
    series: pd.Series,
    method: NormalizeMethod = "minmax",
) -> pd.Series:
    """Normalize a numeric series to [0, 1] (minmax) or z-scores."""
    s = pd.to_numeric(series, errors="coerce")
    if method == "zscore":
        std = s.std(ddof=0)
        if std == 0 or np.isnan(std):
            return pd.Series(np.zeros(len(s)), index=s.index)
        return (s - s.mean()) / std

    # minmax
    lo, hi = s.min(), s.max()
    if hi == lo or pd.isna(lo):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - lo) / (hi - lo)


def normalize_indicators(
    df: pd.DataFrame,
    columns: list[str],
    method: NormalizeMethod = "minmax",
    suffix: str = "_norm",
) -> pd.DataFrame:
    """
    Add normalized versions of indicator columns.

    Normalization is applied globally across the provided frame.
    """
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            raise KeyError(f"Column not found for normalization: {col}")
        out[f"{col}{suffix}"] = normalize_series(out[col], method=method)
    return out


def calculate_growth_rate(
    series: pd.Series,
    periods: int = 1,
) -> pd.Series:
    """
    Period-over-period growth rate as a percentage.

    growth = (x_t / x_{t-periods} - 1) * 100
    """
    s = pd.to_numeric(series, errors="coerce")
    prev = s.shift(periods)
    with np.errstate(divide="ignore", invalid="ignore"):
        growth = (s / prev - 1.0) * 100.0
    return growth.replace([np.inf, -np.inf], np.nan)


def calculate_cagr(
    start_value: float,
    end_value: float,
    periods: float,
) -> float:
    """
    Compound Annual Growth Rate.

    CAGR = (end / start) ** (1 / periods) - 1
    Returns percentage points.
    """
    if periods <= 0:
        raise ValueError("periods must be positive")
    if start_value <= 0 or end_value <= 0:
        return float("nan")
    return float(((end_value / start_value) ** (1.0 / periods) - 1.0) * 100.0)


def cagr_by_group(
    df: pd.DataFrame,
    value_col: str,
    group_cols: list[str],
    time_col: str = "year",
) -> pd.DataFrame:
    """Compute CAGR for each group between first and last observed year."""

    def _cagr(g: pd.DataFrame) -> float:
        g = g.sort_values(time_col)
        if len(g) < 2:
            return float("nan")
        start = float(g[value_col].iloc[0])
        end = float(g[value_col].iloc[-1])
        periods = float(g[time_col].iloc[-1] - g[time_col].iloc[0])
        try:
            return calculate_cagr(start, end, periods)
        except ValueError:
            return float("nan")

    frames = []
    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row[f"{value_col}_cagr"] = _cagr(g)
        frames.append(row)
    return pd.DataFrame(frames)


def calculate_employment_growth(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add YoY employment growth (%) within each industry-region series.
    """
    out = df.sort_values(["industry", "region", "year"]).copy()
    out["employment_growth"] = out.groupby(["industry", "region"])["employment"].transform(
        calculate_growth_rate
    )
    # First year has no prior period — fill with 0 for composite index stability
    out["employment_growth"] = out["employment_growth"].fillna(0.0)
    return out


def calculate_digital_development_index(
    df: pd.DataFrame,
    weights: Optional[dict[str, float]] = None,
    method: NormalizeMethod = "minmax",
) -> pd.DataFrame:
    """
    Composite Digital Development Index (DDI).

    DDI = Σ w_i * normalize(component_i)

    Default components:
      - technology_investment
      - digital_index (digital adoption proxy)
      - rd_ratio (R&D intensity)
      - employment_growth
    """
    cfg = load_settings()
    weights = weights or cfg["indicators"]["composite_weights"]
    method = method or cfg["indicators"].get("normalize_method", "minmax")

    out = df.copy()
    if "employment_growth" not in out.columns:
        out = calculate_employment_growth(out)

    # Map weight keys to columns
    component_map = {
        "technology_investment": "technology_investment",
        "digital_index": "digital_index",
        "rd_ratio": "rd_ratio",
        "employment_growth": "employment_growth",
    }

    # Normalize weights
    total_w = sum(weights.get(k, 0.0) for k in component_map)
    if total_w <= 0:
        raise ValueError("Composite weights must sum to a positive value")

    ddi = pd.Series(0.0, index=out.index)
    for key, col in component_map.items():
        w = weights.get(key, 0.0) / total_w
        if col not in out.columns:
            raise KeyError(f"Missing component column for DDI: {col}")
        norm = normalize_series(out[col], method=method)
        out[f"{col}_norm"] = norm
        ddi = ddi + w * norm

    # Scale to 0–100 for interpretability
    out["digital_development_index"] = (ddi * 100).round(2)
    return out


def enrich_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply full indicator enrichment pipeline to a clean panel.
    """
    cfg = load_settings()
    out = calculate_employment_growth(df)
    out = calculate_digital_development_index(
        out,
        weights=cfg["indicators"]["composite_weights"],
        method=cfg["indicators"].get("normalize_method", "minmax"),
    )

    # YoY growth for key metrics
    for col in ["digital_index", "technology_investment", "revenue_growth"]:
        out[f"{col}_yoy"] = out.groupby(["industry", "region"])[col].transform(
            calculate_growth_rate
        )

    return out


def ranking_table(
    df: pd.DataFrame,
    metric: str = "digital_development_index",
    year: Optional[int] = None,
    group_by: str = "industry",
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Rank groups by average metric for a given year (or latest year).
    """
    if metric not in df.columns:
        raise KeyError(f"Metric not found: {metric}")

    data = df.copy()
    if year is None:
        year = int(data["year"].max())
    data = data[data["year"] == year]

    ranked = (
        data.groupby(group_by, as_index=False)[metric]
        .mean()
        .sort_values(metric, ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    ranked["year"] = year
    return ranked
