"""
Statistical analysis module.

Descriptive statistics, correlation analysis, and linear regression
for digital industry growth drivers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from src.config import load_settings


@dataclass
class RegressionResult:
    """Structured linear regression output."""

    target: str
    features: list[str]
    n_observations: int
    r_squared: float
    adj_r_squared: float
    rmse: float
    intercept: float
    coefficients: dict[str, float]
    p_values: dict[str, float]
    std_errors: dict[str, float]
    significant_features: list[str]
    interpretation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AnalyticsBundle:
    """Aggregated analytics artifacts for reporting / LLM context."""

    descriptive: dict[str, Any] = field(default_factory=dict)
    correlations: dict[str, Any] = field(default_factory=dict)
    regression: dict[str, Any] = field(default_factory=dict)
    trend_summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def descriptive_statistics(
    df: pd.DataFrame,
    columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Compute descriptive statistics for numeric indicators.

    Includes count, mean, std, min, quartiles, max, skewness, kurtosis.
    """
    cols = columns or list(df.select_dtypes(include=[np.number]).columns)
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return pd.DataFrame()

    desc = df[cols].describe().T
    desc["skewness"] = df[cols].skew()
    desc["kurtosis"] = df[cols].kurtosis()
    desc["missing"] = df[cols].isna().sum()
    return desc.round(4)


def correlation_matrix(
    df: pd.DataFrame,
    columns: Optional[list[str]] = None,
    method: str = "pearson",
) -> pd.DataFrame:
    """Compute pairwise correlation matrix."""
    cols = columns or [
        "digital_index",
        "technology_investment",
        "revenue_growth",
        "employment",
        "rd_ratio",
        "digital_development_index",
        "employment_growth",
    ]
    cols = [c for c in cols if c in df.columns]
    return df[cols].corr(method=method).round(4)


def correlation_with_target(
    df: pd.DataFrame,
    target: str = "revenue_growth",
    features: Optional[list[str]] = None,
    method: str = "pearson",
) -> pd.DataFrame:
    """Rank features by absolute correlation with a target variable."""
    feat = features or [
        c
        for c in df.select_dtypes(include=[np.number]).columns
        if c not in {target, "year"}
    ]
    rows = []
    for col in feat:
        if col not in df.columns:
            continue
        pair = df[[col, target]].dropna()
        if len(pair) < 3:
            continue
        if method == "spearman":
            r, p = stats.spearmanr(pair[col], pair[target])
        else:
            r, p = stats.pearsonr(pair[col], pair[target])
        rows.append(
            {
                "feature": col,
                "correlation": round(float(r), 4),
                "abs_correlation": round(abs(float(r)), 4),
                "p_value": round(float(p), 6),
                "significant": bool(p < 0.05),
            }
        )
    result = pd.DataFrame(rows).sort_values("abs_correlation", ascending=False)
    return result.reset_index(drop=True)


def linear_regression_analysis(
    df: pd.DataFrame,
    target: Optional[str] = None,
    features: Optional[list[str]] = None,
    significance_level: float = 0.05,
) -> RegressionResult:
    """
    Fit OLS linear regression and return coefficients with p-values.

    Addresses: What factors influence digital industry growth?
    """
    cfg = load_settings()
    target = target or cfg["analytics"]["regression_target"]
    features = features or cfg["analytics"]["regression_features"]
    features = [f for f in features if f in df.columns and f != target]

    if target not in df.columns:
        raise KeyError(f"Target column not found: {target}")
    if not features:
        raise ValueError("No valid regression features available")

    data = df[features + [target]].dropna()
    if len(data) < len(features) + 2:
        raise ValueError("Insufficient observations for regression")

    X = data[features].to_numpy(dtype=float)
    y = data[target].to_numpy(dtype=float)

    # Standardized coefficients for interpretability + raw model for prediction metrics
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegression()
    model.fit(X_scaled, y)
    y_pred = model.predict(X_scaled)

    r2 = float(r2_score(y, y_pred))
    n, k = X_scaled.shape
    adj_r2 = float(1 - (1 - r2) * (n - 1) / max(n - k - 1, 1))
    rmse = float(np.sqrt(mean_squared_error(y, y_pred)))

    # Manual OLS SE / p-values on standardized design (+ intercept)
    X_design = np.column_stack([np.ones(n), X_scaled])
    beta = np.linalg.lstsq(X_design, y, rcond=None)[0]
    residuals = y - X_design @ beta
    dof = max(n - X_design.shape[1], 1)
    mse = float(residuals @ residuals / dof)
    try:
        cov = mse * np.linalg.inv(X_design.T @ X_design)
        se = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        se = np.full(X_design.shape[1], np.nan)

    t_stats = beta / se
    p_values = 2 * stats.t.sf(np.abs(t_stats), dof)

    coef_map = {f: round(float(c), 4) for f, c in zip(features, beta[1:])}
    se_map = {f: round(float(s), 4) for f, s in zip(features, se[1:])}
    p_map = {f: round(float(p), 6) for f, p in zip(features, p_values[1:])}
    significant = [f for f, p in p_map.items() if p < significance_level]

    # Human-readable interpretation
    if significant:
        ranked = sorted(significant, key=lambda f: abs(coef_map[f]), reverse=True)
        top = ranked[0]
        direction = "positively" if coef_map[top] > 0 else "negatively"
        interpretation = (
            f"Model explains {r2:.1%} of variance in {target}. "
            f"Strongest driver: {top} ({direction} associated). "
            f"Significant features (p<{significance_level}): {', '.join(ranked)}."
        )
    else:
        interpretation = (
            f"Model R²={r2:.3f}; no features reached p<{significance_level} significance."
        )

    return RegressionResult(
        target=target,
        features=features,
        n_observations=n,
        r_squared=round(r2, 4),
        adj_r_squared=round(adj_r2, 4),
        rmse=round(rmse, 4),
        intercept=round(float(beta[0]), 4),
        coefficients=coef_map,
        p_values=p_map,
        std_errors=se_map,
        significant_features=significant,
        interpretation=interpretation,
    )


def trend_analysis(
    df: pd.DataFrame,
    metric: str = "digital_index",
    group_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute annual mean trends, optionally by group (industry/region).
    Includes linear slope (units per year) via OLS.
    """
    if metric not in df.columns:
        raise KeyError(f"Metric not found: {metric}")

    group_keys = ["year"] if group_col is None else [group_col, "year"]
    trend = df.groupby(group_keys, as_index=False)[metric].mean()

    def _slope(g: pd.DataFrame) -> float:
        if len(g) < 2:
            return float("nan")
        x = g["year"].to_numpy(dtype=float)
        y = g[metric].to_numpy(dtype=float)
        slope, _, _, _, _ = stats.linregress(x, y)
        return float(slope)

    if group_col:
        slope_rows = [
            {group_col: key, "annual_slope": _slope(g)}
            for key, g in trend.groupby(group_col)
        ]
        slopes = pd.DataFrame(slope_rows)
        trend = trend.merge(slopes, on=group_col, how="left")
    else:
        trend["annual_slope"] = _slope(trend)

    return trend.round(4)


def industry_region_summary(df: pd.DataFrame, year: Optional[int] = None) -> pd.DataFrame:
    """KPI-style summary table for dashboard cards."""
    data = df.copy()
    if year is None:
        year = int(data["year"].max())
    latest = data[data["year"] == year]
    prev = data[data["year"] == year - 1] if year - 1 in set(data["year"]) else None

    metrics = [
        "digital_index",
        "technology_investment",
        "revenue_growth",
        "employment",
        "rd_ratio",
        "digital_development_index",
    ]
    metrics = [m for m in metrics if m in latest.columns]

    rows = []
    for m in metrics:
        current = float(latest[m].mean())
        change = None
        if prev is not None and m in prev.columns:
            prior = float(prev[m].mean())
            change = ((current / prior) - 1) * 100 if prior else None
        rows.append(
            {
                "metric": m,
                "year": year,
                "value": round(current, 2),
                "yoy_change_pct": round(change, 2) if change is not None else None,
            }
        )
    return pd.DataFrame(rows)


def run_full_analytics(df: pd.DataFrame) -> AnalyticsBundle:
    """Execute the standard analytics suite and package results."""
    cfg = load_settings()
    corr_cols = [
        c
        for c in [
            "digital_index",
            "technology_investment",
            "revenue_growth",
            "employment",
            "rd_ratio",
            "digital_development_index",
        ]
        if c in df.columns
    ]

    desc = descriptive_statistics(df, columns=corr_cols)
    corr = correlation_matrix(df, columns=corr_cols, method=cfg["analytics"]["correlation_method"])
    corr_target = correlation_with_target(df, target=cfg["analytics"]["regression_target"])
    reg = linear_regression_analysis(df)
    trend = trend_analysis(df, metric="digital_index", group_col="industry")

    return AnalyticsBundle(
        descriptive=desc.reset_index().rename(columns={"index": "metric"}).to_dict(orient="records"),
        correlations={
            "matrix": corr.to_dict(),
            "vs_revenue_growth": corr_target.to_dict(orient="records"),
        },
        regression=reg.to_dict(),
        trend_summary={
            "by_industry_latest": (
                trend.sort_values("year")
                .groupby("industry")
                .tail(1)[["industry", "digital_index", "annual_slope"]]
                .to_dict(orient="records")
            )
        },
    )
