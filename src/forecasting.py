"""
Time series forecasting module.

Implements linear trend forecasting and ARIMA (statsmodels) with a
graceful fallback when ARIMA cannot be fitted.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
from scipy import stats

from src.config import load_settings

ForecastMethod = Literal["linear_trend", "arima", "ensemble"]


@dataclass
class ForecastResult:
    """Container for historical series + forecast horizon."""

    metric: str
    method: str
    history: pd.DataFrame
    forecast: pd.DataFrame
    metrics: dict[str, float] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "method": self.method,
            "history": self.history.to_dict(orient="records"),
            "forecast": self.forecast.to_dict(orient="records"),
            "metrics": self.metrics,
            "notes": self.notes,
        }


def aggregate_annual_series(
    df: pd.DataFrame,
    metric: str,
    industry: Optional[str] = None,
    region: Optional[str] = None,
) -> pd.Series:
    """
    Build a clean annual mean series for forecasting.

    Optionally filter by industry and/or region.
    """
    if metric not in df.columns:
        raise KeyError(f"Metric not found: {metric}")

    data = df.copy()
    if industry:
        data = data[data["industry"] == industry]
    if region:
        data = data[data["region"] == region]
    if data.empty:
        raise ValueError("No data after applying filters")

    series = data.groupby("year")[metric].mean().sort_index()
    series.name = metric
    return series


def forecast_linear_trend(
    series: pd.Series,
    horizon: int = 5,
    confidence: float = 0.95,
) -> ForecastResult:
    """
    Simple OLS linear trend extrapolation with confidence bands.
    """
    if len(series) < 3:
        raise ValueError("Need at least 3 observations for linear trend forecast")

    years = series.index.to_numpy(dtype=float)
    values = series.to_numpy(dtype=float)
    slope, intercept, r_value, p_value, std_err = stats.linregress(years, values)

    fitted = intercept + slope * years
    resid = values - fitted
    dof = max(len(years) - 2, 1)
    s_err = float(np.sqrt(np.sum(resid**2) / dof))

    last_year = int(years.max())
    future_years = np.arange(last_year + 1, last_year + horizon + 1, dtype=float)
    point = intercept + slope * future_years

    # Approximate prediction interval
    t_crit = float(stats.t.ppf((1 + confidence) / 2, dof))
    x_mean = years.mean()
    ssx = float(np.sum((years - x_mean) ** 2))
    se_pred = s_err * np.sqrt(1 + 1 / len(years) + (future_years - x_mean) ** 2 / ssx)
    lower = point - t_crit * se_pred
    upper = point + t_crit * se_pred

    history = pd.DataFrame(
        {
            "year": years.astype(int),
            "actual": values,
            "fitted": fitted,
        }
    )
    forecast = pd.DataFrame(
        {
            "year": future_years.astype(int),
            "forecast": np.round(point, 2),
            "lower": np.round(lower, 2),
            "upper": np.round(upper, 2),
        }
    )

    rmse = float(np.sqrt(np.mean(resid**2)))
    return ForecastResult(
        metric=str(series.name),
        method="linear_trend",
        history=history.round(2),
        forecast=forecast,
        metrics={
            "slope_per_year": round(float(slope), 4),
            "r_squared": round(float(r_value**2), 4),
            "p_value": round(float(p_value), 6),
            "rmse": round(rmse, 4),
        },
        notes="OLS linear trend with approximate prediction intervals.",
    )


def forecast_arima(
    series: pd.Series,
    horizon: int = 5,
    order: tuple[int, int, int] = (1, 1, 1),
) -> ForecastResult:
    """
    ARIMA forecast via statsmodels.

    Falls back to linear trend if ARIMA fitting fails.
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError as exc:
        raise ImportError(
            "statsmodels is required for ARIMA forecasting. "
            "Install via: pip install statsmodels"
        ) from exc

    if len(series) < 6:
        result = forecast_linear_trend(series, horizon=horizon)
        result.notes = "ARIMA skipped (series too short); used linear trend fallback."
        result.method = "linear_trend_fallback"
        return result

    try:
        import warnings

        values = series.to_numpy(dtype=float)
        fitted = None
        used_order = order
        # Try configured order, then a more stable differenced MA fallback
        for candidate in (order, (0, 1, 1), (1, 1, 0)):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fitted = ARIMA(values, order=candidate).fit()
                used_order = candidate
                break
            except Exception:  # noqa: BLE001
                continue
        if fitted is None:
            raise RuntimeError("Unable to fit ARIMA with candidate orders")

        fc = fitted.get_forecast(steps=horizon)
        pred = fc.predicted_mean
        ci = fc.conf_int(alpha=0.05)
        order = used_order

        last_year = int(series.index.max())
        future_years = list(range(last_year + 1, last_year + horizon + 1))

        # conf_int may be ndarray or DataFrame depending on version
        if hasattr(ci, "iloc"):
            lower = ci.iloc[:, 0].to_numpy()
            upper = ci.iloc[:, 1].to_numpy()
        else:
            lower = np.asarray(ci)[:, 0]
            upper = np.asarray(ci)[:, 1]

        history = pd.DataFrame(
            {
                "year": series.index.astype(int),
                "actual": series.to_numpy(),
                "fitted": fitted.fittedvalues,
            }
        ).round(2)

        forecast = pd.DataFrame(
            {
                "year": future_years,
                "forecast": np.round(np.asarray(pred), 2),
                "lower": np.round(lower, 2),
                "upper": np.round(upper, 2),
            }
        )

        resid = series.to_numpy(dtype=float) - np.asarray(fitted.fittedvalues)
        # Align lengths if differencing shortens fitted values
        n = min(len(resid), len(series))
        rmse = float(np.sqrt(np.mean(resid[-n:] ** 2)))

        return ForecastResult(
            metric=str(series.name),
            method=f"arima{order}",
            history=history,
            forecast=forecast,
            metrics={
                "aic": round(float(fitted.aic), 2),
                "bic": round(float(fitted.bic), 2),
                "rmse": round(rmse, 4),
            },
            notes=f"ARIMA{order} point forecast with 95% confidence intervals.",
        )
    except Exception as exc:  # noqa: BLE001
        result = forecast_linear_trend(series, horizon=horizon)
        result.notes = f"ARIMA failed ({exc}); used linear trend fallback."
        result.method = "linear_trend_fallback"
        return result


def forecast_ensemble(
    series: pd.Series,
    horizon: int = 5,
    arima_order: tuple[int, int, int] = (1, 1, 1),
) -> ForecastResult:
    """Average linear trend and ARIMA forecasts (simple ensemble)."""
    linear = forecast_linear_trend(series, horizon=horizon)
    arima = forecast_arima(series, horizon=horizon, order=arima_order)

    merged = linear.forecast[["year", "forecast", "lower", "upper"]].merge(
        arima.forecast[["year", "forecast", "lower", "upper"]],
        on="year",
        suffixes=("_lin", "_arima"),
    )
    forecast = pd.DataFrame(
        {
            "year": merged["year"],
            "forecast": ((merged["forecast_lin"] + merged["forecast_arima"]) / 2).round(2),
            "lower": ((merged["lower_lin"] + merged["lower_arima"]) / 2).round(2),
            "upper": ((merged["upper_lin"] + merged["upper_arima"]) / 2).round(2),
        }
    )

    return ForecastResult(
        metric=str(series.name),
        method="ensemble",
        history=linear.history,
        forecast=forecast,
        metrics={
            "linear_r_squared": linear.metrics.get("r_squared", float("nan")),
            "arima_aic": arima.metrics.get("aic", float("nan")),
        },
        notes="Simple average of linear trend and ARIMA forecasts.",
    )


def run_forecast(
    df: pd.DataFrame,
    metric: Optional[str] = None,
    method: ForecastMethod = "ensemble",
    horizon: Optional[int] = None,
    industry: Optional[str] = None,
    region: Optional[str] = None,
) -> ForecastResult:
    """
    High-level forecast API used by the dashboard and pipeline.
    """
    cfg = load_settings()
    metric = metric or cfg["forecasting"]["default_metric"]
    horizon = horizon or int(cfg["forecasting"]["horizon_years"])
    order = tuple(cfg["forecasting"].get("arima_order", [1, 1, 1]))

    series = aggregate_annual_series(df, metric=metric, industry=industry, region=region)

    if method == "linear_trend":
        return forecast_linear_trend(series, horizon=horizon)
    if method == "arima":
        return forecast_arima(series, horizon=horizon, order=order)  # type: ignore[arg-type]
    return forecast_ensemble(series, horizon=horizon, arima_order=order)  # type: ignore[arg-type]


def forecast_to_frame(result: ForecastResult) -> pd.DataFrame:
    """Combine history and forecast into a single plotting frame."""
    hist = result.history[["year", "actual"]].rename(columns={"actual": "value"})
    hist["series"] = "Historical"

    fut = result.forecast[["year", "forecast"]].rename(columns={"forecast": "value"})
    fut["series"] = "Forecast"

    bands = result.forecast[["year", "lower", "upper"]].copy()
    combined = pd.concat([hist, fut], ignore_index=True)
    return combined.merge(bands, on="year", how="left")
