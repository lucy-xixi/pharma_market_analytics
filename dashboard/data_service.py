"""Cached access to the pharmaceutical sales panel for the Streamlit app."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_pharma_settings  # noqa: E402
from src.pharma_features import add_growth_features, cagr_table  # noqa: E402
from src.pharma_forecast import GRAIN_TOTAL, evaluate_grouped_forecast  # noqa: E402
from src.pharma_quality import prepare_sales_panel  # noqa: E402
from src.pharma_schema import load_pharma_panel  # noqa: E402


@st.cache_data(show_spinner=False)
def load_sales_bundle() -> tuple[pd.DataFrame, dict]:
    """Prepare the configured pharma CSV. Does not read other raw files."""
    raw = load_pharma_panel()
    analytical, quality = prepare_sales_panel(raw)
    return add_growth_features(analytical), quality


def evaluation_settings() -> dict:
    return load_pharma_settings().get("evaluation", {})


def filter_panel(
    df: pd.DataFrame,
    products: tuple[str, ...],
    regions: tuple[str, ...],
    month_start: str | None = None,
    month_end: str | None = None,
) -> pd.DataFrame:
    if not products or not regions:
        return df.iloc[0:0].copy()
    out = df[df["product"].isin(products) & df["region"].isin(regions)].copy()
    if month_start:
        out = out[out["month"] >= month_start]
    if month_end:
        out = out[out["month"] <= month_end]
    return out


def monthly_total(df: pd.DataFrame) -> pd.DataFrame:
    """One row per month: sum of the current filter, then calendar growth."""
    if df.empty:
        return df
    grouped = (
        df.groupby("month", as_index=False)[["sales_amount", "sales_volume"]]
        .sum()
        .sort_values("month")
    )
    grouped["product"] = "筛选合计"
    grouped["region"] = "筛选合计"
    grouped["data_source"] = ",".join(sorted(df["data_source"].dropna().astype(str).unique()))
    grouped["data_type"] = ",".join(sorted(df["data_type"].dropna().astype(str).unique()))
    return add_growth_features(grouped)


def forecast_input_frame(
    df: pd.DataFrame,
    products: tuple[str, ...],
    regions: tuple[str, ...],
) -> pd.DataFrame:
    """Rows for forecasting. The month slider is not an argument and does not apply."""
    return filter_panel(df, products, regions)


@st.cache_data(show_spinner=False)
def forecast_selection(
    products: tuple[str, ...],
    regions: tuple[str, ...],
    grain: str = GRAIN_TOTAL,
) -> dict:
    panel, _quality = load_sales_bundle()
    selected = forecast_input_frame(panel, products, regions)
    settings = evaluation_settings()
    return evaluate_grouped_forecast(
        selected,
        grain=grain,
        test_months=int(settings.get("test_months", 6)),
        horizon_months=int(settings.get("horizon_months", 6)),
        min_train_months=int(settings.get("min_train_months", 18)),
        season_length=int(settings.get("season_length", 12)),
        value_col=str(settings.get("target_column", "sales_amount")),
    )


def product_cagr(df: pd.DataFrame) -> pd.DataFrame:
    return cagr_table(df, group_cols=["product"], value_col="sales_amount")
