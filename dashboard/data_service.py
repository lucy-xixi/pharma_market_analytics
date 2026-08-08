"""
Dashboard data access layer.

Caches enriched panel loads and analytics so Streamlit pages stay responsive.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analytics import (  # noqa: E402
    correlation_matrix,
    descriptive_statistics,
    industry_region_summary,
    linear_regression_analysis,
    run_full_analytics,
)
from src.config import get_db_path, get_processed_dir, load_settings  # noqa: E402
from src.data_cleaning import (  # noqa: E402
    DataQualityReport,
    build_quality_monitoring_summary,
    clean_dataset,
    generate_quality_report,
)
from src.data_loader import (  # noqa: E402
    load_from_sqlite,
    load_primary_panel,
    save_to_sqlite,
)
from src.forecasting import ForecastResult, run_forecast  # noqa: E402
from src.indicator_engine import enrich_indicators, ranking_table  # noqa: E402
from src.llm_summary import InsightReport, build_context_from_dataframe, generate_insights  # noqa: E402


def _report_from_dict(payload: dict) -> DataQualityReport:
    return DataQualityReport(
        source=payload.get("source", "unknown"),
        n_rows=int(payload.get("n_rows", 0)),
        n_columns=int(payload.get("n_columns", 0)),
        missing_by_column=payload.get("missing_by_column", {}),
        missing_pct_by_column=payload.get("missing_pct_by_column", {}),
        duplicate_rows=int(payload.get("duplicate_rows", 0)),
        dtype_issues=list(payload.get("dtype_issues", [])),
        range_violations=payload.get("range_violations", {}),
        outlier_counts=payload.get("outlier_counts", {}),
        created_at=payload.get("created_at", ""),
    )


def _load_saved_intake_report() -> DataQualityReport | None:
    path = get_processed_dir() / "data_quality_report.json"
    if not path.exists():
        return None
    try:
        import json

        return _report_from_dict(json.loads(path.read_text(encoding="utf-8")))
    except Exception:  # noqa: BLE001
        return None


def ensure_data_ready() -> pd.DataFrame:
    """
    Return enriched industry panel, building it from raw CSV if DB is empty.
    """
    db_path = get_db_path()
    processed = get_processed_dir() / "industry_panel_clean.csv"

    if db_path.exists():
        try:
            df = load_from_sqlite()
            if not df.empty:
                return df
        except Exception:  # noqa: BLE001
            pass

    if processed.exists():
        df = pd.read_csv(processed)
        if not df.empty:
            return df

    # Bootstrap from raw
    raw = load_primary_panel()
    clean, _ = clean_dataset(raw, persist_report=False)
    enriched = enrich_indicators(clean)
    try:
        save_to_sqlite(enriched)
    except Exception:  # noqa: BLE001
        pass
    return enriched


@st.cache_data(show_spinner=False)
def get_panel() -> pd.DataFrame:
    return ensure_data_ready()


@st.cache_data(show_spinner=False)
def get_quality_monitoring() -> dict:
    """
    Enterprise data quality monitoring snapshot.

    Compares source intake (pre-clean) vs analytical store (post-prep).
    """
    panel = get_panel()
    # Assess the analytical store on core indicator columns only
    core_cols = [
        c
        for c in [
            "year",
            "industry",
            "region",
            "digital_index",
            "technology_investment",
            "revenue_growth",
            "employment",
            "rd_ratio",
        ]
        if c in panel.columns
    ]
    store_report = generate_quality_report(
        panel[core_cols],
        source="analytical_store (SQLite / processed)",
    )
    store_summary = build_quality_monitoring_summary(
        store_report,
        stage="analytical_store",
    )

    intake_report = _load_saved_intake_report()
    if intake_report is None:
        try:
            raw = load_primary_panel()
            intake_report = generate_quality_report(
                raw,
                source="digital_industry_indicators.csv",
            )
        except Exception:  # noqa: BLE001
            intake_report = store_report

    intake_summary = build_quality_monitoring_summary(
        intake_report,
        stage="source_intake",
    )

    return {
        "primary": store_summary,
        "source_intake": intake_summary,
        "analytical_store": store_summary,
        "records_removed": max(
            0,
            int(intake_summary["total_records"]) - int(store_summary["total_records"]),
        ),
        "score_improvement": round(
            float(store_summary["data_quality_score"])
            - float(intake_summary["data_quality_score"]),
            1,
        ),
    }


@st.cache_data(show_spinner=False)
def get_kpis(year: int | None = None) -> pd.DataFrame:
    return industry_region_summary(get_panel(), year=year)


@st.cache_data(show_spinner=False)
def get_rankings(metric: str, year: int, group_by: str) -> pd.DataFrame:
    return ranking_table(get_panel(), metric=metric, year=year, group_by=group_by)


@st.cache_data(show_spinner=False)
def get_correlation(columns: tuple[str, ...]) -> pd.DataFrame:
    return correlation_matrix(get_panel(), columns=list(columns))


@st.cache_data(show_spinner=False)
def get_regression() -> dict:
    return linear_regression_analysis(get_panel()).to_dict()


@st.cache_data(show_spinner=False)
def get_forecast(
    metric: str,
    method: str,
    horizon: int,
    industry: str | None,
    region: str | None,
) -> dict:
    result = run_forecast(
        get_panel(),
        metric=metric,
        method=method,  # type: ignore[arg-type]
        horizon=horizon,
        industry=industry or None,
        region=region or None,
    )
    return result.to_dict()


def forecast_from_dict(payload: dict) -> ForecastResult:
    return ForecastResult(
        metric=payload["metric"],
        method=payload["method"],
        history=pd.DataFrame(payload["history"]),
        forecast=pd.DataFrame(payload["forecast"]),
        metrics=payload.get("metrics", {}),
        notes=payload.get("notes", ""),
    )


def run_ai_insight(question: str, use_llm: bool = True) -> InsightReport:
    df = get_panel()
    context = build_context_from_dataframe(df, include_forecast=True)
    return generate_insights(context, user_question=question, use_llm=use_llm)


@lru_cache(maxsize=1)
def theme_colors() -> dict[str, str]:
    cfg = load_settings()
    primary = cfg.get("dashboard", {}).get("theme_primary", "#0B3D5C")
    return {
        "primary": primary,
        "accent": "#1F7A8C",
        "warm": "#C17A3A",
        "muted": "#6B7280",
        "bg": "#F7F9FB",
    }
