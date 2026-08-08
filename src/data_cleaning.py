"""
Data cleaning, validation, and quality reporting module.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.config import get_db_path
from src.data_loader import REQUIRED_COLUMNS

EXPECTED_DTYPES: dict[str, str] = {
    "year": "int",
    "industry": "str",
    "region": "str",
    "digital_index": "float",
    "technology_investment": "float",
    "revenue_growth": "float",
    "employment": "float",
    "rd_ratio": "float",
}

NUMERIC_BOUNDS: dict[str, tuple[float, float]] = {
    "year": (2000, 2100),
    "digital_index": (0, 100),
    "technology_investment": (0, 1000),
    "revenue_growth": (-50, 80),
    "employment": (0, 10_000),
    "rd_ratio": (0, 50),
}


@dataclass
class DataQualityReport:
    """Structured data quality assessment."""

    source: str
    n_rows: int
    n_columns: int
    missing_by_column: dict[str, int] = field(default_factory=dict)
    missing_pct_by_column: dict[str, float] = field(default_factory=dict)
    duplicate_rows: int = 0
    dtype_issues: list[str] = field(default_factory=list)
    range_violations: dict[str, int] = field(default_factory=dict)
    outlier_counts: dict[str, int] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @property
    def is_healthy(self) -> bool:
        """True when no critical quality issues remain after assessment of raw data."""
        high_missing = any(v > 20 for v in self.missing_pct_by_column.values())
        return (
            not high_missing
            and self.duplicate_rows == 0
            and not self.dtype_issues
            and sum(self.range_violations.values()) == 0
        )

    @property
    def overall_missing_pct(self) -> float:
        """Share of all cells that are missing (0–100)."""
        if self.n_rows == 0 or self.n_columns == 0:
            return 0.0
        total_cells = self.n_rows * self.n_columns
        missing_cells = sum(self.missing_by_column.values())
        return round(missing_cells / total_cells * 100, 2)

    @property
    def range_violation_total(self) -> int:
        return int(sum(self.range_violations.values()))

    @property
    def outlier_total(self) -> int:
        return int(sum(self.outlier_counts.values()))


def compute_quality_score(report: DataQualityReport) -> float:
    """
    Composite data quality score (0–100) for enterprise monitoring.

    Deducts from a perfect score based on missingness, duplicates,
    schema/dtype issues, business-rule violations, and outliers.
    """
    if report.n_rows == 0:
        return 0.0

    score = 100.0

    # Missingness: up to -40
    score -= min(40.0, report.overall_missing_pct * 4.0)

    # Duplicates: up to -25 (share of rows)
    dup_rate = report.duplicate_rows / report.n_rows * 100
    score -= min(25.0, dup_rate * 5.0)

    # Schema / dtype issues: -8 each, up to -16
    score -= min(16.0, 8.0 * len(report.dtype_issues))

    # Range / business-rule violations: up to -14
    viol_rate = report.range_violation_total / report.n_rows * 100
    score -= min(14.0, viol_rate * 3.0)

    # Outliers (soft signal): up to -5
    out_rate = report.outlier_total / report.n_rows * 100
    score -= min(5.0, out_rate * 0.5)

    return round(max(0.0, min(100.0, score)), 1)


def validation_status(report: DataQualityReport, score: Optional[float] = None) -> str:
    """
    Gate status for analytical readiness.

    Returns:
        Pass | Warning | Fail
    """
    q = score if score is not None else compute_quality_score(report)
    critical = (
        any(v > 20 for v in report.missing_pct_by_column.values())
        or bool(report.dtype_issues)
        or report.range_violation_total > max(5, int(0.05 * report.n_rows))
    )
    if critical or q < 70:
        return "Fail"
    if report.duplicate_rows > 0 or report.overall_missing_pct > 1.0 or q < 90:
        return "Warning"
    return "Pass"


def build_quality_monitoring_summary(
    report: DataQualityReport,
    stage: str = "analytical_store",
) -> dict[str, Any]:
    """Flatten a quality report into dashboard monitoring KPIs."""
    score = compute_quality_score(report)
    status = validation_status(report, score)
    return {
        "stage": stage,
        "source": report.source,
        "created_at": report.created_at,
        "total_records": report.n_rows,
        "n_columns": report.n_columns,
        "missing_value_pct": report.overall_missing_pct,
        "duplicate_records": report.duplicate_rows,
        "validation_status": status,
        "data_quality_score": score,
        "dtype_issues": report.dtype_issues,
        "range_violations": report.range_violations,
        "outlier_counts": report.outlier_counts,
        "missing_by_column": report.missing_by_column,
        "missing_pct_by_column": report.missing_pct_by_column,
        "is_healthy": report.is_healthy,
    }


def detect_missing(df: pd.DataFrame) -> tuple[dict[str, int], dict[str, float]]:
    """Detect missing values by column."""
    counts = df.isna().sum().astype(int).to_dict()
    pct = {k: round(v / max(len(df), 1) * 100, 2) for k, v in counts.items()}
    return counts, pct


def detect_duplicates(df: pd.DataFrame, subset: Optional[list[str]] = None) -> int:
    """Count duplicate rows (optionally keyed by subset)."""
    keys = subset or ["year", "industry", "region"]
    present = [c for c in keys if c in df.columns]
    if not present:
        return int(df.duplicated().sum())
    return int(df.duplicated(subset=present).sum())


def validate_dtypes(df: pd.DataFrame) -> list[str]:
    """Validate column data types against expected schema."""
    issues: list[str] = []
    for col, expected in EXPECTED_DTYPES.items():
        if col not in df.columns:
            issues.append(f"Missing column: {col}")
            continue
        series = df[col]
        if expected == "int":
            coerced = pd.to_numeric(series, errors="coerce")
            if coerced.isna().sum() > series.isna().sum():
                issues.append(f"{col}: non-integer values detected")
        elif expected == "float":
            coerced = pd.to_numeric(series, errors="coerce")
            if coerced.isna().sum() > series.isna().sum():
                issues.append(f"{col}: non-numeric values detected")
        elif expected == "str":
            if not pd.api.types.is_object_dtype(series) and not pd.api.types.is_string_dtype(series):
                issues.append(f"{col}: expected string/object dtype, got {series.dtype}")
    return issues


def detect_range_violations(df: pd.DataFrame) -> dict[str, int]:
    """Count values outside expected business ranges."""
    violations: dict[str, int] = {}
    for col, (lo, hi) in NUMERIC_BOUNDS.items():
        if col not in df.columns:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        bad = ((numeric < lo) | (numeric > hi)) & numeric.notna()
        violations[col] = int(bad.sum())
    return violations


def detect_outliers_iqr(df: pd.DataFrame, columns: Optional[list[str]] = None) -> dict[str, int]:
    """Count IQR-based outliers for numeric columns."""
    cols = columns or [
        c for c in df.select_dtypes(include=[np.number]).columns if c != "year"
    ]
    counts: dict[str, int] = {}
    for col in cols:
        if col not in df.columns:
            continue
        s = df[col].dropna()
        if s.empty:
            counts[col] = 0
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            counts[col] = 0
            continue
        mask = (df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)
        counts[col] = int(mask.fillna(False).sum())
    return counts


def generate_quality_report(df: pd.DataFrame, source: str = "unknown") -> DataQualityReport:
    """Generate a full data quality report for a DataFrame."""
    missing_counts, missing_pct = detect_missing(df)
    return DataQualityReport(
        source=source,
        n_rows=len(df),
        n_columns=len(df.columns),
        missing_by_column=missing_counts,
        missing_pct_by_column=missing_pct,
        duplicate_rows=detect_duplicates(df),
        dtype_issues=validate_dtypes(df),
        range_violations=detect_range_violations(df),
        outlier_counts=detect_outliers_iqr(df),
    )


def persist_quality_report(report: DataQualityReport, db_path: Optional[Path] = None) -> None:
    """Store quality report JSON in SQLite."""
    path = db_path or get_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS data_quality_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                source_file TEXT,
                report_json TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "INSERT INTO data_quality_reports (created_at, source_file, report_json) VALUES (?, ?, ?)",
            (report.created_at, report.source, report.to_json()),
        )
        conn.commit()


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce columns to expected types."""
    out = df.copy()
    for col, expected in EXPECTED_DTYPES.items():
        if col not in out.columns:
            continue
        if expected in ("int", "float"):
            out[col] = pd.to_numeric(out[col], errors="coerce")
            if expected == "int":
                # Keep as float temporarily if NaNs present; finalize after imputation
                pass
        else:
            out[col] = out[col].astype(str).str.strip()
    return out


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing numeric values using industry-region medians,
    falling back to global median.
    """
    out = df.copy()
    group_cols = [c for c in ["industry", "region"] if c in out.columns]
    numeric_cols = [
        c
        for c in out.select_dtypes(include=[np.number]).columns
        if c != "year"
    ]

    for col in numeric_cols:
        if out[col].isna().sum() == 0:
            continue
        if group_cols:
            out[col] = out.groupby(group_cols, group_keys=False)[col].transform(
                lambda s: s.fillna(s.median())
            )
        out[col] = out[col].fillna(out[col].median())

    if "year" in out.columns:
        out["year"] = out["year"].fillna(out["year"].median()).astype(int)

    return out


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate year-industry-region keys, keeping first occurrence."""
    subset = [c for c in ["year", "industry", "region"] if c in df.columns]
    if subset:
        return df.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)
    return df.drop_duplicates(keep="first").reset_index(drop=True)


def clip_bounds(df: pd.DataFrame) -> pd.DataFrame:
    """Clip numeric indicators to business-valid ranges."""
    out = df.copy()
    for col, (lo, hi) in NUMERIC_BOUNDS.items():
        if col in out.columns and col != "year":
            out[col] = out[col].clip(lower=lo, upper=hi)
    return out


def clean_dataset(
    df: pd.DataFrame,
    source: str = "digital_industry_indicators.csv",
    persist_report: bool = True,
) -> tuple[pd.DataFrame, DataQualityReport]:
    """
    Full cleaning pipeline: report → coerce → dedupe → impute → clip.

    Returns:
        Clean DataFrame and the *pre-clean* quality report.
    """
    # Ensure required columns present
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Cannot clean dataset; missing columns: {missing_cols}")

    report = generate_quality_report(df, source=source)
    if persist_report:
        try:
            persist_quality_report(report)
        except Exception:  # noqa: BLE001
            # Non-fatal: cleaning should proceed even if DB write fails
            pass

    cleaned = coerce_types(df[REQUIRED_COLUMNS])
    cleaned = remove_duplicates(cleaned)
    cleaned = impute_missing(cleaned)
    cleaned = clip_bounds(cleaned)

    # Finalize year dtype
    cleaned["year"] = cleaned["year"].astype(int)

    return cleaned.sort_values(["year", "industry", "region"]).reset_index(drop=True), report
