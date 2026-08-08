"""
Data ingestion module.

Loads CSV datasets, persists panels to SQLite, and exposes a unified panel API.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import PROJECT_ROOT, get_db_path, get_raw_dir, load_settings

REQUIRED_COLUMNS = [
    "year",
    "industry",
    "region",
    "digital_index",
    "technology_investment",
    "revenue_growth",
    "employment",
    "rd_ratio",
]


class DataLoadError(Exception):
    """Raised when dataset loading or schema validation fails."""


def list_raw_files(raw_dir: Optional[Path] = None) -> list[Path]:
    """List CSV files available in the raw data directory."""
    directory = raw_dir or get_raw_dir()
    if not directory.exists():
        return []
    return sorted(directory.glob("*.csv"))


def load_csv(
    filepath: str | Path,
    required_columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Load a CSV file into a DataFrame with optional column validation.

    Args:
        filepath: Path to CSV.
        required_columns: If provided, ensure these columns exist.

    Returns:
        Loaded DataFrame.

    Raises:
        DataLoadError: File missing or schema mismatch.
    """
    path = Path(filepath)
    if not path.is_absolute():
        # Try raw data dir, then project root
        candidates = [get_raw_dir() / path, PROJECT_ROOT / path, path]
        path = next((c for c in candidates if c.exists()), path)

    if not path.exists():
        raise DataLoadError(f"CSV file not found: {filepath}")

    try:
        df = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        raise DataLoadError(f"Failed to read CSV {path}: {exc}") from exc

    if required_columns:
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            raise DataLoadError(f"Missing required columns in {path.name}: {missing}")

    return df


def load_primary_panel(settings: Optional[dict] = None) -> pd.DataFrame:
    """Load the primary digital industry indicators panel."""
    cfg = settings or load_settings()
    filename = cfg["data"]["primary_file"]
    return load_csv(get_raw_dir(cfg) / filename, required_columns=REQUIRED_COLUMNS)


def load_all_raw(settings: Optional[dict] = None) -> dict[str, pd.DataFrame]:
    """Load all CSV files from the raw data directory."""
    cfg = settings or load_settings()
    datasets: dict[str, pd.DataFrame] = {}
    for path in list_raw_files(get_raw_dir(cfg)):
        try:
            datasets[path.stem] = pd.read_csv(path)
        except Exception as exc:  # noqa: BLE001
            raise DataLoadError(f"Failed to load {path}: {exc}") from exc
    return datasets


def init_database(db_path: Optional[Path] = None) -> Path:
    """Create SQLite database and core tables if they do not exist."""
    path = db_path or get_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS industry_indicators (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                year INTEGER NOT NULL,
                industry TEXT NOT NULL,
                region TEXT NOT NULL,
                digital_index REAL,
                technology_investment REAL,
                revenue_growth REAL,
                employment REAL,
                rd_ratio REAL,
                digital_development_index REAL,
                employment_growth REAL,
                UNIQUE(year, industry, region)
            )
            """
        )
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
        conn.commit()
    return path


def save_to_sqlite(
    df: pd.DataFrame,
    table: str = "industry_indicators",
    db_path: Optional[Path] = None,
    if_exists: str = "replace",
) -> int:
    """
    Persist a DataFrame to SQLite.

    Returns:
        Number of rows written.
    """
    path = init_database(db_path)
    with sqlite3.connect(path) as conn:
        df.to_sql(table, conn, if_exists=if_exists, index=False)
    return len(df)


def load_from_sqlite(
    table: str = "industry_indicators",
    db_path: Optional[Path] = None,
    query: Optional[str] = None,
) -> pd.DataFrame:
    """Load data from SQLite table or custom SQL query."""
    path = db_path or get_db_path()
    if not path.exists():
        raise DataLoadError(
            f"Database not found at {path}. Run the data pipeline first."
        )
    with sqlite3.connect(path) as conn:
        if query:
            return pd.read_sql_query(query, conn)
        return pd.read_sql_query(f"SELECT * FROM {table}", conn)


def save_processed_csv(df: pd.DataFrame, filename: str = "industry_panel_clean.csv") -> Path:
    """Write processed DataFrame to data/processed/."""
    from src.config import get_processed_dir

    out_dir = get_processed_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    df.to_csv(out_path, index=False)
    return out_path
