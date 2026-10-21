"""
Field dictionary for the pharmaceutical sales panel.

The canonical file is data/metadata/pharma_data_dictionary.csv.
data/metadata/data_dictionary.csv is the same pharmaceutical dictionary.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import PROJECT_ROOT

DICTIONARY_PATH = PROJECT_ROOT / "data" / "metadata" / "data_dictionary.csv"
METADATA_DOC_PATH = PROJECT_ROOT / "data" / "metadata" / "DATA_DICTIONARY.md"
PHARMA_DICTIONARY_PATH = PROJECT_ROOT / "data" / "metadata" / "pharma_data_dictionary.csv"
PHARMA_DATA_NOTES_PATH = PROJECT_ROOT / "data" / "metadata" / "PHARMA_DATA_NOTES.md"

REQUIRED_DICTIONARY_COLUMNS = [
    "dataset_name",
    "column_name",
    "description",
    "data_type",
    "business_meaning",
]


class DataDictionaryError(Exception):
    """Raised when the data dictionary cannot be loaded or validated."""


def get_dictionary_path() -> Path:
    """Return the canonical CSV path for the data dictionary."""
    return DICTIONARY_PATH


def load_data_dictionary(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the enterprise data dictionary.

    Returns:
        DataFrame with dataset/column metadata.

    Raises:
        DataDictionaryError: Missing file or required columns.
    """
    csv_path = path or DICTIONARY_PATH
    if not csv_path.exists():
        raise DataDictionaryError(f"Data dictionary not found: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:  # noqa: BLE001
        raise DataDictionaryError(f"Failed to read data dictionary: {exc}") from exc

    missing = [c for c in REQUIRED_DICTIONARY_COLUMNS if c not in df.columns]
    if missing:
        raise DataDictionaryError(f"Data dictionary missing columns: {missing}")

    # Normalize whitespace
    for col in REQUIRED_DICTIONARY_COLUMNS:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()

    return df.sort_values(["dataset_name", "column_name"]).reset_index(drop=True)


def load_pharma_data_dictionary(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the pharmaceutical sales dictionary, not the legacy catalog."""
    return load_data_dictionary(path or PHARMA_DICTIONARY_PATH)


def list_datasets(dictionary: Optional[pd.DataFrame] = None) -> list[str]:
    """Return distinct dataset names in the catalog."""
    dd = dictionary if dictionary is not None else load_data_dictionary()
    return sorted(dd["dataset_name"].dropna().unique().tolist())


def get_dataset_dictionary(
    dataset_name: str,
    dictionary: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Return metadata rows for a single dataset."""
    dd = dictionary if dictionary is not None else load_data_dictionary()
    subset = dd[dd["dataset_name"] == dataset_name].copy()
    if subset.empty:
        known = ", ".join(list_datasets(dd))
        raise DataDictionaryError(
            f"Unknown dataset '{dataset_name}'. Known datasets: {known}"
        )
    return subset.reset_index(drop=True)


def get_column_metadata(
    dataset_name: str,
    column_name: str,
    dictionary: Optional[pd.DataFrame] = None,
) -> dict:
    """Return a single column's metadata as a dictionary."""
    subset = get_dataset_dictionary(dataset_name, dictionary)
    rows = subset[subset["column_name"] == column_name]
    if rows.empty:
        raise DataDictionaryError(
            f"Column '{column_name}' not found in dataset '{dataset_name}'"
        )
    return rows.iloc[0].to_dict()


def dataset_summary_table(dictionary: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Compact catalog summary: dataset, column count, stages."""
    dd = dictionary if dictionary is not None else load_data_dictionary()
    stage_col = "source_stage" if "source_stage" in dd.columns else None
    rows = []
    for name, group in dd.groupby("dataset_name"):
        rows.append(
            {
                "dataset_name": name,
                "n_columns": len(group),
                "source_stage": (
                    ", ".join(sorted(group[stage_col].dropna().astype(str).unique()))
                    if stage_col
                    else "n/a"
                ),
                "columns": ", ".join(group["column_name"].tolist()),
            }
        )
    return pd.DataFrame(rows).sort_values("dataset_name").reset_index(drop=True)


def to_markdown_table(
    dataset_name: str,
    dictionary: Optional[pd.DataFrame] = None,
    columns: Optional[list[str]] = None,
) -> str:
    """
    Render a dataset's dictionary as a Markdown table for README / docs.
    """
    subset = get_dataset_dictionary(dataset_name, dictionary)
    cols = columns or [
        "column_name",
        "description",
        "data_type",
        "business_meaning",
    ]
    cols = [c for c in cols if c in subset.columns]
    view = subset[cols].copy()

    headers = [
        {
            "column_name": "Column",
            "description": "Description",
            "data_type": "Data type",
            "business_meaning": "Business meaning",
            "unit": "Unit",
            "nullable": "Nullable",
            "dataset_name": "Dataset",
        }.get(c, c.replace("_", " ").title())
        for c in cols
    ]

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in view.iterrows():
        cells = [str(row[c]).replace("|", "\\|") for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def validate_dataframe_against_dictionary(
    df: pd.DataFrame,
    dataset_name: str,
    dictionary: Optional[pd.DataFrame] = None,
    require_all: bool = True,
) -> list[str]:
    """
    Compare a DataFrame's columns to the data dictionary.

    Returns:
        List of validation issue strings (empty if aligned).
    """
    meta = get_dataset_dictionary(dataset_name, dictionary)
    expected = set(meta["column_name"].tolist())
    actual = set(df.columns.tolist())
    issues: list[str] = []

    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing and require_all:
        issues.append(f"Missing dictionary columns: {missing}")
    if extra:
        issues.append(f"Columns not in dictionary: {extra}")
    return issues
