"""
Pharmaceutical sales panel contract.

The business key is month + product + region. `validate_pharma_contract`
reports problems and does not impute months or rewrite sales values.
`load_pharma_panel` reads the single configured CSV. It does not scan
`data/raw` and it refuses legacy digital-industry and crude-oil filenames.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.config import LEGACY_RAW_FILENAMES, load_pharma_settings

REQUIRED_COLUMNS: list[str] = [
    "month",
    "product",
    "region",
    "sales_amount",
    "sales_volume",
    "data_source",
    "data_type",
]
UNIQUE_KEY: list[str] = ["month", "product", "region"]
ALLOWED_DATA_TYPES = frozenset({"synthetic_demo", "real"})
SYNTHETIC_DATA_TYPE = "synthetic_demo"
SYNTHETIC_DATA_SOURCE = "synthetic_demo"
DATASET_NAME = "pharma_sales_monthly"
PRIMARY_FILENAME = "pharma_sales_monthly_synthetic.csv"


def month_starts(start: str, end: str) -> pd.DatetimeIndex:
    """Inclusive month-start index. Both endpoints must already be month starts."""
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if start_ts.day != 1 or end_ts.day != 1:
        raise ValueError("date_start and date_end must be month-start dates (YYYY-MM-01)")
    months = pd.date_range(start_ts, end_ts, freq="MS")
    if len(months) == 0:
        raise ValueError("date range produced no months")
    return months


def simulate_sales_observation(
    *,
    base_volume: float,
    region_multiplier: float,
    t: int,
    month: int,
    monthly_trend: float,
    season_amplitude: float,
    peak_month: int,
    unit_price: float,
    price_drift_per_month: float,
    noise: float,
    volume_decimals: int = 2,
    amount_decimals: int = 2,
) -> tuple[float, float]:
    """
    One synthetic sales observation.

    Volume is floored at zero before rounding. This function does not describe
    a real product's demand. See data/metadata/PHARMA_DATA_NOTES.md.
    """
    seasonal = 1.0 + float(season_amplitude) * np.cos(
        2.0 * np.pi * (int(month) - int(peak_month)) / 12.0
    )
    level = (
        float(base_volume)
        * float(region_multiplier)
        * (1.0 + float(monthly_trend) * int(t))
        * seasonal
    )
    volume = round(max(0.0, level * (1.0 + float(noise))), int(volume_decimals))
    price = float(unit_price) * (1.0 + float(price_drift_per_month) * int(t))
    amount = round(volume * price, int(amount_decimals))
    return float(volume), float(amount)


def validate_pharma_contract(df: pd.DataFrame) -> list[str]:
    """
    Check the sales-panel contract.

    Returns issue strings. An empty list means the frame satisfies the contract.
    The input frame is not modified. Missing months are not inserted here;
    a gap between observed months is left for the quality checks in a later phase.
    """
    issues: list[str] = []
    if df is None or len(df) == 0:
        return ["panel has no rows"]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        issues.append(f"missing columns: {missing}")
        return issues
    extra = [c for c in df.columns if c not in REQUIRED_COLUMNS]
    if extra:
        issues.append(f"unexpected columns: {extra}")

    work = df[REQUIRED_COLUMNS].copy()
    nulls = {c: int(work[c].isna().sum()) for c in REQUIRED_COLUMNS if work[c].isna().any()}
    if nulls:
        issues.append(f"nulls in required columns: {nulls}")

    months = pd.to_datetime(work["month"], errors="coerce")
    unparseable = int(months.isna().sum())
    if unparseable:
        issues.append(f"unparseable month values: {unparseable}")
    parsed = months.dropna()
    if len(parsed):
        not_start = int((~parsed.dt.is_month_start).sum())
        if not_start:
            issues.append(f"month values must be month-start dates: {not_start}")

    key = pd.DataFrame(
        {
            "month": months.dt.strftime("%Y-%m-%d"),
            "product": work["product"].astype(str).str.strip(),
            "region": work["region"].astype(str).str.strip(),
        }
    )
    duplicate_rows = int(key.duplicated(UNIQUE_KEY).sum())
    if duplicate_rows:
        issues.append(
            "duplicate business keys (month, product, region): " + str(duplicate_rows)
        )

    for column in ("sales_amount", "sales_volume"):
        numeric = pd.to_numeric(work[column], errors="coerce")
        original_missing = work[column].isna()
        non_numeric = int((numeric.isna() & ~original_missing).sum())
        if non_numeric:
            issues.append(f"non-numeric {column}: {non_numeric}")
        negative = int((numeric < 0).fillna(False).sum())
        if negative:
            issues.append(f"negative {column}: {negative}")

    data_type = work["data_type"]
    observed_types = {
        str(v).strip()
        for v in data_type.dropna().unique().tolist()
    }
    invalid_types = sorted(observed_types - ALLOWED_DATA_TYPES)
    if invalid_types:
        issues.append(f"data_type outside {{synthetic_demo, real}}: {invalid_types}")

    source = work["data_source"]
    blank_source = int(
        (
            source.isna()
            | (~source.isna() & source.astype(str).str.strip().eq(""))
        ).sum()
    )
    if blank_source:
        issues.append(f"blank data_source: {blank_source}")

    return issues


def assert_pharma_contract(df: pd.DataFrame) -> None:
    """Raise ValueError when the frame breaks the contract."""
    issues = validate_pharma_contract(df)
    if issues:
        raise ValueError("Pharma panel contract failed: " + "; ".join(issues))


def load_pharma_panel(
    settings: Optional[dict[str, Any]] = None,
    path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Load one configured pharma CSV.

    Does not glob `data/raw`. Refuses legacy digital-industry and crude filenames
    even if a caller passes them explicitly.
    """
    cfg = settings or load_pharma_settings()
    csv_path = Path(path) if path is not None else (
        Path(cfg["paths"]["raw_data"]) / str(cfg["data"]["primary_file"])
    )
    if csv_path.name in LEGACY_RAW_FILENAMES:
        raise ValueError(
            "Refusing to load a legacy digital-industry or crude file "
            f"into the pharma panel: {csv_path.name}"
        )
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Pharma panel not found: {csv_path}. "
            "Run: python scripts/generate_pharma_data.py"
        )
    return pd.read_csv(csv_path)
