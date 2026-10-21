"""Pipeline writes only the pharmaceutical analytical store."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

from scripts.run_pharma_pipeline import run_pharma_pipeline
from src.config import load_pharma_settings


def _panel() -> pd.DataFrame:
    months = pd.date_range("2021-01-01", periods=24, freq="MS")
    rows = []
    for month in months:
        rows.append(
            {
                "month": month.strftime("%Y-%m-%d"),
                "product": "PX-A",
                "region": "East",
                "sales_amount": 100.0 + month.month,
                "sales_volume": 10.0,
                "data_source": "synthetic_demo",
                "data_type": "synthetic_demo",
            }
        )
    return pd.DataFrame(rows)


def test_pipeline_persists_pharma_outputs_without_reading_sibling_csv(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "digital_industry_indicators.csv").write_text("year,industry\n2015,Software\n", encoding="utf-8")
    (raw_dir / "crude_oil_weekly_synthetic.csv").write_text("date,commodity_id\n2020-01-03,WTI\n", encoding="utf-8")

    settings = load_pharma_settings()
    settings["paths"]["raw_data"] = str(raw_dir)
    settings["paths"]["processed_data"] = str(tmp_path / "processed")
    settings["paths"]["database"] = str(tmp_path / "pharma_analytics.db")
    settings["paths"]["reports"] = str(tmp_path / "reports")
    settings["evaluation"]["min_train_months"] = 12

    summary = run_pharma_pipeline(settings=settings, panel=_panel())
    assert summary["rows_analytical"] == 24
    assert summary["data_type"] == ["synthetic_demo"]
    assert summary["forecast_status"] == "ok"

    stored = pd.read_csv(summary["csv_path"])
    assert "digital_index" not in stored.columns
    assert "price_usd_bbl" not in stored.columns
    assert "mom_sales_amount_pct" in stored.columns
    quality = json.loads(Path(summary["quality_path"]).read_text(encoding="utf-8"))
    assert quality["policy"]["missing_months"] == "report_only"
    assert quality["analytical"]["missing_month_count"] == 0

    with sqlite3.connect(summary["database_path"]) as connection:
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        count = connection.execute("SELECT COUNT(*) FROM pharma_sales_monthly").fetchone()[0]
    assert "pharma_sales_monthly" in tables
    assert "commodity_market_series" not in tables
    assert count == 24
    forecast = json.loads(Path(summary["forecast_path"]).read_text(encoding="utf-8"))
    assert forecast["n_test"] == 6
    assert forecast["train_end"] < forecast["test_start"]
