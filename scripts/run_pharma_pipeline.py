"""Run the pharmaceutical sales pipeline only.

Reads the configured pharma CSV. Does not scan data/raw and does not load
digital-industry or crude-oil files.

    python scripts/run_pharma_pipeline.py
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_pharma_settings  # noqa: E402
from src.pharma_features import add_growth_features  # noqa: E402
from src.pharma_forecast import evaluate_panel_forecast  # noqa: E402
from src.pharma_quality import prepare_sales_panel  # noqa: E402
from src.pharma_schema import load_pharma_panel  # noqa: E402


def run_pharma_pipeline(
    settings: Optional[dict[str, Any]] = None,
    panel: Optional[pd.DataFrame] = None,
) -> dict[str, Any]:
    """Clean, feature, persist, and evaluate the pharma panel."""
    cfg = settings or load_pharma_settings()
    raw = panel.copy() if panel is not None else load_pharma_panel(cfg)
    analytical, quality = prepare_sales_panel(raw)
    featured = add_growth_features(analytical)

    processed_dir = Path(cfg["paths"]["processed_data"])
    reports_dir = Path(cfg["paths"]["reports"])
    db_path = Path(cfg["paths"]["database"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    csv_path = processed_dir / "pharma_sales_monthly_clean.csv"
    featured.to_csv(csv_path, index=False)
    quality_path = processed_dir / "pharma_data_quality_report.json"
    quality_path.write_text(json.dumps(quality, indent=2, ensure_ascii=False), encoding="utf-8")

    with sqlite3.connect(db_path) as connection:
        featured.to_sql("pharma_sales_monthly", connection, if_exists="replace", index=False)
        connection.execute("DROP INDEX IF EXISTS ux_pharma_month_product_region")
        connection.execute(
            "CREATE UNIQUE INDEX ux_pharma_month_product_region "
            "ON pharma_sales_monthly(month, product, region)"
        )
        connection.commit()

    evaluation_cfg = cfg.get("evaluation", {})
    forecast = evaluate_panel_forecast(
        featured,
        test_months=int(evaluation_cfg.get("test_months", 6)),
        horizon_months=int(evaluation_cfg.get("horizon_months", 6)),
        min_train_months=int(evaluation_cfg.get("min_train_months", 18)),
        season_length=int(evaluation_cfg.get("season_length", 12)),
        value_col=str(evaluation_cfg.get("target_column", "sales_amount")),
    )
    forecast["series_definition"] = (
        "Total grain: monthly sum of sales_amount across the analytical panel after exclusions. "
        "Product, region, and product-region forecasts are separate series and are not fitted together."
    )
    forecast["data_type"] = sorted(featured["data_type"].dropna().astype(str).unique().tolist())
    forecast_path = reports_dir / "forecast_total.json"
    forecast_path.write_text(json.dumps(forecast, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "rows_raw": int(len(raw)),
        "rows_analytical": int(len(featured)),
        "csv_path": str(csv_path),
        "quality_path": str(quality_path),
        "database_path": str(db_path),
        "forecast_path": str(forecast_path),
        "forecast_status": forecast["status"],
        "selected_forward_model": forecast.get("selected_forward_model"),
        "data_type": forecast["data_type"],
    }


def main() -> None:
    summary = run_pharma_pipeline()
    print(
        "Pharma pipeline wrote "
        f"{summary['rows_analytical']} analytical rows from {summary['rows_raw']} raw rows."
    )
    print(f"CSV: {summary['csv_path']}")
    print(f"Quality: {summary['quality_path']}")
    print(f"SQLite: {summary['database_path']}")
    print(
        f"Forecast status: {summary['forecast_status']}; "
        f"selected model: {summary['selected_forward_model']}"
    )
    print(f"data_type: {summary['data_type']}")
    print("Synthetic demo output is not a real pharmaceutical market finding.")


if __name__ == "__main__":
    main()
