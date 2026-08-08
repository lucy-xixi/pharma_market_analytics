"""
End-to-end data pipeline:

1. Generate sample data (optional)
2. Load raw CSVs
3. Clean & validate
4. Enrich indicators
5. Persist to SQLite + processed CSV
6. Run analytics snapshot → reports/
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analytics import run_full_analytics  # noqa: E402
from src.config import get_processed_dir, load_settings  # noqa: E402
from src.data_cleaning import clean_dataset  # noqa: E402
from src.data_loader import (  # noqa: E402
    init_database,
    load_primary_panel,
    save_processed_csv,
    save_to_sqlite,
)
from src.indicator_engine import enrich_indicators  # noqa: E402


def run(generate: bool = False) -> Path:
    settings = load_settings()
    reports_dir = Path(settings["paths"]["reports"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    if generate:
        from scripts.generate_sample_data import generate_panel, generate_supplementary
        from src.config import get_raw_dir

        raw_dir = get_raw_dir(settings)
        raw_dir.mkdir(parents=True, exist_ok=True)
        panel = generate_panel()
        panel_path = raw_dir / "digital_industry_indicators.csv"
        panel.to_csv(panel_path, index=False)
        print(f"Generated {panel_path} ({len(panel)} rows)")
        for name, frame in generate_supplementary().items():
            path = raw_dir / name
            frame.to_csv(path, index=False)
            print(f"Generated {path} ({len(frame)} rows)")

    print("Loading primary panel...")
    raw = load_primary_panel(settings)
    print(f"  raw rows: {len(raw)}")

    print("Cleaning & validating...")
    clean, quality = clean_dataset(raw, source=settings["data"]["primary_file"])
    print(f"  clean rows: {len(clean)}")
    print(f"  duplicates removed (reported): {quality.duplicate_rows}")
    print(f"  missing (pre-clean): {quality.missing_by_column}")

    print("Computing indicators...")
    enriched = enrich_indicators(clean)

    print("Persisting to SQLite & processed CSV...")
    init_database()
    n = save_to_sqlite(enriched, table="industry_indicators", if_exists="replace")
    out_csv = save_processed_csv(enriched)
    print(f"  wrote {n} rows → database")
    print(f"  wrote {out_csv}")

    print("Running analytics snapshot...")
    bundle = run_full_analytics(enriched)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = reports_dir / f"analytics_snapshot_{stamp}.json"
    report_path.write_text(json.dumps(bundle.to_dict(), indent=2), encoding="utf-8")

    quality_path = get_processed_dir() / "data_quality_report.json"
    quality_path.write_text(quality.to_json(), encoding="utf-8")
    print(f"  analytics → {report_path}")
    print(f"  quality → {quality_path}")
    print("Pipeline complete.")
    return out_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Run digital industry analytics pipeline")
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate synthetic sample CSVs before processing",
    )
    args = parser.parse_args()
    run(generate=args.generate)


if __name__ == "__main__":
    main()
