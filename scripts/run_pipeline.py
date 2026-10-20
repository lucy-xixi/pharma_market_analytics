"""
End-to-end data pipeline:

1. Generate sample data (optional) — digital industry + synthetic crude
2. Load raw CSVs
3. Clean & validate
4. Enrich indicators / commodity features
5. Persist to SQLite + processed CSV
6. Run analytics snapshot → reports/ (digital industry panel)

Existing digital-industry flow is preserved. Commodity flow is additive.
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
from src.commodity_features import enrich_commodity_features, feature_summary  # noqa: E402
from src.config import get_processed_dir, get_raw_dir, load_settings  # noqa: E402
from src.data_cleaning import clean_commodity_dataset, clean_dataset  # noqa: E402
from src.data_loader import (  # noqa: E402
    init_database,
    load_commodity_panel,
    load_primary_panel,
    save_commodity_processed,
    save_processed_csv,
    save_to_sqlite,
)
from src.indicator_engine import enrich_indicators  # noqa: E402


def _generate_samples(settings: dict) -> None:
    from scripts.generate_commodity_data import generate_crude_weekly_panel
    from scripts.generate_sample_data import generate_panel, generate_supplementary

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

    crude = generate_crude_weekly_panel()
    crude_name = settings.get("commodity", {}).get(
        "primary_file", "crude_oil_weekly_synthetic.csv"
    )
    crude_path = raw_dir / crude_name
    crude.to_csv(crude_path, index=False)
    print(
        f"Generated SYNTHETIC {crude_path} ({len(crude)} rows) "
        "[data_source=synthetic_demo — not real market data]"
    )


def run_digital_industry(settings: dict) -> Path:
    """Original digital-industry pipeline branch (unchanged behaviour)."""
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
    print(f"  wrote {n} rows → industry_indicators")
    print(f"  wrote {out_csv}")

    print("Running analytics snapshot...")
    bundle = run_full_analytics(enriched)
    reports_dir = Path(settings["paths"]["reports"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = reports_dir / f"analytics_snapshot_{stamp}.json"
    report_path.write_text(json.dumps(bundle.to_dict(), indent=2), encoding="utf-8")

    quality_path = get_processed_dir() / "data_quality_report.json"
    quality_path.write_text(quality.to_json(), encoding="utf-8")
    print(f"  analytics → {report_path}")
    print(f"  quality → {quality_path}")
    return out_csv


def run_commodity(settings: dict) -> Path | None:
    """
    Phase-1 crude market branch: load → clean → features → SQLite/CSV.

    Does not claim forecasting or live market feeds are implemented.
    """
    commodity_cfg = settings.get("commodity", {})
    if not commodity_cfg.get("enabled", True):
        print("Commodity branch disabled in settings; skipping.")
        return None

    filename = commodity_cfg.get("primary_file", "crude_oil_weekly_synthetic.csv")
    raw_path = get_raw_dir(settings) / filename
    if not raw_path.exists():
        print(f"Commodity raw file missing ({raw_path}); skipping commodity branch.")
        return None

    print("Loading commodity panel (synthetic demo unless otherwise labeled)...")
    raw = load_commodity_panel(settings)
    print(f"  raw rows: {len(raw)}")
    if "data_type" in raw.columns:
        print(f"  data_type values: {sorted(raw['data_type'].dropna().unique().tolist())}")

    print("Cleaning & validating commodity panel...")
    freq = commodity_cfg.get("expected_freq", "W-FRI")
    clean, quality, extras = clean_commodity_dataset(
        raw,
        source=filename,
        expected_freq=freq,
    )
    print(f"  clean rows: {len(clean)}")
    print(
        f"  duplicate excess rows (pre-clean, key=date+commodity_id): "
        f"{quality.duplicate_rows}"
    )
    print(f"  time gap counts by commodity: {extras.time_gaps_by_commodity}")
    print(f"  time gap dates by commodity: {extras.time_gap_dates_by_commodity}")
    print(f"  extreme price flags (retained): {extras.extreme_price_flags}")

    print("Computing commodity features...")
    enriched = enrich_commodity_features(clean, periods=1)
    summary = feature_summary(enriched)
    print(f"  feature coverage:\n{summary.to_string(index=False)}")

    print("Persisting commodity panel...")
    init_database()
    processed_name = commodity_cfg.get(
        "processed_file", "crude_oil_weekly_clean.csv"
    )
    out_csv, n = save_commodity_processed(
        enriched,
        filename=processed_name,
        table=commodity_cfg.get("table", "commodity_market_series"),
    )
    print(f"  wrote {n} rows → commodity_market_series")
    print(f"  wrote {out_csv}")

    processed_dir = get_processed_dir()
    q_path = processed_dir / "commodity_data_quality_report.json"
    payload = {
        "base": quality.to_dict(),
        "commodity_extras": extras.to_dict(),
        "disclaimer": (
            "Rows with data_type=synthetic / data_source=synthetic_demo are "
            "demonstration data only, not live public market feeds."
        ),
    }
    q_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"  commodity quality → {q_path}")
    return out_csv


def run(generate: bool = False) -> Path:
    settings = load_settings()
    Path(settings["paths"]["reports"]).mkdir(parents=True, exist_ok=True)

    if generate:
        _generate_samples(settings)

    out_csv = run_digital_industry(settings)
    run_commodity(settings)
    print("Pipeline complete.")
    return out_csv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run digital industry + commodity analytics pipeline"
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate synthetic sample CSVs (industry + crude demo) before processing",
    )
    args = parser.parse_args()
    run(generate=args.generate)


if __name__ == "__main__":
    main()
