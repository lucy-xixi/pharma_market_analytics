"""
Generate the synthetic pharmaceutical sales panel.

The output is data_source=synthetic_demo and data_type=synthetic_demo.
It is not real sell-in, sell-out, hospital, or patient data.

Rules and limitations: data/metadata/PHARMA_DATA_NOTES.md

Run from the project root:
    python scripts/generate_pharma_data.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_pharma_settings  # noqa: E402
from src.pharma_schema import (  # noqa: E402
    PRIMARY_FILENAME,
    REQUIRED_COLUMNS,
    SYNTHETIC_DATA_SOURCE,
    SYNTHETIC_DATA_TYPE,
    assert_pharma_contract,
    month_starts,
    simulate_sales_observation,
)


def _product_rows(settings: dict[str, Any]) -> list[dict[str, Any]]:
    products = settings.get("generator", {}).get("products")
    if not isinstance(products, list) or not products:
        raise ValueError("pharma generator.products must be a non-empty list")
    return products


def _region_rows(settings: dict[str, Any]) -> list[dict[str, Any]]:
    regions = settings.get("generator", {}).get("regions")
    if not isinstance(regions, list) or not regions:
        raise ValueError("pharma generator.regions must be a non-empty list")
    return regions


def generate_pharma_sales_panel(
    settings: Optional[dict[str, Any]] = None,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Build a complete month × product × region panel.

    Random draws use numpy Generator(default_rng(seed)) in this order:
    product list, region list, then month ascending. One normal draw per row.
    The returned frame is sorted by month, product, region after those draws.
    """
    cfg = settings or load_pharma_settings()
    data_cfg = cfg["data"]
    gen = cfg["generator"]
    used_seed = int(data_cfg["seed"] if seed is None else seed)
    months = month_starts(str(data_cfg["date_start"]), str(data_cfg["date_end"]))
    products = _product_rows(cfg)
    regions = _region_rows(cfg)
    noise_scale = float(gen["noise_scale"])
    volume_decimals = int(gen["volume_decimals"])
    amount_decimals = int(gen["amount_decimals"])
    data_source = str(data_cfg["synthetic_data_source"])
    data_type = str(data_cfg["synthetic_data_type"])
    if data_source != SYNTHETIC_DATA_SOURCE or data_type != SYNTHETIC_DATA_TYPE:
        raise ValueError(
            "The demo generator can only emit data_source=synthetic_demo "
            "and data_type=synthetic_demo"
        )

    for product in products:
        peak = int(product["peak_month"])
        if peak < 1 or peak > 12:
            raise ValueError(f"peak_month must be 1..12 for {product.get('id')}")
        if float(product["base_volume"]) <= 0 or float(product["unit_price"]) <= 0:
            raise ValueError(f"base_volume and unit_price must be positive for {product.get('id')}")
    for region in regions:
        if float(region["volume_multiplier"]) <= 0:
            raise ValueError(f"volume_multiplier must be positive for {region.get('id')}")

    rng = np.random.default_rng(used_seed)
    rows: list[dict[str, Any]] = []
    for product in products:
        for region in regions:
            for t, ts in enumerate(months):
                noise = float(rng.normal(0.0, noise_scale))
                volume, amount = simulate_sales_observation(
                    base_volume=float(product["base_volume"]),
                    region_multiplier=float(region["volume_multiplier"]),
                    t=t,
                    month=int(ts.month),
                    monthly_trend=float(product["monthly_trend"]),
                    season_amplitude=float(product["season_amplitude"]),
                    peak_month=int(product["peak_month"]),
                    unit_price=float(product["unit_price"]),
                    price_drift_per_month=float(product["price_drift_per_month"]),
                    noise=noise,
                    volume_decimals=volume_decimals,
                    amount_decimals=amount_decimals,
                )
                rows.append(
                    {
                        "month": ts.strftime("%Y-%m-%d"),
                        "product": str(product["id"]),
                        "region": str(region["id"]),
                        "sales_amount": amount,
                        "sales_volume": volume,
                        "data_source": data_source,
                        "data_type": data_type,
                    }
                )

    panel = pd.DataFrame(rows, columns=REQUIRED_COLUMNS)
    panel = panel.sort_values(["month", "product", "region"]).reset_index(drop=True)
    assert_pharma_contract(panel)
    return panel


def write_pharma_sales_csv(
    df: pd.DataFrame,
    directory: Path,
    filename: str = PRIMARY_FILENAME,
) -> Path:
    """
    Write one pharma CSV. Does not delete or rewrite sibling files.
    """
    if filename != PRIMARY_FILENAME:
        raise ValueError(
            f"Pharma demo output must be named {PRIMARY_FILENAME}, got {filename}"
        )
    assert_pharma_contract(df)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    df.to_csv(path, index=False)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic_demo pharmaceutical sales "
            "(not real market data)"
        )
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the seed in config/pharma_settings.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for the CSV (default: pharma raw_data path)",
    )
    args = parser.parse_args()

    settings = load_pharma_settings()
    panel = generate_pharma_sales_panel(settings=settings, seed=args.seed)
    if args.output_dir is not None:
        out_dir = args.output_dir
    else:
        out_dir = Path(settings["paths"]["raw_data"])
    path = write_pharma_sales_csv(panel, out_dir)
    print(
        f"Wrote {path} ({len(panel)} rows, "
        f"data_type={SYNTHETIC_DATA_TYPE}). "
        "Synthetic demo only — not real pharmaceutical market data."
    )


if __name__ == "__main__":
    main()
