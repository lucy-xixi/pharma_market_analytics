"""
Generate SYNTHETIC crude-oil market demo data for portfolio / pipeline testing.

IMPORTANT:
  Output is explicitly labeled data_source=synthetic_demo.
  It is NOT real market data and must not be described as live EIA/ICE/NYMEX feeds.

Run from project root:
    python scripts/generate_commodity_data.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# Demo contracts — identifiers only; prices/inventories are simulated
COMMODITIES = {
    "WTI_CRUDE": {"base_price": 65.0, "base_inv": 450.0, "vol": 2.2},
    "BRENT_CRUDE": {"base_price": 68.0, "base_inv": 380.0, "vol": 2.0},
}


def generate_crude_weekly_panel(
    start: str = "2020-01-03",
    end: str = "2024-12-27",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Build a weekly synthetic crude panel.

    Columns:
      date, commodity_id, price_usd_bbl, inventory_mmbbl,
      frequency, data_source, data_type
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, end=end, freq="W-FRI")
    rows: list[dict] = []

    for commodity_id, profile in COMMODITIES.items():
        price = float(profile["base_price"])
        inv = float(profile["base_inv"])
        for i, dt in enumerate(dates):
            # Secular drift + noise
            price = price + rng.normal(0.05, profile["vol"])
            inv = inv + rng.normal(-0.15, 1.8)

            # Scenario shocks (still synthetic): early-2020 stress, mid-2022 spike
            if dt >= pd.Timestamp("2020-03-01") and dt <= pd.Timestamp("2020-04-30"):
                price = price - abs(rng.normal(4.0, 1.5))
            if dt >= pd.Timestamp("2022-02-01") and dt <= pd.Timestamp("2022-06-30"):
                price = price + abs(rng.normal(3.0, 1.2))

            # WTI-style negative print is allowed as a plausible extreme, not an error
            if commodity_id == "WTI_CRUDE" and dt == pd.Timestamp("2020-04-17"):
                price = -5.0 + rng.normal(0, 1.0)

            rows.append(
                {
                    "date": dt.strftime("%Y-%m-%d"),
                    "commodity_id": commodity_id,
                    "price_usd_bbl": round(float(price), 2),
                    "inventory_mmbbl": round(float(max(inv, 50.0)), 2),
                    "frequency": "weekly",
                    "data_source": "synthetic_demo",
                    "data_type": "synthetic",
                }
            )

    df = pd.DataFrame(rows)

    # Controlled quality issues for cleaning demos (~2% missing, few duplicates)
    n = len(df)
    miss_idx = rng.choice(n, size=max(2, n // 50), replace=False)
    for i in miss_idx:
        col = rng.choice(["price_usd_bbl", "inventory_mmbbl"])
        df.loc[i, col] = np.nan

    dup_idx = rng.choice(n, size=3, replace=False)
    df = pd.concat([df, df.iloc[dup_idx]], ignore_index=True)

    # Drop one week for one commodity to create a continuity gap (data quality signal)
    gap_mask = (df["commodity_id"] == "BRENT_CRUDE") & (
        df["date"] == "2021-06-18"
    )
    df = df.loc[~gap_mask].reset_index(drop=True)

    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SYNTHETIC crude oil demo CSVs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    panel = generate_crude_weekly_panel(seed=args.seed)
    path = RAW_DIR / "crude_oil_weekly_synthetic.csv"
    panel.to_csv(path, index=False)
    print(f"Wrote SYNTHETIC demo file {path} ({len(panel)} rows)")
    print("data_source=synthetic_demo — not real market data.")


if __name__ == "__main__":
    main()
