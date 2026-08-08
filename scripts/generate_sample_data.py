"""
Generate realistic synthetic digital industry indicator datasets.

Produces CSV files under data/raw/ for portfolio demonstration.
Run from project root:
    python scripts/generate_sample_data.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

INDUSTRIES = [
    "Software & IT Services",
    "Electronics Manufacturing",
    "Telecommunications",
    "FinTech",
    "E-Commerce",
    "Industrial Automation",
    "Cloud & Data Centers",
    "Digital Media",
]

REGIONS = [
    "North America",
    "Europe",
    "East Asia",
    "Southeast Asia",
    "Latin America",
]

# Baseline profiles (mean digital maturity by industry)
INDUSTRY_BASE = {
    "Software & IT Services": {"digital": 78, "tech": 42, "rd": 12.5, "emp": 85, "rev": 9.5},
    "Electronics Manufacturing": {"digital": 62, "tech": 35, "rd": 8.0, "emp": 120, "rev": 5.5},
    "Telecommunications": {"digital": 70, "tech": 38, "rd": 6.5, "emp": 95, "rev": 4.0},
    "FinTech": {"digital": 82, "tech": 48, "rd": 11.0, "emp": 55, "rev": 14.0},
    "E-Commerce": {"digital": 75, "tech": 40, "rd": 5.5, "emp": 70, "rev": 11.0},
    "Industrial Automation": {"digital": 58, "tech": 32, "rd": 9.0, "emp": 65, "rev": 6.5},
    "Cloud & Data Centers": {"digital": 88, "tech": 55, "rd": 14.0, "emp": 48, "rev": 16.0},
    "Digital Media": {"digital": 72, "tech": 36, "rd": 7.0, "emp": 52, "rev": 8.0},
}

REGION_MULT = {
    "North America": 1.15,
    "Europe": 1.05,
    "East Asia": 1.12,
    "Southeast Asia": 0.88,
    "Latin America": 0.78,
}


def generate_panel(start_year: int = 2015, end_year: int = 2024, seed: int = 42) -> pd.DataFrame:
    """Generate a multi-year industry × region panel dataset."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    for year in range(start_year, end_year + 1):
        t = year - start_year
        for industry in INDUSTRIES:
            base = INDUSTRY_BASE[industry]
            for region in REGIONS:
                mult = REGION_MULT[region]
                # Secular growth + noise
                digital = (
                    base["digital"] * mult
                    + t * rng.uniform(1.2, 2.8)
                    + rng.normal(0, 2.5)
                )
                tech = (
                    base["tech"] * mult
                    + t * rng.uniform(1.5, 3.5)
                    + rng.normal(0, 2.0)
                )
                rd = max(
                    1.0,
                    base["rd"] * (0.95 + 0.05 * mult)
                    + t * rng.uniform(0.05, 0.25)
                    + rng.normal(0, 0.4),
                )
                employment = max(
                    10.0,
                    base["emp"] * mult
                    + t * rng.uniform(1.0, 4.0)
                    + rng.normal(0, 3.0),
                )
                # Revenue growth linked to tech investment & digital index (with noise)
                rev = (
                    base["rev"]
                    + 0.08 * (tech - 30)
                    + 0.04 * (digital - 60)
                    + 0.15 * (rd - 8)
                    + t * 0.15
                    + rng.normal(0, 1.8)
                )

                rows.append(
                    {
                        "year": year,
                        "industry": industry,
                        "region": region,
                        "digital_index": round(float(np.clip(digital, 20, 100)), 2),
                        "technology_investment": round(float(max(tech, 5)), 2),  # USD billions
                        "revenue_growth": round(float(rev), 2),  # percent
                        "employment": round(float(employment), 1),  # thousands
                        "rd_ratio": round(float(rd), 2),  # percent of revenue
                    }
                )

    df = pd.DataFrame(rows)

    # Inject controlled data-quality issues for cleaning demo (~3%)
    n = len(df)
    miss_idx = rng.choice(n, size=max(1, n // 40), replace=False)
    for i, col in zip(miss_idx, rng.choice(["digital_index", "rd_ratio", "employment"], size=len(miss_idx))):
        df.loc[i, col] = np.nan

    # Duplicate a few rows
    dup_idx = rng.choice(n, size=5, replace=False)
    df = pd.concat([df, df.iloc[dup_idx]], ignore_index=True)

    # Shuffle
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def generate_supplementary(seed: int = 42) -> dict[str, pd.DataFrame]:
    """Generate smaller thematic datasets for multi-source ingestion demos."""
    rng = np.random.default_rng(seed + 1)
    years = list(range(2015, 2025))

    tech_invest = pd.DataFrame(
        {
            "year": years,
            "ai_investment_bn": [8, 12, 18, 28, 42, 55, 72, 95, 118, 145]
            + rng.normal(0, 2, 10),
            "cloud_investment_bn": [45, 55, 70, 90, 115, 140, 170, 205, 245, 290]
            + rng.normal(0, 5, 10),
            "iot_investment_bn": [20, 28, 35, 48, 60, 75, 88, 102, 120, 138]
            + rng.normal(0, 3, 10),
        }
    )
    for c in tech_invest.columns:
        if c != "year":
            tech_invest[c] = tech_invest[c].round(2)

    employment_macro = pd.DataFrame(
        {
            "year": years,
            "digital_jobs_million": np.round(28 + np.arange(10) * 2.4 + rng.normal(0, 0.5, 10), 2),
            "skills_gap_index": np.round(55 - np.arange(10) * 1.8 + rng.normal(0, 1.5, 10), 2),
            "remote_work_share": np.round(
                np.clip(12 + np.arange(10) * 4.5 + rng.normal(0, 2, 10), 5, 70), 2
            ),
        }
    )

    return {
        "technology_investment_trends.csv": tech_invest,
        "digital_employment_macro.csv": employment_macro,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate sample digital industry datasets")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    panel = generate_panel(seed=args.seed)
    panel_path = RAW_DIR / "digital_industry_indicators.csv"
    panel.to_csv(panel_path, index=False)
    print(f"Wrote {panel_path} ({len(panel)} rows)")

    for name, frame in generate_supplementary(seed=args.seed).items():
        path = RAW_DIR / name
        frame.to_csv(path, index=False)
        print(f"Wrote {path} ({len(frame)} rows)")


if __name__ == "__main__":
    main()
