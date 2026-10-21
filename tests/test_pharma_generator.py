"""Reproducible synthetic pharmaceutical sales generator."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.generate_pharma_data import generate_pharma_sales_panel, write_pharma_sales_csv
from src.config import PROJECT_ROOT, load_pharma_settings
from src.pharma_schema import (
    PRIMARY_FILENAME,
    REQUIRED_COLUMNS,
    load_pharma_panel,
    validate_pharma_contract,
)


def _spec_sales(
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
    volume_decimals: int,
    amount_decimals: int,
) -> tuple[float, float]:
    """Independent copy of the formula in PHARMA_DATA_NOTES.md."""
    seasonal = 1.0 + season_amplitude * np.cos(
        2.0 * np.pi * (month - peak_month) / 12.0
    )
    level = base_volume * region_multiplier * (1.0 + monthly_trend * t) * seasonal
    volume = round(max(0.0, level * (1.0 + noise)), volume_decimals)
    price = unit_price * (1.0 + price_drift_per_month * t)
    amount = round(volume * price, amount_decimals)
    return float(volume), float(amount)


def _expected_panel(settings: dict) -> pd.DataFrame:
    data_cfg = settings["data"]
    gen = settings["generator"]
    months = pd.date_range(data_cfg["date_start"], data_cfg["date_end"], freq="MS")
    rng = np.random.default_rng(int(data_cfg["seed"]))
    rows: list[dict] = []
    for product in gen["products"]:
        for region in gen["regions"]:
            for t, ts in enumerate(months):
                noise = float(rng.normal(0.0, float(gen["noise_scale"])))
                volume, amount = _spec_sales(
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
                    volume_decimals=int(gen["volume_decimals"]),
                    amount_decimals=int(gen["amount_decimals"]),
                )
                rows.append(
                    {
                        "month": ts.strftime("%Y-%m-%d"),
                        "product": product["id"],
                        "region": region["id"],
                        "sales_amount": amount,
                        "sales_volume": volume,
                        "data_source": data_cfg["synthetic_data_source"],
                        "data_type": data_cfg["synthetic_data_type"],
                    }
                )
    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS).sort_values(
        ["month", "product", "region"]
    ).reset_index(drop=True)


def test_generator_matches_documented_formula_and_grid() -> None:
    settings = load_pharma_settings()
    panel = generate_pharma_sales_panel(settings=settings)
    expected = _expected_panel(settings)
    pd.testing.assert_frame_equal(panel, expected)

    months = pd.date_range("2021-01-01", "2024-12-01", freq="MS")
    products = [item["id"] for item in settings["generator"]["products"]]
    regions = [item["id"] for item in settings["generator"]["regions"]]
    assert products == [
        "PX-CARD-01",
        "PX-ENDO-02",
        "PX-DIAB-03",
        "PX-RESP-04",
        "PX-ANTI-05",
    ]
    assert regions == ["East China", "North China", "South China", "Southwest China"]
    assert len(months) == 48
    assert len(panel) == len(months) * len(products) * len(regions)
    assert list(panel.columns) == REQUIRED_COLUMNS
    assert validate_pharma_contract(panel) == []
    assert set(panel["data_type"]) == {"synthetic_demo"}
    assert set(panel["data_source"]) == {"synthetic_demo"}
    assert panel["sales_amount"].min() >= 0
    assert panel["sales_volume"].min() >= 0
    assert int(panel.duplicated(["month", "product", "region"]).sum()) == 0

    for (_, _), group in panel.groupby(["product", "region"], sort=False):
        observed = pd.to_datetime(group["month"]).sort_values()
        assert list(observed) == list(months)


def test_same_seed_is_reproducible_and_other_seed_changes_volume() -> None:
    baseline = generate_pharma_sales_panel(seed=42)
    again = generate_pharma_sales_panel(seed=42)
    other = generate_pharma_sales_panel(seed=7)
    pd.testing.assert_frame_equal(baseline, again)
    assert not baseline["sales_volume"].equals(other["sales_volume"])
    assert set(other["data_type"]) == {"synthetic_demo"}


def test_generator_does_not_read_csv_files(monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.generate_pharma_data as generator

    def _forbid_read(*args: object, **kwargs: object) -> pd.DataFrame:
        raise AssertionError("pharma generator must not read a CSV")

    monkeypatch.setattr(generator.pd, "read_csv", _forbid_read)
    panel = generator.generate_pharma_sales_panel()
    assert len(panel) == 960


def test_writer_leaves_legacy_csv_files_untouched(tmp_path: Path) -> None:
    digital = tmp_path / "digital_industry_indicators.csv"
    crude = tmp_path / "crude_oil_weekly_synthetic.csv"
    digital.write_text("legacy-digital\n", encoding="utf-8")
    crude.write_text("legacy-crude\n", encoding="utf-8")

    panel = generate_pharma_sales_panel()
    written = write_pharma_sales_csv(panel, tmp_path)
    assert written.name == PRIMARY_FILENAME
    assert written.exists()
    assert digital.read_text(encoding="utf-8") == "legacy-digital\n"
    assert crude.read_text(encoding="utf-8") == "legacy-crude\n"
    saved = pd.read_csv(written)
    assert set(saved["data_type"]) == {"synthetic_demo"}


def test_generator_source_does_not_import_legacy_pipelines() -> None:
    banned = {
        "src.data_loader",
        "src.commodity_features",
        "src.commodity_analytics",
        "src.indicator_engine",
        "src.analytics",
        "src.forecasting",
        "scripts.generate_sample_data",
        "scripts.generate_commodity_data",
        "scripts.run_pipeline",
    }
    for relative in ("scripts/generate_pharma_data.py", "src/pharma_schema.py"):
        tree = ast.parse((PROJECT_ROOT / relative).read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        assert imported.isdisjoint(banned)


def test_checked_in_csv_matches_generator_when_present() -> None:
    """The raw demo file is optional until the generator has been run."""
    path = PROJECT_ROOT / "data" / "raw" / PRIMARY_FILENAME
    if not path.exists():
        pytest.skip("Run python scripts/generate_pharma_data.py to materialize the demo CSV")
    loaded = load_pharma_panel()
    fresh = generate_pharma_sales_panel()
    pd.testing.assert_frame_equal(
        loaded.reset_index(drop=True),
        fresh.reset_index(drop=True),
        check_dtype=False,
    )
