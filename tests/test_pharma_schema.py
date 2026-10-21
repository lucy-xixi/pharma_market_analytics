"""Contract checks for the pharmaceutical sales panel."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.config import LEGACY_RAW_FILENAMES, PROJECT_ROOT, load_pharma_settings
from src.data_dictionary import load_pharma_data_dictionary, validate_dataframe_against_dictionary
from src.pharma_schema import (
    ALLOWED_DATA_TYPES,
    DATASET_NAME,
    PRIMARY_FILENAME,
    REQUIRED_COLUMNS,
    UNIQUE_KEY,
    load_pharma_panel,
    simulate_sales_observation,
    validate_pharma_contract,
)


def _valid_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "month": "2021-01-01",
        "product": "PX-CARD-01",
        "region": "East China",
        "sales_amount": 100.0,
        "sales_volume": 10.0,
        "data_source": "synthetic_demo",
        "data_type": "synthetic_demo",
    }
    row.update(overrides)
    return row


def test_shipped_pharma_config_is_separate_from_legacy_settings() -> None:
    pharma = load_pharma_settings()

    assert pharma["data"]["primary_file"] == PRIMARY_FILENAME
    assert pharma["data"]["dataset_name"] == DATASET_NAME
    assert pharma["data"]["required_columns"] == REQUIRED_COLUMNS
    assert pharma["data"]["unique_key"] == UNIQUE_KEY
    assert set(pharma["data"]["allowed_data_types"]) == ALLOWED_DATA_TYPES
    assert pharma["data"]["synthetic_data_type"] == "synthetic_demo"
    assert pharma["data"]["synthetic_data_source"] == "synthetic_demo"
    assert pharma["data"]["seed"] == 42
    assert pharma["data"]["date_start"] == "2021-01-01"
    assert pharma["data"]["date_end"] == "2024-12-01"
    assert pharma["data"]["frequency"] == "MS"
    assert "commodity" not in pharma
    assert "indicators" not in pharma
    assert Path(pharma["paths"]["database"]).name == "pharma_analytics.db"
    assert Path(pharma["paths"]["raw_data"]).is_absolute()


def test_worked_example_volume_and_amount() -> None:
    volume, amount = simulate_sales_observation(
        base_volume=12000,
        region_multiplier=1.2,
        t=0,
        month=1,
        monthly_trend=0.004,
        season_amplitude=0.06,
        peak_month=1,
        unit_price=86.0,
        price_drift_per_month=0.001,
        noise=0.0,
    )
    assert volume == 15264.0
    assert amount == 1312704.0


def test_large_negative_noise_floors_volume_at_zero() -> None:
    volume, amount = simulate_sales_observation(
        base_volume=100,
        region_multiplier=1.0,
        t=0,
        month=1,
        monthly_trend=0.0,
        season_amplitude=0.0,
        peak_month=1,
        unit_price=10.0,
        price_drift_per_month=0.0,
        noise=-2.0,
    )
    assert volume == 0.0
    assert amount == 0.0


def test_contract_accepts_synthetic_demo_and_real_token() -> None:
    synthetic = pd.DataFrame([_valid_row()])
    assert validate_pharma_contract(synthetic) == []

    real = pd.DataFrame(
        [_valid_row(data_type="real", data_source="authorized_source_name")]
    )
    assert validate_pharma_contract(real) == []


def test_contract_rejects_invalid_type_duplicate_negative_and_mid_month() -> None:
    bad_type = pd.DataFrame([_valid_row(data_type="synthetic")])
    assert any("data_type outside" in issue for issue in validate_pharma_contract(bad_type))

    duplicate = pd.DataFrame([_valid_row(), _valid_row(sales_amount=5.0)])
    assert any("duplicate business keys" in issue for issue in validate_pharma_contract(duplicate))

    negative = pd.DataFrame([_valid_row(sales_amount=-1.0, sales_volume=-2.0)])
    before = negative.copy(deep=True)
    issues = validate_pharma_contract(negative)
    pd.testing.assert_frame_equal(negative, before)
    assert "negative sales_amount: 1" in issues
    assert "negative sales_volume: 1" in issues
    assert negative.loc[0, "sales_amount"] == -1.0

    mid_month = pd.DataFrame([_valid_row(month="2021-01-15")])
    assert any("month-start" in issue for issue in validate_pharma_contract(mid_month))

    missing = pd.DataFrame([{"month": "2021-01-01"}])
    assert any("missing columns" in issue for issue in validate_pharma_contract(missing))


def test_contract_does_not_insert_missing_months() -> None:
    gapped = pd.DataFrame(
        [
            _valid_row(month="2021-01-01"),
            _valid_row(month="2021-03-01", sales_amount=110.0, sales_volume=11.0),
        ]
    )
    before = gapped.copy(deep=True)
    assert validate_pharma_contract(gapped) == []
    pd.testing.assert_frame_equal(gapped, before)
    assert len(gapped) == 2
    assert "2021-02-01" not in set(gapped["month"])


def test_dictionary_matches_contract_columns() -> None:
    dictionary = load_pharma_data_dictionary()
    assert set(dictionary["dataset_name"]) == {DATASET_NAME}
    assert set(dictionary["column_name"]) == set(REQUIRED_COLUMNS)
    frame = pd.DataFrame([_valid_row()])
    assert validate_dataframe_against_dictionary(frame, DATASET_NAME, dictionary) == []


def test_catalog_marks_pharma_file_as_synthetic_demo() -> None:
    catalog = pd.read_csv(PROJECT_ROOT / "data" / "metadata" / "datasets_catalog.csv")
    row = catalog.loc[catalog["file_name"] == PRIMARY_FILENAME].iloc[0]
    assert row["domain"] == "pharma_market"
    assert row["data_type"] == "synthetic_demo"
    assert row["data_source"] == "synthetic_demo"
    assert row["frequency"] == "monthly"


def test_loader_refuses_legacy_files_and_does_not_scan_the_directory(tmp_path: Path) -> None:
    legacy = tmp_path / "digital_industry_indicators.csv"
    legacy.write_text("year,industry\n2015,Software\n", encoding="utf-8")
    crude = tmp_path / "crude_oil_weekly_synthetic.csv"
    crude.write_text("date,commodity_id\n2020-01-03,WTI_CRUDE\n", encoding="utf-8")
    target = tmp_path / PRIMARY_FILENAME
    pd.DataFrame([_valid_row()]).to_csv(target, index=False)

    settings = load_pharma_settings()
    settings["paths"]["raw_data"] = str(tmp_path)
    loaded = load_pharma_panel(settings=settings)
    assert list(loaded.columns) == REQUIRED_COLUMNS
    assert loaded.loc[0, "product"] == "PX-CARD-01"
    assert "industry" not in loaded.columns

    with pytest.raises(ValueError, match="legacy"):
        load_pharma_panel(settings=settings, path=legacy)
    with pytest.raises(ValueError, match="legacy"):
        load_pharma_panel(settings=settings, path=crude)


def test_legacy_raw_filenames_cover_shipped_non_pharma_extracts() -> None:
    assert "digital_industry_indicators.csv" in LEGACY_RAW_FILENAMES
    assert "crude_oil_weekly_synthetic.csv" in LEGACY_RAW_FILENAMES
    raw = PROJECT_ROOT / "data" / "raw"
    assert not (raw / "digital_industry_indicators.csv").exists()
    assert not (raw / "crude_oil_weekly_synthetic.csv").exists()


def test_docs_match_forecast_and_unit_language() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    notes = (PROJECT_ROOT / "data" / "metadata" / "PHARMA_DATA_NOTES.md").read_text(encoding="utf-8")
    app = (PROJECT_ROOT / "dashboard" / "app.py").read_text(encoding="utf-8")
    dictionary = load_pharma_data_dictionary().set_index("column_name")
    assert dictionary.loc["sales_amount", "unit"] == "模拟金额，未指定真实币种"
    assert dictionary.loc["sales_volume", "unit"] == "模拟数量，非真实销售包装单位或处方量"
    assert "Phase 1 does not" not in notes
    assert "or train a forecast model" not in notes
    assert "测试 RMSE 最低" not in readme
    assert "测试 RMSE 更低" not in app
    assert "按产品×地区" in readme
    assert "不参与模型选择" in readme or "不参与选择" in readme
