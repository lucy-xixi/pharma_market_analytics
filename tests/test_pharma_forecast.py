"""Chronological forecast evaluation."""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dashboard.data_service import filter_panel, forecast_input_frame, forecast_selection
from src.pharma_forecast import (
    FORECAST_GRAINS,
    _predict_model,
    error_metrics,
    evaluate_grouped_forecast,
    evaluate_sales_forecast,
    select_model_by_validation_ranks,
)
from src.pharma_quality import prepare_sales_panel


def _monthly(values: list[float], start: str = "2021-01-01") -> pd.Series:
    index = pd.date_range(start, periods=len(values), freq="MS")
    return pd.Series(values, index=index, name="sales_amount")


def test_mape_skips_zero_actuals() -> None:
    metrics = error_metrics([0.0, 10.0], [5.0, 20.0])
    assert metrics["mape"] == pytest.approx(100.0)
    assert metrics["mape_excluded_zeros"] == 1
    assert metrics["mae"] == pytest.approx(7.5)


def test_short_series_records_fallback_and_does_not_forecast() -> None:
    result = evaluate_sales_forecast(_monthly([1.0] * 10), test_months=6, min_train_months=18)
    assert result["status"] == "insufficient_history"
    assert "No model was fit" in result["reason"]
    assert "not used to select" in result["reason"]
    assert result["forward"] == []
    assert result["models"] == {}
    assert result["selected_model"] is None
    assert result["test_metrics"] is None


def test_final_test_values_do_not_change_model_selection() -> None:
    clean = _monthly([float(value) for value in range(1, 37)])
    spiked_values = [float(value) for value in range(1, 37)]
    spiked_values[-6:] = [1000.0] * 6
    spiked = _monthly(spiked_values)
    clean_result = evaluate_sales_forecast(clean)
    spiked_result = evaluate_sales_forecast(spiked)

    assert clean_result["status"] == "ok"
    assert spiked_result["status"] == "ok"
    assert spiked_result["selected_model"] == clean_result["selected_model"]
    assert spiked_result["validation_summary"] == clean_result["validation_summary"]
    assert spiked_result["test_start"] == "2023-07-01"
    assert spiked_result["train_end"] < spiked_result["test_start"]
    for fold in spiked_result["validation_folds"]:
        assert fold["val_end"] < spiked_result["test_start"]
        assert fold["train_end"] < fold["val_start"]
    detail = pd.DataFrame(spiked_result["test_detail"])
    selected = spiked_result["selected_model"]
    assert selected in detail.columns
    assert (detail[selected] != 1000).all()
    assert spiked_result["test_metrics"]["mae"] != clean_result["test_metrics"]["mae"]
    assert "test RMSE" not in spiked_result["selection_rule"]


def test_perfect_line_has_zero_linear_test_error() -> None:
    result = evaluate_sales_forecast(_monthly(list(np.arange(1, 37, dtype=float))))
    assert result["status"] == "ok"
    assert result["selected_model"] == "linear_trend"
    assert result["test_metrics"]["mae"] == pytest.approx(0.0, abs=1e-6)
    assert result["test_metrics"]["rmse"] == pytest.approx(0.0, abs=1e-6)
    assert result["test_metrics"]["mape_excluded_zeros"] == 0
    assert "实际销售额较小" in result["test_metrics"]["mape_note"]
    assert result["validation_end"] < result["test_start"]
    assert result["forward"]
    assert result["selected_forward_model"] == result["selected_model"]
    assert result["intervals"] is None


def test_seasonal_naive_is_skipped_when_training_history_is_shorter_than_a_year() -> None:
    result = evaluate_sales_forecast(
        _monthly(list(range(1, 15))),
        test_months=3,
        min_train_months=8,
    )
    assert result["models"]["seasonal_naive"]["status"] == "failed"
    assert "不使用季节性朴素模型" in result["models"]["seasonal_naive"]["reason"]
    assert result["models"]["naive"]["status"] == "ok"
    assert result["n_train"] == 11
    assert result["train_end"] < result["test_start"]


def test_all_zero_series_warns_and_keeps_mape_undefined() -> None:
    result = evaluate_sales_forecast(_monthly([0.0] * 30))
    assert any("全部为 0" in warning for warning in result["warnings"])
    assert result["status"] == "ok"
    assert result["selected_model"] == "naive"
    assert result["test_metrics"]["mae"] == pytest.approx(0.0)
    assert result["test_metrics"]["mape"] is None
    assert result["test_metrics"]["mape_excluded_zeros"] == 6
    assert result["validation_end"] < result["test_start"]


def test_missing_month_is_warned_and_not_inserted_as_zero() -> None:
    series = _monthly(list(range(1, 37)))
    series = series.drop(series.index[3])
    result = evaluate_sales_forecast(series)
    assert result["n_history"] == 35
    assert result["status"] == "ok"
    assert any("没有填成 0" in warning for warning in result["warnings"])
    assert result["train_end"] < result["test_start"]
    for fold in result["validation_folds"]:
        assert fold["val_end"] < result["test_start"]
    detail = pd.DataFrame(result["test_detail"])
    assert len(detail) == 6
    assert "2021-04-01" not in set(detail["month"])
    assert (detail["actual"] != 0).all()


def test_validation_windows_end_before_the_locked_test() -> None:
    result = evaluate_sales_forecast(_monthly(list(range(1, 49))))
    assert result["n_validation_folds"] == 4
    assert result["test_start"] == "2024-07-01"
    assert result["test_end"] == "2024-12-01"
    assert result["validation_end"] == "2024-06-01"
    assert [
        (fold["val_start"], fold["val_end"]) for fold in result["validation_folds"]
    ] == [
        ("2022-07-01", "2022-12-01"),
        ("2023-01-01", "2023-06-01"),
        ("2023-07-01", "2023-12-01"),
        ("2024-01-01", "2024-06-01"),
    ]
    for fold in result["validation_folds"]:
        assert fold["train_end"] < fold["val_start"]
        assert fold["n_train"] >= 18


def test_history_shorter_than_one_validation_fold_does_not_use_the_test() -> None:
    result = evaluate_sales_forecast(_monthly(list(range(1, 30))))
    assert result["status"] == "insufficient_history"
    assert result["selected_model"] is None
    assert result["test_metrics"] is None
    assert result["test_detail"] == []
    assert "not used to select" in result["reason"]


def test_thirty_observed_months_have_one_validation_fold() -> None:
    result = evaluate_sales_forecast(_monthly(list(range(1, 31))))
    assert result["status"] == "ok"
    assert result["n_validation_folds"] == 1
    assert result["validation_folds"][0]["val_start"] == "2022-07-01"
    assert result["validation_folds"][0]["val_end"] == "2022-12-01"
    assert result["test_start"] == "2023-01-01"
    assert result["validation_end"] < result["test_start"]


def test_average_rank_prefers_balance_over_either_single_metric() -> None:
    choice = select_model_by_validation_ranks(
        {
            "naive": {"mae": 3.0, "rmse": 10.0},
            "seasonal_naive": {"mae": 2.0, "rmse": 9.0},
            "linear_trend": {"mae": 4.0, "rmse": 8.0},
            "arima": {"mae": 5.0, "rmse": 11.0},
        }
    )
    ranking = choice["ranking"]
    assert min(ranking, key=lambda name: ranking[name]["mae"]) == "seasonal_naive"
    assert min(ranking, key=lambda name: ranking[name]["rmse"]) == "linear_trend"
    assert choice["selected_model"] == "seasonal_naive"
    assert ranking["seasonal_naive"]["average_rank"] == pytest.approx(1.5)


def test_average_rank_tie_prefers_the_simpler_model() -> None:
    choice = select_model_by_validation_ranks(
        {
            "naive": {"mae": 1.0, "rmse": 4.0},
            "seasonal_naive": {"mae": 2.0, "rmse": 3.0},
            "linear_trend": {"mae": 3.0, "rmse": 2.0},
            "arima": {"mae": 4.0, "rmse": 1.0},
        }
    )
    assert choice["selected_model"] == "naive"
    assert {
        name: payload["average_rank"] for name, payload in choice["ranking"].items()
    } == {
        "naive": 2.5,
        "seasonal_naive": 2.5,
        "linear_trend": 2.5,
        "arima": 2.5,
    }


def _sales_row(month: str, product: str, region: str, amount: float) -> dict:
    return {
        "month": month,
        "product": product,
        "region": region,
        "sales_amount": amount,
        "sales_volume": 1.0,
        "data_source": "synthetic_demo",
        "data_type": "synthetic_demo",
    }


def _grain_panel() -> pd.DataFrame:
    rows = []
    for index, month in enumerate(pd.date_range("2021-01-01", periods=30, freq="MS")):
        stamp = month.strftime("%Y-%m-%d")
        level = float(index + 1)
        rows.extend(
            [
                _sales_row(stamp, "PX-A", "East", 10.0),
                _sales_row(stamp, "PX-A", "West", 10.0),
                _sales_row(stamp, "PX-B", "East", level),
                _sales_row(stamp, "PX-B", "West", level),
            ]
        )
    return pd.DataFrame(rows)


def _by_id(grouped: dict, series_id: str) -> dict:
    return next(item for item in grouped["series"] if item["series_id"] == series_id)


def test_linear_uses_calendar_month_not_observation_order() -> None:
    index = pd.to_datetime(["2021-01-01", "2021-02-01", "2021-03-01", "2021-05-01", "2021-06-01"])
    train = pd.Series([0.0, 1.0, 2.0, 4.0, 5.0], index=index)
    predicted, reason, _extra = _predict_model("linear_trend", train, pd.DatetimeIndex(["2021-07-01"]), 12)
    assert reason is None
    assert predicted is not None
    assert predicted[0] == pytest.approx(6.0)
    arima_pred, arima_reason, _extra = _predict_model("arima", train, pd.DatetimeIndex(["2021-07-01"]), 12)
    assert arima_pred is None
    assert "缺失月" in arima_reason


def test_each_forecast_grain_keeps_its_own_series_identity() -> None:
    panel = _grain_panel()
    grouped = {grain: evaluate_grouped_forecast(panel, grain=grain) for grain in FORECAST_GRAINS}
    assert grouped["total"]["n_series"] == 1
    assert grouped["product"]["n_series"] == 2
    assert grouped["region"]["n_series"] == 2
    assert grouped["product_region"]["n_series"] == 4

    total = _by_id(grouped["total"], "total")
    product_a = _by_id(grouped["product"], "product:PX-A")
    product_b = _by_id(grouped["product"], "product:PX-B")
    east = _by_id(grouped["region"], "region:East")
    pair = _by_id(grouped["product_region"], "product_region:PX-A|East")

    assert total["product"] is None and total["region"] is None
    assert product_a["product"] == "PX-A" and product_a["region"] is None
    assert east["region"] == "East" and east["product"] is None
    assert pair["product"] == "PX-A" and pair["region"] == "East"
    assert product_a["history"][0]["sales_amount"] == pytest.approx(20.0)
    assert pair["history"][0]["sales_amount"] == pytest.approx(10.0)
    assert total["history"][0]["sales_amount"] == pytest.approx(22.0)
    assert product_a["history"][0]["sales_amount"] != total["history"][0]["sales_amount"]
    assert product_a["selected_model"] == "naive"
    assert product_b["selected_model"] == "linear_trend"
    assert product_a["selected_model"] != product_b["selected_model"]

    for grain, result in grouped.items():
        for item in result["series"]:
            assert item["forecast_grain"] == grain
            assert item["series_id"]
            assert item["status"] == "ok"
            assert item["intervals"] is None
            for fold in item["validation_folds"]:
                assert fold["val_end"] < item["test_start"]
                assert fold["train_end"] < fold["val_start"]


def test_short_group_is_kept_and_does_not_fall_back_to_the_test_set() -> None:
    rows = []
    for index, month in enumerate(pd.date_range("2021-01-01", periods=30, freq="MS")):
        rows.append(_sales_row(month.strftime("%Y-%m-%d"), "PX-LONG", "East", 5.0))
    for month in pd.date_range("2021-01-01", periods=8, freq="MS"):
        rows.append(_sales_row(month.strftime("%Y-%m-%d"), "PX-SHORT", "East", 5.0))
    result = evaluate_grouped_forecast(pd.DataFrame(rows), grain="product")
    assert {item["product"] for item in result["series"]} == {"PX-LONG", "PX-SHORT"}
    short = _by_id(result, "product:PX-SHORT")
    assert short["status"] == "insufficient_history"
    assert short["selected_model"] is None
    assert short["test_metrics"] is None
    assert "not used to select" in short["reason"]
    assert _by_id(result, "product:PX-LONG")["status"] == "ok"


def test_month_slider_changes_descriptive_pages_but_not_the_forecast_page() -> None:
    """Drive the sidebar slider through the app, not only the helper signature."""
    from streamlit.testing.v1 import AppTest

    app_path = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=180)
    at.run()
    narrow = ("2021-01-01", "2021-06-01")
    at.sidebar.select_slider[0].set_value(narrow)
    at.run()

    trend = next(item.value for item in at.caption if str(item.value).startswith("时间范围"))
    assert "2021-01-01 至 2021-06-01" in trend
    assert len(at.dataframe[0].value) == 6

    at.sidebar.radio[0].set_value("增长率")
    at.run()
    growth_months = at.dataframe[0].value["month"].astype(str).tolist()
    assert growth_months[0] == "2021-01-01"
    assert growth_months[-1] == "2021-06-01"
    assert len(growth_months) == 6

    at.sidebar.radio[0].set_value("产品与地区对比")
    at.run()
    comparison = next(item.value for item in at.caption if "对比月份" in str(item.value))
    assert "2021-06-01" in comparison

    at.sidebar.radio[0].set_value("预测与模型评估")
    at.run()
    assert at.error == []
    narrow_frames = [frame.value.copy() for frame in at.dataframe]
    summary = narrow_frames[0].iloc[0]
    assert summary["状态"] == "ok"
    assert summary["验证折数"] == 4
    assert summary["验证期"] == "2022-07-01 至 2024-06-01"
    assert summary["最终测试期"] == "2024-07-01 至 2024-12-01"
    assert list(narrow_frames[-1]["month"].astype(str)) == [
        "2025-01-01",
        "2025-02-01",
        "2025-03-01",
        "2025-04-01",
        "2025-05-01",
        "2025-06-01",
    ]
    headings = [item.value for item in at.subheader]
    assert "验证集指标" in headings
    assert "最终测试集指标" in headings
    assert "未来 6 个月点预测" in headings

    at.sidebar.select_slider[0].set_value(("2021-01-01", "2024-12-01"))
    at.run()
    full_frames = [frame.value.copy() for frame in at.dataframe]
    assert len(narrow_frames) == len(full_frames)
    for left, right in zip(narrow_frames, full_frames):
        pd.testing.assert_frame_equal(left, right)


def test_forecast_input_ignores_the_display_month_window() -> None:
    panel = _grain_panel()
    assert "month_start" not in inspect.signature(forecast_selection).parameters
    displayed = filter_panel(panel, ("PX-A",), ("East",), "2021-06-01", "2021-08-01")
    model_rows = forecast_input_frame(panel, ("PX-A",), ("East",))
    assert displayed["month"].nunique() == 3
    assert model_rows["month"].nunique() == 30
    assert model_rows["month"].max() == panel["month"].max()
    narrowed = evaluate_grouped_forecast(displayed, grain="product_region")
    full = evaluate_grouped_forecast(model_rows, grain="product_region")
    assert narrowed["series"][0]["status"] == "insufficient_history"
    assert full["series"][0]["status"] == "ok"
    assert full["series"][0]["series_id"] == "product_region:PX-A|East"
    assert full["series"][0]["test_end"] == "2023-06-01"


def test_grouped_forecast_sees_only_rows_kept_by_demo_rules() -> None:
    rows = [
        _sales_row(month.strftime("%Y-%m-%d"), "PX-A", "East", 5.0)
        for month in pd.date_range("2021-01-01", periods=30, freq="MS")
    ]
    rows.append(_sales_row("2021-01-01", "PX-A", "East", 999.0))
    rows.append(_sales_row("2020-12-01", "PX-A", "East", -5.0))
    analytical, report = prepare_sales_panel(pd.DataFrame(rows))
    result = evaluate_grouped_forecast(analytical, grain="product_region")
    series = result["series"][0]
    assert report["raw"]["duplicate_key_excess"] == 1
    assert report["raw"]["negative_sales_amount_rows"] == 1
    assert series["series_id"] == "product_region:PX-A|East"
    assert series["status"] == "ok"
    assert {point["sales_amount"] for point in series["history"]} == {5.0}
    assert "2020-12-01" not in {point["month"] for point in series["history"]}

