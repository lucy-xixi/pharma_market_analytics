"""
Chronological forecast evaluation for monthly pharmaceutical sales.

The last `test_months` calendar months are a locked test set. Model choice
uses rolling-origin validation that ends before that test set. MAE and RMSE
on the validation folds are ranked separately; the selected model has the
better average rank. The test set is scored once, after that choice, and
only for the selected model. MAPE is descriptive and is not a selection metric.

Missing months are not filled with zero. Prediction intervals are not
implemented (`intervals` stays None).
"""

from __future__ import annotations

import warnings
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import rankdata

MODEL_NAIVE = "naive"
MODEL_SEASONAL_NAIVE = "seasonal_naive"
MODEL_LINEAR = "linear_trend"
MODEL_ARIMA = "arima"
# Lower number wins a tie on average validation rank.
SIMPLICITY_ORDER = (MODEL_NAIVE, MODEL_SEASONAL_NAIVE, MODEL_LINEAR, MODEL_ARIMA)
SELECTION_RULE = (
    "Lowest average of validation MAE rank and validation RMSE rank. "
    "Ties prefer the simpler model: naive, seasonal_naive, linear_trend, arima. "
    "The final test set is not an input to this rule."
)
MAPE_NOTE = (
    "MAPE 只描述最终测试集，不参与模型选择。"
    "实际值为 0 的月份已排除。"
    "实际销售额较小时，MAPE 会被放大。"
)


def error_metrics(
    actual: pd.Series | np.ndarray,
    predicted: pd.Series | np.ndarray,
) -> dict[str, Any]:
    """MAE, RMSE, and MAPE. MAPE ignores non-finite pairs and actual zeros."""
    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    finite = np.isfinite(actual_values) & np.isfinite(predicted_values)
    if int(finite.sum()) == 0:
        return {
            "mae": None,
            "rmse": None,
            "mape": None,
            "mape_excluded_zeros": 0,
            "n_scored": 0,
        }
    observed = actual_values[finite]
    forecast = predicted_values[finite]
    error = forecast - observed
    zero_actual = observed == 0
    mape_mask = ~zero_actual
    mape = None
    if int(mape_mask.sum()) > 0:
        mape = float(np.mean(np.abs(error[mape_mask] / observed[mape_mask])) * 100.0)
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mape": mape,
        "mape_excluded_zeros": int(zero_actual.sum()),
        "n_scored": int(finite.sum()),
    }


def _as_monthly_series(series: pd.Series) -> pd.Series:
    index = pd.to_datetime(series.index, errors="coerce")
    values = pd.to_numeric(series, errors="coerce")
    out = pd.Series(values.to_numpy(dtype=float), index=index).sort_index()
    out = out[out.index.notna()]
    out = out[~out.index.duplicated(keep="first")]
    out.index = pd.DatetimeIndex(out.index).to_period("M").to_timestamp()
    return out.astype(float)


def _fit_linear(series: pd.Series) -> tuple[float, float]:
    x_values = np.arange(len(series), dtype=float)
    y_values = series.to_numpy(dtype=float)
    slope, intercept, _r, _p, _se = stats.linregress(x_values, y_values)
    return float(slope), float(intercept)


def _predict_linear(slope: float, intercept: float, start_x: int, steps: int) -> np.ndarray:
    x_values = np.arange(start_x, start_x + steps, dtype=float)
    return intercept + slope * x_values


def _naive_forecast(history: pd.Series, future_index: pd.DatetimeIndex) -> np.ndarray:
    """Repeat the last training observation. Does not read later actuals."""
    last_value = float(history.dropna().iloc[-1])
    return np.full(len(future_index), last_value, dtype=float)


def _seasonal_naive(history: pd.Series, future_index: pd.DatetimeIndex, season: int) -> np.ndarray:
    lookup = history.copy()
    lookup.index = pd.DatetimeIndex(lookup.index).to_period("M").to_timestamp()
    predictions: list[float] = []
    for timestamp in future_index:
        lagged = pd.Timestamp(timestamp) - pd.DateOffset(months=season)
        lagged = lagged.to_period("M").to_timestamp()
        if lagged in lookup.index and pd.notna(lookup.loc[lagged]):
            predictions.append(float(lookup.loc[lagged]))
        else:
            predictions.append(float("nan"))
    return np.asarray(predictions, dtype=float)


def _fit_arima(train: pd.Series, steps: int) -> tuple[np.ndarray, str]:
    from statsmodels.tsa.arima.model import ARIMA

    values = train.to_numpy(dtype=float)
    last_error = "ARIMA was not attempted"
    for order in ((1, 1, 1), (1, 1, 0), (0, 1, 1)):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitted = ARIMA(values, order=order).fit()
            forecast = np.asarray(fitted.get_forecast(steps=steps).predicted_mean, dtype=float)
            return forecast, f"arima{order}"
        except Exception as exc:  # noqa: BLE001
            last_error = f"ARIMA{order} failed: {exc}"
    raise RuntimeError(last_error)


DISPLAY_NAMES = {
    MODEL_NAIVE: "朴素基线（上一观测值）",
    MODEL_SEASONAL_NAIVE: "季节性朴素基线（去年同月）",
    MODEL_LINEAR: "线性趋势",
    MODEL_ARIMA: "ARIMA",
}
MODEL_ROLES = {
    MODEL_NAIVE: "baseline",
    MODEL_SEASONAL_NAIVE: "baseline",
    MODEL_LINEAR: "time_series",
    MODEL_ARIMA: "time_series",
}


def _month_stamp(value: object) -> pd.Timestamp:
    return pd.Timestamp(value).to_period("M").to_timestamp()


def _iso(value: object) -> str:
    return _month_stamp(value).strftime("%Y-%m-%d")


def select_model_by_validation_ranks(
    model_scores: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Choose a model from validation MAE and RMSE only.

    `model_scores` maps a model name to ``{"mae": float, "rmse": float}``.
    Those numbers must already be aggregated from validation folds. This
    function does not see the final test set.
    """
    names = [
        name
        for name, scores in model_scores.items()
        if scores.get("mae") is not None
        and scores.get("rmse") is not None
        and np.isfinite(scores["mae"])
        and np.isfinite(scores["rmse"])
    ]
    if not names:
        return {"selected_model": None, "ranking": {}, "selection_rule": SELECTION_RULE}

    mae_rank = rankdata([model_scores[name]["mae"] for name in names], method="average")
    rmse_rank = rankdata([model_scores[name]["rmse"] for name in names], method="average")
    ranking: dict[str, dict[str, Any]] = {}
    for index, name in enumerate(names):
        average_rank = float(mae_rank[index] + rmse_rank[index]) / 2.0
        ranking[name] = {
            "mae": float(model_scores[name]["mae"]),
            "rmse": float(model_scores[name]["rmse"]),
            "mae_rank": float(mae_rank[index]),
            "rmse_rank": float(rmse_rank[index]),
            "average_rank": average_rank,
            "simplicity": SIMPLICITY_ORDER.index(name) if name in SIMPLICITY_ORDER else len(SIMPLICITY_ORDER),
        }
    selected = min(names, key=lambda name: (ranking[name]["average_rank"], ranking[name]["simplicity"]))
    return {"selected_model": selected, "ranking": ranking, "selection_rule": SELECTION_RULE}


def _month_offsets(index: pd.DatetimeIndex, origin: pd.Timestamp) -> np.ndarray:
    """Calendar months from `origin`. A missing month does not shorten later offsets."""
    origin = _month_stamp(origin)
    stamps = pd.DatetimeIndex(pd.to_datetime(index)).to_period("M").to_timestamp()
    return np.array(
        [(stamp.year - origin.year) * 12 + (stamp.month - origin.month) for stamp in stamps],
        dtype=float,
    )


def _calendar_complete(index: pd.Index) -> bool:
    stamps = pd.DatetimeIndex(pd.to_datetime(index)).to_period("M").to_timestamp().sort_values()
    if len(stamps) == 0 or not stamps.is_unique:
        return False
    expected = pd.date_range(stamps.min(), stamps.max(), freq="MS")
    return len(expected) == len(stamps)


def _predict_arima_calendar(
    train: pd.Series,
    future_index: pd.DatetimeIndex,
) -> tuple[Optional[np.ndarray], Optional[str], dict[str, Any]]:
    """Forecast calendar months after a complete training span, then keep observed months."""
    if not _calendar_complete(train.index):
        return None, "训练期内有缺失月，ARIMA 不把不连续的观测当成连续月份。", {}
    train_end = _month_stamp(train.index.max())
    last_future = _month_stamp(pd.DatetimeIndex(future_index).max())
    n_steps = (last_future.year - train_end.year) * 12 + (last_future.month - train_end.month)
    if n_steps < 1:
        return None, "预测月份不在训练期之后。", {}
    try:
        forecast, fitted_name = _fit_arima(train, n_steps)
    except Exception as exc:  # noqa: BLE001
        return None, str(exc), {}
    calendar = pd.date_range(train_end + pd.DateOffset(months=1), periods=n_steps, freq="MS")
    if len(forecast) != len(calendar) or not np.isfinite(forecast).all():
        return None, "ARIMA returned a forecast that does not cover the holdout window.", {}
    lookup = {_month_stamp(ts): float(value) for ts, value in zip(calendar, forecast)}
    values: list[float] = []
    for timestamp in future_index:
        stamp = _month_stamp(timestamp)
        if stamp not in lookup:
            return None, "ARIMA 的日历预测没有覆盖某个实际观测月。", {}
        values.append(lookup[stamp])
    return np.asarray(values, dtype=float), None, {"fitted_name": fitted_name, "display_name": fitted_name}


def _predict_model(
    name: str,
    train: pd.Series,
    future_index: pd.DatetimeIndex,
    season_length: int,
) -> tuple[Optional[np.ndarray], Optional[str], dict[str, Any]]:
    """Forecast `future_index` from `train` only. Missing months are not filled with zero."""
    if name == MODEL_NAIVE:
        if train.dropna().empty:
            return None, "训练期没有观测值。", {}
        return _naive_forecast(train, future_index), None, {}
    if name == MODEL_SEASONAL_NAIVE:
        if len(train) < season_length:
            return None, (
                f"训练期只有 {len(train)} 个月，少于 {season_length} 个月，不使用季节性朴素模型。"
            ), {}
        predicted = _seasonal_naive(train, future_index, season_length)
        if not np.isfinite(predicted).all():
            return None, "训练窗内找不到 12 个月前的同月观测，不使用季节性朴素模型。", {}
        return predicted, None, {}
    if name == MODEL_LINEAR:
        if len(train) < 3:
            return None, (
                f"Linear trend needs at least 3 training months; train length is {len(train)}."
            ), {}
        origin = _month_stamp(train.index.min())
        x_values = _month_offsets(pd.DatetimeIndex(train.index), origin)
        if len(np.unique(x_values)) < 2:
            return None, "线性趋势至少需要两个不同的日历月。", {}
        slope, intercept, _r, _p, _se = stats.linregress(x_values, train.to_numpy(dtype=float))
        predicted = intercept + float(slope) * _month_offsets(future_index, origin)
        return predicted, None, {"train_slope": float(slope), "train_intercept": float(intercept)}
    if name == MODEL_ARIMA:
        return _predict_arima_calendar(train, future_index)
    return None, f"Unknown model {name}", {}


def _score_holdout(
    train: pd.Series,
    actual: pd.Series,
    season_length: int,
) -> dict[str, dict[str, Any]]:
    """Score every model on one holdout. MAE and RMSE are the selection inputs."""
    scored: dict[str, dict[str, Any]] = {}
    future_index = pd.DatetimeIndex(actual.index)
    for name in SIMPLICITY_ORDER:
        predicted, reason, extra = _predict_model(name, train, future_index, season_length)
        if predicted is None:
            payload = _failed_validation(reason or "model failed")
        else:
            metrics = error_metrics(actual.to_numpy(), predicted)
            payload = {
                "status": "ok",
                "reason": "",
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
            }
        payload.update(extra)
        scored[name] = payload
    return scored


def _failed_validation(reason: str) -> dict[str, Any]:
    return {"status": "failed", "reason": reason, "mae": None, "rmse": None}


def _validation_folds(
    pre_test: pd.Series,
    *,
    series_start: pd.Timestamp,
    test_start: pd.Timestamp,
    test_months: int,
    min_train_months: int,
) -> list[dict[str, Any]]:
    """Rolling origins on data that ends before `test_start`. Does not fill gaps."""
    folds: list[dict[str, Any]] = []
    val_start = _month_stamp(series_start) + pd.DateOffset(months=int(min_train_months))
    while _month_stamp(val_start) < _month_stamp(test_start):
        val_end = _month_stamp(val_start) + pd.DateOffset(months=int(test_months) - 1)
        if _month_stamp(val_end) >= _month_stamp(test_start):
            break
        train = pre_test[pre_test.index < _month_stamp(val_start)]
        validation = pre_test[
            (pre_test.index >= _month_stamp(val_start)) & (pre_test.index <= _month_stamp(val_end))
        ]
        if len(train) >= int(min_train_months) and len(validation) > 0:
            folds.append(
                {
                    "train": train,
                    "validation": validation,
                    "train_start": _iso(train.index.min()),
                    "train_end": _iso(train.index.max()),
                    "val_start": _iso(val_start),
                    "val_end": _iso(val_end),
                }
            )
        val_start = _month_stamp(val_start) + pd.DateOffset(months=int(test_months))
    return folds


def _insufficient(result: dict[str, Any], reason: str) -> dict[str, Any]:
    result["status"] = "insufficient_history"
    result["reason"] = (
        reason
        + " No model was fit and no forward forecast was produced."
        + " The final test set was not used to select a model."
    )
    result["selected_model"] = None
    result["selected_forward_model"] = None
    result["validation_folds"] = []
    result["validation_summary"] = {}
    result["test_metrics"] = None
    return result


def evaluate_sales_forecast(
    series: pd.Series,
    *,
    test_months: int = 6,
    horizon_months: int = 6,
    min_train_months: int = 18,
    season_length: int = 12,
) -> dict[str, Any]:
    """Validate before a locked final test window, then score the chosen model once."""
    history = _as_monthly_series(series).dropna()
    result: dict[str, Any] = {
        "status": "insufficient_history",
        "reason": "",
        "target": str(series.name or "sales_amount"),
        "test_months": int(test_months),
        "horizon_months": int(horizon_months),
        "min_train_months": int(min_train_months),
        "validation_horizon_months": int(test_months),
        "validation_step_months": int(test_months),
        "n_history": int(len(history)),
        "n_train": 0,
        "n_test": 0,
        "n_validation_folds": 0,
        "train_start": None,
        "train_end": None,
        "validation_start": None,
        "validation_end": None,
        "test_start": None,
        "test_end": None,
        "models": {},
        "validation_folds": [],
        "validation_summary": {},
        "selected_model": None,
        "selected_forward_model": None,
        "selection_rule": SELECTION_RULE,
        "test_metrics": None,
        "test_detail": [],
        "history": [
            {"month": _iso(timestamp), "sales_amount": float(value)}
            for timestamp, value in history.items()
        ],
        "forward": [],
        "warnings": [],
        "intervals": None,
        "disclaimer": (
            "Metrics describe this series only. synthetic_demo series do not "
            "measure a real pharmaceutical market."
        ),
    }
    if test_months < 1 or horizon_months < 1 or min_train_months < 1:
        return _insufficient(
            result,
            "test_months, horizon_months, and min_train_months must be positive.",
        )
    if len(history) >= 2:
        expected = pd.date_range(history.index.min(), history.index.max(), freq="MS")
        missing_months = len(expected) - len(history)
        if missing_months:
            result["warnings"].append(
                f"观测月份之间有 {missing_months} 个缺失月。"
                "缺失月份没有填成 0。验证和测试只使用实际出现的月份。"
            )
    if len(history) and bool((history.to_numpy(dtype=float) == 0).all()):
        result["warnings"].append(
            "销售额序列全部为 0。点预测可以计算，但 MAPE 不能解释为预测性能。"
        )
    if history.empty:
        return _insufficient(result, "Series has no observed months.")

    series_end = _month_stamp(history.index.max())
    series_start = _month_stamp(history.index.min())
    test_start = series_end - pd.DateOffset(months=int(test_months) - 1)
    test_end = series_end
    result["test_start"] = _iso(test_start)
    result["test_end"] = _iso(test_end)

    pre_test = history[history.index < _month_stamp(test_start)]
    test = history[(history.index >= _month_stamp(test_start)) & (history.index <= test_end)]
    fold_specs = _validation_folds(
        pre_test,
        series_start=series_start,
        test_start=_month_stamp(test_start),
        test_months=int(test_months),
        min_train_months=int(min_train_months),
    )
    if not fold_specs or test.empty:
        return _insufficient(
            result,
            (
                f"Series spans {result['n_history']} observed months from {_iso(series_start)} "
                f"to {_iso(series_end)}. Reliable validation needs an initial training window "
                f"of {min_train_months} months and at least one {test_months}-month validation "
                f"window that ends before the final {test_months} calendar-month test set "
                f"({result['test_start']} to {result['test_end']})."
            ),
        )

    fold_rows: list[dict[str, Any]] = []
    per_model: dict[str, list[dict[str, Any]]] = {name: [] for name in SIMPLICITY_ORDER}
    for fold_number, spec in enumerate(fold_specs, start=1):
        scored = _score_holdout(spec["train"], spec["validation"], season_length)
        for name, payload in scored.items():
            per_model[name].append(payload)
        fold_rows.append(
            {
                "fold": fold_number,
                "train_start": spec["train_start"],
                "train_end": spec["train_end"],
                "n_train": int(len(spec["train"])),
                "val_start": spec["val_start"],
                "val_end": spec["val_end"],
                "n_val": int(len(spec["validation"])),
                "models": scored,
            }
        )
    result["validation_folds"] = fold_rows
    result["n_validation_folds"] = len(fold_rows)
    result["validation_start"] = fold_rows[0]["val_start"]
    result["validation_end"] = fold_rows[-1]["val_end"]

    aggregate: dict[str, dict[str, float]] = {}
    for name, fold_payloads in per_model.items():
        if fold_payloads and all(item["status"] == "ok" for item in fold_payloads):
            aggregate[name] = {
                "mae": float(np.mean([item["mae"] for item in fold_payloads])),
                "rmse": float(np.mean([item["rmse"] for item in fold_payloads])),
            }
    choice = select_model_by_validation_ranks(aggregate)
    ranking = choice["ranking"]
    for name in SIMPLICITY_ORDER:
        entry: dict[str, Any] = {
            "display_name": DISPLAY_NAMES[name],
            "role": MODEL_ROLES[name],
            "metric_window": "validation",
            "mape": None,
        }
        if name in ranking:
            entry.update(
                {
                    "status": "ok",
                    "reason": "",
                    "mae": ranking[name]["mae"],
                    "rmse": ranking[name]["rmse"],
                    "mae_rank": ranking[name]["mae_rank"],
                    "rmse_rank": ranking[name]["rmse_rank"],
                    "average_rank": ranking[name]["average_rank"],
                }
            )
        else:
            reasons = [item["reason"] for item in per_model[name] if item["status"] != "ok" and item["reason"]]
            entry.update(
                _failed_validation(
                    reasons[0] if reasons else "未在全部验证折上成功，不参与模型选择。"
                )
            )
            entry["status"] = "failed"
        result["models"][name] = entry
    result["validation_summary"] = {
        name: {
            "mae_mean": scores["mae"],
            "rmse_mean": scores["rmse"],
            "mae_rank": ranking[name]["mae_rank"],
            "rmse_rank": ranking[name]["rmse_rank"],
            "average_rank": ranking[name]["average_rank"],
            "n_folds": len(fold_rows),
        }
        for name, scores in aggregate.items()
    }

    selected = choice["selected_model"]
    if selected is None:
        result["status"] = "no_model"
        result["reason"] = (
            "Every model failed on at least one validation fold. "
            "No model was fit for the final test set, and the test set was not used to select a model."
        )
        return result

    result["selected_model"] = selected
    result["selected_forward_model"] = selected
    test_train = pre_test
    predicted, reason, extra = _predict_model(
        selected,
        test_train,
        pd.DatetimeIndex(test.index),
        season_length,
    )
    if predicted is None:
        result["status"] = "failed"
        result["reason"] = (
            f"Model {selected} was selected on validation data but could not score the locked "
            f"test window ({reason}). The test set was not used to choose a different model."
        )
        return result

    test_metrics = error_metrics(test.to_numpy(), predicted)
    test_metrics["mape_note"] = MAPE_NOTE
    test_metrics["model"] = selected
    result["test_metrics"] = test_metrics
    result["n_train"] = int(len(test_train))
    result["n_test"] = int(len(test))
    result["train_start"] = _iso(test_train.index.min())
    result["train_end"] = _iso(test_train.index.max())
    detail = pd.DataFrame(
        {
            "month": [_iso(ts) for ts in test.index],
            "actual": test.to_numpy(dtype=float),
            selected: predicted,
        }
    )
    result["test_detail"] = detail.to_dict(orient="records")
    if extra.get("train_slope") is not None:
        result["models"][selected]["test_train_slope"] = extra["train_slope"]
        result["models"][selected]["test_train_intercept"] = extra["train_intercept"]
    if extra.get("fitted_name"):
        result["models"][selected]["test_fitted_name"] = extra["fitted_name"]

    result["status"] = "ok"
    mape_text = "undefined" if test_metrics["mape"] is None else f"{test_metrics['mape']:.3f}"
    result["reason"] = (
        f"验证期 {result['validation_start']} 至 {result['validation_end']} "
        f"（{result['n_validation_folds']} 折）用于选择模型。"
        f"最终测试期 {result['test_start']} 至 {result['test_end']} 未参与模型选择。"
        f"选定模型 {selected} 的测试 MAE={test_metrics['mae']:.3f}，"
        f"RMSE={test_metrics['rmse']:.3f}，MAPE={mape_text} "
        f"（排除零值月 {test_metrics['mape_excluded_zeros']}）。"
        f"{MAPE_NOTE}"
        " 向前预测在记录测试指标之后，用全部历史重新拟合选定模型。"
    )
    if pd.Timestamp(result["train_end"]) >= pd.Timestamp(result["test_start"]):
        result["status"] = "failed"
        result["reason"] = "Train window is not strictly before the test window."
        return result

    future_index = pd.date_range(
        series_end + pd.DateOffset(months=1),
        periods=horizon_months,
        freq="MS",
    )
    forward_values = _forward_forecast(selected, history, future_index, season_length, result)
    if forward_values is None:
        result["warnings"].append("选定模型的全历史重训失败。向前预测为空，模型选择保持不变。")
        return result
    result["forward"] = [
        {"month": _iso(ts), "forecast": float(value)}
        for ts, value in zip(future_index, forward_values)
    ]
    return result


def _forward_forecast(
    selected: str,
    history: pd.Series,
    future_index: pd.DatetimeIndex,
    season_length: int,
    result: dict[str, Any],
) -> Optional[np.ndarray]:
    """Refit the already selected model. A failure does not change the selection."""
    predicted, reason, extra = _predict_model(selected, history, future_index, season_length)
    if predicted is None:
        result["forward_error"] = reason
        return None
    if extra.get("fitted_name") and selected in result["models"]:
        result["models"][selected]["forward_refit"] = extra["fitted_name"]
    return predicted


GRAIN_TOTAL = "total"
GRAIN_PRODUCT = "product"
GRAIN_REGION = "region"
GRAIN_PRODUCT_REGION = "product_region"
FORECAST_GRAINS = (GRAIN_TOTAL, GRAIN_PRODUCT, GRAIN_REGION, GRAIN_PRODUCT_REGION)


def series_from_panel(
    df: pd.DataFrame,
    value_col: str = "sales_amount",
) -> pd.Series:
    """Monthly sum of the rows passed in. This does not group by product or region."""
    work = df.copy()
    work["month"] = pd.to_datetime(work["month"], errors="coerce")
    grouped = work.groupby("month")[value_col].sum().sort_index()
    grouped.index = pd.DatetimeIndex(grouped.index).to_period("M").to_timestamp()
    grouped.name = value_col
    return grouped


def _series_identity(grain: str, product: str | None, region: str | None) -> dict[str, Any]:
    if grain == GRAIN_TOTAL:
        series_id = "total"
        label = "所选产品与地区的销售额合计"
    elif grain == GRAIN_PRODUCT:
        series_id = f"product:{product}"
        label = f"{product}（所选地区合计）"
    elif grain == GRAIN_REGION:
        series_id = f"region:{region}"
        label = f"{region}（所选产品合计）"
    else:
        series_id = f"product_region:{product}|{region}"
        label = f"{product} × {region}"
    return {
        "forecast_grain": grain,
        "product": product,
        "region": region,
        "series_id": series_id,
        "series_label": label,
    }


def iter_forecast_series(
    df: pd.DataFrame,
    grain: str = GRAIN_TOTAL,
    value_col: str = "sales_amount",
):
    """Yield one monthly sales series per grain member.

    `total` sums every passed row by month. `product` sums the passed regions
    within each product. `region` sums the passed products within each region.
    `product_region` keeps each pair. Series are not pooled into one fit.
    """
    if grain not in FORECAST_GRAINS:
        raise ValueError(f"Unknown forecast grain: {grain}")
    if df is None or df.empty:
        return
    if grain == GRAIN_TOTAL:
        yield _series_identity(grain, None, None), series_from_panel(df, value_col)
        return
    group_cols = {
        GRAIN_PRODUCT: ["product"],
        GRAIN_REGION: ["region"],
        GRAIN_PRODUCT_REGION: ["product", "region"],
    }[grain]
    for key, group in df.groupby(group_cols, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        if grain == GRAIN_PRODUCT:
            product, region = str(key[0]), None
        elif grain == GRAIN_REGION:
            product, region = None, str(key[0])
        else:
            product, region = str(key[0]), str(key[1])
        yield _series_identity(grain, product, region), series_from_panel(group, value_col)


def evaluate_grouped_forecast(
    df: pd.DataFrame,
    *,
    grain: str = GRAIN_TOTAL,
    test_months: int = 6,
    horizon_months: int = 6,
    min_train_months: int = 18,
    season_length: int = 12,
    value_col: str = "sales_amount",
) -> dict[str, Any]:
    """Evaluate each grain series on its own history. Short series stay in the result."""
    series_results: list[dict[str, Any]] = []
    if df is None or df.empty:
        return {
            "forecast_grain": grain,
            "value_column": value_col,
            "n_series": 0,
            "series": [],
            "intervals": None,
        }
    for identity, series in iter_forecast_series(df, grain=grain, value_col=value_col):
        try:
            evaluated = evaluate_sales_forecast(
                series,
                test_months=test_months,
                horizon_months=horizon_months,
                min_train_months=min_train_months,
                season_length=season_length,
            )
        except Exception as exc:  # noqa: BLE001
            evaluated = {
                "status": "failed",
                "reason": f"预测失败，序列已保留：{exc}",
                "selected_model": None,
                "selected_forward_model": None,
                "validation_folds": [],
                "test_metrics": None,
                "test_detail": [],
                "forward": [],
                "warnings": [],
                "intervals": None,
                "history": [],
            }
        evaluated.update(identity)
        series_results.append(evaluated)
    return {
        "forecast_grain": grain,
        "value_column": value_col,
        "n_series": len(series_results),
        "series": series_results,
        "intervals": None,
    }


def evaluate_panel_forecast(
    df: pd.DataFrame,
    *,
    test_months: int = 6,
    horizon_months: int = 6,
    min_train_months: int = 18,
    season_length: int = 12,
    value_col: str = "sales_amount",
    grain: str = GRAIN_TOTAL,
) -> dict[str, Any]:
    """Total-grain calls keep the single-series payload. Other grains return every series."""
    grouped = evaluate_grouped_forecast(
        df,
        grain=grain,
        test_months=test_months,
        horizon_months=horizon_months,
        min_train_months=min_train_months,
        season_length=season_length,
        value_col=value_col,
    )
    if grain != GRAIN_TOTAL:
        return grouped
    if not grouped["series"]:
        empty = evaluate_sales_forecast(pd.Series(dtype=float, name=value_col))
        empty.update(_series_identity(GRAIN_TOTAL, None, None))
        return empty
    return grouped["series"][0]
