"""
Pharmaceutical Sales Analytics & Forecasting Platform.

Run from the repository root:

    streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.data_service import (  # noqa: E402
    filter_panel,
    forecast_selection,
    load_sales_bundle,
    monthly_total,
    product_cagr,
)
from src.pharma_forecast import (  # noqa: E402
    GRAIN_PRODUCT,
    GRAIN_PRODUCT_REGION,
    GRAIN_REGION,
    GRAIN_TOTAL,
)

INK = "#14343A"
ACCENT = "#1F6F6A"
AMBER = "#C47B2B"
BLUE = "#3D6B99"
PAPER = "#F4F7F7"

PAGES = ["销售趋势", "增长率", "产品与地区对比", "预测与模型评估", "数据质量"]
METRIC_NOTES = (
    "销售额（sales_amount）是模拟金额，未指定真实币种，并且非负。"
    "销量（sales_volume）是模拟数量，不是真实销售包装单位或处方量，并且非负。"
    "两者都来自 synthetic_demo 生成器，不能据此推断真实产品销量、市场份额或商业机会。"
)


def apply_theme() -> None:
    st.set_page_config(page_title="医药销售分析与预测", layout="wide")
    st.markdown(
        f"""
        <style>
        .stApp {{ background: {PAPER}; color: {INK}; }}
        h1, h2, h3 {{ color: {INK}; letter-spacing: 0; }}
        div[data-testid="stMetric"] {{
            background: white;
            border: 1px solid #D5E0E0;
            border-radius: 8px;
            padding: 8px 12px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def chart_layout(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(
        title={"text": title, "x": 0, "xanchor": "left"},
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"family": "Segoe UI, Microsoft YaHei, sans-serif", "color": INK, "size": 13},
        margin={"l": 68, "r": 96, "t": 108, "b": 56},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "x": 0, "xanchor": "left"},
        height=480,
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#E3EBEB", zeroline=False)
    return fig


def banner(frame: pd.DataFrame) -> None:
    if frame.empty or "data_type" not in frame.columns:
        st.info("当前没有可展示的销售行。模拟数据标记为 synthetic_demo，不能当成真实医药市场结论。")
        return
    kinds = "、".join(sorted(frame["data_type"].dropna().astype(str).unique())) or "无"
    sources = "、".join(sorted(frame["data_source"].dropna().astype(str).unique())) or "无"
    st.info(
        f"数据标记为 data_type={kinds}，data_source={sources}。"
        "这是固定种子生成的模拟销售面板，不能当成真实医药市场的结论。"
    )


def sidebar_filters(frame: pd.DataFrame) -> tuple[str, tuple[str, ...], tuple[str, ...], str, str]:
    page = st.sidebar.radio("页面", PAGES)
    products = tuple(
        st.sidebar.multiselect(
            "产品",
            sorted(frame["product"].unique()),
            default=sorted(frame["product"].unique()),
        )
    )
    regions = tuple(
        st.sidebar.multiselect(
            "地区",
            sorted(frame["region"].unique()),
            default=sorted(frame["region"].unique()),
        )
    )
    months = sorted(frame["month"].unique())
    month_start, month_end = months[0], months[-1]
    if months:
        month_start, month_end = st.sidebar.select_slider(
            "月份范围",
            options=months,
            value=(months[0], months[-1]),
        )
    st.sidebar.caption("环比为 0 表示上一个自然月存在且销售额相同。空白表示无法计算，不是零增长。")
    st.sidebar.caption("月份范围只改变趋势、增长率和对比图的显示，不改变预测使用的历史。")
    st.sidebar.caption("预测模型由验证集 MAE 与 RMSE 的平均名次选择。最终测试集不参与选择。页面不展示预测区间。")
    return page, products, regions, str(month_start), str(month_end)


def _growth_table(monthly: pd.DataFrame) -> pd.DataFrame:
    view = monthly[
        ["month", "sales_amount", "sales_volume", "mom_sales_amount_pct", "yoy_sales_amount_pct"]
    ].copy()
    for column in ("mom_sales_amount_pct", "yoy_sales_amount_pct"):
        view[column] = view[column].map(lambda value: "无法计算" if pd.isna(value) else f"{value:.2f}")
    return view


def page_trend(frame: pd.DataFrame) -> None:
    st.title("销售趋势")
    banner(frame)
    st.caption(METRIC_NOTES)
    monthly = monthly_total(frame)
    if monthly.empty:
        st.warning("当前筛选下没有数据。")
        return
    st.caption(
        f"时间范围 {monthly['month'].min()} 至 {monthly['month'].max()}。"
        "图中的销售额和销量是筛选范围内各产品、各地区的月度合计。"
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("月数", f"{monthly['month'].nunique()}")
    c2.metric("销售额合计", f"{frame['sales_amount'].sum():,.0f}")
    c3.metric("销量合计", f"{frame['sales_volume'].sum():,.0f}")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=monthly["month"],
            y=monthly["sales_amount"],
            mode="lines+markers",
            name="销售额",
            line={"color": ACCENT, "width": 2},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=monthly["month"],
            y=monthly["sales_volume"],
            mode="lines",
            name="销量",
            yaxis="y2",
            line={"color": BLUE, "width": 2},
        )
    )
    fig.update_layout(
        yaxis2={
            "overlaying": "y",
            "side": "right",
            "title": {"text": "销量", "standoff": 12},
            "showgrid": False,
            "automargin": True,
        }
    )
    st.plotly_chart(chart_layout(fig, "月度销售额与销量"), width="stretch")
    st.dataframe(_growth_table(monthly), width="stretch", hide_index=True)


def page_growth(frame: pd.DataFrame) -> None:
    st.title("增长率")
    banner(frame)
    st.caption(
        "环比 = (本月销售额 / 上一自然月销售额 - 1) × 100。"
        "同比 = (本月销售额 / 去年同月销售额 - 1) × 100。"
        "对比月缺失或分母为 0 时结果为“无法计算”，不会把缺失月当成 0，也不会把无法计算显示成 0% 增长。"
    )
    monthly = monthly_total(frame)
    if monthly.empty:
        st.warning("当前筛选下没有数据。")
        return
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=monthly["month"],
            y=monthly["mom_sales_amount_pct"],
            mode="lines+markers",
            name="销售额环比 %",
            connectgaps=False,
            line={"color": ACCENT},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=monthly["month"],
            y=monthly["yoy_sales_amount_pct"],
            mode="lines+markers",
            name="销售额同比 %",
            connectgaps=False,
            line={"color": AMBER},
        )
    )
    st.plotly_chart(chart_layout(fig, "筛选合计的销售额增长率"), width="stretch")
    st.dataframe(_growth_table(monthly), width="stretch", hide_index=True)
    ranking = product_cagr(frame)
    st.subheader("产品 CAGR")
    st.caption(
        "CAGR 是当前筛选时间范围和地区内，各产品销售额合计从首个出现月份到最后出现月份的年化增长率。"
        "它不是固定自然年度的增长率，也只描述这条模拟序列。"
    )
    if ranking.empty:
        st.write("序列太短，无法计算 CAGR。")
    else:
        bar = go.Figure(
            go.Bar(x=ranking["product"], y=ranking["cagr_pct"], marker_color=ACCENT, name="CAGR %")
        )
        st.plotly_chart(chart_layout(bar, "产品销售额 CAGR"), width="stretch")
        st.dataframe(ranking, width="stretch", hide_index=True)


def page_comparison(frame: pd.DataFrame) -> None:
    st.title("产品与地区对比")
    banner(frame)
    st.caption(METRIC_NOTES)
    if frame.empty:
        st.warning("当前筛选下没有数据。")
        return
    latest = frame["month"].max()
    current = frame[frame["month"] == latest].sort_values(["region", "product"])
    st.caption(
        f"对比月份是当前筛选中的最新月 {latest}。"
        "销售额占比的分母是同一月份、同一地区在整张分析表中的全部产品销售额，包含当前未选中的产品。"
        "因此只筛选部分产品时，页面上的占比不必加总为 100%。"
        "环比和同比与增长率页使用同一口径。"
    )
    amount = current.pivot_table(index="product", columns="region", values="sales_amount", aggfunc="sum")
    volume = current.pivot_table(index="product", columns="region", values="sales_volume", aggfunc="sum")
    amount_fig = go.Figure()
    volume_fig = go.Figure()
    for region in amount.columns:
        amount_fig.add_trace(go.Bar(name=str(region), x=amount.index, y=amount[region]))
        volume_fig.add_trace(go.Bar(name=str(region), x=volume.index, y=volume[region]))
    amount_fig.update_layout(barmode="group")
    volume_fig.update_layout(barmode="group")
    st.plotly_chart(chart_layout(amount_fig, f"{latest} 销售额"), width="stretch")
    st.plotly_chart(chart_layout(volume_fig, f"{latest} 销量"), width="stretch")
    share = current.pivot_table(index="product", columns="region", values="sales_share_pct", aggfunc="sum")
    share_fig = go.Figure()
    for region in share.columns:
        share_fig.add_trace(go.Bar(name=str(region), x=share.index, y=share[region]))
    share_fig.update_layout(barmode="group")
    st.plotly_chart(chart_layout(share_fig, f"{latest} 销售额占比 %"), width="stretch")
    columns = [
        "month",
        "product",
        "region",
        "sales_amount",
        "sales_volume",
        "mom_sales_amount_pct",
        "yoy_sales_amount_pct",
        "sales_share_pct",
    ]
    st.dataframe(current[columns], width="stretch", hide_index=True)


GRAIN_OPTIONS = {
    GRAIN_TOTAL: "总量（所选产品与地区的月度合计）",
    GRAIN_PRODUCT: "按产品（在所选地区内分别合计）",
    GRAIN_REGION: "按地区（在所选产品内分别合计）",
    GRAIN_PRODUCT_REGION: "按产品×地区",
}


def _series_summary(item: dict) -> dict:
    test_metrics = item.get("test_metrics") or {}
    selected = item.get("selected_model")
    validation = (item.get("validation_summary") or {}).get(selected) or {}
    return {
        "粒度": item.get("forecast_grain"),
        "序列": item.get("series_label"),
        "产品": item.get("product") or "—",
        "地区": item.get("region") or "—",
        "状态": item.get("status"),
        "验证折数": item.get("n_validation_folds"),
        "验证期": (
            f"{item.get('validation_start') or '—'} 至 {item.get('validation_end') or '—'}"
        ),
        "最终测试期": f"{item.get('test_start') or '—'} 至 {item.get('test_end') or '—'}",
        "入选模型": selected or "—",
        "验证 MAE": validation.get("mae_mean"),
        "验证 RMSE": validation.get("rmse_mean"),
        "测试 MAE": test_metrics.get("mae"),
        "测试 RMSE": test_metrics.get("rmse"),
    }


def _plot_one_series(item: dict) -> None:
    history = pd.DataFrame(item.get("history") or [])
    detail = pd.DataFrame(item.get("test_detail") or [])
    forward = pd.DataFrame(item.get("forward") or [])
    fig = go.Figure()
    if not history.empty:
        fig.add_trace(
            go.Scatter(
                x=history["month"],
                y=history["sales_amount"],
                name="该序列历史销售额",
                line={"color": ACCENT},
            )
        )
    selected = item.get("selected_model")
    if selected and not detail.empty and selected in detail.columns:
        fig.add_trace(
            go.Scatter(
                x=detail["month"],
                y=detail[selected],
                name="最终测试期：入选模型",
                line={"dash": "dot", "color": BLUE},
            )
        )
    if not forward.empty:
        fig.add_trace(
            go.Scatter(
                x=forward["month"],
                y=forward["forecast"],
                name="未来 6 个月点预测",
                line={"dash": "dash", "color": "#9C3B3B"},
            )
        )
    if fig.data:
        st.plotly_chart(
            chart_layout(fig, item.get("series_label") or "单条序列"),
            width="stretch",
        )


def page_forecast(frame: pd.DataFrame, products: tuple[str, ...], regions: tuple[str, ...]) -> None:
    st.title("预测与模型评估")
    banner(frame)
    st.caption(
        "预测使用所选产品和地区的完整历史，不受上方月份滑块影响。"
        "每条序列单独做滚动验证和模型选择。验证集的 MAE、RMSE 用于选择；"
        "最终测试集只评估已经选定的模型，不参与选择。"
        "MAPE 只描述最终测试集，实际值为 0 的月份会排除，销售额较小时 MAPE 可能被放大。"
        "缺失月份不补 0。本页不展示预测区间。"
        "这些结果描述 synthetic_demo 模拟序列，不是真实医药销售预测。"
    )
    grain = st.radio("预测粒度", list(GRAIN_OPTIONS), format_func=lambda key: GRAIN_OPTIONS[key], horizontal=True)
    if frame.empty:
        st.warning("当前筛选下没有数据。")
        return
    grouped = forecast_selection(products, regions, grain)
    st.subheader(f"当前粒度：{GRAIN_OPTIONS.get(grouped.get('forecast_grain'), grain)}")
    series = grouped.get("series") or []
    if not series:
        st.warning("没有可评估的序列。")
        return
    st.dataframe(pd.DataFrame([_series_summary(item) for item in series]), width="stretch", hide_index=True)
    labels = [item.get("series_label") or item.get("series_id") for item in series]
    chosen_label = labels[0] if len(labels) == 1 else st.selectbox("查看一条序列", labels)
    item = next(row for row in series if (row.get("series_label") or row.get("series_id")) == chosen_label)
    st.markdown(
        f"**预测对象：** {item.get('series_label')}。"
        f" 产品 = {item.get('product') or '所选范围合计'}，"
        f" 地区 = {item.get('region') or '所选范围合计'}。"
    )
    for warning in item.get("warnings") or []:
        st.warning(warning)
    st.write(item.get("reason") or "")
    if item.get("status") != "ok":
        st.warning("这条序列没有形成可用的滚动验证，因此没有用最终测试集选择模型，也不展示向前预测。")
        _plot_one_series(item)
        return

    st.subheader("验证集指标")
    st.caption(
        f"验证期 {item.get('validation_start')} 至 {item.get('validation_end')}，"
        f"共 {item.get('n_validation_folds')} 折。下表是验证集 MAE 与 RMSE，用来计算平均名次。"
        "最终测试集没有参与这个选择。"
    )
    validation_rows = []
    for name, payload in (item.get("models") or {}).items():
        validation_rows.append(
            {
                "模型": payload.get("display_name", name),
                "状态": payload.get("status"),
                "验证 MAE": payload.get("mae"),
                "验证 RMSE": payload.get("rmse"),
                "MAE 名次": payload.get("mae_rank"),
                "RMSE 名次": payload.get("rmse_rank"),
                "平均名次": payload.get("average_rank"),
                "说明": payload.get("reason") or "",
            }
        )
    st.dataframe(pd.DataFrame(validation_rows), width="stretch", hide_index=True)

    st.subheader("最终测试集指标")
    test_metrics = item.get("test_metrics") or {}
    selected_name = (item.get("models") or {}).get(item.get("selected_model"), {}).get(
        "display_name", item.get("selected_model")
    )
    st.caption(
        f"测试期 {item.get('test_start')} 至 {item.get('test_end')}。"
        f"只报告已选模型 {selected_name} 的一次测试结果。"
        f" {test_metrics.get('mape_note') or ''}"
    )
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "模型": selected_name,
                    "测试 MAE": test_metrics.get("mae"),
                    "测试 RMSE": test_metrics.get("rmse"),
                    "测试 MAPE %": test_metrics.get("mape"),
                    "MAPE 排除的零值月": test_metrics.get("mape_excluded_zeros"),
                }
            ]
        ),
        width="stretch",
        hide_index=True,
    )
    st.success(
        f"入选模型是 {selected_name}。选择依据是验证集 MAE 名次与 RMSE 名次的平均值，不是测试集 RMSE。"
    )
    _plot_one_series(item)
    forward = pd.DataFrame(item.get("forward") or [])
    st.subheader("未来 6 个月点预测")
    st.caption("这是记录测试指标之后，用该序列全部历史重新拟合入选模型得到的点预测。它不是测试误差，也没有预测区间。")
    if forward.empty:
        st.warning("这条序列没有向前点预测。")
    else:
        st.dataframe(forward, width="stretch", hide_index=True)
    st.caption(item.get("disclaimer") or "")


def page_quality(quality: dict, frame: pd.DataFrame) -> None:
    st.title("数据质量")
    banner(frame)
    st.caption(
        "以下规则只适用于当前演示项目，不是所有真实医药销售数据的通用规则："
        "重复业务键 (month, product, region) 保留输入中的第一行；"
        "负销售额或负销量整行排除，不截成 0；"
        "缺失月份只报告，不自动补 0。"
        "筛选器不改变这份质量报告。IQR 异常值只计数，不删除、不改写。"
    )
    for note in quality["policy_notes"]:
        st.markdown(f"- {note}")
    raw = quality["raw"]
    analytical = quality["analytical"]

    def _block(title: str, profile: dict) -> None:
        st.subheader(title)
        st.metric("行数", profile["n_rows"])
        st.metric("重复业务键（month + product + region）", profile["duplicate_key_excess"])
        st.metric("负销售额行", profile["negative_sales_amount_rows"])
        st.metric("负销量行", profile["negative_sales_volume_rows"])
        st.metric("非数值销售额行", profile["non_numeric_sales_amount_rows"])
        st.metric("非数值销量行", profile["non_numeric_sales_volume_rows"])
        st.metric("非法或无法解析的月份行", profile["invalid_month_rows"])
        st.metric("起止范围内的缺失月份", profile["missing_month_count"])
        if profile["nulls_by_column"]:
            st.write("缺失值行数", profile["nulls_by_column"])

    left, right = st.columns(2)
    with left:
        _block("原始数据", raw)
    with right:
        _block("分析表", analytical)
    st.write(f"从原始表排除的行数：{quality['rows_excluded']}")
    st.write("分析表中的 IQR 异常值标记（数值未修改）")
    st.json(analytical["outlier_flags"])
    gaps = pd.DataFrame(analytical["missing_months"])
    if gaps.empty:
        st.success("分析表里，各产品-地区在自身起止月份之间没有缺失月。")
    else:
        st.dataframe(gaps, width="stretch", hide_index=True)


def main() -> None:
    apply_theme()
    try:
        frame, quality = load_sales_bundle()
    except FileNotFoundError as exc:
        st.title("医药销售分析与预测")
        st.error(str(exc))
        st.code("python scripts/generate_pharma_data.py")
        return
    page, products, regions, month_start, month_end = sidebar_filters(frame)
    if page == "数据质量":
        page_quality(quality, frame)
        return
    selected = filter_panel(frame, products, regions, month_start, month_end)
    if page == "销售趋势":
        page_trend(selected)
    elif page == "增长率":
        page_growth(selected)
    elif page == "产品与地区对比":
        page_comparison(selected)
    else:
        page_forecast(filter_panel(frame, products, regions), products, regions)


if __name__ == "__main__":
    main()
else:
    main()
