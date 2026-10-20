"""
Digital Industry Analytics Platform with AI-assisted Insights — Streamlit Dashboard

Pages:
  1. Overview Dashboard
  2. Indicator Analysis
  3. Forecasting
  4. AI Insight Assistant
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.data_service import (  # noqa: E402
    get_commodity_association,
    get_commodity_ids,
    get_commodity_panel,
    get_commodity_plot_frame,
    get_commodity_qa,
    get_commodity_view,
    get_kpis,
    get_panel,
    get_quality_monitoring,
    theme_colors,
)
from dashboard.export_utils import (  # noqa: E402
    ensure_screenshots_dir,
    export_dataframe_csv,
    export_dataframe_png,
    export_plotly_png,
    portfolio_shot_checklist,
)
from src.analytics import descriptive_statistics, trend_analysis  # noqa: E402
from src.config import load_settings  # noqa: E402
from src.forecasting import forecast_to_frame  # noqa: E402
from src.indicator_engine import ranking_table  # noqa: E402

# ---------------------------------------------------------------------------
# Page config & styling
# ---------------------------------------------------------------------------

settings = load_settings()
COLORS = theme_colors()

st.set_page_config(
    page_title=settings["dashboard"]["page_title"],
    page_icon="◈",
    layout=settings["dashboard"].get("layout", "wide"),
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = f"""
<style>
    .block-container {{ padding-top: 1.5rem; padding-bottom: 2rem; }}
    h1, h2, h3 {{ color: {COLORS['primary']}; letter-spacing: -0.02em; }}
    div[data-testid="stMetric"] {{
        background: linear-gradient(145deg, #ffffff 0%, #eef4f8 100%);
        border: 1px solid #d5e2ea;
        border-radius: 10px;
        padding: 12px 14px;
    }}
    .insight-box {{
        background: linear-gradient(135deg, #0B3D5C 0%, #1F7A8C 100%);
        color: #F7F9FB;
        padding: 1.25rem 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
    }}
    .sidebar-caption {{
        color: #6B7280;
        font-size: 0.85rem;
        margin-top: -0.5rem;
        margin-bottom: 1rem;
    }}
    .dq-banner {{
        background: linear-gradient(135deg, #eef4f8 0%, #f7f9fb 100%);
        border: 1px solid #d5e2ea;
        border-left: 4px solid {COLORS['primary']};
        border-radius: 10px;
        padding: 0.85rem 1.1rem;
        margin: 0.25rem 0 1rem 0;
        color: #374151;
        font-size: 0.92rem;
    }}
    .dq-status-pass {{ color: #157A4C; font-weight: 600; }}
    .dq-status-warning {{ color: #B45309; font-weight: 600; }}
    .dq-status-fail {{ color: #B91C1C; font-weight: 600; }}
    .dq-compact {{
        background: #eef4f8;
        border: 1px solid #d5e2ea;
        border-radius: 8px;
        padding: 0.65rem 1rem;
        margin: 0.2rem 0 1rem 0;
        color: #1f2937;
        font-size: 0.95rem;
    }}
    .exec-note {{
        color: #374151;
        font-size: 0.95rem;
        margin: 0.15rem 0 0.75rem 0;
        padding-left: 0.75rem;
        border-left: 3px solid {COLORS['accent']};
    }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Business-facing labels (presentation only — does not alter analytics)
METRIC_LABELS: dict[str, str] = {
    "digital_index": "Digital Index",
    "technology_investment": "Technology Investment (USD bn)",
    "revenue_growth": "Revenue Growth (%)",
    "employment": "Employment (thousands)",
    "rd_ratio": "R&D Ratio (%)",
    "digital_development_index": "Digital Development Index",
    "employment_growth": "Employment Growth (%)",
}

PRIMARY_KPI_ORDER = [
    "digital_index",
    "technology_investment",
    "revenue_growth",
    "digital_development_index",
]

SECONDARY_KPI_ORDER = ["rd_ratio", "employment"]

DEFAULT_INDUSTRIES = [
    "Cloud & Data Centers",
    "FinTech",
    "Software & IT Services",
    "E-Commerce",
    "Electronics Manufacturing",
]

DEFAULT_REGIONS = [
    "North America",
    "Europe",
    "East Asia",
    "Southeast Asia",
]


def metric_label(column: str) -> str:
    """Map raw column names to business labels."""
    return METRIC_LABELS.get(column, column.replace("_", " ").title())


def apply_plotly_layout(
    fig: go.Figure,
    title: str = "",
    x_title: str | None = None,
    y_title: str | None = None,
) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=COLORS["primary"])),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(247,249,251,0.6)",
        font=dict(family="Segoe UI, Helvetica, Arial, sans-serif", color="#1f2937"),
        margin=dict(l=50, r=24, t=56, b=48),
        legend=dict(orientation="h", yanchor="bottom", y=1.08, x=0),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e5e7eb", title_text=x_title or "")
    fig.update_yaxes(showgrid=True, gridcolor="#e5e7eb", title_text=y_title or "")
    return fig


def exec_note(text: str) -> None:
    """Short executive interpretation above a chart."""
    st.markdown(f"<p class='exec-note'>{text}</p>", unsafe_allow_html=True)


def filtered_frame() -> pd.DataFrame:
    df = get_panel()
    mask = (
        df["year"].between(year_range[0], year_range[1])
        & df["industry"].isin(selected_industries)
        & df["region"].isin(selected_regions)
    )
    return df.loc[mask].copy()


def _export_portfolio_assets_from_ui() -> list[Path]:
    """
    Presentation-only export of representative charts/tables.
    Does not alter analytics results.
    """
    ensure_screenshots_dir()
    df = filtered_frame()
    if df.empty:
        df = get_panel()

    written: list[Path] = []

    trend = (
        df.groupby("year", as_index=False)[
            ["digital_index", "technology_investment", "revenue_growth"]
        ]
        .mean()
        .melt(id_vars="year", var_name="indicator", value_name="value")
    )
    fig_trend = px.line(
        trend,
        x="year",
        y="value",
        color="indicator",
        markers=True,
        color_discrete_sequence=[COLORS["primary"], COLORS["accent"], COLORS["warm"]],
    )
    apply_plotly_layout(fig_trend, "Industry-wide indicator trends")
    written.append(export_plotly_png(fig_trend, "chart_overview_trends.png"))

    corr_cols = [
        c
        for c in [
            "digital_index",
            "technology_investment",
            "revenue_growth",
            "rd_ratio",
            "digital_development_index",
        ]
        if c in df.columns
    ]
    corr = df[corr_cols].corr().round(3)
    fig_corr = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="Tealgrn",
        aspect="auto",
        zmin=-1,
        zmax=1,
    )
    apply_plotly_layout(fig_corr, "Indicator correlation heatmap")
    written.append(export_plotly_png(fig_corr, "chart_correlation_heatmap.png"))

    from src.forecasting import run_forecast

    fc = run_forecast(df, metric="digital_index", method="ensemble", horizon=5)
    plot_df = forecast_to_frame(fc)
    fig_fc = go.Figure()
    hist = plot_df[plot_df["series"] == "Historical"]
    fut = plot_df[plot_df["series"] == "Forecast"]
    fig_fc.add_trace(
        go.Scatter(
            x=hist["year"],
            y=hist["value"],
            mode="lines+markers",
            name="Historical",
            line=dict(color=COLORS["primary"], width=3),
        )
    )
    fig_fc.add_trace(
        go.Scatter(
            x=fut["year"],
            y=fut["value"],
            mode="lines+markers",
            name="Forecast",
            line=dict(color=COLORS["warm"], width=3, dash="dash"),
        )
    )
    apply_plotly_layout(fig_fc, "Digital index forecast")
    written.append(export_plotly_png(fig_fc, "chart_forecast_digital_index.png"))

    latest = int(df["year"].max())
    kpis = get_kpis(latest)
    written.append(
        export_dataframe_png(kpis, "table_kpi_summary.png", title="KPI summary")
    )
    written.append(export_dataframe_csv(kpis, "table_kpi_summary.csv"))

    dq = get_quality_monitoring()["primary"]
    dq_table = pd.DataFrame(
        [
            {"metric": "total_records", "value": dq["total_records"]},
            {"metric": "missing_value_pct", "value": dq["missing_value_pct"]},
            {"metric": "duplicate_records", "value": dq["duplicate_records"]},
            {"metric": "validation_status", "value": dq["validation_status"]},
            {"metric": "data_quality_score", "value": dq["data_quality_score"]},
        ]
    )
    written.append(
        export_dataframe_png(
            dq_table,
            "table_data_quality_summary.png",
            title="Data quality monitoring summary",
        )
    )
    written.append(export_dataframe_csv(dq_table, "table_data_quality_summary.csv"))
    written.append(
        export_dataframe_csv(portfolio_shot_checklist(), "portfolio_screenshot_checklist.csv")
    )
    return written


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## Digital Industry Analytics")
    st.markdown(
        "<p class='sidebar-caption'>Platform with AI-assisted insights</p>",
        unsafe_allow_html=True,
    )
    page = st.radio(
        "Navigate",
        [
            "Overview Dashboard",
            "Data Quality Monitoring",
            "Indicator Analysis",
            "Forecasting",
            "Commodity Market Analytics",
            "AI Insight Assistant",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    df_all = get_panel()
    years = sorted(df_all["year"].unique())
    industries = sorted(df_all["industry"].unique())
    regions = sorted(df_all["region"].unique())

    st.caption("Global filters")
    year_range = st.slider(
        "Year range",
        min_value=int(min(years)),
        max_value=int(max(years)),
        value=(int(min(years)), int(max(years))),
    )
    industry_defaults = [i for i in DEFAULT_INDUSTRIES if i in industries] or industries[:5]
    region_defaults = [r for r in DEFAULT_REGIONS if r in regions] or regions[:4]
    selected_industries = st.multiselect("Industries", industries, default=industry_defaults)
    selected_regions = st.multiselect("Regions", regions, default=region_defaults)
    st.divider()

    with st.expander("Portfolio export", expanded=False):
        st.caption(
            "Export key charts/tables to `reports/screenshots/` for GitHub. "
            "Full-page README shots are still captured in the browser."
        )
        checklist = portfolio_shot_checklist()
        captured = int(checklist["captured"].sum())
        st.write(f"Full-page shots ready: **{captured}/{len(checklist)}**")
        if st.button("Export key charts & tables", use_container_width=True):
            try:
                paths = _export_portfolio_assets_from_ui()
                st.success(f"Exported {len(paths)} files to reports/screenshots/")
                for p in paths:
                    st.caption(f"• {p.name}")
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))
        st.dataframe(
            checklist[["filename", "captured"]],
            use_container_width=True,
            hide_index=True,
        )

    st.divider()
    st.caption(f"v{settings['project']['version']} · SQLite + Streamlit + LLM")


def _status_class(status: str) -> str:
    mapping = {
        "Pass": "dq-status-pass",
        "Warning": "dq-status-warning",
        "Fail": "dq-status-fail",
    }
    return mapping.get(status, "dq-status-warning")


def render_data_quality_compact() -> dict:
    """One-line readiness summary for business pages (presentation only)."""
    monitoring = get_quality_monitoring()
    primary = monitoring["primary"]
    status = str(primary["validation_status"]).upper()
    score = primary["data_quality_score"]
    status_cls = _status_class(primary["validation_status"])
    st.markdown(
        f"""
        <div class="dq-compact">
            <strong>Data Ready</strong>
            &nbsp;|&nbsp; Quality Score: <strong>{score:.1f}</strong>
            &nbsp;|&nbsp; Status:
            <span class="{status_cls}">{status}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return monitoring


def render_data_quality_section(compact: bool = True) -> dict:
    """
    Enterprise data quality gate — full detail for the Data Quality page.
    """
    monitoring = get_quality_monitoring()
    primary = monitoring["primary"]
    status = primary["validation_status"]
    status_cls = _status_class(status)

    if compact:
        return render_data_quality_compact()

    st.subheader("Data Quality Monitoring")
    st.markdown(
        f"""
        <div class="dq-banner">
            Analytical readiness gate for the enterprise data platform.
            Metrics below reflect the <strong>analytical store</strong> after
            multi-source integration, validation, and preparation.
            Validation status:
            <span class="{status_cls}">{status}</span>
            · Quality score improvement vs source intake:
            <strong>{monitoring['score_improvement']:+.1f}</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total records", f"{primary['total_records']:,}")
    c2.metric("Missing value %", f"{primary['missing_value_pct']:.2f}%")
    c3.metric("Duplicate records", f"{primary['duplicate_records']:,}")
    c4.metric("Validation status", status)
    c5.metric(
        "Data quality score",
        f"{primary['data_quality_score']:.1f}",
        help="Composite score (0–100) from missingness, duplicates, schema checks, range rules, and outliers.",
    )

    left, right = st.columns(2)
    with left:
        st.markdown("**Source intake (pre-clean)**")
        src = monitoring["source_intake"]
        st.dataframe(
            pd.DataFrame(
                [
                    {"metric": "Total records", "value": src["total_records"]},
                    {"metric": "Missing value %", "value": src["missing_value_pct"]},
                    {"metric": "Duplicate records", "value": src["duplicate_records"]},
                    {"metric": "Validation status", "value": src["validation_status"]},
                    {"metric": "Data quality score", "value": src["data_quality_score"]},
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
    with right:
        st.markdown("**Analytical store (post-prep)**")
        st.dataframe(
            pd.DataFrame(
                [
                    {"metric": "Total records", "value": primary["total_records"]},
                    {"metric": "Missing value %", "value": primary["missing_value_pct"]},
                    {
                        "metric": "Duplicate records",
                        "value": primary["duplicate_records"],
                    },
                    {
                        "metric": "Validation status",
                        "value": primary["validation_status"],
                    },
                    {
                        "metric": "Data quality score",
                        "value": primary["data_quality_score"],
                    },
                    {
                        "metric": "Records removed in prep",
                        "value": monitoring["records_removed"],
                    },
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )

    miss_df = pd.DataFrame(
        {
            "column": list(primary["missing_pct_by_column"].keys()),
            "missing_pct": list(primary["missing_pct_by_column"].values()),
            "missing_count": [
                primary["missing_by_column"].get(k, 0)
                for k in primary["missing_pct_by_column"]
            ],
        }
    ).sort_values("missing_pct", ascending=False)
    fig = px.bar(
        miss_df,
        x="column",
        y="missing_pct",
        color="missing_pct",
        color_continuous_scale=["#A7C4D4", COLORS["warm"]],
        labels={"missing_pct": "Missing %", "column": "Column"},
    )
    fig.update_layout(coloraxis_showscale=False)
    apply_plotly_layout(fig, "Missingness by Column (Analytical Store)")
    st.plotly_chart(fig, use_container_width=True)

    v1, v2 = st.columns(2)
    with v1:
        st.markdown("**Schema / dtype validation**")
        issues = primary.get("dtype_issues") or []
        if issues:
            for issue in issues:
                st.error(issue)
        else:
            st.success("All required columns passed type validation.")
    with v2:
        st.markdown("**Business-rule range checks**")
        viol = pd.DataFrame(
            [
                {"column": k, "violations": v}
                for k, v in (primary.get("range_violations") or {}).items()
            ]
        )
        st.dataframe(viol, use_container_width=True, hide_index=True)

    st.markdown("---")
    return monitoring


# ---------------------------------------------------------------------------
# Page 1 — Overview
# ---------------------------------------------------------------------------

def page_overview() -> None:
    st.title("Overview Dashboard")
    st.caption(
        "Executive view of digital economy performance across industries and regions."
    )

    render_data_quality_compact()

    df = filtered_frame()
    if df.empty:
        st.warning("No data for the selected filters.")
        return

    latest_year = int(df["year"].max())
    kpis = get_kpis(latest_year)
    kpi_map = {row["metric"]: row for _, row in kpis.iterrows()}

    st.subheader(f"Key performance indicators ({latest_year})")
    cols = st.columns(4)
    for i, metric in enumerate(PRIMARY_KPI_ORDER):
        row = kpi_map.get(metric)
        if row is None:
            continue
        delta = row["yoy_change_pct"]
        cols[i].metric(
            metric_label(metric),
            f"{row['value']:.1f}",
            delta=f"{delta:+.1f}% YoY" if pd.notna(delta) else None,
        )

    with st.expander("Secondary metrics & data quality details"):
        sec_cols = st.columns(2)
        for i, metric in enumerate(SECONDARY_KPI_ORDER):
            row = kpi_map.get(metric)
            if row is None:
                continue
            delta = row["yoy_change_pct"]
            sec_cols[i].metric(
                metric_label(metric),
                f"{row['value']:.1f}",
                delta=f"{delta:+.1f}% YoY" if pd.notna(delta) else None,
            )
        dq = get_quality_monitoring()["primary"]
        st.markdown("**Data quality details**")
        st.dataframe(
            pd.DataFrame(
                [
                    {"Metric": "Total records", "Value": dq["total_records"]},
                    {"Metric": "Missing value %", "Value": dq["missing_value_pct"]},
                    {"Metric": "Duplicate records", "Value": dq["duplicate_records"]},
                    {"Metric": "Validation status", "Value": dq["validation_status"]},
                    {"Metric": "Data quality score", "Value": dq["data_quality_score"]},
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.caption("For full diagnostics, open the Data Quality Monitoring page.")

    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        exec_note(
            "Digital maturity has strengthened steadily across the selected industries."
        )
        trend = (
            df.groupby("year", as_index=False)["digital_index"]
            .mean()
            .rename(columns={"digital_index": "Digital Index"})
        )
        fig = px.line(
            trend,
            x="year",
            y="Digital Index",
            markers=True,
            color_discrete_sequence=[COLORS["primary"]],
        )
        apply_plotly_layout(
            fig,
            "Digital Index Trend Over Time",
            x_title="Year",
            y_title="Digital Index (0–100)",
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        regional = (
            df[df["year"] == latest_year]
            .groupby("region", as_index=False)["digital_development_index"]
            .mean()
            .sort_values("digital_development_index", ascending=True)
        )
        top_region = regional.iloc[-1]["region"] if len(regional) else "N/A"
        exec_note(
            f"{top_region} leads on Digital Development Index among selected regions in {latest_year}."
        )
        fig = px.bar(
            regional,
            x="digital_development_index",
            y="region",
            orientation="h",
            color="digital_development_index",
            color_continuous_scale=["#A7C4D4", COLORS["primary"]],
            labels={
                "digital_development_index": "Digital Development Index",
                "region": "Region",
            },
        )
        fig.update_layout(coloraxis_showscale=False)
        apply_plotly_layout(
            fig,
            f"Regional Digital Development Comparison ({latest_year})",
            x_title="Digital Development Index",
            y_title="",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Industry trajectories")
    exec_note("Most industries show continued digital maturity gains over the decade.")
    ind_trend = trend_analysis(df, metric="digital_index", group_col="industry")
    # Keep chart readable when many industries are selected
    latest_means = (
        ind_trend.sort_values("year")
        .groupby("industry")["digital_index"]
        .last()
        .sort_values(ascending=False)
    )
    focus = latest_means.head(5).index.tolist()
    plot_ind = ind_trend[ind_trend["industry"].isin(focus)].copy()
    fig = px.line(
        plot_ind,
        x="year",
        y="digital_index",
        color="industry",
        markers=True,
        labels={"digital_index": "Digital Index", "year": "Year", "industry": "Industry"},
    )
    apply_plotly_layout(
        fig,
        "Digital Index Trajectories by Industry",
        x_title="Year",
        y_title="Digital Index (0–100)",
    )
    st.plotly_chart(fig, use_container_width=True)
    if len(latest_means) > 5:
        st.caption("Showing top 5 industries by latest Digital Index within current filters.")

    with st.expander("Latest-year industry snapshot"):
        snap = (
            df[df["year"] == latest_year]
            .groupby("industry", as_index=False)
            .agg(
                **{
                    "Digital Index": ("digital_index", "mean"),
                    "Technology Investment (USD bn)": ("technology_investment", "mean"),
                    "Revenue Growth (%)": ("revenue_growth", "mean"),
                    "Digital Development Index": ("digital_development_index", "mean"),
                }
            )
            .round(2)
            .sort_values("Digital Development Index", ascending=False)
        )
        st.dataframe(snap, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Page — Data Quality Monitoring
# ---------------------------------------------------------------------------

def page_data_quality() -> None:
    st.title("Data Quality Monitoring")
    st.caption(
        "Enterprise analytical data quality management — validate readiness before analysis and decision support."
    )
    render_data_quality_section(compact=False)

    st.subheader("Quality management controls")
    st.markdown(
        """
        | Control | Purpose |
        |---------|---------|
        | Missing value detection | Quantify incompleteness before imputation |
        | Duplicate detection | Enforce unique `year × industry × region` keys |
        | Data type validation | Protect statistical engines from schema drift |
        | Range / business rules | Keep indicators inside enterprise-valid bounds |
        | Quality score | Single readiness KPI for analytical release gates |
        """
    )


# ---------------------------------------------------------------------------
# Page 2 — Indicator Analysis
# ---------------------------------------------------------------------------

def page_indicators() -> None:
    st.title("Indicator Analysis")
    st.caption(
        "Compare digital leaders, explore relationships, and identify growth drivers."
    )

    render_data_quality_compact()

    df = filtered_frame()
    if df.empty:
        st.warning("No data for the selected filters.")
        return

    metric_keys = [
        c
        for c in [
            "digital_development_index",
            "digital_index",
            "technology_investment",
            "revenue_growth",
            "rd_ratio",
            "employment",
            "employment_growth",
        ]
        if c in df.columns
    ]
    metric_display = {metric_label(k): k for k in metric_keys}

    c1, c2, c3 = st.columns(3)
    metric_name = c1.selectbox("Metric", list(metric_display.keys()), index=0)
    metric = metric_display[metric_name]
    group_label = c2.selectbox("Rank by", ["Industry", "Region"])
    group_by = group_label.lower()
    year = c3.selectbox("Year", sorted(df["year"].unique(), reverse=True), index=0)

    ranking_title = (
        "Digital Development Leaders Ranking"
        if metric == "digital_development_index"
        else f"{metric_name} Leaders Ranking"
    )
    exec_note(f"Ranking highlights relative performance by {group_label.lower()} in {int(year)}.")
    ranked = ranking_table(df, metric=metric, year=int(year), group_by=group_by, top_n=15)
    ranked_plot = ranked.rename(columns={metric: metric_name, group_by: group_label})
    fig = px.bar(
        ranked_plot,
        x=metric_name,
        y=group_label,
        orientation="h",
        text="rank",
        color=metric_name,
        color_continuous_scale=["#D6EAF0", COLORS["accent"]],
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
    apply_plotly_layout(
        fig,
        f"{ranking_title} ({year})",
        x_title=metric_name,
        y_title="",
    )
    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Distribution by industry")
        fig = px.box(
            df[df["year"] == int(year)],
            x="industry",
            y=metric,
            color_discrete_sequence=[COLORS["primary"]],
            labels={"industry": "Industry", metric: metric_name},
        )
        fig.update_layout(xaxis_tickangle=-30)
        apply_plotly_layout(
            fig,
            f"{metric_name} Distribution by Industry",
            x_title="Industry",
            y_title=metric_name,
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Investment vs growth")
        exec_note(
            "Technology investment shows the strongest visual association with revenue growth."
        )
        sample = df[df["year"] == int(year)].rename(
            columns={
                "technology_investment": "Technology Investment (USD bn)",
                "revenue_growth": "Revenue Growth (%)",
                "industry": "Industry",
                "digital_index": "Digital Index",
            }
        )
        fig = px.scatter(
            sample,
            x="Technology Investment (USD bn)",
            y="Revenue Growth (%)",
            color="Industry",
            size="Digital Index",
            hover_data=["region"] if "region" in sample.columns else None,
            trendline="ols",
        )
        fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.25, x=0))
        apply_plotly_layout(
            fig,
            f"Technology Investment vs Revenue Growth ({year})",
            x_title="Technology Investment (USD bn)",
            y_title="Revenue Growth (%)",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Indicator relationships")
    exec_note("Darker cells indicate stronger co-movement between digital performance metrics.")
    corr_cols = [
        c
        for c in [
            "digital_index",
            "technology_investment",
            "revenue_growth",
            "employment",
            "rd_ratio",
            "digital_development_index",
            "employment_growth",
        ]
        if c in df.columns
    ]
    corr = df[corr_cols].corr().round(2)
    corr_display = corr.rename(index=metric_label, columns=metric_label)
    fig = px.imshow(
        corr_display,
        text_auto=True,
        color_continuous_scale="Tealgrn",
        aspect="auto",
        zmin=-1,
        zmax=1,
    )
    apply_plotly_layout(fig, "How Digital Indicators Move Together")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("What drives revenue growth?")
    try:
        reg = linear_regression_on(df)
        top_driver = None
        if reg.get("significant_features"):
            top_driver = max(
                reg["significant_features"],
                key=lambda f: abs(reg["coefficients"].get(f, 0)),
            )
        if top_driver:
            exec_note(
                f"{metric_label(top_driver)} shows the strongest association with revenue growth "
                f"(model R² = {reg['r_squared']:.1%})."
            )
        else:
            exec_note(reg.get("interpretation", "Regression results for revenue growth drivers."))

        coef_df = pd.DataFrame(
            {
                "Driver": [metric_label(k) for k in reg["coefficients"]],
                "Coefficient": list(reg["coefficients"].values()),
                "P-value": [reg["p_values"][k] for k in reg["coefficients"]],
            }
        )
        fig = px.bar(
            coef_df,
            x="Driver",
            y="Coefficient",
            color="P-value",
            color_continuous_scale=["#1F7A8C", "#E5E7EB"],
        )
        apply_plotly_layout(
            fig,
            "Relative Strength of Growth Drivers",
            x_title="Driver",
            y_title="Standardized coefficient",
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Model diagnostics (R², RMSE, coefficients)"):
            m1, m2, m3 = st.columns(3)
            m1.metric("R²", f"{reg['r_squared']:.3f}")
            m2.metric("Adj. R²", f"{reg['adj_r_squared']:.3f}")
            m3.metric("RMSE", f"{reg['rmse']:.3f}")
            st.caption(reg["interpretation"])
            st.dataframe(coef_df.round(4), use_container_width=True, hide_index=True)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Regression unavailable: {exc}")

    with st.expander("Descriptive statistics"):
        desc = descriptive_statistics(df, columns=corr_cols)
        desc.index = [metric_label(str(i)) for i in desc.index]
        st.dataframe(desc, use_container_width=True)


def linear_regression_on(df: pd.DataFrame) -> dict:
    from src.analytics import linear_regression_analysis

    return linear_regression_analysis(df).to_dict()


# ---------------------------------------------------------------------------
# Page 3 — Forecasting
# ---------------------------------------------------------------------------

def page_forecasting() -> None:
    st.title("Forecasting")
    st.caption(
        "Project future digital economy indicator paths to support planning decisions."
    )

    df = filtered_frame()
    if df.empty:
        st.warning("No data for the selected filters.")
        return

    metric_keys = [
        c
        for c in [
            "digital_index",
            "technology_investment",
            "revenue_growth",
            "digital_development_index",
            "rd_ratio",
        ]
        if c in df.columns
    ]
    metric_display = {metric_label(k): k for k in metric_keys}
    method_display = {
        "Ensemble (recommended)": "ensemble",
        "Linear trend": "linear_trend",
        "ARIMA": "arima",
    }

    c1, c2, c3, c4 = st.columns(4)
    metric_name = c1.selectbox("Forecast metric", list(metric_display.keys()), index=0)
    metric = metric_display[metric_name]
    method_name = c2.selectbox("Method", list(method_display.keys()), index=0)
    method = method_display[method_name]
    horizon = c3.slider("Horizon (years)", 2, 8, 5)
    scope = c4.selectbox("Scope", ["All (aggregate)", "By industry", "By region"])

    industry = region = None
    if scope == "By industry":
        industry = st.selectbox("Industry", sorted(df["industry"].unique()))
    elif scope == "By region":
        region = st.selectbox("Region", sorted(df["region"].unique()))

    try:
        from src.forecasting import run_forecast

        result = run_forecast(
            df,
            metric=metric,
            method=method,  # type: ignore[arg-type]
            horizon=horizon,
            industry=industry,
            region=region,
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"Forecast failed: {exc}")
        return

    end_year = int(result.forecast["year"].max()) if len(result.forecast) else None
    end_value = (
        float(result.forecast["forecast"].iloc[-1]) if len(result.forecast) else None
    )
    if end_year is not None and end_value is not None:
        st.success(
            f"**Outlook:** {metric_name} projected to reach approximately "
            f"**{end_value:.1f}** by **{end_year}**."
        )
        exec_note(
            "Digital maturity is projected to continue increasing through the forecast horizon."
            if metric == "digital_index"
            else f"{metric_name} is projected to continue evolving through {end_year}."
        )

    plot_df = forecast_to_frame(result)
    fig = go.Figure()
    hist = plot_df[plot_df["series"] == "Historical"]
    fut = plot_df[plot_df["series"] == "Forecast"]

    fig.add_trace(
        go.Scatter(
            x=hist["year"],
            y=hist["value"],
            mode="lines+markers",
            name="Historical",
            line=dict(color=COLORS["primary"], width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fut["year"],
            y=fut["value"],
            mode="lines+markers",
            name="Forecast",
            line=dict(color=COLORS["warm"], width=3, dash="dash"),
        )
    )
    band = result.forecast
    fig.add_trace(
        go.Scatter(
            x=list(band["year"]) + list(band["year"][::-1]),
            y=list(band["upper"]) + list(band["lower"][::-1]),
            fill="toself",
            fillcolor="rgba(193, 122, 58, 0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            name="95% interval",
            hoverinfo="skip",
        )
    )
    outlook_title = (
        f"Digital Index Outlook to {end_year}"
        if metric == "digital_index" and end_year
        else f"{metric_name} Outlook to {end_year}"
    )
    apply_plotly_layout(
        fig,
        outlook_title,
        x_title="Year",
        y_title=metric_name,
    )
    fig.update_layout(
        margin=dict(t=72, b=48, l=50, r=24),
        legend=dict(orientation="h", yanchor="top", y=-0.18, x=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Model diagnostics (R², AIC, RMSE)"):
        mcols = st.columns(max(len(result.metrics), 1))
        for i, (k, v) in enumerate(result.metrics.items()):
            label = {
                "r_squared": "Linear R²",
                "linear_r_squared": "Linear R²",
                "aic": "ARIMA AIC",
                "arima_aic": "ARIMA AIC",
                "bic": "ARIMA BIC",
                "rmse": "RMSE",
                "slope_per_year": "Slope / year",
                "p_value": "Trend p-value",
            }.get(k, k.replace("_", " ").title())
            mcols[i % len(mcols)].metric(
                label,
                f"{v:.4g}" if isinstance(v, float) else v,
            )
        st.caption(result.notes)

    with st.expander("Historical and forecast data tables"):
        left, right = st.columns(2)
        with left:
            st.markdown("**Historical**")
            st.dataframe(result.history, use_container_width=True, hide_index=True)
        with right:
            st.markdown("**Forecast**")
            st.dataframe(result.forecast, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Page 4 — AI Insight Assistant
# ---------------------------------------------------------------------------

def page_commodity() -> None:
    st.title("Commodity Market Analytics")
    st.caption(
        "Crude oil price and inventory analytics from the SQLite analytical store "
        "(Phase 1). Default demo rows are synthetic — not live market feeds."
    )

    panel = get_commodity_panel()
    if panel.empty:
        st.warning(
            "No commodity_market_series data found. Run "
            "`python scripts/run_pipeline.py` after generating the synthetic crude panel."
        )
        return

    ids = get_commodity_ids()
    if not ids:
        st.warning("commodity_market_series has no commodity_id values.")
        return

    default_ix = ids.index("WTI_CRUDE") if "WTI_CRUDE" in ids else 0
    commodity_id = st.selectbox("Commodity", ids, index=default_ix)
    view = get_commodity_view(commodity_id)
    qa = get_commodity_qa(commodity_id)

    st.info(
        f"**Provenance:** data_type=`{qa.get('data_type')}` · "
        f"data_source=`{qa.get('data_source')}`. "
        "Do not interpret synthetic_demo rows as live EIA/ICE/NYMEX prints."
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Observations", f"{qa['n_observations']:,}")
    c2.metric("Date start", qa["date_min"] or "—")
    c3.metric("Date end", qa["date_max"] or "—")
    c4.metric("Duplicate excess rows", f"{qa['duplicate_excess_rows']}")
    c5.metric("Missing weeks (W-FRI)", f"{qa['missing_week_count']}")

    if qa["missing_week_dates"]:
        with st.expander("Missing week-ending dates (not filled as observations)"):
            st.write(", ".join(qa["missing_week_dates"]))
            st.caption(
                "Charts reindex to the weekly calendar and leave gaps as NaN so "
                "missing weeks are visible — placeholders are not real prints."
            )

    plot_df = get_commodity_plot_frame(
        commodity_id,
        value_cols=(
            "price_usd_bbl",
            "inventory_mmbbl",
            "price_change_pct",
            "inventory_change_pct",
        ),
    )
    if plot_df.empty:
        st.warning("No plottable rows for the selected commodity.")
        return

    left, right = st.columns(2)
    with left:
        exec_note("Price path uses observed weeks only; calendar gaps break the line.")
        fig = px.line(
            plot_df,
            x="date",
            y="price_usd_bbl",
            markers=True,
            labels={"date": "Date", "price_usd_bbl": "Price (USD/bbl)"},
        )
        fig.update_traces(connectgaps=False)
        apply_plotly_layout(
            fig,
            f"{commodity_id} — Price Trend",
            x_title="Date",
            y_title="Price (USD/bbl)",
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        exec_note("Inventory levels from the analytical store (mmbbl).")
        fig = px.line(
            plot_df,
            x="date",
            y="inventory_mmbbl",
            markers=True,
            labels={"date": "Date", "inventory_mmbbl": "Inventory (mmbbl)"},
        )
        fig.update_traces(connectgaps=False)
        apply_plotly_layout(
            fig,
            f"{commodity_id} — Inventory Trend",
            x_title="Date",
            y_title="Inventory (mmbbl)",
        )
        st.plotly_chart(fig, use_container_width=True)

    left2, right2 = st.columns(2)
    with left2:
        fig = px.line(
            plot_df,
            x="date",
            y="price_change_pct",
            markers=True,
            labels={"date": "Date", "price_change_pct": "Price change (%)"},
        )
        fig.update_traces(connectgaps=False)
        apply_plotly_layout(
            fig,
            f"{commodity_id} — Price Change Rate",
            x_title="Date",
            y_title="Price change (%)",
        )
        st.plotly_chart(fig, use_container_width=True)

    with right2:
        fig = px.line(
            plot_df,
            x="date",
            y="inventory_change_pct",
            markers=True,
            labels={"date": "Date", "inventory_change_pct": "Inventory change (%)"},
        )
        fig.update_traces(connectgaps=False)
        apply_plotly_layout(
            fig,
            f"{commodity_id} — Inventory Change Rate",
            x_title="Date",
            y_title="Inventory change (%)",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Price change vs inventory change")
    assoc = get_commodity_association(commodity_id)
    exec_note(assoc["disclaimer"])
    m1, m2, m3 = st.columns(3)
    m1.metric("Paired observations", f"{assoc['n_paired']}")
    m2.metric(
        "Pearson r",
        "—" if assoc["correlation"] is None else f"{assoc['correlation']:.3f}",
    )
    m3.metric(
        "p-value",
        "—" if assoc["p_value"] is None else f"{assoc['p_value']:.4g}",
    )

    scatter = view.dropna(subset=["price_change_pct", "inventory_change_pct"])
    if scatter.empty:
        st.caption("Not enough paired non-null change rates for a scatter plot.")
    else:
        fig = px.scatter(
            scatter,
            x="inventory_change_pct",
            y="price_change_pct",
            labels={
                "inventory_change_pct": "Inventory change (%)",
                "price_change_pct": "Price change (%)",
            },
            trendline="ols",
        )
        apply_plotly_layout(
            fig,
            f"{commodity_id} — Co-movement (not causation)",
            x_title="Inventory change (%)",
            y_title="Price change (%)",
        )
        st.plotly_chart(fig, use_container_width=True)


def page_ai() -> None:
    st.title("AI Insight Assistant")
    st.caption(
        "Turn validated analytics into executive-ready findings and recommendations."
    )

    st.markdown(
        """
        <div class="insight-box">
            <strong>How it works</strong><br/>
            Statistical outputs (descriptive stats, correlations, regression, trends, forecasts)
            are summarized into a structured decision-support report. An optional LLM can polish
            the narrative; offline demos use a rule-based analyst fallback.
        </div>
        """,
        unsafe_allow_html=True,
    )

    default_q = "Analyze the digital industry trend"
    question = st.text_area(
        "Analysis question",
        value=default_q,
        height=100,
        help="Examples: What factors influence digital industry growth? "
        "Relationship between technology investment and revenue growth?",
    )
    use_llm = st.toggle(
        "Use LLM (requires OPENAI_API_KEY)",
        value=False,
        help="Turn on to call the configured OpenAI-compatible endpoint.",
    )

    examples = st.columns(3)
    prompts = [
        "What factors influence digital industry growth?",
        "Relationship between technology investment and revenue growth?",
        "Which regions are falling behind on digital development?",
    ]
    for col, prompt in zip(examples, prompts):
        if col.button(prompt, use_container_width=True):
            question = prompt
            st.session_state["ai_question"] = prompt

    if "ai_question" in st.session_state:
        question = st.session_state["ai_question"]

    if st.button("Generate insight report", type="primary", use_container_width=True):
        with st.spinner("Synthesizing analytical insight..."):
            # Prefer filtered-frame context for relevance
            from src.llm_summary import build_context_from_dataframe, generate_insights

            df = filtered_frame()
            context = build_context_from_dataframe(df, include_forecast=True)
            report = generate_insights(context, user_question=question, use_llm=use_llm)

        st.session_state["last_insight"] = report.to_dict()

    if "last_insight" in st.session_state:
        report = st.session_state["last_insight"]
        badge = "LLM" if report.get("source") == "llm" else "Rule-based"
        st.success(f"Report generated · source: {badge}")

        st.subheader("Summary")
        st.write(report["summary"])

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Key findings")
            for item in report.get("key_findings", []):
                st.markdown(f"- {item}")
            st.subheader("Potential drivers")
            for item in report.get("drivers", []):
                st.markdown(f"- {item}")
        with c2:
            st.subheader("Recommendations")
            for item in report.get("recommendations", []):
                st.markdown(f"- {item}")

        with st.expander("Structured JSON output"):
            st.json(
                {
                    "summary": report["summary"],
                    "key_findings": report["key_findings"],
                    "drivers": report["drivers"],
                    "recommendations": report["recommendations"],
                }
            )
        if report.get("raw_response"):
            with st.expander("Model / fallback notes"):
                st.code(report["raw_response"])


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

PAGES = {
    "Overview Dashboard": page_overview,
    "Data Quality Monitoring": page_data_quality,
    "Indicator Analysis": page_indicators,
    "Forecasting": page_forecasting,
    "Commodity Market Analytics": page_commodity,
    "AI Insight Assistant": page_ai,
}

PAGES[page]()
