"""
Export key analytical charts/tables to reports/screenshots/ for portfolio use.

This does not capture full Streamlit UI pages — use browser screenshots for those.
Run from project root:

    python scripts/export_portfolio_charts.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.export_utils import export_portfolio_assets  # noqa: E402
from src.analytics import industry_region_summary  # noqa: E402
from src.data_loader import load_from_sqlite  # noqa: E402
from src.forecasting import forecast_to_frame, run_forecast  # noqa: E402
from src.indicator_engine import ranking_table  # noqa: E402


PRIMARY = "#0B3D5C"
ACCENT = "#1F7A8C"
WARM = "#C17A3A"


def _layout(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=PRIMARY)),
        paper_bgcolor="white",
        plot_bgcolor="#F7F9FB",
        margin=dict(l=40, r=20, t=60, b=40),
        font=dict(family="Segoe UI, Helvetica, Arial, sans-serif"),
    )
    return fig


def main() -> None:
    df = load_from_sqlite()
    latest = int(df["year"].max())

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
        color_discrete_sequence=[PRIMARY, ACCENT, WARM],
    )
    _layout(fig_trend, "Industry-wide indicator trends")

    ranked = ranking_table(
        df,
        metric="digital_development_index",
        year=latest,
        group_by="industry",
    )
    fig_rank = px.bar(
        ranked,
        x="digital_development_index",
        y="industry",
        orientation="h",
        color="digital_development_index",
        color_continuous_scale=["#A7C4D4", PRIMARY],
    )
    fig_rank.update_layout(coloraxis_showscale=False, yaxis={"categoryorder": "total ascending"})
    _layout(fig_rank, f"Digital Development Index ranking ({latest})")

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
    _layout(fig_corr, "Indicator correlation heatmap")

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
            line=dict(color=PRIMARY, width=3),
        )
    )
    fig_fc.add_trace(
        go.Scatter(
            x=fut["year"],
            y=fut["value"],
            mode="lines+markers",
            name="Forecast",
            line=dict(color=WARM, width=3, dash="dash"),
        )
    )
    _layout(fig_fc, "Digital index forecast (ensemble)")

    kpis = industry_region_summary(df, year=latest)
    quality = pd.DataFrame(
        [
            {"metric": "total_records", "value": len(df)},
            {"metric": "latest_year", "value": latest},
            {"metric": "industries", "value": df["industry"].nunique()},
            {"metric": "regions", "value": df["region"].nunique()},
        ]
    )

    written = export_portfolio_assets(
        figures={
            "chart_overview_trends.png": fig_trend,
            "chart_indicator_ranking.png": fig_rank,
            "chart_correlation_heatmap.png": fig_corr,
            "chart_forecast_digital_index.png": fig_fc,
        },
        tables={
            "table_kpi_summary": kpis,
            "table_portfolio_meta": quality,
        },
    )
    print("Exported portfolio chart/table assets:")
    for name, path in written.items():
        print(f"  {name} → {path}")
    print(
        "\nNote: Full-page README shots (overview.png, data_quality.png, …) "
        "should still be captured from the Streamlit UI."
    )


if __name__ == "__main__":
    main()
