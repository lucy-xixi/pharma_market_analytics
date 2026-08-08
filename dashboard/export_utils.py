"""
Portfolio presentation helpers for the Streamlit dashboard.

Exports Plotly charts and summary tables to reports/screenshots/
without modifying core analytics logic.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.graph_objects import Figure

from src.config import PROJECT_ROOT

SCREENSHOTS_DIR = PROJECT_ROOT / "reports" / "screenshots"

# Suggested full-page screenshot filenames for README / GitHub portfolio
PORTFOLIO_PAGE_SHOTS = {
    "overview": "overview.png",
    "data_quality": "data_quality.png",
    "indicators": "indicators.png",
    "forecasting": "forecasting.png",
    "ai_insight": "ai_insight.png",
}


def ensure_screenshots_dir() -> Path:
    """Create reports/screenshots/ if needed and return the path."""
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    return SCREENSHOTS_DIR


def export_plotly_png(
    fig: Figure,
    filename: str,
    width: int = 1280,
    height: int = 720,
    scale: int = 2,
) -> Path:
    """
    Export a Plotly figure to PNG under reports/screenshots/.

    Requires the optional `kaleido` package for static image export.
    """
    out_dir = ensure_screenshots_dir()
    path = out_dir / filename
    try:
        fig.write_image(str(path), width=width, height=height, scale=scale)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Plotly static export failed. Install kaleido: pip install kaleido. "
            f"Original error: {exc}"
        ) from exc
    return path


def export_dataframe_csv(df: pd.DataFrame, filename: str) -> Path:
    """Export a summary table as CSV next to portfolio screenshots."""
    out_dir = ensure_screenshots_dir()
    path = out_dir / filename
    df.to_csv(path, index=False)
    return path


def dataframe_to_plotly_table(
    df: pd.DataFrame,
    title: str = "Summary table",
    max_rows: int = 20,
) -> Figure:
    """Render a DataFrame as a Plotly table figure (exportable to PNG)."""
    view = df.head(max_rows).copy()
    # Stringify for stable display
    for col in view.columns:
        view[col] = view[col].astype(str)

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=list(view.columns),
                    fill_color="#0B3D5C",
                    font=dict(color="white", size=12),
                    align="left",
                ),
                cells=dict(
                    values=[view[c].tolist() for c in view.columns],
                    fill_color="#F7F9FB",
                    align="left",
                    font=dict(size=11, color="#1f2937"),
                ),
            )
        ]
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#0B3D5C")),
        margin=dict(l=20, r=20, t=50, b=20),
        height=min(160 + 28 * len(view), 720),
    )
    return fig


def export_dataframe_png(
    df: pd.DataFrame,
    filename: str,
    title: str = "Summary table",
    max_rows: int = 20,
) -> Path:
    """Export a DataFrame as a PNG table image via Plotly + kaleido."""
    fig = dataframe_to_plotly_table(df, title=title, max_rows=max_rows)
    return export_plotly_png(fig, filename, width=1100, height=min(160 + 28 * min(len(df), max_rows), 720))


def fig_to_png_bytes(
    fig: Figure,
    width: int = 1280,
    height: int = 720,
    scale: int = 2,
) -> bytes:
    """Return PNG bytes for Streamlit download buttons."""
    try:
        return fig.to_image(format="png", width=width, height=height, scale=scale)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Unable to render PNG bytes. Install kaleido: pip install kaleido. "
            f"Original error: {exc}"
        ) from exc


def export_portfolio_assets(
    figures: dict[str, Figure],
    tables: Optional[dict[str, pd.DataFrame]] = None,
) -> dict[str, Path]:
    """
    Batch-export named figures/tables for portfolio packaging.

    Args:
        figures: mapping filename → Plotly figure
        tables: optional mapping filename → DataFrame (exported as PNG table + CSV)

    Returns:
        Mapping of logical name → written path
    """
    written: dict[str, Path] = {}
    ensure_screenshots_dir()

    for filename, fig in figures.items():
        name = filename if filename.endswith(".png") else f"{filename}.png"
        written[name] = export_plotly_png(fig, name)

    for filename, frame in (tables or {}).items():
        stem = filename[:-4] if filename.endswith((".png", ".csv")) else filename
        written[f"{stem}.png"] = export_dataframe_png(frame, f"{stem}.png", title=stem)
        written[f"{stem}.csv"] = export_dataframe_csv(frame, f"{stem}.csv")

    return written


def portfolio_shot_checklist() -> pd.DataFrame:
    """Status table of expected full-page README screenshots."""
    ensure_screenshots_dir()
    rows = []
    for key, filename in PORTFOLIO_PAGE_SHOTS.items():
        path = SCREENSHOTS_DIR / filename
        rows.append(
            {
                "page_key": key,
                "filename": filename,
                "relative_path": f"reports/screenshots/{filename}",
                "captured": path.exists() and path.stat().st_size > 0,
            }
        )
    return pd.DataFrame(rows)


def download_button_bytes(label: str, data: bytes, file_name: str, key: str) -> None:
    """Thin Streamlit download helper (imported only when Streamlit is active)."""
    import streamlit as st

    st.download_button(
        label=label,
        data=io.BytesIO(data),
        file_name=file_name,
        mime="image/png",
        key=key,
        use_container_width=True,
    )
