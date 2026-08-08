"""
Capture full Streamlit dashboard page screenshots for the GitHub portfolio.

Prerequisites:
  - streamlit run dashboard/app.py  (http://localhost:8501)
  - pip install playwright && python -m playwright install chromium

Usage:
  python scripts/capture_dashboard_screenshots.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from playwright.sync_api import TimeoutError as PlaywrightTimeout
from playwright.sync_api import sync_playwright

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "reports" / "screenshots"
BASE_URL = "http://localhost:8501"

PAGES = [
    ("Overview Dashboard", "overview.png", False),
    ("Data Quality Monitoring", "data_quality.png", False),
    ("Indicator Analysis", "indicators.png", False),
    ("Forecasting", "forecasting.png", False),
    ("AI Insight Assistant", "ai_insight.png", True),
]


def wait_for_streamlit(page, timeout_ms: int = 90000) -> None:
    page.goto(BASE_URL, wait_until="domcontentloaded", timeout=timeout_ms)
    page.wait_for_selector('[data-testid="stSidebar"]', timeout=timeout_ms)
    page.wait_for_selector('[data-testid="stMain"]', timeout=timeout_ms)
    # Wait until plotly or metrics appear on the default overview page
    page.wait_for_selector('[data-testid="stMetric"], [data-testid="stPlotlyChart"]', timeout=timeout_ms)
    time.sleep(2.0)


def hide_chrome(page) -> None:
    page.add_style_tag(
        content="""
        [data-testid="stToolbar"],
        [data-testid="stDecoration"],
        [data-testid="stStatusWidget"],
        .stDeployButton,
        header[data-testid="stHeader"] {
            display: none !important;
            height: 0 !important;
            visibility: hidden !important;
        }
        """
    )


def select_page(page, label: str) -> None:
    sidebar = page.locator('[data-testid="stSidebar"]')
    # Streamlit radio options
    option = sidebar.locator('[data-testid="stRadioOption"]').filter(has_text=label)
    if option.count() == 0:
        option = sidebar.get_by_text(label, exact=True)
    option.first.click(timeout=20000)
    # Wait for main heading to reflect navigation
    page.locator('[data-testid="stMain"]').get_by_role("heading").first.wait_for(timeout=30000)
    time.sleep(2.5)


def prepare_ai_page(page) -> None:
    try:
        page.get_by_role("button", name="Generate insight report").click(timeout=15000)
        page.get_by_text("Report generated", exact=False).wait_for(timeout=90000)
        time.sleep(2.0)
    except PlaywrightTimeout:
        pass


def expand_viewport_to_content(page) -> None:
    """Grow the viewport so one shot covers sidebar + full main scroll area."""
    height = page.evaluate(
        """() => {
            const app = document.querySelector('[data-testid="stAppViewContainer"]');
            const main = document.querySelector('[data-testid="stMain"]');
            const side = document.querySelector('[data-testid="stSidebar"]');
            const h = Math.max(
                app ? app.scrollHeight : 0,
                main ? main.scrollHeight : 0,
                side ? side.scrollHeight : 0,
                document.body.scrollHeight,
                1200
            );
            return Math.min(h + 80, 5000);
        }"""
    )
    page.set_viewport_size({"width": 1600, "height": int(height)})
    time.sleep(1.0)


def capture_page(page, filename: str) -> Path:
    hide_chrome(page)
    expand_viewport_to_content(page)
    page.evaluate("window.scrollTo(0, 0)")
    time.sleep(0.8)

    out_path = OUT_DIR / filename
    # Capture the whole app shell (sidebar + main) for a true dashboard look
    app = page.locator('[data-testid="stAppViewContainer"]')
    box = app.bounding_box()
    if box:
        page.screenshot(
            path=str(out_path),
            type="png",
            clip={
                "x": 0,
                "y": 0,
                "width": min(box["width"] + box["x"], 1600),
                "height": min(max(box["height"], 1000), 5000),
            },
        )
    else:
        page.screenshot(path=str(out_path), full_page=True, type="png")
    return out_path


def capture_all() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 1600, "height": 1200},
            device_scale_factor=2,
        )
        page = context.new_page()

        print(f"Opening {BASE_URL} ...")
        wait_for_streamlit(page)
        hide_chrome(page)

        for label, filename, needs_ai in PAGES:
            print(f"Capturing full page: {label} → {filename}")
            # Reset viewport before each page so content height is recalculated
            page.set_viewport_size({"width": 1600, "height": 1200})
            select_page(page, label)
            if needs_ai:
                prepare_ai_page(page)
            # Extra wait for Plotly canvases
            time.sleep(1.5)
            out_path = capture_page(page, filename)
            print(f"  saved {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")

        browser.close()

    print("Done. Full dashboard page screenshots written to reports/screenshots/")


if __name__ == "__main__":
    try:
        capture_all()
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        print(
            "Make sure the dashboard is running: streamlit run dashboard/app.py",
            file=sys.stderr,
        )
        sys.exit(1)
