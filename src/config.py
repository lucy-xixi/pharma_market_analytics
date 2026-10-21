"""
Configuration loader for the pharmaceutical sales platform.

`load_pharma_settings` reads only `config/pharma_settings.yaml`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHARMA_CONFIG_PATH = PROJECT_ROOT / "config" / "pharma_settings.yaml"

# Section names that belong to the retired digital-industry configuration.
_LEGACY_SETTING_SECTIONS = frozenset(
    {"commodity", "indicators", "analytics", "forecasting", "llm", "dashboard"}
)
LEGACY_RAW_FILENAMES = frozenset(
    {
        "digital_industry_indicators.csv",
        "technology_investment_trends.csv",
        "digital_employment_macro.csv",
        "crude_oil_weekly_synthetic.csv",
        "industry_panel_clean.csv",
        "crude_oil_weekly_clean.csv",
    }
)


def _load_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML mapping. Raises if the file is missing or not a mapping."""
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, encoding="utf-8") as handle:
        settings = yaml.safe_load(handle)
    if not isinstance(settings, dict):
        raise ValueError(f"Configuration file must be a mapping: {path}")
    return settings


def _resolve_paths(settings: dict[str, Any]) -> dict[str, Any]:
    """Resolve relative `paths` entries against the project root."""
    paths = settings.get("paths", {})
    for key, value in list(paths.items()):
        resolved = Path(value)
        if not resolved.is_absolute():
            paths[key] = str((PROJECT_ROOT / resolved).resolve())
    settings["paths"] = paths
    return settings


def load_pharma_settings(config_path: Path | None = None) -> dict[str, Any]:
    """Load the pharmaceutical sales configuration."""
    settings = _resolve_paths(_load_yaml(config_path or PHARMA_CONFIG_PATH))
    present = _LEGACY_SETTING_SECTIONS.intersection(settings)
    if present:
        names = ", ".join(sorted(present))
        raise ValueError("Pharma settings must not include legacy sections: " + names)
    primary = str(settings.get("data", {}).get("primary_file", ""))
    if primary in LEGACY_RAW_FILENAMES:
        raise ValueError(f"Pharma primary_file must not point at a legacy dataset: {primary}")
    return settings
