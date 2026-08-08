"""
Central configuration loader for the Digital Industry Analytics Platform.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Project root: digital_industry_analytics/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"


def load_env() -> None:
    """Load environment variables from .env if present."""
    env_path = PROJECT_ROOT / ".env"
    load_dotenv(env_path if env_path.exists() else None)


def load_settings(config_path: Path | None = None) -> dict[str, Any]:
    """Load YAML settings and resolve absolute paths."""
    load_env()
    path = config_path or DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, encoding="utf-8") as f:
        settings: dict[str, Any] = yaml.safe_load(f)

    # Resolve relative paths against project root
    paths = settings.get("paths", {})
    for key, value in list(paths.items()):
        resolved = Path(value)
        if not resolved.is_absolute():
            paths[key] = str((PROJECT_ROOT / resolved).resolve())
    settings["paths"] = paths

    # Overlay LLM settings from environment
    llm = settings.setdefault("llm", {})
    if os.getenv("OPENAI_API_KEY"):
        llm["api_key"] = os.getenv("OPENAI_API_KEY")
    if os.getenv("OPENAI_BASE_URL"):
        llm["base_url"] = os.getenv("OPENAI_BASE_URL")
    if os.getenv("LLM_MODEL"):
        llm["model"] = os.getenv("LLM_MODEL")

    return settings


def get_db_path(settings: dict[str, Any] | None = None) -> Path:
    """Return SQLite database path."""
    cfg = settings or load_settings()
    return Path(cfg["paths"]["database"])


def get_raw_dir(settings: dict[str, Any] | None = None) -> Path:
    cfg = settings or load_settings()
    return Path(cfg["paths"]["raw_data"])


def get_processed_dir(settings: dict[str, Any] | None = None) -> Path:
    cfg = settings or load_settings()
    return Path(cfg["paths"]["processed_data"])
