"""Application configuration loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Configuration root must be a mapping: {config_path}")
    overlay_path = config_path.with_name("config.local.yaml")
    if overlay_path.exists():
        with overlay_path.open("r", encoding="utf-8") as handle:
            overlay = yaml.safe_load(handle) or {}
        if not isinstance(overlay, dict):
            raise ValueError(f"Configuration overlay root must be a mapping: {overlay_path}")
        config = _deep_merge(config, overlay)
    required = ("max_hr", "resting_hr", "target_hr", "zones", "timezone_default", "paths")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing required configuration keys: {', '.join(missing)}")
    return config


def resolve_project_path(project_root: Path, configured_path: str) -> Path:
    path = Path(configured_path)
    return path if path.is_absolute() else project_root / path
