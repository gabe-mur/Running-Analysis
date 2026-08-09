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
    _upgrade_cadence_thresholds(config)
    return config


#: Cadence thresholds used to be expressed in Garmin's one-sided strides per
#: minute. They are now total steps per minute, matching the one canonical
#: conversion in ``run_analysis.cadence``.
_LEGACY_CADENCE_KEYS = {
    "high_confidence_walk_cadence_max": "high_confidence_walk_cadence_max_spm",
    "review_low_cadence_max": "review_low_cadence_max_spm",
}

_CADENCE_THRESHOLD_DEFAULTS = {
    "high_confidence_walk_cadence_max_spm": 110,
    "very_low_cadence_max_spm": 130,
    "review_low_cadence_max_spm": 140,
}


def _upgrade_cadence_thresholds(config: dict[str, Any]) -> None:
    """Convert legacy one-sided cadence thresholds in place.

    An existing config.yaml written before the steps-per-minute change would
    otherwise silently compare a one-sided threshold against a doubled value
    and label ordinary running as walking.
    """

    section = config.get("activity_classification")
    if not isinstance(section, dict):
        section = {}
        config["activity_classification"] = section
    for legacy, current in _LEGACY_CADENCE_KEYS.items():
        if legacy in section:
            value = section.pop(legacy)
            section.setdefault(current, float(value) * 2)
    for key, default in _CADENCE_THRESHOLD_DEFAULTS.items():
        section.setdefault(key, default)


def resolve_project_path(project_root: Path, configured_path: str) -> Path:
    path = Path(configured_path)
    return path if path.is_absolute() else project_root / path
