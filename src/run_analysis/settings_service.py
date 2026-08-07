"""Validated settings projection and local overlay persistence."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from .config import _deep_merge, load_config
from .modeling import InsufficientModelDataError, fit_models
from .processing import process_activities
from .weather import update_weather
from .web.schemas import (
    CoachingSettings,
    MovingTimeSettings,
    ProfileSettings,
    SettingsPatch,
    SettingsResponse,
    UploadStage,
    ZoneRange,
)


DEFAULT_WINDOWS = [14, 28, 42, 56, 90, 180, 365]


def validate_config(config: dict[str, Any]) -> None:
    resting = int(config["resting_hr"])
    target = int(config["target_hr"])
    maximum = int(config["max_hr"])
    if not resting < target < maximum:
        raise ValueError("Heart rates must satisfy resting HR < target HR < max HR")
    zones = config["zones"]
    expected = ["z1", "z2", "z3", "z4", "z5"]
    if list(zones) != expected and set(zones) != set(expected):
        raise ValueError("Zones must contain exactly z1, z2, z3, z4, and z5")
    previous_max = None
    for name in expected:
        lower, upper = map(int, zones[name])
        if lower > upper:
            raise ValueError(f"{name} lower bound exceeds upper bound")
        if previous_max is not None and lower <= previous_max:
            raise ValueError("Heart-rate zones must be ordered and non-overlapping")
        previous_max = upper
    if previous_max > maximum:
        raise ValueError("The top of Z5 cannot exceed max HR")


def settings_response(config: dict[str, Any], stages: list[UploadStage] | None = None) -> SettingsResponse:
    reference = config["reference_conditions"]
    weather = config["weather"]
    coaching = {
        "training_goal": "general_fitness",
        "long_run_progression_factor": 1.10,
        "high_load_ratio": 1.30,
        "moderate_intensity_leakage_fraction": 0.17,
        "minimum_days_between_quality_sessions": 4,
        "quality_recency_reference_days": 7,
        "typical_rest_days_between_runs": 1,
        "capacity_retention_half_life_days": 42,
        "capacity_retention_grace_days": 28,
        "minimum_running_days_28d_for_quality": 8,
        "long_run_recency_reference_days": 10,
        "reduced_volume_factor": 0.70,
        "quality_sessions": {
            "short_intervals": True,
            "long_intervals": True,
            "threshold": True,
            "progression": True,
            "hill_repeats": False,
        },
        **config.get("coaching", {}),
    }
    return SettingsResponse(
        max_hr=config["max_hr"],
        resting_hr=config["resting_hr"],
        target_hr=config["target_hr"],
        zones={name: ZoneRange(minimum_bpm=value[0], maximum_bpm=value[1]) for name, value in config["zones"].items()},
        reference_temperature_f=reference["temperature_f"],
        reference_dewpoint_f=reference["dewpoint_f"],
        reference_wind_mph=reference["wind_mph"],
        reference_grade_percent=reference["grade_percent"],
        reference_within_run_minutes=reference["within_run_minutes"],
        weather_privacy_radius_km=weather.get("privacy_jitter_radius_km", 0),
        historical_weather_enabled=bool(weather.get("historical_enabled", False)),
        forecast_weather_enabled=bool(weather.get("forecast_enabled", False)),
        available_fitness_windows=DEFAULT_WINDOWS,
        default_fitness_window=config.get("app", {}).get("default_fitness_window", 28),
        moving_time=MovingTimeSettings.model_validate(config["moving_time"]),
        coaching=CoachingSettings.model_validate(coaching),
        profile=(ProfileSettings.model_validate(config["profile"]) if config.get("profile") else None),
        recalculation=stages or [],
    )


def _patch_to_overlay(patch: SettingsPatch) -> dict[str, Any]:
    values = patch.model_dump(exclude_none=True, mode="python")
    overlay: dict[str, Any] = {}
    for key in ("max_hr", "resting_hr", "target_hr"):
        if key in values:
            overlay[key] = values[key]
    if "zones" in values:
        overlay["zones"] = {
            name: [zone["minimum_bpm"], zone["maximum_bpm"]]
            for name, zone in values["zones"].items()
        }
    reference_map = {
        "reference_temperature_f": "temperature_f",
        "reference_dewpoint_f": "dewpoint_f",
        "reference_wind_mph": "wind_mph",
        "reference_grade_percent": "grade_percent",
        "reference_within_run_minutes": "within_run_minutes",
    }
    reference = {target: values[source] for source, target in reference_map.items() if source in values}
    if reference:
        overlay["reference_conditions"] = reference
    if "weather_privacy_radius_km" in values:
        overlay["weather"] = {"privacy_jitter_radius_km": values["weather_privacy_radius_km"]}
    if "forecast_weather_enabled" in values:
        overlay["weather"] = {
            **overlay.get("weather", {}),
            "forecast_enabled": values["forecast_weather_enabled"],
        }
    if "historical_weather_enabled" in values:
        overlay["weather"] = {
            **overlay.get("weather", {}),
            "historical_enabled": values["historical_weather_enabled"],
        }
    if "moving_time" in values:
        overlay["moving_time"] = values["moving_time"]
    if "coaching" in values:
        overlay["coaching"] = values["coaching"]
    if "profile" in values:
        profile = values["profile"]
        if hasattr(profile.get("birth_date"), "isoformat"):
            profile["birth_date"] = profile["birth_date"].isoformat()
        overlay["profile"] = profile
    if "default_fitness_window" in values:
        overlay["app"] = {"default_fitness_window": values["default_fitness_window"]}
    return overlay


def save_settings_overlay(config_path: Path, patch: SettingsPatch) -> dict[str, Any]:
    base = load_config(config_path)
    overlay_path = config_path.with_name("config.local.yaml")
    existing: dict[str, Any] = {}
    if overlay_path.exists():
        existing = yaml.safe_load(overlay_path.read_text(encoding="utf-8")) or {}
    updated_overlay = _deep_merge(existing, _patch_to_overlay(patch))
    candidate = _deep_merge(base, _patch_to_overlay(patch))
    validate_config(candidate)
    temporary = overlay_path.with_suffix(".yaml.tmp")
    temporary.write_text(yaml.safe_dump(updated_overlay, sort_keys=False), encoding="utf-8")
    temporary.replace(overlay_path)
    return load_config(config_path)


def recalculate_for_settings(connection, config: dict[str, Any], project_root: Path, patch: SettingsPatch) -> list[UploadStage]:
    stages: list[UploadStage] = []
    changed = patch.model_fields_set
    analytical = changed & {
        "max_hr", "resting_hr", "target_hr", "zones", "reference_temperature_f",
        "reference_dewpoint_f", "reference_wind_mph", "reference_grade_percent",
        "reference_within_run_minutes", "moving_time",
    }
    if analytical:
        try:
            summary = process_activities(connection, config, force=True)
            stages.append(UploadStage(name="process", status="complete", detail=str(asdict(summary))))
        except Exception as exc:
            stages.append(UploadStage(name="process", status="failed", detail=str(exc)))
        try:
            summary = fit_models(connection, config, project_root / "output" / "model_results.json")
            stages.append(UploadStage(name="model", status="complete", detail=str(asdict(summary))))
        except InsufficientModelDataError as exc:
            stages.append(UploadStage(name="model", status="deferred", detail=str(exc)))
        except Exception as exc:
            stages.append(UploadStage(name="model", status="failed", detail=str(exc)))
    if "weather_privacy_radius_km" in changed:
        stages.append(UploadStage(name="weather", status="deferred", detail="Privacy radius applies to future or explicitly refreshed weather retrieval; cached coordinates were not silently resent."))
    if "historical_weather_enabled" in changed:
        if patch.historical_weather_enabled:
            try:
                summary = update_weather(connection, config, project_root)
                stages.append(
                    UploadStage(name="weather", status="complete", detail=str(asdict(summary)))
                )
            except Exception as exc:
                stages.append(UploadStage(name="weather", status="failed", detail=str(exc)))
        else:
            stages.append(
                UploadStage(
                    name="weather",
                    status="complete",
                    detail="Historical weather retrieval is disabled; cached local weather was retained.",
                )
            )
    return stages
