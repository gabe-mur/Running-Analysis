"""Validated settings projection and local overlay persistence."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from .config import _deep_merge, load_config
from .modeling import InsufficientModelDataError, fit_models, invalidate_comparison_results
from .processing import process_activities
from .privacy import private_file
from .race_goals import assess_race_goal
from .weather import update_weather
from .web.schemas import (
    CoachingSettings,
    MovingTimeSettings,
    ProfileSettings,
    RacePreparationSettings,
    SettingsPatch,
    SetupSettings,
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
    if int(zones['z1'][0]) <= resting:
        raise ValueError('The bottom of Z1 must exceed resting HR')
    if previous_max > maximum:
        raise ValueError("The top of Z5 cannot exceed max HR")


def settings_response(config: dict[str, Any], stages: list[UploadStage] | None = None, *, project_root: Path | None = None) -> SettingsResponse:
    from .adaptive.settings_view import inspect_settings
    reference = config["reference_conditions"]
    weather = config["weather"]
    configured_coaching = {
        key: value
        for key, value in config.get("coaching", {}).items()
        if key != "minimum_days_between_quality_sessions"
    }
    coaching = {
        "training_goal": "general_fitness",
        "goal_date": None,
        "goal_pace_min_mile": None,
        "general_fitness_progression_fraction": 0.08,
        "long_run_progression_factor": 1.10,
        "long_run_target_progression_fraction": 0.05,
        "high_load_ratio": 1.30,
        "moderate_intensity_leakage_fraction": 0.17,
        "quality_recency_reference_days": 7,
        "typical_rest_days_between_runs": 1,
        "capacity_retention_half_life_days": 84,
        "capacity_retention_grace_days": 28,
        "long_run_retention_half_life_days": 180,
        "long_run_retention_grace_days": 90,
        "minimum_running_days_28d_for_quality": 8,
        "long_run_recency_reference_days": 7,
        "reduced_volume_factor": 0.70,
        "quality_sessions": {
            "fartlek": True,
            "short_intervals": True,
            "long_intervals": True,
            "threshold": True,
            "progression": True,
            "hill_repeats": False,
        },
        **configured_coaching,
    }
    return SettingsResponse(
        beginner_start=bool(config.get('beginner_start')),
        training_background=config.get('training_background'),
        training_background_reported_at=config.get('training_background_reported_at'),
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
        setup=SetupSettings.model_validate(config.get("setup") or {}),
        recalculation=stages or [],
        planner=inspect_settings(config, project_root) if project_root is not None else None,
        race_preparation=RacePreparationSettings.model_validate(config.get('race_preparation') or {}),
    )


def _patch_to_overlay(patch: SettingsPatch) -> dict[str, Any]:
    values = patch.model_dump(exclude_none=True, mode="python")
    overlay: dict[str, Any] = {}
    if patch.beginner_start is not None:
        overlay['beginner_start'] = patch.beginner_start
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
    if 'race_preparation' in patch.model_fields_set:
        if patch.race_preparation is None:
            raise ValueError('Use general fitness to clear the race preparation draft')
        # Include null date so clearing a draft cannot retain the previous date
        # through the recursive YAML merge. One owner, separate from legacy pace.
        overlay['race_preparation'] = patch.race_preparation.model_dump(mode='json')
    if "profile" in values:
        profile = values["profile"]
        if hasattr(profile.get("birth_date"), "isoformat"):
            profile["birth_date"] = profile["birth_date"].isoformat()
        overlay["profile"] = profile
    if "setup" in values:
        # Merged, not replaced: confirming one step must not silently
        # un-confirm the others. Dumped as plain strings because the overlay is
        # YAML, and PyYAML matches representers by exact type -- a StrEnum is a
        # str but is not `str`, so it would fail to serialise.
        overlay["setup"] = patch.setup.model_dump(mode="json", exclude_none=True)
    if "default_fitness_window" in values:
        overlay["app"] = {"default_fitness_window": values["default_fitness_window"]}
    return overlay


def save_settings_overlay(
    config_path: Path,
    patch: SettingsPatch,
    *,
    connection=None,
) -> dict[str, Any]:
    if 'planner_preferences' in patch.model_fields_set:
        raise ValueError('Planner preferences belong to the durable adaptive journal; save them separately')
    base = load_config(config_path)
    if patch.beginner_start is not None:
        if connection is None:
            raise ValueError('A verified app database is required to change starting training')
        if (connection.execute("SELECT 1 FROM app_state WHERE key='adaptive_runtime'").fetchone()
                and patch.beginner_start != bool(base.get('beginner_start'))):
            raise ValueError('Starting training is already adopted; introductory mode cannot reset it')
        background = patch.training_background if 'training_background' in patch.model_fields_set else base.get('training_background')
        if patch.beginner_start and background:
            raise ValueError('Choose either a reported training level or introductory training')
    if 'training_background' in patch.model_fields_set:
        from datetime import datetime, timezone
        from zoneinfo import ZoneInfo
        from .training_background import reported_background
        if connection is None:
            raise ValueError('A verified app database is required to change starting training')
        background = patch.training_background
        if connection is not None and connection.execute(
                "SELECT 1 FROM app_state WHERE key='adaptive_runtime'").fetchone() and background != reported_background(base):
            raise ValueError('Starting training is already adopted. Updating it requires a separate starting-level review.')
        if background and background.last_run_date and background.last_run_date > datetime.now(ZoneInfo(base['timezone_default'])).date():
            raise ValueError('Last run cannot be in the future')
    overlay_path = config_path.with_name("config.local.yaml")
    existing: dict[str, Any] = {}
    if overlay_path.exists():
        existing = yaml.safe_load(overlay_path.read_text(encoding="utf-8")) or {}
    updated_overlay = _deep_merge(existing, _patch_to_overlay(patch))
    candidate = _deep_merge(base, _patch_to_overlay(patch))
    if 'training_background' in patch.model_fields_set:
        background = patch.training_background
        changed = background != reported_background(base)
        value = background.model_dump(mode='json') if background else None
        stamp = (datetime.now(timezone.utc).isoformat() if changed and background else
                 base.get('training_background_reported_at') if background else None)
        for destination in (updated_overlay, candidate):
            destination['training_background'] = value
            destination['training_background_reported_at'] = stamp
    if candidate.get('beginner_start') and candidate.get('training_background'):
        raise ValueError('Choose either a reported training level or introductory training')
    from .watch_zones import watch_zone_patch
    # An explicit watch-source selection must not silently adopt manual fields
    # when the upload contains no watch settings. Unrelated legacy saves remain
    # compatible until the source is explicitly selected/confirmed.
    selecting_watch = (patch.setup is not None and 'zone_method' in patch.setup.model_fields_set
        and patch.setup.zone_method == 'device')
    recorded_zones = watch_zone_patch(candidate, connection, required=selecting_watch)
    candidate = _deep_merge(candidate, recorded_zones)
    updated_overlay = _deep_merge(updated_overlay, recorded_zones)
    # Normal UI saves never submit an analysis coordinate. Derive it on the
    # server; an old setup confirmation/mode is not an opt-in requirement.
    # Explicit target_hr writes from legacy programmatic callers remain
    # compatible unless automatic mode is specified; there is no UI control.
    if ((candidate.get('setup') or {}).get('comparison_hr_mode') == 'automatic'
            or (connection is not None and 'target_hr' not in patch.model_fields_set)):
        from .onboarding import automatic_comparison_hr
        automatic_setup = {'comparison_hr_mode': 'automatic'}
        candidate['setup'] = _deep_merge(candidate.get('setup') or {}, automatic_setup)
        updated_overlay['setup'] = _deep_merge(updated_overlay.get('setup') or {}, automatic_setup)
        candidate['target_hr'] = automatic_comparison_hr(connection, candidate['zones'],
            current=base['target_hr']).recommended_bpm
        updated_overlay['target_hr'] = candidate['target_hr']
    validate_config(candidate)
    from .adaptive.settings_view import validate_settings_change
    validate_settings_change(base, candidate, connection=connection)
    if ('race_preparation' in patch.model_fields_set
            and patch.race_preparation.distance != 'general_fitness'
            and settings_response(base).race_preparation != patch.race_preparation):
        from datetime import datetime
        from zoneinfo import ZoneInfo
        today = datetime.now(ZoneInfo(candidate['timezone_default'])).date()
        if patch.race_preparation.race_date <= today:
            raise ValueError('Choose a future race date')
    # An unrelated strength/settings save must not revalidate an unchanged old
    # pace goal. New pace-free drafts never call the retired pace predictor.
    goal_keys = ('training_goal', 'goal_date', 'goal_pace_min_mile')
    if "coaching" in patch.model_fields_set:
        previous_goal = settings_response(base).coaching.model_dump(mode='json')
        next_goal = settings_response(candidate).coaching.model_dump(mode='json')
        if any(previous_goal[k] != next_goal[k] for k in goal_keys):
            assess_race_goal(connection, candidate)
    temporary = overlay_path.with_suffix(".yaml.tmp")
    temporary.write_text(yaml.safe_dump(updated_overlay, sort_keys=False), encoding="utf-8")
    private_file(temporary)
    temporary.replace(overlay_path)
    private_file(overlay_path)
    if connection is not None and candidate['target_hr'] != base['target_hr']:
        invalidate_comparison_results(connection)
        connection.commit()
    return load_config(config_path)


def _effective_changed_fields(
    previous_config: dict[str, Any],
    config: dict[str, Any],
    submitted_fields: set[str],
) -> set[str]:
    """Return submitted API settings whose effective values actually changed.

    Both Settings forms intentionally submit complete nested objects. Pydantic's
    ``model_fields_set`` therefore describes request presence, not a semantic
    change. Using it directly caused a setup confirmation with identical HR,
    zones, weather, and coaching values to refit every model and discard the
    saved weekly plan.
    """

    before = settings_response(previous_config).model_dump(mode="python")
    after = settings_response(config).model_dump(mode="python")
    changed = {
        field
        for field in submitted_fields
        if before.get(field) != after.get(field)
    }
    # Automatic calibration can change the coordinate without a writable
    # target_hr field in the submitted form. Refit, never relabel old results.
    if before['target_hr'] != after['target_hr']:
        changed.add('target_hr')
    for field in ('zones', 'max_hr', 'resting_hr', 'profile'):
        if before[field] != after[field]:
            changed.add(field)
    return changed


def refresh_automatic_comparison(config_path, connection):
    """Background upload hook; no setup opt-in, unchanged anchors are no-ops."""
    config = load_config(config_path)
    from .onboarding import automatic_comparison_hr
    recommendation = automatic_comparison_hr(connection, config['zones'], current=config['target_hr'])
    if recommendation.recommended_bpm == config['target_hr']:
        return config, False
    updated = save_settings_overlay(Path(config_path), SettingsPatch(setup={
        **(config.get('setup') or {}), 'comparison_hr_mode': 'automatic'}),
        connection=connection)
    return updated, True


def refresh_watch_zones(config_path, connection):
    """Upload hook: current watch HR bundle, preserving adopted load units."""
    from .watch_zones import watch_zone_patch
    config = load_config(config_path)
    patch = watch_zone_patch(config, connection)
    if not patch or all(config.get(k) == v for k,v in patch.items()):
        return config, False
    requested = SettingsPatch(zones={name:dict(minimum_bpm=low, maximum_bpm=high)
        for name,(low,high) in patch['zones'].items()})
    updated = save_settings_overlay(Path(config_path), requested, connection=connection)
    return updated, any(updated[k] != config[k] for k in ('zones','max_hr','resting_hr'))


def recalculate_for_settings(
    connection,
    config: dict[str, Any],
    project_root: Path,
    patch: SettingsPatch,
    *,
    previous_config: dict[str, Any] | None = None,
) -> list[UploadStage]:
    stages: list[UploadStage] = []
    changed = set(patch.model_fields_set)
    if previous_config is not None:
        changed = _effective_changed_fields(previous_config, config, changed)
    if 'race_preparation' in changed:
        stages.append(UploadStage(name='race_preparation', status='deferred',
            detail='Preparation draft saved. Race progression, taper and event accounting are not active; the issued calendar is unchanged.'))
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
    planner_inputs = analytical | changed & {
        "beginner_start", "training_background",
        "coaching",
        "forecast_weather_enabled",
        "weather_privacy_radius_km",
    }
    if planner_inputs:
        connection.execute("DELETE FROM app_state WHERE key='weekly_schedule'")
        connection.commit()
        stages.append(
            UploadStage(
                name="schedule",
                status="deferred",
                detail="The saved schedule was invalidated and will be regenerated from the new settings.",
            )
        )
    return stages

#: Overlay sections that are model machinery rather than facts about the
#: athlete. Only these are cleared by a reset.
_ADVANCED_SECTIONS = ("moving_time", "reference_conditions")

#: Coaching keys that describe the athlete's intent, not the planner's tuning.
#: A reset must not silently discard a race date someone entered.
_PERSONAL_COACHING_KEYS = frozenset(
    {"training_goal", "goal_date", "goal_pace_min_mile", "quality_sessions", "quality_focus", "strength_training"}
)


def reset_advanced_overlay(config_path: Path) -> dict[str, Any]:
    """Return the tuning knobs to their shipped values.

    Deliberately partial. Everything under Advanced is a modelling parameter
    with a defensible default, so putting it back is safe. Heart rates, zones,
    profile, goal, and weather consent are answers about a person, and a button
    labelled "reset" must never be able to erase those -- a destructive action
    hiding behind a harmless-sounding label is worse than no button.
    """

    overlay_path = Path(config_path).with_name("config.local.yaml")
    existing: dict[str, Any] = {}
    if overlay_path.exists():
        existing = yaml.safe_load(overlay_path.read_text(encoding="utf-8")) or {}
    for section in _ADVANCED_SECTIONS:
        existing.pop(section, None)
    coaching = existing.get("coaching")
    if isinstance(coaching, dict):
        kept = {key: value for key, value in coaching.items() if key in _PERSONAL_COACHING_KEYS}
        if kept:
            existing["coaching"] = kept
        else:
            existing.pop("coaching", None)
    from .adaptive.settings_view import validate_settings_change
    # Preview reset against the actual base, without writing an overlay first.
    base = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    validate_settings_change(load_config(config_path), _deep_merge(base, existing))
    temporary = overlay_path.with_suffix(".yaml.tmp")
    temporary.write_text(yaml.safe_dump(existing, sort_keys=False), encoding="utf-8")
    private_file(temporary)
    temporary.replace(overlay_path)
    private_file(overlay_path)
    return load_config(config_path)
