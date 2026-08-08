from __future__ import annotations

from pathlib import Path

import pytest

from run_analysis.db import connect, initialize
from run_analysis.metadata_service import update_run_metadata
from run_analysis.settings_service import save_settings_overlay, settings_response
from run_analysis.web.schemas import (
    ActivityHealthTag,
    RunMetadataPatch,
    SettingsPatch,
    WorkoutType,
    ZoneRange,
    CoachingSettings,
)
from test_web_phase1 import _write_config


def test_settings_are_saved_to_overlay_without_rewriting_documented_base(tmp_path: Path) -> None:
    _write_config(tmp_path)
    base_before = (tmp_path / "config.yaml").read_text()
    config = save_settings_overlay(
        tmp_path / "config.yaml",
        SettingsPatch(
            target_hr=146,
            default_fitness_window=42,
            historical_weather_enabled=True,
        ),
    )
    assert (tmp_path / "config.yaml").read_text() == base_before
    assert (tmp_path / "config.local.yaml").exists()
    assert config["target_hr"] == 146
    assert settings_response(config).default_fitness_window == 42
    assert settings_response(config).historical_weather_enabled is True


def test_invalid_physiology_or_overlapping_zones_are_rejected(tmp_path: Path) -> None:
    _write_config(tmp_path)
    with pytest.raises(ValueError, match="resting HR < target HR < max HR"):
        save_settings_overlay(tmp_path / "config.yaml", SettingsPatch(target_hr=200))
    zones = {
        "z1": ZoneRange(minimum_bpm=128, maximum_bpm=145),
        "z2": ZoneRange(minimum_bpm=140, maximum_bpm=153),
        "z3": ZoneRange(minimum_bpm=154, maximum_bpm=166),
        "z4": ZoneRange(minimum_bpm=167, maximum_bpm=180),
        "z5": ZoneRange(minimum_bpm=181, maximum_bpm=195),
    }
    with pytest.raises(ValueError, match="non-overlapping"):
        save_settings_overlay(tmp_path / "config.yaml", SettingsPatch(zones=zones))


def test_metadata_round_trips_through_database_and_override_csv(tmp_path: Path) -> None:
    database = tmp_path / "test.sqlite"
    overrides = tmp_path / "run_overrides.csv"
    with connect(database) as connection:
        initialize(connection)
        cursor = connection.execute(
            """
            INSERT INTO activities(
                activity_uid,activity_id,sport,lap_count,trackpoint_count,gps_quality,hr_quality,
                elevation_quality,cadence_quality,distance_source,namespaces_json,data_quality_json,
                created_at_utc,updated_at_utc
            ) VALUES ('uid','external','Running',0,0,'missing','missing','missing','missing','device','{}','{}','now','now')
            """
        )
        update_run_metadata(
            connection,
            overrides,
            int(cursor.lastrowid),
            RunMetadataPatch(
                workout_type=WorkoutType.HIKE,
                health_tag=ActivityHealthTag.NORMAL,
                include_in_model=False,
                perceived_exertion=7,
                notes="Confirmed hike",
            ),
        )
        row = connection.execute("SELECT * FROM run_overrides WHERE activity_id='external'").fetchone()
    assert row["workout_type"] == "hike"
    assert row["include_in_model"] == 0
    assert row["perceived_exertion"] == 7
    text = overrides.read_text()
    assert "health_tag" in text.splitlines()[0]
    assert "perceived_exertion" in text.splitlines()[0]
    assert "Confirmed hike" in text


def test_at_least_one_quality_session_type_must_remain_enabled() -> None:
    with pytest.raises(ValueError, match="At least one quality-session type"):
        CoachingSettings(
            training_goal="general_fitness",
            long_run_progression_factor=1.1,
            high_load_ratio=1.3,
            moderate_intensity_leakage_fraction=0.17,
            minimum_days_between_quality_sessions=4,
            quality_recency_reference_days=7,
            typical_rest_days_between_runs=1,
            capacity_retention_half_life_days=42,
            capacity_retention_grace_days=28,
            minimum_running_days_28d_for_quality=8,
            long_run_recency_reference_days=10,
            reduced_volume_factor=0.7,
            quality_sessions={
                "short_intervals": False,
                "long_intervals": False,
                "threshold": False,
                "progression": False,
                "hill_repeats": False,
            },
        )
