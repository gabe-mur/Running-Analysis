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
from fastapi.testclient import TestClient

from run_analysis.web.app import create_app
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


def test_setup_endpoint_reports_what_is_still_defaulted(tmp_path: Path) -> None:
    _write_config(tmp_path)
    database = tmp_path / "data" / "test.sqlite"
    with connect(database) as connection:
        initialize(connection)
    client = TestClient(create_app(tmp_path))
    payload = client.get("/api/setup").json()
    assert payload["complete"] is False
    assert payload["next_step"] == "runs"
    assert any(step["blocking"] for step in payload["steps"])


def test_zone_preview_shows_the_effect_before_anything_is_saved(tmp_path: Path) -> None:
    """The point of a preview is that moving Z2 moves where the evidence for a
    comparison heart rate is, and the athlete should see that first."""
    _write_config(tmp_path)
    database = tmp_path / "data" / "test.sqlite"
    with connect(database) as connection:
        initialize(connection)
    client = TestClient(create_app(tmp_path))
    before = client.get("/api/settings").json()["zones"]
    response = client.post(
        "/api/setup/zone-preview",
        json={"method": "heart_rate_reserve", "max_hr": 194, "resting_hr": 49},
    )
    assert response.status_code == 200
    preview = response.json()
    assert preview["zones"]["z5"]["maximum_bpm"] == 194
    assert "comparison_hr" in preview
    assert client.get("/api/settings").json()["zones"] == before


def test_zone_preview_rejects_reserve_without_a_resting_heart_rate(tmp_path: Path) -> None:
    _write_config(tmp_path)
    database = tmp_path / "data" / "test.sqlite"
    with connect(database) as connection:
        initialize(connection)
    client = TestClient(create_app(tmp_path))
    response = client.post(
        "/api/setup/zone-preview", json={"method": "heart_rate_reserve", "max_hr": 194}
    )
    assert response.status_code == 422


def test_confirmations_persist_and_accumulate(tmp_path: Path) -> None:
    """Confirming one step must not silently un-confirm the others."""
    _write_config(tmp_path)
    database = tmp_path / "data" / "test.sqlite"
    with connect(database) as connection:
        initialize(connection)
    client = TestClient(create_app(tmp_path))
    client.patch("/api/settings", json={"setup": {"confirmed_steps": ["heart_rate"]}})
    client.patch(
        "/api/settings",
        json={"setup": {"confirmed_steps": ["heart_rate", "zones"], "zone_method": "device"}},
    )
    setup = client.get("/api/settings").json()["setup"]
    assert set(setup["confirmed_steps"]) == {"heart_rate", "zones"}
    assert setup["zone_method"] == "device"


def test_the_estimated_max_endpoint_states_its_own_uncertainty(tmp_path: Path) -> None:
    _write_config(tmp_path)
    client = TestClient(create_app(tmp_path))
    payload = client.get("/api/setup/max-hr", params={"age_years": 30}).json()
    assert payload["estimated_max_hr"] == 187
    assert "not a measurement" in payload["caveat"]


def test_reset_restores_tuning_but_never_personal_settings(tmp_path: Path) -> None:
    """A button labelled "reset" must not be able to erase someone's max heart
    rate. It clears modelling parameters and nothing else."""
    _write_config(tmp_path)
    database = tmp_path / "data" / "test.sqlite"
    with connect(database) as connection:
        initialize(connection)
    client = TestClient(create_app(tmp_path))
    shipped = client.get("/api/settings").json()

    client.patch(
        "/api/settings",
        json={
            "max_hr": 201,
            "target_hr": 151,
            "reference_temperature_f": 80.0,
            "moving_time": {**shipped["moving_time"], "minimum_stop_seconds": 42},
            "coaching": {
                **shipped["coaching"],
                "high_load_ratio": 1.9,
                "quality_sessions": {**shipped["coaching"]["quality_sessions"], "hill_repeats": True},
            },
        },
    )
    changed = client.get("/api/settings").json()
    assert changed["max_hr"] == 201 and changed["reference_temperature_f"] == 80.0

    after = client.post("/api/settings/reset-advanced").json()
    # Tuning is back to shipped values.
    assert after["reference_temperature_f"] == shipped["reference_temperature_f"]
    assert after["moving_time"]["minimum_stop_seconds"] == shipped["moving_time"]["minimum_stop_seconds"]
    assert after["coaching"]["high_load_ratio"] == shipped["coaching"]["high_load_ratio"]
    # Everything that describes the athlete survives.
    assert after["max_hr"] == 201
    assert after["target_hr"] == 151
    assert after["coaching"]["quality_sessions"]["hill_repeats"] is True


def test_reset_is_safe_to_run_when_nothing_was_ever_changed(tmp_path: Path) -> None:
    _write_config(tmp_path)
    database = tmp_path / "data" / "test.sqlite"
    with connect(database) as connection:
        initialize(connection)
    client = TestClient(create_app(tmp_path))
    before = client.get("/api/settings").json()
    assert client.post("/api/settings/reset-advanced").status_code == 200
    after = client.get("/api/settings").json()
    assert after["moving_time"] == before["moving_time"]
    assert after["zones"] == before["zones"]


def test_setup_saves_everything_in_one_request(tmp_path: Path) -> None:
    """Setup writes once. Saving step by step let the page show a comparison
    heart rate that no longer matched the zones already stored."""
    _write_config(tmp_path)
    database = tmp_path / "data" / "test.sqlite"
    with connect(database) as connection:
        initialize(connection)
    client = TestClient(create_app(tmp_path))
    shipped = client.get("/api/settings").json()
    response = client.patch(
        "/api/settings",
        json={
            "max_hr": shipped["max_hr"],
            "resting_hr": 49,
            "target_hr": 148,
            "zones": shipped["zones"],
            "coaching": shipped["coaching"],
            "historical_weather_enabled": True,
            "forecast_weather_enabled": False,
            "weather_privacy_radius_km": 2.0,
            "setup": {
                "confirmed_steps": ["heart_rate", "zones", "comparison_hr", "goal", "weather"],
                "zone_method": "device",
            },
        },
    )
    assert response.status_code == 200
    saved = response.json()
    assert saved["target_hr"] == 148
    assert saved["resting_hr"] == 49
    assert saved["setup"]["zone_method"] == "device"
    assert len(saved["setup"]["confirmed_steps"]) == 5
    # And it survives a reload rather than only living in the response.
    assert client.get("/api/settings").json()["target_hr"] == 148
