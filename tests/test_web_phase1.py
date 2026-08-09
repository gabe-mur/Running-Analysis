from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import sqlite3

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from run_analysis.db import SCHEMA_VERSION, connect, initialize
from run_analysis.web.app import create_app
from run_analysis.web.schemas import (
    ActivityHealthTag,
    ConfidenceLevel,
    CurrentHealthStatus,
    FitnessState,
    FitnessTrend,
    LoadContext,
    LoadWindow,
    RecommendationRequest,
    RunMetadataPatch,
)


def _write_config(root: Path, database: str = "data/test.sqlite") -> None:
    (root / "config.yaml").write_text(
        """
max_hr: 195
resting_hr: 50
target_hr: 145
zones:
  z1: [128, 140]
  z2: [141, 153]
  z3: [154, 166]
  z4: [167, 180]
  z5: [181, 195]
timezone_default: America/New_York
paths:
  database: %s
  report: output/report.html
  weather_cache: data/weather_cache
  overrides: run_overrides.csv
reference_conditions:
  temperature_f: 55
  dewpoint_f: 45
  wind_mph: 3
  grade_percent: 0
  within_run_minutes: 22.5
segment_distance_miles: 0.25
moving_time:
  minimum_running_speed_mps: 0.8
  stopped_speed_mps: 0.35
  gps_stopped_speed_mps: 0.8
  stopped_distance_meters: 1.5
  maximum_interval_seconds: 30
  minimum_stop_seconds: 5
  maximum_plausible_speed_mps: 12
elevation:
  smoothing_window_meters: 60
  grade_cost_window_meters: 60
  minimum_gain_change_meters: 0.5
  minimum_grade_distance_meters: 100
  maximum_plausible_grade_percent: 12
segmentation:
  minimum_final_segment_fraction: 0.5
  minimum_plausible_pace_min_mile: 3
  maximum_plausible_pace_min_mile: 30
  minimum_bearing_distance_meters: 20
model:
  window_seconds: 300
  window_sensitivity_seconds: [60, 120, 180, 240, 300]
  fixed_heat_loss_fraction_per_c: 0.002
  fixed_heat_loss_uncertainty_fraction_per_c: 0.002
  heat_personal_match_max_days: 56
  heat_personal_match_min_wbgt_delta_c: 3
  heat_personal_minimum_se_fraction_per_c: 0.0005
  grouped_cv_folds: 5
  ridge_alpha: 1
  minimum_cv_improvement_seconds_per_mile: 2
  run_effect_max_effective_segments: 4
  minimum_run_miles: 1.5
  maximum_stop_fraction: 0.35
  minimum_hr_coverage: 0.8
  minimum_gps_coverage: 0.8
  minimum_reliable_segment_minutes: 5
  maximum_reliable_segment_minutes: 60
  minimum_reliable_hr_bpm: 128
  maximum_reliable_hr_bpm: 166
  maximum_reliable_segment_stop_fraction: 0.15
  # Heart rate drops during a stop and needs minutes of running to catch
  # back up. Windows overlapping that recovery look artificially efficient.
  post_pause_minimum_stop_seconds: 60
  post_pause_suppression_moving_seconds: 180
  random_seed: 145
activity_classification:
  high_confidence_walk_pace_min_mile: 18
  high_confidence_walk_cadence_max_spm: 110
  high_confidence_bike_pace_min_mile: 5.5
  review_slow_pace_min_mile: 13
  very_low_cadence_max_spm: 130
  review_low_cadence_max_spm: 140
  minimum_review_score: 3
  review_limit: 3
weather:
  coordinate_precision: 2
  privacy_jitter_radius_km: 2
  privacy_salt_path: data/weather_privacy_salt
  provider: open_meteo
  endpoint: https://archive-api.open-meteo.com/v1/archive
  model: best_match
  request_timeout_seconds: 1
""" % database,
        encoding="utf-8",
    )


def _window(days: int) -> LoadWindow:
    return LoadWindow(
        days=days,
        distance_miles=10,
        moving_minutes=100,
        zone_load=210,
        hard_minutes=8,
        activity_count=3,
    )


def test_fitness_state_keeps_fitness_and_load_as_distinct_contracts() -> None:
    state = FitnessState(
        as_of=datetime.now(timezone.utc),
        window_days=28,
        fitness_trend=FitnessTrend.STABLE,
        trend_confidence=ConfidenceLevel.MODERATE,
        recent_load=LoadContext(
            trailing_7d=_window(7),
            trailing_14d=_window(14),
            trailing_28d=_window(28),
            acute_to_prior_ratio=1.1,
            confidence=ConfidenceLevel.MODERATE,
        ),
    )
    dumped = state.model_dump(mode="json")
    assert dumped["fitness_trend"] == "stable"
    assert dumped["recent_load"]["trailing_28d"]["zone_load"] == 210
    assert dumped["standardized_pace_at_target_hr"] is None


def test_metadata_and_health_status_are_enumerated() -> None:
    patch = RunMetadataPatch(health_tag=ActivityHealthTag.ILLNESS_RECOVERY)
    assert patch.health_tag == ActivityHealthTag.ILLNESS_RECOVERY
    request = RecommendationRequest(health_status=CurrentHealthStatus.LITTLE_TIRED)
    assert request.health_status == CurrentHealthStatus.LITTLE_TIRED
    with pytest.raises(ValidationError):
        RunMetadataPatch()
    with pytest.raises(ValidationError):
        RecommendationRequest(health_status="fine")


def test_health_endpoint_reports_initialized_database(tmp_path: Path) -> None:
    _write_config(tmp_path)
    database = tmp_path / "data" / "test.sqlite"
    with connect(database) as connection:
        initialize(connection)
        connection.execute(
            """
            INSERT INTO activities(
                activity_uid, sport, lap_count, trackpoint_count, gps_quality,
                hr_quality, elevation_quality, cadence_quality, distance_source,
                namespaces_json, data_quality_json, created_at_utc, updated_at_utc
            ) VALUES ('test', 'Running', 0, 0, 'missing', 'missing', 'missing',
                      'missing', 'missing', '{}', '{}', 'now', 'now')
            """
        )
        connection.commit()

    client = TestClient(create_app(tmp_path))
    response = client.get("/api/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ready"
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["counts"]["activities"] == 1


def test_health_endpoint_does_not_create_a_missing_database(tmp_path: Path) -> None:
    _write_config(tmp_path)
    database = tmp_path / "data" / "test.sqlite"
    response = TestClient(create_app(tmp_path)).get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "setup_required"
    assert not database.exists()


def test_new_user_can_enable_historical_weather_without_a_profile(tmp_path: Path) -> None:
    _write_config(tmp_path)
    client = TestClient(create_app(tmp_path))
    response = client.patch("/api/settings", json={"historical_weather_enabled": True})
    assert response.status_code == 200
    assert response.json()["historical_weather_enabled"] is True


def test_race_goal_settings_require_ten_usable_runs_before_save(tmp_path: Path) -> None:
    _write_config(tmp_path)
    database = tmp_path / "data" / "test.sqlite"
    with connect(database) as connection:
        initialize(connection)
    client = TestClient(create_app(tmp_path))
    coaching = client.get("/api/settings").json()["coaching"]
    coaching.update(
        {
            "training_goal": "10k",
            "goal_date": (date.today() + timedelta(weeks=12)).isoformat(),
            "goal_pace_min_mile": 9.0,
        }
    )

    response = client.patch("/api/settings", json={"coaching": coaching})

    assert response.status_code == 422
    assert "requires 10 usable" in response.json()["detail"]
    assert not (tmp_path / "config.local.yaml").exists()


def test_static_app_and_openapi_are_served(tmp_path: Path) -> None:
    _write_config(tmp_path)
    client = TestClient(create_app(tmp_path))
    page = client.get("/")
    assert page.status_code == 200
    assert "Running Coach" in page.text
    assert "Weekly Plan" in page.text
    schema = client.get("/openapi.json").json()
    assert "/api/health" in schema["paths"]
    assert "HealthResponse" in schema["components"]["schemas"]


def test_private_browser_headers_and_local_host_allowlist(tmp_path: Path) -> None:
    _write_config(tmp_path)
    client = TestClient(create_app(tmp_path))
    response = client.get("/")
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert response.headers["x-frame-options"] == "DENY"
    assert "connect-src 'self'" in response.headers["content-security-policy"]

    rejected = client.get("/api/settings", headers={"host": "run-data.attacker.example"})
    assert rejected.status_code == 400
