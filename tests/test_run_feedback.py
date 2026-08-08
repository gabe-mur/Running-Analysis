from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from run_analysis.db import connect, initialize
from run_analysis.models import Trackpoint
from run_analysis.movement import MovementInterval
from run_analysis.run_feedback import _feedback_text, assess_cardiac_drift, build_mile_splits
from run_analysis.web.app import create_app
from run_analysis.web.schemas import (
    ActivityHealthTag,
    ConfidenceLevel,
    DataQuality,
    DriftAssessment,
    RunMetadata,
    RunSummary,
    SessionDifficulty,
    WorkoutType,
    ZoneBreakdown,
)
from test_web_phase1 import _write_config


def _point(index: int, seconds: int, distance: float, hr: int = 145) -> Trackpoint:
    return Trackpoint(
        lap_index=0,
        track_index=0,
        point_index=index,
        timestamp_utc=datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=seconds),
        latitude=None,
        longitude=None,
        gps_valid=False,
        altitude_m=10 + index,
        distance_m=distance,
        heart_rate_bpm=hr,
        cadence=None,
        run_cadence=None,
        cadence_source=None,
        speed_mps=None,
    )


def _interval(index: int, start_distance: float, distance: float, seconds: float, hr: int = 145) -> MovementInterval:
    start = _point(index, int(index * seconds), start_distance, hr)
    end = _point(index + 1, int((index + 1) * seconds), start_distance + distance, hr)
    return MovementInterval(
        index=index,
        start=start,
        end=end,
        elapsed_s=seconds,
        distance_m=distance,
        distance_source="device",
        device_distance_m=distance,
        gps_distance_m=None,
        computed_speed_mps=distance / seconds,
        recorded_speed_mps=None,
        gps_speed_mps=None,
        moving_time_s=seconds,
        stopped_time_s=0,
        very_slow_time_s=0,
        classification="moving",
        bearing_degrees=None,
        elevation_delta_m=1,
    )


def test_raw_intervals_are_split_at_exact_mile_boundaries() -> None:
    intervals = [_interval(index, index * 500, 500, 300) for index in range(7)]
    splits = build_mile_splits(intervals)
    assert len(splits) == 3
    assert splits[0].distance_miles == pytest.approx(1)
    assert splits[1].distance_miles == pytest.approx(1)
    assert splits[2].is_partial
    assert sum(split.distance_miles for split in splits) == pytest.approx(3500 / 1609.344)


def test_drift_rejects_short_or_variable_intensity_runs() -> None:
    short = [_interval(index, index * 100, 100, 60) for index in range(20)]
    assert not assess_cardiac_drift(short, WorkoutType.EASY).valid
    long = [_interval(index, index * 100, 100, 60) for index in range(40)]
    result = assess_cardiac_drift(long, WorkoutType.INTERVALS)
    assert not result.valid
    assert "Variable-intensity" in result.reason


def test_assessment_is_a_run_specific_single_sentence() -> None:
    difficulty = SessionDifficulty(
        distance_miles=4.4,
        moving_minutes=49,
        elapsed_minutes=59,
        stopped_minutes=10,
        zone_load=96,
        zone_breakdown=ZoneBreakdown(
            zone_seconds={}, zone_fractions={}, easy_minutes=38, moderate_minutes=9.5, hard_minutes=0
        ),
    )
    summary = RunSummary(
        activity_id=1, activity_uid="uid", start_time=None, distance_miles=4.4,
        moving_minutes=49, moving_pace_min_mile=11.1, average_hr_bpm=148,
        gps_quality="gps_complete", assessment_label="Aerobic run",
        workout_type=WorkoutType.EASY, health_tag=ActivityHealthTag.NORMAL,
        data_quality=DataQuality.GOOD, session_difficulty=difficulty,
    )
    assessment, _, _ = _feedback_text(
        summary,
        RunMetadata(workout_type=WorkoutType.EASY),
        DriftAssessment(valid=False, confidence=ConfidenceLevel.LOW, reason="Stops"),
    )

    assert assessment.startswith("Mostly aerobic run")
    assert "stopping" in assessment
    assert assessment.endswith(".") and ". " not in assessment
    assert "evaluated with" not in assessment


def test_run_list_endpoint_exposes_session_difficulty_separately(tmp_path: Path) -> None:
    _write_config(tmp_path)
    database = tmp_path / "data" / "test.sqlite"
    with connect(database) as connection:
        initialize(connection)
        connection.execute(
            """
            INSERT INTO activities(
                activity_uid,activity_id,sport,start_time_utc,start_time_utc_epoch,
                total_elapsed_time_s,total_distance_m,lap_count,trackpoint_count,
                gps_quality,hr_quality,elevation_quality,cadence_quality,distance_source,
                namespaces_json,data_quality_json,created_at_utc,updated_at_utc
            ) VALUES ('uid','external','Running','2026-01-01T12:00:00+00:00',1767268800,
                      3600,8046.72,0,0,'gps_complete','hr_complete','complete','missing','device',
                      '{}','{}','now','now')
            """
        )
        activity_id = connection.execute("SELECT id FROM activities").fetchone()[0]
        connection.execute(
            """
            INSERT INTO activity_metrics(
                activity_id,metrics_json,calculated_at_utc,calculated_moving_time_s,
                elapsed_time_s,moving_pace_min_mile,moving_average_hr_bpm,
                analysis_distance_m,hr_zone_seconds_json,diagnostics_json
            ) VALUES (?, '{}','now',3600,3600,12,145,8046.72,
                      '{"z1":600,"z2":1800,"z3":1200,"z4":0,"z5":0,"below_z1":0,"above_z5":0,"unknown":0}', '{}')
            """,
            (activity_id,),
        )
        connection.commit()
    response = TestClient(create_app(tmp_path)).get("/api/runs")
    assert response.status_code == 200
    run = response.json()[0]
    assert run["distance_miles"] == pytest.approx(5)
    assert run["session_difficulty"]["zone_load"] == pytest.approx(10 + 60 + 60)
    assert run["fitness_observation"] is None
    assert run["workout_type"] == "easy"
    assert run["assessment_label"] == "Moderate, not easy"
    assert run["session_difficulty"]["is_quality_session"] is False
