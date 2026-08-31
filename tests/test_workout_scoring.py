from __future__ import annotations

from datetime import datetime, timedelta, timezone
import sqlite3

import pytest

from run_analysis.models import Trackpoint
from run_analysis.movement import MovementInterval
from run_analysis.web.schemas import (
    ConfidenceLevel,
    QualitySessionType,
    ReadinessFlag,
    RecommendationResponse,
    SessionDifficulty,
    WorkoutAnalysis,
    WorkoutStep,
    WorkoutType,
    ZoneBreakdown,
)
from run_analysis.workout_scoring import (
    _prescription_analysis,
    _recorded_continuous_quality_lap,
    analyze_intervals,
)


def _interval(index: int, lap: int, speed: float, seconds: float, hr: int) -> MovementInterval:
    timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=index * seconds)
    distance = speed * seconds
    start = Trackpoint(lap, 0, index, timestamp, None, None, False, None, None, hr, 84, 84, "run_cadence_extension", speed)
    end = Trackpoint(lap, 0, index + 1, timestamp + timedelta(seconds=seconds), None, None, False, None, None, hr + 2, 86, 86, "run_cadence_extension", speed)
    return MovementInterval(
        index=index, start=start, end=end, elapsed_s=seconds, distance_m=distance,
        distance_source="device", device_distance_m=distance, gps_distance_m=None,
        computed_speed_mps=speed, recorded_speed_mps=speed, gps_speed_mps=None,
        moving_time_s=seconds, stopped_time_s=0, very_slow_time_s=0,
        classification="moving", bearing_degrees=None,
    )


def _laps_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.execute(
        """CREATE TABLE laps(
            activity_id INTEGER, lap_index INTEGER, total_time_s REAL,
            distance_m REAL, average_hr_bpm REAL, maximum_hr_bpm REAL
        )"""
    )
    return connection


def test_recorded_laps_reconstruct_work_recovery_and_hr_kinetics() -> None:
    connection = _laps_connection()
    # warmup, work, recovery, work, recovery, work, cooldown
    speeds = [2.3, 3.5, 1.8, 3.45, 1.8, 3.6, 2.2]
    durations = [300, 110, 85, 112, 87, 108, 240]
    intervals = []
    for lap, (speed, seconds) in enumerate(zip(speeds, durations)):
        hr = 140 + lap * 4
        connection.execute(
            "INSERT INTO laps VALUES (1,?,?,?,?,?)",
            (lap, seconds, speed * seconds, hr + 1, hr + 5),
        )
        intervals.append(_interval(lap, lap, speed, seconds, hr))

    result = analyze_intervals(connection, 1, intervals)

    assert result.available
    assert result.source == "recorded_laps"
    assert result.work_repetition_count == 3
    assert result.recovery_repetition_count == 2
    assert [item.kind for item in result.repetitions] == [
        "warmup", "work", "recovery", "work", "recovery", "work", "cooldown"
    ]
    first_work = next(item for item in result.repetitions if item.kind == "work")
    assert first_work.end_hr_bpm is not None
    assert first_work.recovery_hr_drop_bpm is not None
    assert first_work.average_cadence_spm == 170


def test_raw_pace_stream_infers_repetitions_when_manual_laps_are_absent() -> None:
    connection = _laps_connection()
    intervals: list[MovementInterval] = []
    pattern = [(1.8, 12)]
    for _ in range(4):
        pattern.extend([(3.5, 12), (1.8, 12)])
    for speed, count in pattern:
        for _ in range(count):
            index = len(intervals)
            intervals.append(_interval(index, 0, speed, 5, 145 + min(25, index // 8)))

    result = analyze_intervals(connection, 1, intervals)

    assert result.available
    assert result.source == "pace_stream_inference"
    assert result.work_repetition_count == 4
    assert result.confidence.value == "moderate"


def test_workout_analysis_contract_has_four_dimensions_and_no_composite_score() -> None:
    assert "execution" in WorkoutAnalysis.model_fields
    assert "control" in WorkoutAnalysis.model_fields
    assert "stimulus" in WorkoutAnalysis.model_fields
    assert "recovery" in WorkoutAnalysis.model_fields
    assert "score" not in WorkoutAnalysis.model_fields


def test_prescribed_threshold_uses_recorded_work_lap_not_total_zone_bucket() -> None:
    connection = _laps_connection()
    for lap, (seconds, hr) in enumerate(((660, 144), (1081, 168), (896, 152))):
        connection.execute(
            "INSERT INTO laps VALUES (1,?,?,?,?,?)",
            (lap, seconds, 1600, hr, hr + 5),
        )
    planned_for = datetime(2026, 8, 30, 19, tzinfo=timezone.utc)
    prescription = RecommendationResponse(
        generated_at=planned_for - timedelta(hours=1),
        fitness_state_as_of=planned_for - timedelta(hours=1),
        planned_for=planned_for,
        workout_type=WorkoutType.TEMPO_THRESHOLD,
        quality_session_type=QualitySessionType.THRESHOLD,
        title="Continuous threshold run",
        distance_range_miles=(4.5, 5.0),
        structure=[
            WorkoutStep(
                instruction="Warm up",
                duration_minutes=12,
                target_zones=["Z1", "Z2"],
            ),
            WorkoutStep(
                instruction="Threshold",
                duration_minutes=18,
                target_zones=["upper Z3", "low Z4"],
            ),
            WorkoutStep(
                instruction="Cool down",
                duration_minutes=10,
                target_zones=["Z1", "Z2"],
            ),
        ],
        confidence=ConfidenceLevel.MODERATE,
        readiness=ReadinessFlag.READY,
    )
    difficulty = SessionDifficulty(
        distance_miles=4.41,
        moving_minutes=43.6,
        elapsed_minutes=44,
        stopped_minutes=0.4,
        zone_load=116.6,
        zone_breakdown=ZoneBreakdown(
            easy_minutes=21.5,
            moderate_minutes=6.7,
            hard_minutes=13.9,
        ),
        is_quality_session=True,
    )

    analysis = _prescription_analysis(
        connection,
        {"zones": {"z3": [151, 166]}},
        1,
        difficulty,
        prescription,
        timing_delta_hours=0.6,
        distance_delta_miles=0.09,
        match_confidence="high",
    )

    assert analysis.execution_status == "Completed as prescribed"
    assert analysis.target_work_minutes == 18
    assert analysis.detected_work_minutes == pytest.approx(18.016, abs=0.01)
    assert analysis.detection_source == "recorded_lap_2"
    assert analysis.confidence == ConfidenceLevel.HIGH


def test_threshold_analysis_detects_sustained_manual_lap_without_saved_plan() -> None:
    connection = _laps_connection()
    for lap, (seconds, hr) in enumerate(((660, 144), (1081, 168), (896, 152))):
        connection.execute(
            "INSERT INTO laps VALUES (1,?,?,?,?,?)",
            (lap, seconds, 1600, hr, hr + 5),
        )

    detected = _recorded_continuous_quality_lap(
        connection,
        {"zones": {"z3": [151, 166]}},
        1,
    )

    assert detected == pytest.approx((18.016, 168, 2), abs=0.01)



def test_an_ordinary_run_gets_no_progression_advice() -> None:
    """"Use the weekly plan and see how you feel" is true of every run ever
    done. Printing it on all of them buries the sessions that say something."""
    from run_analysis.web.schemas import (
        ConfidenceLevel,
        DriftAssessment,
        SessionDifficulty,
        WorkoutType,
        ZoneBreakdown,
    )
    from run_analysis.workout_scoring import _generic_analysis

    difficulty = SessionDifficulty(
        distance_miles=5.0,
        moving_minutes=50.0,
        elapsed_minutes=51.0,
        stopped_minutes=1.0,
        zone_load=100.0,
        zone_breakdown=ZoneBreakdown(),
        is_long_run=False,
        is_quality_session=False,
        difficulty_flags=[],
    )
    analysis = _generic_analysis(
        WorkoutType.EASY,
        difficulty,
        DriftAssessment(
            valid=False,
            decoupling_percent=None,
            reason="not enough steady running",
            confidence=ConfidenceLevel.UNAVAILABLE,
        ),
        [],
    )
    assert analysis.progression_recommendation is None
