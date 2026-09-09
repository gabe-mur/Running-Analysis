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
from run_analysis.workout_detection import detect_structured_workout


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
    connection.execute(
        """CREATE TABLE segments(
            activity_id INTEGER, moving_time_s REAL, average_hr_bpm REAL,
            average_grade_percent REAL, metrics_json TEXT,
            is_pathological INTEGER DEFAULT 0
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


def test_recorded_laps_classify_unlabelled_structured_workout_before_signals() -> None:
    connection = _laps_connection()
    speeds = [2.3, 3.5, 1.8, 3.45, 1.8, 3.6, 2.2]
    intervals = []
    for lap, speed in enumerate(speeds):
        seconds = 300 if lap in {0, 6} else 90
        connection.execute(
            "INSERT INTO laps VALUES (1,?,?,?,?,?)",
            (lap, seconds, speed * seconds, 140, 170),
        )
        intervals.append(_interval(lap, lap, speed, seconds, 140))

    detected = detect_structured_workout(
        connection, 1, intervals, z3_floor=154
    )

    assert detected is not None
    assert detected.workout_type == WorkoutType.INTERVALS
    assert detected.source == "recorded_lap_structure"
    assert detected.confidence == ConfidenceLevel.HIGH


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


def test_structured_interval_prescription_has_a_machine_readable_work_dose() -> None:
    connection = _laps_connection()
    planned_for = datetime(2026, 9, 1, 7, tzinfo=timezone.utc)
    prescription = RecommendationResponse(
        generated_at=planned_for - timedelta(hours=1),
        fitness_state_as_of=planned_for - timedelta(hours=1),
        planned_for=planned_for,
        workout_type=WorkoutType.INTERVALS,
        quality_session_type=QualitySessionType.SHORT_INTERVALS,
        title="Short controlled pickups",
        distance_range_miles=(4.0, 4.5),
        structure=[
            WorkoutStep(
                instruction="8 x 1 minute",
                repetitions=8,
                work_duration_minutes=1,
                recovery_duration_minutes=1.5,
                target_zones=["Z4 effort"],
            )
        ],
        confidence=ConfidenceLevel.MODERATE,
        readiness=ReadinessFlag.READY,
    )
    difficulty = SessionDifficulty(
        distance_miles=4.2,
        moving_minutes=42,
        elapsed_minutes=42,
        stopped_minutes=0,
        zone_load=100,
        zone_breakdown=ZoneBreakdown(
            easy_minutes=30,
            moderate_minutes=3,
            hard_minutes=5,
        ),
        is_quality_session=True,
    )

    analysis = _prescription_analysis(
        connection,
        {"zones": {"z3": [151, 166]}},
        1,
        difficulty,
        prescription,
        timing_delta_hours=0.5,
        distance_delta_miles=0,
        match_confidence="high",
    )

    assert analysis.target_work_minutes == 8
    assert analysis.detected_work_minutes == 8
    assert analysis.execution_status == "Completed as prescribed"


def test_unstructured_quality_text_cannot_pass_from_distance_alone() -> None:
    connection = _laps_connection()
    planned_for = datetime(2026, 9, 1, 7, tzinfo=timezone.utc)
    prescription = RecommendationResponse(
        generated_at=planned_for - timedelta(hours=1),
        fitness_state_as_of=planned_for - timedelta(hours=1),
        planned_for=planned_for,
        workout_type=WorkoutType.INTERVALS,
        quality_session_type=QualitySessionType.SHORT_INTERVALS,
        title="Legacy prose-only intervals",
        distance_range_miles=(4.0, 4.5),
        structure=[WorkoutStep(instruction="Run some intervals", target_zones=["Z4"])],
        confidence=ConfidenceLevel.MODERATE,
        readiness=ReadinessFlag.READY,
    )
    difficulty = SessionDifficulty(
        distance_miles=4.2,
        moving_minutes=42,
        elapsed_minutes=42,
        stopped_minutes=0,
        zone_load=100,
        zone_breakdown=ZoneBreakdown(easy_minutes=42),
        is_quality_session=True,
    )

    analysis = _prescription_analysis(
        connection,
        {"zones": {"z3": [151, 166]}},
        1,
        difficulty,
        prescription,
        timing_delta_hours=0.5,
        distance_delta_miles=0,
        match_confidence="high",
    )

    assert analysis.target_work_minutes is None
    assert analysis.execution_status == "Prescription attempted"


def test_prescribed_aerobic_run_keeps_match_but_flags_hr_divergence() -> None:
    connection = _laps_connection()
    planned_for = datetime(2026, 9, 3, 12, tzinfo=timezone.utc)
    prescription = RecommendationResponse(
        generated_at=planned_for - timedelta(hours=1),
        fitness_state_as_of=planned_for - timedelta(hours=1),
        planned_for=planned_for,
        workout_type=WorkoutType.EASY,
        title="Medium-long aerobic run",
        distance_range_miles=(4.2, 4.8),
        target_zones=["Z1", "Z2"],
        structure=[
            WorkoutStep(
                instruction="Stay primarily in Z1–Z2; no fast finish.",
                target_zones=["Z1", "Z2"],
            )
        ],
        confidence=ConfidenceLevel.MODERATE,
        readiness=ReadinessFlag.READY,
    )
    difficulty = SessionDifficulty(
        distance_miles=4.75,
        moving_minutes=55,
        elapsed_minutes=56,
        stopped_minutes=1,
        zone_load=145,
        zone_breakdown=ZoneBreakdown(
            easy_minutes=35,
            moderate_minutes=12,
            hard_minutes=8,
        ),
    )

    analysis = _prescription_analysis(
        connection,
        {"zones": {"z3": [154, 166]}},
        1,
        difficulty,
        prescription,
        timing_delta_hours=3.5,
        distance_delta_miles=0,
        match_confidence="high",
    )

    assert analysis.workout_type == WorkoutType.EASY
    assert analysis.execution_status == "Distance completed; intensity diverged"
    assert analysis.aerobic_intensity_adherence_percent == pytest.approx(63.64, abs=0.01)
    assert analysis.above_prescribed_intensity_minutes == 20
    assert "counts toward recovery" in analysis.summary
    assert analysis.detected_work_minutes is None


def test_prescribed_aerobic_run_with_normal_spillover_is_completed() -> None:
    connection = _laps_connection()
    planned_for = datetime(2026, 9, 3, 12, tzinfo=timezone.utc)
    prescription = RecommendationResponse(
        generated_at=planned_for - timedelta(hours=1),
        fitness_state_as_of=planned_for - timedelta(hours=1),
        planned_for=planned_for,
        workout_type=WorkoutType.EASY,
        title="Easy aerobic run",
        distance_range_miles=(4.0, 4.5),
        target_zones=["Z1", "Z2"],
        confidence=ConfidenceLevel.MODERATE,
        readiness=ReadinessFlag.READY,
    )
    difficulty = SessionDifficulty(
        distance_miles=4.25,
        moving_minutes=45,
        elapsed_minutes=45,
        stopped_minutes=0,
        zone_load=80,
        zone_breakdown=ZoneBreakdown(
            easy_minutes=38,
            moderate_minutes=7,
            hard_minutes=0,
        ),
    )

    analysis = _prescription_analysis(
        connection,
        {"zones": {"z3": [154, 166]}},
        1,
        difficulty,
        prescription,
        timing_delta_hours=1,
        distance_delta_miles=0,
        match_confidence="high",
    )

    assert analysis.execution_status == "Completed as prescribed"
    assert analysis.aerobic_intensity_adherence_percent == pytest.approx(84.44, abs=0.01)


def test_climbing_context_adjusts_adherence_but_not_recorded_hr_load() -> None:
    connection = _laps_connection()
    # A 1.25 energy ratio attributes 20% of this Z3 segment to climbing.
    connection.execute(
        "INSERT INTO segments VALUES (1,660,160,4.0,?,0)",
        ('{"grade_energy_ratio": 1.25}',),
    )
    planned_for = datetime(2026, 9, 3, 12, tzinfo=timezone.utc)
    prescription = RecommendationResponse(
        generated_at=planned_for - timedelta(hours=1),
        fitness_state_as_of=planned_for - timedelta(hours=1),
        planned_for=planned_for,
        workout_type=WorkoutType.EASY,
        title="Easy aerobic run",
        distance_range_miles=(4.0, 4.5),
        target_zones=["Z1", "Z2"],
        confidence=ConfidenceLevel.MODERATE,
        readiness=ReadinessFlag.READY,
    )
    difficulty = SessionDifficulty(
        distance_miles=4.25,
        moving_minutes=50,
        elapsed_minutes=50,
        stopped_minutes=0,
        zone_load=105,
        zone_breakdown=ZoneBreakdown(
            easy_minutes=39,
            moderate_minutes=11,
            hard_minutes=0,
        ),
    )

    analysis = _prescription_analysis(
        connection,
        {"zones": {"z3": [154, 166]}},
        1,
        difficulty,
        prescription,
        timing_delta_hours=1,
        distance_delta_miles=0,
        match_confidence="high",
    )

    assert analysis.aerobic_intensity_adherence_percent == pytest.approx(78)
    assert analysis.grade_attributed_moderate_minutes == pytest.approx(2.2)
    assert analysis.terrain_adjusted_aerobic_adherence_percent == pytest.approx(82.4)
    assert analysis.effective_above_prescribed_intensity_minutes == pytest.approx(8.8)
    assert analysis.execution_status == "Completed as prescribed"
    # Recovery's recorded difficulty object is intentionally untouched.
    assert difficulty.zone_load == 105
    assert difficulty.zone_breakdown.moderate_minutes == 11


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
