from __future__ import annotations

from datetime import datetime, timedelta, timezone
import sqlite3

from run_analysis.models import Trackpoint
from run_analysis.movement import MovementInterval
from run_analysis.web.schemas import WorkoutAnalysis
from run_analysis.workout_scoring import analyze_intervals


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

