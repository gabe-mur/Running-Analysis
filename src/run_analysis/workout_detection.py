"""Detect structured running from recorded workout boundaries.

Recorded laps are the primary evidence because they preserve the boundaries the
athlete or watch workout actually used.  Pace-stream inference is deliberately
only a fallback for files without a usable lap pattern; heart rate summarizes
the work but never sets short-repetition boundaries because it lags effort.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, median
import sqlite3

from .movement import MovementInterval
from .web.schemas import ConfidenceLevel, WorkoutType


@dataclass(frozen=True, slots=True)
class StructuredWorkoutDetection:
    workout_type: WorkoutType
    source: str
    confidence: ConfidenceLevel


def recorded_laps(
    connection: sqlite3.Connection, activity_id: int
) -> list[sqlite3.Row]:
    try:
        return connection.execute(
            """
            SELECT lap_index,total_time_s,distance_m,average_hr_bpm,
                   maximum_hr_bpm,intensity,trigger_method
            FROM laps WHERE activity_id=? ORDER BY lap_index
            """,
            (activity_id,),
        ).fetchall()
    except sqlite3.OperationalError:
        # Small synthetic databases created by older tooling may carry only the
        # core lap columns. Missing trigger metadata is neutral evidence.
        return connection.execute(
            """
            SELECT lap_index,total_time_s,distance_m,average_hr_bpm,
                   maximum_hr_bpm
            FROM laps WHERE activity_id=? ORDER BY lap_index
            """,
            (activity_id,),
        ).fetchall()


def usable_recorded_laps(rows) -> list:
    return [
        row
        for row in rows
        if float(row["total_time_s"] or 0) > 0
        and float(row["distance_m"] or 0) > 0
    ]


def recorded_interval_work_positions(rows) -> set[int]:
    """Return alternating work laps, using recorded boundaries as ground truth."""

    usable = usable_recorded_laps(rows)
    if len(usable) < 4:
        return set()
    speeds = [
        float(row["distance_m"]) / float(row["total_time_s"])
        for row in usable
    ]
    return {
        position
        for position in range(1, len(usable) - 1)
        if speeds[position] >= speeds[position - 1] * 1.10
        and speeds[position] >= speeds[position + 1] * 1.10
        and float(usable[position]["total_time_s"]) >= 30
        and float(usable[position]["distance_m"]) >= 100
    }


def _manual_boundary(row) -> bool:
    keys = set(row.keys()) if hasattr(row, "keys") else set()
    trigger = row["trigger_method"] if "trigger_method" in keys else None
    # Some older exports omit TriggerMethod. A missing value is neutral; an
    # explicit automatic distance/position lap is not evidence of structure.
    return trigger is None or str(trigger).casefold() == "manual"


def _continuous_quality_lap(rows, z3_floor: float) -> bool:
    """Recognize warm-up / sustained work / cool-down lap structure.

    The manual interior boundary is the primary structural evidence. Pace or HR
    must merely corroborate that the middle lap was work rather than an
    arbitrary marker; neither signal is used to invent its boundaries.
    """

    usable = usable_recorded_laps(rows)
    if len(usable) < 3:
        return False
    for position, row in enumerate(usable[1:-1], start=1):
        duration = float(row["total_time_s"] or 0)
        if not 8 * 60 <= duration <= 40 * 60 or not _manual_boundary(row):
            continue
        speed = float(row["distance_m"]) / duration
        surrounding = [usable[position - 1], usable[position + 1]]
        surrounding_speed = sum(float(item["distance_m"] or 0) for item in surrounding) / max(
            1.0, sum(float(item["total_time_s"] or 0) for item in surrounding)
        )
        hr = row["average_hr_bpm"]
        if speed >= surrounding_speed * 1.05 or (
            hr is not None and float(hr) >= z3_floor
        ):
            return True
    return False


def smoothed_speeds(intervals: list[MovementInterval]) -> list[float]:
    raw = [
        item.distance_m / item.moving_time_s
        if item.moving_time_s > 0 and item.distance_m > 0
        else 0.0
        for item in intervals
    ]
    return [median(raw[max(0, index - 2) : index + 3]) for index in range(len(raw))]


def speed_clusters(values: list[float]) -> tuple[float, float]:
    positive = sorted(value for value in values if value > 0)
    if len(positive) < 10:
        return 0.0, 0.0
    low = positive[len(positive) // 4]
    high = positive[(len(positive) * 3) // 4]
    for _ in range(12):
        low_group = [value for value in positive if abs(value - low) <= abs(value - high)]
        high_group = [value for value in positive if abs(value - low) > abs(value - high)]
        if not low_group or not high_group:
            break
        low, high = mean(low_group), mean(high_group)
    return min(low, high), max(low, high)


def inferred_work_groups(intervals: list[MovementInterval]) -> list[tuple[int, int]]:
    """Infer repeated faster bouts only when lap structure is unavailable."""

    if len(intervals) < 20:
        return []
    smooth = smoothed_speeds(intervals)
    low, high = speed_clusters(smooth)
    if low <= 0 or high / low < 1.12:
        return []
    threshold = (low + high) / 2
    raw_groups: list[tuple[int, int]] = []
    start: int | None = None
    for index, speed in enumerate([*smooth, 0.0]):
        fast = index < len(smooth) and speed >= threshold
        if fast and start is None:
            start = index
        elif not fast and start is not None:
            raw_groups.append((start, index))
            start = None
    merged: list[tuple[int, int]] = []
    for group in raw_groups:
        gap = (
            sum(item.elapsed_s for item in intervals[merged[-1][1] : group[0]])
            if merged
            else None
        )
        if merged and gap is not None and gap <= 15:
            merged[-1] = (merged[-1][0], group[1])
        else:
            merged.append(group)
    return [
        group
        for group in merged
        if 30
        <= sum(item.elapsed_s for item in intervals[group[0] : group[1]])
        <= 600
        and sum(item.distance_m for item in intervals[group[0] : group[1]]) >= 100
    ]


def detect_structured_workout(
    connection: sqlite3.Connection,
    activity_id: int,
    intervals: list[MovementInterval],
    *,
    z3_floor: float,
) -> StructuredWorkoutDetection | None:
    """Classify a structured session, preferring recorded laps over signals."""

    laps = recorded_laps(connection, activity_id)
    if len(recorded_interval_work_positions(laps)) >= 2:
        return StructuredWorkoutDetection(
            WorkoutType.INTERVALS,
            "recorded_lap_structure",
            ConfidenceLevel.HIGH,
        )
    if _continuous_quality_lap(laps, z3_floor):
        return StructuredWorkoutDetection(
            WorkoutType.TEMPO_THRESHOLD,
            "recorded_lap_structure",
            ConfidenceLevel.HIGH,
        )
    if len(inferred_work_groups(intervals)) >= 2:
        return StructuredWorkoutDetection(
            WorkoutType.INTERVALS,
            "pace_stream_fallback",
            ConfidenceLevel.MODERATE,
        )
    return None
