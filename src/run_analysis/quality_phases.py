"""Shared detection of structured continuous-quality workout phases."""

from __future__ import annotations

from dataclasses import dataclass
import sqlite3


@dataclass(frozen=True)
class ContinuousQualityPhase:
    lap_index: int
    start_offset_seconds: float
    duration_seconds: float
    distance_miles: float | None
    average_hr_bpm: float | None
    maximum_hr_bpm: float | None

    @property
    def end_offset_seconds(self) -> float:
        return self.start_offset_seconds + self.duration_seconds

    @property
    def pace_min_mile(self) -> float | None:
        if not self.distance_miles or self.distance_miles <= 0:
            return None
        return self.duration_seconds / 60.0 / self.distance_miles


def detect_continuous_quality_phase(
    connection: sqlite3.Connection,
    config: dict,
    activity_id: int,
) -> ContinuousQualityPhase | None:
    """Find a sustained interior work lap bounded by warm-up and cool-down.

    This is the same deliberately conservative rule used by workout analysis:
    the work must be an interior 8--40 minute lap with at least Z3 average HR.
    The elapsed offset lets the fitness model reject every post-work window,
    including a cooldown whose pace and HR have merely stabilized while the
    athlete is still recovering.
    """

    laps = connection.execute(
        """
        SELECT lap_index,total_time_s,distance_m,average_hr_bpm,maximum_hr_bpm
        FROM laps WHERE activity_id=? ORDER BY lap_index
        """,
        (activity_id,),
    ).fetchall()
    if len(laps) < 3:
        return None
    z3_floor = float(config["zones"]["z3"][0])
    candidates = [
        (position, row)
        for position, row in enumerate(laps[1:-1], start=1)
        if 8 * 60 <= float(row["total_time_s"] or 0) <= 40 * 60
        and row["average_hr_bpm"] is not None
        and float(row["average_hr_bpm"]) >= z3_floor
    ]
    if not candidates:
        return None
    position, selected = max(
        candidates,
        key=lambda item: (
            float(item[1]["average_hr_bpm"]),
            float(item[1]["total_time_s"]),
        ),
    )
    distance_m = float(selected["distance_m"] or 0)
    return ContinuousQualityPhase(
        lap_index=int(selected["lap_index"]),
        start_offset_seconds=sum(float(row["total_time_s"] or 0) for row in laps[:position]),
        duration_seconds=float(selected["total_time_s"]),
        distance_miles=distance_m / 1609.344 if distance_m > 0 else None,
        average_hr_bpm=(
            float(selected["average_hr_bpm"])
            if selected["average_hr_bpm"] is not None
            else None
        ),
        maximum_hr_bpm=(
            float(selected["maximum_hr_bpm"])
            if selected["maximum_hr_bpm"] is not None
            else None
        ),
    )
