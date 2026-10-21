"""Shared detection of structured continuous-quality workout phases."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import sqlite3
from .workout_detection import recorded_laps,usable_recorded_laps,recorded_continuous_work


@dataclass(frozen=True)
class ContinuousQualityPhase:
    lap_index: int
    start_offset_seconds: float
    duration_seconds: float
    distance_miles: float | None
    average_hr_bpm: float | None
    maximum_hr_bpm: float | None
    structure: str = 'interior_work'

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
    """Find recorded sustained work, including a progression without cooldown.

    This is the same deliberately conservative rule used by workout analysis:
    work is an 8--40 minute lap with at least Z3 average HR. A terminal
    progression additionally needs an explicit manual boundary and both pace
    and HR contrast with the preceding warm-up.
    The elapsed offset lets the fitness model reject every post-work window,
    including a cooldown whose pace and HR have merely stabilized while the
    athlete is still recovering.
    """

    laps=usable_recorded_laps(recorded_laps(connection,activity_id))
    z3_floor = float(config["zones"]["z3"][0])
    found=recorded_continuous_work(laps,z3_floor)
    if found is None:
        return None
    position,structure=found
    selected=laps[position]
    if selected['average_hr_bpm'] is None or float(selected['average_hr_bpm'])<z3_floor:
        return None
    offset=sum(float(row['total_time_s']) for row in laps[:position])
    if 'start_time_utc' in selected.keys() and selected['start_time_utc'] and laps[0]['start_time_utc']:
        # FIT timer duration excludes pauses; a recorded start is the correct
        # elapsed offset rather than silently shifting work into an earlier lap.
        offset=(datetime.fromisoformat(str(selected['start_time_utc']))-
            datetime.fromisoformat(str(laps[0]['start_time_utc']))).total_seconds()
    distance_m = float(selected["distance_m"] or 0)
    return ContinuousQualityPhase(
        lap_index=int(selected["lap_index"]),
        start_offset_seconds=offset,
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
        structure=structure,
    )
