"""Detect structured running from recorded workout boundaries.

Recorded laps are authoritative evidence because they preserve the boundaries
the athlete or watch workout actually used. Pace-stream inference can still
reconstruct tentative repetitions for run-detail analysis, but it must not
relabel a whole run as quality by itself. Heart rate summarizes the work but
never sets short-repetition boundaries because it lags effort.
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
    # Optional trigger/intensity/timestamp columns must remain visible when
    # present; old small synthetic tables need not manufacture those columns.
    return connection.execute('SELECT * FROM laps WHERE activity_id=? ORDER BY lap_index',
        (activity_id,)).fetchall()


def usable_recorded_laps(rows) -> list:
    return [
        row
        for row in rows
        if float(row["total_time_s"] or 0) > 0
        and float(row["distance_m"] or 0) > 0
    ]


def _lap_value(row, name):
    return row[name] if name in row.keys() else None


def recorded_interval_work_positions(rows, intervals=()) -> set[int]:
    """Return alternating work laps, using recorded boundaries as ground truth."""

    usable = usable_recorded_laps(rows)
    if len(usable) < 4:
        return set()
    speeds = [
        float(row["distance_m"]) / float(row["total_time_s"])
        for row in usable
    ]
    intensity = [str(_lap_value(row, 'intensity') or '').casefold() for row in usable]
    # Recorded Active/Rest alternation is useful for hills, whose uphill work
    # may be SLOWER than the downhill recovery. All-Active auto laps are neutral.
    declared = {p for p in range(1, len(usable)-1)
        if intensity[p] == 'active' and intensity[p+1] in {'rest', 'resting'}
        and float(usable[p]['total_time_s']) >= 30}
    if len(declared)>=2:
        return declared # Declared recovery must not become work just because it is faster downhill.
    by_lap = {}
    for item in intervals:
        if item.start.lap_index == item.end.lap_index:
            by_lap.setdefault(item.start.lap_index, []).append(item)
    def grade(position):
        items=by_lap.get(usable[position]['lap_index'], ())
        values=[i for i in items if i.start.altitude_m is not None and i.end.altitude_m is not None
            and not {'large_recording_gap','implausible_distance_speed','distance_reset'} & set(i.flags)]
        distance=sum(i.distance_m for i in values)
        gain=sum(i.end.altitude_m-i.start.altitude_m for i in values)
        return gain/distance if distance>0 and abs(gain)>=3 else 0.
    selected=set();hills=set()
    for p in range(1,len(usable)-1):
        row=usable[p]
        if float(row['total_time_s']) < 30:
            continue
        if p in declared:
            selected.add(p)
            continue
        if intensity[p] in {'rest','resting'}:
            continue
        if not _manual_boundary(row):
            continue
        contrast=speeds[p]>=speeds[p-1]*1.10 and speeds[p]>=speeds[p+1]*1.10
        # A manually bounded 45-second hill/pickup need not cover100meters.
        minimum_distance=30 if str(_lap_value(row,'trigger_method') or '').casefold()=='manual' else 100
        uphill,following=grade(p),grade(p+1)
        hill=uphill>=.02 and following<=-.02
        downhill_recovery=uphill<=-.02 and following>=.02
        if float(row['distance_m'])>=minimum_distance:
            if hill:
                hills.add(p)
            elif contrast and not downhill_recovery:
                selected.add(p)
    return hills if len(hills)>=2 else selected|hills


def _manual_boundary(row) -> bool:
    keys = set(row.keys()) if hasattr(row, "keys") else set()
    trigger = row["trigger_method"] if "trigger_method" in keys else None
    # Some older exports omit TriggerMethod. A missing value is neutral; an
    # explicit automatic distance/position lap is not evidence of structure.
    return trigger is None or str(trigger).casefold() == "manual"


def recorded_continuous_work(rows, z3_floor: float):
    """Recognize recorded sustained work with or without a cooldown lap.

    The manual interior boundary is the primary structural evidence. Pace or HR
    must merely corroborate that the middle lap was work rather than an
    arbitrary marker; neither signal is used to invent its boundaries. A finish
    without cooldown requires both contrasts and an explicit manual boundary.
    """

    usable = usable_recorded_laps(rows)
    if len(usable) < 2:
        return None
    candidates=[]
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
            candidates.append((position,'interior_work'))
    # Progressions often end at the work effort: no artificial cooldown lap
    # is required. The boundary ending the warmup must be explicitly manual;
    # an auto-mile split plus HR drift must not turn an easy run into quality.
    warm,work=usable[:-1],usable[-1]
    seconds=float(work['total_time_s'])
    warm_seconds=sum(float(r['total_time_s']) for r in warm)
    known_warm=[r for r in warm if r['average_hr_bpm'] is not None]
    warm_hr=(sum(float(r['average_hr_bpm'])*float(r['total_time_s']) for r in known_warm)
        /sum(float(r['total_time_s']) for r in known_warm)) if known_warm else None
    warm_speed=sum(float(r['distance_m']) for r in warm)/warm_seconds
    hr=work['average_hr_bpm']
    if (8*60<=seconds<=40*60 and warm_seconds>=3*60
            and str(_lap_value(warm[-1],'trigger_method') or '').casefold()=='manual'
            and hr is not None and warm_hr is not None
            and float(hr)>=z3_floor and float(hr)>=warm_hr+5
            and float(work['distance_m'])/seconds>=warm_speed*1.05):
        candidates.append((len(usable)-1,'progression_finish'))
    return max(candidates,key=lambda p:(float(usable[p[0]]['average_hr_bpm'] or 0),
        float(usable[p[0]]['total_time_s']))) if candidates else None


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
    """Classify a structured session only from recorded workout boundaries."""

    laps = recorded_laps(connection, activity_id)
    if len(recorded_interval_work_positions(laps,intervals)) >= 2:
        return StructuredWorkoutDetection(
            WorkoutType.INTERVALS,
            "recorded_lap_structure",
            ConfidenceLevel.HIGH,
        )
    continuous=recorded_continuous_work(laps,z3_floor)
    if continuous is not None:
        return StructuredWorkoutDetection(
            WorkoutType.TEMPO_THRESHOLD,
            "recorded_progression_laps" if continuous[1]=='progression_finish' else "recorded_lap_structure",
            ConfidenceLevel.HIGH,
        )
    return None
