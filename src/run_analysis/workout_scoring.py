"""Inspectable, workout-specific analysis.

The four dimensions intentionally remain separate.  There is no composite
score: execution, pacing control, cardiovascular stimulus, and recovery can
disagree without one hiding the others.
"""

from __future__ import annotations

from datetime import datetime
from math import log
from statistics import mean, median, pstdev
import sqlite3

from .movement import MovementInterval
from .processing import _load_points
from .segmentation import METERS_PER_MILE
from .web.schemas import (
    ConfidenceLevel,
    DriftAssessment,
    HistoricalWorkoutComparison,
    IntervalAnalysis,
    IntervalRepetition,
    SessionDifficulty,
    Split,
    WorkoutAnalysis,
    WorkoutAnalysisDimension,
    WorkoutAnalysisMetric,
    WorkoutType,
)


def _pace(seconds: float, distance_m: float) -> float | None:
    return seconds / 60 / (distance_m / METERS_PER_MILE) if seconds > 0 and distance_m > 0 else None


def _clock(seconds: float | None) -> str:
    if seconds is None:
        return "Unavailable"
    rounded = int(round(seconds))
    return f"{rounded // 60}:{rounded % 60:02d}"


def _pace_text(value: float | None) -> str:
    return f"{_clock(value * 60)}/mi" if value else "Unavailable"


def _metric(name: str, value: str, detail: str) -> WorkoutAnalysisMetric:
    return WorkoutAnalysisMetric(name=name, value=value, detail=detail)


def _dimension(
    status: str,
    summary: str,
    confidence: ConfidenceLevel,
    metrics: list[WorkoutAnalysisMetric] | None = None,
) -> WorkoutAnalysisDimension:
    return WorkoutAnalysisDimension(
        status=status, summary=summary, confidence=confidence, metrics=metrics or []
    )


def _cadence_value(interval: MovementInterval) -> float | None:
    """Interval cadence in total steps per minute."""
    values = [
        value
        for value in (interval.start.cadence_spm, interval.end.cadence_spm)
        if value is not None
    ]
    return mean(values) if values else None


def _aggregate(
    intervals: list[MovementInterval],
    start: int,
    end: int,
    index: int,
    kind: str,
    *,
    source: str = "pace_stream_inference",
    duration_override: float | None = None,
    distance_override: float | None = None,
    average_hr_override: float | None = None,
    maximum_hr_override: float | None = None,
) -> IntervalRepetition:
    selected = intervals[start:end]
    elapsed = duration_override if duration_override is not None else sum(item.elapsed_s for item in selected)
    moving = sum(item.moving_time_s for item in selected)
    distance = distance_override if distance_override is not None else sum(item.distance_m for item in selected)
    hr_weight = [
        ((item.start.heart_rate_bpm + item.end.heart_rate_bpm) / 2, item.elapsed_s)
        for item in selected
        if item.start.heart_rate_bpm and item.end.heart_rate_bpm and item.elapsed_s > 0
    ]
    average_hr = (
        average_hr_override
        if average_hr_override is not None
        else sum(value * weight for value, weight in hr_weight) / sum(weight for _, weight in hr_weight)
        if hr_weight
        else None
    )
    hr_values = [
        float(point.heart_rate_bpm)
        for item in selected
        for point in (item.start, item.end)
        if point.heart_rate_bpm
    ]
    cadence_weight = [(_cadence_value(item), item.moving_time_s) for item in selected]
    cadence_weight = [(value, weight) for value, weight in cadence_weight if value and weight > 0]
    return IntervalRepetition(
        index=index,
        kind=kind,
        source=source,
        duration_seconds=max(0.001, elapsed),
        distance_miles=max(0.0, distance) / METERS_PER_MILE,
        pace_min_mile=_pace(moving if moving > 0 else elapsed, distance),
        average_hr_bpm=average_hr,
        end_hr_bpm=hr_values[-1] if hr_values else None,
        minimum_hr_bpm=min(hr_values, default=None),
        maximum_hr_bpm=(maximum_hr_override if maximum_hr_override is not None else max(hr_values, default=None)),
        average_cadence_spm=(
            sum(value * weight for value, weight in cadence_weight) / sum(weight for _, weight in cadence_weight)
            if cadence_weight else None
        ),
    )


def _recorded_lap_analysis(
    connection: sqlite3.Connection,
    activity_id: int,
    intervals: list[MovementInterval],
    z4_floor: float,
) -> IntervalAnalysis | None:
    rows = connection.execute(
        """
        SELECT lap_index,total_time_s,distance_m,average_hr_bpm,maximum_hr_bpm
        FROM laps WHERE activity_id=? ORDER BY lap_index
        """,
        (activity_id,),
    ).fetchall()
    usable = [row for row in rows if float(row["total_time_s"] or 0) > 0 and float(row["distance_m"] or 0) > 0]
    if len(usable) < 4:
        return None
    speeds = [float(row["distance_m"]) / float(row["total_time_s"]) for row in usable]
    work_positions = {
        position
        for position in range(1, len(usable) - 1)
        if speeds[position] >= speeds[position - 1] * 1.10
        and speeds[position] >= speeds[position + 1] * 1.10
        and float(usable[position]["total_time_s"]) >= 30
        and float(usable[position]["distance_m"]) >= 100
    }
    if len(work_positions) < 2:
        return None
    first_work, last_work = min(work_positions), max(work_positions)
    repetitions: list[IntervalRepetition] = []
    for position, row in enumerate(usable):
        lap_index = int(row["lap_index"])
        selected_indexes = [i for i, item in enumerate(intervals) if item.start.lap_index == lap_index]
        start = min(selected_indexes) if selected_indexes else 0
        end = max(selected_indexes) + 1 if selected_indexes else 0
        kind = (
            "work" if position in work_positions
            else "warmup" if position < first_work
            else "cooldown" if position > last_work
            else "recovery"
        )
        repetitions.append(
            _aggregate(
                intervals, start, end, position + 1, kind,
                source="recorded_lap",
                duration_override=float(row["total_time_s"]),
                distance_override=float(row["distance_m"]),
                average_hr_override=float(row["average_hr_bpm"]) if row["average_hr_bpm"] else None,
                maximum_hr_override=float(row["maximum_hr_bpm"]) if row["maximum_hr_bpm"] else None,
            )
        )
    return _summarize_intervals(repetitions, "recorded_laps", ConfidenceLevel.HIGH, intervals, z4_floor)


def _smoothed_speeds(intervals: list[MovementInterval]) -> list[float]:
    raw = [item.distance_m / item.moving_time_s if item.moving_time_s > 0 and item.distance_m > 0 else 0.0 for item in intervals]
    return [median(raw[max(0, i - 2):i + 3]) for i in range(len(raw))]


def _speed_clusters(values: list[float]) -> tuple[float, float]:
    positive = sorted(value for value in values if value > 0)
    if len(positive) < 10:
        return 0.0, 0.0
    low, high = positive[len(positive) // 4], positive[(len(positive) * 3) // 4]
    for _ in range(12):
        low_group = [value for value in positive if abs(value - low) <= abs(value - high)]
        high_group = [value for value in positive if abs(value - low) > abs(value - high)]
        if not low_group or not high_group:
            break
        low, high = mean(low_group), mean(high_group)
    return min(low, high), max(low, high)


def _inferred_interval_analysis(
    intervals: list[MovementInterval], z4_floor: float
) -> IntervalAnalysis:
    unavailable = lambda explanation, count=0: IntervalAnalysis(
        available=False, source="pace_stream_inference", confidence=ConfidenceLevel.LOW,
        work_repetition_count=count, recovery_repetition_count=0, explanation=explanation,
    )
    if len(intervals) < 20:
        return unavailable("Too little raw movement data to infer repetitions.")
    smooth = _smoothed_speeds(intervals)
    low, high = _speed_clusters(smooth)
    if low <= 0 or high / low < 1.12:
        return unavailable("The pace stream does not contain a sufficiently separated fast/recovery pattern.")
    threshold = (low + high) / 2
    raw_groups: list[tuple[int, int]] = []
    start: int | None = None
    for index, speed in enumerate(smooth + [0.0]):
        fast = index < len(smooth) and speed >= threshold
        if fast and start is None:
            start = index
        elif not fast and start is not None:
            raw_groups.append((start, index))
            start = None
    merged: list[tuple[int, int]] = []
    for group in raw_groups:
        gap = sum(item.elapsed_s for item in intervals[merged[-1][1]:group[0]]) if merged else None
        if merged and gap is not None and gap <= 15:
            merged[-1] = (merged[-1][0], group[1])
        else:
            merged.append(group)
    work_groups = [
        group for group in merged
        if 30 <= sum(item.elapsed_s for item in intervals[group[0]:group[1]]) <= 600
        and sum(item.distance_m for item in intervals[group[0]:group[1]]) >= 100
    ]
    if len(work_groups) < 2:
        return unavailable("Fast running was detected, but not enough repeatable work bouts were found.", len(work_groups))
    repetitions: list[IntervalRepetition] = []
    if work_groups[0][0] > 0:
        repetitions.append(_aggregate(intervals, 0, work_groups[0][0], 1, "warmup"))
    for work_index, group in enumerate(work_groups):
        repetitions.append(_aggregate(intervals, group[0], group[1], len(repetitions) + 1, "work"))
        if work_index < len(work_groups) - 1:
            next_start = work_groups[work_index + 1][0]
            if next_start > group[1]:
                repetitions.append(_aggregate(intervals, group[1], next_start, len(repetitions) + 1, "recovery"))
    if work_groups[-1][1] < len(intervals):
        repetitions.append(_aggregate(intervals, work_groups[-1][1], len(intervals), len(repetitions) + 1, "cooldown"))
    return _summarize_intervals(repetitions, "pace_stream_inference", ConfidenceLevel.MODERATE, intervals, z4_floor)


def _summarize_intervals(
    repetitions: list[IntervalRepetition],
    source: str,
    confidence: ConfidenceLevel,
    raw_intervals: list[MovementInterval],
    z4_floor: float,
) -> IntervalAnalysis:
    # Put recovery kinetics on the work rep as well as keeping the recovery row.
    linked = list(repetitions)
    for index, item in enumerate(linked):
        if item.kind != "work" or index + 1 >= len(linked) or linked[index + 1].kind != "recovery":
            continue
        recovery = linked[index + 1]
        start_hr = item.end_hr_bpm or item.maximum_hr_bpm
        minimum_hr = recovery.minimum_hr_bpm
        drop = max(0.0, start_hr - minimum_hr) if start_hr and minimum_hr else None
        linked[index] = item.model_copy(update={
            "recovery_after_seconds": recovery.duration_seconds,
            "recovery_start_hr_bpm": start_hr,
            "recovery_min_hr_bpm": minimum_hr,
            "recovery_hr_drop_bpm": drop,
            "recovery_hr_drop_percent": drop / start_hr * 100 if drop is not None and start_hr else None,
        })
    work = [item for item in linked if item.kind == "work" and item.pace_min_mile]
    recovery = [item for item in linked if item.kind == "recovery"]
    work_speeds = [METERS_PER_MILE / (item.pace_min_mile * 60) for item in work]
    recovery_speeds = [METERS_PER_MILE / (item.pace_min_mile * 60) for item in recovery if item.pace_min_mile]
    times = [item.duration_seconds for item in work]
    cv = pstdev(work_speeds) / mean(work_speeds) * 100 if len(work_speeds) >= 2 else None
    midpoint = max(1, len(work_speeds) // 2)
    first_speed = mean(work_speeds[:midpoint]) if work_speeds else None
    second_speed = mean(work_speeds[midpoint:]) if len(work_speeds) > midpoint else first_speed
    fade = (first_speed - second_speed) / first_speed * 100 if first_speed and second_speed else None
    first_last = (work_speeds[-1] / work_speeds[0] - 1) * 100 if len(work_speeds) >= 2 else None
    prior_speed = mean(work_speeds[max(0, len(work_speeds) - 4):-1]) if len(work_speeds) >= 3 else None
    overspeed = (work_speeds[-1] / prior_speed - 1) * 100 if prior_speed else None
    recovery_times = [item.duration_seconds for item in recovery]
    recovery_cv = pstdev(recovery_times) / mean(recovery_times) * 100 if len(recovery_times) >= 2 and mean(recovery_times) else None
    if cv is not None and cv <= 2.5 and (fade is None or abs(fade) <= 2):
        pattern = "even"
    elif fade is not None and fade < -2:
        pattern = "progressive"
    elif fade is not None and fade > 2:
        pattern = "faded"
    else:
        pattern = "variable"
    work_lap_indexes = {
        int(item.source.split(":")[-1]) for item in []  # reserved for corrected boundaries
    }
    del work_lap_indexes
    work_z45 = 0.0
    # Boundary-independent HR exposure approximation: count raw moving samples
    # whose lap is a recorded work lap.  For inferred reps, rep averages/end HR
    # remain the more reliable short-rep evidence.
    if source == "recorded_laps":
        recorded_work_laps = {item.index - 1 for item in work}
        work_z45 = sum(
            interval.moving_time_s for interval in raw_intervals
            if interval.start.lap_index in recorded_work_laps
            and interval.start.heart_rate_bpm is not None
            and interval.end.heart_rate_bpm is not None
            and (interval.start.heart_rate_bpm + interval.end.heart_rate_bpm) / 2 >= z4_floor
        ) / 60
    separation = (
        (mean(work_speeds) / mean(recovery_speeds) - 1) * 100
        if work_speeds and recovery_speeds and mean(recovery_speeds) > 0 else None
    )
    return IntervalAnalysis(
        available=bool(work), source=source, confidence=confidence,
        work_repetition_count=len(work), recovery_repetition_count=len(recovery),
        mean_work_pace_min_mile=METERS_PER_MILE / mean(work_speeds) / 60 if work_speeds else None,
        median_work_time_seconds=median(times) if times else None,
        mean_work_time_seconds=mean(times) if times else None,
        fastest_work_time_seconds=min(times) if times else None,
        slowest_work_time_seconds=max(times) if times else None,
        work_speed_cv_percent=cv, fade_percent=fade, first_to_last_percent=first_last,
        pacing_pattern=pattern, final_rep_overspeed_percent=overspeed,
        recovery_time_cv_percent=recovery_cv,
        work_minutes=sum(times) / 60,
        work_distance_miles=sum(item.distance_miles for item in work),
        work_z4_z5_minutes=work_z45 if source == "recorded_laps" else None,
        work_recovery_speed_separation_percent=separation,
        explanation=(
            "Work and recovery were reconstructed from recorded Garmin laps."
            if source == "recorded_laps"
            else "Work and recovery were inferred from repeated smoothed pace changes. HR summarizes each bout but does not set boundaries because it lags effort."
        ),
        repetitions=linked,
    )


def analyze_intervals(
    connection: sqlite3.Connection,
    activity_id: int,
    intervals: list[MovementInterval],
    z4_floor: float = 167,
) -> IntervalAnalysis:
    return _recorded_lap_analysis(connection, activity_id, intervals, z4_floor) or _inferred_interval_analysis(intervals, z4_floor)


def _interval_dimensions(analysis: IntervalAnalysis) -> tuple[WorkoutAnalysisDimension, ...]:
    if not analysis.available:
        missing = _dimension(
            "Not enough data", analysis.explanation, ConfidenceLevel.LOW,
            [_metric("Detected work intervals", str(analysis.work_repetition_count), "At least two repeatable efforts are needed.")],
        )
        return missing, missing, missing, missing
    cv = analysis.work_speed_cv_percent
    execution_status = "Strong" if cv is not None and cv <= 3 else "Solid" if cv is not None and cv <= 6 else "Mixed"
    execution = _dimension(
        execution_status,
        f"Found {analysis.work_repetition_count} work reps. Rep times and pacing are shown below.",
        analysis.confidence,
        [
            _metric("Average rep", _clock(analysis.mean_work_time_seconds), f"Mean work pace {_pace_text(analysis.mean_work_pace_min_mile)}"),
            _metric("Typical rep", _clock(analysis.median_work_time_seconds), "Middle rep time after sorting the reps."),
            _metric("Fastest / slowest", f"{_clock(analysis.fastest_work_time_seconds)} / {_clock(analysis.slowest_work_time_seconds)}", "Rep times are most comparable when rep distances match."),
            _metric("Planned target", "Not linked", "This completed run is not linked to a saved workout target."),
        ],
    )
    overspeed = analysis.final_rep_overspeed_percent
    if analysis.pacing_pattern == "faded":
        control_status, control_summary = "Mixed", "The second half slowed relative to the first."
    elif overspeed is not None and overspeed >= 5:
        control_status, control_summary = "Mixed", "The session progressed, but the final rep was disproportionately fast."
    elif cv is not None and cv <= 4:
        control_status, control_summary = "Controlled", "Work repetitions stayed tightly grouped."
    else:
        control_status, control_summary = "Variable", "Work-rep pacing varied enough to review the rep table."
    control = _dimension(
        control_status, control_summary, analysis.confidence,
        [
            _metric("Pacing pattern", str(analysis.pacing_pattern or "Unknown").title(), f"{(analysis.fade_percent or 0):+.1f}% change from the first half to the second; a negative number means faster later."),
            _metric("Rep variation", f"{cv:.1f}%" if cv is not None else "Unavailable", "Lower means the reps were more even."),
            _metric("First → last", f"{analysis.first_to_last_percent:+.1f}%" if analysis.first_to_last_percent is not None else "Unavailable", "Positive means the final rep was faster."),
            _metric("Final-rep overspeed", f"{overspeed:+.1f}%" if overspeed is not None else "Unavailable", "Compared with the preceding three reps."),
        ],
    )
    work_reps = [item for item in analysis.repetitions if item.kind == "work"]
    end_hrs = [item.end_hr_bpm for item in work_reps if item.end_hr_bpm]
    stimulus = _dimension(
        "Measured" if work_reps else "Not enough data",
        "For short reps, pace and end-of-rep HR are more useful than average HR because heart rate lags effort.",
        analysis.confidence,
        [
            _metric("Work volume", f"{analysis.work_minutes:.1f} min · {analysis.work_distance_miles:.2f} mi", "Recoveries are excluded."),
            _metric("Rep-end HR", " → ".join(str(round(value)) for value in end_hrs) if end_hrs else "Unavailable", "HR kinetics across repetitions."),
            _metric("Work Z4/Z5", f"{analysis.work_z4_z5_minutes:.1f} min" if analysis.work_z4_z5_minutes is not None else "Not isolated", "Low Z4 time does not invalidate short repetitions because HR lags effort."),
        ],
    )
    drops = [item.recovery_hr_drop_bpm for item in work_reps if item.recovery_hr_drop_bpm is not None]
    recovery = _dimension(
        "Consistent" if analysis.recovery_time_cv_percent is not None and analysis.recovery_time_cv_percent <= 15 else "Variable" if analysis.recovery_repetition_count else "Not enough data",
        "Shows whether recovery timing and heart-rate drop stayed consistent between reps.",
        analysis.confidence if analysis.recovery_repetition_count else ConfidenceLevel.LOW,
        [
            _metric("Recovery bouts", str(analysis.recovery_repetition_count), "Detected between work reps."),
            _metric("Recovery-time variation", f"{analysis.recovery_time_cv_percent:.1f}%" if analysis.recovery_time_cv_percent is not None else "Unavailable", "Lower indicates more consistent recovery timing."),
            _metric("Median HR drop", f"{median(drops):.0f} bpm" if drops else "Unavailable", "End-of-rep HR to the lowest observed recovery HR."),
        ],
    )
    return execution, control, stimulus, recovery


def _historical_interval_comparison(
    connection: sqlite3.Connection,
    config: dict,
    activity_id: int,
    current_start: datetime,
    current: IntervalAnalysis,
) -> HistoricalWorkoutComparison:
    candidates = connection.execute(
        """
        SELECT a.id,a.start_time_utc
        FROM activities a JOIN run_overrides o ON o.activity_id=a.activity_id
        WHERE o.workout_type='intervals'
          AND COALESCE(o.health_tag,'normal')='normal'
          AND a.id<>? AND a.start_time_utc<?
        ORDER BY a.start_time_utc DESC LIMIT 12
        """,
        (activity_id, current_start.isoformat()),
    ).fetchall()
    current_work = [item for item in current.repetitions if item.kind == "work"]
    current_distance = median([item.distance_miles for item in current_work]) if current_work else None
    current_recovery = median([item.duration_seconds for item in current.repetitions if item.kind == "recovery"]) if current.recovery_repetition_count else None
    best: tuple[float, sqlite3.Row, IntervalAnalysis] | None = None
    for row in candidates:
        points = _load_points(connection, int(row["id"]))
        from .movement import classify_movement
        movement = classify_movement(points, config["moving_time"])
        candidate = analyze_intervals(connection, int(row["id"]), movement.intervals, float(config["zones"]["z4"][0]))
        if not candidate.available:
            continue
        work = [item for item in candidate.repetitions if item.kind == "work"]
        distance = median([item.distance_miles for item in work]) if work else None
        recovery_times = [item.duration_seconds for item in candidate.repetitions if item.kind == "recovery"]
        recovery = median(recovery_times) if recovery_times else None
        structure = abs(candidate.work_repetition_count - current.work_repetition_count) * 2
        if current_distance and distance:
            structure += abs(log(distance / current_distance)) * 8
        if current_recovery and recovery:
            structure += abs(log(recovery / current_recovery)) * 2
        if best is None or structure < best[0]:
            best = (structure, row, candidate)
    if best is None or best[0] > 5:
        return HistoricalWorkoutComparison(
            available=False,
            summary="No sufficiently similar earlier interval workout was found. Comparisons require similar rep count, distance, and recovery structure.",
        )
    _, row, prior = best
    pace_delta = ((current.mean_work_pace_min_mile or 0) - (prior.mean_work_pace_min_mile or 0)) * 60
    cv_delta = (current.work_speed_cv_percent or 0) - (prior.work_speed_cv_percent or 0)
    return HistoricalWorkoutComparison(
        available=True, activity_id=int(row["id"]), date=datetime.fromisoformat(row["start_time_utc"]),
        summary=f"Closest structural match: {prior.work_repetition_count} reps with similar rep distance and recovery.",
        metrics=[
            _metric("Average work pace", f"{abs(pace_delta):.0f} sec/mi {'slower' if pace_delta > 0 else 'faster'}", "Current versus the closest comparable session."),
            _metric("Rep variability", f"{abs(cv_delta):.1f} points {'higher' if cv_delta > 0 else 'lower'}", "Change in speed coefficient of variation."),
        ],
    )


def _generic_analysis(
    workout: WorkoutType,
    difficulty: SessionDifficulty,
    drift: DriftAssessment,
    splits: list[Split],
) -> WorkoutAnalysis:
    known = difficulty.zone_breakdown.easy_minutes + difficulty.zone_breakdown.moderate_minutes + difficulty.zone_breakdown.hard_minutes
    easy_fraction = difficulty.zone_breakdown.easy_minutes / known if known else None
    quality_minutes = difficulty.zone_breakdown.moderate_minutes + difficulty.zone_breakdown.hard_minutes
    quality_fraction = quality_minutes / known if known else None
    paces = [item.pace_min_mile for item in splits if item.pace_min_mile and not item.is_partial]
    speed_cv = pstdev([1 / value for value in paces]) / mean([1 / value for value in paces]) * 100 if len(paces) >= 2 else None
    if workout in {WorkoutType.EASY, WorkoutType.RECOVERY, WorkoutType.LONG}:
        target = 0.90 if workout == WorkoutType.RECOVERY else 0.80
        status = "Kept easy" if easy_fraction is not None and easy_fraction >= target else "Mixed intensity" if easy_fraction is not None else "Limited data"
        execution = _dimension(
            status,
            "Easy and long runs are judged mainly by heart-rate effort, not speed.",
            ConfidenceLevel.MODERATE if known else ConfidenceLevel.LOW,
            [_metric("Easy HR share", f"{easy_fraction * 100:.0f}%" if easy_fraction is not None else "Unavailable", "Z1/Z2 share of known HR time.")],
        )
    elif workout in {WorkoutType.TEMPO_THRESHOLD, WorkoutType.RACE}:
        execution = _dimension(
            "Sustained" if quality_fraction is not None and quality_fraction >= .40 else "Limited hard running" if quality_fraction is not None else "Limited data",
            "Tempo and race efforts are judged by sustained moderate/hard running and steady pacing.",
            ConfidenceLevel.MODERATE if known else ConfidenceLevel.LOW,
            [_metric("Z3+ exposure", f"{quality_minutes:.1f} min · {quality_fraction * 100:.0f}%" if quality_fraction is not None else "Unavailable", "Moderate and hard HR time combined.")],
        )
    else:
        execution = _dimension(
            "Not applicable",
            "This activity counts toward your history and load but has no running-workout target.",
            ConfidenceLevel.UNAVAILABLE,
        )
    control = _dimension(
        "Controlled" if speed_cv is not None and speed_cv <= 5 else "Variable" if speed_cv is not None else "Not enough data",
        "Compares full-mile pacing and heart-rate drift. Pace changes may be intentional in a workout.",
        ConfidenceLevel.MODERATE if speed_cv is not None else ConfidenceLevel.LOW,
        [_metric("Mile-to-mile variation", f"{speed_cv:.1f}%" if speed_cv is not None else "Unavailable", "Coefficient of variation in speed."),
         _metric("Heart-rate drift", f"{drift.decoupling_percent:+.1f}%" if drift.valid and drift.decoupling_percent is not None else "Not measured", drift.reason)],
    )
    stimulus = _dimension(
        "Aerobic" if workout in {WorkoutType.EASY, WorkoutType.RECOVERY, WorkoutType.LONG} else "Quality / mixed" if workout in {WorkoutType.TEMPO_THRESHOLD, WorkoutType.RACE} else "Context only",
        "Distance, duration, and heart-rate load show how much work you completed.",
        ConfidenceLevel.MODERATE,
        [_metric("Volume", f"{difficulty.distance_miles:.2f} mi · {difficulty.moving_minutes:.0f} min", "Moving distance and duration."),
         _metric("Zone load", f"{difficulty.zone_load:.0f}" if difficulty.zone_load is not None else "Unavailable", "Time-in-zone load points.")],
    )
    recovery = _dimension(
        "Use recent context", "One run cannot show whether you are recovered; the weekly plan uses your recent load and health check-in.", ConfidenceLevel.LOW,
        [_metric("Stopped time", f"{difficulty.stopped_minutes:.1f} min", "Useful context, not automatically a failure.")],
    )
    return WorkoutAnalysis(
        workout_type=workout,
        definition="Four independent workout judgments; no composite score is calculated.",
        execution=execution, control=control, stimulus=stimulus, recovery=recovery,
        progression_recommendation="Use the weekly plan and how you feel before making the next workout harder.",
    )


def analyze_workout(
    connection: sqlite3.Connection,
    config: dict,
    activity_id: int,
    start: datetime,
    workout: WorkoutType,
    difficulty: SessionDifficulty,
    drift: DriftAssessment,
    splits: list[Split],
    intervals: list[MovementInterval],
) -> WorkoutAnalysis:
    if workout not in {WorkoutType.INTERVALS, WorkoutType.RUN_WALK}:
        return _generic_analysis(workout, difficulty, drift, splits)
    interval_analysis = analyze_intervals(
        connection, activity_id, intervals, float(config["zones"]["z4"][0])
    )
    execution, control, stimulus, recovery = _interval_dimensions(interval_analysis)
    comparison = _historical_interval_comparison(
        connection, config, activity_id, start, interval_analysis
    ) if workout == WorkoutType.INTERVALS and interval_analysis.available else None
    if not interval_analysis.available:
        progression = "Review or correct the inferred workout boundaries before using this session to progress quality training."
    elif interval_analysis.pacing_pattern == "faded":
        progression = "Repeat this structure with a more conservative opening pace before adding reps or distance."
    elif (interval_analysis.final_rep_overspeed_percent or 0) >= 5:
        progression = "Workout accomplished. Repeat the structure and keep the final rep near the preceding reps before progressing."
    elif (interval_analysis.work_speed_cv_percent or 99) <= 4 and interval_analysis.recovery_repetition_count:
        progression = "Execution was controlled. Progress only if health and recent load are normal; prefer adding controlled work over making the final rep faster."
    else:
        progression = "Repeat once with steadier work and recovery before progressing."
    return WorkoutAnalysis(
        workout_type=workout,
        definition="Execution, control, stimulus, and recovery remain separate; short-rep pace is primary, HR kinetics secondary, and zones tertiary.",
        execution=execution, control=control, stimulus=stimulus, recovery=recovery,
        interval_analysis=interval_analysis,
        historical_comparison=comparison,
        progression_recommendation=progression,
    )
