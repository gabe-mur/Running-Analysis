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
from .quality_phases import detect_continuous_quality_phase
from .segmentation import METERS_PER_MILE
from .web.schemas import (
    ConfidenceLevel,
    DriftAssessment,
    HistoricalWorkoutComparison,
    IntervalAnalysis,
    IntervalRepetition,
    PrescriptionMatchAnalysis,
    RecommendationResponse,
    SessionDifficulty,
    Split,
    WorkoutAnalysis,
    WorkoutAnalysisDimension,
    WorkoutAnalysisMetric,
    WorkoutType,
)
from .terrain_intensity import terrain_moderate_context


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
        FROM activities a
        LEFT JOIN run_overrides o ON o.activity_id=a.activity_id
        LEFT JOIN activity_plan_matches ap ON ap.activity_id=a.id
        LEFT JOIN planned_workout_history ph ON ph.id=ap.planned_workout_id
        WHERE COALESCE(o.workout_type,ph.workout_type)='intervals'
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
        # No progression note for an ordinary run. "Use the weekly plan and see
        # how you feel" is true of every run ever done, so printing it on all
        # of them buries the three sessions where there is something specific
        # to say. Quality sessions set this below; everything else leaves it
        # empty and the interface shows nothing.
        progression_recommendation=None,
    )


def _prescription_analysis(
    connection: sqlite3.Connection,
    config: dict,
    activity_id: int,
    difficulty: SessionDifficulty,
    prescription: RecommendationResponse,
    *,
    timing_delta_hours: float,
    distance_delta_miles: float,
    duration_delta_minutes: float = 0.0,
    match_confidence: str,
    interval_analysis: IntervalAnalysis | None = None,
) -> PrescriptionMatchAnalysis:
    quality_steps = [
        step
        for step in prescription.structure
        if (step.duration_minutes is not None or step.repetitions is not None)
        and any(
            marker in zone.casefold()
            for zone in step.target_zones
            for marker in ("z3", "z4", "z5", "strong", "threshold")
        )
    ]
    target_work = 0.0
    for step in quality_steps:
        if step.repetitions and step.work_duration_minutes:
            target_work += step.repetitions * step.work_duration_minutes
        elif step.repetitions and step.work_duration_range_minutes:
            low, high = step.work_duration_range_minutes
            target_work += step.repetitions * ((low + high) / 2)
        else:
            target_work += float(step.duration_minutes or 0)
    target_work = target_work or None
    detected_work: float | None = None
    source = "heart_rate_zone_exposure"
    if interval_analysis is not None and interval_analysis.available:
        detected_work = interval_analysis.work_minutes
        source = interval_analysis.source
    elif (
        prescription.workout_type == WorkoutType.TEMPO_THRESHOLD
        and target_work is not None
    ):
        laps = connection.execute(
            """
            SELECT lap_index,total_time_s,average_hr_bpm
            FROM laps WHERE activity_id=? ORDER BY lap_index
            """,
            (activity_id,),
        ).fetchall()
        plausible = [
            row
            for row in laps
            if float(row["total_time_s"] or 0) >= target_work * 60 * 0.70
            and float(row["total_time_s"] or 0) <= target_work * 60 * 1.30
            and (
                row["average_hr_bpm"] is None
                or float(row["average_hr_bpm"])
                >= float(config["zones"]["z3"][0])
            )
        ]
        if plausible:
            selected = min(
                plausible,
                key=lambda row: abs(
                    float(row["total_time_s"]) / 60 - target_work
                ),
            )
            detected_work = float(selected["total_time_s"]) / 60
            source = f"recorded_lap_{int(selected['lap_index']) + 1}"
    if detected_work is None and target_work is not None:
        detected_work = (
            difficulty.zone_breakdown.moderate_minutes
            + difficulty.zone_breakdown.hard_minutes
        )
    requires_work_detection = prescription.workout_type in {
        WorkoutType.INTERVALS,
        WorkoutType.TEMPO_THRESHOLD,
        WorkoutType.RACE,
    }
    prescribed_aerobic = (
        prescription.workout_type
        in {WorkoutType.EASY, WorkoutType.RECOVERY, WorkoutType.LONG}
        and not quality_steps
    )
    known_hr_minutes = (
        difficulty.zone_breakdown.easy_minutes
        + difficulty.zone_breakdown.moderate_minutes
        + difficulty.zone_breakdown.hard_minutes
    )
    aerobic_intensity_adherence = (
        difficulty.zone_breakdown.easy_minutes / known_hr_minutes
        if prescribed_aerobic and known_hr_minutes > 0
        else None
    )
    above_prescribed_intensity_minutes = (
        difficulty.zone_breakdown.moderate_minutes
        + difficulty.zone_breakdown.hard_minutes
        if prescribed_aerobic and known_hr_minutes > 0
        else None
    )
    terrain = terrain_moderate_context(
        connection,
        activity_id,
        z3_low_bpm=float(config["zones"]["z3"][0]),
        z3_high_bpm=float(config["zones"]["z3"][1]),
        raw_moderate_minutes=difficulty.zone_breakdown.moderate_minutes,
    )
    effective_above_intensity_minutes = (
        terrain.effective_moderate_minutes
        + difficulty.zone_breakdown.hard_minutes
        if prescribed_aerobic and known_hr_minutes > 0
        else None
    )
    terrain_adjusted_adherence = (
        1.0 - effective_above_intensity_minutes / known_hr_minutes
        if effective_above_intensity_minutes is not None and known_hr_minutes > 0
        else None
    )
    # Use the same aerobic-control standard as the ordinary workout analysis:
    # recovery runs are deliberately stricter; easy and long runs may include
    # limited normal HR spillover without being declared a different workout.
    aerobic_target_share = (
        0.90 if prescription.workout_type == WorkoutType.RECOVERY else 0.80
    )
    aerobic_intensity_close = (
        terrain_adjusted_adherence is not None
        and terrain_adjusted_adherence >= aerobic_target_share
    )
    work_close = bool(
        (
            prescribed_aerobic
            and aerobic_intensity_close
        )
        or (
            not prescribed_aerobic
            and not requires_work_detection
            and target_work is None
        )
        or (
            target_work is not None
            and
            detected_work is not None
            and abs(detected_work - target_work)
            <= max(2.0, target_work * 0.15)
        )
    )
    distance_close = distance_delta_miles <= 0.25
    duration_close = duration_delta_minutes <= 1e-9
    if prescribed_aerobic and not aerobic_intensity_close and aerobic_intensity_adherence is not None:
        adherence_percent = aerobic_intensity_adherence * 100
        adjusted_percent = (terrain_adjusted_adherence or 0.0) * 100
        status = (
            "Distance completed; intensity diverged"
            if distance_close and duration_close
            else "Prescription attempted"
        )
        terrain_text = (
            f"Climbing accounts for {terrain.grade_attributed_minutes:.1f} minutes of the moderate response; "
            f"terrain-adjusted adherence is {adjusted_percent:.1f}%. "
            if terrain.grade_attributed_minutes > 0
            else ""
        )
        summary = (
            "The upload matches the planned aerobic workout, but "
            f"{above_prescribed_intensity_minutes:.1f} minutes were above Z2 "
            f"({adherence_percent:.0f}% raw aerobic HR time). "
            f"{terrain_text}"
            "It stays matched to the aerobic prescription while the observed "
            "heart-rate load counts toward recovery."
        )
        source = (
            "heart_rate_with_grade_context"
            if terrain.grade_attributed_minutes > 0
            else "heart_rate_zone_adherence"
        )
    elif prescribed_aerobic and aerobic_intensity_adherence is None:
        status = "Structure completed"
        summary = (
            "Timing and distance match the saved aerobic prescription, but "
            "heart-rate coverage is insufficient to verify intensity execution."
        )
        source = "heart_rate_unavailable"
    elif (
        prescribed_aerobic
        and aerobic_intensity_adherence is not None
        and aerobic_intensity_adherence < aerobic_target_share
        and aerobic_intensity_close
    ):
        adjusted_percent = (terrain_adjusted_adherence or 0.0) * 100
        status = (
            "Completed as prescribed"
            if distance_close and duration_close
            else "Structure completed"
        )
        summary = (
            f"Raw HR time was {aerobic_intensity_adherence * 100:.0f}% aerobic. "
            f"Climbing accounts for {terrain.grade_attributed_minutes:.1f} minutes "
            f"of the moderate response, bringing terrain-adjusted adherence to "
            f"{adjusted_percent:.1f}%. The full recorded heart-rate load still "
            "counts toward recovery."
        )
        source = "heart_rate_with_grade_context"
    elif work_close and distance_close and duration_close:
        status = "Completed as prescribed"
        summary = (
            "Recorded timing, distance or duration, and work dose match the saved "
            "prescription closely."
        )
    elif work_close:
        status = "Structure completed"
        summary = (
            "The prescribed work dose was detected, with total distance or duration "
            "outside the planned range."
        )
    else:
        status = "Prescription attempted"
        summary = (
            "The upload matches the planned workout slot, but the detected "
            "quality dose differs materially from the prescription."
        )
    confidence = (
        ConfidenceLevel.HIGH
        if match_confidence == "high" and source.startswith("recorded_lap")
        else ConfidenceLevel.MODERATE
    )
    return PrescriptionMatchAnalysis(
        confidence=confidence,
        title=prescription.title,
        planned_for=prescription.planned_for,
        workout_type=prescription.workout_type,
        quality_session_type=prescription.quality_session_type,
        target_distance_range_miles=prescription.distance_range_miles,
        target_duration_range_minutes=prescription.duration_range_minutes,
        timing_delta_hours=timing_delta_hours,
        distance_delta_miles=distance_delta_miles,
        duration_delta_minutes=duration_delta_minutes,
        execution_status=status,
        summary=summary,
        target_work_minutes=target_work,
        detected_work_minutes=detected_work,
        aerobic_intensity_adherence_percent=(
            aerobic_intensity_adherence * 100
            if aerobic_intensity_adherence is not None
            else None
        ),
        terrain_adjusted_aerobic_adherence_percent=(
            terrain_adjusted_adherence * 100
            if terrain_adjusted_adherence is not None
            else None
        ),
        above_prescribed_intensity_minutes=above_prescribed_intensity_minutes,
        effective_above_prescribed_intensity_minutes=effective_above_intensity_minutes,
        grade_attributed_moderate_minutes=terrain.grade_attributed_minutes,
        detection_source=source,
    )


def _recorded_continuous_quality_lap(
    connection: sqlite3.Connection,
    config: dict,
    activity_id: int,
) -> tuple[float, float | None, int] | None:
    selected = detect_continuous_quality_phase(connection, config, activity_id)
    if selected is None:
        return None
    return (
        selected.duration_seconds / 60,
        selected.average_hr_bpm,
        selected.lap_index + 1,
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
    prescription: RecommendationResponse | None = None,
    prescription_timing_delta_hours: float = 0.0,
    prescription_distance_delta_miles: float = 0.0,
    prescription_duration_delta_minutes: float = 0.0,
    prescription_match_confidence: str = "moderate",
) -> WorkoutAnalysis:
    interval_analysis: IntervalAnalysis | None = None
    if workout not in {WorkoutType.INTERVALS, WorkoutType.RUN_WALK}:
        analysis = _generic_analysis(workout, difficulty, drift, splits)
        if workout == WorkoutType.TEMPO_THRESHOLD:
            sustained = _recorded_continuous_quality_lap(
                connection, config, activity_id
            )
            if sustained is not None:
                minutes, average_hr, lap_number = sustained
                phase = detect_continuous_quality_phase(
                    connection, config, activity_id
                )
                execution = analysis.execution.model_copy(
                    update={
                        "status": "Structured threshold work detected",
                        "summary": (
                            "A sustained manual lap separates the threshold "
                            "work from the easy warm-up and cool-down."
                        ),
                        "confidence": ConfidenceLevel.HIGH,
                        "metrics": [
                            *analysis.execution.metrics,
                            _metric(
                                "Continuous work lap",
                                f"{minutes:.1f} min",
                                (
                                    f"Recorded lap {lap_number}; "
                                    f"{_pace_text(phase.pace_min_mile if phase else None)}; "
                                    f"average HR {average_hr:.0f} bpm."
                                ),
                            ),
                        ],
                    }
                )
                analysis = analysis.model_copy(update={"execution": execution})
    else:
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
        analysis = WorkoutAnalysis(
            workout_type=workout,
            definition="Execution, control, stimulus, and recovery remain separate; short-rep pace is primary, HR kinetics secondary, and zones tertiary.",
            execution=execution, control=control, stimulus=stimulus, recovery=recovery,
            interval_analysis=interval_analysis,
            historical_comparison=comparison,
            progression_recommendation=progression,
        )
    if prescription is None or prescription.planned_for is None:
        return analysis
    prescription_analysis = _prescription_analysis(
        connection,
        config,
        activity_id,
        difficulty,
        prescription,
        timing_delta_hours=prescription_timing_delta_hours,
        distance_delta_miles=prescription_distance_delta_miles,
        duration_delta_minutes=prescription_duration_delta_minutes,
        match_confidence=prescription_match_confidence,
        interval_analysis=interval_analysis,
    )
    execution = analysis.execution.model_copy(
        update={
            "status": prescription_analysis.execution_status,
            "summary": prescription_analysis.summary,
            "confidence": prescription_analysis.confidence,
            "metrics": [
                *analysis.execution.metrics,
                _metric(
                    "Prescribed work",
                    (
                        f"{prescription_analysis.target_work_minutes:.0f} min"
                        if prescription_analysis.target_work_minutes is not None
                        else "Structure-based"
                    ),
                    prescription.title,
                ),
                _metric(
                    "Detected work",
                    (
                        f"{prescription_analysis.detected_work_minutes:.1f} min"
                        if prescription_analysis.detected_work_minutes is not None
                        else "Unavailable"
                    ),
                    prescription_analysis.detection_source.replace("_", " "),
                ),
            ],
        }
    )
    return analysis.model_copy(
        update={
            "execution": execution,
            "prescription_match": prescription_analysis,
        }
    )
