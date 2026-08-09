"""Compose the three primary application jobs for the home screen."""

from __future__ import annotations

from datetime import timedelta
import sqlite3
from pathlib import Path

from .fitness_state import build_fitness_state
from .onboarding import setup_state
from .progress import build_progress
from .recommendation import recommend_next_run
from .recommendation_service import ensure_current_weekly_schedule, load_current_status
from .training_status import build_training_status
from .run_feedback import get_run_feedback, list_runs
from .web.schemas import DashboardResponse
from .web.schemas import (
    ConfidenceLevel,
    FitnessHorizon,
    FitnessInterpretation,
    FitnessSignal,
    FitnessTrend,
    SetupNudge,
    WorkoutType,
)

DASHBOARD_WINDOW_DAYS = 90


def _horizon(label: str, progress) -> FitnessHorizon:
    prior_runs = progress.period_comparison.previous.run_count
    confidence = progress.fitness_confidence
    if prior_runs < 3:
        confidence = ConfidenceLevel.LOW if prior_runs else ConfidenceLevel.UNAVAILABLE
    return FitnessHorizon(
        label=label,
        window_days=progress.window_days,
        trend=progress.fitness_trend,
        confidence=confidence,
        pace_change_seconds_per_mile=progress.pace_change_seconds_per_mile,
        current_pace=progress.current_pace,
    )


def _signal_status(trend: FitnessTrend) -> str:
    return {
        FitnessTrend.IMPROVING: "Improved",
        FitnessTrend.DECLINING: "Declined",
        FitnessTrend.STABLE: "About the same",
        FitnessTrend.UNCERTAIN: "No clear change",
        FitnessTrend.INSUFFICIENT_DATA: "Not enough data",
    }[trend]


def _quality_fitness_signal(
    connection: sqlite3.Connection, config: dict, quality_progress
) -> FitnessSignal:
    window_days = quality_progress.window_days
    start = quality_progress.as_of - timedelta(days=window_days)
    candidates = [
        run
        for run in list_runs(connection, limit=500)
        if run.start_time
        and run.start_time > start
        and run.health_tag.value == "normal"
        and run.workout_type
        in {WorkoutType.INTERVALS, WorkoutType.TEMPO_THRESHOLD, WorkoutType.RACE}
    ]
    usable = []
    for run in candidates:
        feedback = get_run_feedback(connection, config, run.activity_id)
        analysis = feedback.workout_analysis if feedback else None
        if analysis is None:
            continue
        if run.workout_type == WorkoutType.INTERVALS:
            intervals = analysis.interval_analysis
            if intervals and intervals.available and intervals.work_repetition_count >= 3:
                usable.append(feedback)
        elif run.workout_type == WorkoutType.TEMPO_THRESHOLD:
            if analysis.execution.status == "Sustained":
                usable.append(feedback)
        else:
            usable.append(feedback)

    count = len(usable)
    confidence = (
        ConfidenceLevel.UNAVAILABLE
        if count == 0
        else ConfidenceLevel.LOW
        if count < 3
        else ConfidenceLevel.MODERATE
        if count < 5
        else ConfidenceLevel.HIGH
    )
    trend = FitnessTrend.INSUFFICIENT_DATA
    status = "Not enough data"
    comparison_detail = ""
    latest_interval = next(
        (
            feedback
            for feedback in usable
            if feedback.run.workout_type == WorkoutType.INTERVALS
            and feedback.workout_analysis
            and feedback.workout_analysis.historical_comparison
            and feedback.workout_analysis.historical_comparison.available
        ),
        None,
    )
    if latest_interval:
        comparison = latest_interval.workout_analysis.historical_comparison
        pace_metric = next(
            (item for item in comparison.metrics if item.name == "Average work pace"),
            None,
        )
        if pace_metric:
            try:
                seconds = float(pace_metric.value.split()[0])
            except (ValueError, IndexError):
                seconds = 0.0
            if seconds < 5:
                trend, status = FitnessTrend.STABLE, "Comparable execution"
            elif "faster" in pace_metric.value:
                trend, status = FitnessTrend.IMPROVING, "Improved in comparable intervals"
            elif "slower" in pace_metric.value:
                trend, status = FitnessTrend.DECLINING, "Down in comparable intervals"
            comparison_detail = f" {pace_metric.value} versus the closest structurally similar interval session."
    elif count >= 2:
        trend, status = FitnessTrend.UNCERTAIN, "No clear trend yet"

    rejected = len(candidates) - count
    detail = f"{count} comparable hard workout{'s' if count != 1 else ''} in the last {window_days} days."
    if comparison_detail:
        detail += comparison_detail
    elif rejected:
        detail += f" {rejected} other labeled workout{'s' if rejected != 1 else ''} could not be compared fairly."
    return FitnessSignal(
        label="High-intensity fitness",
        trend=trend,
        status=status,
        confidence=confidence,
        detail=detail,
    )


#: Weekly-mileage change below this is treated as noise in demonstrated capacity.
CAPACITY_CHANGE_THRESHOLD_MILES = 2.0


def _capacity_signal(load, window_days: int) -> tuple[FitnessSignal, FitnessTrend]:
    """Training capacity from retained demonstrated capacity, not period mileage.

    Period mileage answers "how much did you run lately"; that is training
    volume and gets its own signal. Capacity is what the athlete has shown they
    can sustain, which survives a short gap by design and is the number the
    coaching rules already plan against.
    """

    current = load.capacity_reference_miles
    previous = load.previous_capacity_reference_miles
    if not current:
        return (
            FitnessSignal(
                label="Training capacity",
                trend=FitnessTrend.INSUFFICIENT_DATA,
                status="Not enough data",
                confidence=ConfidenceLevel.UNAVAILABLE,
                detail="There is not enough running history to establish demonstrated capacity.",
            ),
            FitnessTrend.INSUFFICIENT_DATA,
        )
    # A zero baseline means there was no history a window ago, not that
    # capacity grew from nothing. build_progress always supplies a number, so
    # without this an athlete three weeks into running is told they have
    # "Improved" against a baseline that never existed.
    if not previous:
        direction, status = FitnessTrend.UNCERTAIN, "No comparison yet"
    else:
        change = current - previous
        if change >= CAPACITY_CHANGE_THRESHOLD_MILES:
            direction, status = FitnessTrend.IMPROVING, "Improved"
        elif change <= -CAPACITY_CHANGE_THRESHOLD_MILES:
            direction, status = FitnessTrend.DECLINING, "Down"
        else:
            direction, status = FitnessTrend.STABLE, "Holding"
    detail = f"Retained demonstrated capacity: {current:.1f} mi/week"
    if previous:
        detail += (
            f", versus {previous:.1f} mi/week {window_days} days ago."
            " A short gap does not immediately erase what you have shown you can sustain."
        )
    else:
        detail += f". There is no comparable capacity from {window_days} days ago yet."
    return (
        FitnessSignal(
            label="Training capacity",
            trend=direction,
            status=status,
            confidence=ConfidenceLevel.MODERATE if load.confidence != ConfidenceLevel.LOW else ConfidenceLevel.LOW,
            detail=detail,
        ),
        direction,
    )


def _volume_signal(comparison, window_days: int) -> FitnessSignal:
    """Recent period mileage and run count. Volume, deliberately not capacity."""
    current = comparison.current
    previous = comparison.previous
    change = comparison.distance_change_percent
    if change is None or previous.distance_miles <= 0:
        direction, status = FitnessTrend.INSUFFICIENT_DATA, "Not enough data"
    elif change >= 10:
        direction, status = FitnessTrend.IMPROVING, "Up"
    elif change <= -10:
        direction, status = FitnessTrend.DECLINING, "Down"
    else:
        direction, status = FitnessTrend.STABLE, "About the same"
    return FitnessSignal(
        label="Training volume",
        trend=direction,
        status=status,
        confidence=ConfidenceLevel.MODERATE,
        detail=(
            f"Last {window_days} days: {current.distance_miles:.1f} miles in {current.run_count} runs. "
            f"Previous {window_days}: {previous.distance_miles:.1f} miles in {previous.run_count} runs."
        ),
    )


#: Normal runs after a health-tagged one that show current form has moved on.
NORMAL_RUNS_TO_CLEAR_RECOVERY = 3


def _recent_form(state) -> tuple[FitnessTrend, str, str]:
    """Current form, using how the athlete has actually responded since.

    A health-tagged run inside the illness window is context, not a verdict
    that lasts three weeks. Once several normal runs have followed it, their
    measured response is better evidence of current form than the calendar,
    so the recovery label steps aside and the observed response speaks.
    """

    normal_since = state.normal_runs_since_health_event
    still_recovering = (
        state.recent_illness_or_recovery and normal_since < NORMAL_RUNS_TO_CLEAR_RECOVERY
    )
    if still_recovering:
        detail = "Recent illness or recovery runs may be temporarily affecting performance."
        if normal_since:
            detail += (
                f" {normal_since} normal run{'s' if normal_since != 1 else ''} since then;"
                f" {NORMAL_RUNS_TO_CLEAR_RECOVERY - normal_since} more will let recent responses"
                " speak for current form."
            )
        return FitnessTrend.DECLINING, "Recovering", detail
    # Past illness stays visible as context even once form has been re-measured.
    suffix = (
        f" A health-tagged run remains in the last 21 days, but {normal_since} normal runs have followed it."
        if state.recent_illness_or_recovery
        else ""
    )
    if state.recent_performance_anomaly == "unusually_costly":
        return (
            FitnessTrend.DECLINING,
            "Suppressed",
            "The latest comparable run took more effort than usual." + suffix,
        )
    if state.recent_performance_anomaly == "within_recent_range":
        return (
            FitnessTrend.STABLE,
            "Within recent range",
            "The latest comparable run was within your recent range." + suffix,
        )
    if state.recent_performance_anomaly == "unusually_strong":
        return (
            FitnessTrend.IMPROVING,
            "Responding well",
            "The latest comparable run took less effort than usual." + suffix,
        )
    return (
        FitnessTrend.UNCERTAIN,
        "Not enough data",
        "There are too few comparable recent runs to judge current form." + suffix,
    )


def _interpret_fitness(short, long, capacity, state, quality_signal) -> FitnessInterpretation:
    current = capacity.period_comparison.current
    previous = capacity.period_comparison.previous
    longest_change = current.longest_run_miles - previous.longest_run_miles
    capacity_signal, capacity_direction = _capacity_signal(
        capacity.current_load, capacity.window_days
    )
    volume_signal = _volume_signal(capacity.period_comparison, capacity.window_days)
    capacity_up = capacity_direction == FitnessTrend.IMPROVING
    # Not the capacity signal's own detail: repeating it verbatim under the
    # grid that already shows it is noise.
    capacity_summary = (
        f"Last {capacity.window_days} days: {current.distance_miles:.1f} miles in "
        f"{current.run_count} runs, against a retained capacity of "
        f"{capacity.current_load.capacity_reference_miles:.1f} mi/week."
        if capacity.current_load.capacity_reference_miles
        else f"Last {capacity.window_days} days: {current.distance_miles:.1f} miles in {current.run_count} runs."
    )
    illness_context = None
    if state.recent_illness_or_recovery:
        illness_context = "Recent illness/recovery runs still count toward training, but have less influence on the fitness trend."
    short_horizon = _horizon(f"Current {short.window_days}-day aerobic efficiency", short)
    # Kept as a compatibility field for existing API clients. The dashboard no longer
    # mixes a second lookback into its fitness interpretation.
    long_horizon = _horizon(f"Current {long.window_days}-day aerobic efficiency", long)
    if short_horizon.trend == FitnessTrend.DECLINING and capacity_up:
        headline = "Short-term efficiency is down; running capacity is up."
        summary = (
            "These are not contradictory. Pace at the same heart rate has recently been "
            "slower, while the weekly load you have demonstrated you can sustain has grown. "
            "Illness and accumulated fatigue can affect the first signal without erasing the second."
        )
    elif short_horizon.trend == FitnessTrend.IMPROVING and capacity_up:
        headline = "Aerobic efficiency and running capacity are both improving."
        summary = "You are running more efficiently, and the load you can sustain has grown."
    elif short_horizon.trend in {FitnessTrend.STABLE, FitnessTrend.UNCERTAIN} and capacity_up:
        headline = "Training capacity is up; aerobic efficiency has no clear change."
        summary = (
            "The weekly load you can sustain has grown, with no clear change in pace "
            "at the same heart rate."
        )
    elif short_horizon.trend == FitnessTrend.STABLE:
        headline = "Your fitness looks steady."
        summary = "Pace at the same heart rate and your demonstrated capacity are both about the same."
    elif short_horizon.trend in {FitnessTrend.UNCERTAIN, FitnessTrend.INSUFFICIENT_DATA}:
        headline = "There is no clear fitness change yet."
        summary = "Recent runs vary too much, or there are too few comparable runs, to call the trend up or down."
    else:
        headline = "Your recent fitness signals are mixed."
        summary = "Aerobic efficiency, recovery, and training volume are pointing in different directions."
    if longest_change >= 1:
        durability_direction = FitnessTrend.IMPROVING
        durability_status = "Improved"
    elif longest_change <= -1:
        durability_direction = FitnessTrend.DECLINING
        durability_status = "Down"
    else:
        durability_direction = FitnessTrend.STABLE
        durability_status = "Stable"
    recent_direction, recent_status, recent_detail = _recent_form(state)
    signals = [
        FitnessSignal(
            label="Aerobic efficiency",
            trend=short_horizon.trend,
            status=_signal_status(short_horizon.trend),
            confidence=short_horizon.confidence,
            detail=f"Pace at the same heart rate over the last {short_horizon.window_days} days.",
        ),
        FitnessSignal(
            label="Durability",
            trend=durability_direction,
            status=durability_status,
            confidence=ConfidenceLevel.MODERATE,
            detail=f"Longest run: {current.longest_run_miles:.1f} miles now vs {previous.longest_run_miles:.1f} previously.",
        ),
        capacity_signal,
        volume_signal,
        FitnessSignal(
            label="Recent form",
            trend=recent_direction,
            status=recent_status,
            confidence=ConfidenceLevel.MODERATE if recent_status == "Recovering" else state.trend_confidence,
            detail=recent_detail,
        ),
        quality_signal,
    ]
    return FitnessInterpretation(
        headline=headline,
        summary=summary,
        short_term=short_horizon,
        long_term=long_horizon,
        capacity_direction=capacity_direction,
        capacity_summary=capacity_summary,
        signals=signals,
        illness_context=illness_context,
        caveats=[
            "The dashboard verdict uses the robust all-window model; the strict two-minute benchmark is a validation signal, not the primary estimate.",
            "More mileage and longer runs demonstrate greater training capacity, not automatically faster pace at a fixed heart rate.",
            "A respiratory illness can temporarily alter performance; this app records that context but does not diagnose recovery.",
        ],
    )


def build_dashboard(
    connection: sqlite3.Connection,
    config: dict,
    window_days: int = DASHBOARD_WINDOW_DAYS,
    project_root: str | Path = ".",
) -> DashboardResponse:
    progress = build_progress(connection, window_days, config=config)
    long_progress = progress
    capacity_progress = progress
    quality_progress = progress
    quality_signal = _quality_fitness_signal(connection, config, quality_progress)
    status = load_current_status(connection)
    state = build_fitness_state(
        connection,
        config,
        health_status=status.health_status,
        window_days=window_days,
    )
    recommendation = recommend_next_run(state, status, config)
    latest = list_runs(connection, limit=1)
    feedback = get_run_feedback(connection, config, latest[0].activity_id) if latest else None
    setup = setup_state(connection, config)
    pending = [step for step in setup.steps if not step.complete]
    nudge = SetupNudge(
        complete=setup.complete,
        remaining=len(pending),
        detail=(
            pending[0].title + ". " + pending[0].detail
            if pending
            else "Every setting has been confirmed."
        ),
    )
    return DashboardResponse(
        progress=progress,
        training_status=build_training_status(state, config),
        fitness_interpretation=_interpret_fitness(progress, long_progress, capacity_progress, state, quality_signal),
        last_run=feedback,
        recommendation=recommendation,
        current_status=status,
        setup=nudge,
        weekly_schedule=ensure_current_weekly_schedule(connection, config, project_root),
    )
