"""Compose the three primary application jobs for the home screen."""

from __future__ import annotations

from datetime import timedelta
import sqlite3
from pathlib import Path

from .fitness_state import build_fitness_state
from .progress import build_progress
from .recommendation import recommend_next_run
from .recommendation_service import ensure_current_weekly_schedule, load_current_status
from .run_feedback import get_run_feedback, list_runs
from .web.schemas import DashboardResponse
from .web.schemas import (
    ConfidenceLevel,
    FitnessHorizon,
    FitnessInterpretation,
    FitnessSignal,
    FitnessTrend,
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
        FitnessTrend.DECLINING: "Down",
        FitnessTrend.STABLE: "Stable",
        FitnessTrend.UNCERTAIN: "Ambiguous / stable",
        FitnessTrend.INSUFFICIENT_DATA: "Insufficient evidence",
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
    status = "Insufficient recent evidence"
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
        trend, status = FitnessTrend.UNCERTAIN, "Baseline established"

    rejected = len(candidates) - count
    detail = (
        f"{count} usable normal-health quality sessions in the last {window_days} days"
        f"{f'; {rejected} labeled sessions lacked workout-specific evidence' if rejected else ''}."
        f"{comparison_detail} Confidence advances at 3 and 5 usable sessions."
    )
    return FitnessSignal(
        label="High-intensity fitness",
        trend=trend,
        status=status,
        confidence=confidence,
        detail=detail,
    )


def _interpret_fitness(short, long, capacity, state, quality_signal) -> FitnessInterpretation:
    current = capacity.period_comparison.current
    previous = capacity.period_comparison.previous
    distance_change = capacity.period_comparison.distance_change_percent
    longest_change = current.longest_run_miles - previous.longest_run_miles
    capacity_up = bool(
        (distance_change is not None and distance_change >= 10)
        or longest_change >= 1
        or current.run_count >= previous.run_count + 3
    )
    capacity_direction = FitnessTrend.IMPROVING if capacity_up else FitnessTrend.STABLE
    capacity_summary = (
        f"Over the last {capacity.window_days} days you ran {current.distance_miles:.1f} miles across "
        f"{current.run_count} runs, versus {previous.distance_miles:.1f} miles across "
        f"{previous.run_count} runs in the prior {capacity.window_days} days. Longest run: "
        f"{current.longest_run_miles:.1f} vs {previous.longest_run_miles:.1f} miles."
    )
    illness_context = None
    if state.recent_illness_or_recovery:
        illness_context = (
            "Recent runs are tagged illness/recovery. They still count as completed "
            "training load and contribute to aerobic efficiency at reduced reliability weight, making "
            "a short-term dip less diagnostic of durable fitness loss."
        )
    external = short.external_fitness
    external_up = (
        external.vo2_max_trend == FitnessTrend.IMPROVING
        and external.race_prediction_trend == FitnessTrend.IMPROVING
    )
    short_horizon = _horizon(f"Current {short.window_days}-day aerobic efficiency", short)
    # Kept as a compatibility field for existing API clients. The dashboard no longer
    # mixes a second lookback into its fitness interpretation.
    long_horizon = _horizon(f"Current {long.window_days}-day aerobic efficiency", long)
    if short_horizon.trend == FitnessTrend.DECLINING and capacity_up and external_up:
        headline = "Long-term fitness evidence is up; short-term efficiency is down."
        summary = (
            "Garmin VO2 max, Garmin race prediction, volume, and durability point upward. "
            "The recent pace-at-heart-rate dip remains real, but is better interpreted as "
            "current condition/recovery evidence than as a verdict that months of fitness were lost."
        )
    elif short_horizon.trend == FitnessTrend.DECLINING and capacity_up:
        headline = "Short-term efficiency is down; running capacity is up."
        summary = (
            "These are not contradictory. Pace at the same heart rate has recently been "
            "slower, while your ability to sustain more running and longer runs has grown. "
            "Illness and accumulated fatigue can affect the first signal without erasing the second."
        )
    elif short_horizon.trend == FitnessTrend.IMPROVING and capacity_up:
        headline = "Aerobic efficiency and running capacity are both improving."
        summary = f"The pace-at-heart-rate signal and {capacity.window_days}-day training-capacity measures agree."
    else:
        headline = "Fitness is mixed across time horizons."
        summary = (
            "Use aerobic efficiency, current recovery, and training capacity as separate evidence; "
            "none is a complete measure of fitness by itself."
        )
    if longest_change >= 1:
        durability_direction = FitnessTrend.IMPROVING
        durability_status = "Improved"
    elif longest_change <= -1:
        durability_direction = FitnessTrend.DECLINING
        durability_status = "Down"
    else:
        durability_direction = FitnessTrend.STABLE
        durability_status = "Stable"
    if state.recent_illness_or_recovery:
        recent_direction = FitnessTrend.DECLINING
        recent_status = "Recovering"
        recent_detail = "Recent health-tagged runs indicate temporarily suppressed form; current check-in controls coaching."
    elif state.recent_performance_anomaly == "unusually_costly":
        recent_direction = FitnessTrend.DECLINING
        recent_status = "Suppressed"
        recent_detail = "The latest standardized response was unusually costly relative to recent runs."
    elif state.recent_performance_anomaly == "within_recent_range":
        recent_direction = FitnessTrend.STABLE
        recent_status = "Within recent range"
        recent_detail = "The latest standardized response sits within ordinary recent variation."
    else:
        recent_direction = FitnessTrend.UNCERTAIN
        recent_status = "Ambiguous"
        recent_detail = "There is not enough comparable recent evidence to classify current form."
    signals = [
        FitnessSignal(
            label="Aerobic efficiency",
            trend=short_horizon.trend,
            status=_signal_status(short_horizon.trend),
            confidence=short_horizon.confidence,
            detail=f"{short_horizon.window_days}-day standardized pace-at-145 comparison.",
        ),
        FitnessSignal(
            label="Durability",
            trend=durability_direction,
            status=durability_status,
            confidence=ConfidenceLevel.MODERATE,
            detail=f"Longest run changed from {previous.longest_run_miles:.1f} to {current.longest_run_miles:.1f} miles across the compared {capacity.window_days}-day periods.",
        ),
        FitnessSignal(
            label="Training capacity",
            trend=capacity_direction,
            status=_signal_status(capacity_direction),
            confidence=ConfidenceLevel.MODERATE,
            detail=f"Volume and frequency across the current versus prior {capacity.window_days}-day period: {current.distance_miles:.1f} vs {previous.distance_miles:.1f} miles.",
        ),
        FitnessSignal(
            label="Recent form",
            trend=recent_direction,
            status=recent_status,
            confidence=ConfidenceLevel.MODERATE if state.recent_illness_or_recovery else state.trend_confidence,
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
            external.interpretation,
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
    return DashboardResponse(
        progress=progress,
        fitness_interpretation=_interpret_fitness(progress, long_progress, capacity_progress, state, quality_signal),
        last_run=feedback,
        recommendation=recommendation,
        current_status=status,
        weekly_schedule=ensure_current_weekly_schedule(connection, config, project_root),
    )
