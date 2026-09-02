"""Build the compact database-independent state used by coaching rules."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import sqlite3
from statistics import median

from .durability import supported_long_run_capacity
from .easy_baseline import recency_weighted_easy_distance
from .progress import PreparedProgressData, build_progress
from .recovery import athlete_relative_session_load, cumulative_recovery_load
from .run_feedback import get_run_feedback, list_runs
from .web.schemas import (
    ConfidenceLevel,
    ContextEvidence,
    CurrentHealthStatus,
    EvidenceAvailability,
    FitnessState,
    LoadWindow,
    PaceChange,
    WeatherExposureBaseline,
    WorkoutType,
)


def _cumulative_recovery_residual(
    runs,
    as_of: datetime,
    typical_load: LoadWindow,
    *,
    latest_run=None,
    latest_performance_anomaly: str = "within_recent_range",
    latest_drift_percent: float | None = None,
) -> float:
    """Accumulate independently decayed recovery load from recorded runs."""

    session_loads: list[tuple[float, float]] = []
    for run in runs:
        if (
            not run.start_time
            or run.start_time > as_of
            or not run.session_difficulty
            or run.workout_type in {WorkoutType.HIKE, WorkoutType.BIKE}
        ):
            continue
        is_latest = latest_run is not None and run is latest_run
        initial, _ = athlete_relative_session_load(
            run.session_difficulty,
            typical_load,
            performance_anomaly=(
                latest_performance_anomaly
                if is_latest
                else "within_recent_range"
            ),
            drift_percent=latest_drift_percent if is_latest else None,
        )
        elapsed_hours = max(
            0.0,
            (as_of - run.start_time).total_seconds() / 3600.0,
        )
        session_loads.append((initial, elapsed_hours))
    return cumulative_recovery_load(session_loads)


def _change(progress, days: int) -> PaceChange | None:
    value = progress.pace_change_seconds_per_mile
    if value is None:
        return None
    return PaceChange(
        seconds_per_mile=value,
        direction=progress.fitness_trend,
        comparison_days=days,
    )


def _performance_anomaly(points) -> str:
    # Graph-only context points deliberately have no standardized pace and
    # zero trend influence. Performance response must use the same evidence
    # set as the aerobic trend rather than treating the newest visible marker
    # as a scored run.
    evidence = [
        point
        for point in points
        if point.included_in_trend
        and point.trend_weight > 0
        and point.standardized_pace_min_mile is not None
    ]
    if len(evidence) < 4:
        return "unknown"
    latest = evidence[-1]
    prior = [
        point.standardized_pace_min_mile
        for point in evidence[:-1]
        if latest.start_time - timedelta(days=56) < point.start_time < latest.start_time
    ][-8:]
    if len(prior) < 3:
        return "unknown"
    difference = latest.standardized_pace_min_mile - median(prior)
    threshold = max(0.4, latest.uncertainty_95_min_mile)
    if difference > threshold:
        return "unusually_costly"
    if difference < -threshold:
        return "unusually_strong"
    return "within_recent_range"


def _unplanned_moderate_context(runs, as_of: datetime) -> tuple[float | None, int]:
    """Z3 share from normal-health, non-quality running only.

    Planned quality is not "leakage," and illness-affected HR elevation is
    physiological context rather than evidence that easy-day discipline has
    changed. Those sessions still remain in distance, duration, and load.
    """

    quality_types = {WorkoutType.INTERVALS, WorkoutType.TEMPO_THRESHOLD, WorkoutType.RACE}
    excluded_types = quality_types | {WorkoutType.RUN_WALK, WorkoutType.HIKE, WorkoutType.BIKE}
    moderate = known = 0.0
    evidence_runs = 0
    for run in runs:
        if (
            not run.start_time
            or not as_of - timedelta(days=14) < run.start_time.astimezone(timezone.utc) <= as_of
            or run.health_tag.value != "normal"
            or run.workout_type in excluded_types
            or not run.session_difficulty
        ):
            continue
        zones = run.session_difficulty.zone_breakdown.zone_seconds
        evidence_runs += 1
        moderate += float(zones.get("z3", 0) or 0)
        known += sum(
            float(zones.get(name, 0) or 0)
            for name in ("below_z1", "z1", "z2", "z3", "z4", "z5", "above_z5")
        )
    return (moderate / known if known else None, evidence_runs)


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _weather_exposure_baseline(
    connection: sqlite3.Connection,
    runs,
    evaluation_time: datetime,
    *,
    lookback_days: int = 28,
) -> WeatherExposureBaseline | None:
    activity_ids = [
        run.activity_id
        for run in runs
        if run.start_time
        and evaluation_time - timedelta(days=lookback_days)
        < run.start_time.astimezone(timezone.utc)
        <= evaluation_time
        and run.workout_type not in {WorkoutType.HIKE, WorkoutType.BIKE}
    ]
    if not activity_ids:
        return None
    placeholders = ",".join("?" for _ in activity_ids)
    rows = connection.execute(
        f"""
        SELECT temperature_f, apparent_temperature_f, dewpoint_f
        FROM activity_weather
        WHERE activity_id IN ({placeholders})
        """,
        activity_ids,
    ).fetchall()
    apparent = [
        float(row["apparent_temperature_f"] or row["temperature_f"])
        for row in rows
        if row["apparent_temperature_f"] is not None
        or row["temperature_f"] is not None
    ]
    dewpoints = [
        float(row["dewpoint_f"])
        for row in rows
        if row["dewpoint_f"] is not None
    ]
    if not apparent and not dewpoints:
        return None
    return WeatherExposureBaseline(
        lookback_days=lookback_days,
        sample_count=max(len(apparent), len(dewpoints)),
        warm_apparent_temperature_f=_percentile(apparent, 0.75),
        cold_apparent_temperature_f=_percentile(apparent, 0.25),
        humid_dewpoint_f=_percentile(dewpoints, 0.75),
    )


def build_fitness_state(
    connection: sqlite3.Connection,
    config: dict,
    *,
    health_status: CurrentHealthStatus = CurrentHealthStatus.NORMAL,
    window_days: int = 28,
    as_of: datetime | None = None,
    prepared_progress: PreparedProgressData | None = None,
    preloaded_runs=None,
    preloaded_latest_feedback=None,
) -> FitnessState:
    evaluation_time = as_of or datetime.now(timezone.utc)
    if evaluation_time.tzinfo is None:
        raise ValueError("as_of must include a timezone")
    progress = build_progress(
        connection,
        window_days,
        as_of=evaluation_time,
        config=config,
        prepared=prepared_progress,
    )
    progress_14 = build_progress(
        connection,
        14,
        as_of=evaluation_time,
        config=config,
        prepared=prepared_progress,
    )
    runs = list(preloaded_runs) if preloaded_runs is not None else list_runs(connection, limit=500)
    runs = [run for run in runs if run.start_time and run.start_time <= evaluation_time]
    as_of = progress.as_of
    latest = runs[0] if runs else None
    latest_feedback = (
        preloaded_latest_feedback
        if latest and preloaded_latest_feedback is not None
        else get_run_feedback(connection, config, latest.activity_id)
        if latest
        else None
    )
    days_since_last = None
    if latest and latest.start_time:
        days_since_last = max(0.0, (evaluation_time - latest.start_time.astimezone(timezone.utc)).total_seconds() / 86400)

    quality = [
        run for run in runs
        if run.start_time
        and run.health_tag.value == "normal"
        and run.workout_type not in {WorkoutType.RUN_WALK, WorkoutType.HIKE, WorkoutType.BIKE}
        and (
            run.workout_type in {WorkoutType.INTERVALS, WorkoutType.TEMPO_THRESHOLD, WorkoutType.RACE}
            or (run.session_difficulty and run.session_difficulty.is_quality_session)
        )
    ]
    long_runs = [
        run for run in runs
        if run.start_time and run.session_difficulty and run.session_difficulty.is_long_run
    ]
    coaching = config.get("coaching", {})
    retained_long_capacity = supported_long_run_capacity(
        (
            (
                run.start_time.astimezone(timezone.utc),
                run.distance_miles,
                (
                    run.prescribed_distance_range_miles[1]
                    if run.prescribed_distance_range_miles is not None
                    else None
                ),
            )
            for run in runs
            if run.start_time
            and run.health_tag.value == "normal"
            and run.workout_type not in {WorkoutType.RUN_WALK, WorkoutType.HIKE, WorkoutType.BIKE}
        ),
        evaluation_time,
        grace_days=float(coaching.get("long_run_retention_grace_days", 90)),
        half_life_days=float(
            coaching.get("long_run_retention_half_life_days", 180)
        ),
    )
    recent_long_capacity = supported_long_run_capacity(
        (
            (
                run.start_time.astimezone(timezone.utc),
                run.distance_miles,
                (
                    run.prescribed_distance_range_miles[1]
                    if run.prescribed_distance_range_miles is not None
                    else None
                ),
            )
            for run in runs
            if run.start_time
            and evaluation_time - run.start_time.astimezone(timezone.utc)
            <= timedelta(days=30)
            and run.health_tag.value == "normal"
            and run.workout_type
            not in {WorkoutType.RUN_WALK, WorkoutType.HIKE, WorkoutType.BIKE}
        ),
        evaluation_time,
        grace_days=30,
        half_life_days=float(
            coaching.get("long_run_retention_half_life_days", 180)
        ),
    )
    days_since_quality = (
        max(0.0, (evaluation_time - quality[0].start_time.astimezone(timezone.utc)).total_seconds() / 86400)
        if quality else None
    )
    days_since_long = (
        max(0.0, (evaluation_time - long_runs[0].start_time.astimezone(timezone.utc)).total_seconds() / 86400)
        if long_runs else None
    )
    recent_illness = any(
        run.start_time
        and evaluation_time - run.start_time.astimezone(timezone.utc) <= timedelta(days=21)
        and run.health_tag.value in {"illness", "illness_recovery", "injury_affected"}
        for run in runs
    )
    most_recent_abnormal = next(
        (
            run for run in runs
            if run.start_time and run.health_tag.value in {"illness", "illness_recovery", "injury_affected"}
        ),
        None,
    )
    normal_since_health_event = (
        sum(
            run.start_time > most_recent_abnormal.start_time and run.health_tag.value == "normal"
            for run in runs
            if run.start_time
        )
        if most_recent_abnormal and most_recent_abnormal.start_time
        else 0
    )
    ordinary_easy_samples = [
        (run.start_time.astimezone(timezone.utc), run.distance_miles)
        for run in runs
        if run.start_time
        and run.health_tag.value == "normal"
        and run.workout_type == WorkoutType.EASY
        and run.prescribed_planning_role != "support_easy"
        and run.distance_miles > 0
    ]
    typical_easy_run_miles = recency_weighted_easy_distance(
        ordinary_easy_samples,
        evaluation_time,
        half_life_days=float(
            coaching.get("capacity_retention_half_life_days", 84)
        ),
    )

    context = [
        ContextEvidence(
            factor="running history",
            availability=EvidenceAvailability.OBSERVED if runs else EvidenceAvailability.MISSING,
            reliability=ConfidenceLevel.HIGH if len(runs) >= 10 else ConfidenceLevel.LOW,
            detail=f"{len(runs)} activities available to the coaching state.",
        ),
        ContextEvidence(
            factor="heart-rate training load",
            availability=(EvidenceAvailability.OBSERVED if progress.current_load.confidence != ConfidenceLevel.LOW else EvidenceAvailability.INFERRED),
            reliability=progress.current_load.confidence,
            detail="Intensity load uses only recorded HR time; distance and duration remain separate.",
        ),
        ContextEvidence(
            factor="sleep, soreness, stress, nutrition, hydration",
            availability=EvidenceAvailability.MISSING,
            reliability=ConfidenceLevel.UNAVAILABLE,
            detail="Not present in TCX. Current health input is the only same-day recovery context.",
        ),
        ContextEvidence(
            factor="GPS pace evidence",
            availability=(EvidenceAvailability.OBSERVED if latest and latest.data_quality.value == "good" else EvidenceAvailability.INFERRED),
            reliability=(ConfidenceLevel.HIGH if latest and latest.data_quality.value == "good" else ConfidenceLevel.LOW),
            detail="Missing GPS still counts toward time/HR workload but weakens pace comparisons.",
        ),
    ]
    blind_spots = list(progress.blind_spots)
    if health_status == CurrentHealthStatus.NORMAL:
        blind_spots.append("A 'normal' check-in is not medical clearance and does not reveal unreported pain or shortness of breath.")
    unplanned_moderate, unplanned_moderate_runs = _unplanned_moderate_context(
        runs, evaluation_time
    )
    weather_exposure = _weather_exposure_baseline(
        connection, runs, evaluation_time
    )
    context.append(
        ContextEvidence(
            factor="recent weather exposure",
            availability=(
                EvidenceAvailability.OBSERVED
                if weather_exposure and weather_exposure.sample_count >= 3
                else EvidenceAvailability.INFERRED
                if weather_exposure
                else EvidenceAvailability.MISSING
            ),
            reliability=(
                ConfidenceLevel.HIGH
                if weather_exposure and weather_exposure.sample_count >= 6
                else ConfidenceLevel.MODERATE
                if weather_exposure and weather_exposure.sample_count >= 3
                else ConfidenceLevel.LOW
                if weather_exposure
                else ConfidenceLevel.UNAVAILABLE
            ),
            detail=(
                f"Recent adaptation references {weather_exposure.sample_count} runs with weather in the past {weather_exposure.lookback_days} days."
                if weather_exposure
                else "No recent run-weather samples are available for adaptation."
            ),
        )
    )
    performance_anomaly = _performance_anomaly(progress.series)
    latest_drift_percent = (
        latest_feedback.cardiac_drift.decoupling_percent
        if latest_feedback and latest_feedback.cardiac_drift.valid
        else None
    )
    recovery_residual_load = _cumulative_recovery_residual(
        runs,
        evaluation_time,
        progress.current_load.trailing_28d,
        latest_run=latest,
        latest_performance_anomaly=performance_anomaly,
        latest_drift_percent=latest_drift_percent,
    )
    return FitnessState(
        as_of=as_of,
        window_days=window_days,
        fitness_trend=progress.fitness_trend,
        trend_confidence=progress.fitness_confidence,
        standardized_pace_at_target_hr=progress.current_pace,
        short_term_change=_change(progress, window_days),
        medium_term_change=None,
        recent_load=progress.current_load,
        days_since_last_run=days_since_last,
        days_since_quality_run=days_since_quality,
        days_since_long_run=days_since_long,
        last_run=latest.session_difficulty if latest else None,
        last_run_workout_type=latest.workout_type if latest else None,
        last_run_drift_percent=latest_drift_percent,
        recovery_residual_load=recovery_residual_load,
        longest_run_30d_miles=recent_long_capacity,
        retained_long_run_capacity_miles=retained_long_capacity,
        quality_sessions_14d=progress_14.consistency.quality_sessions,
        completed_quality_session_count=len(quality),
        running_days_28d=progress.consistency.running_days,
        typical_easy_run_miles=typical_easy_run_miles,
        easy_fraction_14d=(progress_14.intensity.easy_percent / 100 if progress_14.intensity.easy_percent is not None else None),
        moderate_fraction_14d=unplanned_moderate,
        moderate_evidence_runs_14d=unplanned_moderate_runs,
        hard_fraction_14d=(progress_14.intensity.hard_percent / 100 if progress_14.intensity.hard_percent is not None else None),
        recent_performance_anomaly=performance_anomaly,
        recent_illness_or_recovery=recent_illness,
        normal_runs_since_health_event=normal_since_health_event,
        current_health_status=health_status,
        anomaly_flags=[performance_anomaly] if performance_anomaly not in {"unknown", "within_recent_range"} else [],
        data_quality_flags=(
            list(progress.current_load.flags)
            + (["latest_run_pace_quality_low"] if latest and latest.data_quality.value != "good" else [])
        ),
        context_evidence=context,
        known_blind_spots=blind_spots,
        weather_exposure_baseline=weather_exposure,
    )
