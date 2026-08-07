"""Progress, volume, intensity, consistency, and durability analysis."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from statistics import median
import json
import sqlite3

from .analytics import build_fitness_analytics
from .external_fitness import summarize_external_fitness
from .segmentation import METERS_PER_MILE
from .training_load import (
    TrainingSession,
    acute_to_prior_weekly_ratio,
    calculate_session_load,
    distance_capacity,
    rolling_load,
)
from .vo2_estimation import estimate_local_vo2
from .web.schemas import (
    ConfidenceLevel,
    ConsistencySummary,
    FitnessPoint,
    FitnessBenchmarkSummary,
    FitnessCoverageItem,
    FitnessTrend,
    FitnessTrendPoint,
    IntensitySummary,
    LoadContext,
    LoadWindow,
    PaceValue,
    PeriodComparison,
    PeriodSummary,
    ProgressResponse,
    WorkoutType,
)


AVAILABLE_WINDOWS = (14, 28, 42, 56, 90, 180, 365)


def _pace_display(pace: float) -> str:
    minutes = int(pace)
    seconds = int(round((pace - minutes) * 60))
    if seconds == 60:
        minutes += 1
        seconds = 0
    return f"{minutes}:{seconds:02d}/mi"


def _workout(value: str | None) -> WorkoutType:
    normalized = str(value or "").casefold().strip().replace(" ", "_")
    aliases = {"tempo": "tempo_threshold", "interval": "intervals", "cycling": "bike"}
    try:
        return WorkoutType(aliases.get(normalized, normalized))
    except ValueError:
        return WorkoutType.UNKNOWN


def _trend_evidence_weight(health_tag: str, workout: WorkoutType) -> float:
    """Reliability weight for trend interpretation; load always remains full weight."""
    health_weight = {
        "normal": 1.0,
        "illness_recovery": 0.65,
        "illness": 0.25,
        "injury_affected": 0.25,
    }.get(health_tag, 0.5)
    workout_weight = {
        WorkoutType.INTERVALS: 0.0,
        WorkoutType.TEMPO_THRESHOLD: 0.0,
        WorkoutType.RACE: 0.0,
        WorkoutType.RUN_WALK: 0.5,
        WorkoutType.HIKE: 0.0,
        WorkoutType.BIKE: 0.0,
    }.get(workout, 1.0)
    return health_weight * workout_weight


def _load_window(value) -> LoadWindow:
    return LoadWindow(
        days=value.days,
        distance_miles=value.distance_miles,
        moving_minutes=value.moving_minutes,
        zone_load=value.zone_load,
        hard_minutes=value.hard_minutes,
        activity_count=value.activity_count,
    )


def _sessions(connection: sqlite3.Connection) -> tuple[list[TrainingSession], dict[int, dict]]:
    rows = connection.execute(
        """
        SELECT a.id,a.start_time_utc,a.total_distance_m,m.calculated_moving_time_s,
               m.device_timer_time_s,m.session_zone_load,m.hard_minutes,m.hr_zone_seconds_json,
               m.exclusion_reason,o.workout_type,o.health_tag
        FROM activities a JOIN activity_metrics m ON m.activity_id=a.id
        LEFT JOIN run_overrides o ON o.activity_id=a.activity_id
        WHERE a.start_time_utc IS NOT NULL ORDER BY a.start_time_utc_epoch,a.id
        """
    ).fetchall()
    sessions: list[TrainingSession] = []
    detail: dict[int, dict] = {}
    for row in rows:
        workout = _workout(row["workout_type"])
        exclusion = str(row["exclusion_reason"] or "")
        if workout in {WorkoutType.HIKE, WorkoutType.BIKE} or any(
            marker in exclusion
            for marker in ("probable_walk_or_hike_sensor_signature", "probable_bike_sensor_signature")
        ):
            continue
        moving_s = float(row["calculated_moving_time_s"] or row["device_timer_time_s"] or 0)
        zones = json.loads(row["hr_zone_seconds_json"] or "{}")
        load = calculate_session_load(zones, moving_s)
        session = TrainingSession(
            activity_id=int(row["id"]),
            start_time=datetime.fromisoformat(row["start_time_utc"]),
            distance_miles=float(row["total_distance_m"] or 0) / METERS_PER_MILE,
            moving_minutes=moving_s / 60,
            zone_load=float(row["session_zone_load"]) if row["session_zone_load"] is not None else load.zone_load,
            hard_minutes=float(row["hard_minutes"]) if row["hard_minutes"] is not None else load.hard_minutes,
        )
        sessions.append(session)
        detail[session.activity_id] = {
            "workout": workout,
            "health_tag": str(row["health_tag"] or "normal"),
            "load": load,
            "zones": zones,
        }
    return sessions, detail


def _scored_runs(
    connection: sqlite3.Connection, details: dict[int, dict]
) -> tuple[list[dict], list[FitnessPoint], list[dict], list[FitnessPoint]]:
    rows = connection.execute(
        """
        SELECT a.id,a.start_time_utc,a.total_distance_m,m.session_zone_load,
               mr.result_json,o.workout_type,o.health_tag
        FROM model_runs mr JOIN activities a ON a.id=mr.activity_id
        LEFT JOIN activity_metrics m ON m.activity_id=a.id
        LEFT JOIN run_overrides o ON o.activity_id=a.activity_id
        WHERE mr.model_name='standardized_pace_145'
        ORDER BY a.start_time_utc_epoch,a.id
        """
    ).fetchall()
    analytics_rows: list[dict] = []
    points: list[FitnessPoint] = []
    steady_rows: list[dict] = []
    steady_points: list[FitnessPoint] = []
    for row in rows:
        result = json.loads(row["result_json"])
        standardized = result.get("standardized_pace_145_min_mile")
        if standardized is None or not row["start_time_utc"]:
            continue
        uncertainty = float(result.get("uncertainty_95_min_mile") or 0)
        health_tag = str(row["health_tag"] or "normal")
        activity_id = int(row["id"])
        workout = details.get(activity_id, {}).get("workout") or _workout(row["workout_type"])
        trend_weight = _trend_evidence_weight(health_tag, workout)
        included_in_trend = trend_weight > 0
        if included_in_trend:
            analytics_rows.append(
                {
                    "start_time_utc": row["start_time_utc"],
                    "standardized_pace": float(standardized),
                    "uncertainty_95": uncertainty,
                    "trend_weight": trend_weight,
                }
            )
        points.append(
            FitnessPoint(
                activity_id=activity_id,
                start_time=datetime.fromisoformat(row["start_time_utc"]),
                raw_pace_min_mile=result.get("raw_pace_145_min_mile"),
                standardized_pace_min_mile=float(standardized),
                uncertainty_95_min_mile=uncertainty,
                distance_miles=float(row["total_distance_m"] or 0) / METERS_PER_MILE,
                zone_load=float(row["session_zone_load"]) if row["session_zone_load"] is not None else None,
                workout_type=workout,
                health_tag=health_tag,
                included_in_trend=included_in_trend,
                trend_weight=trend_weight,
                measurement_quality=str(result.get("estimate_quality") or "full_sensor"),
            )
        )
        benchmark = result.get("steady_aerobic_benchmark")
        if benchmark and benchmark.get("standardized_pace_145_min_mile") is not None:
            steady_pace = float(benchmark["standardized_pace_145_min_mile"])
            steady_uncertainty = float(benchmark.get("uncertainty_95_min_mile") or 0)
            if included_in_trend:
                steady_rows.append(
                    {
                        "start_time_utc": row["start_time_utc"],
                        "standardized_pace": steady_pace,
                        "uncertainty_95": steady_uncertainty,
                        "trend_weight": trend_weight,
                    }
                )
            steady_points.append(
                FitnessPoint(
                    activity_id=activity_id,
                    start_time=datetime.fromisoformat(row["start_time_utc"]),
                    raw_pace_min_mile=benchmark.get("raw_pace_145_min_mile"),
                    standardized_pace_min_mile=steady_pace,
                    uncertainty_95_min_mile=steady_uncertainty,
                    distance_miles=float(row["total_distance_m"] or 0) / METERS_PER_MILE,
                    zone_load=float(row["session_zone_load"]) if row["session_zone_load"] is not None else None,
                    workout_type=workout,
                    health_tag=health_tag,
                    included_in_trend=included_in_trend,
                    trend_weight=trend_weight,
                    measurement_quality=str(
                        benchmark.get("estimate_quality")
                        or result.get("estimate_quality")
                        or "full_sensor"
                    ),
                    benchmark_quality=str(
                        benchmark.get("selection_quality") or "strict_observed"
                    ),
                )
            )
    return analytics_rows, points, steady_rows, steady_points


def _period_summary(
    sessions: list[TrainingSession], start: datetime, end: datetime, standardized: float | None
) -> PeriodSummary:
    selected = [session for session in sessions if start < session.start_time <= end]
    return PeriodSummary(
        start_date=start.date(),
        end_date=end.date(),
        run_count=len(selected),
        distance_miles=sum(item.distance_miles for item in selected),
        moving_minutes=sum(item.moving_minutes for item in selected),
        zone_load=(
            sum(float(item.zone_load) for item in selected)
            if selected and all(item.zone_load is not None for item in selected)
            else None
        ),
        standardized_pace_min_mile=standardized,
        longest_run_miles=max((item.distance_miles for item in selected), default=0),
    )


def _percent_change(current: float | None, previous: float | None) -> float | None:
    if current is None or previous in (None, 0):
        return None
    return (current - previous) / previous * 100


def _trend(value: str) -> FitnessTrend:
    mapping = {
        "improving": FitnessTrend.IMPROVING,
        "declining": FitnessTrend.DECLINING,
        "stable": FitnessTrend.STABLE,
        "stable_or_uncertain": FitnessTrend.UNCERTAIN,
    }
    return mapping.get(value, FitnessTrend.INSUFFICIENT_DATA)


def _trend_series(rows: list[dict], days: int) -> list[FitnessTrendPoint]:
    if not rows:
        return []
    analysis = build_fitness_analytics(rows, days)
    if not analysis.get("available"):
        return []
    return [
        FitnessTrendPoint(
            as_of=datetime.fromisoformat(item["as_of_utc"]),
            pace_min_mile=float(item["pace_min_mile"]),
            uncertainty_95_min_mile=float(item["uncertainty_95_min_mile"]),
            run_count=int(item["run_count"]),
        )
        for item in analysis["historical"]
    ]


def _activity_coverage(
    connection: sqlite3.Connection, start: datetime, end: datetime
) -> list[FitnessCoverageItem]:
    rows = connection.execute(
        """
        SELECT a.id,a.start_time_utc,a.total_distance_m,m.exclusion_reason,
               o.workout_type,o.health_tag,mr.result_json
        FROM activities a
        LEFT JOIN activity_metrics m ON m.activity_id=a.id
        LEFT JOIN run_overrides o ON o.activity_id=a.activity_id
        LEFT JOIN model_runs mr ON mr.activity_id=a.id AND mr.model_name='standardized_pace_145'
        WHERE a.start_time_utc_epoch>? AND a.start_time_utc_epoch<=?
        ORDER BY a.start_time_utc_epoch DESC,a.id DESC
        """,
        (start.timestamp(), end.timestamp()),
    ).fetchall()
    output = []
    for row in rows:
        workout = _workout(row["workout_type"])
        health_tag = str(row["health_tag"] or "normal")
        result = json.loads(row["result_json"]) if row["result_json"] else None
        trend_weight = _trend_evidence_weight(health_tag, workout) if result else 0.0
        included = trend_weight > 0
        estimate_quality = str(result.get("estimate_quality") or "full_sensor") if result else ""
        if result and estimate_quality != "full_sensor" and included:
            status = "uncertain_estimate"
            gps_percent = float(result.get("gps_coverage_fraction") or 0.0) * 100.0
            added_uncertainty = float(result.get("fallback_uncertainty_95_min_mile") or 0.0) * 60.0
            reason = (
                f"Scored from Garmin device-distance and HR windows with {gps_percent:.0f}% GPS coverage; "
                f"an added ±{added_uncertainty:.0f} sec/mi uncertainty reduces inverse-variance influence, "
                f"and health/workout context contributes a separate {trend_weight:.0%} weight."
            )
        elif result and trend_weight == 1:
            status = "trend_evidence"
            reason = "Scored at full weight in the modeled fitness trend."
        elif result and included:
            status = "reduced_weight"
            reason = (
                f"Scored at {trend_weight:.0%} context weight because health is "
                f"{health_tag.replace('_', ' ')} and workout type is {workout.value.replace('_', ' ')}."
            )
        elif result:
            status = "context_only"
            reason = f"Scored for display; {workout.value.replace('_', ' ')} does not vote in the running trend."
        elif workout == WorkoutType.INTERVALS:
            status = "workout_specific"
            reason = (
                "Scored in the interval execution/control/stimulus/recovery analysis; intentionally excluded "
                "from the steady aerobic pace@145 trend."
            )
        elif workout in {WorkoutType.HIKE, WorkoutType.BIKE}:
            status = "non_running"
            reason = f"Kept in history/load as {workout.value}; no running-fitness score."
        else:
            status = "unscored"
            raw_reason = str(row["exclusion_reason"] or "no reliable aerobic windows")
            reason = "No modeled fitness point: " + raw_reason.replace("_", " ").replace(";", "; ") + "."
        output.append(
            FitnessCoverageItem(
                activity_id=int(row["id"]),
                start_time=datetime.fromisoformat(row["start_time_utc"]),
                distance_miles=float(row["total_distance_m"] or 0) / METERS_PER_MILE,
                workout_type=workout,
                health_tag=health_tag,
                score_status=status,
                standardized_pace_min_mile=(
                    float(result["standardized_pace_145_min_mile"]) if result else None
                ),
                included_in_trend=included,
                trend_weight=trend_weight,
                reason=reason,
            )
        )
    return output


def build_progress(
    connection: sqlite3.Connection,
    window_days: int = 28,
    *,
    as_of: datetime | None = None,
    config: dict | None = None,
) -> ProgressResponse:
    sessions, details = _sessions(connection)
    analytics_rows, points, steady_rows, steady_points = _scored_runs(connection, details)
    now = datetime.now(timezone.utc)
    if as_of is None:
        as_of = now
        if sessions and sessions[-1].start_time > now + timedelta(days=1):
            as_of = sessions[-1].start_time
    elif as_of.tzinfo is None:
        raise ValueError("as_of must include a timezone")
    analytics_rows = [
        row for row in analytics_rows if datetime.fromisoformat(row["start_time_utc"]) <= as_of
    ]
    steady_rows = [
        row for row in steady_rows if datetime.fromisoformat(row["start_time_utc"]) <= as_of
    ]
    # The selected timeframe is both an analytical comparison window and the
    # visible chart domain.  Keeping a fixed one-year graph made the controls
    # appear inert even when the headline calculation changed.
    chart_start = as_of - timedelta(days=window_days)
    points = [point for point in points if chart_start < point.start_time <= as_of]
    steady_points = [point for point in steady_points if chart_start < point.start_time <= as_of]
    analysis = build_fitness_analytics(analytics_rows, window_days) if analytics_rows else {"available": False}

    current_pace = None
    uncertainty = None
    pace_change = None
    pace_change_uncertainty = None
    trend = FitnessTrend.INSUFFICIENT_DATA
    confidence = ConfidenceLevel.UNAVAILABLE
    definition = f"Robust trailing {window_days}-day estimate of reference-condition pace at target HR"
    current_standardized = previous_standardized = None
    if analysis.get("available"):
        current = analysis["current"]
        current_standardized = float(current["pace_min_mile"])
        current_pace = PaceValue(minutes_per_mile=current_standardized, display=_pace_display(current_standardized))
        uncertainty = float(current["uncertainty_95_min_mile"])
        change = analysis.get("change_prior_window")
        if change:
            pace_change = float(change["pace_change_seconds_per_mile"])
            pace_change_uncertainty = float(change["uncertainty_95_seconds_per_mile"])
            previous_standardized = float(change["prior"]["pace_min_mile"])
            trend = _trend(change["direction"])
        else:
            trend = FitnessTrend.INSUFFICIENT_DATA
        evidence = analysis.get("comparison_evidence_quality", analysis.get("evidence_quality"))
        confidence = {"good": ConfidenceLevel.HIGH, "moderate": ConfidenceLevel.MODERATE}.get(evidence, ConfidenceLevel.LOW)
        if confidence == ConfidenceLevel.LOW and trend in {
            FitnessTrend.IMPROVING,
            FitnessTrend.DECLINING,
        }:
            trend = FitnessTrend.UNCERTAIN
        definition = str(analysis["definition"])

    steady_analysis = build_fitness_analytics(steady_rows, window_days) if steady_rows else {"available": False}
    steady_current_pace = None
    steady_uncertainty = None
    steady_change = None
    steady_trend = FitnessTrend.INSUFFICIENT_DATA
    steady_confidence = ConfidenceLevel.UNAVAILABLE
    if steady_analysis.get("available"):
        steady_current = steady_analysis["current"]
        steady_value = float(steady_current["pace_min_mile"])
        steady_current_pace = PaceValue(minutes_per_mile=steady_value, display=_pace_display(steady_value))
        steady_uncertainty = float(steady_current["uncertainty_95_min_mile"])
        steady_delta = steady_analysis.get("change_prior_window")
        if steady_delta:
            steady_change = float(steady_delta["pace_change_seconds_per_mile"])
            steady_trend = _trend(steady_delta["direction"])
        evidence = steady_analysis.get(
            "comparison_evidence_quality", steady_analysis.get("evidence_quality")
        )
        steady_confidence = {
            "good": ConfidenceLevel.HIGH,
            "moderate": ConfidenceLevel.MODERATE,
        }.get(evidence, ConfidenceLevel.LOW)
        if steady_confidence == ConfidenceLevel.LOW and steady_trend in {
            FitnessTrend.IMPROVING,
            FitnessTrend.DECLINING,
        }:
            steady_trend = FitnessTrend.UNCERTAIN
    steady_summary = FitnessBenchmarkSummary(
        definition=(
            "Fixed-time corroboration at minute 20: a strict continuous 120-second HR 140–150 "
            "window is used when available. Otherwise the nearest reliable window is HR-normalized "
            "and retained with larger stability/sensor uncertainty; weather and grade are standardized to HR 145."
        ),
        trend=steady_trend,
        confidence=steady_confidence,
        current_pace=steady_current_pace,
        uncertainty_95_min_mile=steady_uncertainty,
        pace_change_seconds_per_mile=steady_change,
        eligible_run_count=len(steady_points),
        strict_run_count=sum(
            point.benchmark_quality == "strict_observed" for point in steady_points
        ),
        estimated_run_count=sum(
            point.benchmark_quality == "estimated_fixed_time" for point in steady_points
        ),
        series=steady_points,
        trend_7d=[item for item in _trend_series(steady_rows, 7) if item.as_of > chart_start],
        trend_28d=[item for item in _trend_series(steady_rows, 28) if item.as_of > chart_start],
    )

    current_start = as_of - timedelta(days=window_days)
    previous_end = current_start
    previous_start = previous_end - timedelta(days=window_days)
    current_period = _period_summary(sessions, current_start, as_of, current_standardized)
    previous_period = _period_summary(sessions, previous_start, previous_end, previous_standardized)
    distance_change = _percent_change(current_period.distance_miles, previous_period.distance_miles)
    load_change = _percent_change(current_period.zone_load, previous_period.zone_load)
    comparison = PeriodComparison(
        current=current_period,
        previous=previous_period,
        pace_change_seconds_per_mile=pace_change,
        distance_change_percent=distance_change,
        load_change_percent=load_change,
        interpretation=(
            "Pace and training load moved in different dimensions; inspect both before attributing change to fitness."
            if pace_change is not None and load_change is not None
            else "Comparison is limited by missing standardized pace or HR-derived load."
        ),
    )

    loads = [rolling_load(sessions, as_of, days) for days in (7, 14, 28)]
    any_missing_load = any(
        item.zone_load is None
        for item in sessions
        if as_of - timedelta(days=28) < item.start_time <= as_of
    )
    capacity = distance_capacity(
        sessions,
        as_of,
        retention_half_life_days=float(
            (config or {}).get("coaching", {}).get("capacity_retention_half_life_days", 42)
        ),
        retention_grace_days=int(
            (config or {}).get("coaching", {}).get("capacity_retention_grace_days", 28)
        ),
    )
    current_load = LoadContext(
        trailing_7d=_load_window(loads[0]),
        trailing_14d=_load_window(loads[1]),
        trailing_28d=_load_window(loads[2]),
        acute_to_prior_ratio=acute_to_prior_weekly_ratio(sessions, as_of),
        acute_distance_to_capacity_ratio=capacity.acute_to_capacity_ratio,
        prior_28d_weekly_miles=capacity.prior_28d_weekly_miles,
        sustained_capacity_miles=capacity.sustained_weekly_miles,
        capacity_reference_miles=capacity.reference_miles,
        confidence=ConfidenceLevel.LOW if any_missing_load else ConfidenceLevel.HIGH,
        flags=["some_recent_sessions_missing_hr_load"] if any_missing_load else [],
    )

    recent = [item for item in sessions if current_start < item.start_time <= as_of]
    ordered_dates = sorted(item.start_time for item in recent)
    gaps = [(right - left).total_seconds() / 86400 for left, right in zip(ordered_dates, ordered_dates[1:])]
    consistency = ConsistencySummary(
        running_days=len({item.start_time.date() for item in recent}),
        runs_per_week=len(recent) / window_days * 7,
        longest_gap_days=max(gaps) if gaps else None,
        longest_run_miles=max((item.distance_miles for item in recent), default=0),
        quality_sessions=sum(
            details[item.activity_id]["health_tag"] == "normal"
            and details[item.activity_id]["workout"]
            in {WorkoutType.INTERVALS, WorkoutType.TEMPO_THRESHOLD, WorkoutType.RACE}
            for item in recent
        ),
    )

    easy = moderate = hard = missing = 0.0
    for item in recent:
        load = details[item.activity_id]["load"]
        easy += load.easy_minutes
        moderate += load.moderate_minutes
        hard += load.hard_minutes
        missing += load.unknown_hr_minutes
    known = easy + moderate + hard
    intensity = IntensitySummary(
        easy_percent=easy / known * 100 if known else None,
        moderate_percent=moderate / known * 100 if known else None,
        hard_percent=hard / known * 100 if known else None,
        known_hr_minutes=known,
        missing_hr_minutes=missing,
        confidence=ConfidenceLevel.HIGH if known and missing / (known + missing) <= 0.1 else ConfidenceLevel.LOW,
    )
    blind_spots = []
    if any_missing_load:
        blind_spots.append("Some recent activity intensity is unknown because HR coverage is incomplete.")
    if not analytics_rows:
        blind_spots.append("No runs meet the comparable fitness-observation requirements.")
    if sessions and (as_of - sessions[-1].start_time).days > 14:
        blind_spots.append("The latest activity is stale, so current readiness cannot be inferred reliably.")
    blind_spots.append("Sleep, soreness, stress, nutrition, and unrecorded cross-training are not present in TCX data.")
    return ProgressResponse(
        as_of=as_of,
        window_days=window_days,
        reference_within_run_minutes=float(
            (config or {}).get("reference_conditions", {}).get("within_run_minutes", 20)
        ),
        available_windows=list(AVAILABLE_WINDOWS),
        fitness_trend=trend,
        fitness_confidence=confidence,
        current_pace=current_pace,
        uncertainty_95_min_mile=uncertainty,
        pace_change_seconds_per_mile=pace_change,
        pace_change_uncertainty_95_seconds_per_mile=pace_change_uncertainty,
        definition=definition,
        series=points,
        trend_7d=[item for item in _trend_series(analytics_rows, 7) if item.as_of > chart_start],
        trend_28d=[item for item in _trend_series(analytics_rows, 28) if item.as_of > chart_start],
        steady_aerobic=steady_summary,
        activity_coverage=_activity_coverage(connection, current_start, as_of),
        period_comparison=comparison,
        current_load=current_load,
        consistency=consistency,
        intensity=intensity,
        external_fitness=summarize_external_fitness(connection, as_of),
        local_vo2_estimate=estimate_local_vo2(
            current_pace=current_pace,
            as_of=as_of,
            recent_load=current_load,
            fitness_trend=trend,
            config=config or {},
        ),
        blind_spots=blind_spots,
    )
