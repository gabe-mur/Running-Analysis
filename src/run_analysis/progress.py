"""Progress, volume, intensity, consistency, and durability analysis."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from statistics import median
import json
import sqlite3

from .analytics import build_fitness_analytics
from .fitness_evidence import trend_evidence_reason, trend_evidence_weight
from .quality_phases import detect_continuous_quality_phase
from .run_feedback import _infer_workout_type, _long_run_threshold_from_connection
from .segmentation import METERS_PER_MILE
from .training_load import (
    TrainingSession,
    acute_to_prior_weekly_ratio,
    calculate_session_load,
    continuous_fatigue_load,
    continuous_distance_rate,
    distance_capacity,
    rolling_load,
    short_term_density_half_life_days,
)
from .intensity_balance import assess_intensity_balance
from .vo2_estimation import estimate_local_vo2, vo2_series
from .web.schemas import (
    AerobicChangeEvidence,
    ChangeEvidenceStrength,
    ConfidenceLevel,
    ConsistencySummary,
    FitnessPoint,
    FitnessBenchmarkSummary,
    FitnessCoverageItem,
    QualityPerformancePoint,
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

#: Only used when a caller supplies no configuration at all (tests, tooling).
#: Production callers always pass the athlete's configured comparison HR.
DEFAULT_TARGET_HR_BPM = 145.0


@dataclass(frozen=True, slots=True)
class PreparedProgressData:
    sessions: list[TrainingSession]
    details: dict[int, dict]
    analytics_rows: list[dict]
    points: list[FitnessPoint]
    steady_rows: list[dict]
    steady_points: list[FitnessPoint]
    long_run_threshold_miles: float


def _optional_float(value) -> float | None:
    """A missing score is missing, not a crash.

    Stored result payloads are written by whichever model version was current
    at the time, so a reader must tolerate a key that is absent rather than
    assume every scored run carries every field.
    """
    return None if value is None else float(value)


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


def _trend_evidence_weight(
    health_tag: str, workout: WorkoutType, result: dict | None = None
) -> float:
    """Backward-compatible local name for the shared evidence policy."""

    return trend_evidence_weight(health_tag, workout, result)


def _load_window(value) -> LoadWindow:
    return LoadWindow(
        days=value.days,
        distance_miles=value.distance_miles,
        moving_minutes=value.moving_minutes,
        zone_load=value.zone_load,
        hard_minutes=value.hard_minutes,
        activity_count=value.activity_count,
        zone_load_activity_count=value.zone_load_activity_count,
    )


def _sessions(
    connection: sqlite3.Connection,
    long_run_threshold_miles: float,
) -> tuple[list[TrainingSession], dict[int, dict]]:
    rows = connection.execute(
        """
        SELECT a.id,a.start_time_utc,a.total_distance_m,m.calculated_moving_time_s,
               m.device_timer_time_s,m.session_zone_load,m.hard_minutes,m.hr_zone_seconds_json,
               m.exclusion_reason,m.detected_workout_type,
               COALESCE(o.workout_type,ph.workout_type) AS workout_type,o.health_tag
        FROM activities a JOIN activity_metrics m ON m.activity_id=a.id
        LEFT JOIN run_overrides o ON o.activity_id=a.activity_id
        LEFT JOIN activity_plan_matches ap ON ap.activity_id=a.id
        LEFT JOIN planned_workout_history ph ON ph.id=ap.planned_workout_id
        WHERE a.start_time_utc IS NOT NULL ORDER BY a.start_time_utc_epoch,a.id
        """
    ).fetchall()
    sessions: list[TrainingSession] = []
    detail: dict[int, dict] = {}
    for row in rows:
        explicit_workout = _workout(row["workout_type"])
        exclusion = str(row["exclusion_reason"] or "")
        if explicit_workout in {WorkoutType.HIKE, WorkoutType.BIKE} or any(
            marker in exclusion
            for marker in ("probable_walk_or_hike_sensor_signature", "probable_bike_sensor_signature")
        ):
            continue
        moving_s = float(row["calculated_moving_time_s"] or row["device_timer_time_s"] or 0)
        zones = json.loads(row["hr_zone_seconds_json"] or "{}")
        distance_miles = float(row["total_distance_m"] or 0) / METERS_PER_MILE
        workout = _infer_workout_type(
            row["workout_type"],
            row["detected_workout_type"],
            distance_miles,
            moving_s / 60,
            zones,
            long_run_threshold_miles,
        )
        load = calculate_session_load(zones, moving_s)
        session = TrainingSession(
            activity_id=int(row["id"]),
            start_time=datetime.fromisoformat(row["start_time_utc"]),
            distance_miles=distance_miles,
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
               mr.result_json,COALESCE(o.workout_type,ph.workout_type) AS workout_type,o.health_tag
        FROM model_runs mr JOIN activities a ON a.id=mr.activity_id
        LEFT JOIN activity_metrics m ON m.activity_id=a.id
        LEFT JOIN run_overrides o ON o.activity_id=a.activity_id
        LEFT JOIN activity_plan_matches ap ON ap.activity_id=a.id
        LEFT JOIN planned_workout_history ph ON ph.id=ap.planned_workout_id
        WHERE mr.model_name='standardized_pace_at_target_hr'
        ORDER BY a.start_time_utc_epoch,a.id
        """
    ).fetchall()
    analytics_rows: list[dict] = []
    points: list[FitnessPoint] = []
    steady_rows: list[dict] = []
    steady_points: list[FitnessPoint] = []
    for row in rows:
        result = json.loads(row["result_json"])
        standardized = result.get("standardized_pace_at_target_hr_min_mile")
        if standardized is None or not row["start_time_utc"]:
            continue
        uncertainty = float(result.get("uncertainty_95_min_mile") or 0)
        health_tag = str(row["health_tag"] or "normal")
        activity_id = int(row["id"])
        workout = details.get(activity_id, {}).get("workout") or _workout(row["workout_type"])
        trend_weight = _trend_evidence_weight(health_tag, workout, result)
        included_in_trend = trend_weight > 0
        if included_in_trend:
            analytics_rows.append(
                {
                    "start_time_utc": row["start_time_utc"],
                    "standardized_pace": float(standardized),
                    "uncertainty_95": uncertainty,
                    "trend_weight": trend_weight,
                    "distance_miles": float(row["total_distance_m"] or 0)
                    / METERS_PER_MILE,
                }
            )
        points.append(
            FitnessPoint(
                activity_id=activity_id,
                start_time=datetime.fromisoformat(row["start_time_utc"]),
                raw_pace_min_mile=result.get("raw_pace_at_target_hr_min_mile"),
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
        if benchmark and benchmark.get("standardized_pace_at_target_hr_min_mile") is not None:
            steady_pace = float(benchmark["standardized_pace_at_target_hr_min_mile"])
            steady_uncertainty = float(benchmark.get("uncertainty_95_min_mile") or 0)
            if included_in_trend:
                steady_rows.append(
                    {
                        "start_time_utc": row["start_time_utc"],
                        "standardized_pace": steady_pace,
                        "uncertainty_95": steady_uncertainty,
                        "trend_weight": trend_weight,
                        "distance_miles": float(row["total_distance_m"] or 0)
                        / METERS_PER_MILE,
                    }
                )
            steady_points.append(
                FitnessPoint(
                    activity_id=activity_id,
                    start_time=datetime.fromisoformat(row["start_time_utc"]),
                    raw_pace_min_mile=benchmark.get("raw_pace_at_target_hr_min_mile"),
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


def prepare_progress_data(connection: sqlite3.Connection) -> PreparedProgressData:
    """Load invariant activity/model rows once for a multi-date projection."""

    long_run_threshold_miles = _long_run_threshold_from_connection(connection)
    sessions, details = _sessions(connection, long_run_threshold_miles)
    analytics_rows, points, steady_rows, steady_points = _scored_runs(
        connection, details
    )
    scored_activity_ids = {point.activity_id for point in points}
    for session in sessions:
        metadata = details.get(session.activity_id, {})
        workout = metadata.get("workout", WorkoutType.UNKNOWN)
        if (
            session.activity_id in scored_activity_ids
            or session.distance_miles <= 0
            or session.moving_minutes <= 0
        ):
            continue
        # Every running activity remains visible even when it was manually
        # excluded or no stable aerobic segment can support a pace-at-target-HR
        # estimate. Its actual full-run pace is explicitly context-only: useful
        # for locating and opening the run, but never evidence for the trend.
        raw_pace = session.moving_minutes / session.distance_miles
        points.append(
            FitnessPoint(
                activity_id=session.activity_id,
                start_time=session.start_time,
                raw_pace_min_mile=raw_pace,
                standardized_pace_min_mile=None,
                context_pace_min_mile=raw_pace,
                uncertainty_95_min_mile=0.0,
                distance_miles=session.distance_miles,
                zone_load=session.zone_load,
                workout_type=workout,
                health_tag=metadata.get("health_tag", "normal"),
                included_in_trend=False,
                trend_weight=0.0,
                measurement_quality="unadjusted_workout_context",
            )
        )
    points.sort(key=lambda point: (point.start_time, point.activity_id))
    return PreparedProgressData(
        sessions=sessions,
        details=details,
        analytics_rows=analytics_rows,
        points=points,
        steady_rows=steady_rows,
        steady_points=steady_points,
        long_run_threshold_miles=long_run_threshold_miles,
    )


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


def _change_evidence(
    value: dict | None,
    *,
    basis: str,
    confidence: ConfidenceLevel,
    comparison_run_count: int | None = None,
) -> AerobicChangeEvidence | None:
    if value is None:
        return None
    direction = _trend(
        str(value.get("directional_interpretation", value.get("direction")))
    )
    evidence = ChangeEvidenceStrength(
        str(value.get("evidence_strength", "inconclusive"))
    )
    # Directional probability cannot repair sparse coverage. Keep the numeric
    # estimate visible, but do not label it likely when the contributing
    # period itself has low evidence.
    if confidence in {ConfidenceLevel.LOW, ConfidenceLevel.UNAVAILABLE}:
        direction = FitnessTrend.UNCERTAIN
        evidence = ChangeEvidenceStrength.INCONCLUSIVE
    return AerobicChangeEvidence(
        basis=basis,
        direction=direction,
        evidence=evidence,
        confidence=confidence,
        pace_change_seconds_per_mile=float(
            value["pace_change_seconds_per_mile"]
        ),
        uncertainty_95_seconds_per_mile=float(
            value["uncertainty_95_seconds_per_mile"]
        ),
        probability_faster=float(value["probability_faster"]),
        run_count=int(value.get("run_count", 1)),
        comparison_run_count=comparison_run_count,
        coverage_fraction=float(value.get("coverage_fraction", 0.0)),
        distance_adjusted=bool(value.get("distance_adjusted", False)),
        distance_effect_seconds_per_mile_per_added_mile=_optional_float(
            value.get("distance_effect_seconds_per_mile_per_added_mile")
        ),
        distance_effect_uncertainty_95_seconds_per_mile_per_added_mile=_optional_float(
            value.get(
                "distance_effect_uncertainty_95_seconds_per_mile_per_added_mile"
            )
        ),
        current_weighted_distance_miles=_optional_float(
            value.get("current_weighted_distance_miles")
        ),
        prior_weighted_distance_miles=_optional_float(
            value.get("prior_weighted_distance_miles")
        ),
    )


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
    connection: sqlite3.Connection,
    start: datetime,
    end: datetime,
    long_run_threshold_miles: float,
) -> list[FitnessCoverageItem]:
    rows = connection.execute(
        """
        SELECT a.id,a.start_time_utc,a.total_distance_m,m.exclusion_reason,
               m.calculated_moving_time_s,m.device_timer_time_s,m.hr_zone_seconds_json,
               m.detected_workout_type,
               COALESCE(o.workout_type,ph.workout_type) AS workout_type,o.health_tag,mr.result_json
        FROM activities a
        LEFT JOIN activity_metrics m ON m.activity_id=a.id
        LEFT JOIN run_overrides o ON o.activity_id=a.activity_id
        LEFT JOIN activity_plan_matches ap ON ap.activity_id=a.id
        LEFT JOIN planned_workout_history ph ON ph.id=ap.planned_workout_id
        LEFT JOIN model_runs mr ON mr.activity_id=a.id AND mr.model_name='standardized_pace_at_target_hr'
        WHERE a.start_time_utc_epoch>? AND a.start_time_utc_epoch<=?
        ORDER BY a.start_time_utc_epoch DESC,a.id DESC
        """,
        (start.timestamp(), end.timestamp()),
    ).fetchall()
    output = []
    for row in rows:
        distance_miles = float(row["total_distance_m"] or 0) / METERS_PER_MILE
        moving_s = float(
            row["calculated_moving_time_s"]
            or row["device_timer_time_s"]
            or 0
        )
        workout = _infer_workout_type(
            row["workout_type"],
            row["detected_workout_type"],
            distance_miles,
            moving_s / 60,
            json.loads(row["hr_zone_seconds_json"] or "{}"),
            long_run_threshold_miles,
        )
        health_tag = str(row["health_tag"] or "normal")
        result = json.loads(row["result_json"]) if row["result_json"] else None
        trend_weight = _trend_evidence_weight(health_tag, workout, result) if result else 0.0
        included = trend_weight > 0
        estimate_quality = str(result.get("estimate_quality") or "full_sensor") if result else ""
        if result and estimate_quality != "full_sensor" and included:
            status = "uncertain_estimate"
            gps_percent = float(result.get("gps_coverage_fraction") or 0.0) * 100.0
            added_uncertainty = float(result.get("fallback_uncertainty_95_min_mile") or 0.0) * 60.0
            reason = (
                f"Estimated from Garmin distance because GPS covered {gps_percent:.0f}% of the run. "
                f"Allow about {added_uncertainty:.0f} sec/mi of extra variation. "
                + trend_evidence_reason(health_tag, workout, trend_weight, result)
            )
        elif result and trend_weight >= 0.999:
            status = "trend_evidence"
            reason = trend_evidence_reason(health_tag, workout, trend_weight, result)
        elif result and included:
            status = "reduced_weight"
            reason = trend_evidence_reason(health_tag, workout, trend_weight, result)
        elif result:
            status = "context_only"
            reason = trend_evidence_reason(health_tag, workout, trend_weight, result)
        elif workout not in {WorkoutType.HIKE, WorkoutType.BIKE}:
            status = "context_only"
            raw_reason = str(row["exclusion_reason"] or "no reliable aerobic windows")
            reason = (
                "Shown on the graph at unadjusted full-run pace as workout "
                "context only; 0% influence on the aerobic trend. No adjusted "
                "fitness estimate: "
                + raw_reason.replace("_", " ").replace(";", "; ")
                + "."
            )
        elif workout in {WorkoutType.HIKE, WorkoutType.BIKE}:
            status = "non_running"
            reason = f"Counts in history and activity load as {workout.value}, not running fitness."
        else:
            status = "unscored"
            raw_reason = str(row["exclusion_reason"] or "no reliable aerobic windows")
            reason = "Not used for fitness: " + raw_reason.replace("_", " ").replace(";", "; ") + "."
        output.append(
            FitnessCoverageItem(
                activity_id=int(row["id"]),
                start_time=datetime.fromisoformat(row["start_time_utc"]),
                distance_miles=distance_miles,
                workout_type=workout,
                health_tag=health_tag,
                score_status=status,
                standardized_pace_min_mile=_optional_float(
                    (result or {}).get("standardized_pace_at_target_hr_min_mile")
                ),
                included_in_trend=included,
                trend_weight=trend_weight,
                reason=reason,
            )
        )
    return output


def _quality_performance(
    connection: sqlite3.Connection, start: datetime, end: datetime, config: dict
) -> list[QualityPerformancePoint]:
    rows = connection.execute(
        """
        SELECT a.id,a.start_time_utc,COALESCE(o.workout_type,ph.workout_type) AS workout_type,mr.result_json
        FROM activities a
        LEFT JOIN model_runs mr ON mr.activity_id=a.id
          AND mr.model_name='standardized_pace_at_target_hr'
        LEFT JOIN run_overrides o ON o.activity_id=a.activity_id
        LEFT JOIN activity_plan_matches ap ON ap.activity_id=a.id
        LEFT JOIN planned_workout_history ph ON ph.id=ap.planned_workout_id
        WHERE a.start_time_utc_epoch>? AND a.start_time_utc_epoch<=?
        ORDER BY a.start_time_utc_epoch DESC,a.id DESC
        """,
        (start.timestamp(), end.timestamp()),
    ).fetchall()
    output: list[QualityPerformancePoint] = []
    for row in rows:
        workout = _workout(row["workout_type"])
        if workout == WorkoutType.TEMPO_THRESHOLD and config.get("zones"):
            phase = detect_continuous_quality_phase(
                connection, config, int(row["id"])
            )
            if phase is not None:
                output.append(
                    QualityPerformancePoint(
                        activity_id=int(row["id"]),
                        start_time=datetime.fromisoformat(row["start_time_utc"]),
                        workout_type=workout,
                        source=f"recorded_lap_{phase.lap_index + 1}",
                        duration_minutes=phase.duration_seconds / 60,
                        distance_miles=phase.distance_miles,
                        pace_min_mile=phase.pace_min_mile,
                        average_hr_bpm=phase.average_hr_bpm,
                        maximum_hr_bpm=phase.maximum_hr_bpm,
                    )
                )
                continue
        result = json.loads(row["result_json"]) if row["result_json"] else {}
        quality = result.get("quality_performance")
        if not isinstance(quality, dict) or not quality.get("duration_minutes"):
            continue
        output.append(
            QualityPerformancePoint(
                activity_id=int(row["id"]),
                start_time=datetime.fromisoformat(row["start_time_utc"]),
                workout_type=workout,
                source=str(quality.get("source") or "detected_work_block"),
                duration_minutes=float(quality["duration_minutes"]),
                distance_miles=_optional_float(quality.get("distance_miles")),
                pace_min_mile=_optional_float(quality.get("pace_min_mile")),
                average_hr_bpm=_optional_float(quality.get("average_hr_bpm")),
                maximum_hr_bpm=_optional_float(quality.get("maximum_hr_bpm")),
            )
        )
    return output


def build_progress(
    connection: sqlite3.Connection,
    window_days: int = 28,
    *,
    as_of: datetime | None = None,
    config: dict | None = None,
    prepared: PreparedProgressData | None = None,
) -> ProgressResponse:
    # Every "pace at X bpm" figure below is relative to the configured
    # comparison heart rate. Nothing may hard-code a number here.
    target_hr = float((config or {}).get("target_hr", DEFAULT_TARGET_HR_BPM))
    reference_minutes = float(
        (config or {}).get("reference_conditions", {}).get("within_run_minutes", 20)
    )
    prepared = prepared or prepare_progress_data(connection)
    sessions = prepared.sessions
    details = prepared.details
    analytics_rows = list(prepared.analytics_rows)
    points = list(prepared.points)
    steady_rows = list(prepared.steady_rows)
    steady_points = list(prepared.steady_points)
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
    analysis = (
        build_fitness_analytics(
            analytics_rows,
            window_days,
            target_hr,
            evaluation_time=as_of,
        )
        if analytics_rows
        else {"available": False}
    )

    current_pace = None
    uncertainty = None
    pace_change = None
    pace_change_uncertainty = None
    period_change = None
    within_window_trend = None
    trend = FitnessTrend.INSUFFICIENT_DATA
    confidence = ConfidenceLevel.UNAVAILABLE
    definition = (
        f"Estimated pace at {target_hr:g} bpm, minute {reference_minutes:g}, "
        "and reference conditions "
        f"over the last {window_days} days"
    )
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
        if change:
            prior = change["prior"]
            period_change = _change_evidence(
                {
                    **change,
                    "run_count": current["run_count"],
                    "coverage_fraction": min(
                        current["coverage_fraction"],
                        prior["coverage_fraction"],
                    ),
                },
                basis=(
                    str(change.get("comparison"))
                    if change.get("comparison")
                    else f"last {window_days} days versus the preceding "
                    f"{window_days} days"
                ),
                confidence=confidence,
                comparison_run_count=int(prior["run_count"]),
            )
        slope = analysis.get("within_window_trend")
        if slope:
            slope_confidence = (
                ConfidenceLevel.HIGH
                if slope["run_count"] >= 6
                and slope["coverage_fraction"] >= 0.5
                and analysis.get("days_since_latest_scored_run", float("inf"))
                <= 7
                else ConfidenceLevel.MODERATE
                if slope["run_count"] >= 3
                and slope["coverage_fraction"] >= 0.25
                and analysis.get("days_since_latest_scored_run", float("inf"))
                <= 21
                else ConfidenceLevel.LOW
            )
            within_window_trend = _change_evidence(
                slope,
                basis=str(
                    slope.get(
                        "basis",
                        f"weighted trajectory within the last {window_days} days",
                    )
                ),
                confidence=slope_confidence,
            )

    steady_analysis = (
        build_fitness_analytics(
            steady_rows,
            window_days,
            target_hr,
            evaluation_time=as_of,
        )
        if steady_rows
        else {"available": False}
    )
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
            f"A simple check of pace near {target_hr:g} bpm around minute "
            f"{reference_minutes:g}. It supports the main trend but does not replace it."
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
            "Training volume and aerobic efficiency can change independently."
            if pace_change is not None and load_change is not None
            else "There is not enough comparable pace or heart-rate data for a complete comparison."
        ),
    )

    loads = [rolling_load(sessions, as_of, days) for days in (7, 14, 28)]
    any_missing_load = any(
        item.zone_load is None
        for item in sessions
        if as_of - timedelta(days=28) < item.start_time <= as_of
    )
    capacity_settings = {
        "retention_half_life_days": float(
            (config or {}).get("coaching", {}).get("capacity_retention_half_life_days", 84)
        ),
        "retention_grace_days": int(
            (config or {}).get("coaching", {}).get("capacity_retention_grace_days", 28)
        ),
    }
    capacity = distance_capacity(sessions, as_of, **capacity_settings)
    continuous_half_life_days = float(
        (config or {}).get("coaching", {}).get(
            "continuous_fatigue_half_life_days", 7
        )
    )
    continuous_fatigue = continuous_fatigue_load(
        sessions,
        as_of,
        half_life_days=continuous_half_life_days,
    )
    # Capacity one window back, so the dashboard can say whether demonstrated
    # capacity moved rather than only whether this period's mileage did.
    previous_capacity = distance_capacity(
        sessions, as_of - timedelta(days=window_days), **capacity_settings
    )
    current_load = LoadContext(
        trailing_7d=_load_window(loads[0]),
        trailing_14d=_load_window(loads[1]),
        trailing_28d=_load_window(loads[2]),
        acute_to_prior_ratio=acute_to_prior_weekly_ratio(sessions, as_of),
        acute_distance_to_capacity_ratio=capacity.acute_to_capacity_ratio,
        continuous_fatigue_miles=continuous_fatigue.equivalent_weekly_miles,
        continuous_fatigue_to_capacity_ratio=(
            continuous_fatigue.equivalent_weekly_miles
            / capacity.reference_miles
            if capacity.reference_miles > 0
            else None
        ),
        continuous_distance_miles=continuous_distance_rate(
            sessions,
            as_of,
            half_life_days=continuous_half_life_days,
        ),
        continuous_short_term_distance_miles=continuous_distance_rate(
            sessions,
            as_of,
            half_life_days=short_term_density_half_life_days(
                continuous_half_life_days
            ),
        ),
        prior_28d_weekly_miles=capacity.prior_28d_weekly_miles,
        sustained_capacity_miles=capacity.sustained_weekly_miles,
        capacity_reference_miles=capacity.reference_miles,
        previous_capacity_reference_miles=previous_capacity.reference_miles,
        confidence=ConfidenceLevel.LOW if any_missing_load else ConfidenceLevel.HIGH,
        flags=["some_recent_sessions_missing_hr_load"] if any_missing_load else [],
    )

    trend_28d_points = [
        item for item in _trend_series(analytics_rows, 28) if item.as_of > chart_start
    ]
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
    easy_percent = easy / known * 100 if known else None
    moderate_percent = moderate / known * 100 if known else None
    hard_percent = hard / known * 100 if known else None
    verdict = assess_intensity_balance(
        easy_percent=easy_percent,
        moderate_percent=moderate_percent,
        hard_percent=hard_percent,
        known_minutes=known,
        missing_minutes=missing,
    )
    intensity = IntensitySummary(
        easy_percent=easy_percent,
        moderate_percent=moderate_percent,
        hard_percent=hard_percent,
        known_hr_minutes=known,
        missing_hr_minutes=missing,
        balance=verdict.balance,
        balance_headline=verdict.headline,
        balance_detail=verdict.detail,
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
        target_hr_bpm=target_hr,
        reference_within_run_minutes=reference_minutes,
        available_windows=list(AVAILABLE_WINDOWS),
        fitness_trend=trend,
        fitness_confidence=confidence,
        current_pace=current_pace,
        uncertainty_95_min_mile=uncertainty,
        pace_change_seconds_per_mile=pace_change,
        pace_change_uncertainty_95_seconds_per_mile=pace_change_uncertainty,
        period_change=period_change,
        within_window_trend=within_window_trend,
        definition=definition,
        series=points,
        trend_7d=[item for item in _trend_series(analytics_rows, 7) if item.as_of > chart_start],
        trend_28d=trend_28d_points,
        steady_aerobic=steady_summary,
        activity_coverage=_activity_coverage(
            connection,
            current_start,
            as_of,
            prepared.long_run_threshold_miles,
        ),
        quality_performance=_quality_performance(
            connection, current_start, as_of, config or {}
        ),
        period_comparison=comparison,
        current_load=current_load,
        consistency=consistency,
        intensity=intensity,
        local_vo2_estimate=estimate_local_vo2(
            current_pace=current_pace,
            current_pace_uncertainty_95=uncertainty,
            as_of=as_of,
            recent_load=current_load,
            fitness_trend=trend,
            config=config or {},
            series=vo2_series(trend_28d_points, config=config or {}, as_of=as_of),
        ),
        blind_spots=blind_spots,
    )
