"""Run-detail queries and deterministic, inspectable feedback rules."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
import json
import sqlite3

from .config import load_config, resolve_project_path
from .db import connect, initialize
from .cadence_feedback import build_cadence_analysis
from .movement import MovementInterval, attach_elevation_deltas, classify_movement
from .processing import _load_points
from .segmentation import METERS_PER_MILE
from .training_load import (
    TrainingSession,
    acute_to_prior_weekly_ratio,
    calculate_session_load,
    distance_capacity,
    rolling_load,
)
from .workout_scoring import analyze_workout
from .web.schemas import (
    ActivityHealthTag,
    AdjustmentContribution,
    ConfidenceLevel,
    DataQuality,
    DriftAssessment,
    FitnessObservation,
    LoadContext,
    LoadWindow,
    PaceValue,
    RunFeedback,
    RunMetadata,
    RunSummary,
    SessionDifficulty,
    Split,
    WeatherSnapshot,
    WorkoutAnalysis,
    WorkoutType,
    ZoneBreakdown,
)


def _pace_display(pace: float) -> str:
    minutes = int(pace)
    seconds = int(round((pace - minutes) * 60))
    if seconds == 60:
        minutes += 1
        seconds = 0
    return f"{minutes}:{seconds:02d}/mi"


def _enum_workout(value: str | None) -> WorkoutType:
    aliases = {
        "interval": WorkoutType.INTERVALS,
        "tempo": WorkoutType.TEMPO_THRESHOLD,
        "threshold": WorkoutType.TEMPO_THRESHOLD,
        "walk_run": WorkoutType.RUN_WALK,
        "run/walk": WorkoutType.RUN_WALK,
        "cycling": WorkoutType.BIKE,
    }
    normalized = str(value or "").strip().casefold().replace(" ", "_")
    if normalized in aliases:
        return aliases[normalized]
    try:
        return WorkoutType(normalized)
    except ValueError:
        return WorkoutType.UNKNOWN


def _health_tag(row: sqlite3.Row) -> ActivityHealthTag:
    raw = str(row["health_tag"] or "").strip().casefold() if "health_tag" in row.keys() else ""
    if raw:
        try:
            return ActivityHealthTag(raw)
        except ValueError:
            return ActivityHealthTag.OTHER_ABNORMAL
    return ActivityHealthTag.ILLNESS if row["illness"] else ActivityHealthTag.NORMAL


def _infer_workout_type(
    explicit: str | None,
    distance_miles: float,
    moving_minutes: float,
    zone_seconds: dict[str, float],
) -> WorkoutType:
    selected = _enum_workout(explicit)
    if selected != WorkoutType.UNKNOWN:
        return selected
    if distance_miles >= 7:
        return WorkoutType.LONG
    if moving_minutes > 0:
        return WorkoutType.EASY
    return WorkoutType.UNKNOWN


def _data_quality(row: sqlite3.Row, load_coverage: float) -> DataQuality:
    if row["calculated_moving_time_s"] is None:
        return DataQuality.UNAVAILABLE

    # A Garmin recording does not become analytically incomplete because one
    # occasional trackpoint lacks a coordinate or HR value.  Use the actual
    # coverage when trackpoints are available, while retaining the stored
    # labels as a fallback for synthetic/legacy rows.
    point_count = int(row["observed_trackpoint_count"] or 0)
    if point_count:
        sensor_coverages = (
            int(row["gps_valid_count"] or 0) / point_count,
            int(row["hr_valid_count"] or 0) / point_count,
        )
        if load_coverage < 0.5 or any(coverage < 0.5 for coverage in sensor_coverages):
            return DataQuality.POOR
        if load_coverage < 0.8 or any(coverage < 0.95 for coverage in sensor_coverages):
            return DataQuality.PARTIAL
        return DataQuality.GOOD

    qualities = [str(row[name] or "") for name in ("gps_quality", "hr_quality")]
    if load_coverage < 0.5 or any("missing" in value for value in qualities):
        return DataQuality.POOR
    if load_coverage < 0.8 or any("partial" in value for value in qualities):
        return DataQuality.PARTIAL
    return DataQuality.GOOD


def _split_stride_length(
    distance_m: float, moving_s: float, cadence_weighted: float, cadence_seconds: float
) -> float | None:
    """Metres per step across a split: speed divided by cadence."""
    if moving_s <= 0 or cadence_seconds <= 0:
        return None
    cadence_spm = cadence_weighted / cadence_seconds
    if cadence_spm <= 0:
        return None
    return (distance_m / moving_s) / (cadence_spm / 60.0)


def build_mile_splits(intervals: list[MovementInterval]) -> list[Split]:
    """Create exact mile boundaries by proportionally splitting raw intervals."""

    output: list[Split] = []
    distance = moving_s = elevation = hr_weighted = hr_seconds = 0.0
    cadence_weighted = cadence_seconds = 0.0
    index = 1
    for interval in intervals:
        if interval.distance_m <= 0 or interval.moving_time_s <= 0:
            continue
        remaining = interval.distance_m
        while remaining > 1e-9:
            needed = METERS_PER_MILE - distance
            consumed = min(remaining, needed)
            fraction = consumed / interval.distance_m
            apportioned_time = interval.moving_time_s * fraction
            distance += consumed
            moving_s += apportioned_time
            if interval.elevation_delta_m is not None:
                elevation += interval.elevation_delta_m * fraction
            hrs = [
                float(value)
                for value in (interval.start.heart_rate_bpm, interval.end.heart_rate_bpm)
                if value is not None
            ]
            if hrs:
                hr_weighted += (sum(hrs) / len(hrs)) * apportioned_time
                hr_seconds += apportioned_time
            # Total steps per minute via the one canonical conversion.
            cadences = [
                value
                for value in (interval.start.cadence_spm, interval.end.cadence_spm)
                if value is not None
            ]
            if cadences:
                cadence_weighted += (sum(cadences) / len(cadences)) * apportioned_time
                cadence_seconds += apportioned_time
            remaining -= consumed
            if distance >= METERS_PER_MILE - 1e-6:
                miles = distance / METERS_PER_MILE
                output.append(
                    Split(
                        index=index,
                        distance_miles=miles,
                        moving_minutes=moving_s / 60,
                        pace_min_mile=(moving_s / 60) / miles,
                        average_hr_bpm=hr_weighted / hr_seconds if hr_seconds else None,
                        average_cadence_spm=(
                            cadence_weighted / cadence_seconds if cadence_seconds else None
                        ),
                        stride_length_m=_split_stride_length(
                            distance, moving_s, cadence_weighted, cadence_seconds
                        ),
                        elevation_change_feet=elevation * 3.28084,
                    )
                )
                index += 1
                distance = moving_s = elevation = hr_weighted = hr_seconds = 0.0
                cadence_weighted = cadence_seconds = 0.0
    if distance >= METERS_PER_MILE * 0.05:
        miles = distance / METERS_PER_MILE
        output.append(
            Split(
                index=index,
                distance_miles=miles,
                moving_minutes=moving_s / 60,
                pace_min_mile=(moving_s / 60) / miles,
                average_hr_bpm=hr_weighted / hr_seconds if hr_seconds else None,
                average_cadence_spm=(
                    cadence_weighted / cadence_seconds if cadence_seconds else None
                ),
                stride_length_m=_split_stride_length(
                    distance, moving_s, cadence_weighted, cadence_seconds
                ),
                elevation_change_feet=elevation * 3.28084,
                is_partial=True,
            )
        )
    return output


def assess_cardiac_drift(
    intervals: list[MovementInterval], workout_type: WorkoutType
) -> DriftAssessment:
    moving = [interval for interval in intervals if interval.moving_time_s > 0 and interval.distance_m > 0]
    total_moving = sum(interval.moving_time_s for interval in moving)
    total_elapsed = sum(interval.elapsed_s for interval in intervals)
    stopped = sum(interval.stopped_time_s for interval in intervals)
    if total_moving < 30 * 60:
        return DriftAssessment(valid=False, confidence=ConfidenceLevel.UNAVAILABLE, reason="Run is shorter than 30 moving minutes.")
    if workout_type in {WorkoutType.INTERVALS, WorkoutType.TEMPO_THRESHOLD, WorkoutType.RACE, WorkoutType.RUN_WALK}:
        return DriftAssessment(valid=False, confidence=ConfidenceLevel.UNAVAILABLE, reason="Variable-intensity workouts are not valid steady-state drift tests.")
    if total_elapsed and stopped / total_elapsed > 0.10:
        return DriftAssessment(valid=False, confidence=ConfidenceLevel.LOW, reason="Stops exceed 10% of elapsed interval time.")

    halves = [dict(time=0.0, distance=0.0, hr_weighted=0.0, hr_seconds=0.0) for _ in range(2)]
    completed = 0.0
    for interval in moving:
        midpoint = completed + interval.moving_time_s / 2
        half = 0 if midpoint <= total_moving / 2 else 1
        hrs = [value for value in (interval.start.heart_rate_bpm, interval.end.heart_rate_bpm) if value]
        halves[half]["time"] += interval.moving_time_s
        halves[half]["distance"] += interval.distance_m
        if hrs:
            halves[half]["hr_weighted"] += sum(hrs) / len(hrs) * interval.moving_time_s
            halves[half]["hr_seconds"] += interval.moving_time_s
        completed += interval.moving_time_s
    if any(half["hr_seconds"] / max(half["time"], 1) < 0.8 for half in halves):
        return DriftAssessment(valid=False, confidence=ConfidenceLevel.LOW, reason="Heart-rate coverage is below 80% in one half.")
    efficiencies = []
    for half in halves:
        speed = half["distance"] / half["time"]
        hr = half["hr_weighted"] / half["hr_seconds"]
        efficiencies.append(speed / hr)
    decoupling = (efficiencies[0] - efficiencies[1]) / efficiencies[0] * 100
    return DriftAssessment(
        decoupling_percent=decoupling,
        valid=True,
        confidence=ConfidenceLevel.MODERATE,
        reason="Compares pace per heartbeat in the first and second halves; hills and planned pace changes can affect it.",
    )


def _fitness_observation(row: sqlite3.Row) -> FitnessObservation | None:
    if not row["result_json"]:
        return None
    result = json.loads(row["result_json"])
    raw = result.get("raw_pace_at_target_hr_min_mile")
    standardized = result.get("standardized_pace_at_target_hr_min_mile")
    if raw is None or standardized is None or row["start_time_utc"] is None:
        return None
    contributions: list[AdjustmentContribution] = []
    evidence = result.get("adjustment_evidence", {})
    for raw_name, value in result.get("contributions_min_mile", {}).items():
        evidence_name = (
            raw_name.removesuffix("_adjustment")
            .replace("hr_normalization", "heart_rate")
            .replace("time_normalization", "time")
        )
        item = evidence.get(evidence_name, {})
        confidence_raw = item.get("confidence", "unavailable")
        try:
            confidence = ConfidenceLevel(confidence_raw)
        except ValueError:
            confidence = ConfidenceLevel.UNAVAILABLE
        contributions.append(
            AdjustmentContribution(
                name=raw_name,
                minutes_per_mile=float(value),
                evidence=str(item.get("basis", "No evidence description available")),
                confidence=confidence,
                available=confidence != ConfidenceLevel.UNAVAILABLE,
            )
        )
    uncertainty = float(result.get("uncertainty_95_min_mile") or 0)
    confidence = ConfidenceLevel.MODERATE if uncertainty <= 0.5 else ConfidenceLevel.LOW
    return FitnessObservation(
        activity_id=int(row["id"]),
        activity_uid=str(row["activity_uid"]),
        start_time=datetime.fromisoformat(row["start_time_utc"]),
        raw_pace_at_target_hr=PaceValue(minutes_per_mile=float(raw), display=_pace_display(float(raw))),
        environmental_adjustment_min_mile=float(result.get("environmental_adjustment_min_mile") or 0),
        standardized_pace_at_target_hr=PaceValue(minutes_per_mile=float(standardized), display=_pace_display(float(standardized))),
        uncertainty_95_min_mile=uncertainty,
        comparable_window_minutes=None,
        contributions=contributions,
        confidence=confidence,
        included_in_trend=True,
    )


def _zone_breakdown(zone_seconds: dict[str, float], load) -> ZoneBreakdown:
    known = sum(float(zone_seconds.get(name, 0) or 0) for name in ("below_z1", "z1", "z2", "z3", "z4", "z5", "above_z5"))
    return ZoneBreakdown(
        zone_seconds={name: float(value or 0) for name, value in zone_seconds.items()},
        zone_fractions={
            name: float(value or 0) / known if known else 0.0
            for name, value in zone_seconds.items()
            if name != "unknown"
        },
        easy_minutes=load.easy_minutes,
        moderate_minutes=load.moderate_minutes,
        hard_minutes=load.hard_minutes,
    )


def _difficulty(row: sqlite3.Row, workout: WorkoutType, zone_seconds: dict[str, float]) -> SessionDifficulty:
    moving_s = float(row["calculated_moving_time_s"] or row["device_timer_time_s"] or 0)
    elapsed_s = float(row["elapsed_time_s"] or row["total_elapsed_time_s"] or moving_s)
    load = calculate_session_load(zone_seconds, moving_s)
    distance_miles = float(row["total_distance_m"] or row["analysis_distance_m"] or 0) / METERS_PER_MILE
    perceived_exertion = int(row["perceived_exertion"]) if row["perceived_exertion"] is not None else None
    elevation_gain_ft = float(row["derived_elevation_gain_m"] or 0) * 3.28084
    elevation_loss_ft = float(row["derived_elevation_loss_m"] or 0) * 3.28084
    flags: list[str] = []
    if load.zone_load is None:
        flags.append("intensity_load_unavailable_low_hr_coverage")
    if workout in {WorkoutType.LONG} or distance_miles >= 7:
        flags.append("long_duration_fatigue_possible")
    if load.hard_minutes >= 8:
        flags.append("high_intensity_response")
    if workout in {WorkoutType.INTERVALS, WorkoutType.TEMPO_THRESHOLD, WorkoutType.RACE}:
        flags.append("quality_session")
    if distance_miles > 0 and elevation_gain_ft / distance_miles >= 100:
        flags.append("hilly_session")
    if distance_miles > 0 and elevation_loss_ft / distance_miles >= 100:
        flags.append("substantial_downhill_load")
    return SessionDifficulty(
        distance_miles=distance_miles,
        moving_minutes=moving_s / 60,
        elapsed_minutes=elapsed_s / 60,
        stopped_minutes=max(0.0, elapsed_s - moving_s) / 60,
        zone_load=load.zone_load,
        perceived_exertion=perceived_exertion,
        session_rpe_load=(perceived_exertion * moving_s / 60 if perceived_exertion else None),
        elevation_gain_ft=elevation_gain_ft if elevation_gain_ft > 0 else None,
        elevation_loss_ft=elevation_loss_ft if elevation_loss_ft > 0 else None,
        zone_breakdown=_zone_breakdown(zone_seconds, load),
        is_long_run="long_duration_fatigue_possible" in flags,
        is_quality_session="quality_session" in flags,
        difficulty_flags=flags,
    )


def _load_window(value) -> LoadWindow:
    return LoadWindow(
        days=value.days,
        distance_miles=value.distance_miles,
        moving_minutes=value.moving_minutes,
        zone_load=value.zone_load,
        hard_minutes=value.hard_minutes,
        activity_count=value.activity_count,
    )


def _prior_load_context(connection: sqlite3.Connection, as_of: datetime) -> LoadContext:
    rows = connection.execute(
        """
        SELECT a.id,a.start_time_utc,a.total_distance_m,m.calculated_moving_time_s,
               m.device_timer_time_s,m.session_zone_load,m.hard_minutes,m.hr_zone_seconds_json,
               m.exclusion_reason,o.workout_type
        FROM activities a JOIN activity_metrics m ON m.activity_id=a.id
        LEFT JOIN run_overrides o ON o.activity_id=a.activity_id
        WHERE a.start_time_utc_epoch < ? ORDER BY a.start_time_utc_epoch
        """,
        (as_of.timestamp(),),
    ).fetchall()
    sessions: list[TrainingSession] = []
    incomplete = False
    for row in rows:
        exclusion = str(row["exclusion_reason"] or "")
        if _enum_workout(row["workout_type"]) in {WorkoutType.HIKE, WorkoutType.BIKE} or any(
            marker in exclusion
            for marker in ("probable_walk_or_hike_sensor_signature", "probable_bike_sensor_signature")
        ):
            continue
        moving_s = float(row["calculated_moving_time_s"] or row["device_timer_time_s"] or 0)
        zone_load = row["session_zone_load"]
        hard_minutes = row["hard_minutes"]
        if zone_load is None:
            calculated = calculate_session_load(json.loads(row["hr_zone_seconds_json"] or "{}"), moving_s)
            zone_load, hard_minutes = calculated.zone_load, calculated.hard_minutes
        incomplete = incomplete or zone_load is None
        sessions.append(
            TrainingSession(
                int(row["id"]),
                datetime.fromisoformat(row["start_time_utc"]),
                float(row["total_distance_m"] or 0) / METERS_PER_MILE,
                moving_s / 60,
                float(zone_load) if zone_load is not None else None,
                float(hard_minutes or 0),
            )
        )
    values = [rolling_load(sessions, as_of, days) for days in (7, 14, 28)]
    capacity = distance_capacity(sessions, as_of)
    return LoadContext(
        trailing_7d=_load_window(values[0]),
        trailing_14d=_load_window(values[1]),
        trailing_28d=_load_window(values[2]),
        acute_to_prior_ratio=acute_to_prior_weekly_ratio(sessions, as_of),
        acute_distance_to_capacity_ratio=capacity.acute_to_capacity_ratio,
        prior_28d_weekly_miles=capacity.prior_28d_weekly_miles,
        sustained_capacity_miles=capacity.sustained_weekly_miles,
        capacity_reference_miles=capacity.reference_miles,
        confidence=ConfidenceLevel.LOW if incomplete else ConfidenceLevel.HIGH,
        flags=["some_sessions_missing_hr_load"] if incomplete else [],
    )


def _feedback_text(
    summary: RunSummary,
    metadata: RunMetadata,
    drift: DriftAssessment,
    workout_analysis: WorkoutAnalysis | None = None,
) -> tuple[str, list[str], list[str]]:
    difficulty = summary.session_difficulty
    assert difficulty is not None
    positives: list[str] = []
    cautions: list[str] = []
    known_hr_minutes = (
        difficulty.zone_breakdown.easy_minutes
        + difficulty.zone_breakdown.moderate_minutes
        + difficulty.zone_breakdown.hard_minutes
    )
    easy_share = (
        difficulty.zone_breakdown.easy_minutes / known_hr_minutes
        if known_hr_minutes else None
    )
    health_phrase = {
        ActivityHealthTag.ILLNESS: " while ill",
        ActivityHealthTag.ILLNESS_RECOVERY: " while recovering from illness",
        ActivityHealthTag.INJURY_AFFECTED: " with an injury affecting the effort",
        ActivityHealthTag.OTHER_ABNORMAL: " with abnormal health context",
    }.get(metadata.health_tag, "")
    if metadata.workout_type == WorkoutType.HIKE:
        assessment = f"A hike{health_phrase}: useful time on your feet, kept separate from running fitness."
    elif metadata.workout_type == WorkoutType.BIKE:
        assessment = f"A bike ride{health_phrase}: useful aerobic work, kept separate from running pace."
    elif metadata.workout_type == WorkoutType.INTERVALS and workout_analysis and workout_analysis.interval_analysis:
        intervals = workout_analysis.interval_analysis
        if intervals.available:
            assessment = f"Completed {intervals.work_repetition_count} interval reps; {workout_analysis.control.summary[0].lower() + workout_analysis.control.summary[1:]}"
        else:
            assessment = "Interval workout, but the available data could not reconstruct the reps reliably."
    elif metadata.workout_type in {WorkoutType.TEMPO_THRESHOLD, WorkoutType.RACE}:
        quality_minutes = difficulty.zone_breakdown.moderate_minutes + difficulty.zone_breakdown.hard_minutes
        label = "race" if metadata.workout_type == WorkoutType.RACE else "tempo/threshold run"
        assessment = f"{label.title()}{health_phrase}, with {quality_minutes:.0f} minutes at moderate or hard effort."
    elif metadata.workout_type == WorkoutType.RUN_WALK and workout_analysis and workout_analysis.interval_analysis:
        intervals = workout_analysis.interval_analysis
        assessment = f"Run/walk{health_phrase} with {intervals.work_repetition_count} identifiable running segments."
    else:
        if easy_share is None:
            intensity_phrase = "an aerobic run with incomplete HR coverage"
        elif easy_share >= 0.85:
            intensity_phrase = "a controlled aerobic run"
        elif easy_share >= 0.70:
            intensity_phrase = "a mostly aerobic run"
        else:
            intensity_phrase = "a mixed-intensity run"
        session_label = "Long aerobic run" if difficulty.is_long_run else intensity_phrase.removeprefix("a ").capitalize()
        assessment = f"{session_label}{health_phrase}"
        if difficulty.stopped_minutes >= 5:
            assessment += ", with enough stopping to limit pace comparisons."
        elif drift.valid and drift.decoupling_percent is not None and drift.decoupling_percent > 5:
            assessment += ", but heart-rate cost rose in the second half."
        elif drift.valid and drift.decoupling_percent is not None and abs(drift.decoupling_percent) <= 3:
            assessment += " with steady effort through the second half."
        else:
            assessment += "."
    if difficulty.is_long_run:
        positives.append(f"{difficulty.distance_miles:.1f} miles of durability work")
    elif difficulty.is_quality_session:
        positives.append(f"{difficulty.zone_breakdown.moderate_minutes + difficulty.zone_breakdown.hard_minutes:.0f} minutes at moderate or hard effort")
    else:
        positives.append(f"{difficulty.moving_minutes:.0f} moving minutes completed")
    if metadata.health_tag != ActivityHealthTag.NORMAL:
        cautions.append(f"Health context: {metadata.health_tag.value.replace('_', ' ')}")
    if summary.data_quality in {DataQuality.POOR, DataQuality.UNAVAILABLE}:
        cautions.append("Some sensor data is missing, so the analysis is limited")
    if difficulty.stopped_minutes >= 5:
        cautions.append(f"{difficulty.stopped_minutes:.0f} stopped minutes make elapsed pace less comparable")
    if drift.valid and drift.decoupling_percent is not None and drift.decoupling_percent > 5:
        cautions.append("Heart rate rose relative to pace in the second half")
    if summary.fitness_observation is None:
        cautions.append("This run could not support a fair pace-at-HR comparison")
    return assessment, positives, cautions


RUN_SELECT = """
    SELECT a.*,m.*,
           o.workout_type,o.include_in_model,o.illness,o.notes AS override_notes,o.health_tag,o.perceived_exertion,
           lo.postal_code,lo.locality AS location_locality,lo.region AS location_region,
           w.temperature_f,w.dewpoint_f,w.apparent_temperature_f,w.relative_humidity_percent,
           w.wind_speed_mph,w.wind_gust_mph,w.headwind_mph,w.precipitation_in,w.weather_quality,
           (SELECT COUNT(*) FROM trackpoints tp WHERE tp.activity_id=a.id) AS observed_trackpoint_count,
           (SELECT COUNT(*) FROM trackpoints tp WHERE tp.activity_id=a.id AND tp.gps_valid=1) AS gps_valid_count,
           (SELECT COUNT(*) FROM trackpoints tp WHERE tp.activity_id=a.id AND tp.heart_rate_bpm IS NOT NULL) AS hr_valid_count,
           (SELECT SUM(s.elevation_gain_m) FROM segments s WHERE s.activity_id=a.id AND s.is_pathological=0) AS derived_elevation_gain_m,
           (SELECT SUM(s.elevation_loss_m) FROM segments s WHERE s.activity_id=a.id AND s.is_pathological=0) AS derived_elevation_loss_m,
           (SELECT result_json FROM model_runs mr WHERE mr.activity_id=a.id ORDER BY mr.id DESC LIMIT 1) AS result_json
    FROM activities a LEFT JOIN activity_metrics m ON m.activity_id=a.id
    LEFT JOIN run_overrides o ON o.activity_id=a.activity_id
    LEFT JOIN activity_location_overrides lo ON lo.activity_id=a.id
    LEFT JOIN activity_weather w ON w.activity_id=a.id
"""


def _row_summary(row: sqlite3.Row) -> RunSummary:
    zones = json.loads(row["hr_zone_seconds_json"] or "{}")
    moving_s = float(row["calculated_moving_time_s"] or row["device_timer_time_s"] or 0)
    load = calculate_session_load(zones, moving_s)
    miles = float(row["total_distance_m"] or row["analysis_distance_m"] or 0) / METERS_PER_MILE
    workout = _infer_workout_type(row["workout_type"], miles, moving_s / 60, zones)
    health = _health_tag(row)
    if workout == WorkoutType.HIKE:
        assessment_label = "Hike / time on feet"
    elif workout == WorkoutType.BIKE:
        assessment_label = "Cycling / cross-training"
    elif workout in {WorkoutType.INTERVALS, WorkoutType.TEMPO_THRESHOLD, WorkoutType.RACE}:
        assessment_label = "Intentional quality"
    elif sum(float(zones.get(name, 0) or 0) for name in ("z3", "z4", "z5", "above_z5")) >= moving_s * 0.25:
        assessment_label = "Moderate, not easy"
    else:
        assessment_label = "Aerobic run"
    return RunSummary(
        activity_id=int(row["id"]),
        activity_uid=str(row["activity_uid"]),
        start_time=datetime.fromisoformat(row["start_time_utc"]) if row["start_time_utc"] else None,
        distance_miles=miles,
        moving_minutes=moving_s / 60 if moving_s else None,
        moving_pace_min_mile=float(row["moving_pace_min_mile"]) if row["moving_pace_min_mile"] else None,
        average_hr_bpm=float(row["moving_average_hr_bpm"]) if row["moving_average_hr_bpm"] else None,
        maximum_hr_bpm=float(row["moving_maximum_hr_bpm"] or row["maximum_hr_bpm"]) if (row["moving_maximum_hr_bpm"] or row["maximum_hr_bpm"]) else None,
        temperature_f=float(row["temperature_f"]) if row["temperature_f"] is not None else None,
        gps_quality=str(row["gps_quality"]),
        model_included=bool(row["include_in_model"]) if row["include_in_model"] is not None else bool(row["model_eligible"]) if row["model_eligible"] is not None else None,
        assessment_label=assessment_label,
        workout_type=workout,
        health_tag=health,
        data_quality=_data_quality(row, load.hr_coverage),
        fitness_observation=_fitness_observation(row),
        session_difficulty=_difficulty(row, workout, zones),
    )


def list_runs(connection: sqlite3.Connection, limit: int = 100, offset: int = 0) -> list[RunSummary]:
    rows = connection.execute(
        RUN_SELECT + " ORDER BY a.start_time_utc_epoch DESC,a.id DESC LIMIT ? OFFSET ?",
        (limit, offset),
    ).fetchall()
    return [_row_summary(row) for row in rows]


def get_run_feedback(connection: sqlite3.Connection, config: dict[str, Any], activity_id: int) -> RunFeedback | None:
    row = connection.execute(RUN_SELECT + " WHERE a.id=?", (activity_id,)).fetchone()
    if row is None:
        return None
    summary = _row_summary(row)
    metadata = RunMetadata(
        workout_type=summary.workout_type,
        health_tag=summary.health_tag,
        include_in_model=bool(row["include_in_model"]) if row["include_in_model"] is not None else None,
        perceived_exertion=int(row["perceived_exertion"]) if row["perceived_exertion"] is not None else None,
        notes=str(row["override_notes"] or row["notes"] or ""),
        postal_code=str(row["postal_code"]) if row["postal_code"] else None,
        location_label=(
            ", ".join(
                str(value)
                for value in (row["location_locality"], row["location_region"])
                if value
            )
            or None
        ),
    )
    points = _load_points(connection, activity_id)
    movement = classify_movement(points, config["moving_time"])
    attach_elevation_deltas(points, movement.intervals, float(config["elevation"]["smoothing_window_meters"]))
    drift = assess_cardiac_drift(movement.intervals, metadata.workout_type)
    splits = build_mile_splits(movement.intervals)
    weather = None
    if row["weather_quality"]:
        weather = WeatherSnapshot(
            temperature_f=row["temperature_f"],
            dewpoint_f=row["dewpoint_f"],
            apparent_temperature_f=row["apparent_temperature_f"],
            relative_humidity_percent=row["relative_humidity_percent"],
            wind_speed_mph=row["wind_speed_mph"],
            wind_gust_mph=row["wind_gust_mph"],
            headwind_mph=row["headwind_mph"],
            precipitation_in=row["precipitation_in"],
            quality=row["weather_quality"],
        )
    start = datetime.fromisoformat(row["start_time_utc"]) if row["start_time_utc"] else datetime.now().astimezone()
    workout_analysis = analyze_workout(
        connection,
        config,
        activity_id,
        start,
        metadata.workout_type,
        summary.session_difficulty,
        drift,
        splits,
        movement.intervals,
    )
    assessment, positives, cautions = _feedback_text(
        summary, metadata, drift, workout_analysis
    )
    return RunFeedback(
        run=summary,
        metadata=metadata,
        splits=splits,
        cadence=build_cadence_analysis(connection, activity_id, movement.intervals),
        weather=weather,
        cardiac_drift=drift,
        workout_analysis=workout_analysis,
        load_context_before_run=_prior_load_context(connection, start - timedelta(microseconds=1)),
        assessment=assessment,
        positives=positives,
        cautions=cautions,
    )
