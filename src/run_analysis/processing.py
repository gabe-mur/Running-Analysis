"""Phase 3 activity metrics and quarter-mile segment persistence."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import sqlite3

from .db import initialize, transaction
from .models import Trackpoint
from .movement import attach_elevation_deltas, classify_movement
from .segmentation import METERS_PER_MILE, Segment, build_segments
from .workload import update_workloads
from .training_load import calculate_session_load

PROCESSOR_VERSION = "phase3-v6-device-distance-fallback"


@dataclass(slots=True)
class ProcessingSummary:
    discovered_activities: int = 0
    processed_activities: int = 0
    unchanged_activities: int = 0
    segments_written: int = 0
    pathological_segments: int = 0
    model_eligible_activities: int = 0
    workloads_updated: int = 0


def _fingerprint(config: dict) -> str:
    eligibility_keys = (
        "minimum_run_miles",
        "maximum_stop_fraction",
        "minimum_hr_coverage",
        "minimum_gps_coverage",
    )
    relevant = {
        "version": PROCESSOR_VERSION,
        "moving_time": config["moving_time"],
        "elevation": config["elevation"],
        "segmentation": config["segmentation"],
        "segment_distance_miles": config["segment_distance_miles"],
        "zones": config["zones"],
        "eligibility": {key: config["model"][key] for key in eligibility_keys},
        "activity_classification": config["activity_classification"],
    }
    return hashlib.sha256(json.dumps(relevant, sort_keys=True).encode()).hexdigest()


def _load_points(connection: sqlite3.Connection, activity_id: int) -> list[Trackpoint]:
    rows = connection.execute(
        """
        SELECT lap_index, track_index, point_index, timestamp_utc, latitude,
               longitude, gps_valid, altitude_m, distance_m, heart_rate_bpm,
               cadence, run_cadence, cadence_source, speed_mps, parse_flags_json
        FROM trackpoints WHERE activity_id = ?
        ORDER BY lap_index, track_index, point_index
        """,
        (activity_id,),
    ).fetchall()
    return [
        Trackpoint(
            lap_index=int(row["lap_index"]),
            track_index=int(row["track_index"]),
            point_index=int(row["point_index"]),
            timestamp_utc=datetime.fromisoformat(row["timestamp_utc"]) if row["timestamp_utc"] else None,
            latitude=row["latitude"],
            longitude=row["longitude"],
            gps_valid=bool(row["gps_valid"]),
            altitude_m=row["altitude_m"],
            distance_m=row["distance_m"],
            heart_rate_bpm=row["heart_rate_bpm"],
            cadence=row["cadence"],
            run_cadence=row["run_cadence"],
            cadence_source=row["cadence_source"],
            speed_mps=row["speed_mps"],
            parse_flags=json.loads(row["parse_flags_json"]),
        )
        for row in rows
    ]


def _pace(time_s: float | None, distance_m: float | None) -> float | None:
    if time_s is None or distance_m is None or time_s <= 0 or distance_m <= 0:
        return None
    return (time_s / 60.0) / (distance_m / METERS_PER_MILE)


def _hr_zone_seconds(intervals, zones: dict[str, list[int]]) -> dict[str, float]:
    totals = {name: 0.0 for name in zones}
    totals["below_z1"] = 0.0
    totals["above_z5"] = 0.0
    totals["unknown"] = 0.0
    minimum = min(bounds[0] for bounds in zones.values())
    maximum = max(bounds[1] for bounds in zones.values())
    for interval in intervals:
        if interval.moving_time_s <= 0:
            continue
        values = [
            value
            for value in (interval.start.heart_rate_bpm, interval.end.heart_rate_bpm)
            if value is not None
        ]
        if not values:
            totals["unknown"] += interval.moving_time_s
            continue
        heart_rate = sum(values) / len(values)
        matched = False
        for name, bounds in zones.items():
            if bounds[0] <= heart_rate <= bounds[1]:
                totals[name] += interval.moving_time_s
                matched = True
                break
        if not matched:
            totals["below_z1" if heart_rate < minimum else "above_z5" if heart_rate > maximum else "unknown"] += interval.moving_time_s
    return {name: round(seconds, 3) for name, seconds in totals.items()}


def _moving_hr(intervals) -> tuple[float | None, int | None, float]:
    weighted = 0.0
    covered = 0.0
    maximum = None
    moving_total = sum(interval.moving_time_s for interval in intervals)
    for interval in intervals:
        if interval.moving_time_s <= 0:
            continue
        values = [
            value
            for value in (interval.start.heart_rate_bpm, interval.end.heart_rate_bpm)
            if value is not None
        ]
        if not values:
            continue
        average = sum(values) / len(values)
        weighted += average * interval.moving_time_s
        covered += interval.moving_time_s
        local_maximum = max(values)
        maximum = local_maximum if maximum is None else max(maximum, local_maximum)
    return weighted / covered if covered else None, maximum, covered / moving_total if moving_total else 0.0


def _eligibility(
    row: sqlite3.Row,
    points: list[Trackpoint],
    segments: list[Segment],
    moving_diagnostics: dict,
    config: dict,
    override: sqlite3.Row | None,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    thresholds = config["model"]
    total_points = len(points)
    gps_coverage = sum(point.gps_valid for point in points) / total_points if total_points else 0.0
    hr_coverage = sum(point.heart_rate_bpm is not None for point in points) / total_points if total_points else 0.0
    distance_m = row["total_distance_m"]
    distance_miles = distance_m / METERS_PER_MILE if distance_m else 0.0
    interval_elapsed = float(moving_diagnostics.get("moving_time_s", 0)) + float(
        moving_diagnostics.get("stopped_time_s", 0)
    )
    stop_fraction = (
        float(moving_diagnostics.get("stopped_time_s", 0)) / interval_elapsed if interval_elapsed else 0.0
    )
    analysis_distance = float(moving_diagnostics.get("analysis_distance_m", 0))
    distance_coverage = analysis_distance / distance_m if distance_m and distance_m > 0 else 0.0
    device_distance_coverage = float(moving_diagnostics.get("device_distance_coverage", 0.0))
    reliable_device_distance = (
        device_distance_coverage >= float(thresholds["minimum_gps_coverage"])
        and bool(distance_m)
        and 0.85 <= distance_coverage <= 1.15
    )
    if total_points < 2:
        reasons.append("inadequate_trackpoints")
    if gps_coverage < float(thresholds["minimum_gps_coverage"]) and not reliable_device_distance:
        reasons.append("inadequate_gps")
    if hr_coverage < float(thresholds["minimum_hr_coverage"]):
        reasons.append("inadequate_hr")
    if distance_miles < float(thresholds["minimum_run_miles"]):
        reasons.append("run_too_short")
    if stop_fraction > float(thresholds["maximum_stop_fraction"]):
        reasons.append("excessive_stop_fraction")
    if distance_m and not 0.85 <= distance_coverage <= 1.15:
        reasons.append("unreliable_trackpoint_distance")
    moving_pace = _pace(float(moving_diagnostics.get("moving_time_s", 0)), distance_m)
    cadence_values = [segment.average_cadence for segment in segments if segment.average_cadence is not None]
    mean_cadence = sum(cadence_values) / len(cadence_values) if cadence_values else None
    classification = config["activity_classification"]
    probable_walk = (
        moving_pace is not None
        and mean_cadence is not None
        and moving_pace >= float(classification["high_confidence_walk_pace_min_mile"])
        and mean_cadence <= float(classification["high_confidence_walk_cadence_max"])
    )
    probable_bike = (
        moving_pace is not None
        and distance_miles >= 5
        and moving_pace <= float(classification["high_confidence_bike_pace_min_mile"])
    )
    if probable_walk:
        reasons.append("probable_walk_or_hike_sensor_signature")
    if probable_bike:
        reasons.append("probable_bike_sensor_signature")
    usable_segments = sum(not segment.is_pathological for segment in segments)
    if usable_segments < 4:
        reasons.append("insufficient_valid_segments")
    if override:
        if override["include_in_model"] == 0:
            reasons.append("manual_exclusion")
        if override["workout_type"] and str(override["workout_type"]).casefold() in {
            "interval",
            "intervals",
            "race",
            "walk",
            "walk/jog",
            "hike",
            "hiking",
            "bike",
            "cycling",
        }:
            reasons.append(f"workout_type_{override['workout_type']}")
    # Explicit inclusion can override contextual/manual labels, but never the
    # hard data requirements that make weather/grade modeling impossible.
    hard = {
        "inadequate_trackpoints",
        "inadequate_gps",
        "inadequate_hr",
        "run_too_short",
        "unreliable_trackpoint_distance",
        "insufficient_valid_segments",
        "probable_walk_or_hike_sensor_signature",
        "probable_bike_sensor_signature",
    }
    if override and override["include_in_model"] == 1:
        reasons = [reason for reason in reasons if reason in hard]
    return not reasons, sorted(set(reasons))


def _insert_segments(connection: sqlite3.Connection, activity_id: int, segments: list[Segment]) -> None:
    connection.executemany(
        """
        INSERT INTO segments(
            activity_id, segment_index, metrics_json, start_time_utc, end_time_utc,
            distance_m, moving_time_s, elapsed_time_s, stopped_time_s,
            moving_pace_min_mile, average_hr_bpm, maximum_hr_bpm, average_cadence,
            elevation_gain_m, elevation_loss_m, net_elevation_change_m,
            average_grade_percent, distance_into_run_m, elapsed_minutes_into_run,
            moving_minutes_into_run, gps_complete_fraction, route_bearing_degrees,
            is_pathological, flags_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                activity_id,
                segment.segment_index,
                json.dumps(segment.diagnostics),
                segment.start_time_utc.isoformat() if segment.start_time_utc else None,
                segment.end_time_utc.isoformat() if segment.end_time_utc else None,
                segment.distance_m,
                segment.moving_time_s,
                segment.elapsed_time_s,
                segment.stopped_time_s,
                segment.moving_pace_min_mile,
                segment.average_hr_bpm,
                segment.maximum_hr_bpm,
                segment.average_cadence,
                segment.elevation_gain_m,
                segment.elevation_loss_m,
                segment.net_elevation_change_m,
                segment.average_grade_percent,
                segment.distance_into_run_m,
                segment.elapsed_minutes_into_run,
                segment.moving_minutes_into_run,
                segment.gps_complete_fraction,
                segment.route_bearing_degrees,
                int(segment.is_pathological),
                json.dumps(segment.flags),
            )
            for segment in segments
        ],
    )


def process_activities(
    connection: sqlite3.Connection, config: dict, force: bool = False
) -> ProcessingSummary:
    initialize(connection)
    fingerprint = _fingerprint(config)
    activities = connection.execute("SELECT * FROM activities ORDER BY start_time_utc_epoch, id").fetchall()
    summary = ProcessingSummary(discovered_activities=len(activities))
    for row in activities:
        override = connection.execute(
            "SELECT * FROM run_overrides WHERE activity_id = ?", (row["activity_id"],)
        ).fetchone()
        override_values = dict(override) if override else None
        activity_fingerprint = hashlib.sha256(
            f"{fingerprint}|{json.dumps(override_values, sort_keys=True)}".encode()
        ).hexdigest()
        current = connection.execute(
            "SELECT processing_fingerprint FROM activity_metrics WHERE activity_id = ?", (row["id"],)
        ).fetchone()
        if current and current["processing_fingerprint"] == activity_fingerprint and not force:
            summary.unchanged_activities += 1
            continue
        points = _load_points(connection, int(row["id"]))
        movement = classify_movement(points, config["moving_time"])
        analysis_distance_for_sources = sum(interval.distance_m for interval in movement.intervals)
        movement.diagnostics["device_distance_coverage"] = (
            sum(
                interval.distance_m
                for interval in movement.intervals
                if interval.distance_source == "device"
            )
            / analysis_distance_for_sources
            if analysis_distance_for_sources
            else 0.0
        )
        attach_elevation_deltas(
            points, movement.intervals, float(config["elevation"]["smoothing_window_meters"])
        )
        segments = build_segments(
            movement.intervals,
            float(config["segment_distance_miles"]),
            config["segmentation"],
            config["elevation"],
        )
        eligible, reasons = _eligibility(row, points, segments, movement.diagnostics, config, override)
        distance = row["total_distance_m"]
        elapsed = row["total_elapsed_time_s"]
        device_timer = row["lap_recorded_time_s"]
        moving = float(movement.diagnostics["moving_time_s"])
        stopped = float(movement.diagnostics["stopped_time_s"])
        very_slow = float(movement.diagnostics["very_slow_time_s"])
        analysis_distance = float(movement.diagnostics["analysis_distance_m"])
        interval_elapsed = moving + stopped
        diagnostics = dict(movement.diagnostics)
        diagnostics.update(
            {
                "elapsed_minus_device_timer_s": elapsed - device_timer
                if elapsed is not None and device_timer is not None
                else None,
                "calculated_minus_device_timer_s": moving - device_timer if device_timer is not None else None,
                "gps_trackpoint_coverage": sum(point.gps_valid for point in points) / len(points)
                if points
                else 0.0,
                "hr_trackpoint_coverage": sum(point.heart_rate_bpm is not None for point in points) / len(points)
                if points
                else 0.0,
                "interval_elapsed_s": interval_elapsed,
            }
        )
        zone_seconds = _hr_zone_seconds(movement.intervals, config["zones"])
        session_load = calculate_session_load(zone_seconds, moving)
        moving_average_hr, moving_maximum_hr, moving_hr_coverage = _moving_hr(movement.intervals)
        diagnostics["moving_hr_coverage"] = moving_hr_coverage
        if row["average_hr_bpm"] is not None and row["average_hr_bpm"] > config["max_hr"] + 10:
            diagnostics["summary_hr_warning"] = "implausible_lap_summary_hr_ignored_in_calculated_metrics"
        metrics_json = {
            "processor_version": PROCESSOR_VERSION,
            "quality": {
                "model_eligible": eligible,
                "exclusion_reasons": reasons,
            },
        }
        with transaction(connection):
            connection.execute("DELETE FROM segments WHERE activity_id = ?", (row["id"],))
            connection.execute("DELETE FROM activity_metrics WHERE activity_id = ?", (row["id"],))
            _insert_segments(connection, int(row["id"]), segments)
            connection.execute(
                """
                INSERT INTO activity_metrics(
                    activity_id, metrics_json, calculated_at_utc, processing_fingerprint,
                    elapsed_time_s, device_timer_time_s, calculated_moving_time_s,
                    stopped_time_s, very_slow_time_s, elapsed_pace_min_mile,
                    device_timer_pace_min_mile, moving_pace_min_mile, stop_fraction,
                    moving_average_hr_bpm, moving_maximum_hr_bpm,
                    analysis_distance_m, distance_coverage_fraction, segment_count,
                    pathological_segment_count, model_eligible, exclusion_reason,
                    hr_zone_seconds_json, diagnostics_json, session_zone_load,
                    easy_minutes, moderate_minutes, hard_minutes, hr_load_coverage
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["id"],
                    json.dumps(metrics_json),
                    datetime.now(timezone.utc).isoformat(),
                    activity_fingerprint,
                    elapsed,
                    device_timer,
                    moving,
                    stopped,
                    very_slow,
                    _pace(elapsed, distance),
                    _pace(device_timer, distance),
                    _pace(moving, distance),
                    stopped / interval_elapsed if interval_elapsed else 0.0,
                    moving_average_hr,
                    moving_maximum_hr,
                    analysis_distance,
                    analysis_distance / distance if distance and distance > 0 else 0.0,
                    len(segments),
                    sum(segment.is_pathological for segment in segments),
                    int(eligible),
                    ";".join(reasons) if reasons else None,
                    json.dumps(zone_seconds),
                    json.dumps(diagnostics),
                    session_load.zone_load,
                    session_load.easy_minutes,
                    session_load.moderate_minutes,
                    session_load.hard_minutes,
                    session_load.hr_coverage,
                ),
            )
        summary.processed_activities += 1
        summary.segments_written += len(segments)
        summary.pathological_segments += sum(segment.is_pathological for segment in segments)
        summary.model_eligible_activities += int(eligible)
    summary.workloads_updated = update_workloads(connection, config)
    summary.model_eligible_activities = int(
        connection.execute(
            "SELECT COUNT(*) FROM activity_metrics WHERE model_eligible=1"
        ).fetchone()[0]
    )
    return summary
