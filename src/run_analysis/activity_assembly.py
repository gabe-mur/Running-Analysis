"""Derived activity fields, shared by every input format.

Sensor-coverage grades, summary heart rates, distance provenance, and local
time are computed here rather than in each parser, so a run means the same
thing whether it arrived as TCX or FIT. A second copy of this logic would
drift, and the drift would show up as two formats disagreeing about the same
run.
"""

from __future__ import annotations

from statistics import fmean
from zoneinfo import ZoneInfo

from .models import Activity
from .timeparse import parse_datetime


COMPLETE_SENSOR_COVERAGE = 0.95


def quality(prefix: str, present: int, total: int) -> str:
    """Describe material sensor coverage, not literal point-for-point perfection."""
    if total == 0 or present == 0:
        return f"{prefix}_missing"
    if present / total >= COMPLETE_SENSOR_COVERAGE:
        return f"{prefix}_complete"
    return f"{prefix}_partial"



def finish_activity(activity: Activity, default_timezone: str) -> None:
    points = activity.trackpoints
    timestamps = [point.timestamp_utc for point in points if point.timestamp_utc is not None]
    if timestamps:
        activity.start_time_utc = min(timestamps)
        if len(timestamps) > 1:
            activity.total_elapsed_time_s = (max(timestamps) - min(timestamps)).total_seconds()
    elif any(lap.start_time_utc for lap in activity.laps):
        activity.start_time_utc = min(lap.start_time_utc for lap in activity.laps if lap.start_time_utc)
        activity.parse_warnings.append("start_time_from_lap_no_trackpoint_time")
    else:
        activity.start_time_utc = parse_datetime(activity.activity_id, activity.parse_warnings, "activity_id")
        if activity.start_time_utc:
            activity.parse_warnings.append("start_time_from_activity_id")

    lap_times = [lap.total_time_s for lap in activity.laps if lap.total_time_s is not None]
    lap_distances = [lap.distance_m for lap in activity.laps if lap.distance_m is not None]
    activity.lap_recorded_time_s = sum(lap_times) if lap_times else None
    activity.total_distance_m = sum(lap_distances) if lap_distances else None
    lap_calories = [lap.calories for lap in activity.laps if lap.calories is not None]
    activity.calories = sum(lap_calories) if lap_calories else None
    weighted_hr = [
        (lap.average_hr_bpm, lap.total_time_s)
        for lap in activity.laps
        if lap.average_hr_bpm is not None and lap.total_time_s is not None and lap.total_time_s > 0
    ]
    if weighted_hr:
        activity.average_hr_bpm = sum(hr * seconds for hr, seconds in weighted_hr) / sum(
            seconds for _, seconds in weighted_hr
        )
    else:
        summary_values = [lap.average_hr_bpm for lap in activity.laps if lap.average_hr_bpm is not None]
        activity.average_hr_bpm = fmean(summary_values) if summary_values else None
    maxima = [lap.maximum_hr_bpm for lap in activity.laps if lap.maximum_hr_bpm is not None]
    activity.maximum_hr_bpm = max(maxima) if maxima else None

    count = len(points)
    activity.gps_quality = quality("gps", sum(point.gps_valid for point in points), count)
    activity.hr_quality = quality("hr", sum(point.heart_rate_bpm is not None for point in points), count)
    activity.elevation_quality = quality("elevation", sum(point.altitude_m is not None for point in points), count)
    activity.cadence_quality = quality("cadence", sum(point.cadence is not None for point in points), count)
    device_distance_present = bool(lap_distances) or any(point.distance_m is not None for point in points)
    activity.distance_source = "device" if device_distance_present else "unknown"

    valid_positions = [(point.latitude, point.longitude) for point in points if point.gps_valid]
    timezone_source = "configured_default"
    if valid_positions:
        mean_lat = fmean(position[0] for position in valid_positions if position[0] is not None)
        mean_lon = fmean(position[1] for position in valid_positions if position[1] is not None)
        if 40.45 <= mean_lat <= 41.0 and -74.30 <= mean_lon <= -73.65:
            timezone_source = "gps_nyc"
        else:
            activity.parse_warnings.append("timezone_default_used_outside_nyc")
    else:
        activity.parse_warnings.append("timezone_default_used_without_gps")
    activity.timezone_name = default_timezone
    activity.timezone_source = timezone_source
    if activity.start_time_utc:
        activity.start_time_local = activity.start_time_utc.astimezone(ZoneInfo(default_timezone))
    activity.parse_warnings = sorted(set(activity.parse_warnings))
