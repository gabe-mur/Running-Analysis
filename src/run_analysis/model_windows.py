"""Build physiological modeling windows directly from raw trackpoint intervals."""

from __future__ import annotations

from datetime import datetime, timedelta
from math import atan2, cos, degrees, radians, sin
from typing import Any
import json
import sqlite3

from .movement import (
    MovementInterval,
    attach_elevation_deltas,
    classify_movement,
    pause_restart_offsets,
)
from .physiology import grade_energy_ratio
from .processing import _load_points
from .weather import interpolate_hourly, wind_components

METERS_PER_MILE = 1609.344


def _hourly_for_activity(connection: sqlite3.Connection, activity_id: int) -> dict:
    row = connection.execute(
        "SELECT derived_weather_json FROM activity_weather WHERE activity_id=?", (activity_id,)
    ).fetchone()
    if not row:
        return {}
    derived = json.loads(row["derived_weather_json"])
    cache_ids = [int(value) for value in derived.get("cache_ids", [])]
    if not cache_ids:
        return {}
    placeholders = ",".join("?" for _ in cache_ids)
    cache_rows = connection.execute(
        f"SELECT response_json FROM weather_cache WHERE id IN ({placeholders}) ORDER BY date_local",
        cache_ids,
    ).fetchall()
    combined: dict[str, list] = {}
    for cache_row in cache_rows:
        daily = json.loads(cache_row["response_json"])
        for key, values in daily.items():
            combined.setdefault(key, []).extend(values)
    if combined.get("time"):
        order = sorted(range(len(combined["time"])), key=lambda index: combined["time"][index])
        for key, values in combined.items():
            if len(values) == len(order):
                combined[key] = [values[index] for index in order]
    return combined


#: Fallbacks when a configuration predates post-pause suppression.
DEFAULT_POST_PAUSE_MINIMUM_STOP_SECONDS = 60.0
DEFAULT_POST_PAUSE_SUPPRESSION_MOVING_SECONDS = 180.0


def _overlaps_post_pause_recovery(
    moving_start_s: float,
    moving_end_s: float,
    pause_restarts: list[float] | None,
    settings: dict,
) -> bool:
    """Whether a window overlaps the heart-rate recovery after a long stop."""

    if not pause_restarts:
        return False
    suppression = float(
        settings.get(
            "post_pause_suppression_moving_seconds",
            DEFAULT_POST_PAUSE_SUPPRESSION_MOVING_SECONDS,
        )
    )
    if suppression <= 0:
        return False
    return any(
        moving_start_s < restart + suppression and moving_end_s > restart
        for restart in pause_restarts
    )


def _pause_restarts(intervals: list[MovementInterval], config: dict) -> list[float]:
    return pause_restart_offsets(
        intervals,
        float(
            config["model"].get(
                "post_pause_minimum_stop_seconds",
                DEFAULT_POST_PAUSE_MINIMUM_STOP_SECONDS,
            )
        ),
    )


class _WindowAccumulator:
    def __init__(self, grade_window_m: float, maximum_grade: float):
        self.grade_window_m = grade_window_m
        self.maximum_grade = maximum_grade
        self.intervals: list[MovementInterval] = []
        self.moving_s = 0.0
        self.elapsed_s = 0.0
        self.stopped_s = 0.0
        self.distance_m = 0.0
        self.gps_distance_m = 0.0
        self.device_distance_m = 0.0
        self.hr_weighted = 0.0
        self.hr_seconds = 0.0
        self.bearing_x = 0.0
        self.bearing_y = 0.0
        self.bearing_distance = 0.0
        self.elevation_distance = 0.0
        self.elevation_net = 0.0
        self.grade_micro_distance = 0.0
        self.grade_micro_delta = 0.0
        self.grade_cost_weighted = 0.0
        self.grade_cost_distance = 0.0
        self.unstable_grade_distance = 0.0

    def _flush_grade(self) -> None:
        if self.grade_micro_distance <= 0:
            return
        grade = self.grade_micro_delta / self.grade_micro_distance * 100.0
        if abs(grade) <= self.maximum_grade:
            self.grade_cost_weighted += (
                grade_energy_ratio(grade, self.maximum_grade) * self.grade_micro_distance
            )
            self.grade_cost_distance += self.grade_micro_distance
        else:
            self.unstable_grade_distance += self.grade_micro_distance
        self.grade_micro_distance = 0.0
        self.grade_micro_delta = 0.0

    def add(self, interval: MovementInterval) -> None:
        self.intervals.append(interval)
        self.moving_s += interval.moving_time_s
        self.elapsed_s += interval.elapsed_s
        self.stopped_s += interval.stopped_time_s
        self.distance_m += interval.distance_m
        if interval.start.gps_valid and interval.end.gps_valid:
            self.gps_distance_m += interval.distance_m
        if interval.distance_source == "device":
            self.device_distance_m += interval.distance_m
        if interval.moving_time_s > 0:
            heart_rates = [
                value
                for value in (interval.start.heart_rate_bpm, interval.end.heart_rate_bpm)
                if value is not None
            ]
            if heart_rates:
                average_hr = sum(heart_rates) / len(heart_rates)
                self.hr_weighted += average_hr * interval.moving_time_s
                self.hr_seconds += interval.moving_time_s
        if interval.bearing_degrees is not None and interval.distance_m > 0:
            angle = radians(interval.bearing_degrees)
            self.bearing_x += sin(angle) * interval.distance_m
            self.bearing_y += cos(angle) * interval.distance_m
            self.bearing_distance += interval.distance_m
        if interval.elevation_delta_m is not None and interval.distance_m > 0:
            self.elevation_distance += interval.distance_m
            self.elevation_net += interval.elevation_delta_m
            self.grade_micro_distance += interval.distance_m
            self.grade_micro_delta += interval.elevation_delta_m
            if self.grade_micro_distance >= self.grade_window_m:
                self._flush_grade()

    def finish(
        self,
        activity: sqlite3.Row,
        hourly: dict,
        moving_start_s: float,
        config: dict,
        pause_restarts: list[float] | None = None,
    ) -> tuple[dict[str, Any] | None, str | None]:
        if self.grade_micro_distance >= min(20.0, self.grade_window_m / 3.0):
            self._flush_grade()
        if not self.intervals or self.moving_s <= 0 or self.distance_m <= 0:
            return None, "empty_window"
        settings = config["model"]
        average_hr = self.hr_weighted / self.hr_seconds if self.hr_seconds else None
        # Preserve within-window dynamics for the stricter fixed-time benchmark.
        # A progressive or recovery transition can have an innocuous average HR
        # while still violating the steady-state assumption.
        half_moving_s = self.moving_s / 2.0
        completed_moving_s = 0.0
        half_distance = [0.0, 0.0]
        half_time = [0.0, 0.0]
        half_hr_weighted = [0.0, 0.0]
        half_hr_seconds = [0.0, 0.0]
        heart_rates: list[float] = []
        for interval in self.intervals:
            midpoint = completed_moving_s + interval.moving_time_s / 2.0
            half = 0 if midpoint <= half_moving_s else 1
            half_distance[half] += interval.distance_m
            half_time[half] += interval.moving_time_s
            values = [
                float(value)
                for value in (interval.start.heart_rate_bpm, interval.end.heart_rate_bpm)
                if value is not None
            ]
            heart_rates.extend(values)
            if values and interval.moving_time_s > 0:
                half_hr_weighted[half] += sum(values) / len(values) * interval.moving_time_s
                half_hr_seconds[half] += interval.moving_time_s
            completed_moving_s += interval.moving_time_s
        half_hr = [
            half_hr_weighted[index] / half_hr_seconds[index]
            if half_hr_seconds[index]
            else None
            for index in range(2)
        ]
        half_speed = [
            half_distance[index] / half_time[index] if half_time[index] else None
            for index in range(2)
        ]
        heart_rate_change = (
            half_hr[1] - half_hr[0]
            if half_hr[0] is not None and half_hr[1] is not None
            else None
        )
        speed_change_fraction = (
            half_speed[1] / half_speed[0] - 1.0
            if half_speed[0] and half_speed[1] is not None
            else None
        )
        stop_fraction = self.stopped_s / self.elapsed_s if self.elapsed_s else 0.0
        gps_fraction = self.gps_distance_m / self.distance_m
        device_fraction = self.device_distance_m / self.distance_m
        grade_coverage = self.grade_cost_distance / self.distance_m
        position_minutes = (moving_start_s + self.moving_s / 2.0) / 60.0
        reason = None
        if _overlaps_post_pause_recovery(
            moving_start_s, moving_start_s + self.moving_s, pause_restarts, settings
        ):
            # Heart rate has not caught back up to the effort, so this window
            # would understate the true cost of the pace.
            reason = "post_pause_hr_recovery"
        elif position_minutes < float(settings["minimum_reliable_segment_minutes"]):
            reason = "warmup_hr_lag"
        elif position_minutes > float(settings["maximum_reliable_segment_minutes"]):
            reason = "long_duration_drift"
        elif average_hr is None or not (
            float(settings["minimum_reliable_hr_bpm"])
            <= average_hr
            <= float(settings["maximum_reliable_hr_bpm"])
        ):
            reason = "outside_submaximal_hr_range"
        elif self.hr_seconds / self.moving_s < float(settings["minimum_hr_coverage"]):
            reason = "inadequate_window_hr"
        elif (
            gps_fraction < float(settings["minimum_gps_coverage"])
            and device_fraction < float(settings["minimum_gps_coverage"])
        ):
            reason = "inadequate_window_distance"
        elif stop_fraction > float(settings["maximum_reliable_segment_stop_fraction"]):
            reason = "stop_transition_contamination"
        if reason:
            return None, reason
        start = self.intervals[0].start.timestamp_utc
        end = self.intervals[-1].end.timestamp_utc
        if start is None or end is None:
            return None, "missing_window_time"
        moment = start + (end - start) / 2
        weather = interpolate_hourly(hourly, moment)
        if weather["temperature_f"] is None:
            return None, "missing_window_weather"
        bearing = None
        if self.bearing_distance >= float(config["segmentation"]["minimum_bearing_distance_meters"]):
            bearing = (degrees(atan2(self.bearing_x, self.bearing_y)) + 360.0) % 360.0
        headwind, tailwind, crosswind = wind_components(
            weather["wind_speed_mph"], weather["wind_direction_degrees"], bearing
        )
        pace = (self.moving_s / 60.0) / (self.distance_m / METERS_PER_MILE)
        average_grade = (
            self.elevation_net / self.elevation_distance * 100.0 if self.elevation_distance else 0.0
        )
        return (
            {
                "activity_id": int(activity["id"]),
                "external_activity_id": activity["activity_id"],
                "start_time_utc": activity["start_time_utc"],
                "window_start_time_utc": start.isoformat(),
                "window_end_time_utc": end.isoformat(),
                "moving_pace_min_mile": pace,
                "average_hr_bpm": average_hr,
                "heart_rate_range_bpm": max(heart_rates) - min(heart_rates) if heart_rates else None,
                "heart_rate_change_bpm": heart_rate_change,
                "speed_change_fraction": speed_change_fraction,
                "moving_minutes_into_run": position_minutes,
                "average_grade_percent": average_grade,
                # Elevation is optional for the primary HR/weather score.  A GPS
                # coordinate does not imply that the corresponding altitude is
                # present or trustworthy, so grade-aware analyses must filter on
                # this field instead of discarding the entire cardiovascular
                # window here.
                "grade_energy_ratio": (
                    self.grade_cost_weighted / self.grade_cost_distance
                    if grade_coverage >= 0.7
                    else None
                ),
                "grade_complete_fraction": grade_coverage,
                "gps_complete_fraction": gps_fraction,
                "device_distance_fraction": device_fraction,
                "uses_device_distance_fallback": (
                    gps_fraction < float(settings["minimum_gps_coverage"])
                    and device_fraction >= float(settings["minimum_gps_coverage"])
                ),
                "gps_sufficient_for_shared_parameters": (
                    gps_fraction >= float(settings["minimum_gps_coverage"])
                ),
                "weather_location_estimated": str(
                    activity["weather_quality"]
                    if "weather_quality" in activity.keys()
                    else ""
                ).endswith("estimated_location"),
                "stopped_time_s": self.stopped_s,
                "elapsed_time_s": self.elapsed_s,
                "distance_m": self.distance_m,
                "route_bearing_degrees": bearing,
                "temperature_f": weather["temperature_f"],
                "dewpoint_f": weather["dewpoint_f"],
                "relative_humidity_percent": weather["relative_humidity_percent"],
                "wind_speed_mph": weather["wind_speed_mph"],
                "headwind_signed_mph": (
                    float(headwind or 0.0) - float(tailwind or 0.0)
                    if headwind is not None or tailwind is not None
                    else None
                ),
                "crosswind_mph": crosswind,
                "previous_7d_miles": activity["previous_7d_miles"],
                "previous_28d_miles": activity["previous_28d_miles"],
                "days_since_previous_run": activity["days_since_previous_run"],
                "days_since_previous_hard_run": activity["days_since_previous_hard_run"],
                "run_moving_pace": activity["run_moving_pace"],
                "moving_average_hr_bpm": activity["moving_average_hr_bpm"],
                "health_tag": (
                    activity["health_tag"] if "health_tag" in activity.keys() else "normal"
                ),
                "workout_type": (
                    activity["workout_type"] if "workout_type" in activity.keys() else "unknown"
                ),
                "unstable_micrograde_distance_m": self.unstable_grade_distance,
            },
            None,
        )


def _windows_for_intervals(
    intervals: list[MovementInterval],
    activity: sqlite3.Row,
    hourly: dict,
    window_seconds: int,
    config: dict,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    elevation = config["elevation"]
    restarts = _pause_restarts(intervals, config)
    accumulator = _WindowAccumulator(
        float(elevation["grade_cost_window_meters"]),
        float(elevation["maximum_plausible_grade_percent"]),
    )
    output = []
    counters: dict[str, int] = {}
    completed_moving_s = 0.0
    for interval in intervals:
        accumulator.add(interval)
        if accumulator.moving_s >= window_seconds:
            row, reason = accumulator.finish(
                activity, hourly, completed_moving_s, config, restarts
            )
            if row:
                output.append(row)
            else:
                assert reason is not None
                counters[reason] = counters.get(reason, 0) + 1
            completed_moving_s += accumulator.moving_s
            accumulator = _WindowAccumulator(
                float(elevation["grade_cost_window_meters"]),
                float(elevation["maximum_plausible_grade_percent"]),
            )
    counters["retained"] = len(output)
    return output, counters


def _overlapping_windows_for_intervals(
    intervals: list[MovementInterval],
    activity: sqlite3.Row,
    hourly: dict,
    window_seconds: int,
    stride_seconds: int,
    config: dict,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Build moving-time windows with overlap while retaining intervening stops."""

    total_moving = sum(interval.moving_time_s for interval in intervals)
    output: list[dict[str, Any]] = []
    counters: dict[str, int] = {}
    start_target = 0.0
    elevation = config["elevation"]
    restarts = _pause_restarts(intervals, config)
    while start_target + window_seconds <= total_moving + 1e-9:
        accumulator = _WindowAccumulator(
            float(elevation["grade_cost_window_meters"]),
            float(elevation["maximum_plausible_grade_percent"]),
        )
        completed = 0.0
        started = False
        for interval in intervals:
            next_completed = completed + interval.moving_time_s
            if not started and next_completed <= start_target:
                completed = next_completed
                continue
            started = True
            accumulator.add(interval)
            completed = next_completed
            if accumulator.moving_s >= window_seconds:
                break
        row, reason = accumulator.finish(
            activity, hourly, start_target, config, restarts
        )
        if row:
            output.append(row)
        else:
            assert reason is not None
            counters[reason] = counters.get(reason, 0) + 1
        start_target += stride_seconds
    counters["retained"] = len(output)
    return output, counters


def load_model_window_sets(
    connection: sqlite3.Connection,
    config: dict,
    window_seconds: tuple[int, ...],
) -> tuple[dict[int, list[dict[str, Any]]], dict[int, dict[str, int]]]:
    activities = connection.execute(
        """
        SELECT a.id,a.activity_id,a.start_time_utc,m.previous_7d_miles,m.previous_28d_miles,
               m.days_since_previous_run,m.days_since_previous_hard_run,
               m.moving_pace_min_mile AS run_moving_pace,m.moving_average_hr_bpm,
               COALESCE(o.health_tag,'normal') AS health_tag,aw.weather_quality,
               COALESCE(o.workout_type,'unknown') AS workout_type
        FROM activities a JOIN activity_metrics m ON m.activity_id=a.id
        JOIN activity_weather aw ON aw.activity_id=a.id
        LEFT JOIN run_overrides o ON o.activity_id=a.activity_id
        WHERE m.model_eligible=1
        ORDER BY a.start_time_utc_epoch
        """
    ).fetchall()
    results = {seconds: [] for seconds in window_seconds}
    diagnostics = {seconds: {} for seconds in window_seconds}
    for activity in activities:
        points = _load_points(connection, int(activity["id"]))
        movement = classify_movement(points, config["moving_time"])
        attach_elevation_deltas(
            points, movement.intervals, float(config["elevation"]["smoothing_window_meters"])
        )
        hourly = _hourly_for_activity(connection, int(activity["id"]))
        for seconds in window_seconds:
            rows, counters = _windows_for_intervals(
                movement.intervals, activity, hourly, seconds, config
            )
            results[seconds].extend(rows)
            target = diagnostics[seconds]
            for name, count in counters.items():
                target[name] = target.get(name, 0) + count
    for seconds in window_seconds:
        diagnostics[seconds]["run_count"] = len(
            {row["activity_id"] for row in results[seconds]}
        )
    return results, diagnostics


def load_overlapping_model_windows(
    connection: sqlite3.Connection,
    config: dict,
    *,
    window_seconds: int,
    stride_seconds: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Load overlapping windows for the primary run-level prediction."""

    activities = connection.execute(
        """
        SELECT a.id,a.activity_id,a.start_time_utc,m.previous_7d_miles,m.previous_28d_miles,
               m.days_since_previous_run,m.days_since_previous_hard_run,
               m.moving_pace_min_mile AS run_moving_pace,m.moving_average_hr_bpm,
               COALESCE(o.health_tag,'normal') AS health_tag,aw.weather_quality,
               COALESCE(o.workout_type,'unknown') AS workout_type
        FROM activities a JOIN activity_metrics m ON m.activity_id=a.id
        JOIN activity_weather aw ON aw.activity_id=a.id
        LEFT JOIN run_overrides o ON o.activity_id=a.activity_id
        WHERE m.model_eligible=1
        ORDER BY a.start_time_utc_epoch
        """
    ).fetchall()
    rows: list[dict[str, Any]] = []
    diagnostics: dict[str, int] = {}
    for activity in activities:
        points = _load_points(connection, int(activity["id"]))
        movement = classify_movement(points, config["moving_time"])
        attach_elevation_deltas(
            points, movement.intervals, float(config["elevation"]["smoothing_window_meters"])
        )
        activity_rows, counters = _overlapping_windows_for_intervals(
            movement.intervals,
            activity,
            _hourly_for_activity(connection, int(activity["id"])),
            window_seconds,
            stride_seconds,
            config,
        )
        rows.extend(activity_rows)
        for key, value in counters.items():
            diagnostics[key] = diagnostics.get(key, 0) + value
    return rows, diagnostics
