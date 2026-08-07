"""Interval-level moving/stopped classification with explicit diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from statistics import median

from .geo import haversine_m, initial_bearing_degrees
from .models import Trackpoint


@dataclass(slots=True)
class MovementInterval:
    index: int
    start: Trackpoint
    end: Trackpoint
    elapsed_s: float
    distance_m: float
    distance_source: str
    device_distance_m: float | None
    gps_distance_m: float | None
    computed_speed_mps: float | None
    recorded_speed_mps: float | None
    gps_speed_mps: float | None
    moving_time_s: float
    stopped_time_s: float
    very_slow_time_s: float
    classification: str
    bearing_degrees: float | None
    elevation_delta_m: float | None = None
    flags: list[str] = field(default_factory=list)


@dataclass(slots=True)
class MovementResult:
    intervals: list[MovementInterval]
    diagnostics: dict[str, float | int]


def _valid_gps_distance(first: Trackpoint, second: Trackpoint) -> float | None:
    if not first.gps_valid or not second.gps_valid:
        return None
    assert first.latitude is not None and first.longitude is not None
    assert second.latitude is not None and second.longitude is not None
    return haversine_m(first.latitude, first.longitude, second.latitude, second.longitude)


def _recorded_speed(first: Trackpoint, second: Trackpoint) -> float | None:
    values = [value for value in (first.speed_mps, second.speed_mps) if value is not None and value >= 0]
    return median(values) if values else None


def classify_movement(
    points: list[Trackpoint], settings: dict[str, float | int]
) -> MovementResult:
    minimum_running = float(settings["minimum_running_speed_mps"])
    stopped_speed = float(settings["stopped_speed_mps"])
    gps_stopped_speed = float(settings.get("gps_stopped_speed_mps", minimum_running))
    stopped_distance = float(settings["stopped_distance_meters"])
    maximum_interval = float(settings["maximum_interval_seconds"])
    minimum_stop = float(settings["minimum_stop_seconds"])
    maximum_plausible_speed = float(settings.get("maximum_plausible_speed_mps", 12))

    intervals: list[MovementInterval] = []
    low_candidates: list[bool] = []
    diagnostics: dict[str, float | int] = {
        "invalid_time_intervals": 0,
        "distance_resets": 0,
        "distance_spikes": 0,
        "large_intervals": 0,
        "mixed_gap_intervals": 0,
        "uncertain_intervals": 0,
    }

    for index, (first, second) in enumerate(zip(points, points[1:])):
        flags: list[str] = []
        if first.timestamp_utc is None or second.timestamp_utc is None:
            diagnostics["invalid_time_intervals"] += 1
            continue
        elapsed = (second.timestamp_utc - first.timestamp_utc).total_seconds()
        if elapsed <= 0:
            diagnostics["invalid_time_intervals"] += 1
            continue

        device_distance = None
        if first.distance_m is not None and second.distance_m is not None:
            device_distance = second.distance_m - first.distance_m
            if device_distance < -1.0:
                diagnostics["distance_resets"] += 1
                flags.append("distance_reset")
                device_distance = None
            elif device_distance < 0:
                flags.append("minor_negative_distance_clamped")
                device_distance = 0.0
        gps_distance = _valid_gps_distance(first, second)
        if device_distance is not None:
            distance = device_distance
            distance_source = "device"
        elif gps_distance is not None:
            distance = gps_distance
            distance_source = "gps"
        else:
            distance = 0.0
            distance_source = "unknown"
            flags.append("distance_unavailable")

        computed_speed = distance / elapsed
        gps_speed = gps_distance / elapsed if gps_distance is not None else None
        recorded_speed = _recorded_speed(first, second)
        if computed_speed > maximum_plausible_speed:
            diagnostics["distance_spikes"] += 1
            flags.append("implausible_distance_speed")
        if elapsed > maximum_interval:
            diagnostics["large_intervals"] += 1
            flags.append("large_recording_gap")

        distance_low = distance <= max(stopped_distance, stopped_speed * elapsed)
        recorded_low = recorded_speed is None or recorded_speed <= stopped_speed
        gps_low = gps_speed is None or gps_speed <= gps_stopped_speed
        # Garmin can carry the final pre-pause speed into the point preceding a
        # long auto-pause gap. Unchanged device distance plus low GPS velocity
        # is stronger evidence than that stale endpoint speed.
        stationary_device_gap = (
            elapsed > maximum_interval
            and device_distance is not None
            and device_distance <= stopped_distance
            and gps_low
        )
        if stationary_device_gap and not recorded_low:
            flags.append("stale_endpoint_speed_ignored_for_stationary_gap")
        low_candidates.append(stationary_device_gap or (distance_low and recorded_low and gps_low))
        bearing = None
        if gps_distance is not None and gps_distance > 0.5:
            assert first.latitude is not None and first.longitude is not None
            assert second.latitude is not None and second.longitude is not None
            bearing = initial_bearing_degrees(
                first.latitude, first.longitude, second.latitude, second.longitude
            )
        intervals.append(
            MovementInterval(
                index=index,
                start=first,
                end=second,
                elapsed_s=elapsed,
                distance_m=distance,
                distance_source=distance_source,
                device_distance_m=device_distance,
                gps_distance_m=gps_distance,
                computed_speed_mps=computed_speed,
                recorded_speed_mps=recorded_speed,
                gps_speed_mps=gps_speed,
                moving_time_s=elapsed,
                stopped_time_s=0.0,
                very_slow_time_s=0.0,
                classification="moving",
                bearing_degrees=bearing,
                flags=flags,
            )
        )

    # A low-speed sample is only a stop when it belongs to a sustained low
    # sequence. This intentionally leaves isolated slow jogging as moving.
    run_start = 0
    while run_start < len(intervals):
        if not low_candidates[run_start]:
            run_start += 1
            continue
        run_end = run_start + 1
        while run_end < len(intervals) and low_candidates[run_end]:
            # Nonconsecutive source indexes indicate a malformed time interval
            # was skipped and should break stop persistence.
            if intervals[run_end].index != intervals[run_end - 1].index + 1:
                break
            run_end += 1
        duration = sum(interval.elapsed_s for interval in intervals[run_start:run_end])
        if duration >= minimum_stop:
            for interval in intervals[run_start:run_end]:
                interval.classification = "stopped"
                interval.moving_time_s = 0.0
                interval.stopped_time_s = interval.elapsed_s
                interval.flags.append("sustained_stop_evidence")
        run_start = run_end

    for interval in intervals:
        if interval.classification == "stopped":
            continue
        slow = interval.computed_speed_mps is not None and interval.computed_speed_mps < minimum_running
        if interval.elapsed_s > maximum_interval and slow:
            evidence = [
                speed
                for speed in (interval.recorded_speed_mps, interval.gps_speed_mps)
                if speed is not None and speed >= minimum_running
            ]
            if evidence and interval.distance_m > stopped_distance:
                representative_speed = max(minimum_running, median(evidence))
                estimated_moving = min(interval.elapsed_s, interval.distance_m / representative_speed)
                interval.moving_time_s = estimated_moving
                interval.stopped_time_s = interval.elapsed_s - estimated_moving
                interval.classification = "mixed_gap"
                interval.flags.append("mixed_gap_moving_time_estimated")
                diagnostics["mixed_gap_intervals"] += 1
            else:
                interval.classification = "uncertain_moving"
                interval.flags.append("slow_large_gap_kept_as_moving")
                diagnostics["uncertain_intervals"] += 1
        if slow:
            interval.very_slow_time_s = interval.moving_time_s

    diagnostics.update(
        {
            "interval_count": len(intervals),
            "moving_time_s": sum(interval.moving_time_s for interval in intervals),
            "stopped_time_s": sum(interval.stopped_time_s for interval in intervals),
            "very_slow_time_s": sum(interval.very_slow_time_s for interval in intervals),
            "analysis_distance_m": sum(interval.distance_m for interval in intervals),
            "stopped_intervals": sum(interval.classification == "stopped" for interval in intervals),
        }
    )
    return MovementResult(intervals, diagnostics)


def smooth_elevation(
    points: list[Trackpoint], intervals: list[MovementInterval], window_meters: float
) -> list[float | None]:
    if not points:
        return []
    cumulative = [0.0]
    for interval in intervals:
        cumulative.append(cumulative[-1] + interval.distance_m)
    # Skipped malformed intervals can make this shorter than points. Fall back
    # to raw altitude rather than pretending indexes still align.
    if len(cumulative) != len(points):
        return [point.altitude_m for point in points]
    half_window = window_meters / 2.0
    smoothed: list[float | None] = []
    left = 0
    right = 0
    for index, position in enumerate(cumulative):
        while left < len(points) and cumulative[left] < position - half_window:
            left += 1
        while right + 1 < len(points) and cumulative[right + 1] <= position + half_window:
            right += 1
        values = [points[item].altitude_m for item in range(left, right + 1) if points[item].altitude_m is not None]
        smoothed.append(median(values) if values else None)
    return smoothed


def attach_elevation_deltas(
    points: list[Trackpoint], intervals: list[MovementInterval], window_meters: float
) -> None:
    smoothed = smooth_elevation(points, intervals, window_meters)
    if len(smoothed) != len(points):
        return
    for interval_index, interval in enumerate(intervals):
        # In the usual case interval.index equals its first point index. If an
        # invalid interval was skipped, preserve missing elevation diagnostics.
        first_index = interval.index
        second_index = first_index + 1
        if second_index >= len(smoothed):
            continue
        first = smoothed[first_index]
        second = smoothed[second_index]
        if first is not None and second is not None:
            interval.elevation_delta_m = second - first
