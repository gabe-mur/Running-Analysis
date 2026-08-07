"""Distance-based segments that preserve moving and stopped time separately."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from math import atan2, cos, degrees, radians, sin
from typing import Any

from .movement import MovementInterval
from .physiology import grade_energy_ratio

METERS_PER_MILE = 1609.344


@dataclass(slots=True)
class Segment:
    segment_index: int
    start_time_utc: datetime | None
    end_time_utc: datetime | None
    distance_m: float
    moving_time_s: float
    elapsed_time_s: float
    stopped_time_s: float
    moving_pace_min_mile: float | None
    average_hr_bpm: float | None
    maximum_hr_bpm: int | None
    average_cadence: float | None
    elevation_gain_m: float | None
    elevation_loss_m: float | None
    net_elevation_change_m: float | None
    average_grade_percent: float | None
    distance_into_run_m: float
    elapsed_minutes_into_run: float
    moving_minutes_into_run: float
    gps_complete_fraction: float
    route_bearing_degrees: float | None
    is_pathological: bool
    flags: list[str]
    diagnostics: dict[str, Any]


@dataclass(slots=True)
class _Builder:
    index: int
    start_time: datetime | None = None
    end_time: datetime | None = None
    distance: float = 0.0
    elapsed: float = 0.0
    moving: float = 0.0
    stopped: float = 0.0
    gps_distance: float = 0.0
    elevation_covered_distance: float = 0.0
    elevation_net: float = 0.0
    elevation_gain: float = 0.0
    elevation_loss: float = 0.0
    grade_window_distance: float = 0.0
    grade_window_elevation_delta: float = 0.0
    grade_cost_weighted_distance: float = 0.0
    grade_cost_distance: float = 0.0
    unstable_grade_distance: float = 0.0
    hr_weighted: float = 0.0
    hr_seconds: float = 0.0
    maximum_hr: int | None = None
    cadence_weighted: float = 0.0
    cadence_seconds: float = 0.0
    bearing_x: float = 0.0
    bearing_y: float = 0.0
    bearing_distance: float = 0.0
    classifications: dict[str, float] = field(default_factory=dict)
    flags: set[str] = field(default_factory=set)


def _interpolate_time(interval: MovementInterval, fraction: float) -> datetime | None:
    if interval.start.timestamp_utc is None:
        return None
    return interval.start.timestamp_utc + timedelta(seconds=interval.elapsed_s * fraction)


def _interpolated_average(first: int | None, second: int | None, start: float, end: float) -> float | None:
    if first is None and second is None:
        return None
    if first is None:
        return float(second)
    if second is None:
        return float(first)
    start_value = first + (second - first) * start
    end_value = first + (second - first) * end
    return (start_value + end_value) / 2.0


def _add_piece(
    builder: _Builder,
    interval: MovementInterval,
    start_fraction: float,
    end_fraction: float,
    distance: float,
    gain_deadband_m: float,
    grade_window_m: float,
    maximum_grade_percent: float,
) -> None:
    fraction = end_fraction - start_fraction
    elapsed = interval.elapsed_s * fraction
    moving = interval.moving_time_s * fraction
    stopped = interval.stopped_time_s * fraction
    if builder.start_time is None:
        builder.start_time = _interpolate_time(interval, start_fraction)
    builder.end_time = _interpolate_time(interval, end_fraction)
    builder.distance += distance
    builder.elapsed += elapsed
    builder.moving += moving
    builder.stopped += stopped
    builder.flags.update(interval.flags)
    builder.classifications[interval.classification] = (
        builder.classifications.get(interval.classification, 0.0) + elapsed
    )
    if interval.start.gps_valid and interval.end.gps_valid:
        builder.gps_distance += distance
    if interval.elevation_delta_m is not None:
        delta = interval.elevation_delta_m * fraction
        builder.elevation_covered_distance += distance
        builder.elevation_net += delta
        if interval.elevation_delta_m >= gain_deadband_m:
            builder.elevation_gain += max(0.0, delta)
        elif interval.elevation_delta_m <= -gain_deadband_m:
            builder.elevation_loss += max(0.0, -delta)
        if distance > 0:
            builder.grade_window_distance += distance
            builder.grade_window_elevation_delta += delta
            if builder.grade_window_distance >= grade_window_m:
                _flush_grade_window(builder, maximum_grade_percent)
    if moving > 0:
        hr = _interpolated_average(
            interval.start.heart_rate_bpm,
            interval.end.heart_rate_bpm,
            start_fraction,
            end_fraction,
        )
        if hr is not None:
            builder.hr_weighted += hr * moving
            builder.hr_seconds += moving
            candidates = [
                value
                for value in (interval.start.heart_rate_bpm, interval.end.heart_rate_bpm)
                if value is not None
            ]
            if candidates:
                local_max = max(candidates)
                builder.maximum_hr = local_max if builder.maximum_hr is None else max(builder.maximum_hr, local_max)
        cadence = _interpolated_average(
            interval.start.cadence,
            interval.end.cadence,
            start_fraction,
            end_fraction,
        )
        if cadence is not None:
            builder.cadence_weighted += cadence * moving
            builder.cadence_seconds += moving
    if interval.bearing_degrees is not None and distance > 0:
        angle = radians(interval.bearing_degrees)
        builder.bearing_x += sin(angle) * distance
        builder.bearing_y += cos(angle) * distance
        builder.bearing_distance += distance


def _flush_grade_window(builder: _Builder, maximum_grade_percent: float) -> None:
    if builder.grade_window_distance <= 0:
        return
    grade_percent = builder.grade_window_elevation_delta / builder.grade_window_distance * 100.0
    if abs(grade_percent) <= maximum_grade_percent:
        builder.grade_cost_weighted_distance += grade_energy_ratio(
            grade_percent, maximum_grade_percent
        ) * builder.grade_window_distance
        builder.grade_cost_distance += builder.grade_window_distance
    else:
        builder.unstable_grade_distance += builder.grade_window_distance
    builder.grade_window_distance = 0.0
    builder.grade_window_elevation_delta = 0.0


def _finish(
    builder: _Builder,
    *,
    distance_into_run: float,
    elapsed_into_run: float,
    moving_into_run: float,
    target_distance: float,
    final: bool,
    settings: dict[str, float | int],
    elevation_settings: dict[str, float | int],
) -> Segment:
    flags = set(builder.flags)
    minimum_final_fraction = float(settings["minimum_final_segment_fraction"])
    minimum_pace = float(settings["minimum_plausible_pace_min_mile"])
    maximum_pace = float(settings["maximum_plausible_pace_min_mile"])
    minimum_bearing_distance = float(settings["minimum_bearing_distance_meters"])
    minimum_grade_distance = float(elevation_settings["minimum_grade_distance_meters"])
    maximum_grade = float(elevation_settings.get("maximum_plausible_grade_percent", 12))
    grade_window_m = float(elevation_settings.get("grade_cost_window_meters", 60))
    if builder.grade_window_distance >= min(20.0, grade_window_m / 3.0):
        _flush_grade_window(builder, maximum_grade)
    pace = None
    if builder.distance > 0 and builder.moving > 0:
        pace = (builder.moving / 60.0) / (builder.distance / METERS_PER_MILE)
    if final and builder.distance < target_distance * minimum_final_fraction:
        flags.add("short_final_segment")
    if builder.moving <= 0:
        flags.add("no_moving_time")
    if pace is not None and (pace < minimum_pace or pace > maximum_pace):
        flags.add("implausible_pace")
    if "implausible_distance_speed" in flags:
        flags.add("distance_discontinuity")

    elevation_fraction = builder.elevation_covered_distance / builder.distance if builder.distance > 0 else 0.0
    elevation_available = elevation_fraction >= 0.8
    net = builder.elevation_net if elevation_available else None
    gain = builder.elevation_gain if elevation_available else None
    loss = builder.elevation_loss if elevation_available else None
    grade = None
    if elevation_available and builder.distance >= minimum_grade_distance:
        grade = builder.elevation_net / builder.distance * 100.0
        if abs(grade) > maximum_grade:
            flags.add("implausible_grade")
    bearing = None
    if builder.bearing_distance >= minimum_bearing_distance:
        bearing = (degrees(atan2(builder.bearing_x, builder.bearing_y)) + 360.0) % 360.0
    pathological_flags = {
        "short_final_segment",
        "no_moving_time",
        "implausible_pace",
        "distance_discontinuity",
        "implausible_grade",
    }
    grade_cost_coverage = builder.grade_cost_distance / builder.distance if builder.distance else 0.0
    integrated_grade_energy_ratio = (
        builder.grade_cost_weighted_distance / builder.grade_cost_distance
        if grade_cost_coverage >= 0.7 and builder.grade_cost_distance
        else None
    )
    return Segment(
        segment_index=builder.index,
        start_time_utc=builder.start_time,
        end_time_utc=builder.end_time,
        distance_m=builder.distance,
        moving_time_s=builder.moving,
        elapsed_time_s=builder.elapsed,
        stopped_time_s=builder.stopped,
        moving_pace_min_mile=pace,
        average_hr_bpm=builder.hr_weighted / builder.hr_seconds if builder.hr_seconds else None,
        maximum_hr_bpm=builder.maximum_hr,
        average_cadence=builder.cadence_weighted / builder.cadence_seconds if builder.cadence_seconds else None,
        elevation_gain_m=gain,
        elevation_loss_m=loss,
        net_elevation_change_m=net,
        average_grade_percent=grade,
        distance_into_run_m=distance_into_run,
        elapsed_minutes_into_run=elapsed_into_run / 60.0,
        moving_minutes_into_run=moving_into_run / 60.0,
        gps_complete_fraction=builder.gps_distance / builder.distance if builder.distance else 0.0,
        route_bearing_degrees=bearing,
        is_pathological=bool(flags & pathological_flags),
        flags=sorted(flags),
        diagnostics={
            "classification_seconds": builder.classifications,
            "hr_coverage_fraction": builder.hr_seconds / builder.moving if builder.moving else 0.0,
            "cadence_coverage_fraction": builder.cadence_seconds / builder.moving if builder.moving else 0.0,
            "elevation_coverage_fraction": elevation_fraction,
            "bearing_distance_m": builder.bearing_distance,
            "grade_cost_window_meters": grade_window_m,
            "grade_cost_coverage_fraction": grade_cost_coverage,
            "grade_energy_ratio": integrated_grade_energy_ratio,
            "unstable_micrograde_distance_m": builder.unstable_grade_distance,
        },
    )


def build_segments(
    intervals: list[MovementInterval],
    segment_distance_miles: float,
    settings: dict[str, float | int],
    elevation_settings: dict[str, float | int],
) -> list[Segment]:
    target = segment_distance_miles * METERS_PER_MILE
    gain_deadband = float(elevation_settings["minimum_gain_change_meters"])
    grade_window = float(elevation_settings.get("grade_cost_window_meters", 60))
    maximum_grade = float(elevation_settings.get("maximum_plausible_grade_percent", 12))
    segments: list[Segment] = []
    builder = _Builder(index=0)
    run_distance = 0.0
    run_elapsed = 0.0
    run_moving = 0.0

    for interval in intervals:
        if interval.distance_m <= 1e-9:
            _add_piece(
                builder, interval, 0.0, 1.0, 0.0, gain_deadband, grade_window, maximum_grade
            )
            run_elapsed += interval.elapsed_s
            run_moving += interval.moving_time_s
            continue
        consumed = 0.0
        while consumed < interval.distance_m - 1e-9:
            needed = target - builder.distance
            take = min(needed, interval.distance_m - consumed)
            start_fraction = consumed / interval.distance_m
            end_fraction = (consumed + take) / interval.distance_m
            _add_piece(
                builder,
                interval,
                start_fraction,
                end_fraction,
                take,
                gain_deadband,
                grade_window,
                maximum_grade,
            )
            elapsed_piece = interval.elapsed_s * (end_fraction - start_fraction)
            moving_piece = interval.moving_time_s * (end_fraction - start_fraction)
            run_distance += take
            run_elapsed += elapsed_piece
            run_moving += moving_piece
            consumed += take
            if builder.distance >= target - 1e-6:
                segments.append(
                    _finish(
                        builder,
                        distance_into_run=run_distance,
                        elapsed_into_run=run_elapsed,
                        moving_into_run=run_moving,
                        target_distance=target,
                        final=False,
                        settings=settings,
                        elevation_settings=elevation_settings,
                    )
                )
                builder = _Builder(index=len(segments))
    if builder.distance > 0 or builder.elapsed > 0:
        segments.append(
            _finish(
                builder,
                distance_into_run=run_distance,
                elapsed_into_run=run_elapsed,
                moving_into_run=run_moving,
                target_distance=target,
                final=True,
                settings=settings,
                elevation_settings=elevation_settings,
            )
        )
    return segments
