from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from run_analysis.geo import initial_bearing_degrees
from run_analysis.models import Trackpoint
from run_analysis.movement import MovementInterval, classify_movement
from run_analysis.segmentation import METERS_PER_MILE, build_segments


MOVEMENT_SETTINGS = {
    "minimum_running_speed_mps": 0.8,
    "stopped_speed_mps": 0.35,
    "gps_stopped_speed_mps": 0.8,
    "stopped_distance_meters": 1.5,
    "maximum_interval_seconds": 30,
    "minimum_stop_seconds": 5,
    "maximum_plausible_speed_mps": 12,
}
SEGMENT_SETTINGS = {
    "minimum_final_segment_fraction": 0.5,
    "minimum_plausible_pace_min_mile": 3,
    "maximum_plausible_pace_min_mile": 30,
    "minimum_bearing_distance_meters": 20,
}
ELEVATION_SETTINGS = {
    "minimum_gain_change_meters": 0.5,
    "minimum_grade_distance_meters": 100,
}


def point(
    seconds: float,
    distance: float,
    speed: float,
    *,
    hr: int = 145,
    altitude: float = 5,
    index: int = 0,
) -> Trackpoint:
    return Trackpoint(
        lap_index=0,
        track_index=0,
        point_index=index,
        timestamp_utc=datetime(2024, 7, 1, tzinfo=timezone.utc) + timedelta(seconds=seconds),
        latitude=None,
        longitude=None,
        gps_valid=False,
        altitude_m=altitude,
        distance_m=distance,
        heart_rate_bpm=hr,
        cadence=80,
        run_cadence=80,
        cadence_source="run_cadence_extension",
        speed_mps=speed,
    )


def test_sustained_stop_removed_but_slow_jog_retained() -> None:
    points = [
        point(0, 0, 0, index=0),
        point(1, 0, 0, index=1),
        point(6, 0, 0, index=2),
        point(16, 6, 0.6, index=3),
    ]
    result = classify_movement(points, MOVEMENT_SETTINGS)
    assert result.intervals[0].classification == "stopped"
    assert result.intervals[1].classification == "stopped"
    assert result.intervals[2].classification == "moving"
    assert result.diagnostics["stopped_time_s"] == pytest.approx(6)
    assert result.diagnostics["moving_time_s"] == pytest.approx(10)
    assert result.diagnostics["very_slow_time_s"] == pytest.approx(10)


def test_large_gap_distinguishes_autopause_smart_recording_and_mixed_gap() -> None:
    autopause = classify_movement([point(0, 0, 0), point(60, 0, 0, index=1)], MOVEMENT_SETTINGS)
    smart = classify_movement([point(0, 0, 2), point(60, 120, 2, index=1)], MOVEMENT_SETTINGS)
    mixed = classify_movement([point(0, 0, 2), point(60, 30, 2, index=1)], MOVEMENT_SETTINGS)
    assert autopause.intervals[0].classification == "stopped"
    assert smart.intervals[0].classification == "moving"
    assert mixed.intervals[0].classification == "mixed_gap"
    assert mixed.intervals[0].moving_time_s == pytest.approx(15)
    assert mixed.intervals[0].stopped_time_s == pytest.approx(45)


def interval(
    index: int,
    first: Trackpoint,
    second: Trackpoint,
    distance: float,
    *,
    moving: float,
    stopped: float = 0,
    elevation_delta: float | None = None,
) -> MovementInterval:
    return MovementInterval(
        index=index,
        start=first,
        end=second,
        elapsed_s=moving + stopped,
        distance_m=distance,
        distance_source="device",
        device_distance_m=distance,
        gps_distance_m=None,
        computed_speed_mps=distance / (moving + stopped),
        recorded_speed_mps=distance / moving if moving else 0,
        gps_speed_mps=None,
        moving_time_s=moving,
        stopped_time_s=stopped,
        very_slow_time_s=0,
        classification="moving" if not stopped else "mixed_gap",
        bearing_degrees=None,
        elevation_delta_m=elevation_delta,
    )


def test_quarter_mile_segments_exclude_stop_time_from_pace() -> None:
    quarter = 0.25 * METERS_PER_MILE
    p0 = point(0, 0, 2, index=0)
    p1 = point(100, quarter / 2, 2, index=1)
    p2 = point(160, quarter / 2, 0, index=2)
    p3 = point(260, quarter, 2, index=3)
    intervals = [
        interval(0, p0, p1, quarter / 2, moving=100),
        interval(1, p1, p2, 0, moving=0, stopped=60),
        interval(2, p2, p3, quarter / 2, moving=100),
    ]
    segments = build_segments(intervals, 0.25, SEGMENT_SETTINGS, ELEVATION_SETTINGS)
    assert len(segments) == 1
    assert segments[0].distance_m == pytest.approx(quarter)
    assert segments[0].moving_time_s == pytest.approx(200)
    assert segments[0].elapsed_time_s == pytest.approx(260)
    assert segments[0].stopped_time_s == pytest.approx(60)
    assert segments[0].moving_pace_min_mile == pytest.approx(200 / 60 / 0.25)
    # The trackpoints record 80 via Garmin's one-sided RunCadence extension,
    # so the segment must store 160 total steps per minute.
    assert segments[0].average_cadence_spm == pytest.approx(160)


def test_segment_cadence_is_not_doubled_for_plain_tcx_cadence() -> None:
    quarter = 0.25 * METERS_PER_MILE
    p0 = point(0, 0, 2, index=0)
    p1 = point(100, quarter, 2, index=1)
    for candidate in (p0, p1):
        candidate.cadence_source = "cadence"
    segments = build_segments(
        [interval(0, p0, p1, quarter, moving=100)], 0.25, SEGMENT_SETTINGS, ELEVATION_SETTINGS
    )
    assert segments[0].average_cadence_spm == pytest.approx(80)


def test_segment_grade_uses_smoothed_interval_change() -> None:
    quarter = 0.25 * METERS_PER_MILE
    p0 = point(0, 0, 2, altitude=5)
    p1 = point(200, quarter, 2, altitude=9, index=1)
    segments = build_segments(
        [interval(0, p0, p1, quarter, moving=200, elevation_delta=4)],
        0.25,
        SEGMENT_SETTINGS,
        ELEVATION_SETTINGS,
    )
    assert segments[0].net_elevation_change_m == pytest.approx(4)
    assert segments[0].average_grade_percent == pytest.approx(4 / quarter * 100)


def test_bearing_is_clockwise_from_true_north() -> None:
    assert initial_bearing_degrees(40, -74, 41, -74) == pytest.approx(0)
    assert initial_bearing_degrees(40, -74, 40, -73) == pytest.approx(89.68, abs=0.1)

