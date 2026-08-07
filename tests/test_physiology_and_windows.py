from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from run_analysis.model_windows import _WindowAccumulator
from run_analysis.models import Trackpoint
from run_analysis.movement import MovementInterval
from run_analysis.physiology import (
    estimated_shade_wbgt_f,
    grade_energy_ratio,
    minetti_running_cost_j_per_kg_m,
    relative_humidity_from_dewpoint,
    wet_bulb_temperature_f,
)


def config() -> dict:
    return yaml.safe_load((Path(__file__).parents[1] / "config.example.yaml").read_text())


def test_minetti_cost_matches_published_level_and_ten_percent_values() -> None:
    assert minetti_running_cost_j_per_kg_m(0) == pytest.approx(3.6)
    # These are evaluations of the paper's fitted fifth-order polynomial;
    # individual table means need not lie exactly on the fitted curve.
    assert minetti_running_cost_j_per_kg_m(10) == pytest.approx(5.968214)
    assert minetti_running_cost_j_per_kg_m(-10) == pytest.approx(2.151706)
    assert grade_energy_ratio(10) == pytest.approx(5.968214 / 3.6)


def test_wet_bulb_and_shade_wbgt_are_physically_ordered() -> None:
    assert relative_humidity_from_dewpoint(80, 80) == pytest.approx(100)
    wet_bulb = wet_bulb_temperature_f(80, 50)
    shade_wbgt = estimated_shade_wbgt_f(80, relative_humidity_percent=50)
    assert wet_bulb < shade_wbgt < 80


def _interval(index: int, start: datetime) -> MovementInterval:
    first = Trackpoint(
        lap_index=0,
        track_index=0,
        point_index=index,
        timestamp_utc=start,
        latitude=40.7,
        longitude=-73.9,
        gps_valid=True,
        altitude_m=10 + index,
        distance_m=index * 150,
        heart_rate_bpm=145,
        cadence=80,
        run_cadence=80,
        cadence_source="test",
        speed_mps=2.5,
    )
    second = Trackpoint(
        lap_index=0,
        track_index=0,
        point_index=index + 1,
        timestamp_utc=start + timedelta(seconds=60),
        latitude=40.701,
        longitude=-73.9,
        gps_valid=True,
        altitude_m=11 + index,
        distance_m=(index + 1) * 150,
        heart_rate_bpm=145,
        cadence=80,
        run_cadence=80,
        cadence_source="test",
        speed_mps=2.5,
    )
    return MovementInterval(
        index=index,
        start=first,
        end=second,
        elapsed_s=60,
        distance_m=150,
        distance_source="device",
        device_distance_m=150,
        gps_distance_m=150,
        computed_speed_mps=2.5,
        recorded_speed_mps=2.5,
        gps_speed_mps=2.5,
        moving_time_s=60,
        stopped_time_s=0,
        very_slow_time_s=0,
        classification="moving",
        bearing_degrees=0,
        elevation_delta_m=1,
    )


def test_five_minute_window_is_derived_from_raw_intervals() -> None:
    settings = config()
    start = datetime(2026, 7, 1, 12, tzinfo=timezone.utc)
    accumulator = _WindowAccumulator(60, 12)
    for index in range(5):
        accumulator.add(_interval(index, start + timedelta(seconds=index * 60)))
    hourly = {
        "time": [start.timestamp(), (start + timedelta(hours=1)).timestamp()],
        "temperature_2m": [70, 70],
        "relative_humidity_2m": [50, 50],
        "dew_point_2m": [50, 50],
        "apparent_temperature": [70, 70],
        "precipitation": [0, 0],
        "surface_pressure": [1010, 1010],
        "wind_speed_10m": [3, 3],
        "wind_direction_10m": [0, 0],
        "wind_gusts_10m": [5, 5],
    }
    activity = {
        "id": 1,
        "activity_id": "synthetic",
        "start_time_utc": start.isoformat(),
        "previous_7d_miles": 10,
        "previous_28d_miles": 40,
        "days_since_previous_run": 2,
        "days_since_previous_hard_run": 5,
        "run_moving_pace": 10,
        "moving_average_hr_bpm": 145,
    }
    row, reason = accumulator.finish(activity, hourly, moving_midpoint_s=450, config=settings)
    assert reason is None
    assert row is not None
    assert row["moving_pace_min_mile"] == pytest.approx(10.72896)
    assert row["average_hr_bpm"] == pytest.approx(145)
    assert row["grade_energy_ratio"] > 1


def test_missing_elevation_does_not_discard_hr_gps_window() -> None:
    settings = config()
    start = datetime(2026, 7, 1, 12, tzinfo=timezone.utc)
    accumulator = _WindowAccumulator(60, 12)
    for index in range(5):
        interval = _interval(index, start + timedelta(seconds=index * 60))
        interval.elevation_delta_m = None
        accumulator.add(interval)
    hourly = {
        "time": [start.timestamp(), (start + timedelta(hours=1)).timestamp()],
        "temperature_2m": [70, 70],
        "relative_humidity_2m": [50, 50],
        "dew_point_2m": [50, 50],
        "apparent_temperature": [70, 70],
        "precipitation": [0, 0],
        "surface_pressure": [1010, 1010],
        "wind_speed_10m": [3, 3],
        "wind_direction_10m": [0, 0],
        "wind_gusts_10m": [5, 5],
    }
    activity = {
        "id": 1,
        "activity_id": "synthetic-no-altitude",
        "start_time_utc": start.isoformat(),
        "previous_7d_miles": 10,
        "previous_28d_miles": 40,
        "days_since_previous_run": 2,
        "days_since_previous_hard_run": 5,
        "run_moving_pace": 10,
        "moving_average_hr_bpm": 145,
    }
    row, reason = accumulator.finish(activity, hourly, moving_midpoint_s=450, config=settings)
    assert reason is None
    assert row is not None
    assert row["grade_energy_ratio"] is None
    assert row["grade_complete_fraction"] == 0


def test_device_distance_scores_zero_gps_window_as_fallback() -> None:
    settings = config()
    start = datetime(2026, 7, 1, 12, tzinfo=timezone.utc)
    accumulator = _WindowAccumulator(60, 12)
    for index in range(5):
        interval = _interval(index, start + timedelta(seconds=index * 60))
        interval.start.gps_valid = False
        interval.end.gps_valid = False
        interval.start.latitude = interval.start.longitude = None
        interval.end.latitude = interval.end.longitude = None
        interval.gps_distance_m = None
        interval.bearing_degrees = None
        accumulator.add(interval)
    hourly = {
        "time": [start.timestamp(), (start + timedelta(hours=1)).timestamp()],
        "temperature_2m": [70, 70],
        "relative_humidity_2m": [50, 50],
        "dew_point_2m": [50, 50],
        "apparent_temperature": [70, 70],
        "precipitation": [0, 0],
        "surface_pressure": [1010, 1010],
        "wind_speed_10m": [3, 3],
        "wind_direction_10m": [0, 0],
        "wind_gusts_10m": [5, 5],
    }
    activity = {
        "id": 2,
        "activity_id": "synthetic-device-distance",
        "start_time_utc": start.isoformat(),
        "previous_7d_miles": 10,
        "previous_28d_miles": 40,
        "days_since_previous_run": 2,
        "days_since_previous_hard_run": 5,
        "run_moving_pace": 10,
        "moving_average_hr_bpm": 145,
        "weather_quality": "hourly_interpolated_estimated_location",
    }
    row, reason = accumulator.finish(activity, hourly, moving_midpoint_s=450, config=settings)
    assert reason is None
    assert row is not None
    assert row["gps_complete_fraction"] == 0
    assert row["device_distance_fraction"] == 1
    assert row["uses_device_distance_fallback"] is True
    assert row["weather_location_estimated"] is True


def test_ninety_percent_gps_is_not_a_reduced_weight_fallback() -> None:
    settings = config()
    start = datetime(2026, 7, 1, 12, tzinfo=timezone.utc)
    accumulator = _WindowAccumulator(60, 12)
    for index in range(10):
        interval = _interval(index, start + timedelta(seconds=index * 60))
        if index == 0:
            interval.start.gps_valid = False
            interval.end.gps_valid = False
        accumulator.add(interval)
    hourly = {
        "time": [start.timestamp(), (start + timedelta(hours=1)).timestamp()],
        "temperature_2m": [70, 70],
        "relative_humidity_2m": [50, 50],
        "dew_point_2m": [50, 50],
        "apparent_temperature": [70, 70],
        "precipitation": [0, 0],
        "surface_pressure": [1010, 1010],
        "wind_speed_10m": [3, 3],
        "wind_direction_10m": [0, 0],
        "wind_gusts_10m": [5, 5],
    }
    activity = {
        "id": 3,
        "activity_id": "synthetic-mostly-gps",
        "start_time_utc": start.isoformat(),
        "previous_7d_miles": 10,
        "previous_28d_miles": 40,
        "days_since_previous_run": 2,
        "days_since_previous_hard_run": 5,
        "run_moving_pace": 10,
        "moving_average_hr_bpm": 145,
        "weather_quality": "hourly_interpolated",
    }
    row, reason = accumulator.finish(activity, hourly, moving_midpoint_s=600, config=settings)
    assert reason is None
    assert row is not None
    assert row["gps_complete_fraction"] == pytest.approx(0.9)
    assert row["uses_device_distance_fallback"] is False
