from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import gzip

import pytest

from fit_builder import build_run
from run_analysis.cadence import FIT_ONE_SIDED_CADENCE_SOURCE
from run_analysis.db import connect, initialize
from run_analysis.fit import TimerEvent, parse_fit, timer_pauses
from run_analysis.importer import import_files, parser_for
from run_analysis.tcx import parse_tcx

START = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)


@pytest.fixture()
def run_file(tmp_path: Path) -> Path:
    path = tmp_path / "run.fit"
    build_run(path, start=START)
    return path


def test_the_1989_epoch_is_converted_not_treated_as_unix_time(run_file: Path) -> None:
    """Read as Unix seconds these timestamps land in 2010, and every run would
    silently drop out of every trailing window."""
    activity = parse_fit(run_file).activities[0]
    assert activity.start_time_utc == START


def test_positions_are_converted_from_semicircles(run_file: Path) -> None:
    point = parse_fit(run_file).activities[0].trackpoints[0]
    assert point.latitude == pytest.approx(40.71, abs=1e-4)
    assert point.longitude == pytest.approx(-73.99, abs=1e-4)
    assert point.gps_valid


def test_altitude_is_descaled_to_metres(run_file: Path) -> None:
    """Stored as (metres + 500) * 5, so an unscaled read gives 2650 m."""
    assert parse_fit(run_file).activities[0].trackpoints[0].altitude_m == pytest.approx(30.0)


def test_one_sided_cadence_is_doubled_like_the_tcx_extension(run_file: Path) -> None:
    """The quiet catastrophe: without this every FIT run reads as a walk and is
    dropped from the model, with nothing raised anywhere."""
    point = parse_fit(run_file).activities[0].trackpoints[0]
    assert point.cadence == 82
    assert point.cadence_source == FIT_ONE_SIDED_CADENCE_SOURCE
    assert point.cadence_spm == 164.0


def test_the_sport_enum_becomes_the_string_the_rest_of_the_app_matches(tmp_path: Path) -> None:
    """race_goals and audit both query WHERE sport = 'Running' literally."""
    running = tmp_path / "run.fit"
    build_run(running, start=START, sport=1)
    assert parse_fit(running).activities[0].sport == "Running"
    cycling = tmp_path / "ride.fit"
    build_run(cycling, start=START, sport=2)
    assert parse_fit(cycling).activities[0].sport == "Biking"


def test_a_gzipped_export_reads_the_same(tmp_path: Path) -> None:
    """Strava returns whatever was uploaded, often still gzipped."""
    plain = tmp_path / "run.fit"
    payload = build_run(plain, start=START)
    zipped = tmp_path / "run.fit.gz"
    zipped.write_bytes(gzip.compress(payload))
    assert parse_fit(zipped).activities[0].start_time_utc == parse_fit(plain).activities[0].start_time_utc


def test_summary_fields_survive_a_file_with_no_lap_messages(tmp_path: Path) -> None:
    path = tmp_path / "nolap.fit"
    build_run(path, start=START, include_lap=False)
    activity = parse_fit(path).activities[0]
    assert activity.total_distance_m == pytest.approx(330.0)
    assert activity.average_hr_bpm == 150


def test_sensor_coverage_is_graded_the_same_way_as_tcx(run_file: Path) -> None:
    activity = parse_fit(run_file).activities[0]
    assert activity.gps_quality == "gps_complete"
    assert activity.hr_quality == "hr_complete"
    assert activity.cadence_quality == "cadence_complete"
    assert activity.distance_source == "device"


def test_timer_events_become_closed_pause_intervals() -> None:
    base = START
    events = [
        TimerEvent(base + timedelta(seconds=60), stopped=True),
        TimerEvent(base + timedelta(seconds=150), stopped=False),
    ]
    pauses = timer_pauses(events)
    assert pauses.stopped_seconds == 90.0


def test_an_unterminated_pause_is_dropped_rather_than_assumed() -> None:
    """Inventing an end for it would silently inflate stopped time."""
    pauses = timer_pauses([TimerEvent(START, stopped=True)])
    assert pauses.intervals == []
    assert pauses.stopped_seconds == 0.0


def test_repeated_stops_without_starts_do_not_nest() -> None:
    events = [
        TimerEvent(START + timedelta(seconds=10), stopped=True),
        TimerEvent(START + timedelta(seconds=20), stopped=True),
        TimerEvent(START + timedelta(seconds=40), stopped=False),
    ]
    assert timer_pauses(events).stopped_seconds == 30.0


@pytest.mark.parametrize(
    "name,expected",
    [("a.tcx", parse_tcx), ("a.fit", parse_fit), ("a.FIT", parse_fit), ("a.fit.gz", parse_fit), ("a.gpx", None)],
)
def test_the_parser_is_chosen_by_extension(name: str, expected) -> None:
    assert parser_for(Path(name)) is expected


def test_a_fit_file_imports_end_to_end(tmp_path: Path) -> None:
    build_run(tmp_path / "run.fit", start=START)
    database = tmp_path / "test.sqlite"
    with connect(database) as connection:
        initialize(connection)
        summary = import_files(connection, tmp_path, "America/New_York")
        assert summary.activities_added == 1
        assert summary.failed_files == 0
        row = connection.execute(
            "SELECT sport, cadence_quality, total_distance_m FROM activities"
        ).fetchone()
        assert row["sport"] == "Running"
        assert row["cadence_quality"] == "cadence_complete"
        stored = connection.execute(
            "SELECT cadence, cadence_source FROM trackpoints LIMIT 1"
        ).fetchone()
        # Raw one-sided value is preserved; the doubling stays a read-time
        # conversion so it can always be audited back to the device.
        assert stored["cadence"] == 82
        assert stored["cadence_source"] == FIT_ONE_SIDED_CADENCE_SOURCE


def test_the_same_run_in_both_formats_is_not_counted_twice(tmp_path: Path) -> None:
    """A watch FIT and a Strava TCX of one run disagree slightly on distance.
    Counted twice, they inflate mileage, load, and demonstrated capacity."""
    from test_tcx import make_tcx

    tcx_path = make_tcx(tmp_path / "run.tcx")
    activity = parse_tcx(tcx_path).activities[0]
    assert activity.start_time_utc is not None
    # Same run, as the watch recorded it: eight seconds earlier in file
    # creation time and 1.5% off on distance because Strava re-derived it.
    build_run(
        tmp_path / "run.fit",
        start=activity.start_time_utc + timedelta(seconds=8),
        points=3,
        seconds_per_point=10,
        metres_per_point=20.3,
    )
    database = tmp_path / "test.sqlite"
    with connect(database) as connection:
        initialize(connection)
        summary = import_files(connection, tmp_path, "America/New_York")
    assert summary.activities_added == 1
    assert summary.duplicate_activities == 1


def test_two_genuinely_different_runs_are_both_kept(tmp_path: Path) -> None:
    build_run(tmp_path / "morning.fit", start=START, points=12)
    build_run(tmp_path / "evening.fit", start=START + timedelta(hours=8), points=12)
    database = tmp_path / "test.sqlite"
    with connect(database) as connection:
        initialize(connection)
        summary = import_files(connection, tmp_path, "America/New_York")
    assert summary.activities_added == 2


def test_a_recorded_pause_is_trusted_over_a_stale_endpoint_speed(tmp_path: Path) -> None:
    """The device says the timer stopped, so no speed heuristic gets a vote."""
    from run_analysis.movement import classify_movement
    from run_analysis.processing import _load_points

    build_run(tmp_path / "run.fit", start=START, points=6, pauses=[(15, 105)])
    database = tmp_path / "test.sqlite"
    with connect(database) as connection:
        initialize(connection)
        import_files(connection, tmp_path, "America/New_York")
        activity_id = connection.execute("SELECT id FROM activities").fetchone()["id"]
        points = _load_points(connection, int(activity_id))

    assert any(point.pause_after_s for point in points), "pause never reached the database"
    settings = {
        "minimum_running_speed_mps": 0.8,
        "stopped_speed_mps": 0.35,
        "gps_stopped_speed_mps": 0.8,
        "stopped_distance_meters": 1.5,
        "maximum_interval_seconds": 30,
        "minimum_stop_seconds": 5,
        "maximum_plausible_speed_mps": 12,
    }
    result = classify_movement(points, settings)
    paused = [i for i in result.intervals if "recorded_timer_pause" in i.flags]
    assert paused, "the recorded pause was not honoured"
    assert all(interval.classification == "stopped" for interval in paused)
    assert all(interval.moving_time_s == 0.0 for interval in paused)


def test_tcx_runs_keep_inferring_stops_because_they_have_no_timer_events(tmp_path: Path) -> None:
    from test_tcx import make_tcx

    activity = parse_tcx(make_tcx(tmp_path / "run.tcx")).activities[0]
    assert all(point.pause_after_s is None for point in activity.trackpoints)
