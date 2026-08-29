from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlparse
import json

import pytest
import yaml

from run_analysis.db import connect, initialize
from run_analysis.forecast import (
    _active_nws_alerts,
    _alert_blocks_outdoor_run,
    _forecast_options_from_response,
    _forecast_response,
    _forecast_values,
    choose_planned_forecast,
    choose_planned_forecasts,
    get_planned_forecast,
)
from run_analysis.importer import import_files
from run_analysis.processing import process_activities
from run_analysis.geo import haversine_m
from run_analysis.weather import (
    HOURLY_VARIABLES,
    anonymize_coordinates,
    interpolate_hourly,
    save_activity_postal_code,
    update_weather,
    wind_components,
)
from test_tcx import make_tcx


def _historical_weather_config() -> dict:
    config = yaml.safe_load((Path(__file__).parents[1] / "config.example.yaml").read_text())
    config["weather"]["historical_enabled"] = True
    return config


def test_weather_interpolation_and_precipitation_semantics() -> None:
    hourly = {
        "time": [0, 3600],
        "temperature_2m": [50, 60],
        "relative_humidity_2m": [70, 50],
        "dew_point_2m": [40, 44],
        "apparent_temperature": [48, 59],
        "precipitation": [0.0, 0.2],
        "surface_pressure": [1000, 1002],
        "wind_speed_10m": [4, 8],
        "wind_direction_10m": [350, 10],
        "wind_gusts_10m": [6, 10],
    }
    values = interpolate_hourly(hourly, datetime.fromtimestamp(1800, timezone.utc))
    assert values["temperature_f"] == pytest.approx(55)
    assert values["dewpoint_f"] == pytest.approx(42)
    assert values["wind_direction_degrees"] == pytest.approx(0)
    assert values["precipitation_in"] == pytest.approx(0.2)


def test_planned_forecast_is_a_separate_opt_in(tmp_path: Path) -> None:
    called = False

    def downloader(_url: str, _timeout: float) -> dict:
        nonlocal called
        called = True
        return {}

    with connect(tmp_path / "forecast.sqlite") as connection:
        initialize(connection)
        result = get_planned_forecast(
            connection,
            {"weather": {"forecast_enabled": False}},
            tmp_path,
            datetime.now(timezone.utc),
            downloader=downloader,
        )
    assert result is None
    assert called is False


def test_forecast_carries_extreme_weather_detection_fields() -> None:
    moment = datetime(2026, 12, 10, 12, tzinfo=timezone.utc)
    values = _forecast_values(
        {
            "hourly": {
                "time": [int(moment.timestamp())],
                "temperature_2m": [20.0],
                "snowfall": [1.25],
                "visibility": [300.0],
                "weather_code": [75],
            }
        },
        moment,
    )

    assert values["snowfall_in"] == pytest.approx(1.25)
    assert values["visibility_miles"] == pytest.approx(300 / 1609.344)
    assert values["weather_code"] == 75


def test_planning_forecast_response_is_cached_by_location(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    def downloader(_url: str, _timeout: float) -> dict:
        nonlocal calls
        calls += 1
        return {"hourly": {"time": []}}

    monkeypatch.setattr(
        "run_analysis.forecast._recent_route_centroid", lambda *_args: (40.0, -74.0)
    )
    config = {
        "weather": {
            "forecast_enabled": True,
            "privacy_jitter_radius_km": 0,
            "privacy_salt_path": "weather_salt",
            "request_timeout_seconds": 1,
        }
    }
    with connect(tmp_path / "forecast-cache.sqlite") as connection:
        initialize(connection)
        first = _forecast_response(
            connection, config, tmp_path, downloader
        )
        second = _forecast_response(
            connection, config, tmp_path, downloader
        )

    assert first == second == {"hourly": {"time": []}}
    assert calls == 1


def test_nws_active_alerts_are_parsed_blocking_and_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = datetime.now(timezone.utc)
    calls = 0

    def downloader(url: str, _timeout: float) -> dict:
        nonlocal calls
        calls += 1
        assert "point=" in url
        return {
            "features": [
                {
                    "id": "https://api.weather.gov/alerts/test-warning",
                    "properties": {
                        "event": "Tornado Warning",
                        "headline": "Tornado Warning issued for the test area",
                        "severity": "Extreme",
                        "urgency": "Immediate",
                        "certainty": "Observed",
                        "onset": (now - timedelta(hours=1)).isoformat(),
                        "expires": (now + timedelta(hours=2)).isoformat(),
                    },
                },
                {
                    "id": "https://api.weather.gov/alerts/test-watch",
                    "properties": {
                        "event": "Tornado Watch",
                        "headline": "Tornado Watch issued for the test area",
                        "severity": "Severe",
                        "urgency": "Future",
                        "certainty": "Possible",
                        "onset": (now - timedelta(hours=1)).isoformat(),
                        "expires": (now + timedelta(hours=2)).isoformat(),
                    },
                },
            ]
        }

    monkeypatch.setattr(
        "run_analysis.forecast._recent_route_centroid", lambda *_args: (40.0, -74.0)
    )
    config = {
        "weather": {
            "emergency_alerts_enabled": True,
            "privacy_jitter_radius_km": 0,
            "privacy_salt_path": "weather_salt",
            "request_timeout_seconds": 1,
        }
    }
    with connect(tmp_path / "alerts.sqlite") as connection:
        initialize(connection)
        first, checked = _active_nws_alerts(
            connection, config, tmp_path, downloader=downloader
        )
        second, cached_checked = _active_nws_alerts(
            connection, config, tmp_path, downloader=downloader
        )

    assert checked is True and cached_checked is True
    assert calls == 1
    assert [item.blocks_outdoor_run for item in first] == [True, False]
    assert [item.alert_id for item in second] == [item.alert_id for item in first]


@pytest.mark.parametrize(
    "event",
    [
        "High Wind Warning",
        "Excessive Heat Warning",
        "Wind Chill Warning",
        "Flood Warning",
        "Coastal Flood Warning",
        "Lake Effect Snow Warning",
    ],
)
def test_common_dangerous_nws_warnings_block_outdoor_running(event: str) -> None:
    assert _alert_blocks_outdoor_run(
        event,
        "Severe",
        "Expected",
        "Likely",
    ) is True


def test_weekly_planner_selects_each_days_time_from_forecast_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    day = (datetime.now(timezone.utc) + timedelta(days=2)).date()
    candidates = [datetime.combine(day, datetime.min.time(), timezone.utc) + timedelta(hours=hour) for hour in (7, 12, 19)]
    response = {
        "hourly": {
            "time": [int(item.timestamp()) for item in candidates],
            "temperature_2m": [60.0, 80.0, 70.0],
            "relative_humidity_2m": [70.0, 50.0, 55.0],
            "dew_point_2m": [50.0, 60.0, 55.0],
            "apparent_temperature": [60.0, 82.0, 70.0],
            "precipitation_probability": [50.0, 0.0, 0.0],
            "precipitation": [0.1, 0.0, 0.0],
            "wind_speed_10m": [4.0, 8.0, 5.0],
            "wind_direction_10m": [0.0, 0.0, 0.0],
            "wind_gusts_10m": [6.0, 12.0, 8.0],
        }
    }
    monkeypatch.setattr("run_analysis.forecast._forecast_response", lambda *_args: response)
    with connect(tmp_path / "forecast.sqlite") as connection:
        initialize(connection)
        selected, weather = choose_planned_forecast(
            connection, {"weather": {"forecast_enabled": True}}, tmp_path, candidates
        )
    # Rain is neutral, so equally low-stress conditions keep the earliest slot.
    assert selected.hour == 7
    assert weather is not None and weather.temperature_f == pytest.approx(60)


def test_minor_weather_differences_do_not_push_ready_run_to_evening(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    day = (datetime.now(timezone.utc) + timedelta(days=2)).date()
    candidates = [
        datetime.combine(day, datetime.min.time(), timezone.utc)
        + timedelta(hours=hour)
        for hour in (7, 12, 19)
    ]
    response = {
        "hourly": {
            "time": [int(item.timestamp()) for item in candidates],
            "temperature_2m": [74.0, 70.0, 66.0],
            "relative_humidity_2m": [55.0, 55.0, 55.0],
            "dew_point_2m": [57.0, 53.0, 49.0],
            "apparent_temperature": [75.0, 71.0, 67.0],
            "precipitation_probability": [0.0, 0.0, 0.0],
            "precipitation": [0.0, 0.0, 0.0],
            "wind_speed_10m": [7.0, 5.0, 2.0],
            "wind_direction_10m": [0.0, 0.0, 0.0],
            "wind_gusts_10m": [10.0, 8.0, 4.0],
        }
    }
    monkeypatch.setattr(
        "run_analysis.forecast._forecast_response", lambda *_args: response
    )

    with connect(tmp_path / "minor-weather.sqlite") as connection:
        initialize(connection)
        selected, _ = choose_planned_forecast(
            connection,
            {"weather": {"forecast_enabled": True}},
            tmp_path,
            candidates,
        )

    assert selected.hour == 7


def test_same_day_evening_slot_remains_valid_after_its_clock_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixed_now = datetime(2026, 8, 25, 23, tzinfo=timezone.utc)

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed_now.astimezone(tz) if tz else fixed_now.replace(tzinfo=None)

    monkeypatch.setattr("run_analysis.forecast.datetime", FixedDateTime)
    evening = datetime(2026, 8, 25, 19, tzinfo=timezone.utc)

    options = _forecast_options_from_response([evening], None)

    assert options == [(evening, None)]


def test_multi_day_planner_reuses_one_forecast_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_day = (datetime.now(timezone.utc) + timedelta(days=2)).date()
    groups = [
        [
            datetime.combine(
                first_day + timedelta(days=offset),
                datetime.min.time(),
                timezone.utc,
            )
            + timedelta(hours=hour)
            for hour in (7, 12, 19)
        ]
        for offset in range(2)
    ]
    times = [item for group in groups for item in group]
    response = {
        "hourly": {
            "time": [int(item.timestamp()) for item in times],
            "temperature_2m": [65.0] * len(times),
            "relative_humidity_2m": [50.0] * len(times),
            "dew_point_2m": [50.0] * len(times),
            "apparent_temperature": [65.0] * len(times),
            "precipitation_probability": [0.0] * len(times),
            "precipitation": [0.0] * len(times),
            "wind_speed_10m": [4.0] * len(times),
            "wind_direction_10m": [0.0] * len(times),
            "wind_gusts_10m": [6.0] * len(times),
        }
    }
    calls = 0

    def fake_response(*_args) -> dict:
        nonlocal calls
        calls += 1
        return response

    monkeypatch.setattr("run_analysis.forecast._forecast_response", fake_response)
    with connect(tmp_path / "forecast.sqlite") as connection:
        initialize(connection)
        results = choose_planned_forecasts(
            connection,
            {"weather": {"forecast_enabled": True}},
            tmp_path,
            groups,
        )
    assert calls == 1
    assert len(results) == 2


def test_meteorological_wind_direction_components() -> None:
    # Wind FROM north is a headwind northbound and a tailwind southbound.
    assert wind_components(10, 0, 0) == pytest.approx((10, 0, 0))
    assert wind_components(10, 0, 180) == pytest.approx((0, 10, 0), abs=1e-10)
    assert wind_components(10, 0, 90) == pytest.approx((0, 0, 10), abs=1e-10)


def test_missing_bearing_keeps_raw_wind_but_not_components() -> None:
    assert wind_components(10, 270, None) == (None, None, None)


def test_coordinate_anonymization_is_stable_and_bounded() -> None:
    first = anonymize_coordinates(40.71, -73.95, 2.0, b"local-test-salt-value")
    second = anonymize_coordinates(40.71, -73.95, 2.0, b"local-test-salt-value")
    assert first == second
    assert first != (40.71, -73.95)
    assert haversine_m(40.71, -73.95, *first) <= 2010


def test_weather_cache_and_segment_persistence_without_network(tmp_path: Path) -> None:
    make_tcx(tmp_path / "run.tcx")
    config = _historical_weather_config()
    config["paths"]["weather_cache"] = "weather-cache"

    def fake_download(url: str, timeout: float) -> dict:
        query = parse_qs(urlparse(url).query)
        start = datetime.fromisoformat(query["start_date"][0]).replace(tzinfo=timezone.utc)
        end = datetime.fromisoformat(query["end_date"][0]).replace(tzinfo=timezone.utc) + timedelta(days=1)
        times = []
        cursor = start
        while cursor < end:
            times.append(int(cursor.timestamp()))
            cursor += timedelta(hours=1)
        hourly = {"time": times}
        values = {
            "temperature_2m": 75.0,
            "relative_humidity_2m": 60.0,
            "dew_point_2m": 58.0,
            "apparent_temperature": 76.0,
            "precipitation": 0.0,
            "surface_pressure": 1008.0,
            "wind_speed_10m": 8.0,
            "wind_direction_10m": 0.0,
            "wind_gusts_10m": 12.0,
        }
        for variable in HOURLY_VARIABLES:
            hourly[variable] = [values[variable]] * len(times)
        return {"hourly": hourly, "hourly_units": {name: "test" for name in HOURLY_VARIABLES}}

    with connect(tmp_path / "test.sqlite") as connection:
        import_files(connection, tmp_path, "America/New_York")
        process_activities(connection, config)
        first = update_weather(connection, config, tmp_path, downloader=fake_download)
        second = update_weather(connection, config, tmp_path, downloader=fake_download)
        assert first.api_requests == 1
        assert first.activities_updated == 1
        assert first.segments_updated == 1
        assert second.api_requests == 0
        assert second.cache_hits == 1
        activity_weather = connection.execute("SELECT * FROM activity_weather").fetchone()
        segment = connection.execute("SELECT * FROM segments").fetchone()
        assert activity_weather["temperature_f"] == pytest.approx(75)
        assert segment["dewpoint_f"] == pytest.approx(58)
        assert segment["weather_quality"] == "hourly_interpolated"


def test_no_gps_activity_uses_nearby_anonymized_run_location(tmp_path: Path) -> None:
    make_tcx(tmp_path / "gps.tcx")
    make_tcx(
        tmp_path / "no-gps.tcx",
        start="2024-07-02T12:00:00Z",
        end="2024-07-02T12:00:20Z",
        activity_id="2024-07-02T12:00:00Z",
        gps=False,
    )
    config = _historical_weather_config()
    config["paths"]["weather_cache"] = "weather-cache"
    config["weather"]["estimated_location_sources"] = {
        "2024-07-02T12:00:00Z": "2024-07-01T12:00:00Z"
    }

    def fake_download(url: str, timeout: float) -> dict:
        query = parse_qs(urlparse(url).query)
        start = datetime.fromisoformat(query["start_date"][0]).replace(tzinfo=timezone.utc)
        end = datetime.fromisoformat(query["end_date"][0]).replace(tzinfo=timezone.utc) + timedelta(days=1)
        times = []
        cursor = start
        while cursor < end:
            times.append(int(cursor.timestamp()))
            cursor += timedelta(hours=1)
        hourly = {"time": times}
        for variable in HOURLY_VARIABLES:
            hourly[variable] = [0.0 if variable == "precipitation" else 60.0] * len(times)
        return {"hourly": hourly, "hourly_units": {}}

    with connect(tmp_path / "test.sqlite") as connection:
        import_files(connection, tmp_path, "America/New_York")
        process_activities(connection, config)
        summary = update_weather(connection, config, tmp_path, downloader=fake_download)
        no_gps = connection.execute(
            """SELECT aw.weather_quality,aw.derived_weather_json
               FROM activities a JOIN activity_weather aw ON aw.activity_id=a.id
               WHERE a.gps_quality='gps_missing'"""
        ).fetchone()
    assert summary.estimated_location_activities == 1
    assert summary.skipped_no_gps == 0
    assert no_gps["weather_quality"] == "hourly_interpolated_estimated_location"
    assert json.loads(no_gps["derived_weather_json"])["location_basis"] == "nearest_run_centroid_estimate"


def test_postal_code_location_is_persisted_for_no_gps_activity(tmp_path: Path) -> None:
    make_tcx(tmp_path / "no-gps.tcx", gps=False)
    config = _historical_weather_config()
    observed_url = ""

    def fake_geocode(url: str, timeout: float) -> dict:
        nonlocal observed_url
        observed_url = url
        return {
            "results": [
                {
                    "name": "Chicago",
                    "admin1": "New York",
                    "country_code": "US",
                    "postcodes": ["60601"],
                    "latitude": 40.71,
                    "longitude": -73.95,
                }
            ]
        }

    with connect(tmp_path / "test.sqlite") as connection:
        import_files(connection, tmp_path, "America/New_York")
        activity_id = int(connection.execute("SELECT id FROM activities").fetchone()[0])
        resolved = save_activity_postal_code(
            connection,
            activity_id,
            "60601",
            config,
            downloader=fake_geocode,
        )
        stored = connection.execute(
            "SELECT * FROM activity_location_overrides WHERE activity_id=?", (activity_id,)
        ).fetchone()
    assert "name=60601" in observed_url
    assert resolved["locality"] == "Chicago"
    assert stored["postal_code"] == "60601"
    assert stored["latitude"] == pytest.approx(40.71)
