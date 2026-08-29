"""Privacy-preserving forecast lookup for a planned run.

The forecast location is not a home address or an exact trackpoint.  It is the
rounded centroid of the most recent GPS activity, displaced with the same
locally salted deterministic jitter used by historical weather retrieval.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import json
import sqlite3

from .environmental_stress import assess_training_weather
from .weather import (
    _download_json,
    anonymize_coordinates,
    interpolate_hourly,
    load_or_create_privacy_salt,
)
from .web.schemas import ConfidenceLevel, PlannedWeather, WeatherEmergencyAlert


FORECAST_HOURLY_VARIABLES = (
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "apparent_temperature",
    "precipitation_probability",
    "precipitation",
    "snowfall",
    "visibility",
    "weather_code",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
)

NWS_ALERT_CACHE_KEY = "nws_active_weather_alerts"
NWS_ALERT_CACHE_SECONDS = 60
NWS_ALERT_STALE_FALLBACK_SECONDS = 3600
FORECAST_CACHE_KEY = "open_meteo_planning_forecast"
FORECAST_CACHE_SECONDS = 600
FORECAST_STALE_FALLBACK_SECONDS = 3600


def _download_nws_json(url: str, timeout: float) -> dict:
    request = Request(
        url,
        headers={
            "User-Agent": "RunningAnalysis/1.0 (local personal training app)",
            "Accept": "application/geo+json",
        },
    )
    with urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed configured API
        return json.loads(response.read().decode("utf-8"))


def _alert_datetime(value) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _alert_blocks_outdoor_run(
    event: str,
    severity: str,
    urgency: str,
    certainty: str,
) -> bool:
    if urgency.lower() == "past" or certainty.lower() == "unlikely":
        return False
    normalized = event.lower()
    dangerous = (
        "tornado",
        "hurricane",
        "tropical storm",
        "blizzard",
        "ice storm",
        "winter storm",
        "winter weather",
        "lake effect snow",
        "severe thunderstorm",
        "flash flood",
        "flood",
        "coastal flood",
        "extreme wind",
        "high wind",
        "dust storm",
        "snow squall",
        "storm surge",
        "tsunami",
        "avalanche",
        "extreme heat",
        "excessive heat",
        "extreme cold",
        "wind chill",
    )
    is_warning = "warning" in normalized or "emergency" in normalized
    serious_cap_level = severity.lower() in {"moderate", "severe", "extreme"}
    return is_warning and serious_cap_level and any(
        marker in normalized for marker in dangerous
    )


def _parse_nws_alerts(payload: dict) -> list[WeatherEmergencyAlert]:
    alerts: list[WeatherEmergencyAlert] = []
    for feature in payload.get("features") or []:
        properties = feature.get("properties") or {}
        event = str(properties.get("event") or "Weather alert")
        severity = str(properties.get("severity") or "Unknown")
        urgency = str(properties.get("urgency") or "Unknown")
        certainty = str(properties.get("certainty") or "Unknown")
        alert_id = str(feature.get("id") or properties.get("id") or "")
        if not alert_id:
            continue
        alerts.append(
            WeatherEmergencyAlert(
                alert_id=alert_id,
                event=event,
                headline=str(properties.get("headline") or event),
                severity=severity,
                urgency=urgency,
                certainty=certainty,
                onset=_alert_datetime(
                    properties.get("onset") or properties.get("effective")
                ),
                ends=_alert_datetime(properties.get("ends")),
                expires=_alert_datetime(properties.get("expires")),
                blocks_outdoor_run=_alert_blocks_outdoor_run(
                    event, severity, urgency, certainty
                ),
                source_url=alert_id,
            )
        )
    return alerts


def _recent_route_centroid(
    connection: sqlite3.Connection, precision: int
) -> tuple[float, float] | None:
    row = connection.execute(
        """
        SELECT ROUND(AVG(t.latitude), ?) AS latitude_key,
               ROUND(AVG(t.longitude), ?) AS longitude_key
        FROM trackpoints t JOIN activities a ON a.id=t.activity_id
        WHERE t.gps_valid=1
          AND t.activity_id=(
              SELECT t2.activity_id
              FROM trackpoints t2 JOIN activities a2 ON a2.id=t2.activity_id
              WHERE t2.gps_valid=1
              GROUP BY t2.activity_id
              ORDER BY a2.start_time_utc_epoch DESC, t2.activity_id DESC
              LIMIT 1
          )
        """,
        (precision, precision),
    ).fetchone()
    if not row or row["latitude_key"] is None or row["longitude_key"] is None:
        return None
    return float(row["latitude_key"]), float(row["longitude_key"])


def _active_nws_alerts(
    connection: sqlite3.Connection,
    config: dict,
    project_root: str | Path,
    *,
    downloader: Callable[[str, float], dict] = _download_nws_json,
) -> tuple[list[WeatherEmergencyAlert], bool]:
    weather_config = config.get("weather", {})
    if not bool(weather_config.get("emergency_alerts_enabled", False)):
        return [], False
    precision = int(weather_config.get("coordinate_precision", 2))
    centroid = _recent_route_centroid(connection, precision)
    if centroid is None:
        return [], False
    salt_path = Path(project_root) / weather_config.get(
        "privacy_salt_path", "data/weather_privacy_salt"
    )
    latitude, longitude = anonymize_coordinates(
        centroid[0],
        centroid[1],
        float(weather_config.get("privacy_jitter_radius_km", 0)),
        load_or_create_privacy_salt(salt_path),
    )
    point_key = f"{latitude:.4f},{longitude:.4f}"
    now = datetime.now(timezone.utc)
    cached_alerts: list[WeatherEmergencyAlert] = []
    cached_age: float | None = None
    row = connection.execute(
        "SELECT value_json FROM app_state WHERE key=?", (NWS_ALERT_CACHE_KEY,)
    ).fetchone()
    if row:
        try:
            cached = json.loads(row[0])
            fetched_at = _alert_datetime(cached.get("fetched_at_utc"))
            if cached.get("point_key") == point_key and fetched_at:
                cached_age = (now - fetched_at.astimezone(timezone.utc)).total_seconds()
                cached_alerts = [
                    WeatherEmergencyAlert.model_validate(item)
                    for item in cached.get("alerts") or []
                ]
                if cached_age <= NWS_ALERT_CACHE_SECONDS:
                    return cached_alerts, True
        except (TypeError, ValueError, json.JSONDecodeError):
            cached_alerts = []
            cached_age = None

    endpoint = str(
        weather_config.get(
            "emergency_alerts_endpoint",
            "https://api.weather.gov/alerts/active",
        )
    )
    url = f"{endpoint}?{urlencode({'point': point_key})}"
    timeout = min(8.0, float(weather_config.get("request_timeout_seconds", 30)))
    try:
        alerts = _parse_nws_alerts(downloader(url, timeout))
    except Exception:
        if cached_age is not None and cached_age <= NWS_ALERT_STALE_FALLBACK_SECONDS:
            return cached_alerts, False
        return [], False

    connection.execute(
        """
        INSERT INTO app_state(key,value_json,updated_at_utc) VALUES (?,?,?)
        ON CONFLICT(key) DO UPDATE SET
            value_json=excluded.value_json,
            updated_at_utc=excluded.updated_at_utc
        """,
        (
            NWS_ALERT_CACHE_KEY,
            json.dumps(
                {
                    "point_key": point_key,
                    "fetched_at_utc": now.isoformat(),
                    "alerts": [item.model_dump(mode="json") for item in alerts],
                }
            ),
            now.isoformat(),
        ),
    )
    return alerts, True


def _alerts_for_time(
    alerts: list[WeatherEmergencyAlert], planned_at: datetime
) -> list[WeatherEmergencyAlert]:
    moment = planned_at.astimezone(timezone.utc)
    return [
        alert
        for alert in alerts
        if (alert.onset is None or alert.onset.astimezone(timezone.utc) <= moment)
        and (
            (alert.ends or alert.expires) is None
            or (alert.ends or alert.expires).astimezone(timezone.utc) >= moment
        )
    ]


def _forecast_url(endpoint: str, latitude: float, longitude: float) -> str:
    parameters = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join(FORECAST_HOURLY_VARIABLES),
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "precipitation_unit": "inch",
        "timeformat": "unixtime",
        "timezone": "GMT",
        "forecast_days": 16,
    }
    return f"{endpoint}?{urlencode(parameters)}"


def _forecast_values(response: dict, planned_at: datetime) -> dict[str, float | None]:
    values = interpolate_hourly(response.get("hourly") or {}, planned_at)
    hourly = response.get("hourly") or {}
    times = [float(value) for value in hourly.get("time") or []]
    probabilities = hourly.get("precipitation_probability") or []
    probability = None
    if times and probabilities:
        target = planned_at.astimezone(timezone.utc).timestamp()
        index = min(range(len(times)), key=lambda item: abs(times[item] - target))
        if index < len(probabilities):
            probability = probabilities[index]
    values["precipitation_probability_percent"] = probability
    if times:
        target = planned_at.astimezone(timezone.utc).timestamp()
        index = min(range(len(times)), key=lambda item: abs(times[item] - target))
        for source, destination in (
            ("snowfall", "snowfall_in"),
            ("weather_code", "weather_code"),
        ):
            series = hourly.get(source) or []
            values[destination] = series[index] if index < len(series) else None
        visibility = hourly.get("visibility") or []
        visibility_meters = visibility[index] if index < len(visibility) else None
        values["visibility_miles"] = (
            float(visibility_meters) / 1609.344
            if visibility_meters is not None
            else None
        )
    return values


def _forecast_response(
    connection: sqlite3.Connection,
    config: dict,
    project_root: str | Path,
    downloader: Callable[[str, float], dict],
) -> dict | None:
    weather_config = config.get("weather", {})
    if not bool(weather_config.get("forecast_enabled", False)):
        return None
    precision = int(weather_config.get("coordinate_precision", 2))
    centroid = _recent_route_centroid(connection, precision)
    if centroid is None:
        return None
    salt_path = Path(project_root) / weather_config.get(
        "privacy_salt_path", "data/weather_privacy_salt"
    )
    latitude, longitude = anonymize_coordinates(
        centroid[0],
        centroid[1],
        float(weather_config.get("privacy_jitter_radius_km", 0)),
        load_or_create_privacy_salt(salt_path),
    )
    endpoint = str(
        weather_config.get("forecast_endpoint", "https://api.open-meteo.com/v1/forecast")
    )
    point_key = f"{latitude:.4f},{longitude:.4f}"
    now = datetime.now(timezone.utc)
    cached_response: dict | None = None
    cached_age: float | None = None
    row = connection.execute(
        "SELECT value_json FROM app_state WHERE key=?", (FORECAST_CACHE_KEY,)
    ).fetchone()
    if row:
        try:
            cached = json.loads(row[0])
            fetched_at = _alert_datetime(cached.get("fetched_at_utc"))
            if (
                cached.get("point_key") == point_key
                and cached.get("endpoint") == endpoint
                and fetched_at
                and isinstance(cached.get("response"), dict)
            ):
                cached_age = (
                    now - fetched_at.astimezone(timezone.utc)
                ).total_seconds()
                cached_response = cached["response"]
                if cached_age <= FORECAST_CACHE_SECONDS:
                    return cached_response
        except (TypeError, ValueError, json.JSONDecodeError):
            cached_response = None
            cached_age = None
    timeout = min(8.0, float(weather_config.get("request_timeout_seconds", 30)))
    try:
        response = downloader(_forecast_url(endpoint, latitude, longitude), timeout)
    except Exception:
        if (
            cached_response is not None
            and cached_age is not None
            and cached_age <= FORECAST_STALE_FALLBACK_SECONDS
        ):
            return cached_response
        return None
    if response.get("error"):
        return None
    connection.execute(
        """
        INSERT INTO app_state(key,value_json,updated_at_utc) VALUES (?,?,?)
        ON CONFLICT(key) DO UPDATE SET
            value_json=excluded.value_json,
            updated_at_utc=excluded.updated_at_utc
        """,
        (
            FORECAST_CACHE_KEY,
            json.dumps(
                {
                    "point_key": point_key,
                    "endpoint": endpoint,
                    "fetched_at_utc": now.isoformat(),
                    "response": response,
                }
            ),
            now.isoformat(),
        ),
    )
    return response


def _planned_weather(
    response: dict,
    planned_at: datetime,
    emergency_alerts: list[WeatherEmergencyAlert] | None = None,
    emergency_alerts_checked: bool = False,
) -> PlannedWeather | None:
    values = _forecast_values(response, planned_at.astimezone(timezone.utc))
    applicable_alerts = _alerts_for_time(emergency_alerts or [], planned_at)
    if values.get("temperature_f") is None and not applicable_alerts:
        return None
    return PlannedWeather(
        forecast_time=planned_at,
        temperature_f=values.get("temperature_f"),
        dewpoint_f=values.get("dewpoint_f"),
        apparent_temperature_f=values.get("apparent_temperature_f"),
        wind_speed_mph=values.get("wind_speed_mph"),
        wind_gust_mph=values.get("wind_gust_mph"),
        precipitation_probability_percent=values.get("precipitation_probability_percent"),
        precipitation_in=values.get("precipitation_in"),
        snowfall_in=values.get("snowfall_in"),
        visibility_miles=values.get("visibility_miles"),
        weather_code=(
            int(values["weather_code"])
            if values.get("weather_code") is not None
            else None
        ),
        emergency_alerts_checked=emergency_alerts_checked,
        emergency_alerts=applicable_alerts,
        confidence=ConfidenceLevel.MODERATE,
    )


def _choose_from_forecast_response(
    candidates: list[datetime],
    response: dict | None,
) -> tuple[datetime, PlannedWeather | None]:
    return _forecast_options_from_response(candidates, response)[0]


def _forecast_rank(
    option: tuple[datetime, PlannedWeather | None],
) -> tuple[float, float, float, float, float]:
    """Rank forecasts only by material training stress.

    Candidate order is stable, so equally safe conditions retain the ordinary
    morning/noon/evening order. Tiny raw differences in wind or temperature
    must not move an otherwise identical workout to the end of the day.
    """
    _, weather = option
    if weather is None:
        return 2.0, 101.0, 200.0, 200.0, 200.0
    # The shared model already combines meaningful wind, apparent temperature,
    # humidity, precipitation, relative spikes, and absolute extremes.
    return assess_training_weather(weather).score, 0.0, 0.0, 0.0, 0.0


def _forecast_options_from_response(
    candidates: list[datetime],
    response: dict | None,
    emergency_alerts: list[WeatherEmergencyAlert] | None = None,
    emergency_alerts_checked: bool = False,
) -> list[tuple[datetime, PlannedWeather | None]]:
    """Return all usable times, ordered by forecast quality."""
    now = datetime.now(timezone.utc)
    valid = [
        candidate for candidate in candidates
        if (
            candidate.astimezone(timezone.utc) <= now + timedelta(days=16)
            and (
                candidate.astimezone(timezone.utc) >= now - timedelta(hours=2)
                or candidate.date() == now.astimezone(candidate.tzinfo).date()
            )
        )
    ]
    if not valid:
        fallback = now + timedelta(minutes=15)
        return [(fallback, None)]
    options = []
    for candidate in valid:
        weather = _planned_weather(
            response or {},
            candidate,
            emergency_alerts,
            emergency_alerts_checked,
        )
        if weather is not None:
            options.append((candidate, weather))
    if not options:
        return [(valid[len(valid) // 2], None)]
    return sorted(options, key=_forecast_rank)


def planned_forecast_options(
    connection: sqlite3.Connection,
    config: dict,
    project_root: str | Path,
    candidate_groups: list[list[datetime]],
    *,
    downloader: Callable[[str, float], dict] = _download_json,
    alert_downloader: Callable[[str, float], dict] = _download_nws_json,
) -> list[list[tuple[datetime, PlannedWeather | None]]]:
    """Return every weather-backed time option from one forecast response."""
    response = _forecast_response(connection, config, project_root, downloader)
    emergency_alerts, alerts_checked = _active_nws_alerts(
        connection,
        config,
        project_root,
        downloader=alert_downloader,
    )
    return [
        _forecast_options_from_response(
            candidates, response, emergency_alerts, alerts_checked
        )
        for candidates in candidate_groups
    ]


def choose_planned_forecasts(
    connection: sqlite3.Connection,
    config: dict,
    project_root: str | Path,
    candidate_groups: list[list[datetime]],
    *,
    downloader: Callable[[str, float], dict] = _download_json,
) -> list[tuple[datetime, PlannedWeather | None]]:
    """Choose times for several days from one shared forecast response."""
    return [
        options[0]
        for options in planned_forecast_options(
            connection,
            config,
            project_root,
            candidate_groups,
            downloader=downloader,
        )
    ]


def choose_planned_forecast(
    connection: sqlite3.Connection,
    config: dict,
    project_root: str | Path,
    candidates: list[datetime],
    *,
    downloader: Callable[[str, float], dict] = _download_json,
) -> tuple[datetime, PlannedWeather | None]:
    """Choose a day's time from forecast candidates without a saved preference."""
    return choose_planned_forecasts(
        connection,
        config,
        project_root,
        [candidates],
        downloader=downloader,
    )[0]


def get_planned_forecast(
    connection: sqlite3.Connection,
    config: dict,
    project_root: str | Path,
    planned_at: datetime,
    *,
    downloader: Callable[[str, float], dict] = _download_json,
    alert_downloader: Callable[[str, float], dict] = _download_nws_json,
) -> PlannedWeather | None:
    """Return an hourly forecast when the requested time is in forecast range.

    Network or provider failures intentionally return ``None``. A forecast is
    useful context, but it must never prevent the local coaching rules from
    producing a recommendation.
    """

    moment = planned_at.astimezone(timezone.utc)
    now = datetime.now(timezone.utc)
    if moment < now - timedelta(hours=2) or moment > now + timedelta(days=16):
        return None

    response = _forecast_response(connection, config, project_root, downloader)
    emergency_alerts, alerts_checked = _active_nws_alerts(
        connection,
        config,
        project_root,
        downloader=alert_downloader,
    )
    return _planned_weather(
        response or {}, planned_at, emergency_alerts, alerts_checked
    )
