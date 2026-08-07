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
import sqlite3

from .weather import (
    _download_json,
    anonymize_coordinates,
    interpolate_hourly,
    load_or_create_privacy_salt,
)
from .web.schemas import ConfidenceLevel, PlannedWeather


FORECAST_HOURLY_VARIABLES = (
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "apparent_temperature",
    "precipitation_probability",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
)


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
    timeout = min(8.0, float(weather_config.get("request_timeout_seconds", 30)))
    try:
        response = downloader(_forecast_url(endpoint, latitude, longitude), timeout)
    except Exception:
        return None
    return None if response.get("error") else response


def _planned_weather(response: dict, planned_at: datetime) -> PlannedWeather | None:
    values = _forecast_values(response, planned_at.astimezone(timezone.utc))
    if values.get("temperature_f") is None:
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
        confidence=ConfidenceLevel.MODERATE,
    )


def choose_planned_forecast(
    connection: sqlite3.Connection,
    config: dict,
    project_root: str | Path,
    candidates: list[datetime],
    *,
    downloader: Callable[[str, float], dict] = _download_json,
) -> tuple[datetime, PlannedWeather | None]:
    """Choose a day's time from forecast candidates without a saved preference."""
    now = datetime.now(timezone.utc)
    valid = [
        candidate for candidate in candidates
        if now - timedelta(hours=2) <= candidate.astimezone(timezone.utc) <= now + timedelta(days=16)
    ]
    if not valid:
        fallback = now + timedelta(minutes=15)
        return fallback, None
    response = _forecast_response(connection, config, project_root, downloader)
    forecasts = [weather for candidate in valid if (weather := _planned_weather(response, candidate))] if response else []
    if not forecasts:
        return valid[len(valid) // 2], None

    def rank(weather: PlannedWeather) -> tuple[bool, float, float, float, float]:
        apparent = weather.apparent_temperature_f or weather.temperature_f or 70.0
        dewpoint = weather.dewpoint_f or 50.0
        wind = max(weather.wind_speed_mph or 0.0, (weather.wind_gust_mph or 0.0) * 0.5)
        precipitation = weather.precipitation_probability_percent or 0.0
        caution = apparent >= 85 or dewpoint >= 70 or wind >= 20
        return caution, precipitation, apparent, dewpoint, wind

    selected = min(forecasts, key=rank)
    return selected.forecast_time, selected


def get_planned_forecast(
    connection: sqlite3.Connection,
    config: dict,
    project_root: str | Path,
    planned_at: datetime,
    *,
    downloader: Callable[[str, float], dict] = _download_json,
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
    return _planned_weather(response, planned_at) if response else None
