"""Cached Open-Meteo historical weather and segment-time interpolation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from math import cos, pi, radians, sin, sqrt
from typing import Callable
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import hashlib
import hmac
import json
import secrets
import sqlite3
import time

from .privacy import private_directory, private_file

from .db import initialize, transaction

HOURLY_VARIABLES = (
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "apparent_temperature",
    "precipitation",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
)

FIELD_MAP = {
    "temperature_2m": "temperature_f",
    "relative_humidity_2m": "relative_humidity_percent",
    "dew_point_2m": "dewpoint_f",
    "apparent_temperature": "apparent_temperature_f",
    "precipitation": "precipitation_in",
    "surface_pressure": "surface_pressure_hpa",
    "wind_speed_10m": "wind_speed_mph",
    "wind_direction_10m": "wind_direction_degrees",
    "wind_gusts_10m": "wind_gust_mph",
}


@dataclass(slots=True)
class WeatherSummary:
    historical_weather_enabled: bool = True
    gps_activities: int = 0
    estimated_location_activities: int = 0
    cache_hits: int = 0
    api_requests: int = 0
    cache_days_added: int = 0
    activities_updated: int = 0
    segments_updated: int = 0
    skipped_no_gps: int = 0
    failures: int = 0


def load_or_create_privacy_salt(path: str | Path) -> bytes:
    salt_path = Path(path)
    if salt_path.exists():
        private_file(salt_path)
        value = salt_path.read_bytes()
        if len(value) < 16:
            raise ValueError(f"Weather privacy salt is unexpectedly short: {salt_path}")
        return value
    private_directory(salt_path.parent)
    value = secrets.token_bytes(32)
    salt_path.write_bytes(value)
    private_file(salt_path)
    return value


def anonymize_coordinates(
    latitude: float,
    longitude: float,
    radius_km: float,
    salt: bytes,
) -> tuple[float, float]:
    """Deterministically displace a rounded coordinate within a radius.

    The local random salt makes the offset stable for caching but prevents an
    observer who only knows this algorithm from reversing all offsets. Radius
    uses sqrt(U) so points are distributed uniformly over the disk area.
    """
    if radius_km <= 0:
        return latitude, longitude
    message = f"{latitude:.6f}|{longitude:.6f}".encode()
    digest = hmac.new(salt, message, hashlib.sha256).digest()
    radial_unit = int.from_bytes(digest[:8], "big") / (2**64 - 1)
    angle_unit = int.from_bytes(digest[8:16], "big") / (2**64 - 1)
    distance_km = radius_km * sqrt(radial_unit)
    angle = 2 * pi * angle_unit
    latitude_offset = distance_km * cos(angle) / 111.32
    longitude_scale = max(0.1, cos(radians(latitude)))
    longitude_offset = distance_km * sin(angle) / (111.32 * longitude_scale)
    return round(latitude + latitude_offset, 4), round(longitude + longitude_offset, 4)


def wind_components(
    wind_speed_mph: float | None,
    wind_from_degrees: float | None,
    runner_bearing_degrees: float | None,
) -> tuple[float | None, float | None, float | None]:
    """Return headwind, tailwind, and absolute crosswind components.

    Meteorological direction is where wind comes FROM. A north wind (0°)
    therefore produces positive headwind for a northbound runner and tailwind
    for a southbound runner.
    """
    if wind_speed_mph is None or wind_from_degrees is None or runner_bearing_degrees is None:
        return None, None, None
    angle = radians(wind_from_degrees - runner_bearing_degrees)
    signed_headwind = wind_speed_mph * cos(angle)
    return max(signed_headwind, 0.0), max(-signed_headwind, 0.0), abs(wind_speed_mph * sin(angle))


def _linear(left: float | None, right: float | None, fraction: float) -> float | None:
    if left is None:
        return right
    if right is None:
        return left
    return left + (right - left) * fraction


def _direction(left: float | None, right: float | None, fraction: float) -> float | None:
    if left is None:
        return right
    if right is None:
        return left
    delta = ((right - left + 180.0) % 360.0) - 180.0
    return (left + delta * fraction) % 360.0


def interpolate_hourly(hourly: dict, timestamp: datetime) -> dict[str, float | None]:
    times = hourly.get("time") or []
    if not times:
        return {target: None for target in FIELD_MAP.values()}
    epoch = timestamp.astimezone(timezone.utc).timestamp()
    numeric_times = [float(item) for item in times]
    if epoch <= numeric_times[0]:
        left = right = 0
        fraction = 0.0
    elif epoch >= numeric_times[-1]:
        left = right = len(numeric_times) - 1
        fraction = 0.0
    else:
        right = next(index for index, value in enumerate(numeric_times) if value >= epoch)
        left = right - 1
        span = numeric_times[right] - numeric_times[left]
        fraction = (epoch - numeric_times[left]) / span if span else 0.0
    result: dict[str, float | None] = {}
    for source, target in FIELD_MAP.items():
        values = hourly.get(source) or []
        left_value = values[left] if left < len(values) else None
        right_value = values[right] if right < len(values) else None
        if source == "wind_direction_10m":
            result[target] = _direction(left_value, right_value, fraction)
        elif source == "precipitation":
            # Hourly precipitation is a preceding-hour accumulation, so a
            # nearest-hour selection is less misleading than linear blending.
            result[target] = left_value if fraction < 0.5 else right_value
        else:
            result[target] = _linear(left_value, right_value, fraction)
    return result


def _request_url(endpoint: str, latitude: float, longitude: float, start: date, end: date, model: str) -> str:
    parameters = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "hourly": ",".join(HOURLY_VARIABLES),
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "precipitation_unit": "inch",
        "timeformat": "unixtime",
        "timezone": "GMT",
    }
    if model != "best_match":
        parameters["models"] = model
    return f"{endpoint}?{urlencode(parameters)}"


def _download_json(url: str, timeout: float) -> dict:
    request = Request(url, headers={"User-Agent": "garmin-run-analysis/0.1"})
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as error:  # network and malformed response are both retried
            last_error = error
            if attempt < 2:
                time.sleep(1.0 * (attempt + 1))
    assert last_error is not None
    raise last_error


def save_activity_postal_code(
    connection: sqlite3.Connection,
    activity_id: int,
    postal_code: str | None,
    config: dict,
    *,
    downloader: Callable[[str, float], dict] = _download_json,
) -> dict | None:
    """Resolve and persist a coarse user-supplied location for a no-GPS run."""

    activity = connection.execute(
        """SELECT a.id,COUNT(t.id) AS point_count,SUM(t.gps_valid) AS valid_gps
           FROM activities a LEFT JOIN trackpoints t ON t.activity_id=a.id
           WHERE a.id=? GROUP BY a.id""",
        (activity_id,),
    ).fetchone()
    if not activity:
        raise LookupError("Run not found")
    if postal_code is None:
        connection.execute(
            "DELETE FROM activity_location_overrides WHERE activity_id=?", (activity_id,)
        )
        connection.commit()
        return None
    if int(activity["valid_gps"] or 0) > 0:
        raise ValueError("A ZIP-code location override is only needed for a run with no GPS.")

    weather_config = config["weather"]
    endpoint = str(
        weather_config.get(
            "geocoding_endpoint", "https://geocoding-api.open-meteo.com/v1/search"
        )
    )
    url = f"{endpoint}?{urlencode({'name': postal_code, 'count': 10, 'language': 'en', 'format': 'json', 'countryCode': 'US'})}"
    response = downloader(url, float(weather_config.get("request_timeout_seconds", 30)))
    candidates = [
        item
        for item in response.get("results") or []
        if str(item.get("country_code") or "").upper() == "US"
        and item.get("latitude") is not None
        and item.get("longitude") is not None
    ]
    exact = [item for item in candidates if postal_code in (item.get("postcodes") or [])]
    selected = (exact or candidates or [None])[0]
    if selected is None:
        raise ValueError(f"ZIP code {postal_code} was not found by Open-Meteo geocoding.")
    locality = str(selected.get("name") or selected.get("admin2") or "").strip() or None
    region = str(selected.get("admin1") or "").strip() or None
    connection.execute(
        """
        INSERT INTO activity_location_overrides(
            activity_id,postal_code,latitude,longitude,locality,region,country_code,source,updated_at_utc
        ) VALUES (?,?,?,?,?,?,?,?,?)
        ON CONFLICT(activity_id) DO UPDATE SET
            postal_code=excluded.postal_code,latitude=excluded.latitude,
            longitude=excluded.longitude,locality=excluded.locality,region=excluded.region,
            country_code=excluded.country_code,source=excluded.source,
            updated_at_utc=excluded.updated_at_utc
        """,
        (
            activity_id,
            postal_code,
            float(selected["latitude"]),
            float(selected["longitude"]),
            locality,
            region,
            "US",
            "open_meteo_geocoding",
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    connection.commit()
    return {
        "postal_code": postal_code,
        "locality": locality,
        "region": region,
        "latitude": float(selected["latitude"]),
        "longitude": float(selected["longitude"]),
    }


def _split_days(response: dict) -> dict[str, dict]:
    hourly = response.get("hourly") or {}
    times = hourly.get("time") or []
    by_day: dict[str, dict] = {}
    for index, raw_time in enumerate(times):
        day = datetime.fromtimestamp(float(raw_time), timezone.utc).date().isoformat()
        daily = by_day.setdefault(day, {"time": []})
        daily["time"].append(raw_time)
        for variable in HOURLY_VARIABLES:
            values = hourly.get(variable) or []
            daily.setdefault(variable, []).append(values[index] if index < len(values) else None)
    return by_day


def _required_locations(
    connection: sqlite3.Connection,
    precision: int,
    estimated_location_max_days: float,
    estimated_location_sources: dict[str, str] | None = None,
) -> list[dict]:
    rows = connection.execute(
        """
        SELECT a.id AS activity_row_id, a.activity_id, a.start_time_utc,
               a.start_time_utc_epoch,
               ROUND(AVG(CASE WHEN t.gps_valid THEN t.latitude END), ?) AS latitude_key,
               ROUND(AVG(CASE WHEN t.gps_valid THEN t.longitude END), ?) AS longitude_key,
               SUM(CASE WHEN t.gps_valid THEN 1 ELSE 0 END) AS valid_gps_points,
               lo.postal_code,lo.latitude AS override_latitude,
               lo.longitude AS override_longitude,lo.locality,lo.region
        FROM activities a LEFT JOIN trackpoints t ON t.activity_id = a.id
        LEFT JOIN activity_location_overrides lo ON lo.activity_id=a.id
        WHERE a.start_time_utc IS NOT NULL
        GROUP BY a.id
        ORDER BY a.start_time_utc_epoch
        """,
        (precision, precision),
    ).fetchall()
    known = [row for row in rows if int(row["valid_gps_points"] or 0) > 0]
    known_by_external_id = {str(row["activity_id"]): row for row in known}
    estimated_location_sources = estimated_location_sources or {}
    output: list[dict] = []
    maximum_seconds = estimated_location_max_days * 86400.0
    for row in rows:
        item = dict(row)
        item["location_basis"] = "recorded_route_centroid"
        item["location_source_activity_id"] = int(row["activity_row_id"])
        item["location_offset_days"] = 0.0
        if int(row["valid_gps_points"] or 0) == 0:
            if row["override_latitude"] is not None and row["override_longitude"] is not None:
                item["latitude_key"] = round(float(row["override_latitude"]), precision)
                item["longitude_key"] = round(float(row["override_longitude"]), precision)
                item["location_basis"] = "postal_code_centroid_estimate"
                item["location_source_activity_id"] = int(row["activity_row_id"])
                item["location_label"] = ", ".join(
                    value for value in (row["locality"], row["region"]) if value
                ) or f"ZIP {row['postal_code']}"
                output.append(item)
                continue
            target_epoch = float(row["start_time_utc_epoch"])
            source_external_id = estimated_location_sources.get(str(row["activity_id"]))
            nearest = (
                known_by_external_id.get(str(source_external_id))
                if source_external_id
                else None
            )
            if nearest is None:
                continue
            offset_seconds = abs(float(nearest["start_time_utc_epoch"]) - target_epoch)
            if offset_seconds > maximum_seconds:
                continue
            item["latitude_key"] = nearest["latitude_key"]
            item["longitude_key"] = nearest["longitude_key"]
            item["location_basis"] = "nearest_run_centroid_estimate"
            item["location_source_activity_id"] = int(nearest["activity_row_id"])
            item["location_offset_days"] = offset_seconds / 86400.0
            item["location_label"] = "Confirmed nearby run"
        output.append(item)
    return output


def _remove_unapproved_estimated_weather(
    connection: sqlite3.Connection, approved_activity_ids: set[int]
) -> None:
    stale: list[int] = []
    for row in connection.execute(
        "SELECT activity_id,derived_weather_json FROM activity_weather"
    ).fetchall():
        derived = json.loads(row["derived_weather_json"] or "{}")
        if (
            derived.get("location_basis")
            in {"nearest_run_centroid_estimate", "postal_code_centroid_estimate"}
            and int(row["activity_id"]) not in approved_activity_ids
        ):
            stale.append(int(row["activity_id"]))
    if not stale:
        return
    placeholders = ",".join("?" for _ in stale)
    weather_columns = ",".join(
        f"{column}=NULL"
        for column in (
            *FIELD_MAP.values(),
            "headwind_mph",
            "tailwind_mph",
            "crosswind_mph",
            "weather_quality",
        )
    )
    with transaction(connection):
        connection.execute(
            f"DELETE FROM activity_weather WHERE activity_id IN ({placeholders})", stale
        )
        connection.execute(
            f"UPDATE segments SET {weather_columns} WHERE activity_id IN ({placeholders})",
            stale,
        )


def _cache_key(provider: str, latitude: float, longitude: float, day: str) -> tuple:
    return provider, latitude, longitude, day


def _ensure_cache(
    connection: sqlite3.Connection,
    requirements: dict[tuple[float, float, int], set[str]],
    config: dict,
    cache_directory: Path,
    summary: WeatherSummary,
    downloader: Callable[[str, float], dict],
    force: bool,
) -> None:
    provider = str(config["provider"])
    endpoint = str(config["endpoint"])
    model = str(config.get("model", "best_match"))
    timeout = float(config.get("request_timeout_seconds", 30))
    private_directory(cache_directory)
    consecutive_failures = 0
    for (latitude, longitude, _year), required_days in requirements.items():
        missing: list[str] = []
        for day in sorted(required_days):
            exists = connection.execute(
                """SELECT 1 FROM weather_cache
                   WHERE provider=? AND latitude_key=? AND longitude_key=? AND date_local=?""",
                _cache_key(provider, latitude, longitude, day),
            ).fetchone()
            if exists and not force:
                summary.cache_hits += 1
            else:
                missing.append(day)
        if not missing:
            continue
        start = date.fromisoformat(min(missing))
        end = date.fromisoformat(max(missing))
        url = _request_url(endpoint, latitude, longitude, start, end, model)
        try:
            response = downloader(url, timeout)
            if response.get("error"):
                raise RuntimeError(str(response.get("reason") or response))
        except Exception:
            summary.failures += 1
            consecutive_failures += 1
            if consecutive_failures >= 3:
                return
            continue
        consecutive_failures = 0
        summary.api_requests += 1
        request_hash = hashlib.sha256(url.encode()).hexdigest()
        cache_file = cache_directory / f"{request_hash}.json"
        cache_file.write_text(json.dumps(response, indent=2) + "\n", encoding="utf-8")
        private_file(cache_file)
        daily = _split_days(response)
        with transaction(connection):
            for day in missing:
                if day not in daily:
                    summary.failures += 1
                    continue
                connection.execute(
                    """
                    INSERT INTO weather_cache(
                        provider, latitude_key, longitude_key, date_local,
                        response_json, fetched_at_utc, cache_file, request_url, hourly_units_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(provider, latitude_key, longitude_key, date_local) DO UPDATE SET
                        response_json=excluded.response_json,
                        fetched_at_utc=excluded.fetched_at_utc,
                        cache_file=excluded.cache_file,
                        request_url=excluded.request_url,
                        hourly_units_json=excluded.hourly_units_json
                    """,
                    (
                        provider,
                        latitude,
                        longitude,
                        day,
                        json.dumps(daily[day]),
                        datetime.now(timezone.utc).isoformat(),
                        str(cache_file),
                        url,
                        json.dumps(response.get("hourly_units") or {}),
                    ),
                )
                summary.cache_days_added += 1


def _load_hourly(
    connection: sqlite3.Connection, provider: str, latitude: float, longitude: float, days: set[str]
) -> tuple[dict, list[int]]:
    combined = {"time": []}
    cache_ids: list[int] = []
    for day in sorted(days):
        row = connection.execute(
            """SELECT id,response_json FROM weather_cache
               WHERE provider=? AND latitude_key=? AND longitude_key=? AND date_local=?""",
            _cache_key(provider, latitude, longitude, day),
        ).fetchone()
        if not row:
            continue
        cache_ids.append(int(row["id"]))
        daily = json.loads(row["response_json"])
        for key, values in daily.items():
            combined.setdefault(key, []).extend(values)
    return combined, cache_ids


def _midpoint(start: str | None, end: str | None, fallback: datetime) -> datetime:
    if not start or not end:
        return fallback
    first = datetime.fromisoformat(start)
    second = datetime.fromisoformat(end)
    return first + (second - first) / 2


def update_weather(
    connection: sqlite3.Connection,
    config: dict,
    project_root: str | Path,
    *,
    force: bool = False,
    downloader: Callable[[str, float], dict] = _download_json,
) -> WeatherSummary:
    initialize(connection)
    weather_config = config["weather"]
    if not bool(weather_config.get("historical_enabled", False)):
        return WeatherSummary(historical_weather_enabled=False)
    precision = int(weather_config["coordinate_precision"])
    privacy_radius = float(weather_config.get("privacy_jitter_radius_km", 0))
    privacy_salt_path = Path(project_root) / weather_config.get(
        "privacy_salt_path", "data/weather_privacy_salt"
    )
    privacy_salt = load_or_create_privacy_salt(privacy_salt_path)
    provider = str(weather_config["provider"])
    locations = _required_locations(
        connection,
        precision,
        float(weather_config.get("estimated_location_max_days", 14)),
        {
            str(target): str(source)
            for target, source in dict(
                weather_config.get("estimated_location_sources") or {}
            ).items()
        },
    )
    directly_located = sum(
        row["location_basis"] == "recorded_route_centroid" for row in locations
    )
    estimated_locations = len(locations) - directly_located
    total_no_gps = int(
        connection.execute(
            "SELECT COUNT(*) FROM activities WHERE gps_quality='gps_missing'"
        ).fetchone()[0]
    )
    summary = WeatherSummary(
        gps_activities=directly_located,
        estimated_location_activities=estimated_locations,
        skipped_no_gps=max(0, total_no_gps - estimated_locations),
    )
    _remove_unapproved_estimated_weather(
        connection,
        {
            int(row["activity_row_id"])
            for row in locations
            if row["location_basis"] != "recorded_route_centroid"
        },
    )
    requirements: dict[tuple[float, float, int], set[str]] = {}
    activity_days: dict[int, set[str]] = {}
    anonymized_locations: dict[int, tuple[float, float]] = {}
    for row in locations:
        start = datetime.fromisoformat(row["start_time_utc"])
        segment_times = connection.execute(
            "SELECT start_time_utc,end_time_utc FROM segments WHERE activity_id=?",
            (row["activity_row_id"],),
        ).fetchall()
        times = [start]
        times.extend(_midpoint(item[0], item[1], start) for item in segment_times)
        days = {moment.astimezone(timezone.utc).date().isoformat() for moment in times}
        activity_days[int(row["activity_row_id"])] = days
        anonymized = anonymize_coordinates(
            float(row["latitude_key"]),
            float(row["longitude_key"]),
            privacy_radius,
            privacy_salt,
        )
        anonymized_locations[int(row["activity_row_id"])] = anonymized
        for day in days:
            key = (anonymized[0], anonymized[1], date.fromisoformat(day).year)
            requirements.setdefault(key, set()).add(day)
    cache_directory = Path(project_root) / config["paths"]["weather_cache"]
    _ensure_cache(
        connection,
        requirements,
        weather_config,
        cache_directory,
        summary,
        downloader,
        force,
    )

    weather_columns = list(FIELD_MAP.values()) + ["headwind_mph", "tailwind_mph", "crosswind_mph"]
    for row in locations:
        activity_id = int(row["activity_row_id"])
        latitude, longitude = anonymized_locations[activity_id]
        hourly, cache_ids = _load_hourly(
            connection, provider, latitude, longitude, activity_days[activity_id]
        )
        if not hourly.get("time"):
            continue
        start = datetime.fromisoformat(row["start_time_utc"])
        segment_rows = connection.execute(
            """SELECT id,start_time_utc,end_time_utc,moving_time_s,route_bearing_degrees
               FROM segments WHERE activity_id=? ORDER BY segment_index""",
            (activity_id,),
        ).fetchall()
        weighted = {name: 0.0 for name in weather_columns}
        weights = {name: 0.0 for name in weather_columns}
        wind_direction_x = 0.0
        wind_direction_y = 0.0
        wind_direction_weight = 0.0
        segment_derived: list[dict] = []
        with transaction(connection):
            for segment in segment_rows:
                moment = _midpoint(segment["start_time_utc"], segment["end_time_utc"], start)
                values = interpolate_hourly(hourly, moment)
                headwind, tailwind, crosswind = wind_components(
                    values["wind_speed_mph"],
                    values["wind_direction_degrees"],
                    segment["route_bearing_degrees"],
                )
                values.update(
                    {"headwind_mph": headwind, "tailwind_mph": tailwind, "crosswind_mph": crosswind}
                )
                quality = (
                    "hourly_interpolated_estimated_location"
                    if row["location_basis"] != "recorded_route_centroid"
                    else "hourly_interpolated"
                )
                connection.execute(
                    """
                    UPDATE segments SET temperature_f=?,dewpoint_f=?,relative_humidity_percent=?,
                        apparent_temperature_f=?,wind_speed_mph=?,wind_gust_mph=?,
                        wind_direction_degrees=?,precipitation_in=?,surface_pressure_hpa=?,
                        headwind_mph=?,tailwind_mph=?,crosswind_mph=?,weather_quality=? WHERE id=?
                    """,
                    (
                        values["temperature_f"], values["dewpoint_f"],
                        values["relative_humidity_percent"], values["apparent_temperature_f"],
                        values["wind_speed_mph"], values["wind_gust_mph"],
                        values["wind_direction_degrees"], values["precipitation_in"],
                        values["surface_pressure_hpa"], headwind, tailwind, crosswind,
                        quality, segment["id"],
                    ),
                )
                weight = float(segment["moving_time_s"] or 0) or 1.0
                for name in weather_columns:
                    if name == "wind_direction_degrees":
                        continue
                    if values.get(name) is not None:
                        weighted[name] += float(values[name]) * weight
                        weights[name] += weight
                if values.get("wind_direction_degrees") is not None:
                    angle = radians(float(values["wind_direction_degrees"]))
                    wind_direction_x += sin(angle) * weight
                    wind_direction_y += cos(angle) * weight
                    wind_direction_weight += weight
                segment_derived.append(
                    {"segment_id": segment["id"], "time_utc": moment.isoformat(), **values}
                )
                summary.segments_updated += 1
            activity_values = {
                name: weighted[name] / weights[name] if weights[name] else None for name in weather_columns
            }
            if wind_direction_weight:
                from math import atan2, degrees

                activity_values["wind_direction_degrees"] = (
                    degrees(atan2(wind_direction_x, wind_direction_y)) + 360.0
                ) % 360.0
            # Route-relative components are segment-dependent; average them in
            # the same moving-time-weighted manner as other run conditions.
            derived = {
                "coordinate_key": [latitude, longitude],
                "coordinate_privacy": {
                    "source_centroid_decimal_places": precision,
                    "deterministic_jitter_radius_km": privacy_radius,
                    "salt_stored_locally": True,
                },
                "location_basis": row["location_basis"],
                "location_source_activity_id": row["location_source_activity_id"],
                "location_offset_days": row["location_offset_days"],
                "location_label": row.get("location_label"),
                "cache_ids": cache_ids,
                "interpolation": "linear_at_segment_midpoint; precipitation_nearest_hour",
                "segments": segment_derived,
                "activity": activity_values,
            }
            primary_cache_id = cache_ids[0] if cache_ids else None
            connection.execute(
                """
                INSERT INTO activity_weather(
                    activity_id,weather_cache_id,derived_weather_json,
                    temperature_f,dewpoint_f,relative_humidity_percent,apparent_temperature_f,
                    wind_speed_mph,wind_gust_mph,wind_direction_degrees,precipitation_in,
                    surface_pressure_hpa,headwind_mph,tailwind_mph,crosswind_mph,weather_quality
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(activity_id) DO UPDATE SET
                    weather_cache_id=excluded.weather_cache_id,
                    derived_weather_json=excluded.derived_weather_json,
                    temperature_f=excluded.temperature_f,dewpoint_f=excluded.dewpoint_f,
                    relative_humidity_percent=excluded.relative_humidity_percent,
                    apparent_temperature_f=excluded.apparent_temperature_f,
                    wind_speed_mph=excluded.wind_speed_mph,wind_gust_mph=excluded.wind_gust_mph,
                    wind_direction_degrees=excluded.wind_direction_degrees,
                    precipitation_in=excluded.precipitation_in,surface_pressure_hpa=excluded.surface_pressure_hpa,
                    headwind_mph=excluded.headwind_mph,tailwind_mph=excluded.tailwind_mph,
                    crosswind_mph=excluded.crosswind_mph,weather_quality=excluded.weather_quality
                """,
                (
                    activity_id, primary_cache_id, json.dumps(derived),
                    activity_values["temperature_f"], activity_values["dewpoint_f"],
                    activity_values["relative_humidity_percent"], activity_values["apparent_temperature_f"],
                    activity_values["wind_speed_mph"], activity_values["wind_gust_mph"],
                    activity_values["wind_direction_degrees"], activity_values["precipitation_in"],
                    activity_values["surface_pressure_hpa"], activity_values["headwind_mph"],
                    activity_values["tailwind_mph"], activity_values["crosswind_mph"],
                    (
                        "hourly_interpolated_estimated_location"
                        if row["location_basis"] != "recorded_route_centroid"
                        else "hourly_interpolated"
                    ),
                ),
            )
        summary.activities_updated += 1
    return summary
