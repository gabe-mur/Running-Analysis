"""Namespace-tolerant, non-mutating TCX parsing."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from zoneinfo import ZoneInfo
import xml.etree.ElementTree as ET

from .models import Activity, Lap, ParsedTCX, Trackpoint


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _children(parent: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in parent if local_name(child.tag) == name]


def _child(parent: ET.Element, name: str) -> ET.Element | None:
    return next((child for child in parent if local_name(child.tag) == name), None)


def _descendant(parent: ET.Element, name: str) -> ET.Element | None:
    return next((element for element in parent.iter() if local_name(element.tag) == name), None)


def _text(element: ET.Element | None) -> str | None:
    if element is None or element.text is None:
        return None
    value = element.text.strip()
    return value or None


def _float(element: ET.Element | None, flags: list[str], field_name: str) -> float | None:
    value = _text(element)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        flags.append(f"malformed_{field_name}")
        return None


def _int(element: ET.Element | None, flags: list[str], field_name: str) -> int | None:
    value = _float(element, flags, field_name)
    return round(value) if value is not None else None


def parse_datetime(value: str | None, warnings: list[str], field_name: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        warnings.append(f"malformed_{field_name}")
        return None
    if parsed.tzinfo is None:
        # TCX track and lap timestamps are UTC. Some Smashrun Activity/Id values
        # omit the suffix while retaining the UTC clock time.
        warnings.append(f"naive_{field_name}_treated_as_utc")
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _load_xml(path: Path) -> tuple[ET.Element, list[str], str, list[str]]:
    raw = path.read_bytes()
    warnings: list[str] = []
    encoding = "utf-8"
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as utf8_error:
        try:
            # Five observed Smashrun files declare UTF-8 but contain cp1252
            # punctuation in Notes. Decode in memory only; source files remain
            # byte-for-byte untouched.
            root = ET.fromstring(raw.decode("cp1252"))
            encoding = "cp1252_fallback"
            warnings.append("declared_utf8_repaired_as_cp1252")
        except (UnicodeDecodeError, ET.ParseError) as fallback_error:
            raise ValueError(
                f"Invalid XML ({utf8_error}); cp1252 fallback failed ({fallback_error})"
            ) from fallback_error

    namespaces: list[str] = []
    try:
        namespaces = sorted({uri for _, (_, uri) in ET.iterparse(path, events=("start-ns",))})
    except ET.ParseError:
        # Namespace declarations are already represented in expanded element
        # names after the successful in-memory fallback.
        namespaces = sorted(
            {
                element.tag[1:].split("}", 1)[0]
                for element in root.iter()
                if element.tag.startswith("{")
            }
        )
    return root, namespaces, encoding, warnings


def _parse_lap(element: ET.Element, index: int, warnings: list[str]) -> Lap:
    flags: list[str] = []
    start = parse_datetime(element.attrib.get("StartTime"), warnings, "lap_start_time")
    avg = _child(element, "AverageHeartRateBpm")
    maximum = _child(element, "MaximumHeartRateBpm")
    return Lap(
        lap_index=index,
        start_time_utc=start,
        total_time_s=_float(_child(element, "TotalTimeSeconds"), flags, "lap_time"),
        distance_m=_float(_child(element, "DistanceMeters"), flags, "lap_distance"),
        calories=_int(_child(element, "Calories"), flags, "calories"),
        average_hr_bpm=_int(_descendant(avg, "Value") if avg is not None else None, flags, "average_hr"),
        maximum_hr_bpm=_int(_descendant(maximum, "Value") if maximum is not None else None, flags, "maximum_hr"),
        maximum_speed_mps=_float(_child(element, "MaximumSpeed"), flags, "maximum_speed"),
        intensity=_text(_child(element, "Intensity")),
        trigger_method=_text(_child(element, "TriggerMethod")),
    )


def _parse_trackpoint(
    element: ET.Element,
    lap_index: int,
    track_index: int,
    point_index: int,
    warnings: list[str],
) -> Trackpoint:
    flags: list[str] = []
    timestamp = parse_datetime(_text(_child(element, "Time")), flags, "trackpoint_time")
    position = _child(element, "Position")
    latitude = _float(_child(position, "LatitudeDegrees") if position is not None else None, flags, "latitude")
    longitude = _float(_child(position, "LongitudeDegrees") if position is not None else None, flags, "longitude")
    in_range = (
        latitude is not None
        and longitude is not None
        and -90 <= latitude <= 90
        and -180 <= longitude <= 180
    )
    zero_pair = in_range and latitude == 0 and longitude == 0
    gps_valid = bool(in_range and not zero_pair)
    if zero_pair:
        flags.append("zero_zero_position")
    elif (latitude is None) != (longitude is None):
        flags.append("incomplete_position")
    elif latitude is not None and not in_range:
        flags.append("out_of_range_position")

    hr = _child(element, "HeartRateBpm")
    direct_cadence = _int(_child(element, "Cadence"), flags, "cadence")
    run_cadence = _int(_descendant(element, "RunCadence"), flags, "run_cadence")
    cadence = run_cadence if run_cadence is not None else direct_cadence
    cadence_source = "run_cadence_extension" if run_cadence is not None else "cadence" if direct_cadence is not None else None
    warnings.extend(flag for flag in flags if flag.startswith("malformed_"))
    return Trackpoint(
        lap_index=lap_index,
        track_index=track_index,
        point_index=point_index,
        timestamp_utc=timestamp,
        latitude=latitude,
        longitude=longitude,
        gps_valid=gps_valid,
        altitude_m=_float(_child(element, "AltitudeMeters"), flags, "altitude"),
        distance_m=_float(_child(element, "DistanceMeters"), flags, "distance"),
        heart_rate_bpm=_int(_descendant(hr, "Value") if hr is not None else None, flags, "heart_rate"),
        cadence=cadence,
        run_cadence=run_cadence,
        cadence_source=cadence_source,
        speed_mps=_float(_descendant(element, "Speed"), flags, "speed"),
        parse_flags=sorted(set(flags)),
    )


COMPLETE_SENSOR_COVERAGE = 0.95


def _quality(prefix: str, present: int, total: int) -> str:
    """Describe material sensor coverage, not literal point-for-point perfection."""
    if total == 0 or present == 0:
        return f"{prefix}_missing"
    if present / total >= COMPLETE_SENSOR_COVERAGE:
        return f"{prefix}_complete"
    return f"{prefix}_partial"


def _creator_name(root: ET.Element, activity_element: ET.Element) -> str | None:
    if root.attrib.get("creator"):
        return root.attrib["creator"].strip() or None
    creator = _child(activity_element, "Creator")
    return _text(_descendant(creator, "Name")) if creator is not None else None


def _finish_activity(activity: Activity, default_timezone: str) -> None:
    points = activity.trackpoints
    timestamps = [point.timestamp_utc for point in points if point.timestamp_utc is not None]
    if timestamps:
        activity.start_time_utc = min(timestamps)
        if len(timestamps) > 1:
            activity.total_elapsed_time_s = (max(timestamps) - min(timestamps)).total_seconds()
    elif any(lap.start_time_utc for lap in activity.laps):
        activity.start_time_utc = min(lap.start_time_utc for lap in activity.laps if lap.start_time_utc)
        activity.parse_warnings.append("start_time_from_lap_no_trackpoint_time")
    else:
        activity.start_time_utc = parse_datetime(activity.activity_id, activity.parse_warnings, "activity_id")
        if activity.start_time_utc:
            activity.parse_warnings.append("start_time_from_activity_id")

    lap_times = [lap.total_time_s for lap in activity.laps if lap.total_time_s is not None]
    lap_distances = [lap.distance_m for lap in activity.laps if lap.distance_m is not None]
    activity.lap_recorded_time_s = sum(lap_times) if lap_times else None
    activity.total_distance_m = sum(lap_distances) if lap_distances else None
    lap_calories = [lap.calories for lap in activity.laps if lap.calories is not None]
    activity.calories = sum(lap_calories) if lap_calories else None
    weighted_hr = [
        (lap.average_hr_bpm, lap.total_time_s)
        for lap in activity.laps
        if lap.average_hr_bpm is not None and lap.total_time_s is not None and lap.total_time_s > 0
    ]
    if weighted_hr:
        activity.average_hr_bpm = sum(hr * seconds for hr, seconds in weighted_hr) / sum(
            seconds for _, seconds in weighted_hr
        )
    else:
        summary_values = [lap.average_hr_bpm for lap in activity.laps if lap.average_hr_bpm is not None]
        activity.average_hr_bpm = fmean(summary_values) if summary_values else None
    maxima = [lap.maximum_hr_bpm for lap in activity.laps if lap.maximum_hr_bpm is not None]
    activity.maximum_hr_bpm = max(maxima) if maxima else None

    count = len(points)
    activity.gps_quality = _quality("gps", sum(point.gps_valid for point in points), count)
    activity.hr_quality = _quality("hr", sum(point.heart_rate_bpm is not None for point in points), count)
    activity.elevation_quality = _quality("elevation", sum(point.altitude_m is not None for point in points), count)
    activity.cadence_quality = _quality("cadence", sum(point.cadence is not None for point in points), count)
    device_distance_present = bool(lap_distances) or any(point.distance_m is not None for point in points)
    activity.distance_source = "device" if device_distance_present else "unknown"

    valid_positions = [(point.latitude, point.longitude) for point in points if point.gps_valid]
    timezone_source = "configured_default"
    if valid_positions:
        mean_lat = fmean(position[0] for position in valid_positions if position[0] is not None)
        mean_lon = fmean(position[1] for position in valid_positions if position[1] is not None)
        if 40.45 <= mean_lat <= 41.0 and -74.30 <= mean_lon <= -73.65:
            timezone_source = "gps_nyc"
        else:
            activity.parse_warnings.append("timezone_default_used_outside_nyc")
    else:
        activity.parse_warnings.append("timezone_default_used_without_gps")
    activity.timezone_name = default_timezone
    activity.timezone_source = timezone_source
    if activity.start_time_utc:
        activity.start_time_local = activity.start_time_utc.astimezone(ZoneInfo(default_timezone))
    activity.parse_warnings = sorted(set(activity.parse_warnings))


def parse_tcx(path: str | Path, default_timezone: str = "America/New_York") -> ParsedTCX:
    source_path = Path(path)
    root, namespaces, encoding, file_warnings = _load_xml(source_path)
    activities: list[Activity] = []
    activity_elements = [element for element in root.iter() if local_name(element.tag) == "Activity"]
    for activity_element in activity_elements:
        warnings = list(file_warnings)
        laps: list[Lap] = []
        trackpoints: list[Trackpoint] = []
        for lap_index, lap_element in enumerate(_children(activity_element, "Lap")):
            laps.append(_parse_lap(lap_element, lap_index, warnings))
            for track_index, track in enumerate(_children(lap_element, "Track")):
                for point_index, point in enumerate(_children(track, "Trackpoint")):
                    trackpoints.append(
                        _parse_trackpoint(point, lap_index, track_index, point_index, warnings)
                    )
        activity = Activity(
            activity_id=_text(_child(activity_element, "Id")),
            sport=activity_element.attrib.get("Sport", "Unknown"),
            notes=_text(_child(activity_element, "Notes")),
            creator=_creator_name(root, activity_element),
            namespaces=namespaces,
            laps=laps,
            trackpoints=trackpoints,
            parse_warnings=warnings,
        )
        _finish_activity(activity, default_timezone)
        activities.append(activity)
    return ParsedTCX(activities, namespaces, sorted(set(file_warnings)), encoding)
