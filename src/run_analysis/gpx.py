"""Recorded GPX tracks, including common HR/cadence extensions.

A planned route is not a workout. Never manufacture timestamps, HR, laps,
timer events or watch-zone settings to turn one into an activity.
"""
from pathlib import Path
import math
from statistics import fmean
import xml.etree.ElementTree as ET

from .activity_assembly import finish_activity
from .cadence import GPX_UNSPECIFIED_CADENCE_SOURCE
from .geo import haversine_m
from .models import Activity, Lap, ParsedTCX, Trackpoint
from .tcx import local_name
from .timeparse import parse_datetime

GPX_NAMESPACES = {'', 'http://www.topografix.com/GPX/1/0', 'http://www.topografix.com/GPX/1/1'}
EXTENSION_NAMESPACES = {
    '', 'http://www.garmin.com/xmlschemas/TrackPointExtension/v1',
    'http://www.garmin.com/xmlschemas/TrackPointExtension/v2',
    'http://www.cluetrust.com/XML/GPXDATA/1/0',
}


class _NoDTD(ET.TreeBuilder):
    def doctype(self, name, pubid, system):
        raise ValueError('GPX document types and entities are not supported')


def _namespace(tag):
    return tag[1:].split('}', 1)[0] if tag.startswith('{') else ''


def _children(element, name):
    return [child for child in element if local_name(child.tag) == name
            and _namespace(child.tag) in GPX_NAMESPACES]


def _text(element, name):
    children = _children(element, name)
    return children[0].text.strip() if children and children[0].text else None


def _number(value, flags, name, *, minimum=None, maximum=None):
    if value is None:
        return None
    try:
        number = float(value)
        if (not math.isfinite(number) or (minimum is not None and number < minimum)
                or (maximum is not None and number > maximum)):
            raise ValueError
        return number
    except (ValueError, TypeError):
        flags.append(f'invalid_gpx_{name}')
        return None


def _extension(element, names):
    for container in _children(element, 'extensions'):
        for child in container.iter():
            if local_name(child.tag) in names and _namespace(child.tag) in EXTENSION_NAMESPACES:
                return child.text.strip() if child.text else None
    return None


def _sport(value, creator, warnings):
    # GPX has no standard numeric sport enum. Only interpret the known Strava
    # export codes in Strava files; retain unknown types instead of guessing.
    if creator and creator.casefold().startswith('stravagpx'):
        value = {'9': 'running', '53': 'running', '1': 'cycling',
                 '4': 'hiking', '10': 'walking'}.get(value, value)
    normalized = (value or '').casefold().replace(' ', '').replace('_', '')
    sport = {'run': 'Running', 'running': 'Running', 'trailrun': 'Running',
             'trailrunning': 'Running', 'virtualrun': 'Running',
             'cycling': 'Biking', 'biking': 'Biking', 'ride': 'Biking',
             'walk': 'Walking', 'walking': 'Walking', 'hike': 'Hiking', 'hiking': 'Hiking'}.get(normalized)
    if sport:
        return sport
    warnings.append('gpx_sport_unspecified' if not value else 'gpx_sport_unrecognized')
    return value or 'Unknown'


def parse_gpx(path: str | Path, default_timezone: str = 'America/New_York') -> ParsedTCX:
    source = Path(path)
    if source.stat().st_size > 50 * 1024 * 1024:
        raise ValueError('GPX file exceeds 50 MB')
    root = ET.parse(source, parser=ET.XMLParser(target=_NoDTD())).getroot()
    if local_name(root.tag) != 'gpx' or _namespace(root.tag) not in GPX_NAMESPACES:
        raise ValueError('Expected a GPX activity file')
    namespaces = sorted({_namespace(element.tag) for element in root.iter()} - {''})
    creator = root.attrib.get('creator') or None
    activities = []
    for track in _children(root, 'trk'):
        warnings = []
        points = []
        segment_seconds = 0.0
        gps_distance = 0.0
        sport = _sport(_text(track, 'type'), creator, warnings)
        for segment_index, segment in enumerate(_children(track, 'trkseg')):
            segment_points = []
            for index, element in enumerate(_children(segment, 'trkpt')):
                flags = ['gpx_segment_start'] if points and index == 0 else []
                latitude = _number(element.get('lat'), flags, 'latitude', minimum=-90, maximum=90)
                longitude = _number(element.get('lon'), flags, 'longitude', minimum=-180, maximum=180)
                timestamp = parse_datetime(_text(element, 'time'), flags, 'gpx_point_time')
                if timestamp is None:
                    flags.append('gpx_point_time_missing')
                valid = latitude is not None and longitude is not None and (latitude, longitude) != (0, 0)
                hr = _number(_extension(element, {'hr', 'heartrate'}), flags, 'heart_rate', minimum=1, maximum=300)
                cadence = _number(_extension(element, {'cad', 'cadence'}), flags, 'cadence', minimum=0, maximum=1000)
                point = Trackpoint(
                    lap_index=0, track_index=segment_index, point_index=index,
                    timestamp_utc=timestamp, latitude=latitude, longitude=longitude, gps_valid=valid,
                    altitude_m=_number(_text(element, 'ele'), flags, 'altitude'),
                    distance_m=None, heart_rate_bpm=round(hr) if hr is not None else None,
                    cadence=round(cadence) if cadence is not None else None,
                    run_cadence=None,
                    cadence_source=GPX_UNSPECIFIED_CADENCE_SOURCE if cadence is not None else None,
                    speed_mps=_number(_extension(element, {'speed'}), flags, 'speed', minimum=0),
                    parse_flags=flags,
                )
                points.append(point)
                segment_points.append(point)
            times = [p.timestamp_utc for p in segment_points if p.timestamp_utc is not None]
            if len(times) > 1:
                segment_seconds += (max(times) - min(times)).total_seconds()
            for first, second in zip(segment_points, segment_points[1:]):
                if (first.timestamp_utc is not None and second.timestamp_utc is not None
                        and second.timestamp_utc > first.timestamp_utc and first.gps_valid and second.gps_valid):
                    gps_distance += haversine_m(first.latitude, first.longitude, second.latitude, second.longitude)
        if not any(a.timestamp_utc and b.timestamp_utc and b.timestamp_utc > a.timestamp_utc
                   and 'gpx_segment_start' not in b.parse_flags for a, b in zip(points, points[1:])):
            raise ValueError('GPX needs a recorded track with timestamps; routes alone are not workouts')
        # One derived summary, not a claim that GPX contains a recorded lap.
        lap = Lap(0, None, segment_seconds, None, None, None, None, None, None, None)
        activity = Activity(None, sport, _text(track, 'desc') or _text(track, 'name'),
                            creator, namespaces, [lap], points, parse_warnings=warnings)
        finish_activity(activity, default_timezone)
        activity.activity_id = activity.start_time_utc.isoformat() if activity.start_time_utc else None
        activity.total_distance_m = gps_distance
        activity.distance_source = 'gps'
        hrs = [p.heart_rate_bpm for p in points if p.heart_rate_bpm is not None]
        activity.average_hr_bpm = fmean(hrs) if hrs else None
        activity.maximum_hr_bpm = max(hrs) if hrs else None
        warnings = activity.parse_warnings + [flag for p in points for flag in p.parse_flags
                                               if flag != 'gpx_segment_start']
        if not hrs:
            warnings.append('gpx_heart_rate_missing')
        if any(p.cadence is not None for p in points):
            warnings.append('gpx_cadence_units_unspecified')
        warnings.append('gpx_distance_derived_from_coordinates')
        activity.parse_warnings = sorted(set(warnings))
        activities.append(activity)
    if not activities:
        raise ValueError('GPX needs a recorded track with timestamps; routes alone are not workouts')
    return ParsedTCX(activities, namespaces,
                     sorted({w for activity in activities for w in activity.parse_warnings}), 'xml_declared')
