"""Offline terrain provenance and consistency checks for comparable pace.

Recorded altitude is not a terrain reference merely because it is present.
Map tiles are optional local HGT-format DEM files; this module never makes requests.
Raw source points are preserved, including when a correction is withheld.
"""
from collections import defaultdict
from dataclasses import replace
from contextvars import ContextVar
from functools import lru_cache
from math import cos, floor, isfinite, radians
from pathlib import Path
import re

import numpy as np

from .geo import haversine_m
from .movement import attach_elevation_deltas

VERSION = 'terrain-quality-v4-signal-surfaces'
# Engineering tolerances, not published physiological or universal GPS limits.
REVISIT_RADIUS_M = 10.
REVISIT_MINIMUM_SECONDS = 120.
REVISIT_VERTICAL_TOLERANCE_M = 6.
REVISIT_MINIMUM_PAIRS = 3
_TILE_LOOKUPS = ContextVar('terrain_tile_lookups', default=None)


def _finite(value):
    return isinstance(value, (int, float)) and isfinite(value)


def recorded_elevation_source(creator):
    """A small explicit capability registry; unknown devices stay unknown.

    'Enhanced altitude' is FIT numeric precision, not evidence of a barometer.
    A barometer-capable recorder is not a guarantee against drift, either.
    """
    name = str(creator or '').lower().replace('_', ' ')
    if re.fullmatch(r'(?:garmin )?(?:forerunner\s*|fr\s*)245(?: music)?', name):
        return 'recorded_gps_only'
    if re.fullmatch(r'(?:garmin )?(?:forerunner\s*|fr\s*)265(?:s| small)?', name):
        return 'recorded_barometer_capable'
    return 'recorded_unknown'


def assess_recorded_elevation(points, creator=None):
    """Find inconsistent return visits without requiring a history of runs.

    Multiple samples on both visits must disagree; one GPS outlier is not a
    run-level rejection. Bridges/stacked paths can also be ambiguous, so a
    rejection withholds a correction rather than declaring the terrain flat.
    """
    source = recorded_elevation_source(creator)
    known = [p for p in points if _finite(p.altitude_m)]
    buckets = defaultdict(list)
    pairs = bad = 0
    first_bad, second_bad = set(), set()
    max_difference = 0.
    origin = next((p.latitude for p in points if p.gps_valid and _finite(p.latitude)), 0.)
    longitude_scale = 111195. * max(.01, cos(radians(origin)))
    for index, point in enumerate(points):
        if not (point.gps_valid and _finite(point.latitude) and _finite(point.longitude)
                and _finite(point.altitude_m) and point.timestamp_utc):
            continue
        cell = (floor(point.latitude * 111195. / REVISIT_RADIUS_M),
                floor(point.longitude * longitude_scale / REVISIT_RADIUS_M))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for previous_index, previous in buckets.get((cell[0]+dx, cell[1]+dy), ()):
                    seconds = (point.timestamp_utc - previous.timestamp_utc).total_seconds()
                    if seconds < REVISIT_MINIMUM_SECONDS:
                        continue
                    distance = haversine_m(point.latitude, point.longitude, previous.latitude, previous.longitude)
                    if distance > REVISIT_RADIUS_M:
                        continue
                    pairs += 1
                    difference = abs(point.altitude_m - previous.altitude_m)
                    max_difference = max(max_difference, difference)
                    # Permit a generous 12% slope across the horizontal mismatch
                    # as well as the sensor tolerance.
                    if difference > REVISIT_VERTICAL_TOLERANCE_M + .12 * distance:
                        bad += 1
                        first_bad.add(previous_index)
                        second_bad.add(index)
        # Bound dense stationary sampling without discarding separate visits.
        entries = buckets[cell]
        if not entries or (point.timestamp_utc-entries[-1][1].timestamp_utc).total_seconds() >= 5:
            entries.append((index, point))
    contradicted = bad >= REVISIT_MINIMUM_PAIRS and len(first_bad) >= 2 and len(second_bad) >= 2
    consistent = pairs >= REVISIT_MINIMUM_PAIRS and bad == 0
    # A successful local loop check is not an independent terrain reference
    # for the rest of a route. GPS-only/unknown elevation still needs a map.
    usable = bool(known) and not contradicted and source == 'recorded_barometer_capable'
    status = 'rejected' if contradicted else 'supported' if usable else 'unverified' if known else 'missing'
    reason = ('Repeated visits to nearby positions have conflicting elevations.' if contradicted
        else 'Barometer-capable recorder; no contradictory return visits detected.' if usable
        else 'Return visits agree, but GPS-only or unknown elevation still needs a terrain reference.' if consistent
        else 'Recorded elevation lacks a supported source or consistent return-visit check.' if known
        else 'Recorded elevation is missing.')
    return dict(source=source, status=status, usable=usable, reason=reason,
        repeat_pairs=pairs, inconsistent_pairs=bad,
        conflicting_sample_indices=sorted(first_bad | second_bad) if contradicted else [],
        maximum_repeat_difference_m=max_difference,
        thresholds=dict(radius_m=REVISIT_RADIUS_M, minimum_seconds=REVISIT_MINIMUM_SECONDS,
            vertical_tolerance_m=REVISIT_VERTICAL_TOLERANCE_M, minimum_pairs=REVISIT_MINIMUM_PAIRS))


def tile_directory(connection, config):
    configured = config.get('elevation', {}).get('dem_directory')
    database = connection.execute('PRAGMA database_list').fetchone()[2]
    # An explicit absolute directory avoids making a website process's cwd
    # decide which private maps belong to an athlete's database.
    if configured:
        path = Path(configured).expanduser()
        if not path.is_absolute():
            raise ValueError('elevation.dem_directory must be an absolute local path')
        return path
    return Path(database).resolve().parent / 'elevation_tiles' if database else None


def tile_signature(connection, config):
    directory = tile_directory(connection, config)
    tiles = tuple((str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns)
                 for p in sorted(directory.glob('*.hgt'))) if directory and directory.is_dir() else ()
    return tiles


@lru_cache(maxsize=2)
def _read_tile(filename, size, mtime_ns):
    side = int((size // 2) ** .5)
    if side not in (1201, 3601) or size != side*side*2:
        raise ValueError('Invalid SRTM HGT tile dimensions')
    # Signed 16-bit big-endian, north-to-south rows, west-to-east columns.
    values = np.fromfile(filename, dtype='>i2').reshape(side, side)
    values.flags.writeable = False
    return values


def map_elevation(latitude, longitude, directory):
    if directory is None or not (_finite(latitude) and _finite(longitude)):
        return None
    if not (-90 < latitude < 90 and -180 <= longitude < 180):
        return None
    south, west = floor(latitude), floor(longitude)
    name = f'{"N" if south >= 0 else "S"}{abs(south):02d}{"E" if west >= 0 else "W"}{abs(west):03d}.hgt'
    path = directory / name
    lookups = _TILE_LOOKUPS.get()
    if lookups is not None and path in lookups:
        tile = lookups[path]
    else:
        try:
            stat = path.stat()
            tile = _read_tile(str(path), stat.st_size, stat.st_mtime_ns)
        except (OSError, ValueError):
            tile = None
        if lookups is not None:
            lookups[path] = tile
    if tile is None:
        return None
    row, column = (south+1-latitude)*(tile.shape[0]-1), (longitude-west)*(tile.shape[1]-1)
    top, left = int(row), int(column)
    bottom, right = min(top+1, tile.shape[0]-1), min(left+1, tile.shape[1]-1)
    cells = [int(tile[r, col]) for r, col in ((top,left),(top,right),(bottom,left),(bottom,right))]
    if -32768 in cells:  # A void is not sea level or an interpolation anchor.
        return None
    dy, dx = row-top, column-left
    return (cells[0]*(1-dx)+cells[1]*dx)*(1-dy)+(cells[2]*(1-dx)+cells[3]*dx)*dy


def attach_surface_deltas(points, intervals, heights, sources, window_meters):
    """Smooth each contiguous source separately; never join different datums.

    Gaps, source switches and missing movement intervals are boundaries, not
    synthetic climbs. A constant offset within either source cancels exactly.
    """
    if len(points) != len(heights) or len(points) != len(sources):
        raise ValueError('Each point requires a terrain height and source')
    for interval in intervals:
        interval.elevation_delta_m = None
    by_index = {interval.index: interval for interval in intervals}
    start = 0
    while start < len(points):
        if not _finite(heights[start]) or sources[start] is None:
            start += 1
            continue
        stop = start+1
        while (stop < len(points) and _finite(heights[stop])
               and sources[stop] == sources[start] and stop-1 in by_index):
            stop += 1
        selected = [replace(p, altitude_m=h) for p, h in zip(points[start:stop], heights[start:stop])]
        local = [replace(by_index[i], index=i-start) for i in range(start, stop-1)]
        attach_elevation_deltas(selected, local, window_meters)
        for interval in local:
            by_index[interval.index+start].elevation_delta_m = interval.elevation_delta_m
        start = stop


def prepare_terrain(points, intervals, config, creator=None, directory=None):
    """Map-default terrain with local provisional recorded-surface exceptions."""
    from .surface_mismatch import surface_candidates
    assessment = assess_recorded_elevation(points, creator)
    if directory is not None and not directory.is_dir():
        directory = None
    # Resolve each tile once for this run, not once per GPS second. No tile
    # availability result survives the call; new/changed maps are still seen.
    token = _TILE_LOOKUPS.set({})
    try:
        mapped = [map_elevation(p.latitude, p.longitude, directory) if p.gps_valid else None for p in points]
    finally:
        _TILE_LOOKUPS.reset(token)
    map_fraction = sum(v is not None for v in mapped)/len(points) if points else 0.
    barometric = assessment['source'] == 'recorded_barometer_capable'
    # Preserve the released preference for good barometric evidence. Ground
    # tiles otherwise remain the default, not an optional bridge-height lookup.
    map_default = map_fraction >= .9 and not barometric
    heights = list(mapped) if map_default else [None]*len(points)
    sources = ['local_dem' if _finite(v) else None for v in heights]
    if map_default:
        assessment = {**assessment, 'recorded_status': assessment['status'],
            'source': 'local_dem', 'status': 'supported', 'usable': True,
            'surface': 'bare_earth', 'bridge_deck_supported': False,
            'reason': 'Elevation from local ground-terrain tiles. Bridge-deck height is not guaranteed.'}

    # Repeat disagreement is local: withhold both conflicting visits and their
    # smoothing neighbourhood, not every later sample in a partial recording.
    distance = [0.]*len(points)
    by_index = {i.index:i for i in intervals}
    for i in range(len(points)-1):
        distance[i+1] = distance[i] + (by_index[i].distance_m if i in by_index else 0.)
    bad = np.zeros(len(points),dtype=bool)
    half = float(config['elevation']['smoothing_window_meters'])/2
    travel = np.asarray(distance)
    for index in assessment['conflicting_sample_indices']:
        bad |= abs(travel-travel[index]) <= half
    recorded_valid = [p.gps_valid and _finite(p.altitude_m) and not bad[i] for i,p in enumerate(points)]

    if barometric:
        for i,p in enumerate(points):
            if recorded_valid[i]:
                heights[i],sources[i] = p.altitude_m,'recorded_barometer_capable'
        # A known local conflict cannot be resolved by substituting the ground
        # underneath an unknown surface. Keep those sections missing.
        assessment = {**assessment, 'usable':any(s is not None for s in sources)}
        if assessment['conflicting_sample_indices']:
            assessment.update(status='partial', calibration_eligible=False,
                reason='Conflicting return elevations withheld locally; other recorded sections retained.')

    comparison = surface_candidates(points, intervals, mapped, config) if not barometric else {'mask':[False]*len(points)}
    chosen = 0
    for i,candidate in enumerate(comparison['mask']):
        if not candidate:
            continue
        # Even below the whole-run map-coverage threshold, usable GPS sections
        # can be compared. A missing start does not invalidate later terrain.
        if recorded_valid[i]:
            heights[i],sources[i] = points[i].altitude_m,'recorded_surface'
            chosen += 1
        else:
            heights[i],sources[i] = None,None

    attach_surface_deltas(points, intervals, heights, sources, float(config['elevation']['smoothing_window_meters']))
    withheld_grade = 0
    for interval in intervals:
        i = interval.index
        if (i+1<len(sources) and sources[i] == sources[i+1] == 'recorded_surface'
                and interval.elevation_delta_m is not None
                and (interval.distance_m<=0 or abs(interval.elevation_delta_m)*100
                     > float(config['elevation']['maximum_plausible_grade_percent'])*interval.distance_m)):
            interval.elevation_delta_m = None
            withheld_grade += 1
    known = any(i.elevation_delta_m is not None for i in intervals)
    if chosen:
        assessment.update(source='mixed_recorded_surface' if any(s=='local_dem' for s in sources) else 'recorded_surface',
            status='provisional', usable=known, surface='recorded_surface_and_ground',
            surface_profile_uncertain=True, bridge_deck_supported=False,
            calibration_eligible=False,
            reason='Local surface mismatch: recorded grades retained provisionally; no bridge catalogue, endpoint interpolation or altitude offset.')
    elif not known:
        assessment.update(usable=False, status='rejected' if assessment['conflicting_sample_indices'] else 'unverified')
        if assessment['conflicting_sample_indices']:
            assessment['reason'] = 'Conflicting return elevations withheld locally; no usable recorded grade remains. Ground maps cannot resolve different bridge or road levels.'
    return {**assessment, 'map_coverage_fraction':map_fraction, 'version':VERSION,
        'surface_recorded_points':chosen, 'surface_candidate_points':sum(comparison['mask']),
        'recorded_conflict_points':int(sum(bad)),
        'local_conflict_withheld_points':sum(bool(bad[i]) and sources[i] is None for i in range(len(points))),
        'surface_implausible_grade_intervals':withheld_grade,
        'surface_detection':'Signal-only provisional inference; cannot prove which surface is correct.'}
