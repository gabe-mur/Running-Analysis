"""Offline endpoint-interpolation experiment; never used by the live pipeline."""
from dataclasses import replace

from .bridge_geometry import (
    BridgePath, MATCH_RADIUS_M, MINIMUM_MATCHED_POINTS, MINIMUM_MATCHED_SPAN_M,
    _finite, _nearest, _projector, bridge_paths_from_osm,
)
from .elevation_quality import assess_recorded_elevation
from .movement import attach_elevation_deltas
from math import hypot

VERSION = 'bridge-terrain-prototype-v1'


def prototype_bridge_profile(points, mapped, paths, elevation_lookup, *, creator=None,
                             radius_m=MATCH_RADIUS_M):
    """Return the approximate endpoint profile and a separate guarded profile.

    The guarded profile retains supported barometric data. Otherwise matched
    bridge/underpass ambiguity is missing terrain, not inferred flat running.
    Each bridge estimate uses position along its geometry, so turning around
    halfway through does not invent a new endpoint at the turnaround.
    """
    if len(points) != len(mapped):
        raise ValueError('Every source point needs a corresponding map sample')
    if radius_m <= 0:
        raise ValueError('Positive geometry matching radius required')
    approximate = list(mapped)
    estimated = [False]*len(points)
    matched = [False]*len(points)
    nearest = [float('inf')]*len(points)
    candidates = []
    for path in paths:
        project = _projector(path.coordinates)
        shape = [project(*p) for p in path.coordinates]
        cumulative = [0.]
        for a, b in zip(shape, shape[1:]):
            cumulative.append(cumulative[-1]+hypot(b[0]-a[0], b[1]-a[1]))
        if cumulative[-1] <= 0:
            continue
        left, right = elevation_lookup(*path.coordinates[0]), elevation_lookup(*path.coordinates[-1])
        anchors = _finite(left) and _finite(right)
        hits = []
        x0, x1 = min(p[0] for p in shape)-radius_m, max(p[0] for p in shape)+radius_m
        y0, y1 = min(p[1] for p in shape)-radius_m, max(p[1] for p in shape)+radius_m
        for index, point in enumerate(points):
            if not point.gps_valid or not (_finite(point.latitude) and _finite(point.longitude)):
                continue
            position = project(point.latitude, point.longitude)
            if not (x0 <= position[0] <= x1 and y0 <= position[1] <= y1):
                continue
            distance, along = _nearest(position, shape, cumulative)
            if distance <= radius_m:
                matched[index] = True
                hits.append((index, along, distance))
        if hits:
            span = max(v for _, v, _ in hits)-min(v for _, v, _ in hits)
            follows = len(hits) >= MINIMUM_MATCHED_POINTS and span >= MINIMUM_MATCHED_SPAN_M
            if follows:
                for index, along, distance in hits:
                    if distance < nearest[index]:
                        nearest[index] = distance
                        # Missing reference/anchors stay missing, never zero.
                        approximate[index] = (left+(right-left)*along/cumulative[-1]
                            if anchors and _finite(mapped[index]) else None)
                        estimated[index] = approximate[index] is not None
            candidates.append(dict(way_ids=list(path.way_ids), name=path.name,
                length_m=cumulative[-1], matched_indices=[i for i, _, _ in hits],
                matched_span_m=span, start_height_m=left, end_height_m=right,
                endpoint_estimate_available=anchors,
                follows_geometry=follows,
                status=('proximity_only' if not follows else
                    'approximate_endpoint_interpolation' if anchors else 'missing_anchors'),
                measured_deck_height=False, anchors_are_assumed_ground_contact=True))
    quality = assess_recorded_elevation(points, creator)
    if quality['usable']:
        guarded = [p.altitude_m if _finite(p.altitude_m) else None for p in points]
        sources = ['recorded_barometer' if v is not None else 'missing' for v in guarded]
    else:
        guarded = [None if match or not _finite(value) else value for match, value in zip(matched, mapped)]
        sources = ['bridge_level_uncertain' if match else 'ground_map' if value is not None else 'missing'
                   for match, value in zip(matched, guarded)]
    return dict(version=VERSION, endpoint_estimate=approximate, guarded_elevations=guarded,
        bridge_mask=matched, estimated_mask=estimated, point_sources=sources,
        recorded_quality=quality, bridges=candidates,
        limitations=['Geometry proximity alone cannot distinguish travelling over versus under a bridge.',
            'Endpoint interpolation removes false ground dips but cannot reconstruct an unknown deck hump.',
            'Missing OSM bridge tags or incomplete map coverage are not proof that terrain is ordinary ground.',
            'A connected path endpoint is not independently verified ground contact.'])


def attach_prototype_deltas(points, intervals, elevations, window_meters):
    """Smooth valid chunks separately; never leak filled bridge gaps into grades."""
    if len(points) != len(elevations):
        raise ValueError('Every point needs a corresponding elevation')
    for interval in intervals:
        interval.elevation_delta_m = None
    selected = [replace(p, altitude_m=v if _finite(v) else None) for p, v in zip(points, elevations)]
    by_index = {interval.index: interval for interval in intervals}
    start = 0
    while start < len(selected):
        if selected[start].altitude_m is None:
            start += 1
            continue
        stop = start+1
        while stop < len(selected) and selected[stop].altitude_m is not None:
            stop += 1
        local = [replace(by_index[i], index=i-start) for i in range(start, stop-1) if i in by_index]
        attach_elevation_deltas(selected[start:stop], local, window_meters)
        for interval in local:
            by_index[interval.index+start].elevation_delta_m = interval.elevation_delta_m
        start = stop
