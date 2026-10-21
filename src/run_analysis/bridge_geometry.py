"""Offline global OSM bridge geometry; no regional rules or height interpolation."""
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
import json
from math import cos, hypot, isfinite, radians
from pathlib import Path

from .geo import EARTH_RADIUS_M


# Geometric matching tolerances, not claims of GPS accuracy or deck level.
MATCH_RADIUS_M = 25.
MINIMUM_MATCHED_SPAN_M = 50.
MINIMUM_MATCHED_POINTS = 3


def _finite(value):
    return isinstance(value, (int, float)) and isfinite(value)


@dataclass(frozen=True)
class BridgePath:
    way_ids: tuple[int, ...]
    name: str
    coordinates: tuple[tuple[float, float], ...]  # latitude, longitude


def bridge_paths_from_osm(payloads):
    """Join unbranched, connected bridge road/path ways, not outline polygons.

    Nodes must actually connect; nearby parallel decks are not merged. Branches
    terminate a chain and remain ambiguous ground-contact anchors. Separate
    highway/layer values remain separate. Missing geometry is reported.
    """
    nodes, ways = {}, {}
    for payload in payloads:
        for item in payload.get('elements', []):
            if item.get('type') == 'node':
                lat, lon = item.get('lat'), item.get('lon')
                if _finite(lat) and _finite(lon) and -90 < lat < 90 and -180 <= lon <= 180:
                    nodes[item['id']] = (lat, lon)
            elif item.get('type') == 'way':
                ways[item['id']] = item
    groups = defaultdict(dict)
    skipped = []
    for identifier, way in sorted(ways.items()):
        tags = way.get('tags', {})
        if not tags.get('highway') or tags.get('bridge') in (None, 'no'):
            continue
        ids = way.get('nodes', [])
        if len(ids) < 2 or ids[0] == ids[-1] or any(n not in nodes for n in ids):
            skipped.append(identifier)
            continue
        groups[(tags['highway'], tags.get('layer', ''))][identifier] = way
    paths = []
    for group in groups.values():
        adjacency = defaultdict(list)
        for identifier, way in group.items():
            for node in (way['nodes'][0], way['nodes'][-1]):
                adjacency[node].append(identifier)
        used = set()

        def walk(start, identifier):
            current = start
            collected, identifiers, names = [], [], set()
            while identifier not in used:
                used.add(identifier)
                way = group[identifier]
                ids = list(way['nodes'])
                if ids[0] != current:
                    ids.reverse()
                collected.extend(ids if not collected else ids[1:])
                identifiers.append(identifier)
                if way.get('tags', {}).get('name'):
                    names.add(way['tags']['name'])
                current = ids[-1]
                following = [i for i in adjacency[current] if i not in used]
                if len(adjacency[current]) != 2 or not following:
                    break
                identifier = following[0]
            if collected[0] == collected[-1]:
                skipped.extend(identifiers)  # No entry/exit for a closed loop.
            else:
                paths.append(BridgePath(tuple(identifiers), ' / '.join(sorted(names)) or 'Unnamed bridge',
                    tuple(nodes[n] for n in collected)))

        for node in sorted(adjacency):
            if len(adjacency[node]) != 2:
                for identifier in sorted(adjacency[node]):
                    if identifier not in used:
                        walk(node, identifier)
        for identifier, way in sorted(group.items()):
            if identifier not in used:
                walk(way['nodes'][0], identifier)
    return paths, dict(skipped_way_ids=sorted(skipped), paths=len(paths),
        attribution='© OpenStreetMap contributors; ODbL; https://www.openstreetmap.org/copyright')


def _projector(coordinates):
    latitude = sum(p[0] for p in coordinates) / len(coordinates)
    longitude = coordinates[0][1]
    scale = EARTH_RADIUS_M * radians(1.)

    def project(lat, lon):
        # Wrap across the dateline; local bridge projection, not whole-world XY.
        delta = (lon-longitude+180.) % 360. - 180.
        return delta*scale*cos(radians(latitude)), (lat-latitude)*scale
    return project


def _nearest(point, shape, cumulative):
    best = (float('inf'), 0.)
    for a, b, start, stop in zip(shape, shape[1:], cumulative, cumulative[1:]):
        length = stop-start
        if length <= 0:
            continue
        dx, dy = b[0]-a[0], b[1]-a[1]
        fraction = max(0., min(1., ((point[0]-a[0])*dx+(point[1]-a[1])*dy)/(length*length)))
        distance = hypot(point[0]-a[0]-fraction*dx, point[1]-a[1]-fraction*dy)
        if distance < best[0]:
            best = distance, start+fraction*length
    return best



def match_bridge_paths(points, paths, *, radius_m=MATCH_RADIUS_M):
    """Return proximity and along-structure masks, not inferred deck heights.

    An across-road underpass touch is proximity-only. Following a path remains
    a geometric inference, not proof of the vertical level or complete coverage.
    """
    if radius_m <= 0:
        raise ValueError('Positive geometry matching radius required')
    nearby = [False] * len(points)
    following = [False] * len(points)
    matched_paths = []
    gps = [(i, p) for i, p in enumerate(points)
           if p.gps_valid and _finite(p.latitude) and _finite(p.longitude)]
    if not gps:
        return dict(nearby=nearby, following=following, paths=matched_paths)
    latitudes = [p.latitude for _, p in gps]
    longitudes = [p.longitude for _, p in gps]
    route_bounds = min(latitudes), max(latitudes), min(longitudes), max(longitudes)
    for path in paths:
        # Cheap geographic rejection before scanning dense activity samples.
        # Skip this optimization at the dateline; the wrapped projection below
        # remains the source of geometric matches there.
        lats, lons = zip(*path.coordinates)
        margin_lat = radius_m / 111195.
        margin_lon = margin_lat / max(.01, abs(cos(radians(sum(lats)/len(lats)))))
        if max(lons)-min(lons) < 180 and route_bounds[3]-route_bounds[2] < 180:
            if (route_bounds[1] < min(lats)-margin_lat or route_bounds[0] > max(lats)+margin_lat
                    or route_bounds[3] < min(lons)-margin_lon or route_bounds[2] > max(lons)+margin_lon):
                continue
        project = _projector(path.coordinates)
        shape = [project(*p) for p in path.coordinates]
        cumulative = [0.]
        for a, b in zip(shape, shape[1:]):
            cumulative.append(cumulative[-1] + hypot(b[0]-a[0], b[1]-a[1]))
        if cumulative[-1] <= 0:
            continue
        xs, ys = [p[0] for p in shape], [p[1] for p in shape]
        bounds = min(xs)-radius_m, max(xs)+radius_m, min(ys)-radius_m, max(ys)+radius_m
        hits = []
        for index, point in gps:
            xy = project(point.latitude, point.longitude)
            if not (bounds[0] <= xy[0] <= bounds[1] and bounds[2] <= xy[1] <= bounds[3]):
                continue
            distance, along = _nearest(xy, shape, cumulative)
            if distance <= radius_m:
                nearby[index] = True
                hits.append((index, along))
        if not hits:
            continue
        span = max(v for _, v in hits)-min(v for _, v in hits)
        follows = len(hits) >= MINIMUM_MATCHED_POINTS and span >= MINIMUM_MATCHED_SPAN_M
        if follows:
            for index, _ in hits:
                following[index] = True
        matched_paths.append(dict(way_ids=list(path.way_ids), follows_geometry=follows,
                                  matched_span_m=span, matched_points=len(hits)))
    return dict(nearby=nearby, following=following, paths=matched_paths)


def bridge_geometry_directory(connection, config):
    configured = config.get('elevation', {}).get('bridge_geometry_directory')
    if configured:
        path = Path(configured).expanduser()
        if not path.is_absolute():
            raise ValueError('elevation.bridge_geometry_directory must be an absolute local path')
        return path
    database = connection.execute('PRAGMA database_list').fetchone()[2]
    return Path(database).resolve().parent / 'bridge_geometry' if database else None


def bridge_geometry_signature(connection, config):
    directory = bridge_geometry_directory(connection, config)
    return tuple((str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns)
                 for p in sorted(directory.glob('*.json'))) if directory and directory.is_dir() else ()


@lru_cache(maxsize=2)
def _cached_bridge_paths(signature):
    payloads = []
    invalid_files = 0
    for name, _, _ in signature:
        try:
            payload = json.loads(Path(name).read_text())
            if not isinstance(payload, dict) or not isinstance(payload.get('elements'), list):
                raise ValueError('Expected an OSM elements list')
            payloads.append(payload)
        except (OSError, ValueError):
            invalid_files += 1
    # An incomplete/corrupt optional cache must not break the Progress page.
    try:
        paths, info = bridge_paths_from_osm(payloads)
    except (AttributeError, KeyError, TypeError, ValueError):
        paths, info = [], {'paths': 0, 'skipped_way_ids': []}
        invalid_files += 1
    return tuple(paths), {**info, 'invalid_files': invalid_files,
        'coverage': 'Cached geometry only; absence of a match is not proof that no bridge exists.'}


def load_bridge_paths(connection, config):
    return _cached_bridge_paths(bridge_geometry_signature(connection, config))
