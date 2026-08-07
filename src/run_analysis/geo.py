"""Small geodesic helpers used by movement, segments, and wind features."""

from __future__ import annotations

from math import asin, atan2, cos, degrees, radians, sin, sqrt

EARTH_RADIUS_M = 6_371_008.8


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    value = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * asin(min(1.0, sqrt(value)))


def initial_bearing_degrees(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return initial travel bearing clockwise from true north."""
    first = radians(lat1)
    second = radians(lat2)
    delta_lon = radians(lon2 - lon1)
    y = sin(delta_lon) * cos(second)
    x = cos(first) * sin(second) - sin(first) * cos(second) * cos(delta_lon)
    return (degrees(atan2(y, x)) + 360.0) % 360.0

