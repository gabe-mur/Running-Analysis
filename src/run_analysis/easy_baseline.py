"""Stable athlete-relative references for ordinary easy running."""

from __future__ import annotations

from datetime import datetime
from math import exp, log
from typing import Iterable


def ordinary_easy_sample_distance(
    actual_distance_miles: float,
    planning_role: str | None,
    prescribed_range_miles: tuple[float, float] | None,
) -> float | None:
    """Return the distance allowed to teach the ordinary easy baseline.

    Support and medium-long sessions serve different purposes and are not
    ordinary-run evidence. For a matched ordinary prescription, clamp modest
    execution noise to the prescribed range: the coach may progress that
    range, but going short or long once must not silently redefine a normal
    session and feed a self-reinforcing frequency loop.
    """

    if actual_distance_miles <= 0:
        return None
    if planning_role not in {None, "ordinary_easy"}:
        return None
    if planning_role == "ordinary_easy" and prescribed_range_miles is not None:
        low, high = prescribed_range_miles
        return min(max(actual_distance_miles, low), high)
    return actual_distance_miles


def recency_weighted_easy_distance(
    samples: Iterable[tuple[datetime, float]],
    as_of: datetime,
    *,
    half_life_days: float,
) -> float | None:
    """Return a continuous-time weighted median without a window cliff.

    Each sample retains half its influence per configured half-life.  This
    gives recent ordinary aerobic runs more voice while avoiding the abrupt
    baseline change caused by a run crossing a 28-day boundary.
    """

    weighted: list[tuple[float, float]] = []
    decay_rate = log(2.0) / max(1.0, half_life_days)
    for occurred_at, distance in samples:
        if distance <= 0 or occurred_at > as_of:
            continue
        age_days = max(
            0.0,
            (as_of - occurred_at).total_seconds() / 86400.0,
        )
        weighted.append((float(distance), exp(-decay_rate * age_days)))
    if not weighted:
        return None
    weighted.sort(key=lambda item: item[0])
    threshold = sum(weight for _, weight in weighted) / 2.0
    cumulative = 0.0
    for distance, weight in weighted:
        cumulative += weight
        if cumulative + 1e-12 >= threshold:
            return distance
    return weighted[-1][0]
