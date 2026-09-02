"""Smoothly retained evidence of demonstrated running durability."""

from __future__ import annotations

from datetime import datetime
from math import exp, log
from typing import Iterable


def retained_long_run_capacity(
    runs: Iterable[tuple[datetime, float]],
    as_of: datetime,
    *,
    grace_days: float = 90.0,
    half_life_days: float = 180.0,
) -> float:
    """Return the strongest recency-decayed single-run distance.

    A demonstrated distance remains fully credited through ``grace_days`` and
    then fades continuously. This avoids the artificial capacity cliff caused
    by treating day 30 as materially different from day 31.
    """

    if half_life_days <= 0:
        raise ValueError("half_life_days must be positive")
    best = 0.0
    for started_at, distance_miles in runs:
        if distance_miles <= 0 or started_at > as_of:
            continue
        age_days = max(0.0, (as_of - started_at).total_seconds() / 86400.0)
        decay_days = max(0.0, age_days - grace_days)
        retained = distance_miles * exp(-log(2.0) * decay_days / half_life_days)
        best = max(best, retained)
    return best


def supported_long_run_capacity(
    runs: Iterable[tuple[datetime, float, float | None]],
    as_of: datetime,
    *,
    grace_days: float = 90.0,
    half_life_days: float = 180.0,
) -> float:
    """Return long-run capacity supported by a prescription or repetition.

    Completing a planned progression supports the prescribed distance even
    when execution lands beyond its range.  The surplus still becomes training
    load immediately, but a single over-distance outing is not enough evidence
    to make that entire surplus the next safe long-run anchor.  Repeating the
    distance supplies that evidence, represented by the second-strongest
    retained surplus exposure.  Runs without a known prescription remain
    ordinary historical evidence and are credited in full.
    """

    if half_life_days <= 0:
        raise ValueError("half_life_days must be positive")

    directly_supported = 0.0
    surplus_exposures: list[float] = []
    for started_at, distance_miles, prescribed_high_miles in runs:
        if distance_miles <= 0 or started_at > as_of:
            continue
        age_days = max(0.0, (as_of - started_at).total_seconds() / 86400.0)
        decay_days = max(0.0, age_days - grace_days)
        retention = exp(-log(2.0) * decay_days / half_life_days)
        if prescribed_high_miles is None or prescribed_high_miles <= 0:
            supported_distance = distance_miles
        else:
            supported_distance = min(distance_miles, prescribed_high_miles)
            if distance_miles > prescribed_high_miles:
                surplus_exposures.append(distance_miles * retention)
        directly_supported = max(
            directly_supported,
            supported_distance * retention,
        )

    repeated_surplus = (
        sorted(surplus_exposures, reverse=True)[1]
        if len(surplus_exposures) >= 2
        else 0.0
    )
    return max(directly_supported, repeated_surplus)
