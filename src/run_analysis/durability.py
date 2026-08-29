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
