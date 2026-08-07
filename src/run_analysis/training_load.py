"""Inspectable, distance-aware session and rolling training-load calculations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Iterable


# Edwards-style time-in-zone weighting. Below-Z1 time is recorded as easy
# volume but carries no Edwards points; above-Z5 is capped at the Z5 weight.
ZONE_WEIGHTS: dict[str, float] = {
    "below_z1": 0.0,
    "z1": 1.0,
    "z2": 2.0,
    "z3": 3.0,
    "z4": 4.0,
    "z5": 5.0,
    "above_z5": 5.0,
}


@dataclass(frozen=True, slots=True)
class SessionLoad:
    zone_load: float | None
    easy_minutes: float
    moderate_minutes: float
    hard_minutes: float
    known_hr_minutes: float
    unknown_hr_minutes: float
    hr_coverage: float


@dataclass(frozen=True, slots=True)
class TrainingSession:
    activity_id: int
    start_time: datetime
    distance_miles: float
    moving_minutes: float
    zone_load: float | None
    hard_minutes: float


@dataclass(frozen=True, slots=True)
class RollingLoad:
    days: int
    distance_miles: float
    moving_minutes: float
    zone_load: float | None
    hard_minutes: float
    activity_count: int


@dataclass(frozen=True, slots=True)
class DistanceCapacity:
    recent_7d_miles: float
    prior_28d_weekly_miles: float
    sustained_weekly_miles: float
    retained_sustained_miles: float
    reference_miles: float
    acute_to_capacity_ratio: float | None


def calculate_session_load(zone_seconds: dict[str, float], moving_time_s: float) -> SessionLoad:
    """Calculate intensity-weighted load without filling missing HR time."""

    normalized = {name: max(0.0, float(zone_seconds.get(name, 0.0) or 0.0)) for name in (*ZONE_WEIGHTS, "unknown")}
    known_seconds = sum(normalized[name] for name in ZONE_WEIGHTS)
    unknown_seconds = max(normalized["unknown"], max(0.0, float(moving_time_s)) - known_seconds)
    coverage = min(1.0, known_seconds / moving_time_s) if moving_time_s > 0 else 0.0
    # Below 50% coverage, a precise intensity score is more misleading than a
    # missing value. Duration and distance remain available independently.
    zone_load = None
    if coverage >= 0.5:
        zone_load = sum(normalized[name] / 60.0 * weight for name, weight in ZONE_WEIGHTS.items())
    return SessionLoad(
        zone_load=zone_load,
        easy_minutes=sum(normalized[name] for name in ("below_z1", "z1", "z2")) / 60.0,
        moderate_minutes=normalized["z3"] / 60.0,
        hard_minutes=sum(normalized[name] for name in ("z4", "z5", "above_z5")) / 60.0,
        known_hr_minutes=known_seconds / 60.0,
        unknown_hr_minutes=unknown_seconds / 60.0,
        hr_coverage=coverage,
    )


def rolling_load(
    sessions: Iterable[TrainingSession], as_of: datetime, days: int
) -> RollingLoad:
    start = as_of - timedelta(days=days)
    selected = [session for session in sessions if start < session.start_time <= as_of]
    known_zone_loads = [session.zone_load for session in selected if session.zone_load is not None]
    return RollingLoad(
        days=days,
        distance_miles=sum(session.distance_miles for session in selected),
        moving_minutes=sum(session.moving_minutes for session in selected),
        zone_load=sum(known_zone_loads) if known_zone_loads else None,
        hard_minutes=sum(session.hard_minutes for session in selected),
        activity_count=len(selected),
    )


def acute_to_prior_weekly_ratio(sessions: Iterable[TrainingSession], as_of: datetime) -> float | None:
    """Compare the last 7 days with the preceding 28-day weekly mean.

    Zone load is preferred when every selected session has usable HR coverage;
    moving time is a transparent fallback when it is not.
    """

    records = list(sessions)
    acute_start = as_of - timedelta(days=7)
    prior_start = acute_start - timedelta(days=28)
    acute = [item for item in records if acute_start < item.start_time <= as_of]
    prior = [item for item in records if prior_start < item.start_time <= acute_start]
    if not prior:
        return None
    all_selected = acute + prior
    use_zone_load = bool(all_selected) and all(item.zone_load is not None for item in all_selected)
    if use_zone_load:
        acute_value = sum(float(item.zone_load) for item in acute)
        prior_week = sum(float(item.zone_load) for item in prior) / 4.0
    else:
        acute_value = sum(item.moving_minutes for item in acute)
        prior_week = sum(item.moving_minutes for item in prior) / 4.0
    return acute_value / prior_week if prior_week > 0 else None


def distance_capacity(
    sessions: Iterable[TrainingSession],
    as_of: datetime,
    *,
    lookback_days: int = 365,
    retention_grace_days: int = 28,
    retention_half_life_days: float = 42.0,
) -> DistanceCapacity:
    """Compare current mileage with retained, demonstrated four-week capacity.

    The immediate preceding four weeks remain visible, but a short illness,
    trip, or other disruption cannot instantly redefine normal capacity.  The
    best completed 28-day block before the acute week is retained in full for
    a grace period and then decays gradually.
    """

    records = [item for item in sessions if item.start_time <= as_of]
    end = as_of.date()
    acute_start = as_of - timedelta(days=7)
    prior_start = acute_start - timedelta(days=28)
    recent = sum(item.distance_miles for item in records if acute_start < item.start_time <= as_of)
    prior = sum(item.distance_miles for item in records if prior_start < item.start_time <= acute_start) / 4.0
    if not records:
        return DistanceCapacity(0.0, 0.0, 0.0, 0.0, 0.0, None)

    daily: dict[date, float] = {}
    for item in records:
        day = item.start_time.astimezone(as_of.tzinfo).date()
        daily[day] = daily.get(day, 0.0) + item.distance_miles
    history_start = max(min(daily), end - timedelta(days=lookback_days - 1))
    last_completed_end = acute_start.astimezone(as_of.tzinfo).date()
    best_value = 0.0
    completed_windows: list[tuple[date, float]] = []
    candidate = history_start
    while candidate <= last_completed_end:
        window_start = candidate - timedelta(days=27)
        value = sum(miles for day, miles in daily.items() if window_start <= day <= candidate) / 4.0
        completed_windows.append((candidate, value))
        # A rolling 28-day total often stays flat across adjacent dates. Keep
        # the latest equally strong endpoint so retention is measured from the
        # end of the demonstrated block, not its first plateau day.
        if value >= best_value:
            best_value = value
        candidate += timedelta(days=1)

    retained = max(
        (
            value
            * 0.5
            ** (
                max(0, (end - window_end).days - retention_grace_days)
                / max(1.0, retention_half_life_days)
            )
            for window_end, value in completed_windows
        ),
        default=0.0,
    )
    reference = max(prior, retained)
    ratio = recent / reference if reference > 0 else None
    return DistanceCapacity(recent, prior, best_value, retained, reference, ratio)
