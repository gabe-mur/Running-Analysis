"""Inspectable, distance-aware session and rolling training-load calculations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from math import log, sqrt
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
    zone_load_activity_count: int


@dataclass(frozen=True, slots=True)
class DistanceCapacity:
    recent_7d_miles: float
    prior_28d_weekly_miles: float
    sustained_weekly_miles: float
    retained_sustained_miles: float
    reference_miles: float
    acute_to_capacity_ratio: float | None


@dataclass(frozen=True, slots=True)
class ContinuousFatigue:
    """Exponentially decayed, athlete-relative weekly load equivalent."""

    equivalent_weekly_miles: float
    half_life_days: float
    session_count: int


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
        zone_load_activity_count=len(known_zone_loads),
    )


def continuous_fatigue_load(
    sessions: Iterable[TrainingSession],
    as_of: datetime,
    *,
    half_life_days: float = 7.0,
    baseline_days: int = 28,
) -> ContinuousFatigue:
    """Return a boundary-free distance/duration/intensity load equivalent.

    Each session is measured against the athlete's ordinary recent session
    using distance, duration, and HR-derived zone load. Its contribution then
    halves smoothly every ``half_life_days``. Multiplication by ``ln(2)``
    normalizes a steady weekly training rate back to that familiar weekly-mile
    unit, so the result can be compared directly with demonstrated capacity.
    """

    records = [session for session in sessions if session.start_time <= as_of]
    half_life = max(0.1, float(half_life_days))
    baseline = rolling_load(records, as_of, max(1, int(baseline_days)))
    if baseline.activity_count <= 0 or baseline.distance_miles <= 0:
        return ContinuousFatigue(0.0, half_life, 0)
    typical_distance = baseline.distance_miles / baseline.activity_count
    typical_minutes = (
        baseline.moving_minutes / baseline.activity_count
        if baseline.moving_minutes > 0
        else None
    )
    known_load_count = (
        baseline.zone_load_activity_count or baseline.activity_count
    )
    typical_zone_load = (
        baseline.zone_load / known_load_count
        if baseline.zone_load is not None
        and baseline.zone_load > 0
        and known_load_count > 0
        else None
    )

    equivalent = 0.0
    for session in records:
        evidence: list[tuple[float, float]] = []
        if typical_distance > 0:
            evidence.append((session.distance_miles / typical_distance, 0.45))
        if typical_minutes and session.moving_minutes > 0:
            evidence.append((session.moving_minutes / typical_minutes, 0.30))
        if typical_zone_load and session.zone_load is not None:
            evidence.append((session.zone_load / typical_zone_load, 0.25))
        if not evidence:
            continue
        relative_load = sum(value * weight for value, weight in evidence) / sum(
            weight for _, weight in evidence
        )
        age_days = max(
            0.0,
            (as_of - session.start_time).total_seconds() / 86400.0,
        )
        equivalent += (
            log(2.0)
            * typical_distance
            * relative_load
            * 0.5 ** (age_days / half_life)
        )
    return ContinuousFatigue(equivalent, half_life, len(records))


def continuous_distance_rate(
    sessions: Iterable[TrainingSession],
    as_of: datetime,
    *,
    half_life_days: float = 7.0,
) -> float:
    """Return boundary-free completed mileage as an equivalent weekly rate.

    Unlike ``continuous_fatigue_load``, this is deliberately distance-only.
    It seeds the mileage allocator with work already completed before a fresh
    21-day regeneration, preventing each new calendar origin from creating a
    new mileage budget.
    """

    half_life = max(0.1, float(half_life_days))
    normalization = log(2.0) * 7.0 / half_life
    return sum(
        normalization
        * max(0.0, session.distance_miles)
        * 0.5
        ** (
            max(0.0, (as_of - session.start_time).total_seconds() / 86400.0)
            / half_life
        )
        for session in sessions
        if session.start_time <= as_of
    )


def short_term_density_half_life_days(
    load_half_life_days: float,
    *,
    recovery_half_life_hours: float = 12.0,
) -> float:
    """Bridge immediate recovery and the slower continuous load signal.

    The geometric midpoint supplies a distinct short-term density timescale
    without introducing another independently tuned window. With the shipped
    12-hour recovery and seven-day load half-lives this is about 1.9 days.
    """

    recovery_days = max(0.1, recovery_half_life_hours / 24.0)
    return sqrt(recovery_days * max(0.1, load_half_life_days))


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
    retention_half_life_days: float = 84.0,
) -> DistanceCapacity:
    """Compare current mileage with retained, demonstrated four-week capacity.

    The immediate preceding four weeks remain visible, but a short illness,
    trip, or other disruption cannot instantly redefine normal capacity. The
    best completed 28-day block before the acute week is retained in full for
    a grace period and then decays gradually. A more recent strong seven-day
    exposure can re-confirm that retained capacity, but is capped at the
    sustained 28-day evidence so one spike cannot invent a higher baseline.
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
    recent_confirmation = 0.0
    candidate = history_start
    while candidate <= end:
        seven_start = candidate - timedelta(days=6)
        seven = sum(
            miles
            for day, miles in daily.items()
            if seven_start <= day <= candidate
        )
        confirmation = min(seven, best_value)
        age_days = max(0, (end - candidate).days)
        confirmation *= 0.5 ** (
            max(0, age_days - retention_grace_days)
            / max(1.0, retention_half_life_days)
        )
        recent_confirmation = max(recent_confirmation, confirmation)
        candidate += timedelta(days=1)
    reference = max(prior, retained, recent_confirmation)
    ratio = recent / reference if reference > 0 else None
    return DistanceCapacity(recent, prior, best_value, retained, reference, ratio)
