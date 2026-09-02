"""Athlete-relative recovery estimates from recorded training data.

Recovery is deliberately represented as decaying load, not a categorical
number of mandatory rest hours.  The same residual load can permit an easy
run while making another long or quality session unattractive.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log2
from typing import Iterable

from .web.schemas import FitnessState, LoadWindow, SessionDifficulty


RECOVERY_HALF_LIFE_HOURS = 12.0
EASY_RUN_RESIDUAL_LIMIT = 0.55
TAXING_RUN_RESIDUAL_LIMIT = 0.25
# Below this residue, the effect on an ordinary easy-run distance would be
# smaller than the prescription's half-mile display precision.
EASY_VOLUME_RESIDUAL_FLOOR = 0.10


@dataclass(frozen=True, slots=True)
class RecoveryEstimate:
    initial_load: float
    residual_load: float
    elapsed_hours: float
    half_life_hours: float
    hours_until_easy: float
    hours_until_taxing: float
    distance_ratio: float | None
    duration_ratio: float | None
    zone_load_ratio: float | None


def _weighted_mean(values: list[tuple[float | None, float]]) -> float | None:
    available = [(value, weight) for value, weight in values if value is not None]
    if not available:
        return None
    return sum(value * weight for value, weight in available) / sum(
        weight for _, weight in available
    )


def athlete_relative_session_load(
    session: SessionDifficulty,
    typical: LoadWindow,
    *,
    performance_anomaly: str = "unknown",
    drift_percent: float | None = None,
) -> tuple[float, dict[str, float | None]]:
    """Measure a completed session against this athlete's ordinary run.

    Distance and duration are stable primary evidence. Recorded HR load adds
    intensity when coverage exists; zone fractions provide a smaller fallback
    when it does not. RPE, hills/downhills, and an objectively costly response
    contribute continuously instead of switching the run into a hard bucket.
    """

    typical_distance = (
        typical.distance_miles / typical.activity_count
        if typical.activity_count and typical.distance_miles > 0
        else None
    )
    typical_minutes = (
        typical.moving_minutes / typical.activity_count
        if typical.activity_count and typical.moving_minutes > 0
        else None
    )
    known_zone_load_count = (
        typical.zone_load_activity_count or typical.activity_count
    )
    typical_zone_load = (
        typical.zone_load / known_zone_load_count
        if known_zone_load_count
        and typical.zone_load is not None
        and typical.zone_load > 0
        else None
    )
    distance_ratio = (
        session.distance_miles / typical_distance if typical_distance else None
    )
    duration_ratio = (
        session.moving_minutes / typical_minutes if typical_minutes else None
    )
    zone_load_ratio = (
        session.zone_load / typical_zone_load
        if session.zone_load is not None and typical_zone_load
        else None
    )
    relative_work = _weighted_mean(
        [
            (distance_ratio, 0.45),
            (duration_ratio, 0.30),
            (zone_load_ratio, 0.25),
        ]
    )
    if relative_work is None:
        relative_work = 1.0

    known_zone_minutes = (
        session.zone_breakdown.easy_minutes
        + session.zone_breakdown.moderate_minutes
        + session.zone_breakdown.hard_minutes
    )
    # Zone load prices the observed intensity response. Fractions are only the
    # fallback when HR-derived load could not be calculated. Workout labels do
    # not change recovery: a run recorded as easy that becomes unusually hard
    # must cost more, while merely calling the same work "long" or "quality"
    # must not create a categorical recovery cliff.
    zone_factor = 1.0
    if zone_load_ratio is None and known_zone_minutes > 0:
        moderate_fraction = (
            session.zone_breakdown.moderate_minutes / known_zone_minutes
        )
        hard_fraction = session.zone_breakdown.hard_minutes / known_zone_minutes
        zone_factor += 0.15 * moderate_fraction + 0.50 * hard_fraction
    rpe_factor = (
        1.0
        + 0.07 * max(0, session.perceived_exertion - 6)
        if session.perceived_exertion is not None
        else 1.0
    )
    mechanical_flags = set(session.difficulty_flags)
    mechanical_factor = 1.0 + 0.08 * (
        int("hilly_session" in mechanical_flags)
        + int("substantial_downhill_load" in mechanical_flags)
    )
    drift_strength = (
        min(1.0, max(0.0, (drift_percent - 5.0) / 7.0))
        if drift_percent is not None
        else 0.0
    )
    if performance_anomaly == "unusually_strong":
        drift_strength *= 0.25
    response_strength = max(
        drift_strength,
        0.75 if performance_anomaly == "unusually_costly" else 0.0,
    )
    response_factor = 1.0 + 0.20 * response_strength

    load = (
        relative_work
        * zone_factor
        * rpe_factor
        * mechanical_factor
        * response_factor
    )
    return max(0.10, min(4.0, load)), {
        "distance_ratio": distance_ratio,
        "duration_ratio": duration_ratio,
        "zone_load_ratio": zone_load_ratio,
        "rpe_factor": rpe_factor,
        "mechanical_factor": mechanical_factor,
        "response_factor": response_factor,
    }


def decay_recovery_load(
    initial_load: float,
    elapsed_hours: float,
    *,
    half_life_hours: float = RECOVERY_HALF_LIFE_HOURS,
) -> float:
    """Exponentially decay transient session load with elapsed clock time."""

    return max(0.0, initial_load) * 0.5 ** (
        max(0.0, elapsed_hours) / half_life_hours
    )


def cumulative_recovery_load(
    session_loads: Iterable[tuple[float, float]],
    *,
    half_life_hours: float = RECOVERY_HALF_LIFE_HOURS,
) -> float:
    """Sum independently decayed residue from completed sessions.

    Each pair is ``(initial_load, elapsed_hours)``. Keeping this arithmetic in
    one pure function lets persisted fitness state and projected planning use
    the same recovery accounting across reloads.
    """

    return sum(
        decay_recovery_load(
            initial_load,
            elapsed_hours,
            half_life_hours=half_life_hours,
        )
        for initial_load, elapsed_hours in session_loads
    )


def prior_typical_load(state: FitnessState) -> LoadWindow:
    """Return the 28-day baseline immediately before the latest session.

    The rolling window stored on ``FitnessState`` includes the run whose
    recovery cost is being evaluated. Leaving it in both numerator and
    denominator makes an unusually large or hard session look more ordinary
    than it was.
    """

    window = state.recent_load.trailing_28d
    session = state.last_run
    if (
        session is None
        or state.days_since_last_run is None
        or state.days_since_last_run > window.days
        or window.activity_count <= 1
    ):
        return window
    zone_load = window.zone_load
    if zone_load is not None and session.zone_load is not None:
        zone_load = max(0.0, zone_load - session.zone_load)
    known_zone_load_count = window.zone_load_activity_count
    if known_zone_load_count and session.zone_load is not None:
        known_zone_load_count -= 1
    return LoadWindow(
        days=window.days,
        distance_miles=max(0.0, window.distance_miles - session.distance_miles),
        moving_minutes=max(0.0, window.moving_minutes - session.moving_minutes),
        zone_load=zone_load,
        hard_minutes=max(
            0.0,
            window.hard_minutes - session.zone_breakdown.hard_minutes,
        ),
        activity_count=window.activity_count - 1,
        zone_load_activity_count=known_zone_load_count,
    )


def _hours_to_limit(
    residual_load: float,
    limit: float,
    half_life_hours: float,
) -> float:
    if residual_load <= limit:
        return 0.0
    return half_life_hours * log2(residual_load / limit)


def estimate_recovery(state: FitnessState) -> RecoveryEstimate | None:
    """Estimate current readiness from the latest recorded/planned session."""

    if state.last_run is None or state.days_since_last_run is None:
        return None
    initial, evidence = athlete_relative_session_load(
        state.last_run,
        prior_typical_load(state),
        performance_anomaly=state.recent_performance_anomaly,
        drift_percent=state.last_run_drift_percent,
    )
    elapsed_hours = max(0.0, state.days_since_last_run * 24.0)
    residual = (
        state.recovery_residual_load
        if state.recovery_residual_load is not None
        else decay_recovery_load(initial, elapsed_hours)
    )
    return RecoveryEstimate(
        initial_load=initial,
        residual_load=residual,
        elapsed_hours=elapsed_hours,
        half_life_hours=RECOVERY_HALF_LIFE_HOURS,
        hours_until_easy=_hours_to_limit(
            residual, EASY_RUN_RESIDUAL_LIMIT, RECOVERY_HALF_LIFE_HOURS
        ),
        hours_until_taxing=_hours_to_limit(
            residual, TAXING_RUN_RESIDUAL_LIMIT, RECOVERY_HALF_LIFE_HOURS
        ),
        distance_ratio=evidence["distance_ratio"],
        duration_ratio=evidence["duration_ratio"],
        zone_load_ratio=evidence["zone_load_ratio"],
    )
