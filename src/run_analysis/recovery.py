"""Athlete-relative recovery estimates from recorded training data.

Recovery is deliberately represented as decaying load, not a categorical
number of mandatory rest hours.  The same residual load can permit an easy
run while making another long or quality session unattractive.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log2

from .web.schemas import FitnessState, LoadWindow, SessionDifficulty


RECOVERY_HALF_LIFE_HOURS = 12.0
EASY_RUN_RESIDUAL_LIMIT = 0.55
TAXING_RUN_RESIDUAL_LIMIT = 0.25
QUALITY_SESSION_LOAD_FACTOR = 1.35
LONG_RUN_LOAD_FACTOR = 1.15
RECOVERY_RUN_LOAD_FACTOR = 0.70
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
    typical_zone_load = (
        typical.zone_load / typical.activity_count
        if typical.activity_count
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
    # fallback when HR-derived load could not be calculated. Workout role is
    # accounted for separately below so completing a prescribed quality run
    # does not make its projected recovery cost disappear on upload. Long-run
    # recovery retains its separately calibrated distance-led behavior.
    zone_factor = 1.0
    if zone_load_ratio is None and known_zone_minutes > 0:
        moderate_fraction = (
            session.zone_breakdown.moderate_minutes / known_zone_minutes
        )
        hard_fraction = session.zone_breakdown.hard_minutes / known_zone_minutes
        zone_factor += 0.15 * moderate_fraction + 0.50 * hard_fraction
    session_type_factor = (
        QUALITY_SESSION_LOAD_FACTOR
        if session.is_quality_session
        else 1.0
    )

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
        * session_type_factor
        * rpe_factor
        * mechanical_factor
        * response_factor
    )
    return max(0.10, min(4.0, load)), {
        "distance_ratio": distance_ratio,
        "duration_ratio": duration_ratio,
        "zone_load_ratio": zone_load_ratio,
        "session_type_factor": session_type_factor,
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
        state.recent_load.trailing_28d,
        performance_anomaly=state.recent_performance_anomaly,
        drift_percent=state.last_run_drift_percent,
    )
    elapsed_hours = max(0.0, state.days_since_last_run * 24.0)
    residual = decay_recovery_load(initial, elapsed_hours)
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
