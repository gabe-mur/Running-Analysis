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
    performance_response: str = "unknown",
    drift_percent: float | None = None,
    prescribed_intensity_factor: float | None = None,
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
    actual_volume_work = _weighted_mean(
        [(distance_ratio, 0.60), (duration_ratio, 0.40)]
    )

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
    if performance_response == "stronger_than_recent":
        drift_strength *= 0.25
    response_strength = max(
        drift_strength,
        0.75 if performance_response == "higher_cost_than_recent" else 0.0,
    )
    response_factor = 1.0 + 0.20 * response_strength

    observed_load = relative_work * zone_factor
    # Short reps can complete the prescribed muscular/metabolic dose before HR
    # catches up. Preserve that planned intensity on the *actual* distance and
    # duration; observed surplus remains free to raise load above this floor.
    prescribed_load_floor = (
        actual_volume_work * prescribed_intensity_factor
        if actual_volume_work is not None
        and prescribed_intensity_factor is not None
        and prescribed_intensity_factor > 1.0
        else None
    )
    load = (
        max(observed_load, prescribed_load_floor or 0.0)
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
        "prescribed_intensity_factor": prescribed_intensity_factor,
        "prescribed_load_floor": prescribed_load_floor,
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


def load_window_before_session(
    window: LoadWindow,
    session: SessionDifficulty | None,
) -> LoadWindow:
    """Remove one evaluated session from an aggregate reference window."""

    if (
        session is None
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


def ordinary_session_reference(
    window: LoadWindow,
    ordinary_distance_miles: float | None,
) -> LoadWindow:
    """Express aggregate training evidence as one ordinary aerobic session.

    Long and quality runs belong in the load history, but they must not make
    the unit used to price the next ordinary run larger merely because those
    sessions are longer.  Distance is anchored to the robust easy-run
    baseline; duration and HR load retain the athlete's observed per-mile
    relationships.  The same conversion is used before and after upload.
    """

    if (
        ordinary_distance_miles is None
        or ordinary_distance_miles <= 0
        or window.activity_count <= 0
        or window.distance_miles <= 0
    ):
        return window
    distance = float(ordinary_distance_miles)
    moving_per_mile = window.moving_minutes / window.distance_miles
    zone_load_per_mile = (
        window.zone_load / window.distance_miles
        if window.zone_load is not None and window.zone_load > 0
        else None
    )
    hard_minutes_per_mile = window.hard_minutes / window.distance_miles
    return LoadWindow(
        days=window.days,
        distance_miles=distance,
        moving_minutes=distance * moving_per_mile,
        zone_load=(
            distance * zone_load_per_mile
            if zone_load_per_mile is not None
            else None
        ),
        hard_minutes=distance * hard_minutes_per_mile,
        activity_count=1,
        zone_load_activity_count=(
            1 if zone_load_per_mile is not None else 0
        ),
    )


def prior_typical_load(state: FitnessState) -> LoadWindow:
    """Return the 28-day baseline immediately before the latest session.

    The rolling window stored on ``FitnessState`` includes the run whose
    recovery cost is being evaluated. Leaving it in both numerator and
    denominator makes an unusually large or hard session look more ordinary
    than it was.
    """

    window = state.recent_load.trailing_28d
    if (
        state.last_run is None
        or state.days_since_last_run is None
        or state.days_since_last_run > window.days
    ):
        prior = window
    else:
        prior = load_window_before_session(window, state.last_run)
    return ordinary_session_reference(prior, state.typical_easy_run_miles)


def projected_recovery_reference_miles(state: FitnessState) -> float:
    """Return the ordinary-session distance used on both sides of upload.

    The robust ordinary-easy baseline keeps long and quality sessions from
    enlarging the unit used to price recovery. Before upload, the current
    trailing history precedes the proposed run; afterward, ``prior_typical_load``
    removes that run and reconstructs the same evidence at this distance.
    """

    if state.typical_easy_run_miles is not None:
        return state.typical_easy_run_miles
    window = state.recent_load.trailing_28d
    if window.activity_count > 0 and window.distance_miles > 0:
        return window.distance_miles / window.activity_count
    return 1.0


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
    response_applies_to_last = (
        state.recent_performance_response_activity_id is None
        or state.last_run_activity_id is None
        or state.recent_performance_response_activity_id
        == state.last_run_activity_id
    )
    initial, evidence = athlete_relative_session_load(
        state.last_run,
        prior_typical_load(state),
        performance_response=(
            state.recent_performance_response
            if response_applies_to_last
            else "within_recent_range"
        ),
        drift_percent=state.last_run_drift_percent,
        prescribed_intensity_factor=state.last_run_prescribed_intensity_factor,
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
