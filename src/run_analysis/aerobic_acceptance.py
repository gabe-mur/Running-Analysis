"""Shared, load-aware aerobic-gap acceptance; not a universal rest-day limit."""

from dataclasses import dataclass
from datetime import date
from math import ceil
from typing import Iterable


RECOVERY_EXCEPTION_MARKERS = (
    "replaced with aerobic", "recovery guardrail", "readiness guardrail", "recovery is incomplete",
)


def cadence_exceptions(reasons: Iterable[str]) -> tuple[str, ...]:
    """Preserve the gate's existing explanation-based exception convention."""
    return tuple(dict.fromkeys(reason for reason in reasons
        if any(marker in reason.casefold() for marker in RECOVERY_EXCEPTION_MARKERS)))


@dataclass(frozen=True)
class AerobicAcceptance:
    rejected: bool
    reason: str
    evidence: dict


def assess_aerobic_exposure(*, sessions: Iterable[tuple[date, float]],
                            target_trajectory: Iterable[tuple[float, float]],
                            ordinary_miles: float | None, planning_mode: str | None,
                            exception_reasons: Iterable[str] = (), rounding_miles: float = .5,
                            enabled: bool = True) -> AerobicAcceptance:
    """Apply the existing projection-gate contract to finalized program facts."""
    if not enabled:
        return AerobicAcceptance(False, "disabled", {})
    targets = tuple(target_trajectory)
    if not targets or ordinary_miles is None or planning_mode is None:
        return AerobicAcceptance(False, "missing_context", {})
    if planning_mode != "established":
        return AerobicAcceptance(False, "non_established", {})
    if tuple(exception_reasons):
        return AerobicAcceptance(False, "recovery_exception", {})
    runs = tuple(sessions)
    miles = sum(miles for _, miles in runs)
    low, high = (sum(x[i] for x in targets) / 7 for i in (0, 1))
    if miles >= low - max(0, rounding_miles) - 1e-9:
        return AerobicAcceptance(False, "funded", {})
    rate = sum((low + high) / 2 for low, high in targets) / len(targets)
    if rate <= 0:
        return AerobicAcceptance(False, "nonpositive_target", {})
    expected = 7 * ordinary_miles / rate
    allowed = max(4, ceil(2 * expected - 1e-9))
    dates = sorted(day for day, _ in runs)
    separations = [(b - a).days for a, b in zip(dates, dates[1:])]
    longest = max(separations, default=0)
    evidence = dict(planned_midpoint_miles=miles, integrated_target=(low, high),
        ordinary_easy_midpoint_miles=ordinary_miles, target_midpoint_rate=rate,
        expected_date_separation_days=expected, allowed_date_separation_days=allowed,
        observed_date_separations_days=separations, maximum_date_separation_days=longest)
    return AerobicAcceptance(longest > allowed,
        "underfunded_aerobic_exposure_gap" if longest > allowed else "spacing_within_allowance", evidence)
