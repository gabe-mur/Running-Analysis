"""Offline full-path funding accounting, independent of the hard/soft cut.

Retain date-local under/overload and the existing full-program recurring-rate
objective. Do not also levy a hard-prefix quota. No new weights, recovery
thresholds, mileage credit, catch-up debt, or release-gate changes.
"""
from contextlib import contextmanager
from math import isfinite, sqrt
from time import perf_counter

from run_analysis import weekly_schedule as planner
from run_analysis.selection_objective import CONTINUATION_FUNDING_SCORER


def full_path_funding(volume, rate, target, *, legacy_residual):
    """One physical-path price; legacy residual is audited, never carried over.

    Inputs have already been generated from actual history plus the exact
    candidate sequence. No hard length or boundary session enters this price.
    This does not establish that the inherited volume/rate terms are calibrated.
    """
    started = perf_counter()
    low, high = target
    ordinary, squared = volume['ordinary_miles'], volume['squared_violation']
    if (not all(isfinite(v) for v in (rate, low, high, ordinary, squared, legacy_residual))
            or rate < 0 or not 0 < low <= high or ordinary <= 0 or squared < 0
            or legacy_residual < 0):
        raise ValueError('Require finite nonnegative load and positive ordered references')
    rate_violation = max(0., low-rate, rate-high)
    hard_quota = legacy_residual-rate_violation
    if hard_quota < -1e-8:
        raise ValueError('Legacy residual does not contain the full-program rate violation')
    weight = planner.TARGET_FIT_UNIT / ordinary**2
    path_cost, rate_cost = weight*squared, weight*rate_violation**2
    return dict(version='physical_path_funding_v1', target_violation=sqrt(squared+rate_violation**2),
        target_cost=path_cost+rate_cost, volume_cost=path_cost, rate_cost=rate_cost,
        rate_violation=rate_violation, removed_hard_quota_violation=max(0., hard_quota),
        removed_cost=weight*(legacy_residual**2-rate_violation**2),
        seconds=perf_counter()-started)


@contextmanager
def boundary_funding_policy():
    """Diagnostic scope only; not a user option or a default workflow change."""
    metrics = dict(evaluations=0, seconds=0.)
    def evaluate(*args, **kwargs):
        result = full_path_funding(*args, **kwargs)
        metrics['evaluations'] += 1
        metrics['seconds'] += result['seconds']
        return result
    token = CONTINUATION_FUNDING_SCORER.set(evaluate)
    try:
        yield metrics
    finally:
        CONTINUATION_FUNDING_SCORER.reset(token)
