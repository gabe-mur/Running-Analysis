"""Canonical, displayable distance targets; independent of adherence bounds."""

from contextlib import contextmanager
from contextvars import ContextVar
from math import ceil, floor, isfinite
from functools import wraps


CANONICAL_DISTANCE_TARGETS: ContextVar[bool] = ContextVar('canonical_distance_targets', default=False)


class UnrepresentableDistanceTarget(ValueError):
    """No target fits both the model's dose bounds and the display precision."""


def reject_unrepresentable_program(score):
    """An infeasible candidate loses search; unrelated errors still propagate."""
    @wraps(score)
    def checked(*args, **kwargs):
        try:
            return score(*args, **kwargs)
        except UnrepresentableDistanceTarget as error:
            details = kwargs.get('diagnostic_details')
            if details is not None:
                details.update(unrepresentable_distance_target=str(error), finalized_coaching_cost=0.)
            return (float('inf'), 0., 0., 0.)
    return checked


@contextmanager
def canonical_distance_policy(enabled=True):
    """Offline comparison switch; legacy production output stays unchanged."""
    token = CANONICAL_DISTANCE_TARGETS.set(enabled)
    try:
        yield
    finally:
        CANONICAL_DISTANCE_TARGETS.reset(token)


def rounded_distance_target(miles, *, minimum=0.1, maximum=None):
    """Nearest tenth (ties upward), constrained before any scoring takes place.

    A hard ceiling wins over nearest rounding. A band containing no displayable
    target is rejected rather than silently violating either limit.
    """
    if not isfinite(miles) or not isfinite(minimum) or (maximum is not None and not isfinite(maximum)):
        raise ValueError('Distance target and limits must be finite')
    low = max(1, ceil(minimum * 10 - 1e-9))
    high = floor(maximum * 10 + 1e-9) if maximum is not None else None
    if high is not None and high < low:
        raise UnrepresentableDistanceTarget(f'Dose limits contain no positive tenth-mile target: {miles=}, {minimum=}, {maximum=}')
    units = max(low, floor(miles * 10 + .5 + 1e-9))
    return min(units, high) / 10 if high is not None else units / 10
