"""Shadow response review: declared expectations, not learned fatigue labels.

Self-reported exertion complements observed load; it is not added to HR load.
No personal-history minimum and no automatic reward for one easier session.
"""
from dataclasses import dataclass
from math import isfinite

from .workload_accounting import Quantity


@dataclass(frozen=True)
class ResponseExpectation:
    source: str
    effort: Quantity | None = None
    feeling: Quantity | None = None

    def __post_init__(self):
        if not self.source.strip():
            raise ValueError('Name the expectation assumptions')
        for value, ceiling in ((self.effort, 10), (self.feeling, 100)):
            if value is not None and value.high > ceiling:
                raise ValueError('Expectation is outside the report scale')


def assess_response(expected: ResponseExpectation, *, effort=None, feeling=None,
                    feedback_conflict=False, readiness_concern=False):
    """Compare to the expectation issued before execution, never refit it here.

    Effort is CR10 (higher is harder); feeling is 0–100 (higher is better).
    Absent reports are unknown, not compliance. Current readiness concerns stay
    independent of whether a previous session felt good. Pain/illness handling
    remains owned by the existing readiness workflow, not an exertion score.
    """
    channels = {}
    for name, value, bounds, ceiling, higher_is_harder in (
        ('effort', effort, expected.effort, 10, True),
        ('feeling', feeling, expected.feeling, 100, False),
    ):
        if value is not None and (isinstance(value, bool) or not isfinite(value) or not 0 <= value <= ceiling):
            raise ValueError(f'Invalid {name} report')
        if value is None or bounds is None:
            state = 'unknown'
        elif bounds.low <= value <= bounds.high:
            state = 'as_expected'
        elif (value > bounds.high) == higher_is_harder:
            state = 'more_demanding_than_expected'
        else:
            state = 'less_demanding_than_expected'
        channels[name] = dict(status=state, observed=value, expected=bounds)
    concerns = [name for name, value in channels.items()
                if value['status'] == 'more_demanding_than_expected']
    if readiness_concern:
        concerns.append('current_readiness')
    if feedback_conflict:
        concerns.append('conflicting_imports')
    return dict(channels=channels, review_reasons=concerns,
        review_needed=bool(concerns), expectation_source=expected.source,
        planner_influence=False, automatic_progression=False,
        interpretation='Response discrepancy, not a diagnosis or measured recovery time')
