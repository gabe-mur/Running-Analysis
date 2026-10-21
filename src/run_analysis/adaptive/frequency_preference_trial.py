"""Offline coaching-prior trial, not connected to scheduler selection.

The declared target volume and reference fatigue are frozen before comparing
candidates. Descriptive training plans motivate a smooth center, not a proven
optimal frequency or a physiological conversion from days to fatigue.
"""
from dataclasses import dataclass
from math import isfinite


# Miles per seven days -> running DAYS per seven days. Published group means
# count sessions, not days; using them as day anchors is a single-session/day
# approximation. Low-volume and 20-mile anchors are explicit coaching choices.
ANCHORS = ((0., 0.), (5., 3.), (12., 3.), (20., 4.),
           (26.7, 4.1), (36.4, 4.9), (66.9, 6.8))


def frequency_center(target_miles_per_7):
    if not isfinite(target_miles_per_7) or target_miles_per_7 < 0:
        raise ValueError('Finite nonnegative target volume required')
    for (a, x), (b, y) in zip(ANCHORS, ANCHORS[1:]):
        if target_miles_per_7 <= b:
            return x+(y-x)*(target_miles_per_7-a)/(b-a)
    return 6.8  # No extrapolation to doubles or increasing beyond sampled range.


def running_days_per_7(days, *, span_days):
    """Count distinct dates, not sessions; caller declares a complete exposure."""
    if not isinstance(span_days, int) or isinstance(span_days, bool) or span_days <= 0:
        raise ValueError('Positive integer exposure span required')
    unique = set(days)
    if len(unique) > span_days:
        raise ValueError('More running dates than exposure dates')
    return 7*len(unique)/span_days


@dataclass(frozen=True)
class FrequencyPreference:
    target_miles_per_7: float
    reference_fatigue: float
    coefficient: float

    def __post_init__(self):
        frequency_center(self.target_miles_per_7)
        if not isfinite(self.reference_fatigue) or self.reference_fatigue <= 0:
            raise ValueError('Positive finite frozen reference fatigue required')
        if not isfinite(self.coefficient) or self.coefficient < 0:
            raise ValueError('Finite nonnegative coefficient required')

    @property
    def center(self):
        return frequency_center(self.target_miles_per_7)

    def cost(self, running_days):
        if not isfinite(running_days) or not 0 <= running_days <= 7:
            raise ValueError('Running-day rate must be in [0, 7]')
        # At coefficient .01, one extra/missing day costs 1% of the frozen
        # reference fatigue. Maximum 4% keeps distant alternatives available.
        return self.reference_fatigue*self.coefficient*min((running_days-self.center)**2, 4.)
