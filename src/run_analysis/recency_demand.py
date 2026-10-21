"""Conditional cumulative-demand ordering, not a physiological recovery model."""
from dataclasses import dataclass
from datetime import date
from math import inf, isfinite

from .workload_accounting import Quantity

CHANNELS = ('miles', 'running_minutes', 'cardio', 'work_minutes')


@dataclass(frozen=True)
class DemandTerm:
    channel: str
    variable: str
    coefficient: float
    bounds: Quantity | None

    def __post_init__(self):
        if self.channel not in CHANNELS or not self.variable or not isfinite(self.coefficient) or self.coefficient < 0:
            raise ValueError('Named nonnegative demand coefficient required')


@dataclass(frozen=True)
class DemandEvent:
    key: str
    day: date
    terms: tuple[DemandTerm, ...]
    accounting_key: str
    purpose: str
    structure: tuple[float, ...] | None
    source: str

    def __post_init__(self):
        if not self.key or type(self.day) is not date or not self.accounting_key or not self.purpose:
            raise ValueError('Identified, dated and unit-tagged demand required')
        if self.source not in {'observed', 'planned'}:
            raise ValueError('Named demand source required')
        if {t.channel for t in self.terms} != set(CHANNELS):
            raise ValueError('Every channel must be explicit, including unknowns')
        if self.structure is not None and any(not isfinite(x) or x <= 0 for x in self.structure):
            raise ValueError('Positive recorded/prescribed work-bout seconds required')


def _validate(history, next_event):
    if len({e.key for e in (*history, next_event)}) != len(history)+1:
        raise ValueError('Duplicate actual/projected identity')
    if any(e.day >= next_event.day for e in history):
        raise ValueError('History must be earlier; same-day/future ordering is unsupported')
    if any(e.accounting_key != next_event.accounting_key for e in history):
        raise ValueError('Cardio accounting units must match')


def _ordering(intervals):
    left = all(high <= 1e-8 for low, high in intervals)
    right = all(low >= -1e-8 for low, high in intervals)
    if left and right:
        return 'equal'
    if left:
        return 'left_no_greater'
    if right:
        return 'right_no_greater'
    if any(high < -1e-8 for low, high in intervals) and any(low > 1e-8 for low, high in intervals):
        return 'crossing_recency_tradeoff'
    return 'uncertain'


def compare_recency(left_history, left_next, right_history, right_next, *, response_review=None):
    left_history, right_history = tuple(left_history), tuple(right_history)
    _validate(left_history, left_next)
    _validate(right_history, right_next)
    if left_next.accounting_key != right_next.accounting_key:
        raise ValueError('Compare the same cardio accounting')
    variables = {}
    for e in (*left_history, left_next, *right_history, right_next):
        for t in e.terms:
            k = (t.channel, t.variable)
            if k in variables and variables[k] != t.bounds:
                raise ValueError('Shared variables must have identical uncertainty bounds')
            variables[k] = t.bounds
    ages = sorted({0, *((left_next.day-e.day).days for e in left_history),
                      *((right_next.day-e.day).days for e in right_history)})
    curves = {channel: [] for channel in CHANNELS}
    for age in ages:
        coefficients = {}
        for sign, history, target in ((1., left_history, left_next), (-1., right_history, right_next)):
            for event in (*history, target):
                if (target.day-event.day).days > age:
                    continue
                for t in event.terms:
                    k = (t.channel, t.variable)
                    coefficients[k] = coefficients.get(k, 0.)+sign*t.coefficient
        for channel in CHANNELS:
            low = high = 0.
            for k, coefficient in coefficients.items():
                if k[0] != channel or abs(coefficient) < 1e-12:
                    continue
                q = variables[k]
                a, b = (q.low, q.high) if q is not None else (0., inf)
                if coefficient > 0:
                    low += coefficient*a
                    high += coefficient*b
                else:
                    low += coefficient*b
                    high += coefficient*a
            curves[channel].append((low, high))
    orders = {k: _ordering(v) for k, v in curves.items()}
    values = set(orders.values())
    overall = ('equal' if values == {'equal'} else
               'left_no_greater' if values <= {'equal', 'left_no_greater'} else
               'right_no_greater' if values <= {'equal', 'right_no_greater'} else
               'no_joint_order')
    comparable = left_next.purpose == right_next.purpose and left_next.structure == right_next.structure
    return dict(left_next_key=left_next.key, right_next_key=right_next.key,
        left_next_day=left_next.day, right_next_day=right_next.day,
        same_next_purpose_and_structure=comparable,
        next_dose_changed=left_next.terms != right_next.terms,
        recency_demand_order=overall, channel_orders=orders,
        cutoff_differences={k: [dict(age_days=age, left_minus_right_low=lo if isfinite(lo) else None,
            left_minus_right_high=hi if isfinite(hi) else None) for age, (lo, hi) in zip(ages, v)]
            for k, v in curves.items()},
        response_review=response_review,
        response_action='review_reported_concern' if response_review and response_review.get('review_needed') else
                        'no_clearance_inferred',
        physiology_established=False, progression_accepted=False, running_authorized=False,
        planner_influence=False,
        limitations=['Ordering conditional on the supplied ledger and declared shared demand variables',
            'No nonlinear interactions, muscular fatigue or calibrated hours-to-recovery',
            'Date recency, not a precise elapsed-hours model; same-day order unsupported',
            'No superiority claim across changed purposes, goals or unmeasured response'])
