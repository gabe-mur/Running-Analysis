"""Offline additive volume-path pricing; no short-window maximum or re-anchor.

Keep the existing distance EWMA, one ordinary-session allowance and fit weight.
Score each projected date once against its candidate-independent target. Divide
the squared-mile-day integral by the existing load half-life, not the horizon
length: a shared future date has the same price after expected completion.
This normalization is an explicit modeling choice, not a physiological claim.
"""
from math import isfinite, log, sqrt
from time import perf_counter
from contextlib import contextmanager

from run_analysis import weekly_schedule as planner
from run_analysis.selection_objective import VOLUME_PATH_SCORER


def volume_path(days, states, targets, config, *, trace=False):
    started = perf_counter()
    if not days or len(days) != len(states) or len(days) != len(targets):
        raise ValueError('Require nonempty aligned days, states and targets')
    opening = states[0].recent_load.continuous_distance_miles
    if opening is None or not isfinite(opening) or opening < 0:
        raise ValueError('Volume-path prototype requires an observed continuous mileage rate')
    half_life = float(config.get('coaching', {}).get('continuous_fatigue_half_life_days', 7))
    ordinary = sum(planner.typical_easy_distance(states[0])) / 2
    if not isfinite(half_life) or half_life <= 0 or not isfinite(ordinary) or ordinary <= 0:
        raise ValueError('Require positive finite load half-life and ordinary-session reference')
    decay = .5 ** (1 / half_life)
    normalization = log(2) * 7 / half_life
    tolerance = max(.1, ordinary * normalization)
    hours = planner._planned_impulse_hours(days)
    under, over, rows = [], [], []
    rate = opening
    for i, (day, state, limits, hour) in enumerate(zip(days, states, targets, hours, strict=True)):
        if day.date != state.as_of.date() or (i and (day.date-days[i-1].date).days != 1):
            raise ValueError('Require consecutive aligned physical dates')
        low, high = limits
        if not (isfinite(low) and isfinite(high) and 0 < low <= high):
            raise ValueError('Target ranges must be finite, positive and ordered')
        recommendation = day.recommendation
        miles = (planner._midpoint(recommendation) if recommendation
                 and recommendation.workout_type != planner.WorkoutType.REST else 0.)
        if not isfinite(miles) or miles < 0:
            raise ValueError('Prescribed miles must be finite and nonnegative')
        if recommendation and miles and (recommendation.planned_for is None
                                         or recommendation.planned_for.date() != day.date):
            raise ValueError('Running prescriptions require a same-day physical timestamp')
        rate = planner._advance_continuous_mileage_rate(rate, miles, daily_decay=decay,
            session_normalization=normalization, impulse_hour=hour,
            opening_hour=planner._clock_hour(state.as_of) if i == 0 else 0.)
        deficit, excess = max(0., low-tolerance-rate), max(0., rate-high-tolerance)
        under.append(deficit**2/half_life)
        over.append(excess**2/half_life)
        if trace:
            rows.append(dict(date=day.date.isoformat(),miles=miles,rate=rate,low=low,high=high,
                tolerance=tolerance,deficit=deficit,excess=excess,
                squared_violation=under[-1]+over[-1]))
    squared = sum(under) + sum(over)
    return dict(squared_violation=squared, equivalent_violation=sqrt(squared),
        cost=planner.TARGET_FIT_UNIT*squared/ordinary**2,
        underload_squared_violation=sum(under), overload_squared_violation=sum(over),
        ordinary_miles=ordinary, normalization_days=half_life, tolerance=tolerance,
        days=rows, seconds=perf_counter()-started)


def replace_volume_cost(original, old_path, path, ordinary):
    """Remove the old path once; preserve funding/rate cost and other components."""
    base = original['target_violation'] - old_path
    if base < -1e-8:
        raise ValueError('Old path exceeds the recorded target violation')
    residual = max(0., base)
    target_cost = planner.TARGET_FIT_UNIT*(residual/ordinary)**2 + path['cost']
    return dict(total=original['total']-original['target']+target_cost,
                target=target_cost, path=path['cost'], residual_target_violation=residual)


@contextmanager
def volume_path_policy(enabled=True):
    """One offline build workflow; legacy controls can replay their saved policy."""
    metrics=dict(evaluations=0,seconds=0.)
    def assess(*args,**kwargs):
        result=volume_path(*args,**kwargs)
        metrics['evaluations']+=1
        metrics['seconds']+=result['seconds']
        return result
    token=VOLUME_PATH_SCORER.set(assess if enabled else None)
    try:
        yield metrics
    finally:
        VOLUME_PATH_SCORER.reset(token)
