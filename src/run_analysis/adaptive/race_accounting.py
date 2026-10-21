"""Conditional event work in the SAME demand axes, separate from training credit.

Forecast bounds are deliberately broad. A race declaration is not medical
clearance, an observed completion, or measured distance for the spike guard.
"""
from datetime import timedelta
from math import exp
import numpy as np

from .program_objective_trial import AXES, demand_value
from .soft_horizon_trial import day_weight
from .race_event import dated_purpose_fraction, no_race_recovery


def event_demand(revision, references):
    """Use supported athlete forecasts, not an invented finish pace/HR.

    Running-axis exposure scales from the long forecast; duration/cardio/work
    from the supported quality forecast. Broad bounds disclose the extrapolation
    from intermittent quality to sustained racing; this is not calibration.
    """
    from run_analysis.workload_accounting import Quantity
    if revision is None or revision.goal is None:
        return ()
    refs = {role: (miles, demand) for role, miles, demand in references}
    miles = revision.goal.profile.distance_miles
    result = []
    for axis in range(3):
        reference_miles, demand = refs['long' if axis == 0 else 'quality']
        q = demand[axis]
        result.append(Quantity(q.low*miles/reference_miles*.8,
            q.high*miles/reference_miles*1.5, None,
            'conditional race extrapolation from supported athlete forecast; not a confidence interval'))
    # Race effort is sustained, not the interval work-to-total-distance ratio.
    # Keep a broad 25–100% work-duration scenario; do not silently price a
    # marathon as a few repetitions spread across 26 miles.
    result[2] = Quantity(result[0].low*.25,result[0].high,None,
        'conditional sustained race-work fraction 25–100%; not measured quality equivalence')
    return tuple(result)


def event_cost(context, future, *, day_weight_function=None):
    if not context.race_demand or context.race_goal.race_date < context.decision:
        return None
    day = context.race_goal.race_date
    nominal = lambda q: np.array([demand_value(v, context.scenario.bound_fraction) for v in q])/context.scales
    demand = nominal(context.race_demand)
    axes = .5*demand*demand
    for row in (*context.history, *future):
        # The reserved event date has no candidate training prescription.
        # The shared admission adapter rejects event-date collisions. Keeping
        # their score finite lets search report/reject them without crashing.
        weight = 1. if row.source == 'observed' else (day_weight(context, row.day)
            if day_weight_function is None else day_weight_function(row.day))
        age = abs((day-row.day).days)-context.scenario.timing_compression_days
        axes += weight*demand*nominal(row.demand)*exp(-age/context.scenario.memory_days)
    axes *= 7/(context.end-context.stage_start).days
    raw = float(np.average(axes, weights=context.scenario.axis_weights))
    return dict(raw_demand_cost=raw, cost=raw*context.exchange,
        demand_by_axis=dict(zip(AXES, axes.tolist())), date=day,
        training_credit=0., observed=False, participation_cleared=False,
        forecast_basis='bounded quality/long athlete references; sustained race uncertainty retained')


def apply_purpose_window(context, temporal):
    revision = context.race_goal
    if (revision is None or revision.goal is None) and not context.race_goal_history:
        return temporal
    output = {}
    for name, value in temporal.items():
        daily = []
        for row in value['daily']:
            factor = dated_purpose_fraction(context,row['day'],name)
            daily.append(dict(row,loss=row['loss']*factor))
        loss = sum(row['loss'] for row in daily)/(context.end-context.stage_start).days
        from .program_objective_trial import PURPOSES
        output[name] = dict(value, daily=daily, loss=loss,
            weighted_loss=loss*context.policy.purpose_weights[PURPOSES.index(name)],
            dated_race_purpose_relief=True)
    return output


def allowed_dose(revision, day, dose, stage, resolution=None):
    if revision is None or revision.goal is None:
        return True
    phase = revision.phase(day)
    if no_race_recovery(revision, resolution, day):
        return True
    if phase == 'race':
        return False
    if phase == 'post_race' and dose.role != 'easy':
        return False
    if phase == 'taper' and dose.role == 'long':
        return dose.miles <= stage.long_miles*revision.training_fraction(day)+1e-8
    if phase == 'taper' and dose.role == 'quality':
        return dose.key == 'quality:reduced'
    return True
