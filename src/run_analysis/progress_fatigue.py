"""Read-only display of the scheduler's residual demand, not a readiness model.

Use its verified load ledger, normalization, daily age and demand scenario.
The display's weighted residual index is NOT the optimizer's programme score:
the cost of the next session also depends on that session's own dose. Historical
values are reconstructed with the current committed model, not archived forecasts.
"""
from datetime import datetime, time, timedelta
from math import ceil, exp
from statistics import mean, median
from zoneinfo import ZoneInfo

from .adaptive.program_objective_trial import AXES, DemandScenario, demand_value
from .workload_accounting import Quantity


def residual(rows, day, scales, scenario, zone, *, entering=False):
    """Same normalized preceding-demand sum as objective_trial.components.

    Before a session: exclude its entire local date, matching daily scheduling.
    History: include completed work on the date. No intraday recovery is invented.
    """
    totals = [0., 0., 0.]
    for row in rows:
        age = (day-datetime.fromisoformat(row['start_time']).astimezone(zone).date()).days
        if age < 0 or entering and age == 0:
            continue
        decay = exp(-age/scenario.memory_days)
        for i, axis in enumerate(AXES):
            totals[i] += demand_value(Quantity(**row[axis]), scenario.bound_fraction)/scales[i]*decay
    return dict(by_axis=dict(zip(AXES, totals)),
        value=sum(v*w for v,w in zip(totals, scenario.axis_weights))/sum(scenario.axis_weights))


def fatigue_history(ledger, sessions, points, details, end, window_days):
    empty = dict(status='unavailable', reason=ledger.get('reason') or 'No verified scheduler history.',
        daily=[], runs=[], current=None, usual=None, comparison=None, comparison_days=42)
    model = ledger.get('fatigue_model')
    if not model or not model['history_start']:
        return empty
    zone = ZoneInfo(model['timezone'])
    scenario = DemandScenario(**model['scenario'])
    scales = model['scales']
    rows = [r for r in ledger['rows'] if r['accounting_key'] == model['accounting_key']]
    found = {r['activity_id'] for r in rows}
    today = end.astimezone(zone).date()
    first = datetime.fromisoformat(model['history_start']).astimezone(zone).date()
    # Five e-folds is a coverage check (<0.7% remaining per old contribution),
    # not a reset: every available earlier row still contributes to the sum.
    lead_days = ceil(5*scenario.memory_days)
    missing = [s.start_time.astimezone(zone).date() for s in sessions
        if s.activity_id not in found and s.start_time <= end]

    def sample(day, *, entering=False):
        complete = first <= day-timedelta(days=lead_days) and not any(
            day-timedelta(days=lead_days) <= d <= day and (not entering or d < day) for d in missing)
        nominal = residual(rows, day, scales, scenario, zone, entering=entering)
        bounds = [residual(rows, day, scales, DemandScenario(**{**model['scenario'], 'bound_fraction': f}),
            zone, entering=entering)['value'] for f in (0., 1.)]
        return dict(complete=complete, value=nominal['value'] if complete else None,
            low=bounds[0] if complete else None, high=bounds[1] if complete else None,
            by_axis=nominal['by_axis'] if complete else None)

    daily = []
    for offset in range(max(window_days, 28)+1):
        day = today-timedelta(days=max(window_days, 28)-offset)
        at = min(datetime.combine(day, time(12), zone), end)
        daily.append(dict(as_of=at.isoformat(), **sample(day)))
    recent = [r['value'] for r in daily[-28:] if r['complete']]
    usual = mean(recent) if len(recent) == 28 and sum(recent) > 0 else None
    current = daily[-1]
    runs = []
    for s in sessions:
        point = points.get(s.activity_id)
        if not end-timedelta(days=42) < s.start_time <= end or not point or not point.included_in_trend:
            continue
        if details[s.activity_id]['health_tag'] != 'normal':
            continue
        before = sample(s.start_time.astimezone(zone).date(), entering=True)
        if not before['complete'] or point.standardized_pace_min_mile is None:
            continue
        runs.append(dict(activity_id=s.activity_id, start_time=s.start_time.isoformat(),
            adjusted_pace=point.standardized_pace_min_mile, uncertainty_95=point.uncertainty_95_min_mile,
            fatigue=before['value'], low=before['low'], high=before['high']))
    comparison = None
    if len(runs) >= 6:
        boundary = median(r['fatigue'] for r in runs)
        lower = [r for r in runs if r['fatigue'] < boundary]
        higher = [r for r in runs if r['fatigue'] > boundary]
        if len(lower) >= 3 and len(higher) >= 3:
            comparison = dict(lower_runs=len(lower), higher_runs=len(higher),
                lower_pace=mean(r['adjusted_pace'] for r in lower),
                higher_pace=mean(r['adjusted_pace'] for r in higher),
                difference_seconds=(mean(r['adjusted_pace'] for r in higher)-mean(r['adjusted_pace'] for r in lower))*60)
    return dict(status='available' if current['complete'] else 'incomplete',
        reason=None if current['complete'] else 'Recent runs are missing verified scheduler accounting.',
        daily=daily[-(window_days+1):], runs=runs, current=current, usual=usual,
        comparison=comparison, comparison_days=42, memory_days=scenario.memory_days,
        coverage_lead_days=lead_days, snapshot_as_of=ledger['snapshot_as_of'],
        model_basis='Current committed scheduler; daily accounting; historical reconstruction',
        unit='Weighted normalized residual demand; not recovery percent or programme score')
