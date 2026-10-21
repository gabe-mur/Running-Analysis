"""Pure actual-future-run error calibration for the opt-in WIP pace outlook.

Audited fractional-error quantiles and short-horizon error growth. No database,
research-file reads, planner calls, fitted physiological ceiling or probability
coverage claim. A range must be earned from completed out-of-sample forecasts.
"""
from collections import Counter
import math

import numpy as np

from .training_response import fit_response


def weights(rows):
    origins=Counter(r['origin'] for r in rows)
    outcomes=Counter(i for r in rows for i in r['outcome_ids'])
    w=np.array([sum(1/outcomes[i] for i in r['outcome_ids'])/len(r['outcome_ids'])/
        origins[r['origin']] for r in rows])
    return w/w.sum()


def weighted_quantile(values,w,probabilities):
    values,w=np.asarray(values,float),np.asarray(w,float)
    if not len(values) or np.any(w<=0) or not np.isfinite(values).all() or not np.isfinite(w).all():
        raise ValueError('Finite observations and positive weights required')
    ix=np.argsort(values,kind='stable');x,w=values[ix],w[ix]
    positions=(np.cumsum(w)-w/2)/w.sum()
    return np.interp(probabilities,positions,x)


def matched_error_growth(rows):
    """Same-origin two/four-week error growth; no forced sqrt-time scaling."""
    a={r['origin']:r for r in rows if r['horizon']==14}
    b={r['origin']:r for r in rows if r['horizon']==28}
    common=sorted(set(a)&set(b))
    if len(common)<8:return None
    values=[]
    for group in ([a[o] for o in common],[b[o] for o in common]):
        w=weights(group)
        values.append((float(w@np.array([r['elapsed']*28 for r in group])),
            float(np.sqrt(w@np.square([r['log_error'] for r in group])))))
    if min(v for pair in values for v in pair)<=0 or values[1][0]<=values[0][0]:return None
    power=math.log(values[1][1]/values[0][1])/math.log(values[1][0]/values[0][0])
    # Negative/undefined growth is not silently clipped to a chosen positive
    # number. Such history has not earned this extrapolated range yet.
    if not math.isfinite(power) or power<0:return None
    return dict(exponent=power,matched_origins=len(common),
        latest_outcome=max(r['outcome_end'] for r in rows if r['origin'] in common and r['horizon'] in {14,28}))


def response_fits(policies,origin):
    fits={policy:{model:fit_response([r for r in rows if r['outcome_end']<origin],model)
        for model in ('calendar','log_miles')} for policy,rows in policies.items()}
    return None if any(f is None for variants in fits.values() for f in variants.values()) else fits


def response_midpoint(fits,row,start):
    values=[start*math.exp(fits[p][m]['coefficients']['elapsed' if m=='calendar' else 'log_miles']*
        row['elapsed' if m=='calendar' else 'log_miles']) for p in fits for m in ('calendar','log_miles')]
    return (min(values)+max(values))/2


def error_history(all_responses,groups_by_bandwidth,raw_targets):
    """Reproduce audited 28/42 joint hindcasts; actual future runs are target.

    Cohort membership is conditional on completed future training. Every fit
    outcome ends before its forecast origin; no today's retrospective targets.
    """
    errors={days:[] for days in groups_by_bandwidth}
    primary='gap7_fraction0.500'
    for origin in sorted({r['origin'] for r in raw_targets}):
        fits={days:response_fits(groups,origin) for days,groups in groups_by_bandwidth.items()}
        if any(value is None for value in fits.values()):continue
        rows_by_bw={days:{r['target_id']:r for r in by_policy[primary] if r['origin']==origin}
            for days,by_policy in all_responses.items()}
        for source in (r for r in raw_targets if r['origin']==origin):
            for days in errors:
                row=rows_by_bw[days][source['target_id']]
                predicted=response_midpoint(fits[days],row,row['start_pace'])
                errors[days].append(dict(origin=origin,target_id=row['target_id'],horizon=row['horizon'],
                    outcome_ids=row['outcome_ids'],outcome_end=row['outcome_end'],elapsed=row['elapsed'],
                    actual=source['actual'],predicted=predicted,log_error=math.log(predicted/source['actual']),
                    regime=row['consistency']['regime']))
    return errors


def calibrate(errors,*,candidate=42,control=28):
    growth=matched_error_growth(errors[control])
    rows=[r for r in errors[candidate] if r['horizon'] in {14,28} and r['regime']=='consistent']
    if (growth is None or len({r['origin'] for r in rows})<8
            or len({i for r in rows for i in r['outcome_ids']})<3):return None
    return dict(exponent=growth['exponent'],errors=rows,matched_origins=growth['matched_origins'],
        minimum_horizon_days=min(r['elapsed']*28 for r in rows),
        calibration_origins=len({r['origin'] for r in rows}),calibration_outcome_runs=len({i for r in rows for i in r['outcome_ids']}),
        latest_calibration_outcome=max(r['outcome_end'] for r in rows),
        latest_growth_outcome=growth['latest_outcome'])


def error_radius(calibration,days):
    """Working range only at/after the earliest tested elapsed horizon.

    A power fitted to multi-week forecast errors says nothing about present
    estimate uncertainty. In particular, zero elapsed time is NOT zero error.
    Do not extend a sublinear power law down to a falsely certain origin.
    """
    if days<0 or not math.isfinite(days):raise ValueError('Nonnegative finite horizon required')
    rows=calibration['errors'];power=calibration['exponent']
    if days<min(r['elapsed']*28 for r in rows):return None
    values=np.array([abs(r['log_error'])*(days/(r['elapsed']*28))**power for r in rows])
    return float(weighted_quantile(values,weights(rows),[.8])[0])
