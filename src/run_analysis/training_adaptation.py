"""Experimental separation of training-linked change and temporary innovation.

No absolute starting-pace regressor or repeated return-to-population-mean term.
These are statistical components, not independently measured fitness/recovery.
"""
from collections import Counter
from datetime import datetime,timedelta
import math

import numpy as np

from .aerobic_forecast_core import fit_trend

KEYS=('elapsed','trend','training_change','temporary')


def training_stock(training,times,*,memory_days=28.,input_days=7.):
    """Two-stage causal exposure: recent running, then delayed accumulated dose.

    At completion an activity adds no instantaneous adaptation. Every completed
    run contributes continuously afterward. At a steady rate the state has
    weekly-mile units. Memory constants are tested assumptions, not biomarkers.
    """
    if min(memory_days,input_days)<=0 or not all(math.isfinite(v) for v in (memory_days,input_days)):
        raise ValueError('Positive finite training memory required')
    output=[]
    for at in times:
        value=0.
        for run in training:
            age=(at-run.get('available_at',run['time'])).total_seconds()/86400
            if age<=0:continue
            if abs(memory_days-input_days)<1e-9:
                kernel=7*age/(memory_days**2)*math.exp(-age/memory_days)
            else:
                kernel=7/(memory_days-input_days)*(math.exp(-age/memory_days)-math.exp(-age/input_days))
            value+=run['miles']*kernel
        output.append(value)
    return np.array(output)


def add_components(examples,rows,variance,training,*,memory_days=28.,temporary_days=7.):
    """Causal start innovation; delayed training change at EACH future outcome."""
    if not math.isfinite(temporary_days) or temporary_days<=0:
        raise ValueError('Positive finite temporary-response memory required')
    dates=[datetime.fromisoformat(r['start_time']) for r in rows]
    y=np.log([r['pace_min_mile'] for r in rows])
    by_id={r['activity_id']:i for i,r in enumerate(rows)}
    times=np.array([(d-dates[0]).total_seconds()/86400 for d in dates])
    output=[]
    for e in examples:
        i=by_id[e['origin_id']];origin=datetime.fromisoformat(e['origin'])
        # Innovation is a change from the older, already available trajectory,
        # not a label identifying that change as fatigue or true fitness.
        older=np.flatnonzero(times[:i+1] <= times[i]-7)
        if not len(older):continue
        fit=fit_trend(y[older],times[older],variance[older])
        if fit is None:continue
        expected=fit['log_level']+fit['slope']*((origin-dates[0]).total_seconds()/86400-times[older[-1]])
        innovation=e['start_level']-expected
        outcome_dates=[datetime.fromisoformat(v['at']) for v in e['training_by_outcome']]
        weights=np.array([v['weight'] for v in e['training_by_outcome']])
        stocks=training_stock(training,[origin]+outcome_dates,memory_days=memory_days)
        if min(stocks)<=0:continue
        elapsed=np.array([(d-origin).total_seconds()/86400 for d in outcome_dates])
        output.append({**e,'trend':e['start_trend'],
            'training_change':float(weights@np.log(stocks[1:])-math.log(stocks[0])),
            'temporary':float(innovation*(weights@(-np.expm1(-elapsed/temporary_days)))),
            'start_innovation':innovation,'start_stock':float(stocks[0]),
            'future_stock':float(math.exp(weights@np.log(stocks[1:]))),
            'memory_days':memory_days,'temporary_days':temporary_days})
    return output


def fit_change(examples,*,keys=KEYS,penalty=4.):
    """Zero-intercept change regression: zero elapsed/dose change => zero gain.

    RMS scaling preserves the zero anchor. Temporary displacement is a nuisance
    component and is never recycled as training adaptation in a long rollout.
    """
    if not math.isfinite(penalty) or penalty<=0:
        raise ValueError('Positive finite fitting penalty required')
    if len(examples)<8:return None
    x=np.array([[e[k] for k in keys] for e in examples]);y=np.array([e['change'] for e in examples])
    if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Finite training examples required')
    counts=Counter(e['target_id'] for e in examples)
    w=np.array([1/counts[e['target_id']] for e in examples])
    scales=np.sqrt(np.average(x*x,axis=0,weights=w));scales[scales<1e-10]=1
    z=x/scales
    beta=np.linalg.solve(z.T@(w[:,None]*z)+penalty*np.eye(len(keys)),z.T@(w*y))/scales
    return dict(keys=tuple(keys),coefficients={k:float(v) for k,v in zip(keys,beta)},
        examples=len(examples),latest_outcome=max(e['outcome_end'] for e in examples))


def change_components(fit,row):
    return {key:fit['coefficients'][key]*row[key] for key in fit['keys']}


def projected_training(training,at,days,*,volume,growth,target_volume,memory_days=28.):
    """Conditional exposure path; completed history plus expected daily volume.

    Daily fractional mileage is an integration approximation, NOT a prescribed
    daily run. No planner search is performed. Current training is never
    replaced with the target volume; the delayed state evolves from history.
    """
    if not isinstance(days,int) or not 0<=days<=365:
        raise ValueError('Projection within one year required')
    if not all(math.isfinite(v) for v in (volume,growth,target_volume)) or min(volume,target_volume)<=0 or growth<=0:
        raise ValueError('Positive finite progression inputs required')
    if any(r.get('available_at',r['time'])>at for r in training):
        raise ValueError('Future recordings cannot initialize exposure')
    future=[dict(time=at+timedelta(days=d+.5),miles=min(target_volume,volume*(1+growth)**((d+.5)/7))/7)
        for d in range(days)]
    return training_stock([*training,*future],[at+timedelta(days=d) for d in range(days+1)],memory_days=memory_days)
