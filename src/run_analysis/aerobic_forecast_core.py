"""Shared, pure helpers for the audited training-conditioned pace prototype.

No planner calls, database writes or research-script imports.
"""
from collections import Counter
from datetime import datetime, timedelta
import math

import numpy as np

from .aerobic_performance import LocalPaceTrend


def damped_days(horizon, half_life):
    if not math.isfinite(horizon) or horizon < 0:
        raise ValueError('Nonnegative finite horizon required')
    if half_life is None:
        return horizon
    if not math.isfinite(half_life) or half_life <= 0:
        raise ValueError('Positive finite trend half life required')
    tau = half_life / math.log(2)
    return -tau * math.expm1(-horizon / tau)


def fit_trend(y, t, variance, *, history_days=84, feature=None):
    """Independent run fit with training-only weights/feature centering."""
    y, t, variance = (np.asarray(v, dtype=float) for v in (y, t, variance))
    if not (len(y) == len(t) == len(variance)) or len(y) < 3:
        return None
    if not all(np.isfinite(v).all() for v in (y, t, variance)) or np.any(variance <= 0):
        raise ValueError('Finite aligned observations and positive variance required')
    age = t - t[-1]
    ix = age >= -history_days
    if feature is not None:
        feature = np.asarray(feature, dtype=float)
        if feature.shape != y.shape:
            raise ValueError('Aligned feature required')
        ix &= np.isfinite(feature)
    indices = np.flatnonzero(ix)
    if len(indices) < 8 or np.ptp(t[ix]) < 28:
        return None
    x = np.column_stack([np.ones(len(indices)), age[ix] / 84])
    center = scale = None
    if feature is not None:
        center = float(np.mean(feature[ix]))
        scale = float(np.std(feature[ix]))
        if scale < 1e-10:
            return None
        x = np.column_stack([x, (feature[ix] - center) / scale])
    w = 1 / variance[ix]
    w /= w.sum()
    effective = float(1 / (w @ w))
    if effective <= x.shape[1] + 1 or np.linalg.matrix_rank(x) < x.shape[1]:
        return None
    root = np.sqrt(w)
    beta = np.linalg.lstsq(root[:, None] * x, root * y[ix], rcond=None)[0]
    residual = y[ix] - x @ beta
    return dict(log_level=float(beta[0]), slope=float(beta[1] / 84),
        feature_coefficient=float(beta[2]) if feature is not None else None,
        feature_center=center, feature_scale=scale, count=len(indices),
        effective_count=effective, span_days=float(np.ptp(t[ix])),
        rms_log_error=float(np.sqrt(w @ residual**2)))


def training_between(runs, start, end):
    """Physical completed running, after origin and strictly before outcomes.

    A manual exclusion from aerobic scoring does not erase completed training.
    Non-running activities are excluded explicitly by the data reader.
    """
    if end <= start:
        raise ValueError('Positive training interval required')
    chosen = [r for r in runs if start < r['time'] < end and r.get('available_at',r['time']) <= end]
    days = (end-start).total_seconds()/86400
    factor = 7/days
    return dict(volume=sum(r['miles'] for r in chosen)*factor,
        frequency=len(chosen)*factor,
        minutes=sum(r['minutes'] for r in chosen)*factor,
        quality=sum(r['quality'] for r in chosen)*factor,
        long_miles=sum(r['miles'] for r in chosen if r['long'])*factor,
        total_miles=sum(r['miles'] for r in chosen), days=days,
        run_ids=[r['id'] for r in chosen])


def outcome_training(training,origin,dates,weights,fatigue,zone):
    """Match every outcome to its own preceding training, never its own load.

    Since the outcome is a weighted mean of log pace and our candidate models
    are linear in their inputs, use the SAME weights for each input. Earlier
    outcome runs count as completed training for later outcomes, not themselves.
    """
    weights=np.asarray(weights,float)
    if len(dates)!=len(weights) or not len(dates) or np.any(weights<0) or not np.isfinite(weights).all() or weights.sum()<=0:
        raise ValueError('Aligned finite positive outcome weights required')
    weights=weights/weights.sum()
    exposures=[training_between(training,origin,at) for at in dates]
    fields=('volume','frequency','minutes','quality','long_miles','total_miles','days')
    aggregate={key:float(weights@np.array([e[key] for e in exposures])) for key in fields}
    union=sorted({i for e in exposures for i in e['run_ids']})
    ids=set(union)
    aggregate['run_ids']=union
    aggregate['union_miles']=sum(r['miles'] for r in training if r['id'] in ids)
    values=[fatigue.get(at.astimezone(zone).date()-timedelta(days=1)) for at in dates]
    aggregate['end_fatigue']=float(weights@values) if all(v is not None for v in values) else None
    aggregate['by_outcome']=[dict(at=at.isoformat(),weight=float(weight),
        **exposure) for at,weight,exposure in zip(dates,weights,exposures)]
    return aggregate


def make_examples(rows, variance, training, fatigue, zone, *, horizons=(14,28,56,84)):
    dates = [datetime.fromisoformat(r['start_time']) for r in rows]
    t = np.array([(d-dates[0]).total_seconds()/86400 for d in dates])
    y = np.log([r['pace_min_mile'] for r in rows])
    examples = []
    for i, recorded_start in enumerate(dates):
        origin=datetime.fromisoformat(rows[i].get('available_at',rows[i]['start_time']))
        if i < 7 or t[i]-t[0] < 28:
            continue
        gaps = np.flatnonzero(np.diff(t[:i+1])>28)
        first = int(gaps[-1]+1) if len(gaps) else 0
        fit = fit_trend(y[first:i+1],t[first:i+1],variance[first:i+1])
        if fit is None:
            continue
        curve = LocalPaceTrend(y[first:i+1],variance[first:i+1],
            np.zeros((i+1-first,0)),np.zeros((0,0)),t[first:i+1])
        level = curve.level(t[i])['log_pace']
        prior = training_between(training,origin-timedelta(days=28),origin)
        for horizon in horizons:
            selected = [j for j in range(i+1,len(rows)) if abs((dates[j]-origin).total_seconds()/86400-horizon)<=7]
            if len(selected)<2 or np.any(np.diff(t[i:max(selected)+1])>28):
                continue
            start=dates[selected[0]]
            end=max(datetime.fromisoformat(rows[j].get('available_at',rows[j]['start_time'])) for j in selected)
            weights = 1/variance[selected];weights /= weights.sum()
            delivered=outcome_training(training,origin,[dates[j] for j in selected],weights,fatigue,zone)
            elapsed = float(weights@t[selected]-(origin-dates[0]).total_seconds()/86400)
            actual = float(weights@y[selected])
            examples.append(dict(origin=origin.isoformat(), outcome_start=start.isoformat(),
                outcome_end=end.isoformat(), horizon=horizon,
                origin_id=rows[i]['activity_id'],
                target_id=','.join(str(rows[j]['activity_id']) for j in selected),
                outcome_ids=[rows[j]['activity_id'] for j in selected],
                outcome_runs=len(selected), start_pace=math.exp(level), actual=math.exp(actual),
                change=actual-level, elapsed=elapsed/28,
                start_level=level, start_trend=fit['slope']*damped_days(elapsed,56),
                linear_trend=fit['slope']*elapsed,
                prior_volume=prior['volume'], prior_quality=prior['quality'],
                delivered_volume=delivered['volume'], delivered_quality=delivered['quality'],
                volume_change=delivered['volume']-prior['volume'],
                delivered_frequency=delivered['frequency'],delivered_minutes=delivered['minutes'],
                delivered_long_miles=delivered['long_miles'], delivered_total_miles=delivered['total_miles'],
                delivered_run_ids=delivered['run_ids'], exposure_days=delivered['days'],
                delivered_union_miles=delivered['union_miles'],
                training_by_outcome=delivered['by_outcome'],end_fatigue=delivered['end_fatigue'],
                response_variance=float(weights@variance[selected]+variance[i])))
    return examples


def ridge_fit(rows, keys, *, penalty=4.):
    """Fixed statistical shrinkage, training-only scaling, no sign constraints."""
    if penalty<0 or not math.isfinite(penalty):
        raise ValueError('Finite nonnegative penalty required')
    if len(rows)<8:
        return None
    x = np.array([[r[k] for k in keys] for r in rows],float)
    y = np.array([r['change'] for r in rows])
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Known training covariates required')
    # Overlapping transitions are not independent new votes. Repeated exact
    # outcome blocks share one total weight. We report remaining overlap too.
    counts = Counter(r['target_id'] for r in rows)
    w = np.array([1/counts[r['target_id']] for r in rows])
    mean = np.average(x,axis=0,weights=w)
    scale = np.sqrt(np.average((x-mean)**2,axis=0,weights=w));scale[scale<1e-10]=1
    z = np.column_stack([np.ones(len(rows)),(x-mean)/scale])
    regularizer = np.diag([0.]+[penalty]*len(keys))
    beta = np.linalg.lstsq(z.T@(w[:,None]*z)+regularizer,z.T@(w*y),rcond=None)[0]
    return dict(keys=keys,mean=mean,scale=scale,beta=beta,training_examples=len(rows),
        training_targets=len(counts),latest_outcome=max(r['outcome_end'] for r in rows))


def predict_change(fit, row):
    x = np.array([row[k] for k in fit['keys']],float)
    return float(np.r_[1.,(x-fit['mean'])/fit['scale']]@fit['beta'])

