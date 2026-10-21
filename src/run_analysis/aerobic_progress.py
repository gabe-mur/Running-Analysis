"""Prototype sustained aerobic performance, separate from per-run pace.

Reuse the existing robust trailing estimator, but expose shared calibration
error instead of pretending that it is independent on every run or in each
comparison period. This module does not fit normalization parameters, infer
fitness from training load, or feed the scheduler. Ranges are conditional model
sensitivities, not empirically calibrated physiological confidence intervals.
"""
from collections import defaultdict
from datetime import datetime, timedelta
import math

import numpy as np

from .analytics import MIN_MEASUREMENT_SIGMA_MIN_MILE, _directional_evidence
from .uncertainty import overlapping_variance


VERSION = 'sustained-aerobic-progress-v2-variance-envelope'
EVIDENCE_DAYS = 28


def _date(row):
    at = datetime.fromisoformat(row['start_time'])
    if at.tzinfo is None:
        raise ValueError('Timezone-aware observations required')
    return at


def _period(rows, end, covariance):
    selected = [r for r in rows if end-timedelta(days=EVIDENCE_DAYS) < _date(r) <= end]
    if not selected:
        return None
    pace = np.array([r['pace_min_mile'] for r in selected])
    sigma = np.array([max(MIN_MEASUREMENT_SIGMA_MIN_MILE, r['uncertainty_95_min_mile']/1.96) for r in selected])
    context = np.array([min(1., float(r['trend_weight'])) for r in selected])
    base = context/sigma**2
    weight = base.copy()
    level = float(np.average(pace, weights=weight))
    # Identical Huber location policy to analytics._window_estimate.
    for _ in range(8):
        residual = pace-level
        scale = max(5/60, 1.4826*float(np.median(abs(residual-np.median(residual)))))
        weight = base*np.minimum(1.,1.345*scale/np.maximum(abs(residual),1e-12))
        level = float(np.average(pace, weights=weight))
    weight /= weight.sum()
    effective = float(1/(weight@weight))
    gradients = np.array([r['shared_log_speed_gradient'] for r in selected])
    pace_gradients = -pace[:,None]*gradients
    gradient = weight@pace_gradients
    # The original per-run interval includes parameter and non-parameter error.
    # Remove only the explicitly identified shared component; no invented
    # cancellation of missing terrain, fatigue, or other unmeasured effects.
    shared = np.einsum('ij,jk,ik->i', pace_gradients, covariance, pace_gradients)
    independent = np.maximum(sigma**2-shared, 0.)
    measurement_variance = float(np.sum(weight**2*independent))
    variance = float(np.sum(weight*(pace-level)**2))
    residual_scores = defaultdict(float)
    for r, w, p in zip(selected, weight, pace):
        residual_scores[math.floor(_date(r).timestamp()/86400/7)] += float(w*(p-level))
    count_blocks = len(residual_scores)
    correction = count_blocks/(count_blocks-1) if count_blocks > 1 else 1.
    block_variance = correction*sum(s*s for s in residual_scores.values())
    # Week clustering is a sensitivity guard against serially related runs,
    # never permission to replace measured scatter with a smaller standard error.
    day_variance = max(variance/max(1.,effective),block_variance)
    random_variance = overlapping_variance(measurement_variance,day_variance)
    shared_variance = max(0., float(gradient@covariance@gradient))
    total = random_variance+shared_variance
    span = (_date(selected[-1])-_date(selected[0])).total_seconds()/86400
    return dict(pace_min_mile=level, estimated_range_95_min_mile=[max(.1,level-1.96*math.sqrt(total)),level+1.96*math.sqrt(total)],
        uncertainty_95_min_mile=1.96*math.sqrt(total), run_count=len(selected), effective_run_count=effective,
        coverage_span_days=span, between_run_spread_seconds_per_mile=math.sqrt(variance)*60,
        evidence_start=selected[0]['start_time'], evidence_end=selected[-1]['start_time'],
        random_variance=random_variance, measurement_variance=measurement_variance,
        observed_variance=day_variance, shared_calibration_variance=shared_variance,
        shared_pace_gradient=gradient.tolist(), weekly_blocks=count_blocks,
        evidence='established' if effective>=3 and span>=14 else 'provisional')


def _change(current, prior, covariance, *, comparison_days=28):
    if prior is None:
        return None
    c, p = current['pace_min_mile'],prior['pace_min_mile']
    change = math.log(p/c)  # Positive = faster at the same reference effort.
    gradient = np.array(prior['shared_pace_gradient'])/p-np.array(current['shared_pace_gradient'])/c
    shared = max(0.,float(gradient@covariance@gradient))
    random = prior['random_variance']/p**2+current['random_variance']/c**2
    se = math.sqrt(shared+random)
    evidence = _directional_evidence(-change,se)
    established = current['evidence']==prior['evidence']=='established'
    # Numerical normal-tail calculations are conditional on this model; do not
    # turn two isolated, precise-looking observations into an established trend.
    return dict(speed_change_percent=100*math.expm1(change),
        estimated_change_range_95_percent=[100*math.expm1(change-1.96*se),100*math.expm1(change+1.96*se)],
        pace_change_seconds_per_mile=(c-p)*60,
        direction=evidence['directional_interpretation'] if established else 'provisional',
        evidence_strength=evidence['evidence_strength'] if established else 'provisional',
        conditional_probability_faster=evidence['probability_faster'],
        shared_calibration_log_variance=shared,
        shared_calibration_log_variance_if_periods_were_independent=(prior['shared_calibration_variance']/p**2+current['shared_calibration_variance']/c**2),
        comparison=(f'Most recent 28-day level versus the 28-day level {comparison_days} days earlier, anchored to the latest comparable run'))


def build_aerobic_progress(observations, *, as_of, covariance, history_days=90):
    """One current state; history_days crops the curve, never changes the state.

    Every history point uses its preceding 28 days and one coherent calibration
    supplied by the caller. It is a retrospectively harmonized reconstruction,
    not a claim that today's calibration was known at each historical date.
    """
    if as_of.tzinfo is None or history_days < 1:
        raise ValueError('Timezone-aware as_of and positive history length required')
    covariance = np.asarray(covariance,dtype=float)
    if covariance.ndim!=2 or covariance.shape[0]!=covariance.shape[1] or not np.isfinite(covariance).all():
        raise ValueError('Finite square calibration covariance required')
    if not np.allclose(covariance,covariance.T) or np.linalg.eigvalsh(covariance).min() < -1e-10:
        raise ValueError('Calibration covariance must be symmetric positive semidefinite')
    seen = set(); rows = []
    for row in sorted(observations,key=lambda r:(_date(r),r['activity_id'])):
        if row['activity_id'] in seen:
            raise ValueError('Duplicate physical activity')
        seen.add(row['activity_id'])
        if _date(row)>as_of or not row.get('included_in_trend') or row.get('trend_weight',0)<=0:
            continue
        gradient = np.asarray(row['shared_log_speed_gradient'],dtype=float)
        if (gradient.shape!=(len(covariance),) or not np.isfinite(gradient).all()
            or not math.isfinite(row['pace_min_mile']) or row['pace_min_mile']<=0
            or not math.isfinite(row['uncertainty_95_min_mile']) or row['uncertainty_95_min_mile']<0
            or not math.isfinite(row['trend_weight'])):
            raise ValueError('Finite positive pace and aligned uncertainty inputs required')
        rows.append(row)
    if not rows:
        return dict(version=VERSION,available=False,reason='No comparable aerobic observations')
    latest = _date(rows[-1]);current = _period(rows,latest,covariance)
    prior = _period(rows,latest-timedelta(days=EVIDENCE_DAYS),covariance)
    prior_90 = _period(rows,latest-timedelta(days=90),covariance)
    curve = []
    dates = sorted({_date(r) for r in rows})
    start = as_of-timedelta(days=history_days)
    # Keep one real predecessor so the display can clip the existing segment
    # at the viewport edge instead of restarting at the first visible run.
    predecessor = next((at for at in reversed(dates) if at < start), None)
    for at in dates:
        if start <= at or at == predecessor:
            level = _period(rows,at,covariance)
            before = _period(rows,at-timedelta(days=EVIDENCE_DAYS),covariance)
            latest_on_date = [r for r in rows if _date(r)==at][-1]
            curve.append(dict(as_of=at.isoformat(),activity_id=latest_on_date['activity_id'],
                              change=_change(level,before,covariance),**level))
    return dict(version=VERSION,available=True,evidence_days=EVIDENCE_DAYS,
        as_of=as_of.isoformat(),last_observation=latest.isoformat(),
        days_since_observation=(as_of-latest).total_seconds()/86400,
        current=current, prior=prior,change=_change(current,prior,covariance),
        change_90d=_change(current,prior_90,covariance,comparison_days=90),history=curve,
        definition='Sustained comparable aerobic performance, not training load or laboratory VO2max',
        assumptions=dict(location='Existing Huber robust, uncertainty/context-weighted 28-day pace estimate',
            shared_error='Supplied HR/time/wind/heat calibration covariance retained across runs and periods',
            remaining_error='Larger of non-parameter per-run mean uncertainty and observed between-run/weekly-cluster variance; shared calibration added separately',
            evidence='Three effective observations spanning 14 days in each period for a directional interpretation; otherwise provisional',
            historical_basis='One current calibration applied consistently to history; not a prospective backtest',
            missing_days='No new observation does not assert maintenance or detraining; show evidence age instead'),
        limitations=['A better estimate of sustained performance from the same sensors, not an independent new physiological measurement.',
            'Shared calibration cancellation applies only to identified parameter errors; biases outside the model remain.',
            'Week-level residual correlation is a sensitivity check, not a fitted biological recovery law.',
            'Ranges and directional probabilities are conditional model estimates, not empirically calibrated guarantees.'])


def progress_from_comparability(report, *, as_of, history_days=90, config=None):
    """Reuse the exact cached pace evidence; never load sensor windows twice."""
    observations=report.get('runs',[])
    if any('shared_log_speed_gradient' not in r for r in observations if r.get('included_in_trend')):
        return dict(version=VERSION,available=False,reason='Shared calibration details are unavailable')
    calibration=report['calibration']
    from .pace_temperature import comparison_covariance
    covariance=comparison_covariance(calibration)
    result=build_aerobic_progress(observations,as_of=as_of,covariance=covariance,history_days=history_days)
    if config is not None and result.get('available'):
        from .vo2_estimation import reserve_fitness_level
        def level(row):
            return reserve_fitness_level(row['pace_min_mile'],row['uncertainty_95_min_mile'],config)
        current=level(result['current'])
        result['fitness_level']=dict(current,as_of=result['as_of'],last_observation=result['last_observation'],
            target_hr_bpm=config.get('target_hr'),evidence_days=EVIDENCE_DAYS,
            history=[dict(as_of=row['as_of'],activity_id=row['activity_id'],run_count=row['run_count'],
                evidence=row['evidence'],**level(row)) for row in result['history']],
            definition='Absolute level of adjusted sustained aerobic performance on a VO2-equivalent scale',
            assumptions=['Same adjusted pace, reference HR and 28-day evidence as aerobic progress.',
                'One existing primary equation is used consistently; no quality-dependent blend of equations.',
                'Absolute-scale uncertainty includes shared equation/max-HR error; it is not independent noise at each date.',
                'Not a lab VO2max, race prediction or an independent physiological signal.'])
    return result
