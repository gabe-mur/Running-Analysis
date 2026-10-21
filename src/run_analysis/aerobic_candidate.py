"""Connected Progress prototype; never used by scheduling or the legacy VO2 card."""
from datetime import datetime, timedelta
import json
import math

import numpy as np

from .aerobic_coherent_history import CoherentHistory
from .aerobic_gaussian import GaussianChangeProfile
from .analytics import MIN_MEASUREMENT_SIGMA_MIN_MILE
from .read_cache import invariant_read
from .vo2_estimation import reserve_fitness_level


VERSION = 'coherent-aerobic-progress-v1'


def measurement_inputs(rows, cov):
    m=[];g=[]
    for row in rows:
        p=float(row['pace_min_mile']);gradient=-np.asarray(row['shared_log_speed_gradient'],float)
        precision=row['comparison_precision'];sample=precision['sampling_precision']
        shared=max(0.,float(gradient@cov@gradient))
        if sample:
            variance=(max(v['sampling_sd_min_mile'] for v in sample)/p)**2+sum(
                v for k,v in precision['log_variance_parts'].items() if k not in ('parameter','whole_run_dispersion'))
        else:
            variance=max(0.,(row['uncertainty_95_min_mile']/1.96/p)**2-shared)
        m.append(max(1e-12,(MIN_MEASUREMENT_SIGMA_MIN_MILE/p)**2-shared,variance)/min(1.,row['trend_weight']))
        g.append(gradient)
    return np.asarray(m),np.asarray(g)


def _fit(report):
    rows=sorted([r for r in report.get('runs',[]) if r.get('included_in_trend') and r.get('trend_weight',0)>0],
        key=lambda r:(r['start_time'],r['activity_id']))
    if not rows:
        return dict(version=VERSION,available=False,reason='No comparable aerobic observations')
    if any(not r.get('comparison_precision') for r in rows):
        return dict(version=VERSION,available=False,reason='Comparable run precision is unavailable')
    if len({r['activity_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate physical activity in aerobic evidence')
    from .pace_temperature import comparison_covariance
    calibration=report['calibration'];cov=comparison_covariance(calibration)
    dates=[datetime.fromisoformat(r['start_time']) for r in rows];end=dates[-1]
    times=np.array([(d-end).total_seconds()/86400 for d in dates])
    y=np.log([r['pace_min_mile'] for r in rows]);m,g=measurement_inputs(rows,cov)
    curve=CoherentHistory(y,m,g,cov,times)

    def level(anchor):
        value=curve.level(anchor)
        if not value['available']:
            return None
        ix=value['indices'];pace=value['pace_min_mile'];var=value.get('conditional_variance')
        sigma=rows[ix[0]]['uncertainty_95_min_mile'] if var is None else pace*math.expm1(1.96*math.sqrt(max(0.,var)))
        weights=np.array(value.get('weights',[1.]));effective=float(1/(weights@weights))
        span=(dates[ix[-1]]-dates[ix[0]]).total_seconds()/86400
        return dict(pace_min_mile=pace,uncertainty_95_min_mile=sigma,
            estimated_range_95_min_mile=[max(.1,pace-sigma),pace+sigma],run_count=len(ix),
            effective_run_count=effective,coverage_span_days=span,
            evidence='established' if effective>=3 and span>=14 and not value['unconverged_kernels'] else 'provisional',
            evidence_start=dates[ix[0]].isoformat(),evidence_end=dates[ix[-1]].isoformat(),
            uncertainty_kind='Conditional level sensitivity, not calibrated physiological coverage')

    def change(anchor, days, *, assess=False):
        point=curve.compare(anchor,days)
        if not point['available']:
            return None
        now=level(anchor);prior=level(anchor-days)
        result=dict(speed_change_percent=point['change_percent'],
            pace_change_seconds_per_mile=60*(now['pace_min_mile']-prior['pace_min_mile']),
            direction='estimated',evidence_strength='point_only',inference_method='bootstrap-likelihood',
            comparison_days=days,comparison='Two exact trailing 28-day levels from the same historical curve')
        if now['evidence']!='established' or prior['evidence']!='established':
            result.update(direction='provisional',evidence_strength='provisional')
            return result
        if not assess:
            return result
        ix=np.flatnonzero((times>anchor-118)&(times<=anchor))
        ages=anchor-times[ix]
        labels=np.where(ages<28,0,np.where(ages<56,1,np.where(ages>=90,2,3)))
        observed=sorted(set(labels));groups=np.array([observed.index(k) for k in labels])
        period=1 if days==28 else 2
        if period not in observed or len(ix)<=len(observed):
            return result
        contrast=np.zeros(len(observed));contrast[observed.index(0)]=-1;contrast[observed.index(period)]=1
        model=GaussianChangeProfile(y[ix],m[ix],g[ix],cov,groups,times=times[ix]-anchor,future_time=2.,mode='nuisance_residual')
        test=model.bootstrap_test(contrast,0.,draws=127,seed=20261205+days,decision_only=True)
        warnings=bool(test['observed_unconverged_kernels'] or test['optimizer_warning_draws'])
        aligned=point['estimate']*test['estimate']>0
        conclusive=not test['covered_95'] and aligned and not warnings
        result.update(direction=('improving' if point['estimate']>0 else 'declining') if conclusive else 'stable_or_uncertain',
            evidence_strength='clear' if conclusive else 'inconclusive',
            zero_change_included=test['covered_95'],bootstrap_budget=127,
            zero_change_pvalue_bounds=test['pvalue_bounds'],optimizer_warning=warnings)
        return result

    history=[]
    for i,t in enumerate(times):
        history.append(dict(as_of=dates[i].isoformat(),activity_id=rows[i]['activity_id'],
            change=change(float(t),28),**level(float(t))))
    return dict(version=VERSION,available=True,evidence_days=28,last_observation=end.isoformat(),
        current=level(0.),prior=level(-28.),change=change(0.,28,assess=True),
        change_90d=change(0.,90,assess=True),history=history,
        definition='Adjusted sustained aerobic performance; not laboratory VO2max or training load',
        assumptions=dict(location='One local Gaussian log-pace mean per exact trailing 28-day period',
            uncertainty='Shared adjustment covariance retained; joint likelihood tests do not overwrite curve points',
            history='Current normalization harmonizes history; not an as-known-then backtest'))


def progress_from_comparability(connection, report, *, as_of, history_days=90, config=None):
    if as_of.tzinfo is None or history_days<1:
        raise ValueError('Timezone-aware date and positive history required')
    # Date-dependent presentation stays outside the cache. No TTL, rounded
    # clock, sidecar research file, persisted estimate or planner refresh.
    selected={**report,'runs':[r for r in report.get('runs',[]) if datetime.fromisoformat(r['start_time'])<=as_of]}
    result=invariant_read(connection,(VERSION,json.dumps(selected,sort_keys=True)),lambda:_fit(selected))
    if not result.get('available'):
        return result
    end=datetime.fromisoformat(result['last_observation']);start=as_of-timedelta(days=history_days)
    before=[r for r in result['history'] if datetime.fromisoformat(r['as_of'])<start]
    result['history']=before[-1:]+[r for r in result['history'] if datetime.fromisoformat(r['as_of'])>=start]
    result.update(as_of=as_of.isoformat(),days_since_observation=(as_of-end).total_seconds()/86400)
    if config is not None:
        def convert(row):
            return reserve_fitness_level(row['pace_min_mile'],row['uncertainty_95_min_mile'],config)
        result['fitness_level']=dict(convert(result['current']),as_of=result['as_of'],last_observation=result['last_observation'],
            target_hr_bpm=config.get('target_hr'),evidence_days=28,
            history=[dict(as_of=r['as_of'],activity_id=r['activity_id'],run_count=r['run_count'],evidence=r['evidence'],**convert(r)) for r in result['history']],
            definition='Adjusted sustained performance on a VO2-equivalent scale, not measured VO2max')
    return result
