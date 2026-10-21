"""Read-only WIP pace outlook, conditional on the goal's volume progression.

42-day responses fitted to consistent training, with actual-future-run errors.
This is not a planner objective, physiological fitness model, race predictor or
calibrated long-horizon confidence interval. No research artifacts are read.
"""
from datetime import date, datetime, timedelta
import json
import math
from zoneinfo import ZoneInfo

import numpy as np

from .aerobic_candidate import measurement_inputs
from .aerobic_performance import LocalPaceTrend
from .aerobic_forecast_core import make_examples
from .pace_temperature import comparison_covariance
from .progress import prepare_progress_data
from .read_cache import invariant_read
from .training_response import (HORIZONS,POLICIES,annotate,benchmark_examples,benchmark_levels,
    apply_benchmark_levels,classify_consistency,features,fit_response)
from .pace_forecast_error import calibrate,error_history,error_radius

VERSION='42-day-empirical-error-outlook-v5'
SMOOTHING_DAYS=42.


def projected_curve(fits,*,start,end,current_pace,prior_miles,volume,growth,target_volume,error_calibration=None):
    """Response midpoint; production bounds use observed forecast errors.

    Four training-only consistency definitions, each fitting sustained observed
    log-pace change per consistent training day and log-cumulative-mile response.
    The center is the arithmetic midpoint of scenario extrema, NOT a separately
    fitted expected value. With error_calibration, range bounds are replaced
    by empirical error bounds. Without it the legacy scenario envelope remains
    available for audit controls; the live API always requires calibration.
    No absolute-pace reversion, imposed time taper, or zero-growth floor.
    """
    horizon=(end-start).days
    if not 0<horizon<=365 or not all(math.isfinite(v) for v in (current_pace,prior_miles,volume,growth,target_volume)):
        raise ValueError('Finite inputs and a projection within one year required')
    if min(current_pace,prior_miles,volume,target_volume)<=0 or growth<=0:
        raise ValueError('Positive pace, completed training and progression required')
    if error_calibration is not None and error_radius(error_calibration,horizon) is None:
        raise ValueError('Target is earlier than the tested forecast-error horizons')
    if set(fits)!=set(POLICIES) or any(set(fits[p])!={'calendar','log_miles'} for p in POLICIES):
        raise ValueError('All fixed response variants required')
    days=np.arange(horizon+1)
    miles=np.r_[0.,np.cumsum([min(target_volume,volume*(1+growth)**((d-.5)/7))/7 for d in range(1,horizon+1)])]
    paths=[]
    for policy in POLICIES:
        for model in ('calendar','log_miles'):
            row=days/28 if model=='calendar' else np.log1p(miles/prior_miles)
            key='elapsed' if model=='calendar' else 'log_miles'
            changes=fits[policy][model]['coefficients'][key]*row
            if not np.isfinite(changes).all() or np.any(np.abs(changes)>2):
                raise ValueError('Unusable projected response')
            paths.append(current_pace*np.exp(changes))
    points=[]
    for offset in sorted(set(range(0,horizon+1,7))|{horizon}):
        values=[path[offset] for path in paths]
        low=float(min(values));high=float(max(values))
        center=(low+high)/2
        if error_calibration is not None:
            radius=error_radius(error_calibration,offset)
            low,high=(None,None) if radius is None else (center*math.exp(-radius),center*math.exp(radius))
        points.append(dict(date=(start+timedelta(days=offset)).isoformat(),
            pace_min_mile=center,low_pace_min_mile=low,high_pace_min_mile=high))
    return points


def _fit_history(connection,report,performance,as_of,zone):
    hidden=set(performance.get('hidden_activity_ids',()))
    rows=sorted([dict(r) for r in report.get('runs',()) if r.get('included_in_trend')
        and r.get('trend_weight',0)>0 and r['activity_id'] not in hidden
        and not r.get('manually_excluded') and r.get('workout_type') not in {'hike','bike'}
        and datetime.fromisoformat(r['start_time'])<=as_of],key=lambda r:(r['start_time'],r['activity_id']))
    if len(rows)<8:
        return None
    # Do not combine an old training era across a recording break with a new
    # one just to manufacture enough training-response examples.
    gaps=[i for i in range(1,len(rows)) if (datetime.fromisoformat(rows[i]['start_time'])-
        datetime.fromisoformat(rows[i-1]['start_time'])).total_seconds()>28*86400]
    if gaps:rows=rows[gaps[-1]:]
    metadata={r['id']:dict(r) for r in connection.execute('SELECT id,start_time_utc,total_elapsed_time_s FROM activities')}
    for r in rows:
        meta=metadata[r['activity_id']]
        r['available_at']=(datetime.fromisoformat(meta['start_time_utc'])+timedelta(seconds=meta['total_elapsed_time_s'] or 0)).isoformat()
    rows=[r for r in rows if datetime.fromisoformat(r['available_at'])<=as_of]
    if len(rows)<8 or rows[-1]['start_time']!=performance.get('last_observation'):
        return None
    covariance=comparison_covariance(report['calibration'])
    m,g=measurement_inputs(rows,covariance)
    variance=m+np.einsum('ij,jk,ik->i',g,covariance,g)
    prepared=prepare_progress_data(connection)
    training=[];health=[]
    for s in prepared.sessions:
        role=prepared.details[s.activity_id]['workout'].value
        if role in {'hike','bike'}:continue
        end=s.start_time+timedelta(seconds=metadata[s.activity_id]['total_elapsed_time_s'] or 0)
        if end>as_of:continue
        tag=prepared.details[s.activity_id].get('health_tag','normal')
        training.append(dict(id=s.activity_id,time=s.start_time,available_at=end,miles=s.distance_miles,
            minutes=s.moving_minutes,quality=role in {'intervals','tempo_threshold','race'},long=role=='long'))
        if tag!='normal':health.append(dict(at=s.start_time,status=tag,activity_id=s.activity_id))
    if connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='user_health_reports'").fetchone():
        health.extend(dict(at=datetime.fromisoformat(r['reported_at']),status=r['status'])
            for r in connection.execute('SELECT reported_at,status FROM user_health_reports')
            if datetime.fromisoformat(r['reported_at'])<=as_of)
    examples=make_examples(rows,variance,training,{},zone,horizons=HORIZONS)
    completed=[e for e in examples if datetime.fromisoformat(e['outcome_end'])<as_of]
    era_start=datetime.fromisoformat(rows[0]['start_time'])
    raw_responses=features(completed,training,era_start)
    measured=annotate(raw_responses,training,health,era_start=era_start)
    levels={days:benchmark_levels(rows,m,g,covariance,bandwidth_days=days) for days in (28,SMOOTHING_DAYS)}
    all_by_bandwidth={days:{policy:[{**r,'consistency':classify_consistency(r['consistency'],**options)}
        for r in apply_benchmark_levels(measured,values)] for policy,options in POLICIES.items()}
        for days,values in levels.items()}
    groups_by_bandwidth={days:{policy:[r for r in responses if r['consistency']['regime']=='consistent']
        for policy,responses in policies.items()} for days,policies in all_by_bandwidth.items()}
    groups=groups_by_bandwidth[SMOOTHING_DAYS]
    fits={policy:{model:fit_response(groups[policy],model,as_of=as_of)
        for model in ('calendar','log_miles')} for policy in POLICIES}
    if any(f is None for variants in fits.values() for f in variants.values()):return None
    # Preserve the exact audited 42-day candidate: error growth is learned
    # from the 28-day CONTROL's actual-run misses, then applied to 42-day
    # consistent-training misses. No smoothed benchmark is an error target.
    calibration=calibrate(error_history(all_by_bandwidth,groups_by_bandwidth,raw_responses))
    if calibration is None:return None
    dates=[datetime.fromisoformat(r['start_time']) for r in rows]
    times=np.array([(d-dates[-1]).total_seconds()/86400 for d in dates])
    curve=LocalPaceTrend(np.log([r['pace_min_mile'] for r in rows]),m,g,covariance,times,bandwidth_days=SMOOTHING_DAYS)
    history=[dict(date=d.astimezone(zone).date().isoformat(),pace_min_mile=curve.level(t)['pace_min_mile'])
        for d,t in zip(dates,times) if t>=-84]
    primary=groups['gap7_fraction0.500']
    return dict(fits=fits,prior_miles=sum(r['miles'] for r in training if r['time']>=era_start),examples=len(primary),
        outcome_runs=len({i for e in primary for i in e['outcome_ids']}),
        sensitivity=[dict(policy=p,response_examples=len(groups[p]),origins=fits[p]['calendar']['origins']) for p in POLICIES],
        training_runs=len(training),latest_training=max((r['available_at'] for r in training)).isoformat(),
        latest_activity_id=rows[-1]['activity_id'],history=history,
        current_pace=math.exp(levels[SMOOTHING_DAYS][rows[-1]['activity_id']]),error_calibration=calibration)


def pace_outlook(connection,report,performance,goal,*,as_of,timezone):
    unavailable=lambda reason:dict(available=False,reason=reason,version=VERSION)
    if goal.get('status') not in {'supporting_volume','progressing'} or len(goal.get('projection',{}).get('points',()))<2:
        return unavailable('Refresh your Weekly Plan for an updated pace outlook.')
    if not performance.get('available') or not performance.get('current',{}).get('available'):
        return unavailable('Comparable runs will establish your pace outlook.')
    if performance.get('days_since_observation',0)>28:
        return unavailable('Add recent runs to update your pace outlook.')
    zone=ZoneInfo(timezone)
    # This cache covers only the historical fit. Fresh uploads/overrides/HR
    # settings invalidate it; date-dependent goal projection stays outside.
    selected={**report,'runs':[r for r in report.get('runs',()) if datetime.fromisoformat(r['start_time'])<=as_of]}
    key=(VERSION,timezone,json.dumps(selected,sort_keys=True),tuple(performance.get('hidden_activity_ids',())),performance['last_observation'])
    fitted=invariant_read(connection,key,lambda:_fit_history(connection,selected,performance,as_of,zone),cacheable=lambda r:r is not None)
    if not fitted:
        return unavailable('More completed training and comparable runs will establish your pace outlook.')
    start=as_of.astimezone(zone).date()
    projection=goal['projection'];end=date.fromisoformat(projection['target_date'] or projection['points'][-1]['date'])
    try:
        points=projected_curve(fitted['fits'],start=start,end=end,
            current_pace=fitted['current_pace'],prior_miles=fitted['prior_miles'],
            volume=goal['volume_per_7'],growth=goal['volume_growth_per_7'],target_volume=goal['target_volume_per_7'],
            error_calibration=fitted['error_calibration'])
    except (ValueError,OverflowError):
        return unavailable('The current progression does not support a pace outlook yet.')
    return dict(available=True,version=VERSION,as_of=as_of.isoformat(),reference_hr_bpm=performance['target_hr_bpm'],
        current_pace_min_mile=fitted['current_pace'],date=end.isoformat(),
        expected_pace_min_mile=points[-1]['pace_min_mile'],low_pace_min_mile=points[-1]['low_pace_min_mile'],
        high_pace_min_mile=points[-1]['high_pace_min_mile'],points=points,history=fitted['history'],
        date_basis='target' if projection['target_date'] else 'taper',
        latest_training=fitted['latest_training'],latest_activity_id=fitted['latest_activity_id'],
        training_runs=fitted['training_runs'],response_examples=fitted['examples'],outcome_runs=fitted['outcome_runs'],
        consistency_sensitivity=fitted['sensitivity'],smoothing_days=SMOOTHING_DAYS,
        error_growth_exponent=fitted['error_calibration']['exponent'],
        range_minimum_horizon_days=fitted['error_calibration']['minimum_horizon_days'],
        calibration_examples=len(fitted['error_calibration']['errors']),
        calibration_origins=fitted['error_calibration']['calibration_origins'],
        calibration_outcome_runs=fitted['error_calibration']['calibration_outcome_runs'],
        error_growth_matched_origins=fitted['error_calibration']['matched_origins'],
        latest_calibration_outcome=fitted['error_calibration']['latest_calibration_outcome'],
        center_kind='Midpoint of fitted training-response scenarios, not a separately fitted expected value',
        method='42-day aerobic smoothing; consistent-training response and historical actual-future-run forecast errors',
        range_kind='Empirical forecast-error working range, not a calibrated prediction interval',
        assumptions='Continue consistent training along the projected volume path; extrapolate the historical sustained or diminishing response',
        limitations=['Short-term retrospective validation with current pace corrections held fixed',
            'Training association is not causal proof; temporary variation cannot be identified reliably',
            '7/10-day gaps and 50%/67% usual-training floors are diagnostic sensitivity definitions, not physiological cutoffs',
            'Sustained versus diminishing response forms remain long-term extrapolation assumptions',
            'Range uses the 80th percentile of extrapolated absolute out-of-sample log errors, not certified 80% coverage',
            'Error growth learned from matched two/four-week 28-day control forecasts in mixed history; transferred to 42-day consistent-training errors',
            'No range before the earliest tested forecast horizon; present estimate uncertainty is not calibrated by future-run errors',
            'Unmarked health and absent recordings do not prove healthy uninterrupted training',
            'Long-term response and future adherence are not validated',
            'Range does not include every source of future performance variation',
            'Aerobic benchmark at the reference HR, not a prescribed pace or pace sustainable for the target long-run distance'])
