"""Explore prior-weather anomaly beyond absolute heat, without live changes.

Primary: same-location, same-clock-time temperature minus the preceding seven
days' temperature. Three/fourteen days are fixed sensitivity checks. Archived
full calendar weather is used, never just weather on days the athlete ran.
"""
from argparse import ArgumentParser
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import sqlite3
from time import perf_counter
from zoneinfo import ZoneInfo

import numpy as np

from run_analysis.config import load_config, resolve_project_path
from run_analysis.geo import haversine_m
from run_analysis.privacy import private_file
from run_analysis.read_cache import invariant_snapshot
from scripts.audit_additional_pace_signals import BASE, collect, fit, design, contrast, finite, day


MINIMAL=tuple(dict.fromkeys(BASE+('temperature_f','temperature_squared','heat_squared','hour_sin','hour_cos')))
CONTROLLED=MINIMAL+('previous_7d_miles','previous_28d_miles','log_days_since_run','log_days_since_hard','stop_fraction')
ANOMALY='temperature_anomaly_c'


def archives(connection,root):
    out=[]
    for row in connection.execute('SELECT cache_file,min(latitude_key) AS lat,min(longitude_key) AS lon FROM weather_cache GROUP BY cache_file'):
        path=resolve_project_path(root,row['cache_file'])
        if not path.exists():continue
        data=json.loads(path.read_text());hourly=data.get('hourly',{})
        times=hourly.get('time',[]);temperatures=hourly.get('temperature_2m',[])
        if not times or len(times)!=len(temperatures) or not isinstance(times[0],(int,float)):continue
        if data.get('hourly_units',{}).get('temperature_2m') not in ('°F','F','fahrenheit'):
            raise ValueError('Known Fahrenheit archive units required')
        out.append(dict(lat=row['lat'],lon=row['lon'],times=np.asarray(times,float),
            temperature=np.asarray([float(v) if finite(v) else np.nan for v in temperatures]),
            source=path.name))
    return out


def interpolate(archive,moment):
    times=archive['times'];at=moment.timestamp()
    if at<times[0] or at>times[-1]:return None
    index=int(np.searchsorted(times,at))
    if index<len(times) and times[index]==at:
        value=float(archive['temperature'][index])
    else:
        if index==0 or index==len(times) or times[index]-times[index-1]>3600.1:return None
        left,right=archive['temperature'][index-1:index+1]
        value=float(left+(right-left)*(at-times[index-1])/(times[index]-times[index-1]))
    return value if math.isfinite(value) else None


def prior_anomaly(archive,moment,days,zone):
    local=moment.astimezone(ZoneInfo(zone))
    # Calendar-local same-hour arithmetic retains the intended clock time
    # across DST. No future date, current-day mean or next-day temperature.
    references=[(local-timedelta(days=i)).astimezone(timezone.utc) for i in range(1,days+1)]
    now=interpolate(archive,moment);prior=[interpolate(archive,v) for v in references]
    if now is None or any(value is None for value in prior):return None
    return dict(current_f=now,prior_mean_f=float(np.mean(prior)),
        anomaly_c=(now-float(np.mean(prior)))*5/9,days=days,
        first_prior=references[-1].isoformat(),last_prior=references[0].isoformat())


def weather_features(rows,cache,locations,zones,*,radius_km=10):
    result=[];missing=[]
    for row in rows:
        identity=row['activity_id'];location=locations.get(identity)
        if not location:
            missing.append(dict(activity_id=identity,reason='no_cached_weather_location'));continue
        when=datetime.fromisoformat(row['start_time_utc'])+timedelta(minutes=row['sampled_minutes'])
        candidates=sorted([(haversine_m(location[0],location[1],a['lat'],a['lon'])/1000,a) for a in cache],key=lambda pair:(pair[0],pair[1]['source']))
        features={}
        for window in (3,7,14):
            for distance,archive in candidates:
                if distance>radius_km:break
                value=prior_anomaly(archive,when,window,zones.get(identity) or 'America/New_York')
                if value is not None:
                    features[str(window)]=dict(value,distance_km=distance,source=archive['source']);break
        if '7' not in features:
            missing.append(dict(activity_id=identity,reason='no_complete_preceding_week_nearby'))
        result.append(dict(row,relative_weather=features,
            temperature_squared=row['temperature_f']**2 if finite(row.get('temperature_f')) else None,
            heat_squared=row['heat_c']**2 if finite(row.get('heat_c')) else None))
    return result,missing


def recent_run_temperature(row,history,location,days):
    """Actual during-run weather, regardless of the clock time it occurred."""
    when=datetime.fromisoformat(row['start_time_utc'])
    previous=[p for p in history if 0<(when-datetime.fromisoformat(p['start_time_utc'])).total_seconds()/86400<=days
        and finite(p.get('temperature_f')) and p.get('location') is not None and location is not None
        and haversine_m(*location,*p['location'])<=10000]
    if not previous or not finite(row.get('temperature_f')):return None
    mean=float(np.mean([p['temperature_f'] for p in previous]))
    return dict(current_f=row['temperature_f'],prior_mean_f=mean,anomaly_c=(row['temperature_f']-mean)*5/9,
        preceding_runs=len(previous),prior_ids=[p['activity_id'] for p in previous],days=days,
        basis='Actual recorded run temperatures; no same-clock requirement or daily maximum')


def paired_range(cases,*,draws=4000):
    if not cases:return None
    labels=np.array([math.floor(datetime.fromisoformat(c['date']).timestamp()/86400/28) for c in cases])
    blocks=[np.flatnonzero(labels==label) for label in np.unique(labels)]
    delta=np.array([c['candidate_absolute_error']-c['baseline_absolute_error'] for c in cases])
    rng=np.random.default_rng(20261015)
    samples=[float(delta[np.concatenate([blocks[i] for i in rng.integers(len(blocks),size=len(blocks))])].mean()) for _ in range(draws)]
    return dict(calendar_blocks=len(blocks),mean_error_difference_seconds=float(delta.mean()),
        descriptive_95_block_range=np.quantile(samples,[.025,.975]).tolist())


def forward(rows,keys,*,outcome='pace'):
    cases=[]
    for index in range(24,len(rows)):
        train=rows[:index];anchor=rows[index-1];target=rows[index]
        if day(target)-day(anchor)>14 or day(train[-1])-day(train[0])<56:continue
        values={}
        for name,features in (('baseline',keys),('candidate',keys+(ANOMALY,))):
            fitted=fit(train,features,outcome=outcome,draws=2)
            x,_=design([anchor,target],features,fitted['spec'])
            prediction=anchor[outcome]*math.exp(float((x[1]-x[0])@fitted['beta']))
            values[name]=dict(prediction=prediction,error=60*(prediction-target[outcome]))
        cases.append(dict(activity_id=target['activity_id'],date=target['start_time_utc'],anchor_id=anchor['activity_id'],
            prior_runs=index,baseline_absolute_error=abs(values['baseline']['error']),
            candidate_absolute_error=abs(values['candidate']['error']),**values))
    def summarize(items):
        return dict(cases=len(items),
            baseline_mae_seconds=float(np.mean([c['baseline_absolute_error'] for c in items])) if items else None,
            candidate_mae_seconds=float(np.mean([c['candidate_absolute_error'] for c in items])) if items else None,
            baseline_rmse_seconds=float(np.sqrt(np.mean([c['baseline']['error']**2 for c in items]))) if items else None,
            candidate_rmse_seconds=float(np.sqrt(np.mean([c['candidate']['error']**2 for c in items]))) if items else None)
    half=len(cases)//2
    return dict(summary=summarize(cases),first_half=summarize(cases[:half]),second_half=summarize(cases[half:]),
        paired_blocks=paired_range(cases),cases=cases,
        scope='Earlier-run coefficients, previous-run anchored conditional prediction; current normalization and held-out context are fixed, not a full pre-run forecast')


def analyze(rows,window=7,*,omit=None,keys=CONTROLLED,outcome='pace'):
    cohort=[dict(r,**{ANOMALY:r['relative_weather'][str(window)]['anomaly_c']}) for r in rows
        if str(window) in r['relative_weather'] and r['activity_id']!=omit]
    if len(cohort)<24:return dict(runs=len(cohort),available=False)
    parameters=fit(cohort,keys+(ANOMALY,),outcome=outcome)
    # At fixed actual run weather: compare 10 F cooler than recent days with
    # equal-to-recent weather. This is not an absolute-temperature correction.
    effect=contrast(cohort,parameters,keys+(ANOMALY,),{ANOMALY:(0.,-10*5/9)})
    return dict(available=True,runs=len(cohort),window_days=window,omit_activity_id=omit,outcome=outcome,
        calendar_blocks=parameters['calendar_blocks'],cooler_by_10f_at_same_absolute_weather=effect,
        forward=forward(cohort,keys,outcome=outcome))


def audit(connection,config,root):
    rows,coverage=collect(connection,config)
    cache=archives(connection,root)
    locations={r['activity_id']:(r['latitude_key'],r['longitude_key']) for r in connection.execute(
        'SELECT a.activity_id,w.latitude_key,w.longitude_key FROM activity_weather a JOIN weather_cache w ON w.id=a.weather_cache_id')}
    zones={r['id']:r['timezone_name'] for r in connection.execute('SELECT id,timezone_name FROM activities')}
    all_rows,missing=weather_features(rows,cache,locations,zones)
    recent=[r for r in all_rows if r['start_time_utc']>='2026-03-01']
    print(f'Weather coverage: {sum("7" in r["relative_weather"] for r in recent)}/{len(recent)} recent runs',flush=True)
    # Primary cohort and controls are fixed; no horizon chosen from results.
    variants={str(w):analyze(recent,w) for w in (3,7,14)}
    variants['7_without_july7']=analyze(recent,7,omit=34)
    variants['7_minimal_controls']=analyze(recent,7,keys=MINIMAL)
    variants['7_raw_sections']=analyze(recent,7,outcome='observed')
    variants['7_all_years']=analyze(all_rows,7)
    # User clarification: also compare with the actual weather of preceding
    # runs, even if those were at different times. Never substitute a day's
    # high or daily mean for what the athlete actually ran in.
    history=[dict(r,location=locations.get(r['activity_id'])) for r in connection.execute(
        '''SELECT a.id AS activity_id,a.start_time_utc,w.temperature_f FROM activities a
        JOIN activity_weather w ON w.activity_id=a.id WHERE lower(a.sport) IN ('running','run')''')]
    training_rows=[]
    for row in recent:
        features={str(days):value for days in (3,7,14)
            if (value:=recent_run_temperature(row,history,locations.get(row['activity_id']),days)) is not None}
        training_rows.append(dict(row,relative_weather=features))
    run_variants={str(w):analyze(training_rows,w) for w in (3,7,14)}
    run_variants['7_without_july7']=analyze(training_rows,7,omit=34)
    run_variants['7_minimal_controls']=analyze(training_rows,7,keys=MINIMAL)
    run_variants['7_raw_sections']=analyze(training_rows,7,outcome='observed')
    july=next((r for r in recent if r['activity_id']==34),None)
    return dict(version='relative-weather-audit-v1',live_changed=False,source_database_read_only=True,
        coverage=coverage,cached_archives=len(cache),weather_missing=missing,variants=variants,
        july7=july,observations=all_rows,recent_run_temperature_variants=run_variants,
        recent_run_temperature_july7=next((r['relative_weather'] for r in training_rows if r['activity_id']==34),None),
        assumptions=[
            'Primary prior seven full calendar days at the run observation clock time; three/fourteen-day sensitivities fixed before seeing outcomes.',
            'User clarification adds a separate actual-training-temperature baseline: equally weighted prior running activities in 3/7/14 days within 10 km, regardless of clock time.',
            'Daily prior temperatures include non-running days, use local-time/DST arithmetic and never include future weather.',
            'Each anomaly uses one complete archived grid series within 10 km, chosen by distance not performance; no sparse-day means or mixed-grid jumps.',
            'Current temperature and previous-day values are from the same grid; actual run temperature, squared temperature, humidity-sensitive heat and squared heat remain controls.',
            'Primary controls also include recorded HR, within-run position, duration, wind, device, coarse route area, time of day, recent volume and recovery spacing.',
            '56-day calendar intercepts/slopes and fixed ridge 4 reuse the earlier signal audit; no tuning to a favorable effect or horizon.',
            'Every forward target after 24 priors is evaluated once against its previous eligible run, provided the gap is at most 14 days and training spans at least 56 days.',
            'Forward coefficients use earlier observations only; pace normalization itself is the existing full-history representation, not a complete as-known-then replay.',
            'Block ranges and multiple sensitivity checks are exploratory single-athlete associations, not causal acclimatization or validated physiological intervals.',
            'No weather download, coordinate transmission, HR/pace correction update, graph change or scheduling change.'
        ])


def main():
    parser=ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():parser.error('New output path required')
    root=Path.cwd();config=load_config(root/'config.yaml');database=resolve_project_path(root,config['paths']['database']);started=perf_counter()
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True) as connection:
        connection.row_factory=sqlite3.Row
        with invariant_snapshot(connection):
            result=audit(connection,config,root);assert connection.total_changes==0
    result['elapsed_seconds']=perf_counter()-started
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');private_file(args.output)
    print(json.dumps({name:{k:v for k,v in value.items() if k!='forward'}|{'forward_summary':value.get('forward',{}).get('summary')}
        for name,value in result['variants'].items()},indent=2))


if __name__=='__main__':main()
