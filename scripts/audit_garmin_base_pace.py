"""Read-only sustained observed pace/HR audit; no planner or live model changes."""
from argparse import ArgumentParser
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import sqlite3

import numpy as np

from run_analysis.config import load_config, resolve_project_path
from run_analysis.model_windows import load_model_window_sets
from run_analysis.pace_comparability import fit_comparison_calibration, compare_run_windows
from run_analysis.read_cache import invariant_snapshot
from run_analysis.wellness_signals import read_wellness


def steady(w, start=10., maximum_minute=60.):
    seconds=w['elapsed_time_s']-w['stopped_time_s']
    return (w['moving_minutes_into_run']-seconds/120>=start-1e-8
        and w['moving_minutes_into_run']+seconds/120<=maximum_minute+1e-8
        and w.get('heart_rate_change_bpm') is not None and abs(w['heart_rate_change_bpm'])<=5
        and w.get('speed_change_fraction') is not None and abs(w['speed_change_fraction'])<=.1)


def summarize(rows, meta):
    grouped=defaultdict(list)
    for w in rows:grouped[w['activity_id']].append(w)
    output=[]
    for key, windows in grouped.items():
        seconds=sum(w['elapsed_time_s']-w['stopped_time_s'] for w in windows)
        distance=sum(w['distance_m'] for w in windows)
        def avg(field):
            pairs=[(w[field],w['elapsed_time_s']-w['stopped_time_s']) for w in windows if w.get(field) is not None]
            return sum(v*s for v,s in pairs)/sum(s for _,s in pairs) if pairs else None
        output.append(dict(activity_id=key,date=meta[key]['start_time_local'],device=meta[key]['creator'],
            role=windows[0]['workout_type'],sections=len(windows),minutes=seconds/60,
            pace=seconds/60/(distance/1609.344),hr=avg('average_hr_bpm'),
            temperature_f=avg('temperature_f'),dewpoint_f=avg('dewpoint_f'),
            grade_percent=avg('average_grade_percent'),grade_ratio=avg('grade_energy_ratio'),
            minutes_into_run=[min(w['moving_minutes_into_run'] for w in windows),max(w['moving_minutes_into_run'] for w in windows)]))
    paces=[r['pace'] for r in output]
    return dict(runs=len(output),sections=len(rows),minutes=sum(r['minutes'] for r in output),
        median_pace=float(np.median(paces)) if paces else None,
        descriptive_run_pace_10_90=np.quantile(paces,[.1,.9]).tolist() if paces else None,
        per_run=sorted(output,key=lambda r:r['date']))


def audit(output):
    cfg=load_config();cfg['target_hr']=148
    path=resolve_project_path(Path.cwd(),cfg['paths']['database'])
    now=datetime.now(timezone.utc)
    broad=deepcopy(cfg)
    broad['model']['minimum_reliable_segment_minutes']=0
    broad['model']['maximum_reliable_segment_minutes']=1440
    broad['model']['maximum_reliable_hr_bpm']=cfg['max_hr']
    with sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True) as db:
        db.row_factory=sqlite3.Row
        with invariant_snapshot(db):
            meta={r['id']:dict(r) for r in db.execute('SELECT id,start_time_utc,start_time_local,creator FROM activities')}
            sets,counters=load_model_window_sets(db,broad,(120,300,600),check_elevation=True)
            recent=lambda w,days:datetime.fromisoformat(w['start_time_utc'])>=now-timedelta(days=days)
            # Many imported ordinary runs have no purpose label. Their actual
            # steady HR/pace still count; unknown is NOT grounds to drop them.
            routine=lambda w:w['workout_type'] not in {'intervals','tempo_threshold','hill_repeats','race'}
            cohorts={}
            for days in (28,56,90):
                for size in (300,600):
                    for tolerance in (1,2,3):
                        chosen=[w for w in sets[size] if recent(w,days) and steady(w)
                            and abs(w['average_hr_bpm']-148)<=tolerance and routine(w)]
                        cohorts[f'{days}d_{size}s_hr148pm{tolerance}']=summarize(chosen,meta)
            for days in (28,56):
                cohorts[f'{days}d_300s_hr148pm2_all_later_minutes']=summarize([
                    w for w in sets[300] if recent(w,days) and steady(w,maximum_minute=1440.)
                    and abs(w['average_hr_bpm']-148)<=2 and routine(w)],meta)
            s=sets[300]
            near_pace=lambda w:abs(w['moving_pace_min_mile']-(9+5/60))<=.25
            comparisons={}
            for name, predicate in (
                ('recent56_steady_at_905',lambda w:recent(w,56) and steady(w) and near_pace(w)),
                ('all_2026_steady_at_905',lambda w:w['start_time_utc']>='2026-01-01' and steady(w) and near_pace(w)),
                ('all_2026_at_148',lambda w:w['start_time_utc']>='2026-01-01' and steady(w) and abs(w['average_hr_bpm']-148)<=2),
                ('recent56_early_at_905',lambda w:recent(w,56) and w['moving_minutes_into_run']<10 and near_pace(w)),
                ('recent56_fast_at_148_including_transitions',lambda w:recent(w,56)
                    and abs(w['average_hr_bpm']-148)<=2 and w['moving_pace_min_mile']<=9+5/60+.25)):
                comparisons[name]=summarize([w for w in s if predicate(w)],meta)
            comparisons['recent56_two_minute_fast_at_148']=summarize([w for w in sets[120]
                if recent(w,56) and abs(w['average_hr_bpm']-148)<=2
                and w['moving_pace_min_mile']<=9+5/60+.25],meta)
            comparisons['recent56_two_minute_fast_at_148_details']=[{
                key:w.get(key) for key in ('activity_id','start_time_utc','window_start_time_utc',
                    'moving_pace_min_mile','average_hr_bpm','heart_rate_change_bpm','speed_change_fraction',
                    'moving_minutes_into_run','average_grade_percent','grade_energy_ratio')}
                for w in sets[120] if recent(w,56) and abs(w['average_hr_bpm']-148)<=2
                and w['moving_pace_min_mile']<=9+5/60+.25]
            regular=[w for w in s if steady(w,5.) and w['average_hr_bpm']<=cfg['model']['maximum_reliable_hr_bpm']]
            calibration=fit_comparison_calibration(regular,cfg)
            normalized=[]
            for row in cohorts['56d_300s_hr148pm2']['per_run']:
                windows=[w for w in s if w['activity_id']==row['activity_id'] and steady(w)
                    and abs(w['average_hr_bpm']-148)<=2 and routine(w)]
                result=compare_run_windows(windows,[1.]*len(windows),cfg,calibration)
                normalized.append(dict(activity_id=row['activity_id'],date=row['date'],raw_pace=row['pace'],
                    adjusted=result['pace_min_mile'],contributions=result['contributions_seconds_per_mile']))
            last_runs=[dict(r) for r in db.execute('''SELECT a.id,a.start_time_local,a.creator,
                m.moving_pace_min_mile,m.moving_average_hr_bpm,m.detected_workout_type,m.model_eligible
                FROM activities a JOIN activity_metrics m ON m.activity_id=a.id
                WHERE a.sport='Running' ORDER BY a.start_time_utc DESC LIMIT 15''')]
    wellness=read_wellness(Path.cwd()) or {}
    result=dict(version='garmin-base-direct-observation-v1',as_of=now.isoformat(),
        target_hr=148,garmin_pace=9+5/60,cohorts=cohorts,comparisons=comparisons,
        secondary_adjustments=normalized,calibration=calibration,last_runs=last_runs,
        reported_vo2=[r for r in wellness.get('series',[]) if r.get('reported_vo2max_ml_kg_min')][-15:],
        diagnostics=counters,live_changed=False,planner_called=False,
        limitations=['Unlabelled ordinary runs remain included; only explicitly labelled quality/race runs are excluded from the primary cohort.',
            'Per-run summaries avoid treating sections as independent runs.',
            'Bands/ranges are descriptive observed variation, not confidence intervals.',
            'Primary raw sections are 10–60 moving minutes, five/ten-minute nonoverlapping blocks, HR half change <=5 bpm and speed half change <=10%.',
            'Existing model eligibility, distance/HR coverage and post-pause suppression remain; early/HR-expanded sections are diagnostic only.',
            'Secondary adjusted values use the existing model, not an independent validation or Garmin reverse engineering.'])
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as handle:json.dump(result,handle,indent=2,allow_nan=False)
    print(json.dumps(dict(output=str(output),cohorts={k:{f:v[f] for f in ('runs','sections','median_pace','descriptive_run_pace_10_90')} for k,v in cohorts.items()},
        comparisons={k:{f:v[f] for f in ('runs','sections','median_pace')} for k,v in comparisons.items() if isinstance(v,dict)},
        secondary_adjustments=normalized),indent=2))


if __name__=='__main__':
    p=ArgumentParser(description=__doc__);p.add_argument('output',type=Path)
    audit(p.parse_args().output)
