"""Disposable synthetic first-outing audit; never opens the athlete's database.

Tests software evidence availability, not physiological validity or an appropriate
weekly starting dose. Run with .venv/bin/python scripts/audit_beginner_calibration.py.
"""
from datetime import datetime, timedelta, timezone
import json
from math import exp
from math import log
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tests'))
from fit_builder import (FitWriter, FILE_ID, SESSION, RECORD, ENUM, UINT8,
                         UINT16, UINT32, SINT32, fit_timestamp, semicircles)
from test_web_phase1 import _write_config
from run_analysis.config import load_config
from run_analysis.db import connect
from run_analysis.importer import import_files
from run_analysis.processing import process_activities
from run_analysis.run_feedback import list_runs, get_run_feedback
from run_analysis.training_background import calibration_activity
from run_analysis.adaptive.adaptive_candidate_pipeline import load_athlete, compile_athlete


CASES = (
    dict(name='continuous_10', minutes=10, pace=12.5),
    dict(name='continuous_20', minutes=20, pace=11.),
    dict(name='continuous_30', minutes=30, pace=9.5),
    dict(name='run_walk_20', minutes=20, pace=12., run_seconds=60, walk_seconds=90),
    dict(name='mostly_run_30', minutes=30, pace=11., run_seconds=240, walk_seconds=60),
    dict(name='warmup_hr_10', minutes=10, pace=12.5, initial_hr=110),
    dict(name='hard_20', minutes=20, pace=11., target_hr=178),
    dict(name='missing_hr_20', minutes=20, pace=11., missing_hr=True),
)


def recorded_outing(path, start, case):
    writer=FitWriter()
    writer.message(FILE_ID,{0:(ENUM,4),1:(UINT16,1),2:(UINT16,3121),4:(UINT32,fit_timestamp(start))})
    distance=0.; hr=case.get('initial_hr',145.); observed=[]; running_seconds=0
    for seconds in range(0,round(case['minutes']*60)+1,10):
        cycle=case.get('run_seconds',case['minutes']*60+1)+case.get('walk_seconds',0)
        running=seconds%cycle<case.get('run_seconds',cycle)
        if case.get('blocks'):
            edge=0
            for active,length in case['blocks']:
                edge+=length
                running=active
                if seconds<edge:
                    break
        speed=1609.344/(case['pace']*60) if running else case.get('walk_speed',1.2)
        target=case.get('target_hr',145.) if running else 120.
        if seconds:
            distance+=speed*10
            running_seconds+=10*running
            hr=target+(hr-target)*exp(-10/45)
        observed.append(round(hr))
        values={253:(UINT32,fit_timestamp(start+timedelta(seconds=seconds))),
            0:(SINT32,semicircles(40.71+distance/111195)),1:(SINT32,semicircles(-73.99)),
            2:(UINT16,2650),4:(UINT8,80 if running else 50),
            5:(UINT32,round(distance*100)),6:(UINT16,round(speed*1000))}
        if not case.get('missing_hr'):values[3]=(UINT8,round(hr))
        writer.message(RECORD,values)
    fields={253:(UINT32,fit_timestamp(start)),2:(UINT32,fit_timestamp(start)),5:(ENUM,1),
        7:(UINT32,round(case['minutes']*60000)),8:(UINT32,round(case['minutes']*60000)),9:(UINT32,round(distance*100))}
    if not case.get('missing_hr'):fields.update({16:(UINT8,round(sum(observed)/len(observed))),17:(UINT8,max(observed))})
    writer.message(SESSION,fields)
    path.write_bytes(writer.build())
    return running_seconds/60


def audit_case(root,case):
    root.mkdir();_write_config(root)
    cfg=load_config(root/'config.yaml')
    cfg.setdefault('weather',{}).update(historical_enabled=False,forecast_enabled=False)
    start=datetime.now(timezone.utc)-timedelta(days=1)
    source=root/'calibration.fit'
    true_running=recorded_outing(source,start,case)
    database=root/cfg['paths']['database'];database.parent.mkdir()
    with connect(database) as db:
        import_files(db,root,cfg['timezone_default'],paths=[source])
        process_activities(db,cfg)
        listed=list_runs(db)[0]
        feedback=get_run_feedback(db,cfg,listed.activity_id)
        difficulty=feedback.run.session_difficulty
        metrics=dict(db.execute('SELECT model_eligible,exclusion_reason FROM activity_metrics').fetchone())
        calibrated=calibration_activity(db,cfg,reported_at=start-timedelta(seconds=1))
        result=dict(case=case['name'],outgoing_minutes=case['minutes'],
            synthetic_running_minutes=true_running,role=listed.workout_type.value,
            distance_miles=round(listed.distance_miles,3),**metrics,
            easy_hr_minutes=round(difficulty.zone_breakdown.easy_minutes,2) if difficulty else None,
            existing_easy_calibration_accepted=calibrated is not None)
    try:
        at=datetime.now(timezone.utc)
        inputs=load_athlete(database,cfg,at)
        result['physical_roles']=[r.role for r in inputs['history']]
        session=inputs['sessions'][0]
        result['whole_outing_pace_minutes_per_mile']=round(session.moving_seconds/60/(session.meters/1609.344),3)
        result['known_running_pace_minutes_per_mile']=case['pace']
        result['known_longest_running_bout_minutes']=min(case['minutes'],case.get('run_seconds',case['minutes']*60)/60)
        result['recorded_load_bounds']=[round(inputs['history'][0].demand[1].low,3),
                                       round(inputs['history'][0].demand[1].high,3)]
        try:
            candidate=compile_athlete(inputs,at)
            result['current_full_plan_seed']=candidate['stage'].weekly_equivalent_miles
        except ValueError as error:result['current_full_plan_seed']=str(error)
    except ValueError as error:result['physical_evidence_error']=str(error)
    return result


if __name__=='__main__':
    with TemporaryDirectory(prefix='beginner-calibration-') as folder:
        results=[audit_case(Path(folder)/str(i),case) for i,case in enumerate(CASES)]
    print(json.dumps(dict(scope='Synthetic evidence-path audit, not physiological or progression validation',
        cases=results,
        progression_sanity_check=dict(
            assumption='If a fixed-frequency running-time dose were increased only at the general 4% weekly volume rate; NOT an actual produced beginner schedule',
            minutes_after_9_weeks={str(n):round(n*1.04**9,2) for n in (10,20,30)},
            weeks_10_to_30=round(log(3)/log(1.04),2))),indent=2))
