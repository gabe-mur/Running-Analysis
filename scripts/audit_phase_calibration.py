"""Full-archive chronological phase calibration; no live writes or future fitting.

Historical phase timing/distance is supplied retrospectively for diagnostics.
The four existing sandbox prescriptions remain separate pre-run forecasts.
"""
from argparse import ArgumentParser
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3
from statistics import median
from time import perf_counter

from scripts.audit_athlete_projection import assumptions, json_ready
from scripts.audit_workload_contract import load_sessions
from scripts.audit_workload_expectation import app_session_types
from run_analysis.config import load_config
from run_analysis.projection_evidence import evidence_from_activity
from run_analysis.athlete_projection import snapshot
from run_analysis.phase_calibration import (
    observed_phases, learn_phase_residuals, correction, calibrated_forecast,
)


def summarize(rows):
    result = {}
    for metric in ('pace','load'):
        paired = [r for r in rows if r[f'{metric}_baseline_error'] is not None]
        # A session, not a rep or phase, is the independent aggregation unit.
        grouped = defaultdict(list)
        for r in paired:
            grouped[r['key']].append(r)
        baseline = [sum(abs(r[f'{metric}_baseline_error']) for r in g)/len(g) for g in grouped.values()]
        candidate = [sum(abs(r[f'{metric}_candidate_error']) for r in g)/len(g) for g in grouped.values()]
        result[metric] = dict(session_count=len(grouped), phase_count=len(paired),
            baseline_median_session_mean_absolute_fraction=median(baseline) if baseline else None,
            candidate_median_session_mean_absolute_fraction=median(candidate) if candidate else None,
            improved_sessions=sum(b>a for a,b in zip(candidate,baseline)),
            worsened_sessions=sum(a>b for a,b in zip(candidate,baseline)))
    return result


def audit(database, config, *, pooled_history=False):
    started = perf_counter()
    lookback=None if pooled_history else 84.
    roles = app_session_types(database)
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True) as db:
        eligible={str(k):not ill and health in (None,'','normal') and inc!=0
            for k,ill,health,inc in db.execute('''SELECT a.id,o.illness,o.health_tag,o.include_in_model
                FROM activities a LEFT JOIN run_overrides o ON o.activity_id=a.activity_id''')}
    records=[]
    inventory=defaultdict(Counter)
    summary_only=[]
    def missing_timing(activity,key,path):
        group=inventory[str(activity.start_time_utc.year)]
        group['runs']+=1
        group['with_recorded_hr']+=int(any(p.heart_rate_bpm is not None for p in activity.trackpoints))
        group['summary_only_missing_elapsed']+=1
        summary_only.append(dict(key=key,start=activity.start_time_utc,
            lap_timer_seconds=activity.lap_recorded_time_s,meters=activity.total_distance_m,
            reason='Missing elapsed-time trace; retained in inventory, not fabricated as rest or phase evidence'))
    def collect(activity,entry,profile):
        available=max(entry.end,activity.start_time_utc+timedelta(seconds=activity.lap_recorded_time_s or 0),
            *(p.timestamp_utc for p in activity.trackpoints if p.timestamp_utc is not None))
        e=evidence_from_activity(activity,config,key=entry.key,available_at=available,
            workout_type=roles[entry.key],profile=profile,capacity_eligible=eligible[entry.key])
        phases=observed_phases(activity,e,config,profile)
        group=inventory[str(activity.start_time_utc.year)]
        group['runs']+=1
        group['with_recorded_hr']+=int(any(p.heart_rate_bpm is not None for p in activity.trackpoints))
        group['with_phase_layout']+=bool(phases)
        group['capacity_eligible']+=e.capacity_eligible
        records.append((e,phases))
    load_sessions(database,config,datetime.min.replace(tzinfo=timezone.utc),
        datetime(2026,9,25,tzinfo=timezone.utc),session_types=roles,activity_observer=collect,
        missing_timing_observer=missing_timing)
    history=[]
    residuals=[]
    diagnostics=[]
    cases=[]
    for actual,observations in records:
        state=snapshot(history,actual.start-timedelta(seconds=1),actual.accounting_key)
        day=str(actual.start.date())
        if day in {'2026-09-18','2026-09-20','2026-09-22','2026-09-24'}:
            phases,refs,problem=assumptions(state,day,config)
            case=dict(day=day,problem=problem,actual_load=actual.cardio,
                actual_elapsed_seconds=(actual.end-actual.start).total_seconds(),actual_moving_seconds=actual.moving_seconds)
            if phases:
                if actual.role in {'easy','long'}:
                    p=phases[0]
                    warmup=float(config.get('model',{}).get('minimum_reliable_segment_minutes',5))*60
                    phases=(replace(p,name='warmup',seconds=warmup,meters=None),
                            replace(p,name='easy',finish_total_distance=True))
                try:
                    case['forecast']=calibrated_forecast(state,phases,residuals,lookback_days=lookback)
                except ValueError as exc:
                    case['problem']=str(exc)
            cases.append(case)
        learned=learn_phase_residuals(state,actual,observations)
        if observations and actual.capacity_eligible:
            for o,sample in zip(observations,learned):
                c=correction(residuals,as_of=state.as_of,accounting_key=state.accounting_key,
                    phase=sample.phase,seconds=sample.seconds,lookback_days=lookback)
                diagnostics.append(dict(key=actual.key,day=day,role=actual.role,phase=sample.phase,
                    origin=sample.origin,hr_coverage=sample.hr_coverage,
                    pace_baseline_error=1/sample.pace_ratio-1 if sample.pace_ratio else None,
                    pace_candidate_error=c['pace_ratio']['factor']/sample.pace_ratio-1 if sample.pace_ratio else None,
                    load_baseline_error=1/sample.load_ratio-1 if sample.load_ratio else None,
                    load_candidate_error=c['load_ratio']['factor']/sample.load_ratio-1 if sample.load_ratio else None,
                    correction=c))
        # Append only after forecasting/scoring the held-out completion.
        residuals.extend(learned)
        history.append(actual)
    return dict(version='chronological_phase_calibration_candidate_v1',
        calibration_lookback_days=lookback,pooled_history_diagnostic_only=pooled_history,
        original_session_count=len(records)+len(summary_only),trace_session_count=len(records),
        summary_only=summary_only,inventory_by_year=dict(inventory),
        diagnostics=diagnostics,cases=cases,overall=summarize(diagnostics),
        by_phase={p:summarize([r for r in diagnostics if r['phase']==p]) for p in sorted({r['phase'] for r in diagnostics})},
        by_year={y:summarize([r for r in diagnostics if r['day'].startswith(y)]) for y in sorted(inventory)},
        elapsed_seconds=perf_counter()-started,planner_influence=False,
        limitations=['Fixed current HR references/roles/health metadata, not reconstructed historical settings',
            'Availability at completion is an explicit simulation assumption',
            'Historical phase diagnostics condition on observed boundaries/dose, not verified issued prescriptions',
            'Full archive supplies test origins; capacity and default calibration both retain the 84-day window',
            'No context adjustment or uncertainty calibration; intervals remain a small sample',
            'Forecast time assumes uninterrupted execution; actual moving and elapsed times are reported separately',
            'Four existing sandbox cases were already inspected: not untouched validation'])


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--database',required=True,type=Path)
    parser.add_argument('--config',type=Path,default=Path('config.yaml'))
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--pooled-history',action='store_true',help='Reproduce the rejected across-years correction diagnostic')
    args=parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output path')
    result=audit(args.database,load_config(args.config),pooled_history=args.pooled_history)
    with args.output.open('x') as f:
        json.dump(result,f,default=json_ready,indent=2,allow_nan=False)
        f.write('\n')
    print(json.dumps({k:result[k] for k in ('original_session_count','inventory_by_year','overall','by_phase','by_year','elapsed_seconds')},indent=2))
    for c in result['cases']:
        f=c.get('forecast')
        print(c['day'],c['problem'] or dict(base=f['baseline_cardio_load'],candidate=f['cardio_load'],
            base_seconds=f['baseline_seconds'],candidate_seconds=f['execution_seconds'],actual=c['actual_load']))


if __name__=='__main__':
    main()
