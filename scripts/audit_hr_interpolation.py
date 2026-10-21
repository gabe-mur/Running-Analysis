"""Read-only, target-HR interpolation diagnostic; no live refit or planner.

Earlier runs supply slopes. Later runs supply independent recorded target-HR
windows and neighboring, non-overlapping anchor windows at another HR. Targets
are never used to calibrate a slope. This is a local interpolation check, not
a laboratory fitness test or permission to deploy a correction.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sqlite3

import numpy as np

from run_analysis.config import load_config
from run_analysis.model_windows import load_model_window_sets
from run_analysis.objective_modeling import (
    _shared_parameter_eligible, _within_run_hr_time_slopes,
    pace_to_speed_mps, speed_to_pace_min_mile,
)


def hr_only_slope(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['activity_id']].append(row)
    x, y = [], []
    for run in grouped.values():
        if len(run) < 2:
            continue
        hr=np.array([r['average_hr_bpm'] for r in run],dtype=float)
        speed=np.array([pace_to_speed_mps(r['moving_pace_min_mile']) for r in run])
        x.extend(hr-hr.mean()); y.extend(speed-speed.mean())
    x, y = np.array(x), np.array(y)
    if len(x)<20 or x@x<=1e-12:
        raise ValueError('Insufficient independent HR calibration data')
    slope=float(x@y/(x@x))
    for _ in range(8):
        residual=y-slope*x
        scale=1.4826*np.median(np.abs(residual-np.median(residual)))
        weights=np.minimum(1,1.345*max(scale,1e-9)/np.maximum(np.abs(residual),1e-12))
        slope=float((weights*x)@y/((weights*x)@x))
    return slope


def audit(primary, calibration, config):
    grouped=defaultdict(list)
    for row in primary:
        if _shared_parameter_eligible(row): grouped[row['activity_id']].append(row)
    runs=sorted(grouped,key=lambda i:(grouped[i][0]['start_time_utc'],i))
    target=float(config['target_hr'])
    low,high=map(float,config['zones']['z2'])
    reference=float(config['reference_conditions']['within_run_minutes'])
    errors=defaultdict(lambda:defaultdict(list))
    by_side=defaultdict(lambda:defaultdict(list))
    folds=[]; pairs=0; counts=defaultdict(int)
    def stable(row):
        return (low<=row['average_hr_bpm']<=high
                and abs(row['moving_minutes_into_run']-reference)<=10
                and row.get('heart_rate_change_bpm') is not None
                and row.get('speed_change_fraction') is not None
                and abs(row.get('heart_rate_change_bpm') or 0)<=3
                and abs(row.get('speed_change_fraction') or 0)<=0.10
                and row.get('grade_energy_ratio') is not None
                and abs(row.get('average_grade_percent') or 0)<=1)
    for left,right in ((.5,.65),(.65,.8),(.8,1)):
        start,end=int(len(runs)*left),int(len(runs)*right)
        training_ids=set(runs[:start])
        training=[r for r in calibration if r['activity_id'] in training_ids and _shared_parameter_eligible(r)]
        try:
            joint,_,_=_within_run_hr_time_slopes(training,[pace_to_speed_mps(r['moving_pace_min_mile']) for r in training])
            simple=hr_only_slope(training)
        except ValueError:
            continue
        models={'no_correction':0.0,'hr_only':simple,'hr_controlling_time':joint}
        folds.append({'training_runs':start,'test_runs':end-start,'hr_only_slope':simple,'time_controlled_hr_slope':joint})
        for activity_id in runs[start:end]:
            selected=[r for r in grouped[activity_id] if stable(r)]
            targets=[r for r in selected if abs(r['average_hr_bpm']-target)<=1]
            anchors=[r for r in selected if abs(r['average_hr_bpm']-target)>1]
            for truth in targets:
                candidates=[r for r in anchors
                            if 2<=abs(r['average_hr_bpm']-truth['average_hr_bpm'])<=7
                            and abs(r['moving_minutes_into_run']-truth['moving_minutes_into_run'])<=5
                            and abs(r['average_grade_percent']-truth['average_grade_percent'])<=.5]
                if not candidates: continue
                # Choose by temporal proximity, never by which pace fits best.
                anchor=min(candidates,key=lambda r:(abs(r['moving_minutes_into_run']-truth['moving_minutes_into_run']),r['moving_minutes_into_run']))
                delta=truth['average_hr_bpm']-anchor['average_hr_bpm']
                side='anchor_below_target' if delta>0 else 'anchor_above_target'
                pairs+=1;counts[side]+=1
                for name,slope in models.items():
                    prediction=speed_to_pace_min_mile(pace_to_speed_mps(anchor['moving_pace_min_mile'])+slope*delta)
                    signed=(prediction-truth['moving_pace_min_mile'])*60
                    errors[name][activity_id].append(signed)
                    by_side[side][name].append(signed)
    if not errors:
        return {'status':'insufficient_matched_observations','eligible_runs':len(runs),'folds':folds}
    ids=sorted(errors['no_correction'])
    mae={name:np.array([np.mean(np.abs(values[i])) for i in ids]) for name,values in errors.items()}
    comparisons={}
    for name in ('hr_only','hr_controlling_time'):
        delta=mae[name]-mae['no_correction']
        rng=np.random.default_rng(20261013)
        interval=np.quantile(np.mean(rng.choice(delta,size=(2000,len(delta)),replace=True),axis=1),[.025,.975])
        comparisons[name]={'error_change_seconds_per_mile':float(delta.mean()),'run_bootstrap_95':interval.tolist()}
    return {'status':'evaluated','eligible_runs':len(runs),'held_out_matched_runs':len(ids),'matched_target_windows':pairs,
            'run_balanced_mae_seconds_per_mile':{name:float(values.mean()) for name,values in mae.items()},
            'comparisons':comparisons,'pair_counts_by_direction':dict(counts),
            'directional_signed_error_seconds_per_mile':{side:{name:float(np.mean(values)) for name,values in models.items()} for side,models in by_side.items()},
            'folds':folds,
            'scope':'Local interpolation to independently recorded near-target-HR windows. Observational, selection-limited, and not causal fitness validation.'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);args=parser.parse_args()
    config=load_config('config.yaml')
    with sqlite3.connect('file:data/run_analysis.sqlite?mode=ro',uri=True) as c:
        c.row_factory=sqlite3.Row
        sets,_=load_model_window_sets(c,config,(120,300))
        result=audit(sets[120],sets[300],config)
    rendered=json.dumps(result,indent=2);print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(rendered+'\n')


if __name__=='__main__': main()
