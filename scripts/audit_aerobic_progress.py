"""Numerical prototype only: no live model, planner, database or UI changes."""
from argparse import ArgumentParser
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3

import numpy as np

from run_analysis.aerobic_progress import build_aerobic_progress
from run_analysis.config import load_config
from run_analysis.model_windows import load_overlapping_model_windows
from run_analysis.objective_modeling import _primary_window_weight, _pre_quality_indexes
from run_analysis.pace_comparability import build_pace_comparability, _features
from run_analysis.quality_phases import detect_continuous_quality_phase
from run_analysis.read_cache import invariant_snapshot


def main():
    parser=ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();config=load_config('config.yaml')
    with sqlite3.connect('file:data/run_analysis.sqlite?mode=ro',uri=True) as c:
        c.row_factory=sqlite3.Row
        with invariant_snapshot(c):
            report=build_pace_comparability(c,config)
            windows,_=load_overlapping_model_windows(c,config,
                window_seconds=int(config['model']['primary_window_seconds']),
                stride_seconds=int(config['model']['primary_window_stride_seconds']),check_elevation=True)
            grouped=defaultdict(list)
            for row in windows:grouped[row['activity_id']].append(row)
            covariance=np.zeros((4,4));covariance[:3,:3]=report['calibration']['covariance']
            covariance[3,3]=report['calibration']['heat']['sd']**2
            observations=[]
            for run in report['runs']:
                rows=grouped[run['activity_id']]
                phase=detect_continuous_quality_phase(c,config,run['activity_id']) if rows[0].get('workout_type')=='tempo_threshold' else None
                rows=[rows[i] for i in _pre_quality_indexes(rows,list(range(len(rows))),phase)]
                features=[_features(r,config) for r in rows]
                vectors=[[-f['hr_log'],-f['time'],f['wind_log'],f['heat'] or 0.] for f in features]
                gradient=np.average(vectors,axis=0,weights=[_primary_window_weight(r,config) for r in rows])
                observations.append(dict(run,shared_log_speed_gradient=gradient.tolist()))
            assert c.total_changes==0
    at=datetime.now(timezone.utc)
    views={str(days):build_aerobic_progress(observations,as_of=at,covariance=covariance,history_days=days)
           for days in (14,28,90,365)}
    assert all(v['current']==views['28']['current'] and v['change']==views['28']['change']
               and v['change_90d']==views['28']['change_90d'] for v in views.values())
    result=dict(database_unchanged=True,live_model_changed=False,planner_changed=False,
        reference_hr_bpm=config['target_hr'],calibration_covariance=covariance.tolist(),
        observations=observations,views=views,
        remaining_work='Numerical estimator prototype; no UI promotion or independent fitness-accuracy acceptance.',
        gradient_basis='First-order weighted-mean feature sensitivity, matching existing comparable-pace uncertainty; robust-weight derivatives are not modeled.')
    from run_analysis.privacy import private_file
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');private_file(args.output)
    print(json.dumps({k:views['28'][k] for k in ('current','prior','change','change_90d','days_since_observation')},indent=2))
    print('Display-window invariance: passed; saved:',args.output)


if __name__=='__main__':main()
