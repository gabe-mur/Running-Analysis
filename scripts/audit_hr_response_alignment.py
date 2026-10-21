"""Read-only held-out evaluation of delayed-effort HR alignment.

Run-centered coherence is not independently measured fitness. The zero-response
control and all fixed response times are retained, including failures.
"""
from argparse import ArgumentParser
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np

from run_analysis.config import load_config, resolve_project_path
from run_analysis.hr_response_alignment import TAUS, held_out_tau, training_losses, corrected_windows
from run_analysis.model_windows import load_overlapping_model_windows
from run_analysis.objective_modeling import _primary_window_weight
from run_analysis.pace_comparability import build_pace_comparability, compare_run_windows
from run_analysis.privacy import private_file
from run_analysis.read_cache import invariant_snapshot


def audit(connection, config):
    base=build_pace_comparability(connection,config)
    originals={r['activity_id']:r for r in base['runs'] if r['included_in_trend'] and r['trend_weight']>0}
    windows,_=load_overlapping_model_windows(connection,config,window_seconds=120,stride_seconds=60,
                                             check_elevation=True,hr_response=True)
    groups=defaultdict(list)
    for row in windows:
        if row['activity_id'] in originals:
            groups[row['activity_id']].append(row)
    losses={identity:training_losses(rows[0].get('_hr_response_training',[]),config,base['calibration'])
            for identity,rows in groups.items()}
    results=[]
    for identity,rows in groups.items():
        before=originals[identity]
        check=compare_run_windows(rows,[_primary_window_weight(r,config) for r in rows],config,base['calibration'])
        np.testing.assert_allclose(check['pace_min_mile'],before['pace_min_mile'],atol=1e-10,rtol=1e-10)
        index=held_out_tau(losses,identity)
        order=sorted([key for key in groups if originals[key]['start_time']<before['start_time']],key=lambda key:originals[key]['start_time'])
        prefix={key:losses[key] for key in order}
        prefix_index=held_out_tau(prefix,identity)
        variants=[]
        for ti,tau in enumerate(TAUS):
            selected=corrected_windows(rows,ti)
            if not selected:
                variants.append(None);continue
            result=compare_run_windows(selected,[_primary_window_weight(r,config) for r in selected],config,base['calibration'])
            variants.append(dict(tau=tau,pace=result['pace_min_mile'],delta_seconds=60*(result['pace_min_mile']-before['pace_min_mile']),
                windows=len(selected),source_windows=len(rows)))
        results.append(dict(activity_id=identity,date=before['start_time'],before_pace=before['pace_min_mile'],
            held_out_tau=TAUS[index],prefix_tau=TAUS[prefix_index],losses=losses[identity],
            held_out=variants[index],prefix=variants[prefix_index],variants=variants))
    valid=[r for r in results if r['losses'] is not None]
    early=[r for r in valid if r['date']>='2026-03-01']
    def summary(rows):
        old=[r['losses'][0] for r in rows]
        new=[r['losses'][TAUS.index(r['held_out_tau'])] for r in rows]
        prefix=[r['losses'][TAUS.index(r['prefix_tau'])] for r in rows]
        return dict(runs=len(rows),mean_loss_before=float(np.mean(old)),mean_loss_held_out=float(np.mean(new)),
            mean_loss_prefix=float(np.mean(prefix)),improved_runs=sum(b<a for a,b in zip(old,new)),
            worsened_runs=sum(b>a for a,b in zip(old,new)),
            median_score_shift_seconds=float(np.median([r['held_out']['delta_seconds'] for r in rows if r['held_out']])))
    return dict(version='hr-response-alignment-audit-v1',live_changed=False,source_database_read_only=True,
        tau_candidates=list(TAUS),calibration=base['calibration'],summary=summary(valid),recent_summary=summary(early),runs=results,
        assumptions=[
            'First-order terrain-effort response; instantaneous zero-response candidate included.',
            'Response time selected without the target physical run; every training run supplies one centered loss.',
            'Separate earlier-run-only selection retained; shared HR/drift calibration is frozen full-history, not a full historical forecast.',
            'Training observes HR through configured max HR, before fitness-window HR censoring; no score/outlier rank selection.',
            'Current normalization, role/health policy and canonical scores preserved outside this experiment.',
            'Within-run consistency is not physiological validation, proof of sensor correctness or proof that all outliers are artifacts.',
            'No sleep, medication, readiness or new fitness-preference weights.',
            'Window ratios approximate a response to terrain-adjusted speed; wind kinetics and nonlinear/asymmetric HR dynamics are not identified.',
            'Long moving recording gaps are not assumed to be settled running; unsupported response windows are missing, not zero.'
        ])


def main():
    parser=ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():parser.error('New output path required')
    root=Path(__file__).resolve().parents[1];config=load_config(root/'config.yaml')
    database=resolve_project_path(root,config['paths']['database'])
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True) as connection:
        connection.row_factory=sqlite3.Row
        with invariant_snapshot(connection):
            result=audit(connection,config);assert connection.total_changes==0
    result['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with args.output.open('x') as target:json.dump(result,target,indent=2,allow_nan=False)
    private_file(args.output)
    print(json.dumps(dict(summary=result['summary'],recent=result['recent_summary'],
        examples=[r for r in result['runs'] if r['activity_id'] in (352,353,356,33,34,373,377,390)]),indent=2))


if __name__=='__main__':main()
