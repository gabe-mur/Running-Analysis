"""Compatibility of one shared mean point with comparison-specific inference.

Both target the same Gaussian period log means. This does not transfer the
profile intervals to the different production pace-domain Huber estimand.
"""
from argparse import ArgumentParser
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from scripts.aerobic_joint_profile import GaussianChangeProfile
from scripts.audit_aerobic_three_layers import CONTRASTS
from scripts.audit_small_aerobic_changes import describe


def cases(data, influence):
    original=np.array(data['actual_log_pace']);n=len(original)
    yield dict(label='baseline',mask=np.ones(n,bool),y=original,
        proposals={k:v['estimate'] for k,v in influence['baseline']['contrasts'].items()})
    for i,row in enumerate(influence['leave_one_out']):
        mask=np.arange(n)!=i
        yield dict(label='leave_one_out',activity_id=row['activity_id'],mask=mask,y=original[mask],
            proposals={k:v['estimate']+row['contrast_shift_log'][k] for k,v in influence['baseline']['contrasts'].items()})
    index={r['activity_id']:i for i,r in enumerate(influence['leave_one_out'])}
    for row in influence['single_run_stress']:
        y=original.copy();y[index[row['activity_id']]]+=math.log1p(row['pace_change_percent']/100)
        yield dict(label=row.get('category','extreme_data_stress'),activity_id=row['activity_id'],
            pace_change_percent=row['pace_change_percent'],mask=np.ones(n,bool),y=y,
            proposals={k:v['estimate']+row['contrast_shift_log'][k] for k,v in influence['baseline']['contrasts'].items()})


def summarize_compatibility(results):
    summaries={}
    for label in sorted({v['label'] for v in results}):
        values=[v for v in results if v['label']==label]
        summaries[label]=dict(point_in_confidence_set=dict(accepted=sum(v['covered_95'] for v in values),
                tested=len(values),all_accepted=all(v['covered_95'] for v in values)),
            absolute_peak_difference_log=describe([abs(v['peak_difference_log']) for v in values]),
            evaluated_draws=describe([v['evaluated_draws'] for v in values]),
            outer_warning_cases=sum(bool(v['observed_unconverged_kernels']) for v in values),
            inner_warning_draws=sum(v['optimizer_warning_draws'] for v in values))
    return summaries


def audit(frozen_file,influence_file,*,draws,seed):
    data=json.loads(frozen_file.read_text())['statistical_inputs']
    influence=json.loads(influence_file.read_text())
    assert influence['frozen_sha256']==hashlib.sha256(frozen_file.read_bytes()).hexdigest()
    m=np.array(data['measurement_variance']);g=np.array(data['gradients']);groups=np.array(data['groups']);times=np.array(data['times'])
    results=[]
    for number,case in enumerate(cases(data,influence)):
        mask=case['mask']
        model=GaussianChangeProfile(case['y'],m[mask],g[mask],data['shared_covariance'],groups[mask],
            times=times[mask],future_time=data['future_time'],mode='nuisance_residual')
        for target,contrast in CONTRASTS.items():
            result=model.bootstrap_test(contrast,case['proposals'][target],draws=draws,seed=seed+number,decision_only=True)
            results.append(dict({k:v for k,v in case.items() if k not in ('mask','y','proposals')},target=target,
                shared_point=case['proposals'][target],profile_peak=result['estimate'],
                peak_difference_log=result['estimate']-case['proposals'][target],**{
                    k:result[k] for k in ('statistic','covered_95','pvalue_bounds','evaluated_draws',
                        'observed_unconverged_kernels','optimizer_warning_draws','nesting_refits')}))
    summaries=summarize_compatibility(results)
    for label,summary in summaries.items():print(label,json.dumps(summary),flush=True)
    root=Path(__file__).resolve().parents[1]
    return dict(version='aerobic-shared-point-profile-compatibility-v2-descriptive-counts',deployed=False,no_database_access=True,
        draws=draws,seed=seed,summary=summaries,results=results,
        frozen_sha256=hashlib.sha256(frozen_file.read_bytes()).hexdigest(),influence_sha256=hashlib.sha256(influence_file.read_bytes()).hexdigest(),
        source_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in (
            'scripts/audit_aerobic_shared_profile_compatibility.py','scripts/aerobic_joint_profile.py')},
        assumptions=['Shared point is the common full-ML period log-mean vector; nuisance-residual likelihood tests the same mean contrasts.',
            'A confidence set is defined by its coverage for the target parameter, not by requiring its profile peak to supply the displayed point.',
            'This audit checks numerical compatibility of the two summaries, not physiological truth or new confidence-set coverage.',
            'Fixed dependent sensitivity cases have descriptive counts only, not a binomial Monte Carlo coverage confidence interval.',
            'Comparison-specific nuisance fits never change the shared current-state point.',
            'No confidence set is forcibly expanded to contain the point; disagreement and optimizer warnings remain visible.',
            'The different production Huber pace statistic is not assigned these log-mean intervals.',
            'Early decisions preserve the complete fixed-bootstrap-budget answer; p-value bounds are not exact p-values.',
            'No live curve, model coefficient, VO2, planner or original evidence change.'])


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--frozen',type=Path,default=Path('output/aerobic-joint-bootstrap-validation-20261014.json'))
    parser.add_argument('--influence',type=Path,default=Path('output/aerobic-shared-influence-everyday-20261014.json'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--draws',type=int,default=63);parser.add_argument('--seed',type=int,default=20261129)
    args=parser.parse_args()
    if args.output.exists() or args.draws<19:parser.error('New output and at least 19 bootstrap draws required')
    result=audit(args.frozen,args.influence,draws=args.draws,seed=args.seed)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
