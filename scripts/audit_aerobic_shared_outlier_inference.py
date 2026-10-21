"""Fixed-cohort inference sensitivity to the worst two prespecified stresses."""
from argparse import ArgumentParser
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from scripts.aerobic_joint_profile import GaussianChangeProfile
from scripts.audit_aerobic_three_layers import CONTRASTS


def selected_cases(influence):
    cases=influence['single_run_stress']
    return dict(worst_slower=max(cases,key=lambda v:v['current_shift_seconds_per_mile']),
                worst_faster=min(cases,key=lambda v:v['current_shift_seconds_per_mile']))


def audit(frozen_file,influence_file,*,draws,seed):
    frozen=json.loads(frozen_file.read_text());data=frozen['statistical_inputs']
    influence=json.loads(influence_file.read_text())
    assert influence['frozen_sha256']==hashlib.sha256(frozen_file.read_bytes()).hexdigest()
    index={r['activity_id']:i for i,r in enumerate(influence['leave_one_out'])}
    original=np.array(data['actual_log_pace']);samples={};points={}
    stresses=dict(baseline=None,**selected_cases(influence))
    for label,stress in stresses.items():
        y=original.copy()
        if stress is not None:y[index[stress['activity_id']]]+=math.log1p(stress['pace_change_percent']/100)
        model=GaussianChangeProfile(y,data['measurement_variance'],data['gradients'],data['shared_covariance'],
            data['groups'],times=data['times'],future_time=data['future_time'],mode='ml')
        samples[label]={};points[label]={}
        for target,contrast in CONTRASTS.items():
            free=model.fit(contrast)
            baseline=influence['baseline']['contrasts'][target]['estimate']
            points[label][target]=dict(estimate=free['estimate'],current_pace_min_mile=math.exp(free['beta'][0]),
                conditional_se=free['conditional_se'])
            samples[label][target]={}
            proposals=dict(zero=0.) if stress is None else dict(zero=0.,unmodified_point=baseline)
            for name,value in proposals.items():
                result=model.bootstrap_test(contrast,value,draws=draws,seed=seed,free=free)
                samples[label][target][name]=result
                print(json.dumps(dict(case=label,target=target,proposal=name,pvalue=result['pvalue'],
                    accepted=result['covered_95'],warnings=result['optimizer_warning_draws'])),flush=True)
    root=Path(__file__).resolve().parents[1]
    return dict(version='aerobic-shared-outlier-inference-v1',deployed=False,no_database_access=True,
        draws=draws,seed=seed,stresses=stresses,points=points,tests=samples,
        frozen_sha256=hashlib.sha256(frozen_file.read_bytes()).hexdigest(),
        influence_sha256=hashlib.sha256(influence_file.read_bytes()).hexdigest(),
        source_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in (
            'scripts/audit_aerobic_shared_outlier_inference.py','scripts/aerobic_joint_profile.py')},
        scope=['Worst positive/negative current-level shifts selected by the complete fixed influence audit, not inference p-values.',
            'Same full ML likelihood as the shared point; nuisance-residual inference is not borrowed.',
            'Unmodified fitted point is a sensitivity reference, not a known physiological ground truth.',
            'All full-budget p-values retained, including numerical warnings; no new clinical or sensor-error threshold.',
            'No evidence edit, live signal, VO2, adjustment coefficient or planner change.'])


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--frozen',type=Path,default=Path('output/aerobic-joint-bootstrap-validation-20261014.json'))
    parser.add_argument('--influence',type=Path,default=Path('output/aerobic-shared-influence-20261014.json'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--draws',type=int,default=127);parser.add_argument('--seed',type=int,default=20261127)
    args=parser.parse_args()
    if args.output.exists() or args.draws<19:parser.error('New output and at least 19 draws required')
    result=audit(args.frozen,args.influence,draws=args.draws,seed=args.seed)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
