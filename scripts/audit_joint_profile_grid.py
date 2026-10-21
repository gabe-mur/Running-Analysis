"""Bootstrap checks of proposed changes on an already-frozen private cohort."""
from argparse import ArgumentParser
import hashlib
import json
import math
from pathlib import Path

from scripts.aerobic_joint_profile import GaussianChangeProfile
from scripts.audit_aerobic_three_layers import CONTRASTS


def main():
    p=ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--draws',type=int,default=127);p.add_argument('--seed',type=int,default=20261109)
    p.add_argument('--target',choices=list(CONTRASTS),default='ninety_day');p.add_argument('--mode',choices=['ml','nuisance_residual'],default='nuisance_residual')
    p.add_argument('--changes-percent',type=float,nargs='+',default=[-10,-7.5,-5,-2.5,0,2.5,5,7.5,10])
    args=p.parse_args()
    if args.output.exists() or args.draws<19 or not all(math.isfinite(v) and v>-100 for v in args.changes_percent):
        p.error('New output, at least 19 draws and finite changes above -100% required')
    source=json.loads(args.input.read_text());cohort=source['statistical_inputs']
    model=GaussianChangeProfile(cohort['actual_log_pace'],cohort['measurement_variance'],cohort['gradients'],cohort['shared_covariance'],
        cohort['groups'],times=cohort['times'],future_time=cohort['future_time'],mode=args.mode)
    contrast=CONTRASTS[args.target];free=model.fit(contrast);checks=[]
    for proposed in sorted(set(args.changes_percent)):
        test=model.bootstrap_test(contrast,math.log1p(proposed/100),draws=args.draws,seed=args.seed,free=free)
        checks.append(dict(proposed_change_percent=proposed,**test))
        print(json.dumps(dict(proposed_change_percent=proposed,pvalue=test['pvalue'],accepted=test['covered_95'],
            numerical_warnings=test['optimizer_warning_draws'])),flush=True)
    root=Path(__file__).resolve().parents[1]
    result=dict(version='aerobic-bootstrap-change-grid-v1',deployed=False,no_database_access=True,input=str(args.input),
        input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),target=args.target,mode=args.mode,
        point_change_percent=100*math.expm1(free['estimate']),checks=checks,
        source_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ('scripts/aerobic_joint_profile.py','scripts/audit_joint_profile_grid.py')},
        scope='Proposed-change tests on a fixed grid, not interpolated interval endpoints or proof of a connected confidence set. Common random seed across proposed changes; full bootstrap budget used for each p-value.')
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
