"""Read-only coverage audit for joint change/persistence/day-variance profiles."""
from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np

from run_analysis import pace_comparability as pc
from run_analysis.config import load_config,resolve_project_path
from run_analysis.read_cache import invariant_snapshot
from scripts.audit_aerobic_three_layers import inputs,simulations,CONTRASTS,rate
from scripts.audit_progress_responsiveness import at,build_from_windows,covariance
from scripts.audit_small_aerobic_changes import describe
from scripts.aerobic_serial_model import fit_serial_sensitivity
from scripts.aerobic_joint_profile import GaussianChangeProfile,LR_95


def audit(database,config,end,precision_file,draws,seed,only_scenarios=None,gains=(0.,.005,.01,.02),targets=('ninety_day',),
    modes=('ml','nuisance_residual'),bootstrap_draws=0):
    precision=json.loads(precision_file.read_text())
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True) as connection:
        connection.row_factory=sqlite3.Row
        with invariant_snapshot(connection):
            sets,_=pc.load_model_window_sets(connection,config,(int(config['model'].get('window_seconds',300)),),check_elevation=True)
            windows,_=pc.load_overlapping_model_windows(connection,config,window_seconds=int(config['model']['primary_window_seconds']),
                stride_seconds=int(config['model']['primary_window_stride_seconds']),check_elevation=True)
            sets={k:[r for r in v if datetime.fromisoformat(r['start_time_utc'])<=end] for k,v in sets.items()}
            windows=[r for r in windows if datetime.fromisoformat(r['start_time_utc'])<=end]
            report=build_from_windows(connection,config,sets,windows);before=deepcopy(report)
            rows,m,g,groups,latest,used,eligible=inputs(report,precision,end);cov=covariance(report)
            times=np.array([(at(r)-latest).total_seconds()/86400 for r in rows])
            actual_y=np.log([r['pace_min_mile'] for r in rows]);actual={}
            for mode in modes:
                model=GaussianChangeProfile(actual_y,m,g,cov,groups,times=times,future_time=2.,mode=mode)
                actual[mode]={target:model.interval(CONTRASTS[target]) for target in targets}
                if bootstrap_draws:
                    for target in targets:actual[mode][target]['bootstrap_zero_test']=model.bootstrap_test(CONTRASTS[target],0.,draws=bootstrap_draws,seed=seed+900000)
            cells={};warning_examples=[]
            def old_selected(y,m,g,c,groups,*,contrasts):
                return fit_serial_sensitivity(y,m,g,c,groups,times=times,future_time=2.,contrasts=contrasts)['conditional_selected_fit']
            def observe(scenario,gain,y,fit,truths):
                for mode in modes:
                    model=GaussianChangeProfile(y,m,g,cov,groups,times=times,future_time=2.,mode=mode)
                    for target in targets:
                        truth=truths[target][1];result=model.likelihood_ratio(CONTRASTS[target],truth)
                        key=(scenario,gain,mode,target)
                        if bootstrap_draws:
                            index=len(cells.get(key,[]));scenario_number=int.from_bytes(hashlib.sha256(scenario.encode()).digest()[:4])
                            result['bootstrap']=model.bootstrap_test(CONTRASTS[target],truth,draws=bootstrap_draws,seed=seed+scenario_number*10000+index,decision_only=True)
                        if (result['unconverged_kernels'] or result.get('bootstrap',{}).get('optimizer_warning_draws')) and len(warning_examples)<4:
                            warning_examples.append(dict(scenario=scenario,gain=gain,mode=mode,target=target,y=y.tolist(),result=result))
                        cells.setdefault(key,[]).append(dict(truth=truth,**result))
                        if len(cells[key])==draws:
                            covered=sum(s['covered_95'] for s in cells[key]);bad=sum(bool(s['unconverged_kernels']) for s in cells[key])
                            boot='' if not bootstrap_draws else f'; bootstrap {sum(v["bootstrap"]["covered_95"] for v in cells[key])}/{draws}'
                            print(f'{scenario} gain={gain*100:g}% {mode} {target}: {covered}/{draws} covered; {bad} optimizer warnings{boot}',flush=True)
            baseline=simulations(rows,m,g,cov,groups,latest,eligible,draws,seed,old_selected,serial_controls=True,
                only_scenarios=only_scenarios,gains=gains,observer=observe)
            assert report==before and connection.total_changes==0
    summaries=[]
    for (scenario,gain,mode,target),values in cells.items():
        summary=dict(scenario=scenario,gain_percent=gain*100,mode=mode,target=target,
            coverage_95=rate([v['covered_95'] for v in values]),statistic=describe([v['statistic'] for v in values]),
            bias_log=float(np.mean([v['estimate']-v['truth'] for v in values])),
            optimizer_warning_draws=sum(bool(v['unconverged_kernels']) for v in values),
            free_kernel_counts={k:sum(v['free_kernel']==k for v in values) for k in sorted({v['free_kernel'] for v in values})})
        if bootstrap_draws:
            summary['bootstrap']=dict(coverage_95=rate([v['bootstrap']['covered_95'] for v in values]),
                pvalue_lower_bounds=describe([v['bootstrap']['pvalue_bounds'][0] for v in values]),
                pvalue_upper_bounds=describe([v['bootstrap']['pvalue_bounds'][1] for v in values]),
                evaluated_draws=describe([v['bootstrap']['evaluated_draws'] for v in values]),
                nesting_refits=sum(v['bootstrap']['nesting_refits'] for v in values),
                inner_optimizer_warning_draws=sum(v['bootstrap']['optimizer_warning_draws'] for v in values),
                outer_optimizer_warning_draws=sum(bool(v['bootstrap']['observed_unconverged_kernels']) for v in values),
                inner_draws_per_history=bootstrap_draws)
        summaries.append(summary)
    root=Path(__file__).resolve().parents[1]
    return dict(version='aerobic-joint-profile-v3-exact-early-decision',deployed=False,source_database_read_only=True,as_of=end.isoformat(),
        precision_sha256=hashlib.sha256(precision_file.read_bytes()).hexdigest(),seed=seed,draws=draws,
        scope=dict(scenarios=only_scenarios,gains_percent=[v*100 for v in gains],targets=list(targets),modes=list(modes),bootstrap_draws=bootstrap_draws),
        statistical_inputs=dict(actual_log_pace=actual_y.tolist(),measurement_variance=m.tolist(),gradients=g.tolist(),
            shared_covariance=cov.tolist(),groups=groups.tolist(),times=times.tolist(),future_time=2.),
        source_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in
            ('scripts/aerobic_joint_profile.py','scripts/aerobic_serial_model.py','scripts/audit_aerobic_three_layers.py','scripts/audit_aerobic_joint_profile.py')},
        actual=actual,baseline=baseline,profile_coverage=summaries,numerical_warning_examples=warning_examples,
        assumptions=['Same physical run precision inventory, shared calibration covariance and nine predeclared correlation kernels.',
            'Each proposed change refits all period day SDs and searches all kernels; other period means are analytically profiled.',
            'ML uses the full Gaussian likelihood; nuisance_residual uses a Gaussian likelihood after projecting out means orthogonal to the tested contrast.',
            'The latter can have contrast-specific profile peaks; it is an inference experiment, not a replacement shared current-state contract.',
            'Critical statistic is the predeclared asymptotic chi-square(1) 95% value, not a tuned band multiplier; small-sample coverage must be checked.',
            'Coverage evaluates the LR at known simulated truth without computing interval endpoints; truth is not used for the free point fit.',
            'Optional parametric bootstrap simulates from the constrained fitted null and refits both hypotheses, including kernel search, using (1+exceedances)/(B+1) p-values; fitted Gaussian null may still be misspecified.',
            'Outer coverage tests stop inner sampling only when remaining fixed-budget draws cannot change the 5% decision; return p-value bounds, not invented exact p-values/quantiles. Actual-data tests use all draws.',
            'Kernel family/grid and measurement uncertainty fixed; projected multistart BFGS numerical convergence reported separately from coverage.',
            'No model, UI, calibration coefficient, training recommendation, planner, readiness estimate or canonical evidence change.'])


def main():
    p=ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--precision',type=Path,default=Path('output/run-pace-precision-coverage-20261014.json'))
    p.add_argument('--draws',type=int,default=128);p.add_argument('--seed',type=int,default=20261101)
    p.add_argument('--scenarios',nargs='+');p.add_argument('--gains-percent',type=float,nargs='+',default=[0,.5,1,2])
    p.add_argument('--targets',nargs='+',choices=list(CONTRASTS),default=['ninety_day'])
    p.add_argument('--modes',nargs='+',choices=['ml','nuisance_residual'],default=['ml','nuisance_residual'])
    p.add_argument('--bootstrap-draws',type=int,default=0)
    p.add_argument('--as-of',type=datetime.fromisoformat,default=datetime.fromisoformat('2026-10-14T22:30:00+00:00'))
    args=p.parse_args()
    if args.output.exists() or args.as_of.tzinfo is None or args.draws<2 or (args.bootstrap_draws!=0 and args.bootstrap_draws<19) or not all(np.isfinite(v) and v>=0 for v in args.gains_percent):
        p.error('New output, timezone-aware date, at least two draws and finite nonnegative gains required')
    root=Path(__file__).resolve().parents[1];config=load_config(root/'config.yaml')
    result=audit(resolve_project_path(root,config['paths']['database']),config,args.as_of,args.precision,args.draws,args.seed,
        args.scenarios,tuple(v/100 for v in args.gains_percent),tuple(args.targets),tuple(args.modes),args.bootstrap_draws)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({mode:{k:{'estimate':v['estimate'],'range_95':v['range_95']} for k,v in values.items()}
        for mode,values in result['actual'].items()},indent=2))


if __name__=='__main__':main()
