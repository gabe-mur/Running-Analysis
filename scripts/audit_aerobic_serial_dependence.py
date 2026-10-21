"""Read-only serial dependence experiment; no application imports or changes."""
from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime,timedelta
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np

from run_analysis import pace_comparability as pc
from run_analysis.config import load_config,resolve_project_path
from run_analysis.read_cache import invariant_snapshot
from scripts.audit_aerobic_three_layers import inputs,simulations,CONTRASTS
from scripts.audit_progress_responsiveness import at,build_from_windows,covariance
from scripts.aerobic_three_layer_model import fit_period_layers
from scripts.aerobic_serial_model import correlation_matrix,fit_serial_layers,fit_serial_sensitivity


def audit(database,config,end,precision_file,draws,seed,support_deviance=None):
    precision=json.loads(precision_file.read_text())
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True) as connection:
        connection.row_factory=sqlite3.Row
        with invariant_snapshot(connection):
            sets,_=pc.load_model_window_sets(connection,config,(int(config['model'].get('window_seconds',300)),),check_elevation=True)
            windows,_=pc.load_overlapping_model_windows(connection,config,
                window_seconds=int(config['model']['primary_window_seconds']),stride_seconds=int(config['model']['primary_window_stride_seconds']),check_elevation=True)
            sets={k:[r for r in v if datetime.fromisoformat(r['start_time_utc'])<=end] for k,v in sets.items()}
            windows=[r for r in windows if datetime.fromisoformat(r['start_time_utc'])<=end]
            report=build_from_windows(connection,config,sets,windows);before=deepcopy(report)
            rows,m,g,groups,latest,used,eligible=inputs(report,precision,end);cov=covariance(report)
            times=np.array([(at(r)-latest).total_seconds()/86400 for r in rows]);augmented=np.r_[times,2.]
            y=np.log([r['pace_min_mile'] for r in rows])
            actual=fit_serial_sensitivity(y,m,g,cov,groups,times=times,future_time=2.,contrasts=CONTRASTS,support_deviance=support_deviance)
            # One sensitivity calculation per draw supplies both blind plug-in
            # and envelope results; keyed deterministic samples are shared.
            cache={};support_cache={}
            def candidate(y,m,g,c,groups,*,contrasts,scenario):
                fit=fit_serial_sensitivity(y,m,g,c,groups,times=times,future_time=2.,contrasts=contrasts,support_deviance=support_deviance)
                cache[(scenario,y.tobytes())]=dict(fit['conditional_selected_fit'],selected_kernel=fit['selected_kernel'])
                if support_deviance is not None:support_cache[(scenario,y.tobytes())]=fit['supported_kernel_fit']
                return fit
            def plugin(y,m,g,c,groups,*,contrasts,scenario):
                return cache.pop((scenario,y.tobytes()))
            def screened(y,m,g,c,groups,*,contrasts,scenario):
                return support_cache.pop((scenario,y.tobytes()))
            def oracle(y,m,g,c,groups,*,contrasts,scenario):
                # This is an oracle only for the explicitly generated Gaussian
                # serial controls, NOT for historical/heavy-tail donor noise.
                if scenario=='persistent_days':r=correlation_matrix(augmented,run_rho=.7)
                elif scenario.startswith('calendar_days_'):r=correlation_matrix(augmented,half_life_days=float(scenario.split('_')[-1]))
                else:r=np.eye(len(augmented))
                return fit_serial_layers(y,m,g,c,groups,correlation=r[:-1,:-1],future_correlation=r[:-1,-1],contrasts=contrasts)
            simulations_out={}
            methods=[('independent',fit_period_layers,False),('known_serial_kernel',oracle,True),
                ('all_kernel_envelope',candidate,True),('selected_kernel_plugin',plugin,True)]
            if support_deviance is not None:methods.append(('supported_kernel_envelope',screened,True))
            for name,model,scenario_model in methods:
                simulations_out[name]=simulations(rows,m,g,cov,groups,latest,eligible,draws,seed,model,
                    serial_controls=True,scenario_model=scenario_model)
                print(f'{name}: {len(simulations_out[name]["cells"])} cells complete',flush=True)
            assert not cache and not support_cache and report==before and connection.total_changes==0
    return dict(version='aerobic-serial-dependence-v2-stratified',deployed=False,source_database_read_only=True,support_deviance=support_deviance,
        precision_sha256=hashlib.sha256(precision_file.read_bytes()).hexdigest(),as_of=end.isoformat(),last_run=latest.isoformat(),
        actual=actual,simulation=simulations_out,
        assumptions=['All original ten control families retained; added calendar-time half lives of 3 and 7 days.',
            'Known-kernel control uses generator correlation only in the three explicit Gaussian serial cases.',
            'Blind grid fixed before evaluation: calendar half lives 0,1,3,7,14 days; run-index correlations .3,.5,.7,.9.',
            'Period-specific variances remain learned, shared calibration remains separate and cross-period covariance is retained.',
            'Blind point selects minimum restricted deviance evaluated at moment amplitudes, not maximized REML or an exact profile likelihood.',
            'Plug-in conditional intervals ignore kernel-selection uncertainty; all-kernel ranges are sensitivity envelopes, not labelled exact 95% intervals.',
            'Optional supported-kernel screen uses a declared restricted-deviance difference at moment amplitudes, not a formal likelihood-ratio confidence set; its union is only a sensitivity envelope.',
            'Future comparison retains correlation with previous days, measurement error and shared-calibration cancellation.',
            'No symptom attribution, readiness forecast, production score, adjustment coefficient, UI, planner or canonical evidence change.'])


def main():
    p=ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--precision',type=Path,default=Path('output/run-pace-precision-coverage-20261014.json'))
    p.add_argument('--draws',type=int,default=512);p.add_argument('--seed',type=int,default=20261016)
    p.add_argument('--support-deviance',type=float,default=None)
    p.add_argument('--as-of',type=datetime.fromisoformat,default=datetime.fromisoformat('2026-10-14T22:30:00+00:00'))
    args=p.parse_args()
    if args.output.exists() or args.as_of.tzinfo is None or args.draws<2 or (args.support_deviance is not None and (not np.isfinite(args.support_deviance) or args.support_deviance<0)):
        p.error('New output path, timezone-aware date, at least two draws and an optional finite nonnegative support screen required')
    root=Path(__file__).resolve().parents[1];config=load_config(root/'config.yaml')
    result=audit(resolve_project_path(root,config['paths']['database']),config,args.as_of,args.precision,args.draws,args.seed,args.support_deviance)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    fit=result['actual'];print(json.dumps(dict(selected_kernel=fit['selected_kernel'],monthly=fit['contrasts']['monthly'],
        selected_monthly=fit['conditional_selected_fit']['contrasts']['monthly']),indent=2))


if __name__=='__main__':main()
