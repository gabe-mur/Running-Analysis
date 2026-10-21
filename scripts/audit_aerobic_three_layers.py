"""Read-only three-layer experiment with one current state for all comparisons."""
from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime,timedelta,timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3

import numpy as np

from run_analysis import pace_comparability as pc, aerobic_progress as ap
from run_analysis.config import load_config,resolve_project_path
from run_analysis.analytics import MIN_MEASUREMENT_SIGMA_MIN_MILE
from run_analysis.read_cache import invariant_snapshot
from scripts.aerobic_three_layer_model import fit_layers,fit_period_layers
from scripts.audit_progress_responsiveness import at,build_from_windows,covariance,speed_factor
from scripts.audit_small_aerobic_changes import noise_library,draw_noise,describe
from scripts.audit_aerobic_interval_coverage import wilson


CONTRASTS=dict(monthly=[-1.,1.,0.,0.],ninety_day=[-1.,0.,1.,0.])


def measurement_inputs(report,precision,rows):
    """Existing per-run precision, independent of a comparison anchor.

    Keep this shared by the frozen-cohort and historical reconstruction audits;
    a viewport or period grouping must not alter the measurement inputs.
    """
    index={r['activity_id']:r for r in precision['runs']}
    m=[];g=[];used=[]
    for row in rows:
        value=index[row['activity_id']]
        if not math.isclose(value['uncertainty_95_min_mile'],row['uncertainty_95_min_mile'],rel_tol=1e-9,abs_tol=1e-9):
            raise ValueError('Precision inventory is stale; rerun it on this evidence')
        p=row['pace_min_mile'];gradient=-np.array(row['shared_log_speed_gradient'])
        if value['sampling_precision']:
            sampling=max(v['sampling_sd_min_mile'] for v in value['sampling_precision'])/p
            variance=sampling**2+sum(v for k,v in value['log_variance_parts'].items() if k not in ('parameter','whole_run_dispersion'))
            source='time_block_sampling_plus_unmeasured_conditions'
        else:
            variance=max(0.,(row['uncertainty_95_min_mile']/1.96/p)**2-float(gradient@covariance(report)@gradient))
            source='retained_whole_dispersion_fallback'
        # Preserve the inherited TOTAL 10 sec/mi SD floor. Identified shared
        # calibration already fills part of that floor; do not add it twice.
        shared=max(0.,float(gradient@covariance(report)@gradient))
        m.append(max(1e-12,(MIN_MEASUREMENT_SIGMA_MIN_MILE/p)**2-shared,variance)/min(1.,row['trend_weight']))
        g.append(gradient);used.append(source)
    return np.array(m),np.array(g),used


def inputs(report,precision,end):
    eligible=sorted([r for r in report['runs'] if r['included_in_trend'] and r['trend_weight']>0 and at(r)<=end],key=at)
    end=at(eligible[-1]);rows=[r for r in eligible if end-timedelta(days=118)<at(r)<=end]
    m,g,used=measurement_inputs(report,precision,rows)
    groups=[]
    for row in rows:
        days=(end-at(row)).total_seconds()/86400
        groups.append(0 if days<28 else 1 if days<56 else 2 if days>=90 else 3)
    if set(groups)!={0,1,2,3}:raise ValueError('This audit needs all four comparison periods; no missing period is invented')
    return rows,np.array(m),np.array(g),np.array(groups),end,used,eligible


def rate(covered):
    count=int(sum(covered));return dict(covered=count,n=len(covered),rate=count/len(covered),
        monte_carlo_wilson_95=wilson(count,len(covered)))


def summarize(samples):
    result={}
    for target in ('level','monthly','ninety_day','next_day'):
        values=[s[target] for s in samples]
        result[target]=dict(coverage_95=rate([v['covered'] for v in values]),
            bias_log=float(np.mean([v['estimate']-v['truth'] for v in values])),
            half_width_log=describe([(v['range'][1]-v['range'][0])/2 for v in values]))
    result['estimated_day_sd']=describe([s['day_sd'] for s in samples])
    selected=[s['selected_kernel'] for s in samples if s.get('selected_kernel') is not None]
    if selected:
        result['selected_kernels']={k:selected.count(k) for k in sorted(set(selected))}
        result['by_selected_kernel']={k:dict(n=selected.count(k),**{target:rate([s[target]['covered'] for s in samples if s.get('selected_kernel')==k])
            for target in ('level','monthly','ninety_day','next_day')}) for k in sorted(set(selected))}
    support=[s['supported_kernels'] for s in samples if s.get('supported_kernels') is not None]
    if support:
        result['supported_kernel_count']=describe([len(v) for v in support])
        result['support_inclusion']={k:sum(k in v for v in support) for k in sorted({k for v in support for k in v})}
    return result


def simulations(rows,m,g,cov,groups,end,eligible,draws,seed,model=fit_layers,*,serial_controls=False,scenario_model=False,
    observer=None,only_scenarios=None,gains=(0.,.005,.01,.02)):
    donors=[r for r in eligible if at(r)>=end-timedelta(days=180)]
    libraries={}
    for days in (7,14):
        for detrend in (False,True):libraries[(days,detrend)]=noise_library(donors,days,detrend)[0]
    scenarios=['measurement_only','independent_days','understated_measurement',
        'historical_7','historical_14','detrended_7','detrended_14','persistent_days','heavy_tail_days','unequal_day_variance']
    if serial_controls:scenarios+=['calendar_days_3','calendar_days_7']
    if only_scenarios is not None and (not only_scenarios or not set(only_scenarios)<=set(scenarios)):
        raise ValueError('Nonempty subset of known simulation scenarios required')
    cells=[];base=math.log(float(np.median([r['pace_min_mile'] for r in donors])))
    for number,scenario in enumerate(scenarios):
        if only_scenarios is not None and scenario not in only_scenarios:continue
        for gain in gains:
            rng=np.random.default_rng(seed+number*100)
            latent=np.array([base-math.log(speed_factor(at(r),end-timedelta(days=84),gain,84)) for r in rows])
            means=np.array([latent[groups==i].mean() for i in range(4)])
            samples=[]
            for _ in range(draws):
                delta=rng.multivariate_normal(np.zeros(len(cov)),cov)
                future_m=float(np.median(m[groups==0]));future_g=g[groups==0].mean(axis=0)
                if 'historical' in scenario or 'detrended' in scenario:
                    days=int(scenario.split('_')[-1]);future_row=dict(rows[-1],start_time=(end+timedelta(days=2)).isoformat())
                    sequence=draw_noise(rows+[future_row],libraries[(days,'detrended' in scenario)],rng)
                    noise=sequence[:-1];future_noise=sequence[-1]
                else:
                    multiplier=2. if scenario=='understated_measurement' else 1.
                    noise=rng.normal(0.,np.sqrt(m)*multiplier)
                    if scenario in ('independent_days','understated_measurement'):
                        noise+=rng.normal(0.,.035,len(rows))
                    elif scenario=='heavy_tail_days':noise+=rng.standard_t(3,len(rows))*.035/math.sqrt(3)
                    elif scenario=='unequal_day_variance':noise+=rng.normal(0.,np.where(groups==0,.055,.015))
                    elif scenario=='persistent_days':
                        spell=rng.normal(0.,.035)
                        for i in range(len(rows)):
                            spell=.7*spell+rng.normal(0.,.035*math.sqrt(1-.7**2));noise[i]+=spell
                    elif scenario.startswith('calendar_days_'):
                        half_life=float(scenario.split('_')[-1]);spell=rng.normal(0.,.035)
                        for i in range(len(rows)):
                            rho=0. if i==0 else 2**(-(at(rows[i])-at(rows[i-1])).total_seconds()/86400/half_life)
                            spell=rho*spell+rng.normal(0.,.035*math.sqrt(1-rho**2));noise[i]+=spell
                y=latent+noise+g@delta
                fit=model(y,m,g,cov,groups,contrasts=CONTRASTS,**({'scenario':scenario} if scenario_model else {}))
                sd=0. if scenario=='measurement_only' else .055 if scenario=='unequal_day_variance' else .035
                if 'historical' in scenario or 'detrended' in scenario:
                    # Sample another donor spell, not a made-up independent SD.
                    future=means[0]+future_noise+float(future_g@delta)
                else:
                    future_rho=2**(-2/float(scenario.split('_')[-1])) if scenario.startswith('calendar_days_') else None
                    future_day=(future_rho*spell+rng.normal(0.,.035*math.sqrt(1-future_rho**2)) if future_rho is not None
                        else .7*spell+rng.normal(0.,.035*math.sqrt(1-.7**2)) if scenario=='persistent_days'
                        else rng.standard_t(3)*.035/math.sqrt(3) if scenario=='heavy_tail_days' else rng.normal(0.,sd))
                    future=means[0]+future_day+rng.normal(0.,math.sqrt(future_m)*(2. if scenario=='understated_measurement' else 1.))+float(future_g@delta)
                targets=dict(level=(fit['periods'][0],means[0]),
                    monthly=(fit['contrasts']['monthly'],means[1]-means[0]),
                    ninety_day=(fit['contrasts']['ninety_day'],means[2]-means[0]),
                    next_day=(fit['next_comparable_measured_run'],future))
                if observer is not None:observer(scenario,gain,y,fit,targets)
                samples.append(dict(day_sd=fit['extra_day_log_sd'],selected_kernel=fit.get('selected_kernel'),supported_kernels=fit.get('supported_kernels'),**{name:dict(estimate=v['estimate'],truth=truth,
                    range=v['range_95'],covered=v['range_95'][0]<=truth<=v['range_95'][1]) for name,(v,truth) in targets.items()}))
            cells.append(dict(scenario=scenario,gain_percent=gain*100,draws=draws,**summarize(samples)))
    return dict(cells=cells,draws_per_cell=draws,seed=seed,
        truth='Mean latent log pace at observed dates in each fixed period; endpoint gain is not the answer key.',
        next_day_scope='Next measured standardized run at unchanged period-average level, current median run precision and mean adjustment exposure; shared calibration and donor-spell continuation retained.')


def audit(database,config,end,precision_file,draws,period_variance=False):
    precision=json.loads(precision_file.read_text());digest=hashlib.sha256(precision_file.read_bytes()).hexdigest()
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True) as connection:
        connection.row_factory=sqlite3.Row
        with invariant_snapshot(connection):
            size=int(config['model'].get('window_seconds',300))
            sets,_=pc.load_model_window_sets(connection,config,(size,),check_elevation=True)
            windows,_=pc.load_overlapping_model_windows(connection,config,
                window_seconds=int(config['model']['primary_window_seconds']),stride_seconds=int(config['model']['primary_window_stride_seconds']),check_elevation=True)
            sets={k:[r for r in v if datetime.fromisoformat(r['start_time_utc'])<=end] for k,v in sets.items()}
            windows=[r for r in windows if datetime.fromisoformat(r['start_time_utc'])<=end]
            report=build_from_windows(connection,config,sets,windows);before=deepcopy(report)
            rows,m,g,groups,latest,used,eligible=inputs(report,precision,end);cov=covariance(report)
            y=np.log([r['pace_min_mile'] for r in rows])
            model=fit_period_layers if period_variance else fit_layers
            fit=model(y,m,g,cov,groups,contrasts=CONTRASTS)
            shipped=ap.progress_from_comparability(report,as_of=end,history_days=1,config=config)
            simulation=simulations(rows,m,g,cov,groups,latest,eligible,draws,20261016,model)
            assert report==before and connection.total_changes==0
    pace=math.exp(fit['periods'][0]['estimate'])
    return dict(version='aerobic-three-layer-audit-v3-measured-prediction',period_specific_variance=period_variance,
        deployed=False,source_database_read_only=True,
        precision_inventory=str(precision_file),precision_sha256=digest,as_of=end.isoformat(),last_run=latest.isoformat(),
        source_counts={name:used.count(name) for name in set(used)},
        measurement_log_sd=describe(np.sqrt(m)),measurement_sd_seconds_per_mile=describe(np.sqrt(m)*np.exp(y)*60),
        fit=fit,current_pace_min_mile=pace,current_range_95_min_mile=[math.exp(v) for v in fit['periods'][0]['range_95']],
        day_sd_seconds_per_mile=fit['extra_day_log_sd']*pace*60,
        monthly_change_percent=100*math.expm1(fit['contrasts']['monthly']['estimate']),
        monthly_range_95_percent=[100*math.expm1(v) for v in fit['contrasts']['monthly']['range_95']],
        next_day_range_95_min_mile=[math.exp(v) for v in fit['next_comparable_measured_run']['range_95']],
        shipped_current_pace=shipped['current']['pace_min_mile'],shipped_change=shipped['change'],simulation=simulation,
        assumptions=['One current state fitted jointly with prior monthly, prior 90-day and intervening period means; history selector cannot change it.',
            'Means are constant within periods; gradual-gain truth uses the equally weighted latent log pace at observed run dates.',
            'Run precision from the existing block inventory; common adjustment covariance retained; unsupported runs keep full-dispersion fallback.',
            'Inherited total 10 sec/mi SD floor retained after subtracting identified shared covariance; health/context reliability scaling retained.',
            ('Period-specific day variances estimated by separate generalized Paule-Mandel moment equations, bounded at zero.' if period_variance else 'One common day variance estimated by generalized Paule-Mandel moment equation Q(tau)=n-p, bounded at zero.'),
            ('Finite-variance uncertainty uses residual-projection covariance and Welch-Satterthwaite t degrees of freedom, bounded by total residual df.' if period_variance else 'Modified Hartung-Knapp scale max(1,Q/df), with Student-t df=n-p; covariance-aware extension is experimental.'),
            'Baseline day effects independent and Gaussian; persistent/heavy-tail/unequal-variance cases explicitly challenge or distinguish the assumptions.',
            'Next-day measured-run prediction assumes unchanged period mean, current median run precision and mean adjustment exposure; shared error cancels only through the explicit exposure difference.',
            'Unobserved true next-day performance has a separate interval; neither prediction is raw outdoor pace, readiness or a training-effect forecast.',
            'No workout, live score, threshold, UI, canonical evidence or planner change.'])


def main():
    p=ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--precision',type=Path,default=Path('output/run-pace-precision-coverage-20261014.json'))
    p.add_argument('--draws',type=int,default=512)
    p.add_argument('--period-variance',action='store_true')
    p.add_argument('--as-of',type=datetime.fromisoformat,default=datetime.now(timezone.utc))
    args=p.parse_args()
    if args.output.exists() or args.as_of.tzinfo is None or args.draws<2:p.error('New output path, timezone-aware date and at least two draws required')
    root=Path(__file__).resolve().parents[1];config=load_config(root/'config.yaml')
    result=audit(resolve_project_path(root,config['paths']['database']),config,args.as_of,args.precision,args.draws,args.period_variance)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('current_pace_min_mile','current_range_95_min_mile','day_sd_seconds_per_mile','monthly_change_percent','monthly_range_95_percent','next_day_range_95_min_mile')},indent=2))


if __name__=='__main__':main()
