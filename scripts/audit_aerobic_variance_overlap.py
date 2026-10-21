"""Compare two predeclared overlap-accounting candidates; never deploy either.

Envelope: max(claimed independent mean variance, observed mean variance).
Excess moments: claimed independent mean variance plus residual mean variance
beyond its expected measurement/shared contribution. No fitted multiplier.
"""
from argparse import ArgumentParser
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path

import numpy as np

from run_analysis.config import load_config, resolve_project_path
from scripts.audit_aerobic_interval_coverage import audit, period_parts, coverage
from scripts.audit_progress_responsiveness import at


VARIANTS = ('variance_envelope', 'excess_moments')


def candidate_period(rows, end, cov):
    p, parts, design = period_parts(rows, end, cov, include_design=True)
    w = design['weight']; independent = design['independent']; g = design['gradients']
    expected_cov = np.diag(independent)+g@cov@g.T
    # Conditional on these weights, observed residuals contain both independent
    # measurement error and variation caused by shared coefficient error.
    center = np.eye(len(w))-np.ones((len(w),1))*w[None,:]
    residual_cov = center@expected_cov@center.T
    expected_scatter = float(w@np.diag(residual_cov))
    observed_scatter = float(w@((design['pace']-p['pace_min_mile'])**2))
    w2 = float(w@w)
    excess_independent_mean = max(0., observed_scatter-expected_scatter)*w2/max(1-w2, 1e-12)
    weeks = np.array([math.floor(at(r).timestamp()/86400/7) for r in design['rows']])
    groups = np.unique(weeks); count = len(groups)
    correction = count/(count-1) if count>1 else 1.
    observed_cluster = 0.; expected_cluster = 0.
    for group in groups:
        wb = w*(weeks==group)
        score = wb@center
        observed_cluster += float(wb@(design['pace']-p['pace_min_mile']))**2
        expected_cluster += float(score@expected_cov@score)
    excess_cluster = max(0., correction*(observed_cluster-expected_cluster))
    parts['variance_envelope'] = max(parts['measurement'], parts['day'])
    parts['excess_moments'] = parts['measurement']+max(excess_independent_mean, excess_cluster)
    return p, parts


def evaluate_candidates(rows, end, cov):
    current, cp = candidate_period(rows, end, cov)
    out = dict(level=dict(estimate=current['pace_min_mile'], reported=current['random_variance'], **cp,
        established=current['evidence']=='established'))
    for name, days in (('monthly',28),('ninety_day',90)):
        prior, pp = candidate_period(rows, end-timedelta(days=days), cov)
        c,p = current['pace_min_mile'], prior['pace_min_mile']
        gradient = np.array(prior['shared_pace_gradient'])/p-np.array(current['shared_pace_gradient'])/c
        out[name] = dict(estimate=math.log(p/c), shared=max(0.,float(gradient@cov@gradient)),
            reported=current['random_variance']/c**2+prior['random_variance']/p**2,
            **{k:cp[k]/c**2+pp[k]/p**2 for k in ('measurement','day',*VARIANTS)},
            established=current['evidence']==prior['evidence']=='established')
    return out


def persistence_inventory(rows, end):
    """Descriptive association, not identification of noise versus real fitness."""
    eligible=sorted([r for r in rows if r.get('included_in_trend') and r['trend_weight']>0 and at(r)<=end],key=at)
    end=at(eligible[-1]);result=[]
    for days in (90,180):
        selected=[r for r in eligible if at(r)>=end-timedelta(days=days)]
        times=np.array([(at(r)-at(selected[0])).total_seconds()/86400 for r in selected])
        for field in ('pace_min_mile','observed_pace_min_mile'):
            if any(field not in r for r in selected):continue
            logs=np.log([r[field] for r in selected])
            for detrend in (False,True):
                design=np.column_stack([np.ones(len(times)),times-times.mean()])
                residual=logs-design@np.linalg.lstsq(design,logs,rcond=None)[0] if detrend else logs-logs.mean()
                lags=[]
                for lag in (1,2,3):
                    # No crossing the summer interruption; use contiguous spells.
                    pairs=[(i-lag,i) for i in range(lag,len(times))
                        if np.max(np.diff(times[i-lag:i+1]))<=14]
                    left=np.array([residual[a] for a,b in pairs]);right=np.array([residual[b] for a,b in pairs])
                    lags.append(dict(lag_runs=lag,pairs=len(pairs),
                        pearson_correlation=float(np.corrcoef(left,right)[0,1]) if len(pairs)>=3 and left.std()>0 and right.std()>0 else None,
                        median_gap_days=float(np.median([times[b]-times[a] for a,b in pairs])) if pairs else None))
                result.append(dict(lookback_days=days,runs=len(selected),field=field,
                    linear_trend_removed=detrend,lags=lags))
    return dict(cases=result,interpretation='Neighboring performance association can reflect genuine fitness, fatigue, seasons or model error. It does not identify an AR noise parameter.')


def stress_checks(draws=512, seed=20261015):
    """Fresh seed and independent synthetic cadence; no tuning to these results.

    Independent errors below/at/above their declared SD and persistent AR(1)
    extra variation challenge the measurement floor and excess-scatter logic.
    """
    end = datetime(2026,10,14,tzinfo=timezone.utc)
    cov = np.array([[.0001]])
    cells = []
    for every in (2,4):
        template = [dict(activity_id=i, start_time=(end-timedelta(days=every*(60-i))).isoformat(),
            pace_min_mile=10., uncertainty_95_min_mile=1.96*(.15 if i%3 else .35),
            trend_weight=.7 if i%5==0 else 1., included_in_trend=True,
            shared_log_speed_gradient=[.4 if i%4 else 1.2]) for i in range(61)]
        truth = evaluate_candidates(template,end,cov)
        for scale in (.5,1.,2.):
            for persistent in (False,True):
                rng = np.random.default_rng(seed+every*100+int(persistent))
                samples = {k:[] for k in truth}
                for _ in range(draws):
                    delta = rng.normal(0.,math.sqrt(cov[0,0]))
                    spell = rng.normal(0.,.035) if persistent else 0.
                    rows = []
                    for row in template:
                        if persistent:spell=.7*spell+rng.normal(0.,.035*math.sqrt(1-.7**2))
                        sigma = max(10/60,row['uncertainty_95_min_mile']/1.96)
                        g = row['shared_log_speed_gradient'][0]
                        independent = math.sqrt(max(0., sigma*sigma-100*g*g*cov[0,0]))
                        noise = rng.normal(0.,scale*independent/10)+spell-g*delta
                        rows.append(dict(row,pace_min_mile=10*math.exp(noise)))
                    result = evaluate_candidates(rows,end,cov)
                    for k,v in result.items():samples[k].append(v)
                cells.append(dict(run_spacing_days=every,measurement_error_scale=scale,
                    extra_ar1_variation=persistent,results={k:dict(
                        full=coverage(v,truth[k]['estimate'],('measurement','day','shared')),
                        **{name:coverage(v,truth[k]['estimate'],(name,'shared')) for name in VARIANTS})
                        for k,v in samples.items()}))
    return dict(draws=draws,seed=seed,cells=cells,
        note='Underestimated per-run error and extra persistent variation are intentional misspecification stresses, not assumed athlete physiology.')


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--draws',type=int,default=512)
    p.add_argument('--as-of',type=datetime.fromisoformat,default=datetime.now(timezone.utc))
    args = p.parse_args()
    if args.output.exists():p.error('Choose a new output path')
    if args.draws<2 or args.as_of.tzinfo is None:p.error('At least two draws and timezone-aware date required')
    root=Path(__file__).resolve().parents[1];config=load_config(root/'config.yaml')
    result=audit(resolve_project_path(root,config['paths']['database']),config,args.as_of,args.draws,
        evaluator=evaluate_candidates,extra_variants=VARIANTS,source_diagnostic=persistence_inventory)
    result['version']='aerobic-variance-overlap-audit-v1'
    result['candidate_definitions']={
        'variance_envelope':'Keep the larger of claimed measurement mean variance and observed scatter/cluster mean variance.',
        'excess_moments':'Add only positive observed scatter/cluster excess beyond expected independent/shared error, to the measurement floor.'}
    result['stress']=stress_checks(args.draws)
    result['assumptions'].extend([
        'Both candidate formulas were specified before their coverage results; no coefficient is fitted to coverage.',
        'Excess-moment expectations condition on Huber weights that actually depend on observations; robustness/small-sample effects can remain.',
        'Both candidates preserve full shared calibration error, the independent measurement floor, and the shipped point estimate.',
        'Neither candidate is deployed; separate synthetic stress uses a fresh seed and cadence.'])
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(cells=len(result['simulation']['cells']),stress_cells=len(result['stress']['cells']),
        production_model_changed=False),indent=2))


if __name__=='__main__':main()
