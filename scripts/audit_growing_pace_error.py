"""Test nonzero baseline plus growing forecast error without changing live code.

The central pace forecasts and actual-future-run targets are frozen. Match
origins across two/four-week errors before estimating a baseline variance and
growth term. Chronological tests embargo every calibration outcome at origin;
long-horizon descriptive transfers are reported separately, not as independent
validation. No May result selects a formula or coefficient.
"""
from argparse import ArgumentParser
import json
import math
from pathlib import Path

import numpy as np

from run_analysis.pace_forecast_error import matched_error_growth,weights,weighted_quantile
from run_analysis.privacy import private_file
from scripts.audit_contextual_training_response import sequential_blocks
from scripts.audit_long_horizon_error_growth import coverage_summary

METHODS=('constant','existing_power','baseline_linear_variance','baseline_quadratic_variance')


def matched_moments(rows,minimum_origins=8):
    by={h:{r['origin']:r for r in rows if r['horizon']==h} for h in (14,28)}
    origins=sorted(set(by[14])&set(by[28]))
    if len(origins)<minimum_origins:return None
    # Both horizons get exactly the same per-origin weights. Otherwise a
    # different mixture of overlapping outcomes can masquerade as error growth.
    pairs=[by[h][o] for o in origins for h in (14,28)]
    w=weights(pairs).reshape(-1,2).sum(axis=1)
    return dict(rows=pairs,origin_weights=w,
        horizons=[[by[h][o] for o in origins] for h in (14,28)],origins=len(origins))


def fit_shape(rows,method,minimum_origins=8):
    if method=='constant':return dict(method=method,baseline=1.,growth=0.,power=1.)
    if method=='existing_power':
        fitted=matched_error_growth(rows)
        return None if fitted is None else dict(method=method,baseline=0.,growth=1.,power=2*fitted['exponent'],
            latest_shape_outcome=fitted['latest_outcome'])
    matched=matched_moments(rows,minimum_origins)
    if matched is None:return None
    power={'baseline_linear_variance':1.,'baseline_quadratic_variance':2.}[method]
    w=matched['origin_weights']
    x=np.array([w@np.array([r['elapsed']**power for r in group]) for group in matched['horizons']])
    y=np.array([w@np.square([r['log_error'] for r in group]) for group in matched['horizons']])
    if x[1]<=x[0]:return None
    growth=float((y[1]-y[0])/(x[1]-x[0]));baseline=float(y[0]-growth*x[0])
    # Do not invent an intercept or force rising uncertainty when the matched
    # observations fail to identify both nonnegative components.
    if min(baseline,growth)<0 or not np.isfinite([baseline,growth]).all():return None
    return dict(method=method,baseline=baseline,growth=growth,power=power,
        origins=matched['origins'],latest_shape_outcome=max(r['outcome_end'] for r in matched['rows']))


def scale(shape,days):
    if days<0 or not math.isfinite(days):raise ValueError('Nonnegative finite horizon required')
    return math.sqrt(shape['baseline']+shape['growth']*(days/28)**shape['power'])


def calibrate_range(shape,rows,minimum_origins=8):
    if shape is None or len({r['origin'] for r in rows})<minimum_origins:return None
    s=np.array([scale(shape,r['elapsed']*28) for r in rows])
    if np.any(s<=0):return None
    factor=float(weighted_quantile(np.abs([r['log_error'] for r in rows])/s,weights(rows),[.8])[0])
    return dict(shape=shape,factor=factor,origins=len({r['origin'] for r in rows}),
        latest_calibration=max(r['outcome_end'] for r in rows),
        calibration_outcome_ids=sorted({i for r in rows for i in r['outcome_ids']}))


def radius(model,days):return model['factor']*scale(model['shape'],days)


def score_target(target,model):
    r=radius(model,target['elapsed']*28)
    return {**target,'method':model['shape']['method'],'radius':r,
        'covered':abs(target['log_error'])<=r,
        'width_seconds':120*target['predicted']*math.sinh(r),
        'latest_calibration':model['latest_calibration'],
        'latest_shape_outcome':model['shape'].get('latest_shape_outcome'),
        'calibration_outcome_ids':model['calibration_outcome_ids'],
        'starting_radius':radius(model,0),
        'shape':model['shape']}


def summarize(rows):
    result={}
    for label,horizons in (('all',(14,28,56,84,112)),('short',(14,28)),('long',(56,84,112))):
        chosen=[r for r in rows if r['horizon'] in horizons]
        result[label]={}
        for method in METHODS:
            subset=[r for r in chosen if r['method']==method]
            result[label][method]=dict(all=coverage_summary(subset),
                sequential=coverage_summary(sequential_blocks(subset)))
    return result


def chronological(control,candidate,*,consistent=False):
    predictions=[];rejections={m:0 for m in METHODS}
    pairwise={m:[] for m in METHODS if m!='existing_power'}
    for target in candidate:
        prior=[r for r in control if r['horizon'] in (14,28) and r['outcome_end']<target['origin']]
        residuals=[r for r in candidate if r['horizon'] in (14,28) and r['outcome_end']<target['origin']
            and (not consistent or r['regime']=='consistent')]
        models={m:calibrate_range(fit_shape(prior,m),residuals) for m in METHODS}
        for m,model in models.items():
            if model is None:rejections[m]+=1
        for method,items in pairwise.items():
            if models[method] is not None and models['existing_power'] is not None:
                items.extend(score_target(target,models[m]) for m in (method,'existing_power'))
        # Identical test cases for all formulas; a failed fit does not get to
        # silently drop its difficult cases from a headline comparison.
        if any(m is None for m in models.values()):continue
        predictions.extend(score_target(target,model) for model in models.values())
    return dict(predictions=predictions,summary=summarize(predictions),unavailable_by_method=rejections,
        pairwise={m:dict(predictions=r,summary=summarize(r)) for m,r in pairwise.items()})


def horizon_holdout(control,candidate,*,consistent=False):
    result={}
    for horizon in (56,84,112):
        by={h:{r['origin']:r for r in control if r['horizon']==h} for h in (14,28)}
        targets=[r for r in candidate if r['horizon']==horizon]
        common=set(by[14])&set(by[28])&{r['origin'] for r in targets}
        shape_rows=[by[h][o] for o in sorted(common) for h in (14,28)]
        residuals=[r for r in candidate if r['horizon'] in (14,28) and r['origin'] in common
            and (not consistent or r['regime']=='consistent')]
        models={m:calibrate_range(fit_shape(shape_rows,m),residuals) for m in METHODS}
        scored=[score_target(r,model) for r in targets if r['origin'] in common for model in models.values() if model]
        result[str(horizon)]=dict(origins=len(common),
            methods={m:coverage_summary([r for r in scored if r['method']==m]) for m in METHODS})
    return result


def audit(source):
    control=[r for r in source['predictions'] if r['bandwidth']==28 and r['mode']=='joint']
    candidate=[r for r in source['predictions'] if r['bandwidth']==42 and r['mode']=='joint']
    output=dict(version='growing-pace-error-audit-v1',as_of=source['as_of'],
        target='Unchanged corrected actual future-run means, not each candidate smoother',
        formulas={'baseline_linear_variance':'variance = baseline + growth × elapsed weeks',
            'baseline_quadratic_variance':'variance = baseline + growth × elapsed weeks squared',
            'existing_power':'The previously learned multi-week power; zero-origin behavior retained as audit control only',
            'constant':'No horizon growth; not a proposed growing band'},
        results={},live_forecast_changed=False,planner_changed=False,database_changed=False)
    for consistent in (False,True):
        scope='consistent_magnitude' if consistent else 'all_recorded_contexts'
        residuals=[r for r in candidate if r['horizon'] in (14,28) and (not consistent or r['regime']=='consistent')]
        current={}
        for method in METHODS:
            model=calibrate_range(fit_shape([r for r in control if r['horizon'] in (14,28)],method),residuals)
            if model is None:current[method]=dict(available=False);continue
            pace=source['final_sensitivity']['42']['may_joint'];r=radius(model,source['target_days'])
            current[method]=dict(available=True,model=model,
                radii_log={str(d):radius(model,d) for d in (0,7,14,28,56,84,112,source['target_days'])},
                may_center=pace,may_low=pace*math.exp(-r),may_high=pace*math.exp(r))
        output['results'][scope]=dict(current=current,chronological=chronological(control,candidate,consistent=consistent),
            horizon_holdouts=horizon_holdout(control,candidate,consistent=consistent))
    output['limitations']=['No independent 211-day forecast outcomes',
        'Short calibration is sparse; histories and outcome windows overlap',
        'Mixed-context growth is transferred to consistent-training residual magnitudes',
        'Long observed outcomes are health affected; not uninterrupted healthy training validation',
        'Fitted intercept is extrapolated forecast residual variance, not direct measurement of latent present fitness uncertainty',
        'Baseline/growth independence and linear/quadratic variance are tested assumptions, not physiological facts',
        'Horizon holdout uses future information relative to origins; it is not chronological validation',
        'No coefficient or method chosen from May pace or curve appearance']
    return output


def main():
    p=ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=Path('output/future-run-smoothing-20261020.json'))
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();result=audit(json.loads(args.input.read_text()))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');private_file(args.output)
    print(json.dumps({scope:dict(current={m:{k:v for k,v in r.items() if k!='model'} for m,r in v['current'].items()},
        paired_tests={m:{h:{method:r['all'] for method,r in stats.items()
            if method in (m,'existing_power')} for h,stats in values['summary'].items()}
            for m,values in v['chronological']['pairwise'].items()},
        unavailable=v['chronological']['unavailable_by_method'])
        for scope,v in result['results'].items()},indent=2))


if __name__=='__main__':main()
