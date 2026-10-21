"""Read-only empirical dose-response test, without prescribed adaptation delays.

Frozen retrospective HR/weather/grade corrections; strict outcome embargo.
Candidate shapes are tested, not physiology or causal identification. Actual
future training is supplied only as a conditional input during hindcasts.
No live chart, training policy, settings, journal or database is changed.
"""
from argparse import ArgumentParser
from collections import Counter
from datetime import datetime,timedelta
import json
import math
from pathlib import Path
import sqlite3
from urllib.request import urlopen
from zoneinfo import ZoneInfo

import numpy as np

from run_analysis.aerobic_candidate import measurement_inputs
from run_analysis.aerobic_forecast_core import make_examples
from run_analysis.aerobic_performance import LocalPaceTrend
from run_analysis.config import load_config,resolve_project_path
from run_analysis.pace_temperature import comparison_covariance
from run_analysis.privacy import private_file
from run_analysis.progress import prepare_progress_data
from run_analysis.training_response import HORIZONS,BASE_MODELS,MODELS,HISTORY,features,benchmark_examples,fit_response







def observed_blocks(examples,horizon=28):
    """Descriptive successive nonoverlapping response periods, not a forecast."""
    end='';blocks=[]
    for e in sorted((r for r in examples if r['horizon']==horizon),key=lambda r:r['origin']):
        if e['origin']<end:continue
        blocks.append(dict(origin=e['origin'],end=e['outcome_end'],
            matched_exposure_miles=e['delivered_total_miles'],distinct_pre_outcome_miles=e['delivered_union_miles'],
            start_pace=e['start_pace'],end_pace=e['actual'],
            faster_seconds=60*(e['start_pace']-e['actual'])))
        end=e['outcome_end']
    return blocks




def predict(fit,row):
    return sum(b*row[k] for k,b in fit['coefficients'].items())


def score(rows):
    if not rows:return dict(pairs=0,outcome_runs=0)
    counts=Counter(i for r in rows for i in r['outcome_ids'])
    w=np.array([sum(1/counts[i] for i in r['outcome_ids'])/len(r['outcome_ids']) for r in rows]);w/=w.sum()
    errors=np.array([(r['predicted']-r['actual'])*60 for r in rows])
    return dict(pairs=len(rows),outcome_runs=len(counts),mae_seconds=float(w@abs(errors)),
        bias_seconds=float(w@errors),log_mae=float(w@np.abs([r['log_error'] for r in rows])))


def choose(prior):
    """Choose using completed earlier hindcasts, balancing horizons equally."""
    losses={}
    eligible=[model for model in MODELS if len({r['origin'] for r in prior if r['model']==model})>=8]
    if not eligible:return None
    # A recent-history candidate cannot win merely by skipping difficult early
    # origins. All eligible models are scored on exactly the same cases.
    common=set.intersection(*[{(r['origin'],r['target_id']) for r in prior if r['model']==model} for model in eligible])
    if len({o for o,t in common})<8:return None
    for model in eligible:
        chosen=[r for r in prior if r['model']==model and (r['origin'],r['target_id']) in common]
        by_h=[score([r for r in chosen if r['horizon']==h]) for h in HORIZONS]
        losses[model]=float(np.mean([v['log_mae'] for v in by_h if v['pairs']]))
    return min(losses,key=losses.get) if losses else None


def disjoint(rows):
    """Deterministic outcome-disjoint subset, never chosen by error magnitude."""
    used=set();chosen=[]
    for r in sorted(rows,key=lambda v:(v['outcome_end'],v['origin'])):
        if used.isdisjoint(r['outcome_ids']):
            chosen.append(r);used.update(r['outcome_ids'])
    return chosen


def residual_range(predicted,prior):
    """Empirical 10th–90th residual percentiles, NOT a certified 80% interval."""
    selected=disjoint(prior)
    if len(selected)<3:return dict(available=False,calibration_blocks=len(selected))
    q=np.quantile([-r['log_error'] for r in selected],[.1,.9])
    return dict(available=True,low_pace=float(predicted*math.exp(q[0])),
        high_pace=float(predicted*math.exp(q[1])),calibration_blocks=len(selected),
        calibration_outcome_runs=len({i for r in selected for i in r['outcome_ids']}),
        latest_calibration_outcome=max(r['outcome_end'] for r in selected))


def evaluate(examples):
    predictions=[];selected=[]
    for origin in sorted({r['origin'] for r in examples}):
        train=[r for r in examples if r['outcome_end']<origin]
        prior=[r for r in predictions if r['outcome_end']<origin]
        winner=choose(prior)
        fits={model:fit_response(train,model,as_of=datetime.fromisoformat(origin)) for model in MODELS}
        for e in (r for r in examples if r['origin']==origin):
            for model,fit in fits.items():
                if fit is None:continue
                predicted=e['start_pace']*math.exp(predict(fit,e))
                p=dict(model=model,origin=origin,horizon=e['horizon'],target_id=e['target_id'],
                    outcome_ids=e['outcome_ids'],outcome_end=e['outcome_end'],actual=e['actual'],
                    predicted=predicted,persistence=e['start_pace'],log_error=math.log(predicted/e['actual']),
                    latest_training_outcome=fit['latest_outcome'],coefficients=fit['coefficients'])
                predictions.append(p)
                if model==winner:
                    # Only previously SELECTED, completed, same-horizon errors
                    # calibrate this prediction. No post-hoc best-model errors.
                    errors=[r for r in selected if r['horizon']==e['horizon'] and r['outcome_end']<origin]
                    selected.append({**p,'range':residual_range(predicted,errors),
                        'latest_selection_outcome':max(r['outcome_end'] for r in prior)})
    return predictions,selected


def summarize(predictions,selected):
    result={}
    for h in HORIZONS:
        rows=[r for r in predictions if r['horizon']==h]
        models={model:score([r for r in rows if r['model']==model]) for model in MODELS}
        base=[{**r,'predicted':r['persistence'],'log_error':math.log(r['persistence']/r['actual'])}
            for r in rows if r['model']=='miles']
        chosen=[r for r in selected if r['horizon']==h]
        ranges=[r for r in chosen if r['range']['available']]
        result[str(h)]=dict(models=models,persistence=score(base),selected=score(chosen),
            selected_persistence=score([{**r,'predicted':r['persistence'],
                'log_error':math.log(r['persistence']/r['actual'])} for r in chosen]),
            selected_disjoint=score(disjoint(chosen)),selected_models=dict(Counter(r['model'] for r in chosen)),
            calibrated_ranges=len(ranges),observed_range_coverage=(sum(r['range']['low_pace']<=r['actual']<=r['range']['high_pace'] for r in ranges)/len(ranges) if ranges else None))
    return result


def scenario(fit,goal,*,at,current_pace,prior_miles,minutes_per_mile,days,mode='goal'):
    """Direct response, never a rolling reversion or hand-chosen time taper."""
    total=0.;points=[]
    for d in range(days+1):
        if d:
            volume=goal['volume_per_7']*(1+goal['volume_growth_per_7'])**((d-.5)/7)
            if mode=='goal':volume=min(volume,goal['target_volume_per_7'])
            elif mode=='constant':volume=goal['volume_per_7']
            elif mode!='uncapped':raise ValueError('Known volume scenario required')
            total+=volume/7
        if d%7==0 or d==days:
            row=dict(elapsed=d/28,miles=total/100,minutes=total*minutes_per_mile/1000,
                log_miles=math.log1p(total/prior_miles))
            points.append(dict(days=d,date=(at+timedelta(days=d)).date().isoformat(),completed_miles=total,
                pace=current_pace*math.exp(predict(fit,row))))
    return points


def main():
    parser=ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    with urlopen('http://127.0.0.1:8000/api/progress-wip?window_days=365&include_exploratory=false&include_legacy_aerobic=false',timeout=150) as f:data=json.load(f)
    at=datetime.fromisoformat(data['as_of']);report=data['pace_comparison'];performance=data['aerobic_performance']
    hidden=set(performance['hidden_activity_ids'])
    rows=sorted([dict(r) for r in report['runs'] if r['included_in_trend'] and r['trend_weight']>0
        and r['activity_id'] not in hidden and not r.get('manually_excluded') and r.get('workout_type') not in {'hike','bike'}
        and datetime.fromisoformat(r['start_time'])<=at],key=lambda r:(r['start_time'],r['activity_id']))
    gaps=[i for i in range(1,len(rows)) if (datetime.fromisoformat(rows[i]['start_time'])-datetime.fromisoformat(rows[i-1]['start_time'])).total_seconds()>28*86400]
    if gaps:rows=rows[gaps[-1]:]
    config=load_config(root/'config.yaml');database=resolve_project_path(root,config['paths']['database'])
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True) as c:
        c.row_factory=sqlite3.Row;prepared=prepare_progress_data(c)
        metadata={r['id']:dict(r) for r in c.execute('SELECT id,start_time_utc,total_elapsed_time_s FROM activities')}
        for r in rows:
            meta=metadata[r['activity_id']]
            r['available_at']=(datetime.fromisoformat(meta['start_time_utc'])+timedelta(seconds=meta['total_elapsed_time_s'] or 0)).isoformat()
        rows=[r for r in rows if datetime.fromisoformat(r['available_at'])<=at]
        training=[]
        for s in prepared.sessions:
            role=prepared.details[s.activity_id]['workout'].value
            end=s.start_time+timedelta(seconds=metadata[s.activity_id]['total_elapsed_time_s'] or 0)
            if role in {'hike','bike'} or end>at:continue
            training.append(dict(id=s.activity_id,time=s.start_time,available_at=end,miles=s.distance_miles,
                minutes=s.moving_minutes,quality=role in {'intervals','tempo_threshold','race'},long=role=='long'))
        assert c.total_changes==0
    covariance=comparison_covariance(report['calibration']);m,g=measurement_inputs(rows,covariance)
    variance=m+np.einsum('ij,jk,ik->i',g,covariance,g)
    era_start=datetime.fromisoformat(rows[0]['start_time'])
    base=make_examples(rows,variance,training,{},ZoneInfo(data['timezone']),horizons=HORIZONS)
    raw_examples=features(base,training,era_start)
    raw_predictions,raw_selected=evaluate(raw_examples)
    examples=features(benchmark_examples(base,rows,m,g,covariance),training,era_start)
    predictions,selected=evaluate(examples)
    summary=summarize(predictions,selected)
    winner=choose(predictions);fits={model:fit_response(examples,model,as_of=at) for model in MODELS}
    goal=data['goal_journey'];end=datetime.fromisoformat(goal['projection']['target_date'])
    horizon=(end.date()-at.astimezone(ZoneInfo(data['timezone'])).date()).days
    prior_miles=sum(r['miles'] for r in training if r['time']>=era_start)
    recent=[r for r in training if at-timedelta(days=28)<=r['available_at']<=at]
    minutes_per_mile=sum(r['minutes'] for r in recent)/sum(r['miles'] for r in recent)
    scenarios={}
    for model,fit in fits.items():
        if fit:
            scenarios[model]={mode:scenario(fit,goal,at=at.astimezone(ZoneInfo(data['timezone'])),current_pace=performance['current']['pace_min_mile'],
                prior_miles=prior_miles,minutes_per_mile=minutes_per_mile,days=horizon,mode=mode)
                for mode in ('goal','constant','uncapped')}
    ranges={}
    if winner:
        for h in HORIZONS:
            point=next((r for r in scenarios[winner]['goal'] if r['days']==h),None)
            if point:ranges[str(h)]=residual_range(point['pace'],[r for r in selected if r['horizon']==h])
    result=dict(version='empirical-training-response-v1',as_of=at.isoformat(),current_pace=performance['current']['pace_min_mile'],
        era_start=era_start.isoformat(),aerobic_runs=len(rows),training_runs=len(training),
        latest_training_id=max(training,key=lambda r:r['available_at'])['id'],era_miles=prior_miles,
        horizons=HORIZONS,target='Causal Progress benchmark at each outcome; same model as current, no later runs',
        examples=examples,predictions=predictions,selected_predictions=selected,summary=summary,
        raw_run_control=dict(summary=summarize(raw_predictions,raw_selected),selected_model=choose(raw_predictions)),
        observed_response_blocks=observed_blocks(examples),
        observed_blocks_basis='End pace averages causal benchmark levels at outcome runs; exposure matches each outcome, not all miles through the printed end date',
        final_fits=fits,selected_model=winner,scenarios=scenarios,empirical_ranges=ranges,
        may_horizon_days=horizon,may_empirical_range_available=horizon in [h for h in HORIZONS if ranges.get(str(h),{}).get('available')],
        protocol=['No fixed adaptation delay, taper, sign restriction or ridge penalty',
            'Candidate linear and log-cumulative response forms are assumptions tested against chronological holdouts',
            'Full, 56-, 84-, and 112-day response histories compared on common forecast cases, never selected by May pace',
            'Train outcomes and model-selection outcomes must finish before each forecast origin',
            'Ranges use earlier selected-model residuals, same horizon, disjoint physical outcome runs',
            'Outcome benchmark prefixes share earlier evidence: disjoint outcome runs do not imply independent benchmark errors',
            'Overlap downweighting is not independence; retrospective pace corrections are frozen',
            'The current goal volume cap is a future training scenario, not an inferred physiological ceiling'],
        limitations=['Observational training/time confounding; not causal proof of adaptation',
            'Range percentiles from few blocks are descriptive, not certified coverage',
            'No empirical range is silently extended beyond tested horizons',
            'Future minutes estimated from recent completed minutes/mile only for minute-based candidates'],
        live_model_changed=False,planner_changed=False,database_changed=False)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');private_file(args.output)
    print(json.dumps(dict(summary={h:{k:v for k,v in s.items() if k!='models'} for h,s in summary.items()},
        selected_model=winner,final_fit=fits.get(winner),empirical_ranges=ranges,
        may={model:round(s['goal'][-1]['pace'],4) for model,s in scenarios.items()},
        may_empirical_range_available=result['may_empirical_range_available']),indent=2))


if __name__=='__main__':main()
