"""Chronological, training-conditioned pace-change experiment (offline only).

At each historical origin, predict later comparable running performance using
models learned ONLY from transitions whose outcomes had already finished.
Completed future training is supplied as a conditional scenario, not forecast.
Current retrospective HR/weather/terrain calibration is frozen throughout.
No live pace estimate, planner, training policy or database is modified.
"""
from argparse import ArgumentParser
from collections import Counter
from datetime import date, datetime, timedelta
import json
import math
from pathlib import Path
import sqlite3
from urllib.request import urlopen
from zoneinfo import ZoneInfo

import numpy as np

from run_analysis.aerobic_candidate import measurement_inputs
from run_analysis.aerobic_forecast_core import training_between, outcome_training, make_examples, ridge_fit, predict_change
from run_analysis.aerobic_performance import LocalPaceTrend
from run_analysis.config import load_config, resolve_project_path
from run_analysis.pace_temperature import comparison_covariance
from run_analysis.privacy import private_file
from run_analysis.progress import prepare_progress_data
from scripts.audit_aerobic_pace_forecast import damped_days, fit_trend








MODELS = {
    'calendar': ('elapsed',),
    'state': ('elapsed','start_level','start_trend'),
    'state_volume': ('elapsed','start_level','start_trend','delivered_volume'),
    'state_volume_change': ('elapsed','start_level','start_trend','volume_change'),
    'state_minutes': ('elapsed','start_level','start_trend','delivered_minutes'),
    'state_prior': ('elapsed','start_level','start_trend','prior_volume','prior_quality'),
    'plus_volume': ('elapsed','start_level','start_trend','prior_volume','prior_quality','delivered_volume'),
    'plus_quality': ('elapsed','start_level','start_trend','prior_volume','prior_quality','delivered_quality'),
    'plus_volume_quality': ('elapsed','start_level','start_trend','prior_volume','prior_quality','delivered_volume','delivered_quality'),
    'plus_frequency': ('elapsed','start_level','start_trend','prior_volume','prior_quality','delivered_frequency'),
    'plus_minutes': ('elapsed','start_level','start_trend','prior_volume','prior_quality','delivered_minutes'),
    'state_prior_fatigue': ('elapsed','start_level','start_trend','prior_volume','prior_quality','end_fatigue'),
    'plus_volume_fatigue': ('elapsed','start_level','start_trend','prior_volume','prior_quality','end_fatigue','delivered_volume'),
}






def evaluate(examples, *, penalty=4.):
    predictions = []
    for row in examples:
        # Strict embargo: a training transition's complete outcome block must
        # finish before the current origin. No future response enters fitting.
        training = [r for r in examples if r['horizon']==row['horizon']
            and datetime.fromisoformat(r['outcome_end'])<datetime.fromisoformat(row['origin'])]
        known_fatigue = row['end_fatigue'] is not None
        for name,keys in MODELS.items():
            chosen=training
            if 'end_fatigue' in keys:
                if not known_fatigue:
                    continue
                chosen=[r for r in training if r['end_fatigue'] is not None]
            fit = ridge_fit(chosen,keys,penalty=penalty)
            if fit is None:
                continue
            change = predict_change(fit,row)
            coefficients = {k:float(fit['beta'][j+1]/fit['scale'][j]) for j,k in enumerate(keys)}
            predictions.append(dict(model=name,horizon=row['horizon'],origin=row['origin'],
                origin_id=row['origin_id'],target_id=row['target_id'],outcome_ids=row['outcome_ids'],
                outcome_end=row['outcome_end'],
                predicted=row['start_pace']*math.exp(change),actual=row['actual'],
                persistence=row['start_pace'],actual_change_seconds=60*(row['actual']-row['start_pace']),
                latest_training_outcome=fit['latest_outcome'],training_examples=fit['training_examples'],
                training_targets=fit['training_targets'],coefficients=coefficients,
                delivered_volume=row['delivered_volume'],prior_volume=row['prior_volume']))
    return predictions


def metrics(rows):
    if not rows:
        return dict(pairs=0,unique_outcome_runs=0)
    # Downweight blocks repeatedly reusing the same physical outcome runs.
    # This does not make overlapping blocks independent observations.
    counts=Counter(i for r in rows for i in r['outcome_ids'])
    w=np.array([sum(1/counts[i] for i in r['outcome_ids'])/len(r['outcome_ids']) for r in rows]);w/=w.sum()
    errors=np.array([(r['predicted']-r['actual'])*60 for r in rows])
    return dict(pairs=len(rows),unique_outcome_runs=len(counts),
        origin_start=min(r['origin'] for r in rows),origin_end=max(r['origin'] for r in rows),
        mae_seconds_per_mile=float(w@np.abs(errors)),bias_seconds_per_mile=float(w@errors),
        note='Overlapping outcome blocks/origins remain dependent; no significance claim')


def paired_summary(predictions, candidate, baseline):
    result={}
    for horizon in (14,28,56,84):
        base={(p['origin'],p['target_id']):p for p in predictions if p['model']==baseline and p['horizon']==horizon}
        pairs=[p for p in predictions if p['model']==candidate and p['horizon']==horizon and (p['origin'],p['target_id']) in base]
        a=metrics(pairs);b=metrics([base[p['origin'],p['target_id']] for p in pairs])
        result[str(horizon)]=dict(candidate=a,baseline=b,
            mae_improvement_seconds_per_mile=(b['mae_seconds_per_mile']-a['mae_seconds_per_mile']) if pairs else None,
            coefficient_negative_fraction=(sum(p['coefficients'].get('delivered_volume',0)<0 for p in pairs)/len(pairs)) if pairs and 'volume' in candidate else None)
    return result


def disjoint_summary(predictions,candidate,baseline):
    """Sensitivity with no repeated physical outcome run, not a new fit."""
    result={}
    for horizon in (14,28,56,84):
        base={(p['origin'],p['target_id']):p for p in predictions if p['model']==baseline and p['horizon']==horizon}
        ordered=sorted((p for p in predictions if p['model']==candidate and p['horizon']==horizon
            and (p['origin'],p['target_id']) in base),key=lambda p:(p['outcome_end'],p['origin']))
        chosen=[];used=set()
        for p in ordered:
            if used.isdisjoint(p['outcome_ids']):
                chosen.append(p);used.update(p['outcome_ids'])
        a=metrics(chosen);b=metrics([base[p['origin'],p['target_id']] for p in chosen])
        result[str(horizon)]=dict(candidate=a,baseline=b,
            mae_improvement_seconds_per_mile=b['mae_seconds_per_mile']-a['mae_seconds_per_mile'] if chosen else None,
            note='Deterministic earliest-finish blocks; outcome runs disjoint, training histories still overlap')
    return result


def perfect_adherence_scenario(examples,state,training,program,*,as_of,zone,hard_end,plan_end,origin=None):
    """Short-term conditional scenario, not a reroll or a May projection.

    Program entries are saved day/dose pairs. Beyond hard_end they are only
    tentative continuation and are explicitly labeled, not claimed issued.
    """
    today=as_of.astimezone(zone).date()
    origin=origin or as_of
    if origin>as_of:
        raise ValueError('Forecast origin cannot be in the future')
    prior=training_between(training,as_of-timedelta(days=28),as_of)
    current=state['current']['pace_min_mile']
    slope=state['progress_trend'].get('log_pace_change_per_day')
    if slope is None:
        return dict(available=False,reason='An established starting trend is needed')
    # Match the backtest's observation-conditioned exposure instead of using
    # a full calendar month for a feature learned strictly before run starts.
    # No future sensor precision is available: equal run weights are a stated
    # scenario assumption, not invented HR observations.
    planned=[dict(id=-i-1,time=datetime.combine(r['day'],datetime.min.time(),zone)+timedelta(hours=12),
        miles=r['miles'],minutes=0.,quality=r.get('role')=='quality',long=r.get('role')=='long',
        role=r.get('role','easy'),day=r['day']) for i,r in enumerate(program) if r['day']>=today]
    physical=training+planned
    scenarios=[]
    for horizon in (14,28):
        target=origin.astimezone(zone).date()+timedelta(days=horizon)
        if target+timedelta(days=7)>=plan_end:
            continue
        outcomes=[r for r in planned if r['role'] in {'easy','long'}
            and abs((r['time']-origin).total_seconds()/86400-horizon)<=7]
        if len(outcomes)<2:
            continue
        outcome_dates=[r['time'] for r in outcomes]
        delivered=outcome_training(physical,origin,outcome_dates,np.ones(len(outcomes)),{},zone)
        volume=delivered['volume']
        elapsed=sum((at-origin).total_seconds()/86400 for at in outcome_dates)/len(outcomes)
        row=dict(elapsed=elapsed/28,start_level=math.log(current),
            start_trend=slope*damped_days(elapsed,56),delivered_volume=volume)
        issued=tentative=0.
        by_id={r['id']:r for r in planned}
        for exposure in delivered['by_outcome']:
            issued+=exposure['weight']*sum(by_id[i]['miles'] for i in exposure['run_ids'] if i in by_id and by_id[i]['day']<hard_end)
            tentative+=exposure['weight']*sum(by_id[i]['miles'] for i in exposure['run_ids'] if i in by_id and by_id[i]['day']>=hard_end)
        completed=[e for e in examples if e['horizon']==horizon
            and datetime.fromisoformat(e['outcome_end'])<as_of]
        values=[]
        for penalty in (1.,4.,16.):
            fit=ridge_fit(completed,MODELS['state_volume'],penalty=penalty)
            if fit:
                values.append(dict(penalty=penalty,pace_min_mile=current*math.exp(predict_change(fit,row))))
        if not values:
            continue
        inputs=[e['delivered_volume'] for e in completed]
        scenarios.append(dict(horizon_days=horizon,date=target.isoformat(),training_miles=delivered['total_miles'],
            training_miles_basis='Mean cumulative training before each scenario outcome, not the total of a calendar month',
            outcome_dates=[r['day'].isoformat() for r in outcomes],outcome_weights='equal, future precision unknown',
            issued_miles=issued,tentative_continuation_miles=tentative,volume_per_7=volume,
            primary_pace_min_mile=next(v['pace_min_mile'] for v in values if v['penalty']==4),
            fitting_sensitivity=values,training_input_min=min(inputs),training_input_max=max(inputs),
            outside_training_input_range=not min(inputs)<=volume<=max(inputs),
            note='Conditional on completing saved day/dose program; no future run observation or adherence is inferred'))
    return dict(available=bool(scenarios),current_pace_min_mile=current,origin=origin.isoformat(),
        prior_28d_volume_per_7=prior['volume'],prior_28d_run_ids=prior['run_ids'],
        scenarios=scenarios,forecast_model='state_volume',
        limitations=['Fitting sensitivity is not a prediction interval',
            'Saved continuation after hard_end is tentative, not issued',
            'No seven-month or 17-mile durability extrapolation'])


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--url',default='http://127.0.0.1:8000')
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    with urlopen(args.url+'/api/progress-wip?window_days=365&include_exploratory=false&include_legacy_aerobic=false',timeout=150) as response:
        data=json.load(response)
    state=data['aerobic_performance'];report=data['pace_comparison'];hidden=set(state.get('hidden_activity_ids',[]))
    rows=sorted([r for r in report['runs'] if r.get('included_in_trend') and r.get('trend_weight',0)>0
        and r['activity_id'] not in hidden and not r.get('manually_excluded') and r.get('workout_type') not in {'hike','bike'}],key=lambda r:(r['start_time'],r['activity_id']))
    cov=comparison_covariance(report['calibration']);m,g=measurement_inputs(rows,cov)
    variance=m+np.einsum('ij,jk,ik->i',g,cov,g)
    zone=ZoneInfo(data['timezone'])
    fatigue={datetime.fromisoformat(r['as_of']).astimezone(zone).date():r['short']['value'] for r in data['training_load']['daily']}
    config=load_config(root/'config.yaml');database=resolve_project_path(root,config['paths']['database'])
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True) as c:
        c.row_factory=sqlite3.Row;prepared=prepare_progress_data(c)
        metadata={r['id']:dict(r) for r in c.execute('SELECT id,start_time_utc,total_elapsed_time_s FROM activities')}
        for row in rows:
            meta=metadata[row['activity_id']]
            row['available_at']=(datetime.fromisoformat(meta['start_time_utc'])+timedelta(seconds=meta['total_elapsed_time_s'] or 0)).isoformat()
        training=[]
        for s in prepared.sessions:
            role=prepared.details[s.activity_id]['workout'].value
            if role in {'hike','bike'}:
                continue
            meta=metadata[s.activity_id]
            training.append(dict(id=s.activity_id,time=s.start_time,
                available_at=s.start_time+timedelta(seconds=meta['total_elapsed_time_s'] or 0),miles=s.distance_miles,
                minutes=s.moving_minutes,quality=role in {'intervals','tempo_threshold','race'},long=role=='long'))
        pointer=c.execute("SELECT value_json FROM app_state WHERE key='adaptive_runtime'").fetchone()
        assert c.total_changes==0
    examples=make_examples(rows,variance,training,fatigue,zone)
    comparisons={
        'state_vs_calendar':('state','calendar'),
        'volume_vs_state':('state_volume','state'),
        'volume_change_vs_state':('state_volume_change','state'),
        'minutes_vs_state':('state_minutes','state'),
        'volume_vs_state_prior':('plus_volume','state_prior'),
        'quality_vs_state_prior':('plus_quality','state_prior'),
        'volume_quality_vs_state_prior':('plus_volume_quality','state_prior'),
        'frequency_vs_state_prior':('plus_frequency','state_prior'),
        'minutes_vs_state_prior':('plus_minutes','state_prior'),
        'volume_controlling_end_fatigue':('plus_volume_fatigue','state_prior_fatigue'),
    }
    sensitivity={};primary=None
    for penalty in (1.,4.,16.):
        predictions=evaluate(examples,penalty=penalty)
        if penalty==4:
            primary=predictions
        sensitivity[str(penalty)]={k:paired_summary(predictions,*models) for k,models in comparisons.items()}
    baselines=[]
    by_key={(r['origin'],r['target_id'],r['horizon']):r for r in examples}
    for p in primary:
        if p['model']=='state':
            row=by_key[p['origin'],p['target_id'],p['horizon']]
            for name,change in [('persistence',0.),('linear',row['linear_trend']),('damp56',row['start_trend'])]:
                baselines.append({**p,'model':name,'predicted':p['persistence']*math.exp(change)})
    primary+=baselines
    baseline_comparison={name:paired_summary(primary,'state_volume',name) for name in ['persistence','linear','damp56']}
    as_of=datetime.fromisoformat(data['as_of'])
    current_training=training_between(training,as_of-timedelta(days=28),as_of)
    latest=[dict(id=r['id'],date=r['time'].astimezone(zone).date().isoformat(),miles=r['miles'],
        role='long' if r['long'] else 'quality' if r['quality'] else 'easy/other')
        for r in training if r['id'] in set(current_training['run_ids'])]
    scenario=dict(available=False,reason='No verified saved plan')
    if pointer:
        runtime=Path(json.loads(pointer[0])['root'])
        if not runtime.is_absolute():runtime=root/runtime
        with sqlite3.connect((runtime/'sandbox.sqlite').resolve().as_uri()+'?mode=ro',uri=True) as c:
            committed=c.execute('SELECT event,snapshot FROM sandbox_decisions ORDER BY sequence DESC LIMIT 1').fetchone()
        if committed:
            event=json.loads(committed[0])
            from run_analysis.adaptive.decision_snapshot import restore
            restore(committed[1],expected_fingerprint=event['snapshot_fingerprint'])
            if event['as_of']==data.get('goal_journey',{}).get('as_of'):
                catalog={r['key']:r for r in event['stage_runtime']['catalog']}
                program=[dict(day=date.fromisoformat(day),miles=catalog[key]['miles'],role=catalog[key]['role']) for day,key in event['program']]
                scenario=perfect_adherence_scenario(examples,state,training,program,as_of=as_of,zone=zone,
                    hard_end=date.fromisoformat(event['hard_end']),plan_end=date.fromisoformat(event['end']),
                    origin=datetime.fromisoformat(rows[-1]['available_at']))
    result=dict(version='training-conditioned-pace-audit-v2-per-outcome-training',as_of=data['as_of'],
        reference_hr_bpm=state['target_hr_bpm'],run_count=len(rows),training_activities=len(training),
        example_counts={str(h):sum(r['horizon']==h for r in examples) for h in (14,28,56,84)},
        examples=examples,predictions=primary,primary_penalty=4.,sensitivity=sensitivity,
        volume_vs_unlearned_baselines=baseline_comparison,
        disjoint_outcome_check=disjoint_summary(primary,'state_volume','state'),
        current_training=dict(volume_per_7=current_training['volume'],total_miles=current_training['total_miles'],
            runs=latest,basis='All completed running in the 28 days before current as_of, including long and quality'),
        perfect_adherence_scenario=scenario,
        supersedes='training-conditioned-pace-audit-v1, whose exposure stopped before the first outcome',
        protocol=dict(response='Later +/-7-day multi-run comparable pace minus origin-only fitted pace',
            exposures='For EACH outcome run: all completed running after origin and before that individual run; same weights as log-pace outcomes',
            earlier_outcomes='Earlier outcome runs count as completed training for later outcomes, never for their own pace',
            aggregate='delivered_total_miles is weighted cumulative mileage; delivered_union_miles is distinct physical mileage; neither is a calendar-week maximum',
            baseline='Elapsed time, starting pace/trend and preceding 28-day volume/quality',
            embargo='Only transitions whose complete outcome block finished before prediction origin',
            selection='Fixed candidate feature sets and ridge penalties; none selected using test error',
            meaning='Given this actual delivered training, not a forecast of future adherence'),
        limitations=['Frozen retrospective pace calibration; not prospective HR/weather/elevation refitting',
            'One runner, roughly seven months; overlapping transitions are not independent experiments',
            'No causal claim: training decisions, symptoms, season and fitness affect each other',
            'Outcome is comparable running performance, not measured latent fitness',
            'Quality is completed session frequency, not a validated adaptation-equivalent dose',
            'Recorded gaps can mean rest or missing uploads; completeness cannot be proven',
            'No May/seven-month interval calibration or 17-mile durability forecast'],
        live_model_changed=False,planner_changed=False,database_changed=False)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');private_file(args.output)
    print('examples',result['example_counts'])
    for name,value in sensitivity['4.0'].items():
        print(name,{h:dict(n=v['candidate']['pairs'],
            base=round(v['baseline'].get('mae_seconds_per_mile',0),2),
            candidate=round(v['candidate'].get('mae_seconds_per_mile',0),2),
            improvement=round(v['mae_improvement_seconds_per_mile'],2) if v['mae_improvement_seconds_per_mile'] is not None else None)
            for h,v in value.items()})
    print('Saved',args.output)


if __name__=='__main__':
    main()
