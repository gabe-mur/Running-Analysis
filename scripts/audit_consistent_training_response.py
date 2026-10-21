"""Read-only stratification of the frozen empirical training-response audit.

Regimes depend only on physical training and explicit health records, never on
pace, improvement sign, or prediction error. These diagnostic cutoffs are not
planner constraints, physiological boundaries, or proof that unmarked = healthy.
"""
from argparse import ArgumentParser
from datetime import datetime,timedelta
import json
import math
from pathlib import Path
import sqlite3

import numpy as np

from run_analysis.config import load_config,resolve_project_path
from run_analysis.privacy import private_file
from run_analysis.progress import prepare_progress_data
from run_analysis.training_response import REGIMES,regime,annotate
from scripts.audit_empirical_training_response import (
    HORIZONS,disjoint,fit_response,observed_blocks,predict,score,
)

MODELS=('calendar','miles','log_miles','miles_calendar')






def summaries(examples):
    result={}
    for h in HORIZONS:
        result[str(h)]={}
        for label in REGIMES:
            rows=[e for e in examples if e['horizon']==h and e['consistency']['regime']==label]
            blocks=disjoint(rows)
            changes=[60*(r['start_pace']-r['actual']) for r in rows]
            result[str(h)][label]=dict(examples=len(rows),origins=len({e['origin_id'] for e in rows}),
                outcome_runs=len({i for e in rows for i in e['outcome_ids']}),outcome_disjoint_blocks=len(blocks),
                median_faster_seconds=float(np.median(changes)) if changes else None,
                improvement_fraction=sum(v>0 for v in changes)/len(changes) if changes else None,
                first_origin=min((r['origin'] for r in rows),default=None),last_origin=max((r['origin'] for r in rows),default=None))
    return result


def evaluate(examples):
    """Same outcomes, pooled versus consistent-only fits, strict time embargo.

    The uninterrupted-future stratum is conditional information, just like
    completed future mileage in the original audit. It cannot be known in
    advance; an interruption would require revising that conditional forecast.
    """
    predictions=[]
    for origin in sorted({r['origin'] for r in examples}):
        prior=[r for r in examples if r['outcome_end']<origin]
        for fit_group in ('pooled','consistent_only'):
            train=[r for r in prior if fit_group=='pooled' or r['consistency']['regime']=='consistent']
            fits={model:fit_response(train,model,as_of=datetime.fromisoformat(origin)) for model in MODELS}
            for e in (r for r in examples if r['origin']==origin):
                for model,fit in fits.items():
                    if fit is None:continue
                    predicted=e['start_pace']*math.exp(predict(fit,e))
                    predictions.append(dict(model=model,fit_group=fit_group,regime=e['consistency']['regime'],
                        origin=origin,horizon=e['horizon'],target_id=e['target_id'],outcome_ids=e['outcome_ids'],
                        outcome_end=e['outcome_end'],actual=e['actual'],predicted=predicted,persistence=e['start_pace'],
                        log_error=math.log(predicted/e['actual']),latest_training_outcome=fit['latest_outcome'],
                        training_origins=fit['origins'],coefficients=fit['coefficients']))
    return predictions


def comparisons(predictions):
    result={}
    for h in HORIZONS:
        result[str(h)]={}
        for model in MODELS:
            both={(r['origin'],r['target_id']):r for r in predictions
                if r['model']==model and r['fit_group']=='pooled' and r['horizon']==h and r['regime']=='consistent'}
            clean=[r for r in predictions if r['model']==model and r['fit_group']=='consistent_only'
                and r['horizon']==h and r['regime']=='consistent' and (r['origin'],r['target_id']) in both]
            matched=[both[r['origin'],r['target_id']] for r in clean]
            persistence=[{**r,'predicted':r['persistence'],'log_error':math.log(r['persistence']/r['actual'])} for r in clean]
            result[str(h)][model]=dict(pooled=score(matched),consistent_only=score(clean),persistence=score(persistence),
                consistent_only_disjoint=score(disjoint(clean)),
                all_pooled_consistent_test=score(list(both.values())),
                fitted_growth_fraction=(sum(r['predicted']<r['persistence'] for r in clean)/len(clean) if clean else None))
    return result


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=Path('output/empirical-training-response-benchmark-20261020.json'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    base=json.loads(args.input.read_text());as_of=datetime.fromisoformat(base['as_of'])
    era_start=datetime.fromisoformat(base['era_start'])
    config=load_config(root/'config.yaml');database=resolve_project_path(root,config['paths']['database'])
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True) as c:
        c.row_factory=sqlite3.Row;prepared=prepare_progress_data(c)
        meta={r['id']:dict(r) for r in c.execute('SELECT id,start_time_utc,total_elapsed_time_s FROM activities')}
        training=[];health=[]
        for s in prepared.sessions:
            end=s.start_time+timedelta(seconds=meta[s.activity_id]['total_elapsed_time_s'] or 0)
            if end>as_of:continue
            role=prepared.details[s.activity_id]['workout'].value
            if role in {'hike','bike'}:continue
            tag=prepared.details[s.activity_id]['health_tag']
            training.append(dict(id=s.activity_id,time=s.start_time,available_at=end,miles=s.distance_miles,
                minutes=s.moving_minutes,quality=role in {'intervals','tempo_threshold','race'},long=role=='long',health_tag=tag))
            if tag!='normal':health.append(dict(at=s.start_time,status=tag,activity_id=s.activity_id))
        if c.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='user_health_reports'").fetchone():
            health.extend(dict(at=datetime.fromisoformat(r['reported_at']),status=r['status'])
                for r in c.execute('SELECT reported_at,status FROM user_health_reports') if datetime.fromisoformat(r['reported_at'])<=as_of)
        assert c.total_changes==0
    # The frozen responses and current DB exposure must refer to the same
    # physical history. New recordings after the input as_of are excluded.
    assert max(training,key=lambda r:r['available_at'])['id']==base['latest_training_id']
    policies={}
    for gap in (7,10):
        for fraction in (.5,2/3):
            key=f'gap{gap}_fraction{fraction:.3f}'
            examples=annotate(base['examples'],training,health,era_start=era_start,
                max_gap_days=gap,minimum_fraction=fraction)
            predictions=evaluate(examples)
            fits={model:fit_response([e for e in examples if e['consistency']['regime']=='consistent'],model,as_of=as_of) for model in MODELS}
            policies[key]=dict(summary=summaries(examples),comparisons=comparisons(predictions),examples=examples,
                predictions=predictions,final_consistent_fits=fits,
                observed_consistent_blocks=observed_blocks([e for e in examples if e['consistency']['regime']=='consistent']))
    primary=policies['gap7_fraction0.500']
    result=dict(version='consistent-training-response-v1',source=str(args.input),as_of=base['as_of'],
        primary_policy='gap7_fraction0.500',policies=policies,
        health_events=[dict(at=r['at'].isoformat(),status=r['status'],activity_id=r.get('activity_id')) for r in health if r['at']>=era_start],
        user_context=[dict(period='2026-08-07/2026-08-19',
            provenance='User recalled vacation during the second gap; not an injury attribution',
            effect='Context only; no live health or activity records changed')],
        protocol=['Same frozen causal benchmark responses and exposure as prior test',
            'Regimes use training and health only, never performance or error sign',
            'Baseline is personal previous 28 days; 14-day volume and frequency checked daily',
            '7/10-day maximum gaps and 50%/67% baseline floors are diagnostic sensitivity choices, not physiological cutoffs',
            'Only completed earlier response periods enter historical fits; matched comparisons use identical consistent test outcomes',
            'Conditional uninterrupted-future classification uses future training/health, not information available at forecast origin'],
        limitations=['No health flag means not documented, not proven healthy',
            'Recorded gaps may reflect missing uploads rather than no running',
            'Filtered benchmark origins can retain earlier disrupted history; no historical pace scores were refitted or deleted',
            'Outcome-disjoint blocks still share benchmark history; small cohorts are not independent trials'],
        live_model_changed=False,planner_changed=False,database_changed=False)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');private_file(args.output)
    print(json.dumps(dict(primary_four_week=primary['summary']['28'],
        matched_four_week=primary['comparisons']['28']['miles'],
        sensitivity={key:value['summary']['28']['consistent'] for key,value in policies.items()},
        health_events=result['health_events']),indent=2))


if __name__=='__main__':main()
