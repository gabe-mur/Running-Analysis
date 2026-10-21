"""Read-only separation trial. Frozen pace calibration, chronological holdouts.

No live projection is promoted by this script. Future completed running is a
conditional input. Statistical components are not physiological ground truth.
"""
from argparse import ArgumentParser
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
from run_analysis.config import load_config,resolve_project_path
from run_analysis.pace_temperature import comparison_covariance
from run_analysis.privacy import private_file
from run_analysis.progress import prepare_progress_data
from run_analysis.training_adaptation import add_components,fit_change,change_components,projected_training
from scripts.audit_training_conditioned_pace import metrics


def evaluate(examples,*,penalty=4.):
    predictions=[]
    for e in examples:
        train=[v for v in examples if v['horizon']==e['horizon'] and v['outcome_end']<e['origin']]
        for model,keys in [('time',('elapsed','trend')),('training',('elapsed','trend','training_change')),
            ('separated',('elapsed','trend','training_change','temporary'))]:
            fit=fit_change(train,keys=keys,penalty=penalty)
            if not fit:continue
            parts=change_components(fit,e)
            predictions.append(dict(model=model,horizon=e['horizon'],origin=e['origin'],target_id=e['target_id'],
                outcome_ids=e['outcome_ids'],outcome_end=e['outcome_end'],actual=e['actual'],
                predicted=e['start_pace']*math.exp(sum(parts.values())),
                adaptation_only=e['start_pace']*math.exp(sum(v for k,v in parts.items() if k!='temporary')),
                training_linked_only=e['start_pace']*math.exp(parts.get('training_change',0)),
                persistence=e['start_pace'],components=parts,coefficients=fit['coefficients'],latest_training_outcome=fit['latest_outcome']))
    return predictions


def main():
    parser=ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    with urlopen('http://127.0.0.1:8000/api/progress-wip?window_days=365&include_exploratory=false&include_legacy_aerobic=false',timeout=150) as f:data=json.load(f)
    report=data['pace_comparison'];performance=data['aerobic_performance'];hidden=set(performance['hidden_activity_ids'])
    rows=sorted([dict(r) for r in report['runs'] if r['included_in_trend'] and r['trend_weight']>0
        and r['activity_id'] not in hidden],key=lambda r:(r['start_time'],r['activity_id']))
    config=load_config(root/'config.yaml');database=resolve_project_path(root,config['paths']['database'])
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True) as c:
        c.row_factory=sqlite3.Row;prepared=prepare_progress_data(c)
        metadata={r['id']:dict(r) for r in c.execute('SELECT id,start_time_utc,total_elapsed_time_s FROM activities')}
        for r in rows:
            meta=metadata[r['activity_id']];r['available_at']=(datetime.fromisoformat(meta['start_time_utc'])+timedelta(seconds=meta['total_elapsed_time_s'] or 0)).isoformat()
        training=[]
        for s in prepared.sessions:
            role=prepared.details[s.activity_id]['workout'].value
            if role in {'hike','bike'}:continue
            training.append(dict(id=s.activity_id,time=s.start_time,
                available_at=s.start_time+timedelta(seconds=metadata[s.activity_id]['total_elapsed_time_s'] or 0),
                miles=s.distance_miles,minutes=s.moving_minutes,quality=role in {'intervals','tempo_threshold','race'},long=role=='long'))
        assert c.total_changes==0
    covariance=comparison_covariance(report['calibration']);m,g=measurement_inputs(rows,covariance)
    variance=m+np.einsum('ij,jk,ik->i',g,covariance,g)
    base=make_examples(rows,variance,training,{},ZoneInfo(data['timezone']))
    results={}
    for memory in (14.,28.,56.):
        examples=add_components(base,rows,variance,training,memory_days=memory)
        predictions=evaluate(examples)
        summary={}
        for horizon in (14,28,56,84):
            subset=[p for p in predictions if p['horizon']==horizon]
            summary[str(horizon)]={name:metrics([p for p in subset if p['model']==name]) for name in ['time','training','separated']}
            separated=[p for p in subset if p['model']=='separated']
            summary[str(horizon)]['adaptation_only']=metrics([{**p,'predicted':p['adaptation_only']} for p in separated])
            summary[str(horizon)]['persistence']=metrics([{**p,'predicted':p['persistence']} for p in separated])
        fits={str(p):fit_change([e for e in examples if e['horizon']==28],penalty=p) for p in (1.,4.,16.)}
        results[str(memory)]=dict(examples=examples,predictions=predictions,summary=summary,fits=fits)
        print('memory',memory)
        for h,v in summary.items():print(h,{k:(round(x.get('mae_seconds_per_mile',0),2),x['pairs']) for k,x in v.items()})
        print('fit',fits['4.0'])
        for h in (14,28,56):
            subset=[p for p in predictions if p['horizon']==h and p['model']=='separated']
            score=metrics([{**p,'predicted':p['training_linked_only']} for p in subset])
            summary[str(h)]['training_linked_only']=score
            print('training-only',h,score)
        origin=datetime.fromisoformat(data['as_of'])
        goal=data['goal_journey']
        end=datetime.fromisoformat(goal['projection']['target_date']).date()
        horizon=(end-origin.astimezone(ZoneInfo(data['timezone'])).date()).days
        stock=projected_training(training,origin,horizon,volume=goal['volume_per_7'],growth=goal['volume_growth_per_7'],
            target_volume=goal['target_volume_per_7'],memory_days=memory)
        outlook={str(day):{p:performance['current']['pace_min_mile']*math.exp(f['coefficients']['training_change']*math.log(stock[day]/stock[0]))
            for p,f in fits.items()} for day in (0,7,28,56,horizon)}
        results[str(memory)]['outlook']=outlook
        print('outlook',outlook)
    result=dict(version='training-adaptation-separation-v1',as_of=data['as_of'],results=results,
        current_pace=performance['current']['pace_min_mile'],original_trend=performance['progress_trend'],
        limitations=['Frozen retrospective pace correction','Overlapping outcomes','Statistical innovation is not proven temporary fatigue',
            'Delayed exposure constants are assumptions, not identified physiological time constants','Long-term extrapolation not validated'],
        live_model_changed=False,planner_changed=False)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');private_file(args.output)


if __name__=='__main__':main()
