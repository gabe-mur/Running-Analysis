"""Frozen real-forecast single-volume controls before daily search. No deployment."""
from argparse import ArgumentParser
from dataclasses import replace
from datetime import date,datetime,time
from itertools import combinations
import json
from pathlib import Path
from time import perf_counter
from zoneinfo import ZoneInfo

from scripts.replay_adaptive_search import load_decision
from scripts.research_manifest import freeze_sources
from run_analysis.config import load_config
from run_analysis.adaptive.decision_snapshot import evaluate_snapshot,serialize,restore
from run_analysis.adaptive.new_model_daily_harness import planned_accounted
from run_analysis.continuous_training_dose import ContinuousDoseReference


def run(root,output):
    output.mkdir(exist_ok=False)
    case=load_decision(root,6)
    state,catalog,hs=case['state'],case['catalog'],case['handoffs']
    c=state.context
    single=replace(state,context=replace(c,policy=replace(c.policy,purpose_weights=(1.,0.,1.,1.))))
    # New weights already belong to the canonical serialized score contract.
    assert restore(serialize(single),expected_fingerprint=single.fingerprint)==single
    manifest=freeze_sources(output/'sources.zip',config=load_config(),entrypoint=__name__,
        inputs=(root/'decision-0006.json',root/case['event']['snapshot_file'],root/'scenario.json'))
    stamp=perf_counter();calls=0
    ref=ContinuousDoseReference('sandbox upload accounting');zone=ZoneInfo(state.timezone)
    physical={}
    def evaluate(program,model):
        nonlocal calls
        rows=[]
        for day,key in program:
            if (day,key) not in physical:
                physical[day,key]=planned_accounted(f'candidate:{day}:{key}',
                    datetime.combine(date.fromisoformat(day),time(18),zone),catalog[key],hs[key],ref,available=state.as_of)
            rows.append(physical[day,key])
        result=evaluate_snapshot(model,tuple(rows),tuple(k for _,k in program),reference=ref)
        e=result['evaluation'];calls+=1
        return dict(cost=e.total_cost,eligible=result['decision_eligible'],fatigue=e.weighted_demand_cost,
            delivery=e.delivery_cost,short=e.short_run_cost,losses=e.parts['shortfall_by_purpose'],
            version=e.model_version)
    baseline=tuple(map(tuple,case['event']['program']))
    variants={'baseline':baseline,'empty':(), 'no_easy':tuple((d,k) for d,k in baseline if catalog[k].role!='easy')}
    lower=[]
    for d,k in baseline:
        q=catalog[k]
        key=f'easy:{q.miles-.5:.1f}' if q.role=='easy' else k
        lower.append((d,key if key in hs else k))
    variants['reduce_easy_half_mile']=tuple(lower)
    easy=[(d,k) for d,k in baseline if catalog[k].role=='easy' and d<c.hard_end.isoformat()]
    for i,((d1,k1),(d2,k2)) in enumerate(combinations(easy,2)):
        total=round(catalog[k1].miles+catalog[k2].miles,1)
        for key in hs:
            if catalog[key].role not in ('long','quality'):continue
            remain=round(total-catalog[key].miles,1)
            remainder=f'easy:{remain:.1f}'
            if remainder not in hs:continue
            for first,second in ((d1,d2),(d2,d1)):
                change={first:key,second:remainder}
                variants[f'substitute:{i}:{key}:{first}']=tuple((d,change.get(d,k)) for d,k in baseline)
    records=[]
    for name,program in variants.items():
        old,new=evaluate(program,state),evaluate(program,single)
        if name.startswith('substitute:'):
            assert abs(sum(catalog[k].miles for _,k in program)-sum(catalog[k].miles for _,k in baseline))<1e-8
        records.append(dict(name=name,old=old,single=new,program=program))
    base=records[0]
    assert abs(base['old']['cost']-case['event']['cost'])<1e-12
    assert all(r['single']['cost']>base['single']['cost'] for r in records if r['name'] in ('empty','no_easy'))
    substitutions=[r for r in records if r['name'].startswith('substitute:') and r['single']['eligible']]
    best=min(substitutions,key=lambda r:r['single']['cost']) if substitutions else None
    report=dict(manifest=manifest,evaluations=calls,seconds=perf_counter()-stamp,baseline=base,
        best_substitution=best,substitutions=len(substitutions),
        better_substitutions=[r['name'] for r in substitutions if r['single']['cost']<base['single']['cost']-1e-12],
        controls=records,limitations=['Fixed archived incumbent, not reoptimized under new objective',
            'Existing stage review retained; no new ratio or frequency constraint',
            'Passing controls is not daily stability or progression acceptance'])
    (output/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ('manifest','controls')},indent=2))


if __name__=='__main__':
    p=ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('output/memory-continuation-daily-2-20260929'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.root,a.output)
