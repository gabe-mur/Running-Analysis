"""Bounded real-forecast and structural contrasts for a capped volume curve.

No daily selection or live publication. Coefficients are declared before results;
one physical evaluation supplies every curve comparison, avoiding repeated search.
"""
from argparse import ArgumentParser
from dataclasses import replace
from datetime import date,timedelta
import json
from pathlib import Path
from time import perf_counter

from scripts.replay_adaptive_search import load_decision
from scripts.research_manifest import freeze_sources
from run_analysis.adaptive.decision_snapshot import evaluate_snapshot
from run_analysis.adaptive.volume_benefit_trial import linear_volume_loss,volume_benefit_cost
from run_analysis.adaptive.shared_program_evaluator import ProgramContext,ProgramCandidate,evaluate_program
from run_analysis.adaptive.program_objective_trial import TrialPolicy,TrialSession,DemandScenario,EasyDurationPolicy
from run_analysis.adaptive.timely_purpose_trial import PurposeTimingPolicy
from run_analysis.adaptive.new_model_daily_harness import planned_accounted
from run_analysis.continuous_training_dose import ContinuousDoseReference
from run_analysis.workload_accounting import Quantity
from run_analysis.program_delivery import TargetSegment
from run_analysis.config import load_config
from datetime import datetime,time
from zoneinfo import ZoneInfo


COEFFICIENTS=(0.,.05,.1,.2,.4)
NOMINAL=.1


def frozen_case(root,index,prior):
    case=load_decision(root,index)
    state,catalog,handoffs=(case[k] for k in ('state','catalog','handoffs'))
    records={tuple(map(tuple,r['program'])):r for r in
        (prior['baseline'],*prior['variants'],*prior['dose_path'])}
    reference=ContinuousDoseReference('sandbox upload accounting')
    physical={}
    results=[]
    for program,record in records.items():
        for day,key in program:
            if (day,key) not in physical:
                physical[day,key]=planned_accounted(f'candidate:{day}:{key}',
                    datetime.combine(date.fromisoformat(day),time(18),ZoneInfo(state.timezone)),
                    catalog[key],handoffs[key],reference,available=state.as_of)
        outcome=evaluate_snapshot(state,tuple(physical[p] for p in program),
            tuple(k for _,k in program),reference=reference)
        e=outcome['evaluation']
        assert abs(e.total_cost-record['cost'])<1e-12
        assert outcome['decision_eligible']==record['eligible']
        linear=linear_volume_loss(state.context,e)
        results.append(dict(program=program,hard_miles=record['hard_miles'],eligible=record['eligible'],
            squared_volume=e.parts['shortfall_by_purpose']['volume'],linear_volume=linear,
            demand=e.weighted_demand_cost,short=e.short_run_cost,
            costs={str(a):volume_benefit_cost(state.context,e,linear_coefficient=a,linear_loss=linear)['cost']
                for a in COEFFICIENTS}))
    base=next(r for r in results if r['program']==tuple(map(tuple,prior['baseline']['program'])))
    summary=[]
    for a in COEFFICIENTS:
        eligible=[r for r in results if r['eligible']]
        winner=min(eligible,key=lambda r:(r['costs'][str(a)],r['program']))
        funded=[r for r in eligible if r['hard_miles']>=prior['target_integral']-1e-9]
        best_funded=min(funded,key=lambda r:r['costs'][str(a)])
        summary.append(dict(coefficient=a,best_tested_miles=winner['hard_miles'],
            best_delta=winner['costs'][str(a)]-base['costs'][str(a)],
            best_program=winner['program'],funded_witness_delta=best_funded['costs'][str(a)]-base['costs'][str(a)],
            funded_witness_miles=best_funded['hard_miles'],
            funded_program=best_funded['program']))
    return dict(index=index,target_integral=prior['target_integral'],baseline=base,
        evaluations=len(results),summary=summary,candidates=results)


def structural_controls():
    """R1/R3 motif/dose contrasts under the current rolling/handoff accounting.

    Duration/cardio rates are explicit synthetic scenarios, not new fitted signals.
    Future stress multipliers are sensitivity controls, not recovery categories.
    """
    start=date(2026,9,27);end=start+timedelta(days=54)
    q=lambda x:Quantity.point(x,'Declared R1/R3 pattern sensitivity scenario')
    def session(offset,miles,role,source='planned',factor=1.,scale=1.):
        # Doubled distance and speed preserves workload when scale=2.
        minutes=miles*10/scale
        cardio=miles*(20 if role=='quality' else 10)/scale
        return TrialSession(f'{source}:{offset}',start+timedelta(days=offset),miles,role,
            (q(minutes*factor),q(cardio*factor),q(15 if role=='quality' else 0)),
            1 if role=='quality' else 0,source)
    def programme(easies=((2,3.5),(6,3.6)),*,history=False,factor=1.,scale=1.):
        motif=((0,8.4,'long'),(4,4.5,'quality'),*((d,m,'easy') for d,m in easies))
        return tuple(session(offset,m*scale,role,'observed' if history else 'planned',factor,scale)
            for offset in range(-42,0) if history for day,m,role in motif if offset%7==day) if history else tuple(
            session(offset,m*scale,role,factor=factor,scale=scale)
            for offset in range(54) for day,m,role in motif if offset%7==day)
    patterns=dict(maintained=((2,3.5),(6,3.6)),one_percent_cut=((2,3.4),(6,3.5)),
        no_easy=(),compressed=((1,3.5),(2,3.6)),uneven=((2,2.5),(6,4.6)),
        fragmented=((1,1.),(2,1.),(3,1.),(6,4.1)),concentrated=((2,7.1),),
        surplus=((2,4.5),(6,4.6)))
    panels=[]
    for scale in (1.,2.):
        for memory in (1.,2.,4.):
            for factor in (1.,2.,4.):
                policy=TrialPolicy(weekly_miles=20*scale,ordinary_miles=3.5*scale,long_miles=8.4*scale,
                    ordinary_per_7=2.,purpose_weights=(1.,0.,1.,1.),volume_window_days=28)
                refs=tuple((role,m*scale,session(0,m*scale,role,scale=scale).demand)
                    for role,m in (('easy',3.5),('long',8.4),('quality',4.5)))
                ctx=ProgramContext(programme(history=True,scale=scale),start,start,start+timedelta(days=21),end,
                    (35.,35.,15.),policy,DemandScenario(memory_days=memory),.01,PurposeTimingPolicy(1.),
                    easy_duration_policy=EasyDurationPolicy(10,25,.01,'R1 provisional',7.1*scale),
                    volume_targets=(TargetSegment(start-timedelta(days=33),end,20*scale),),
                    support_targets=(TargetSegment(start-timedelta(days=33),end,7.1*scale),),
                    soft_horizon_taper=True,delivery_contract='sustained_v1',
                    continuation_boundary_references=refs,continuation_guard_days=7)
                scores={}
                for name,easy in patterns.items():
                    rows=programme(easy,factor=factor,scale=scale)
                    candidate=ProgramCandidate(tuple(r for r in rows if r.day<ctx.hard_end),
                        tuple(r for r in rows if r.day>=ctx.hard_end),'R1/R3 supplied diagnostic calendar')
                    e=evaluate_program(ctx,candidate);linear=linear_volume_loss(ctx,e)
                    scores[name]=dict(miles=sum(r.miles for r in rows),demand=e.weighted_demand_cost,
                        short=e.short_run_cost,squared_volume=e.parts['shortfall_by_purpose']['volume'],
                        linear_volume=linear,costs={str(a):volume_benefit_cost(ctx,e,linear_coefficient=a,
                            linear_loss=linear)['cost'] for a in COEFFICIENTS})
                panels.append(dict(scale=scale,memory=memory,future_demand_factor=factor,scores=scores))
    checks=[]
    for a in COEFFICIENTS:
        costs=lambda p,n:p['scores'][n]['costs'][str(a)]
        normal=[p for p in panels if p['future_demand_factor']==1.]
        severe=[p for p in panels if p['future_demand_factor']==4.]
        checks.append(dict(coefficient=a,
            normal_maintained_over_omission=sum(int(costs(p,'maintained')<costs(p,'no_easy')) for p in normal),
            normal_distributed_over_compressed=sum(int(costs(p,'maintained')<costs(p,'compressed')) for p in normal),
            normal_distributed_over_fragments=sum(int(costs(p,'maintained')<costs(p,'fragmented')) for p in normal),
            normal_maintained_over_surplus=sum(int(costs(p,'maintained')<costs(p,'surplus')) for p in normal),
            normal_maintained_over_small_cut=sum(int(costs(p,'maintained')<costs(p,'one_percent_cut')) for p in normal),
            severe_omission_over_full=sum(int(costs(p,'no_easy')<costs(p,'maintained')) for p in severe),
            normal_count=len(normal),severe_count=len(severe)))
    for p in panels:
        if p['scale']!=1.:continue
        other=next(x for x in panels if x['scale']==2. and x['memory']==p['memory'] and
            x['future_demand_factor']==p['future_demand_factor'])
        for n in patterns:
            for a in COEFFICIENTS:
                assert abs(costs_for(p,n,a)-costs_for(other,n,a))<1e-12
    return dict(panels=panels,checks=checks,evaluations=len(panels)*len(patterns),
        limitation='Demand multipliers/rates declared for sensitivity; not observed recovery thresholds')


def costs_for(panel,name,coefficient):
    return panel['scores'][name]['costs'][str(coefficient)]


def main():
    p=ArgumentParser();p.add_argument('--root',type=Path,default=Path('output/single-volume-daily-2-20260929'))
    p.add_argument('--prior',type=Path,default=Path('output/single-volume-funding-diagnosis-20260929'))
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
    inputs=[a.root/'scenario.json']
    for i in (1,3):
        event=a.root/f'decision-{i:04d}.json'
        inputs.extend((event,a.root/json.loads(event.read_text())['snapshot_file'],a.prior/f'event-{i:04d}.json'))
    manifest=freeze_sources(a.output/'sources.zip',config=load_config(),entrypoint=__name__,inputs=tuple(inputs))
    stamp=perf_counter();results=[]
    for i in (1,3):
        prior=json.loads((a.prior/f'event-{i:04d}.json').read_text())
        result=frozen_case(a.root,i,prior);results.append(result)
        print(json.dumps(dict(index=i,summary=[{k:v for k,v in r.items() if not k.endswith('program')}
            for r in result['summary']]),default=str),flush=True)
    controls=structural_controls()
    report=dict(coefficients=COEFFICIENTS,nominal_provisional=NOMINAL,manifest=manifest,
        frozen=results,controls=controls,seconds=perf_counter()-stamp,
        scope='Bounded score-shape ablation; no new search, daily rollout or activation')
    with (a.output/'report.json').open('x') as f:json.dump(report,f,indent=2,default=str)
    print(json.dumps(dict(seconds=report['seconds'],controls=controls['checks'])),flush=True)


if __name__=='__main__':main()
