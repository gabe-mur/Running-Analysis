"""One frozen real-forecast search under the provisional volume-benefit curve."""
from argparse import ArgumentParser
from datetime import date,datetime,time,timedelta
import json
from pathlib import Path
from zoneinfo import ZoneInfo

from scripts.replay_adaptive_search import load_decision
from scripts.research_manifest import freeze_sources
from run_analysis.config import load_config
from run_analysis.adaptive.decision_snapshot import evaluate_snapshot
from run_analysis.adaptive.volume_benefit_trial import volume_benefit_cost
from run_analysis.adaptive.continuous_search_adapter import search
from run_analysis.adaptive.stage_calendar_seeds import complete_stage_seeds
from run_analysis.adaptive.new_model_daily_harness import planned_accounted
from run_analysis.continuous_training_dose import ContinuousDoseReference


def run(root,index,output,*,coefficient=.1,evaluations=10000):
    output.mkdir(exist_ok=False)
    case=load_decision(root,index)
    event,state,catalog,dates,handoffs=(case[k] for k in ('event','state','catalog','dates','handoffs'))
    manifest=freeze_sources(output/'sources.zip',config=load_config(),entrypoint=__name__,
        inputs=(root/f'decision-{index:04d}.json',root/event['snapshot_file'],root/'scenario.json'))
    reference=ContinuousDoseReference('sandbox upload accounting');physical={}
    def evaluate(program):
        for day,key in program:
            if (day,key) not in physical:
                physical[day,key]=planned_accounted(f'candidate:{day}:{key}',
                    datetime.combine(date.fromisoformat(day),time(18),ZoneInfo(state.timezone)),
                    catalog[key],handoffs[key],reference,available=state.as_of)
        r=evaluate_snapshot(state,tuple(physical[p] for p in program),tuple(k for _,k in program),reference=reference)
        score=volume_benefit_cost(state.context,r['evaluation'],linear_coefficient=coefficient)
        return dict(cost=score['cost'],eligible=r['decision_eligible'],score=score,
            suitability=r['suitability'],constraints=r['constraints'])
    previous=tuple(map(tuple,event['program']))
    archived=evaluate(previous)
    assert abs(archived['score']['baseline_cost']-event['cost'])<1e-12
    seeds=set()
    for target in (state.context.policy.ordinary_miles,state.context.policy.ordinary_miles/2):
        seeds.add(min((k for k in handoffs if catalog[k].role=='easy'),
            key=lambda k:(abs(catalog[k].miles-target),k)))
    for role in ('long','quality'):
        seeds.add(max((k for k in handoffs if catalog[k].role==role),key=lambda k:(
            sum(s for p,s in catalog[k].phases if p=='work') if role=='quality' else catalog[k].miles,k)))
    result=search(dates=dates,keys=tuple(sorted(handoffs)),previous=previous,
        previous_end=event['end'],evaluate=evaluate,budget=evaluations,
        frontier_repair=True,stable_days=7,hard_end=state.context.hard_end.isoformat(),
        roles={k:catalog[k].role for k in handoffs},seed_keys=tuple(sorted(seeds)),
        dose_miles={k:catalog[k].miles for k in handoffs},
        seed_programs=complete_stage_seeds(dates,catalog,state.context.policy,
            supporting_miles=state.context.easy_duration_policy.supporting_easy_miles_per_7))
    # Re-evaluate the selected physical plan independently of the search cache.
    replay=evaluate(tuple(map(tuple,result['program'])))
    assert abs(replay['cost']-result['cost'])<1e-12 and replay['eligible']
    hard=[(d,catalog[k].role,catalog[k].miles) for d,k in result['program'] if d<event['hard_end']]
    metric=dict(miles=sum(m for d,r,m in hard),runs=len(hard),
        role_counts={r:sum(role==r for _,role,_ in hard) for r in ('easy','long','quality')},
        role_sequence=hard)
    metric['role_gaps']={role:[(date.fromisoformat(b)-date.fromisoformat(a)).days
        for a,b in zip([d for d,r,m in hard if r==role],[d for d,r,m in hard if r==role][1:])]
        for role in ('long','quality')}
    result.update(manifest=manifest,coefficient=coefficient,snapshot_fingerprint=event['snapshot_fingerprint'],
        archived_trial=archived,selected_replay=replay,hard_training=metric,
        scope='Frozen same-evidence objective comparison; not daily churn or published plan')
    with (output/'result.json').open('x') as f:json.dump(result,f,indent=2,default=str)
    print(json.dumps(dict(seconds=result['seconds'],evaluations=result['evaluations'],
        archived_cost=archived['cost'],new_cost=result['cost'],training=metric,
        local_complete=result['local_neighborhood_complete'])),flush=True)


def inspect_result(root,index,output):
    """Independent rolling-delivery inspection of an already completed search."""
    saved=json.loads((output/'result.json').read_text())
    case=load_decision(root,index)
    state,catalog,handoffs=(case[k] for k in ('state','catalog','handoffs'))
    assert state.fingerprint==saved['snapshot_fingerprint']
    reference=ContinuousDoseReference('sandbox upload accounting')
    evaluations=[]
    for label,program in (('old',case['event']['program']),('trial',saved['program'])):
        rows=tuple(planned_accounted(f'candidate:{day}:{key}',
            datetime.combine(date.fromisoformat(day),time(18),ZoneInfo(state.timezone)),
            catalog[key],handoffs[key],reference,available=state.as_of) for day,key in program)
        result=evaluate_snapshot(state,rows,tuple(k for d,k in program),reference=reference)
        e=result['evaluation']
        score=volume_benefit_cost(state.context,e,linear_coefficient=saved['coefficient'])
        if label=='trial':
            assert result['decision_eligible'] and abs(score['cost']-saved['cost'])<1e-12
        blocks=[]
        for left,right in ((0,7),(7,21),(21,40),(40,54)):
            volume=[r['volume'] for r in e.parts['rolling_volume']['daily']
                if left<=(r['day']-state.context.decision).days<right]
            fractions=[r['delivered']/r['target'] for r in volume]
            blocks.append(dict(offsets=[left,right],mean_fraction=sum(fractions)/len(fractions),
                minimum_fraction=min(fractions)))
        hard_dates=[date.fromisoformat(d) for d,k in program if d<state.context.hard_end.isoformat()]
        all_dates={r.day for r in e.conditional_ledger}
        streaks=[]
        for day in hard_dates:
            count=1
            while day-timedelta(days=count) in all_dates:count+=1
            streaks.append(count)
        evaluations.append(dict(label=label,score=score,blocks=blocks,
            whole_projection_miles=sum(catalog[k].miles for d,k in program),
            longest_hard_run_streak=max(streaks,default=0),
            longest_bracketed_gap=max(((b-a).days for a,b in zip(hard_dates,hard_dates[1:])),default=None)))
    with (output/'inspection.json').open('x') as f:json.dump(evaluations,f,indent=2,default=str)
    print(json.dumps(evaluations),flush=True)


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('--root',type=Path,default=Path('output/single-volume-daily-2-20260929'))
    p.add_argument('--event',type=int,default=1);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--coefficient',type=float,default=.1);p.add_argument('--evaluations',type=int,default=10000)
    p.add_argument('--inspect',action='store_true')
    a=p.parse_args()
    if a.inspect:inspect_result(a.root,a.event,a.output)
    else:run(a.root,a.event,a.output,coefficient=a.coefficient,evaluations=a.evaluations)
