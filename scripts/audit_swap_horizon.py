"""Frozen swap attribution: endpoint effects, matched prefixes and continuations."""
from argparse import ArgumentParser
from dataclasses import replace
from datetime import date,datetime,time,timedelta
import json
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from run_analysis.config import load_config
from run_analysis.continuous_training_dose import ContinuousDoseReference
from run_analysis.adaptive.decision_snapshot import evaluate_snapshot
from run_analysis.adaptive.new_model_daily_harness import planned_accounted,neighbors
from run_analysis.adaptive.frontier_continuation import rebuild_frontiers
from scripts.replay_adaptive_search import load_decision
from scripts.research_manifest import freeze_sources


def summary(v):
    return dict(cost=v.total_cost,demand=v.weighted_demand_cost,delivery=v.delivery_cost,
        short=v.short_run_cost,losses=v.parts['shortfall_by_purpose'])


def run(root,winner_path,output,*,conditional_budget=6000):
    if type(conditional_budget) is not int or conditional_budget<0:
        raise ValueError('Nonnegative conditional proposal budget required')
    a,b=[load_decision(root,i) for i in (9,10)]
    output.mkdir(exist_ok=False)
    manifest=freeze_sources(output/'sources.zip',config=load_config(),
        entrypoint='scripts.audit_swap_horizon',inputs=(winner_path,root/'scenario.json',
            *(root/f'decision-{i:04d}.json' for i in (9,10)),
            *(root/c['event']['snapshot_file'] for c in (a,b))))
    saved_winner=json.loads(winner_path.read_text())
    winner=tuple(map(tuple,saved_winner['program']))
    prior=tuple(map(tuple,a['event']['program']))
    dates=b['dates']; keys=tuple(sorted(b['handoffs']))
    boundary=(b['state'].context.decision+timedelta(days=7)).isoformat()
    roles={k:d.role for k,d in b['catalog'].items()}
    prefix=lambda p:tuple((d,roles[k]) for d,k in p if d<boundary)
    def splice(left,right,cut):
        return tuple(sorted((*[p for p in left if p[0]<cut],*[p for p in right if p[0]>=cut])))
    variants=dict(prior=prior,winner=winner,
        winner_with_prior_week=splice(prior,winner,boundary),
        prior_with_winner_week=splice(winner,prior,boundary))
    # Witness-local order reversal, keeping the winner's exact doses.
    reversed_order=dict(winner)
    reversed_order['2026-10-08'],reversed_order['2026-10-09']=reversed_order['2026-10-09'],reversed_order['2026-10-08']
    variants['winner_order_only_reversed']=tuple(sorted(reversed_order.items()))
    variants['prior_with_new_terminal_long']=tuple(sorted((*prior,('2026-11-04','long:8.4'))))
    ref=ContinuousDoseReference('sandbox upload accounting')
    def price(state,case,p):
        included=tuple((d,k) for d,k in p if state.context.decision.isoformat()<=d<state.context.end.isoformat())
        rows=tuple(planned_accounted(f'candidate:{d}:{k}',
            datetime.combine(date.fromisoformat(d),time(18),ZoneInfo(state.timezone)),
            case['catalog'][k],case['handoffs'][k],ref,available=state.as_of) for d,k in included)
        return evaluate_snapshot(state,rows,tuple(k for _,k in included),reference=ref)
    state=b['state']; old=a['state'].context
    contexts=dict(earlier=(a['state'],a),
        later_fixed=(replace(state,context=replace(state.context,hard_end=old.hard_end,end=old.end)),b),
        hard_only=(replace(state,context=replace(state.context,end=old.end)),b),
        endpoint_only=(replace(state,context=replace(state.context,hard_end=old.hard_end)),b),
        rolled=(state,b),
        earlier_tapered=(replace(a['state'],context=replace(a['state'].context,soft_horizon_taper=True)),a),
        rolled_tapered=(replace(state,context=replace(state.context,soft_horizon_taper=True)),b))
    matrix={}; terminal={}
    for label,(s,c) in contexts.items():
        matrix[label]={}
        for name,p in variants.items():
            result=price(s,c,p); v=result['evaluation']
            matrix[label][name]=dict(**summary(v),eligible=result['decision_eligible'])
            if label=='rolled':
                terminal[name]=dict(
                    volume=[r for r in v.parts['rolling_volume']['daily'] if r['day']>=old.end],
                    timing={purpose:[r for r in data['daily'] if r['day']>=old.end]
                        for purpose,data in v.parts['timing'].items()})
    if (abs(matrix['earlier']['prior']['cost']-a['archived_cost'])>1e-12 or
            abs(matrix['rolled']['winner']['cost']-saved_winner['cost'])>1e-12):
        raise ValueError('Physical decomposition does not reproduce frozen control scores')
    cache={}
    def assess(p,source):
        p=tuple(sorted(p))
        if p not in cache:
            r=price(state,b,p)
            cache[p]=dict(physical=p,score=SimpleNamespace(total_cost=r['evaluation'].total_cost),
                eligible=r['decision_eligible'],parts=summary(r['evaluation']))
        return cache[p] if cache[p]['eligible'] else None
    candidates=[*variants.values()]
    for cut in dates:
        candidates.extend((splice(prior,winner,cut),splice(winner,prior,cut)))
    for day in dates:
        if day>=a['event']['end'] and day not in dict(prior):
            candidates.extend(tuple(sorted((*prior,(day,k)))) for k in keys)
    admitted=[v for p in candidates if (v:=assess(p,'factorial')) is not None]
    fronts=[]
    for pattern in (prefix(prior),prefix(winner)):
        fronts.append(min((v for v in admitted if prefix(v['physical'])==pattern),key=lambda v:v['score'].total_cost))
    print(json.dumps(dict(matrix={l:{n:v['cost'] for n,v in row.items()} for l,row in matrix.items()},
        conditional_seeds=[dict(prefix=prefix(v['physical']),cost=v['score'].total_cost) for v in fronts])),flush=True)
    conditional=rebuild_frontiers(fronts,[v['physical'] for v in sorted(admitted,key=lambda v:v['score'].total_cost)],
        dates=dates,keys=keys,hard_end=state.context.hard_end.isoformat(),prefix_days=7,
        budget=conditional_budget,assess=assess,roles=roles,
        compound_neighbors=lambda v:neighbors(v['physical'],dates,keys))
    for i,lane in enumerate(conditional['lanes']):
        name=f'conditional_{i}'
        for label,(s,c) in contexts.items():
            r=price(s,c,lane['program'])
            matrix[label][name]=dict(**summary(r['evaluation']),eligible=r['decision_eligible'])
    report=dict(manifest=manifest,matrix=matrix,terminal=terminal,conditional=conditional,
        conditional_evaluations=len(cache),forecast_unchanged=a['event']['forecast_load_by_dose']==b['event']['forecast_load_by_dose'],
        hard_end_before=old.hard_end,hard_end_after=state.context.hard_end,
        end_before=old.end,end_after=state.context.end,
        normalization_before=(old.end-old.stage_start).days,
        normalization_after=(state.context.end-state.context.stage_start).days,
        scope='Witness-specific full-score decomposition and bounded conditional continuation search; no live changes or global certificate')
    with (output/'result.json').open('x') as f:
        json.dump(report,f,indent=2,default=lambda v:v.isoformat())
    print(json.dumps(dict(evaluations=len(cache),lanes=[{k:l[k] for k in ('seed_cost','best_cost','status','proposals')} for l in conditional['lanes']])),flush=True)


if __name__=='__main__':
    p=ArgumentParser(); p.add_argument('root',type=Path); p.add_argument('winner',type=Path)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--conditional-budget',type=int,default=6000); args=p.parse_args()
    run(args.root,args.winner,args.output_dir,conditional_budget=args.conditional_budget)
