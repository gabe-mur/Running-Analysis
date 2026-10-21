"""Shared opening-search experiment on immutable decisions; no publication.

Reproduce archived admission/cost first, then use the harness's existing seed
library, objective, dated rules and operators. A frozen comparison is NOT a
daily adherence test: no physiology or prescribed execution is synthesized.
"""
from argparse import ArgumentParser
from dataclasses import replace
from datetime import date
import json
from pathlib import Path

from run_analysis.config import load_config
from run_analysis.adaptive.continuous_search_adapter import search
from run_analysis.adaptive.decimal_dose_search import calendar_representatives
from run_analysis.adaptive.stage_calendar_seeds import complete_stage_seeds
from run_analysis.adaptive.new_model_daily_harness import churn
from run_analysis.adaptive.race_accounting import allowed_dose
from scripts.replay_adaptive_search import load_decision
from scripts.research_manifest import freeze_sources


def run(root, output, *, index, prior_root, prior_index, complete_first=True, evaluations=10000,
        final_winner_evaluations=0,evaluation_observer=None):
    case=load_decision(root,index)
    event,state,catalog,dates,handoffs=(case[k] for k in ('event','state','catalog','dates','handoffs'))
    prior_path=prior_root/f'decision-{prior_index:04d}.json'
    prior=json.loads(prior_path.read_text())
    output.mkdir(exist_ok=False)
    manifest=freeze_sources(output/'sources.zip',config=load_config(),entrypoint=__name__,
        inputs=(root/f'decision-{index:04d}.json',root/event['snapshot_file'],root/'scenario.json',prior_path))
    runtime=event['stage_runtime']
    policy=state.context.policy
    support=state.context.easy_duration_policy.supporting_easy_miles_per_7
    if runtime['stage']['policy']['volume_path']=='gradual':
        policy=replace(policy,weekly_miles=runtime['target_history'][-1]['volume_goal'])
        support=runtime['target_history'][-1]['support_goal']
    supported={k:catalog[k] for k in handoffs}
    seeds=complete_stage_seeds(dates,supported,policy,supporting_miles=support,
        limit=max(1,min(1680,evaluations//5)))
    previous=tuple(tuple(p) for p in prior['program'] if p[1] in handoffs)
    # Retain the pre-existing harness's dated admission filter, not a new
    # race-specific search rule or a changed coaching/fatigue objective.
    race=state.context.race_goal
    if race is not None:
        from types import SimpleNamespace
        stage=SimpleNamespace(**runtime['stage'])
        valid=lambda p:allowed_dose(race,date.fromisoformat(p[0]),catalog[p[1]],stage)
        previous=tuple(p for p in previous if valid(p))
        seeds=tuple(tuple(p for p in seed if valid(p)) for seed in seeds)
    seed_keys=set()
    for target in (state.context.policy.ordinary_miles,state.context.policy.ordinary_miles/2):
        easy=[k for k in handoffs if catalog[k].role=='easy']
        if easy:
            seed_keys.add(min(easy,key=lambda k:(abs(catalog[k].miles-target),k)))
    for role in ('long','quality'):
        choices=[k for k in handoffs if catalog[k].role==role]
        if choices:
            seed_keys.add(max(choices,key=lambda k:(sum(s for p,s in catalog[k].phases if p=='work')
                if role=='quality' else catalog[k].miles,k)))
    def evaluate(program):
        checked=case['evaluate'](program)
        if evaluation_observer is not None:
            evaluation_observer(program,dict(checked))
        return checked
    result=search(dates=dates,keys=tuple(handoffs),previous=previous,evaluate=evaluate,
        budget=evaluations,hard_end=state.context.hard_end.isoformat(),roles={k:catalog[k].role for k in handoffs},
        seed_programs=seeds,seed_keys=tuple(sorted(seed_keys)),dose_miles={k:catalog[k].miles for k in handoffs},
        exploration_keys=calendar_representatives(supported,ordinary=state.context.policy.ordinary_miles,
            support=state.context.easy_duration_policy.supporting_easy_miles_per_7),
        previous_end=prior['end'] if root==prior_root and index!=prior_index else None,
        recombination_dates=(state.context.taper_start.isoformat(),),complete_first=complete_first,
        final_winner_budget=final_winner_evaluations)
    witness=case['evaluate'](tuple(tuple(p) for p in event['program']))
    assert witness['eligible'] and abs(witness['cost']-event['cost'])<1e-12
    result.update(manifest=manifest,source_decision=str(root/f'decision-{index:04d}.json'),
        prior_source=str(prior_path),snapshot_fingerprint=state.fingerprint,
        archived_cost=case['archived_cost'],archived_winner_repriced=witness,
        churn_against_archived=churn(tuple(tuple(p) for p in event['program']),result['program'],state.context.decision,catalog),
        live_changed=False,scope='Frozen search-allocation diagnostic; NOT daily stability or global optimality')
    (output/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:result[k] for k in ('cost','archived_cost','seconds','evaluations',
        'local_neighborhood_complete','churn_against_archived')},indent=2),flush=True)
    print(json.dumps(result['refinement']['opening']),flush=True)
    print(json.dumps(result['refinement']['global_new_by_source']),flush=True)
    return result


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('--index',type=int,required=True)
    parser.add_argument('--prior-root',type=Path,required=True)
    parser.add_argument('--prior-index',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--evaluations',type=int,default=10000)
    parser.add_argument('--final-winner-evaluations',type=int,default=0)
    parser.add_argument('--legacy-opening',action='store_true')
    args=parser.parse_args()
    run(args.root,args.output,index=args.index,prior_root=args.prior_root,prior_index=args.prior_index,
        complete_first=not args.legacy_opening,evaluations=args.evaluations,
        final_winner_evaluations=args.final_winner_evaluations)
