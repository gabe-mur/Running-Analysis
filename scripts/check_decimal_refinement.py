"""Frozen real-HR decision replay for decimal search; never publishes a plan."""
from argparse import ArgumentParser
from dataclasses import replace
from datetime import date, datetime, time
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from scripts.replay_adaptive_search import load_decision
from run_analysis.adaptive.adaptive_candidate_pipeline import resize_easy
from run_analysis.adaptive.continuous_search_adapter import search
from run_analysis.adaptive.decimal_dose_search import calendar_representatives
from run_analysis.adaptive.decision_snapshot import evaluate_snapshot
from run_analysis.adaptive.execution_forecast_contract import issue_execution
from run_analysis.adaptive.new_model_daily_harness import DailyHarness, HarnessPolicy, planned_accounted, churn
from run_analysis.adaptive.stage_calendar_seeds import complete_stage_seeds
from run_analysis.continuous_training_dose import ContinuousDoseReference


def expand(case, policy):
    """Forecast every tenth using the existing command/HR accounting contract."""
    state=case['state']; contracts=dict(state.contracts)
    catalog=dict(case['catalog']); handoffs=dict(case['handoffs'])
    ref=ContinuousDoseReference('sandbox upload accounting')
    helper=SimpleNamespace(policy=policy,zone=ZoneInfo(state.timezone),ref=ref,
        coaching_contracts=contracts,sandbox=SimpleNamespace(context=state.measurement_context))
    base=next(d for d in catalog.values() if d.role=='easy')
    contract=contracts[base.key]
    for n in range(round(contract.ordinary.miles.low*10),round(contract.ordinary.miles.high*10)+1):
        key=f'easy:{n/10:.1f}'
        if key in catalog:
            continue
        dose=resize_easy(base,n/10,contract)
        contracts[key]=contract
        intent=DailyHarness.intent(helper,dose,state.context.decision,state.as_of,f'decimal:{key}')
        handoff=issue_execution(intent,state.phase_history).payload['planning_handoff']
        if not handoff['usable']:
            raise ValueError(f'Unusable fine forecast: {key}')
        catalog[key]=dose; handoffs[key]=handoff
    state=replace(state,contracts=tuple(sorted(contracts.items())))
    runs={(day,key):planned_accounted(f'candidate:{day}:{key}',
        datetime.combine(date.fromisoformat(day),time(18),ZoneInfo(state.timezone)),
        catalog[key],handoffs[key],ref,available=state.as_of)
        for day in case['dates'] for key in handoffs}
    def evaluate(program):
        r=evaluate_snapshot(state,tuple(runs[p] for p in program),tuple(k for _,k in program),reference=ref)
        return dict(cost=r['evaluation'].total_cost,eligible=r['decision_eligible'],
            construction_eligible=r['decision_eligible'])
    archived=evaluate(tuple(map(tuple,case['event']['program'])))
    assert archived['eligible'] and abs(archived['cost']-case['archived_cost'])<1e-12
    return state,catalog,handoffs,evaluate


def main():
    p=ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--indices',type=int,nargs='+',default=[6,7])
    p.add_argument('--evaluations',type=int,default=10000)
    args=p.parse_args()
    args.output.mkdir(exist_ok=False)
    scenario=json.loads((args.root/'scenario.json').read_text())
    policy=HarnessPolicy(**scenario['search_policy'])
    previous=None
    for index in args.indices:
        stamp=perf_counter()
        case=load_decision(args.root,index)
        state,catalog,handoffs,evaluate=expand(case,policy)
        prepared=perf_counter()-stamp
        prior_event=json.loads((args.root/f'decision-{index-1:04d}.json').read_text())
        prior=tuple((d,k) for d,k in (previous or prior_event['program']) if d in case['dates'])
        seed_policy=state.context.policy
        support=state.context.easy_duration_policy.supporting_easy_miles_per_7
        exploration=calendar_representatives(catalog,ordinary=seed_policy.ordinary_miles,support=support)
        seed_keys={min((k for k in handoffs if catalog[k].role=='easy'),
            key=lambda k:(abs(catalog[k].miles-target),k)) for target in (seed_policy.ordinary_miles,seed_policy.ordinary_miles/2)}
        for role in ('long','quality'):
            seed_keys.add(max((k for k in handoffs if catalog[k].role==role),key=lambda k:(
                sum(s for kind,s in catalog[k].phases if kind=='work') if role=='quality' else catalog[k].miles,k)))
        paths=case['event'].get('stage_runtime',{}).get('target_history',())
        if paths and paths[-1].get('ramp_days'):
            path=paths[-1]
            seed_policy=replace(seed_policy,weekly_miles=path['volume_goal'])
            support=path['support_goal']
        result=search(dates=case['dates'],keys=tuple(handoffs),previous=prior,evaluate=evaluate,
            budget=args.evaluations,hard_end=state.context.hard_end.isoformat(),
            roles={k:catalog[k].role for k in handoffs},dose_miles={k:catalog[k].miles for k in handoffs},
            exploration_keys=exploration,seed_keys=tuple(sorted(seed_keys)),previous_end=prior_event['end'],
            seed_programs=complete_stage_seeds(case['dates'],catalog,seed_policy,supporting_miles=support,limit=168))
        result.update(index=index,preparation_seconds=prepared,total_seconds=perf_counter()-stamp,
            archived_cost=case['archived_cost'],churn=churn(prior,result['program'],state.context.decision,catalog),
            scope='Frozen decision replay; second decision carries first replay winner. Not daily simulation.')
        assert result['selected']==evaluate(result['program'])
        (args.output/f'replay-{index:04d}.json').write_text(json.dumps(result,indent=2))
        print(json.dumps({k:result[k] for k in ('index','cost','archived_cost','evaluations','seconds',
            'preparation_seconds','total_seconds','churn','local_neighborhood_complete') }),flush=True)
        previous=result['program']


if __name__=='__main__':
    main()
