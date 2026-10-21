"""Copied-live 2%/3% preference experiment; no live writes or policy deployment.

Reuse the established runner panel, real harness, and perfect-prescription
daily driver. The driver uses transformed recorded HR, not new measured data.
"""
from argparse import ArgumentParser
from dataclasses import asdict,replace,fields
from datetime import date,timedelta
import json
from pathlib import Path
from unittest.mock import patch

from run_analysis.config import load_config
from run_analysis.adaptive.hr_reference import frozen_live_config
from run_analysis.adaptive.new_model_daily_harness import DailyHarness
from run_analysis.adaptive.frequency_range_trial import FrequencyRangePolicy
from scripts.run_adaptive_lifecycle import fork_seed_plan,run as daily_run
from scripts.replay_adaptive_search import load_decision
from scripts.prototype_frequency_ranges import variants


def save(path,value):
    path.write_text(json.dumps(value,indent=2,default=str,allow_nan=False)+'\n')


def profiles(output,panel):
    rows=variants(panel,FrequencyRangePolicy(),weights=(.02,.03))
    save(output/'profiles.json',dict(rows=rows,scope='Existing equal-volume periodic component controls; not full runner-specific searches'))
    print(json.dumps(dict(profiles=[dict(name=r['profile']['name'],
        weights=[dict(weight=s['edge_fraction'],runs=s['selected']['count'],
            fatigue_change=s['fatigue_change_vs_range_percent']) for s in r['midpoint_sweep']]) for r in rows])),flush=True)


def opening(source,output,weight):
    root=output/f'weight-{round(weight*100):02d}'
    at=fork_seed_plan(source,root)
    config=frozen_live_config(load_config(),root=source)
    h=DailyHarness.resume(root,config,as_of=at)
    try:
        old=h.frequency_contract
        h.frequency_contract=replace(old,preference=replace(old.preference,edge_fraction=weight))
        assert replace(h.frequency_contract,preference=old.preference)==old
        before=h.stage,h.target_history,tuple(h.history),h.scales,h.policy
        event=h.replan(at+timedelta(microseconds=1),reason=f'offline_frequency_weight_{weight:.2f}')
        assert (h.stage,h.target_history,tuple(h.history),h.scales,h.policy)==before
        case=load_decision(root,len(h.events))
        assert case['state'].frequency_contract==h.frequency_contract
        e=case['evaluate_details'](tuple(map(tuple,event['program'])))['evaluation']
        report=dict(weight=weight,cost=event['cost'],fatigue=e.weighted_demand_cost,
            delivery=e.delivery_cost,frequency_cost=e.parts['frequency_preference']['cost'],
            seconds=event['seconds'],evaluations=event['evaluations'],miles21=event['volume'],
            hard_schedule=[dict(day=d,role=h.catalog[k].role,miles=h.catalog[k].miles)
                for d,k in event['program'] if d<event['hard_end']],
            source_decisions=len(h.events)-1,search_policy=asdict(h.policy),
            live_changed=False,independently_repriced=True)
        save(output/f'opening-{round(weight*100):02d}.json',report)
        print(json.dumps(report),flush=True)
    finally:
        h.sandbox.close()
    return root


def daily(source,output,weight,database,days):
    seed=output/f'weight-{round(weight*100):02d}'
    config=frozen_live_config(load_config(),root=source)
    latest=json.loads(sorted(seed.glob('decision-*.json'))[-1].read_text())
    # The new screen starts on the next local day, not the historical stage origin.
    from datetime import datetime
    from zoneinfo import ZoneInfo
    start=datetime.fromisoformat(latest['as_of']).astimezone(ZoneInfo(config['timezone_default'])).date()+timedelta(days=1)
    target=output/f'daily-{round(weight*100):02d}'
    with patch('scripts.run_adaptive_lifecycle.load_config',return_value=config):
        report=daily_run(target,database,days=days,evaluations=10000,scenario='prescribed',
            seed_plan=seed,screen_start=start.isoformat(),stability_window=4)
    print(json.dumps(dict(weight=weight,status=report['status'],
        completed_days=report['completed_days'],failures=report['failures'])),flush=True)


def compare(output):
    cases=[load_decision(output/f'weight-{w:02d}',
        len(tuple((output/f'weight-{w:02d}').glob('decision-*.json')))) for w in (2,3)]
    differences=[f.name for f in fields(cases[0]['state'])
        if getattr(cases[0]['state'],f.name)!=getattr(cases[1]['state'],f.name)]
    assert differences==['frequency_contract']
    assert cases[0]['handoffs']==cases[1]['handoffs']
    rows=[]
    for w,c in zip((.02,.03),cases):
        e=c['event'];p=tuple(map(tuple,e['program']))
        r=c['evaluate_details'](p)['evaluation']
        hard=[(d,k) for d,k in p if d<e['hard_end']]
        streak=maximum=0;previous=None
        for d,_ in hard:
            day=date.fromisoformat(d)
            streak=streak+1 if previous and day==previous+timedelta(days=1) else 1
            maximum=max(maximum,streak);previous=day
        rows.append(dict(weight=w,runs21=len(hard),miles21=e['volume'],
            max_run_streak21=maximum,fatigue=r.weighted_demand_cost,
            cost_under_2pct=float(cases[0]['evaluate_details'](p)['evaluation'].total_cost),
            cost_under_3pct=float(cases[1]['evaluate_details'](p)['evaluation'].total_cost),
            hard_schedule=[dict(day=d,role=c['catalog'][k].role,miles=c['catalog'][k].miles) for d,k in hard]))
    report=dict(only_state_difference=differences,identical_forecasts=True,
        comparisons=rows,fatigue_change_percent=100*(rows[1]['fatigue']/rows[0]['fatigue']-1),
        live_changed=False,scope='Matched whole-pipeline opening searches; daily replay is separate')
    save(output/'comparison.json',report)
    print(json.dumps(report),flush=True)


def handoff(output):
    """Fixed-program counterfactuals, NOT a second full daily replay at 2%."""
    root=output/'daily-03'
    n=len(tuple(root.glob('decision-*.json')))
    case=load_decision(root,n);previous=load_decision(root,n-1)
    if case['event']['reason']!='accepted_stage_progression':
        raise ValueError('This attribution requires the actual progression handoff')
    selected=tuple(map(tuple,case['event']['program']))
    prior=tuple(tuple(p) for p in previous['event']['program'] if p[0]>=case['dates'][0])
    cut=(case['state'].context.decision+timedelta(days=3)).isoformat()
    hybrid=tuple(sorted([p for p in prior if p[0]<cut]+[p for p in selected if p[0]>=cut]))
    rows=[]
    for weight in (.02,.03):
        contract=case['state'].frequency_contract
        state=replace(case['state'],frequency_contract=replace(contract,
            preference=replace(contract.preference,edge_fraction=weight)))
        for label,program in [('selected',selected),('preserved_prior',prior),('preserved_near_term_new_tail',hybrid)]:
            r=case['evaluate_details'](program,state_override=state);e=r['evaluation']
            rows.append(dict(weight=weight,candidate=label,eligible=r['decision_eligible'],
                cost=float(e.total_cost),fatigue=e.weighted_demand_cost,
                delivery=e.delivery_cost,frequency=e.parts['frequency_preference']['cost']))
    report=dict(decision_file=n,reason=case['event']['reason'],counterfactuals=rows,
        scope='Identical new-stage state and fixed programmes, coefficient-only repricing; not a 2% daily replay',
        live_changed=False)
    save(output/'handoff-attribution.json',report)
    print(json.dumps(report),flush=True)


def main():
    p=ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('profiles','opening','daily','compare','handoff'))
    p.add_argument('--source',type=Path,default=Path('data/adaptive-planner-v206-coached-r2-20261009'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--weight',type=float,choices=(.02,.03),default=.03)
    p.add_argument('--panel',type=Path,default=Path('output/easy-profile-volume-curve-20260929/report.json'))
    p.add_argument('--database',type=Path,default=Path('data/run_analysis.sqlite'))
    p.add_argument('--days',type=int,default=3)
    args=p.parse_args();args.output.mkdir(exist_ok=True)
    if args.mode=='profiles':profiles(args.output,args.panel)
    elif args.mode=='opening':opening(args.source,args.output,args.weight)
    elif args.mode=='daily':daily(args.source,args.output,args.weight,args.database,args.days)
    elif args.mode=='compare':compare(args.output)
    else:handoff(args.output)


if __name__=='__main__':
    main()
