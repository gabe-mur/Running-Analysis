"""Factor later winner: discovery, forecast inputs and horizon, without search."""
from argparse import ArgumentParser
from dataclasses import replace
from datetime import date,datetime,time,timedelta
import json
from pathlib import Path
from zoneinfo import ZoneInfo

from run_analysis.continuous_training_dose import ContinuousDoseReference
from scripts.decision_snapshot import migrate_snapshot_defaults,evaluate_snapshot
from scripts.forecast_handoff import read_scope
from scripts.new_model_daily_harness import Dose,planned_accounted


def audit(root,*,volume_window=None,event_numbers=(1,3),purpose_terminal=False,steady_terminal=False,taper_soft=False):
    events=[json.loads((root/f'decision-{n:04d}.json').read_text()) for n in event_numbers]
    migrations=[migrate_snapshot_defaults((root/e['snapshot_file']).read_text(),expected_fingerprint=e['snapshot_fingerprint']) for e in events]
    states=[m['state'] for m in migrations]
    if taper_soft:
        if purpose_terminal or steady_terminal:
            raise ValueError('Separate horizon-policy ablations required')
        states=[replace(s,context=replace(s.context,steady_continuation_references=(),soft_horizon_taper=False)) for s in states]
    if volume_window is not None:
        states=[replace(s,context=replace(s.context,policy=replace(s.context.policy,volume_window_days=volume_window))) for s in states]
    scenario=json.loads((root/'scenario.json').read_text())
    catalog={d['key']:Dose(**dict(d,phases=tuple(tuple(p) for p in d['phases']),
        phase_scenarios=tuple(tuple(p) for p in d['phase_scenarios']),forecast_scope=read_scope(d['forecast_scope']))) for d in scenario['catalog']}
    ref=ContinuousDoseReference('sandbox upload accounting')
    def price(state,event,program):
        # A later candidate no longer contains completed dates. Restore the
        # SAME earlier prescription prefix when asking what it would have cost
        # before upload; otherwise this accidentally compares skipping the run.
        consumed={d:k for d,k in events[0]['program'] if d<states[1].context.decision.isoformat()}
        program=tuple({**consumed,**dict(program)}.items())
        p=tuple((d,k) for d,k in program if state.context.decision.isoformat()<=d<state.context.end.isoformat())
        rows=tuple(planned_accounted(f'back:{day}:{key}',datetime.combine(date.fromisoformat(day),time(18),ZoneInfo(state.timezone)),
            catalog[key],event['forecast_handoff_by_dose'][key],ref,available=state.as_of) for day,key in p)
        r=evaluate_snapshot(state,rows,tuple(k for _,k in p),reference=ref)
        v=r['evaluation']
        taper=None
        if taper_soft:
            from run_analysis.adaptive.soft_horizon_trial import tapered_cost
            uniform=tapered_cost(state.context,v,uniform=True)
            if abs(uniform['cost']-v.total_cost)>1e-12:
                raise ValueError('Daily decomposition does not reproduce existing untapered score')
            taper=tapered_cost(state.context,v)
        terminal=None
        if purpose_terminal or steady_terminal:
            if state.context.steady_continuation_references or state.context.soft_horizon_taper:
                raise ValueError('Snapshot already includes terminal accounting; do not charge it twice')
            if purpose_terminal and steady_terminal:
                raise ValueError('Terminal ablations must be priced separately')
            if steady_terminal:
                from run_analysis.adaptive.steady_continuation_trial import terminal_value
            else:
                from run_analysis.adaptive.purpose_terminal_trial import terminal_value
            references={}
            for role in (('easy','long','quality') if steady_terminal else ('long','quality')):
                choices=[d for d in catalog.values() if d.role==role]
                dose=(min(choices,key=lambda d:(abs(d.miles-state.context.policy.ordinary_miles),d.key))
                    if role=='easy' else max(choices,key=lambda d:
                        sum(s for k,s in d.phases if k=='work') if role=='quality' else d.miles))
                physical=planned_accounted('terminal-reference:'+role,
                    datetime.combine(state.context.decision,time(18),ZoneInfo(state.timezone)),
                    dose,event['forecast_handoff_by_dose'][dose.key],ref,available=state.as_of)
                references[role]=(physical.miles,physical.demand) if steady_terminal else physical.demand
            terminal=terminal_value(state.context,v,references)
        return dict(cost=taper['cost'] if taper else v.total_cost+(terminal['cost'] if terminal else 0),base_cost=v.total_cost,taper=taper,
            terminal=terminal,demand=v.weighted_demand_cost,delivery=v.delivery_cost,
            eligible=r['decision_eligible'],funding={k:v.parts[k] for k in ('delivery','targets','timing','easy_duration')})
    rows={}
    for state_name,state,event in (
        ('earlier',states[0],events[0]),
        ('later_fixed_end_frozen_forecast',replace(states[1],context=replace(states[1].context,
            hard_end=states[0].context.hard_end,end=states[0].context.end,
            steady_continuation_references=states[0].context.steady_continuation_references)),events[0]),
        ('later_fixed_end_new_forecast',replace(states[1],context=replace(states[1].context,
            hard_end=states[0].context.hard_end,end=states[0].context.end)),events[1]),
        ('later_rolling_end',states[1],events[1])):
        rows[state_name]={name:price(state,event,e['program']) for name,e in zip(('prior','later_winner'),events)}
    appended=[]
    day=events[0]['end']
    for key in (None,*catalog):
        program=events[0]['program']+([[day,key]] if key else [])
        result=price(states[1],events[1],program)
        if result['eligible']:
            appended.append(dict(dose=key,**result))
    splices=[]
    for n in range(1,(states[1].context.end-states[1].context.decision).days):
        split=(states[1].context.decision+timedelta(days=n)).isoformat()
        for prefix,suffix,label in ((events[0],events[1],'prior_prefix'),(events[1],events[0],'later_prefix')):
            program=[p for p in prefix['program'] if p[0]<split]+[p for p in suffix['program'] if p[0]>=split]
            result=price(states[1],events[1],program)
            if result['eligible']:
                splices.append(dict(split=split,prefix=label,program=program,**result))
    return dict(event_numbers=event_numbers,comparisons=rows,best_prior_extension=min(appended,key=lambda r:r['cost']),
        best_splice=min(splices,key=lambda r:r['cost']) if splices else None,
        best_four_day_prior_splice=min((r for r in splices if r['prefix']=='prior_prefix' and
            r['split']>=(states[1].context.decision+timedelta(days=4)).isoformat()),key=lambda r:r['cost'],default=None),
        volume_window=volume_window,purpose_terminal=purpose_terminal,steady_terminal=steady_terminal,taper_soft=taper_soft,
        snapshot_migrations=[{k:v for k,v in m.items() if k!='state'} for m in migrations],
        forecast_nominal_changed={k:events[1]['forecast_load_by_dose'][k]-v for k,v in events[0]['forecast_load_by_dose'].items()
                                  if v!=events[1]['forecast_load_by_dose'][k]})


if __name__=='__main__':
    p=ArgumentParser()
    p.add_argument('root',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--volume-window',type=int)
    p.add_argument('--events',type=int,nargs=2,default=(1,3))
    p.add_argument('--purpose-terminal',action='store_true')
    p.add_argument('--steady-terminal',action='store_true')
    p.add_argument('--taper-soft',action='store_true')
    a=p.parse_args()
    r=audit(a.root,volume_window=a.volume_window,event_numbers=a.events,
        purpose_terminal=a.purpose_terminal,steady_terminal=a.steady_terminal,taper_soft=a.taper_soft)
    with a.output.open('x') as stream:
        json.dump(r,stream,default=lambda v:v.isoformat(),indent=2)
    print(json.dumps(dict(comparisons={n:{k:v['cost'] for k,v in rows.items()} for n,rows in r['comparisons'].items()},
        best_prior_extension={k:r['best_prior_extension'][k] for k in ('dose','cost')},
        best_splice={k:r['best_splice'][k] for k in ('split','prefix','cost')} if r['best_splice'] else None,
        best_four_day_prior_splice={k:r['best_four_day_prior_splice'][k] for k in ('split','cost')} if r['best_four_day_prior_splice'] else None,
        forecast_differences=r['forecast_nominal_changed'])))
