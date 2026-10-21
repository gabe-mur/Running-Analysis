"""Bounded frozen-plan deferral contrasts; no search, uploads or model changes."""
from argparse import ArgumentParser
from dataclasses import replace
from datetime import date,datetime,time,timedelta
import json
from pathlib import Path
from time import perf_counter
from zoneinfo import ZoneInfo

from run_analysis.config import load_config
from run_analysis.continuous_training_dose import ContinuousDoseReference
from run_analysis.adaptive.decision_snapshot import evaluate_snapshot
from run_analysis.adaptive.new_model_daily_harness import planned_accounted
from run_analysis.adaptive.shared_program_evaluator import ProgramCandidate,evaluate_program
from scripts.replay_adaptive_search import load_decision
from scripts.research_manifest import freeze_sources


def volume_cost(evaluation,context):
    return (evaluation.parts['shortfall_by_purpose']['volume']*context.policy.purpose_weights[0]
            +evaluation.parts.get('volume_benefit',{}).get('added_volume_cost',0.))


def program_candidate(context,rows):
    return ProgramCandidate(tuple(r for r in rows if r.day<context.hard_end),
        tuple(r for r in rows if r.day>=context.hard_end),'unchanged archived physical calendar, explicit diagnostic')


def diagnose(root,index):
    case=load_decision(root,index)
    state,catalog,handoffs=(case[k] for k in ('state','catalog','handoffs'))
    context=state.context
    baseline=tuple(map(tuple,case['event']['program']))
    reference=ContinuousDoseReference('sandbox upload accounting')
    physical,cache={},{}
    started=perf_counter()
    def evaluate(program):
        program=tuple(sorted(program))
        if program not in cache:
            for day,key in program:
                if (day,key) not in physical:
                    physical[day,key]=planned_accounted(f'candidate:{day}:{key}',
                        datetime.combine(date.fromisoformat(day),time(18),ZoneInfo(state.timezone)),
                        catalog[key],handoffs[key],reference,available=state.as_of)
            result=evaluate_snapshot(state,tuple(physical[p] for p in program),tuple(k for _,k in program),reference=reference)
            e=result['evaluation']
            cache[program]=(dict(cost=e.total_cost,fatigue=e.weighted_demand_cost,
                volume=volume_cost(e,context),purpose=e.delivery_cost-volume_cost(e,context),
                short=e.short_run_cost,eligible=result['decision_eligible'],
                miles=sum(catalog[k].miles for _,k in program)),e)
        return cache[program]
    base,evaluate_base=evaluate(baseline)
    assert abs(base['cost']-case['event']['cost'])<1e-12
    origin=context.decision
    def offset(day):return (date.fromisoformat(day)-origin).days
    def contrast(program,**description):
        row,e=evaluate(program)
        assert abs(row['miles']-base['miles'])<1e-9
        return dict(description,**row,delta={k:row[k]-base[k] for k in ('cost','fatigue','volume','purpose','short')},
            program=tuple(sorted(program)))
    rows=[]
    # One-decimal transfers between existing easy sessions. Full calendar,
    # key sessions and total hard+tail distance are retained.
    early=[p for p in baseline if offset(p[0])<14 and catalog[p[1]].role=='easy']
    later=[p for p in baseline if offset(p[0])>=14 and catalog[p[1]].role=='easy']
    for day,key in early:
        for target,other in later:
            for amount in (.1,.5,1.):
                smaller=f'easy:{catalog[key].miles-amount:.1f}'
                larger=f'easy:{catalog[other].miles+amount:.1f}'
                if smaller not in handoffs or larger not in handoffs:continue
                program=tuple((d,smaller if d==day else larger if d==target else k) for d,k in baseline)
                rows.append(contrast(program,kind='equal_volume_dose_transfer',source=day,destination=target,
                    amount=amount,destination_scope='hard' if target<context.hard_end.isoformat() else 'soft'))
                forward=f'easy:{catalog[key].miles+amount:.1f}'
                reduced=f'easy:{catalog[other].miles-amount:.1f}'
                if forward in handoffs and reduced in handoffs:
                    program=tuple((d,forward if d==day else reduced if d==target else k) for d,k in baseline)
                    rows.append(contrast(program,kind='reverse_front_funding',source=target,destination=day,
                        amount=amount,destination_scope='hard'))
    # Identical prescriptions moved to a free later date: no altered HR,
    # distance, duration or forecast structure disguises the timing comparison.
    occupied={d for d,_ in baseline}
    free=[(origin+timedelta(days=n)).isoformat() for n in range(14,(context.end-origin).days)
        if (origin+timedelta(days=n)).isoformat() not in occupied]
    for day,key in early:
        for target in free:
            program=tuple((target if d==day else d,k) for d,k in baseline)
            rows.append(contrast(program,kind='identical_easy_moved',source=day,destination=target,
                amount=catalog[key].miles,destination_scope='hard' if target<context.hard_end.isoformat() else 'soft'))
    # Two balanced finite witnesses, not new optimized calendars or quotas:
    # restore the first 14-day dated integral using the same total distance by
    # extending existing early easies and reducing later ones. Donor order is
    # declared: hard continuation first versus reverse absolute-date order.
    stop=origin+timedelta(days=14)
    target14=sum(max(0,(min(stop,t.end)-max(origin,t.start)).days)*t.miles_per_7_days/7 for t in context.volume_targets)
    miles14=sum(catalog[k].miles for day,k in baseline if date.fromisoformat(day)<stop)
    steps=max(0,__import__('math').ceil((target14-miles14)*10-1e-9))
    for donor_order in ('earliest_later','latest_later'):
        program=dict(baseline)
        donors=sorted((d for d,_ in later),reverse=donor_order=='latest_later')
        receivers=sorted(d for d,_ in early)
        applied=0
        for n in range(steps):
            receiver=receivers[n%len(receivers)]
            up=f'easy:{catalog[program[receiver]].miles+.1:.1f}'
            candidates=[d for d in donors if f'easy:{catalog[program[d]].miles-.1:.1f}' in handoffs]
            if up not in handoffs or not candidates:break
            # Spread reductions across later sessions; do not shrink one donor
            # toward the floor merely because it sorts first.
            donor=candidates[n%len(candidates)]
            down=f'easy:{catalog[program[donor]].miles-.1:.1f}'
            program[receiver],program[donor]=up,down
            applied+=1
        witness=contrast(tuple(program.items()),kind='balanced_front_funding',source=donor_order,
            destination='first_14_days',amount=applied/10,destination_scope='hard')
        witness.update(first14_miles=miles14+applied/10,target14=target14,complete=applied==steps)
        rows.append(witness)
    groups={}
    for kind in ('equal_volume_dose_transfer','identical_easy_moved','reverse_front_funding','balanced_front_funding'):
        for scope in ('hard','soft'):
            members=[r for r in rows if r['kind']==kind and r['destination_scope']==scope and r['eligible']]
            groups[kind+':'+scope]=dict(eligible=len(members),better=int(sum(r['delta']['cost']<-1e-12 for r in members)),
                lower_volume_cost=int(sum(r['delta']['volume']<-1e-12 for r in members)),
                best=min(members,key=lambda r:r['cost']) if members else None)
    # Establish causal credit: moved work cannot change any rolling balance
    # before its ORIGINAL date, and cannot repay the intervening missing dose.
    witnesses=[]
    for kind in ('equal_volume_dose_transfer','identical_easy_moved'):
        members=[r for r in rows if r['kind']==kind and r['eligible']]
        if not members:continue
        row=min(members,key=lambda r:r['cost'])
        _,e=evaluate(row['program'])
        differences=[]
        for a,b in zip(evaluate_base.parts['rolling_volume']['daily'],e.parts['rolling_volume']['daily']):
            assert a['day']==b['day']
            delta=b['volume']['delivered']-a['volume']['delivered']
            if a['day']<date.fromisoformat(row['source']):assert abs(delta)<1e-12
            if date.fromisoformat(row['source'])<=a['day']<date.fromisoformat(row['destination']):
                assert delta<=1e-10
            if abs(delta)>1e-10:differences.append(dict(day=a['day'],delivered_delta=delta))
        witnesses.append(dict(kind=kind,source=row['source'],destination=row['destination'],
            first_affected_day=differences[0]['day'] if differences else None,
            intervening_missing_volume_not_repaid=True))
    # Pure expected-completion control. Consume only unchanged baseline work,
    # retain forecasts, fixed absolute endpoints and target path, and compare
    # a dose transfer whose source remains future for all seven frames.
    controls=[r for r in rows if r['kind']=='equal_volume_dose_transfer' and r['eligible'] and offset(r['source'])>=7]
    rollover=[]
    if controls:
        control=min(controls,key=lambda r:r['cost'])
        _,other_e=evaluate(control['program'])
        base_future=tuple(r for r in evaluate_base.conditional_ledger if r.source=='planned')
        other_future=tuple(r for r in other_e.conditional_ledger if r.source=='planned')
        observed=tuple(r for r in evaluate_base.conditional_ledger if r.source=='observed')
        for shift in range(7):
            decision=origin+timedelta(days=shift)
            consumed=tuple(replace(r,source='observed') for r in base_future if r.day<decision)
            # Snapshot contexts defer physical history to the adapter. Preserve
            # that adapter's actual ledger, not the empty context placeholder.
            c=replace(context,decision=decision,history=(*observed,*consumed))
            a=evaluate_program(c,program_candidate(c,tuple(r for r in base_future if r.day>=decision)))
            b=evaluate_program(c,program_candidate(c,tuple(r for r in other_future if r.day>=decision)))
            if shift==0:
                assert abs(a.total_cost-base['cost'])<1e-12
                assert abs(b.total_cost-control['cost'])<1e-12
            rollover.append(dict(shift_days=shift,source=control['source'],destination=control['destination'],
                total_delta=b.total_cost-a.total_cost,volume_delta=volume_cost(b,c)-volume_cost(a,c),
                fatigue_delta=b.weighted_demand_cost-a.weighted_demand_cost,
                winner='deferred' if b.total_cost<a.total_cost-1e-12 else 'original'))
    cumulative=[]
    for n in (7,14,21,28,40,54):
        stop=min(context.end,origin+timedelta(days=n))
        planned=sum(catalog[k].miles for d,k in baseline if date.fromisoformat(d)<stop)
        target=sum(max(0,(min(stop,t.end)-max(origin,t.start)).days)*t.miles_per_7_days/7 for t in context.volume_targets)
        cumulative.append(dict(days=(stop-origin).days,miles=planned,target=target,ratio=planned/target))
    return dict(index=index,as_of=state.as_of,snapshot=state.fingerprint,baseline=base,cumulative=cumulative,
        groups=groups,causal_controls=witnesses,expected_completion_fixed_endpoint=rollover,
        comparisons=rows,evaluations=len(cache),seconds=perf_counter()-started,
        limitations=['Frozen forecast and history, not a global search or daily simulation',
            'Equal-volume dose transfers can alter per-mile forecasts; identical moves isolate timing',
            'Expected-completion control is scoring arithmetic only, not new observed HR or decision admission',
            'Continuation handoffs remain modeled mean delivery, not executed future training',
            'Raw calendar-window target ratios are diagnostic, not the rolling selection objective'])


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--events',type=int,nargs='+',default=[1,4])
    args=parser.parse_args();args.output.mkdir(exist_ok=False)
    inputs=[args.root/'scenario.json']
    for index in args.events:
        p=args.root/f'decision-{index:04d}.json'
        inputs.extend((p,args.root/json.loads(p.read_text())['snapshot_file']))
    manifest=freeze_sources(args.output/'sources.zip',config=load_config(),entrypoint=__name__,inputs=inputs)
    results=[]
    for index in args.events:
        result=diagnose(args.root,index);results.append(result)
        print(json.dumps(dict(index=index,evaluations=result['evaluations'],seconds=result['seconds'],
            groups={k:{p:v for p,v in r.items() if p!='best'} for k,r in result['groups'].items()})),flush=True)
    with (args.output/'report.json').open('x') as stream:
        json.dump(dict(manifest=manifest,results=results,model_changed=False,live_changed=False),stream,indent=2,default=str,allow_nan=False)


if __name__=='__main__':main()
