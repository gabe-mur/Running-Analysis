"""Search equal-purpose volume slices before integrating any live scheduler."""
from argparse import ArgumentParser
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from statistics import median
from time import perf_counter
from zoneinfo import ZoneInfo
import hashlib
import json

import numpy as np

from scripts.audit_recency_demand import load_evidence, json_ready
from scripts.audit_program_objective import calendar
from scripts.audit_easy_distribution import program_option
from scripts.program_objective_trial import TrialSession, TrialPolicy, DemandScenario, components, describe
from scripts.volume_demand_frontier import PatternCost, LONG, QUALITY, search, exhaustive_small
from run_analysis.easy_dose import AerobicMinimum, EasyDurationReference, minimum_easy_miles
from run_analysis.config import load_config
from run_analysis.program_delivery import ProgramExposure
from run_analysis.program_constraints import assess_program_constraints
from run_analysis.workload_accounting import Quantity


def materialize(state, start, cycles, rates, name='candidate'):
    rows=[]
    for day in range(len(state)*cycles):
        value=int(state[day%len(state)])
        if value==0:
            continue
        role,miles=('long',8.4) if value==LONG else ('quality',4.5) if value==QUALITY else ('easy',value/10)
        demand=tuple(rates[role][axis]['bounds'].scale(miles) for axis in ('running_minutes','cardio'))
        rows.append(TrialSession(f'{name}:{day}',start+timedelta(days=day),miles,role,
            (*demand,Quantity.point(15. if role=='quality' else 0.,'fixed_3x5_work_minutes')),
            1. if role=='quality' else 0.))
    return tuple(rows)


def prior_pattern(name,start,rates):
    rows=calendar(name,start,21,rates)
    state=np.zeros(21,dtype=np.int16)
    for e in rows:
        state[(e.day-start).days]=LONG if e.role=='long' else QUALITY if e.role=='quality' else round(e.miles*10)
    return state


def audit(database,config,as_of,*,volumes=(17.,18.,19.,20.,21.,22.),phase_count=7):
    started=perf_counter()
    digest=hashlib.sha256(database.read_bytes()).hexdigest()
    rows,unresolved,raw_history,rates=load_evidence(database,config,as_of)
    start=max(as_of.astimezone(ZoneInfo(config['timezone_default'])).date(),max(r['day'] for r in rows)+timedelta(days=1))
    history=tuple(TrialSession('actual:'+r['key'],r['day'],r['miles'],r['role'],
        tuple(r['quantities'][c] for c in ('running_minutes','cardio','work_minutes')),0.,'observed') for r in rows)
    recent=[r for r in rows if r['eligible'] and r['role']=='easy' and r['start']>=as_of-timedelta(days=28)]
    scales=np.array([4.4*median((r['quantities'][c].low+r['quantities'][c].high)/2/r['miles'] for r in recent)
        for c in ('running_minutes','cardio')]+[15.])
    entry=json.loads(Path('output/aerobic-entry-panel-20260925.json').read_text())
    startup=entry['summaries']['84']['easy']['variants']['5min_80pct']['uncensored_sustained_entry_minutes']['q75']
    reference=EasyDurationReference(as_of,startup,scales[0]/4.4,'previous_original_startup_q75_current_pace')
    floor=round(minimum_easy_miles(reference,AerobicMinimum(5.,'user_provisional_floor'))*10)
    def engine(scenario,period=21,cycles=3):
        vectors=[]
        for role,miles in (('easy',1.),('long',8.4),('quality',4.5)):
            quantities=[rates[role][c]['bounds'] for c in ('running_minutes','cardio')]
            if any(q is None for q in quantities):
                raise ValueError('Explicit nonzero/unknown-rate scenarios required')
            vectors.append(np.array([miles*(q.low+(q.high-q.low)*scenario.bound_fraction) for q in quantities]
                +[15. if role=='quality' else 0.])/scales)
        preceding=[((start-e.day).days,np.array([q.low+(q.high-q.low)*scenario.bound_fraction for q in e.demand])/scales)
            for e in history]
        return PatternCost(period,cycles,vectors,preceding,memory_days=scenario.memory_days,
            timing_compression_days=scenario.timing_compression_days,axis_weights=scenario.axis_weights)
    # Small-domain exact optimum compared with the SAME heuristic, not oracle-seeded.
    oracle_checks=[]
    for memory in (1.,2.,4.):
        toy=engine(DemandScenario(memory),7,3)
        exact=exhaustive_small(toy)
        for budget in (40,60,80,100):
            result=search(toy,budget,floor=20,ceiling=30,step=10)
            gap=result['cost']-exact['by_easy_tenths'][budget]['cost']
            oracle_checks.append(dict(memory_days=memory,easy_miles=budget/10,gap=gap,
                candidates=exact['total_candidates'],evaluated=result['evaluated_neighbors']))
            if gap>1e-9 or result['budget_exhausted_starts']:
                return dict(status='stop_small_domain_search_gap',oracle_checks=oracle_checks,
                    elapsed_seconds=perf_counter()-started,planner_influence=False)
    nominal=DemandScenario()
    scenarios={'nominal':nominal,'short_memory':DemandScenario(1.),'long_memory':DemandScenario(4.)}
    engines={name:engine(s) for name,s in scenarios.items()}
    results={}
    serial_rates=json.loads(json.dumps(rates,default=json_ready))
    for name,s in scenarios.items():
        case=[]
        for volume in volumes:
            target_tenths=round(volume*30)
            if abs(volume*30-target_tenths)>1e-8:
                raise ValueError('21-day target must resolve to one decimal')
            budget=target_tenths-3*(84+45)
            result=search(engines[name],budget,floor=floor,ceiling=60,phase_count=phase_count)
            plan=materialize(result['state'],start,3,rates)
            full=components(history,plan,stage_start=start,decision=start,end=start+timedelta(days=63),
                scales=scales,policy=TrialPolicy(),scenario=s)
            if abs(full['demand_cost']-result['cost'])>1e-9:
                raise AssertionError('Fast search/publication scorer mismatch')
            exposures=tuple(ProgramExposure(e.key,e.day,e.miles,e.role,'planned',
                2. if e.role=='quality' else e.miles,'threshold' if e.role=='quality' else None,
                (300.,)*3 if e.role=='quality' else None) for e in plan)
            physical=program_option('frontier',exposures,raw_history,serial_rates)
            guard=assess_program_constraints(physical.demands,raw_history,as_of=as_of,
                timezone=config['timezone_default'],plan_complete=True)
            if guard['status']!='passes_checked_constraints_conditionally':
                raise AssertionError('Candidate needs independent distance-guard review')
            first=tuple(e for e in plan if e.day<start+timedelta(days=21))
            easy=[e.miles for e in first if e.role=='easy']
            repeating_gaps={}
            for role in ('long','quality'):
                dates=[e.day for e in plan if e.role==role]
                repeating_gaps[role]=[(b-a).days for a,b in zip(dates,dates[1:])]
            result.update(weekly_equivalent_miles=volume,description_21=describe(plan,history,start,21),
                description_63=describe(plan,history,start,63),easy_distances=easy,
                key_gaps_with_continuation=repeating_gaps,checked_distance_guard=guard['status'],
                main_calendar=[dict(day=e.day,role=e.role,miles=e.miles) for e in first],
                demand_axes=full['demand_by_axis'],minimum_easy_miles=floor/10,
                rescored_under_other_memories={other:float(cost.batch([result['state']])[0]) for other,cost in engines.items()})
            result['first_key_gaps_from_actual_history']={role:dict(
                last_actual=max(e.day for e in history if e.role==role),
                first_planned=min(e.day for e in plan if e.role==role),
                days=(min(e.day for e in plan if e.role==role)-max(e.day for e in history if e.role==role)).days)
                for role in ('long','quality') if any(e.role==role for e in history)}
            case.append(result)
            print(json.dumps(dict(progress=name,volume=volume,cost=result['cost'],
                runs=len(first),evaluated=result['evaluated_neighbors'],seconds=result['elapsed_seconds'])),flush=True)
        for i,row in enumerate(case):
            row['marginal_cost_per_extra_weekly_mile']=None if i==0 else (
                row['cost']-case[i-1]['cost'])/(row['weekly_equivalent_miles']-case[i-1]['weekly_equivalent_miles'])
            row['dominated_by_higher_volume']=[r['weekly_equivalent_miles'] for r in case
                if r['weekly_equivalent_miles']>row['weekly_equivalent_miles'] and r['cost']<=row['cost']]
        results[name]=case
    # A/B/C comparison uses exact matched volume; D intentionally excluded because
    # its eight-day cadence delivers less key work over an aligned full horizon.
    baselines={}
    for label,volume in (('A',17.3),('B',17.3),('C',19.4)):
        baseline=prior_pattern(label,start,rates)
        result=search(engines['nominal'],round(volume*30)-387,floor=floor,ceiling=60,phase_count=phase_count)
        baseline_cost=float(engines['nominal'].batch([baseline])[0])
        fixed=np.where(baseline<0,baseline,0)
        fixed_result=search(engines['nominal'],round(volume*30)-387,floor=floor,ceiling=60,
            phase_count=phase_count,fixed_keys=fixed)
        longer_easy=search(engines['nominal'],round(volume*30)-387,floor=30,ceiling=60,phase_count=phase_count)
        baselines[label]=dict(weekly_miles=volume,baseline_demand=baseline_cost,best_found=result,
            reduction_fraction=1-result['cost']/baseline_cost,
            same_key_dates=fixed_result,same_key_dates_reduction_fraction=1-fixed_result['cost']/baseline_cost,
            exploratory_3_mile_easy_minimum=longer_easy,
            longer_easy_cost_premium_fraction=longer_easy['cost']/result['cost']-1)
    after=hashlib.sha256(database.read_bytes()).hexdigest()
    if after!=digest:
        raise RuntimeError('Database changed during read-only audit')
    return dict(status='best_found_conditional_frontiers_not_scheduler',as_of=as_of,start=start,
        period_days=21,evaluation_days=63,history_count=len(history),unresolved=unresolved,
        rates=rates,scales=scales.tolist(),floor_miles=floor/10,easy_ceiling_miles=6.,
        scenarios={k:asdict(s) for k,s in scenarios.items()},oracle_checks=oracle_checks,
        frontiers=results,matched_original_examples=baselines,
        database_unchanged=True,elapsed_seconds=perf_counter()-started,planner_influence=False,
        limitations=['No global optimality proof for 21-day tenth-mile mixed-discrete search',
            'Repeating 21-day pattern is a bounded search domain, not a coaching rule',
            'Three long/three quality opportunities per 21 days fixed for comparability, not validated cadence',
            'Convex dose overlap can favor fragmentation; minimum aerobic floor is not optimal dose',
            'No ordinary-credit formula, continuity penalty, benefit weight or progression approval used',
            'Per-mile role rates and 1/2/4-day memory are scenarios, not calibrated recovery',
            'Only date-level timing; no HRV, current readiness, reserved dates or weather integrated',
            'Fixed original history and conditional continued training, not daily/upload scheduler replay'])


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--database',type=Path,default=Path('data/run_analysis.sqlite'))
    parser.add_argument('--config',type=Path,default=Path('config.yaml'))
    parser.add_argument('--as-of',type=datetime.fromisoformat,default=datetime.fromisoformat('2026-09-26T04:00:00+00:00'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output path')
    result=audit(args.database,load_config(args.config),args.as_of)
    with args.output.open('x') as stream:
        json.dump(result,stream,default=json_ready,allow_nan=False,indent=2); stream.write('\n')
    print(json.dumps(dict(status=result['status'],elapsed_seconds=result['elapsed_seconds']),indent=2))
