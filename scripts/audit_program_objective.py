"""Bounded objective experiment; read-only original history, never live selection."""
from argparse import ArgumentParser
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime, timedelta
from itertools import product
from pathlib import Path
from statistics import median
from time import perf_counter
from zoneinfo import ZoneInfo
import hashlib
import json

from scripts.audit_recency_demand import load_evidence, json_ready
from scripts.audit_easy_distribution import program_option
from scripts.program_objective_trial import (TrialSession, TrialPolicy, DemandScenario,
    components, credits, describe, exchange_region, score, PURPOSES)
from run_analysis.config import load_config
from run_analysis.workload_accounting import Quantity
from run_analysis.program_constraints import assess_program_constraints
from run_analysis.program_delivery import ProgramExposure
from run_analysis.easy_dose import AerobicMinimum, EasyDurationReference, assess_easy_dose
import numpy as np


# Frozen before inspecting winners. Q is the same 3x5-minute benchmark throughout,
# not a claim that the user's recent 400 m repetitions were threshold work.
MOTIFS = {
    'A': (7,[(0,8.4,'long'),(3,4.5,'quality'),(5,4.4,'easy')]),
    'B': (7,[(0,8.4,'long'),(1,2.2,'easy'),(3,4.5,'quality'),(5,2.2,'easy')]),
    'C': (7,[(0,8.4,'long'),(1,3.,'easy'),(3,4.5,'quality'),(5,3.5,'easy')]),
    'D': (8,[(0,8.4,'long'),(1,4.4,'easy'),(4,4.5,'quality'),(6,4.4,'easy')]),
    'B_shifted': (7,[(0,8.4,'long'),(1,2.2,'easy'),(3,4.5,'quality'),(4,2.2,'easy')]),
    'C_compressed': (7,[(0,8.4,'long'),(1,3.,'easy'),(2,4.5,'quality'),(3,3.5,'easy')]),
    'C_fragmented': (7,[(0,8.4,'long'),(1,2.2,'easy'),(2,2.2,'easy'),(3,4.5,'quality'),(5,2.1,'easy')]),
}


def calendar(name, start, days, rates):
    period,motif = MOTIFS[name]
    result = []
    for offset in range(days):
        for slot,miles,role in motif:
            if offset % period != slot:
                continue
            demand = tuple(rates[role][c]['bounds'].scale(miles) for c in ('running_minutes','cardio'))
            result.append(TrialSession(f'{name}:{offset}',start+timedelta(days=offset),miles,role,
                (*demand,Quantity.point(15. if role=='quality' else 0.,'prescribed_work_minutes')),
                1. if role=='quality' else 0.))
    return tuple(result)


def policies():
    weights = [(1.,1.,1.,1.),*((tuple(2. if i==j else 1. for i in range(4))) for j in range(4))]
    return tuple(TrialPolicy(weekly_miles=m,ordinary_power=p,purpose_weights=w)
        for m,p,w in product((17.5,18.5,19.5),(1.,2.),weights))


def scenarios():
    return tuple(DemandScenario(memory,bound,timing,axes) for memory,bound,timing,axes in
        product((1.,2.,4.),(0.,.5,1.),(0.,.5),((1.,1.,1.),(0.,1.,1.),(1.,0.,1.),(1.,1.,0.))))


def with_delivery(parts, program, start, end, policy):
    # Reuse expensive demand calculation: these policies change only benefit.
    span = (end-start).days
    delivered = sum((credits(e,policy) for e in program if start <= e.day < end),np.zeros(4))
    targets = np.array((policy.weekly_miles,policy.ordinary_per_7,policy.long_per_7,policy.quality_per_7))*span/7
    losses = np.maximum(0,1-delivered/targets)**2
    return dict(parts,delivery=dict(zip(PURPOSES,delivered.tolist())),
        shortfall_by_purpose=dict(zip(PURPOSES,losses.tolist())),delivery_loss=float(np.dot(losses,policy.purpose_weights)))


def audit(database, config, as_of):
    started = perf_counter()
    digest = hashlib.sha256(database.read_bytes()).hexdigest()
    rows,unresolved,raw_history,rates = load_evidence(database,config,as_of)
    today = as_of.astimezone(ZoneInfo(config['timezone_default'])).date()
    start = max(today,max(r['day'] for r in rows)+timedelta(days=1))
    end = start+timedelta(days=56)
    if any(rates[role][axis]['bounds'] is None for role in ('easy','long','quality') for axis in ('running_minutes','cardio')):
        raise ValueError('Declare missing future-rate scenarios before scoring; missing cannot be zero')
    history = tuple(TrialSession('actual:'+r['key'],r['day'],r['miles'],r['role'],
        tuple(r['quantities'][c] for c in ('running_minutes','cardio','work_minutes')),0.,'observed') for r in rows)
    recent_easy = [r for r in rows if r['eligible'] and r['role']=='easy' and r['start'] >= as_of-timedelta(days=28)]
    # Frozen median per-mile demand at 4.4 miles, NOT tolerated ceilings.
    scales = tuple(4.4*median((r['quantities'][c].low+r['quantities'][c].high)/2/r['miles'] for r in recent_easy)
        for c in ('running_minutes','cardio'))+(15.,)
    plans = {k:calendar(k,start,56,rates) for k in MOTIFS}
    policy,scenario = TrialPolicy(),DemandScenario()
    kwargs = dict(stage_start=start,decision=start,end=end,scales=scales,policy=policy)
    # Controls are hypothetical contrasts with no adverse context, not assertions
    # that the real athlete is currently cleared for these sessions.
    funded = plans['C']
    controls = dict(funded=funded,rest=(),
        underfunded=tuple(replace(e,miles=round(e.miles*.6,1),
            demand=tuple(q.scale(round(e.miles*.6,1)/e.miles if i<2 else .6) for i,q in enumerate(e.demand)),
            quality_credit=e.quality_credit*.6) for e in funded),
        missing_easy=tuple(e for e in funded if not (e.role=='easy' and (e.day-start).days%7==5)),
        compressed=plans['C_compressed'])
    policy_grid,scenario_grid = policies(),scenarios()
    contrasts = []
    for si,s in enumerate(scenario_grid):
        values = {name:components((),p,**kwargs,scenario=s) for name,p in controls.items()}
        for pi,pol in enumerate(policy_grid):
            adjusted = {name:with_delivery(values[name],p,start,end,pol) for name,p in controls.items()}
            for other in ('rest','underfunded','missing_easy','compressed'):
                contrasts.append((f'{si}:{pi}:funded_over_{other}',adjusted['funded'],adjusted[other]))
    region = exchange_region(contrasts)
    if not region['feasible'] or region['nominal_exchange'] is None:
        return dict(status='stop_no_bounded_exchange_region',region=region,elapsed_seconds=perf_counter()-started)
    exchange = region['nominal_exchange']
    nominal = {k:components(history,p,**kwargs,scenario=scenario) for k,p in plans.items()}
    results = {}
    serial_rates = json.loads(json.dumps(rates,default=json_ready))
    # Previously frozen original-file startup evidence, not a new fitted cutoff.
    entry = json.loads(Path('output/aerobic-entry-panel-20260925.json').read_text())
    startup = entry['summaries']['84']['easy']['variants']['5min_80pct']['uncensored_sustained_entry_minutes']['q75']
    easy_reference = EasyDurationReference(as_of,startup,scales[0]/4.4,
        'frozen_20260925_original_84d_startup_q75;current_28d_median_pace')
    easy_minimum = AerobicMinimum(5.,'user_selected_provisional_minimum_not_optimum')
    common_tail = tuple(e for e in plans['B'] if e.day >= start+timedelta(days=21))
    for k,p in plans.items():
        prefix = tuple(e for e in p if e.day < start+timedelta(days=21))
        short = components(history,prefix,**dict(kwargs,end=start+timedelta(days=21)),scenario=scenario)
        common = components(history,(*prefix,*common_tail),**kwargs,scenario=scenario)
        external = tuple(ProgramExposure(e.key,e.day,e.miles,e.role,'planned',
            2. if e.role=='quality' else e.miles,'threshold' if e.role=='quality' else None,
            (300.,)*3 if e.role=='quality' else None) for e in p)
        physical = program_option(k,external,raw_history,serial_rates)
        constraints = assess_program_constraints(physical.demands,raw_history,as_of=as_of,
            timezone=config['timezone_default'],plan_complete=True)
        floors = [dict(key=e.key,**assess_easy_dose(e.miles,easy_reference,easy_minimum)) for e in p if e.role=='easy']
        extended = components(history,calendar(k,start,112,rates),
            **dict(kwargs,end=start+timedelta(days=112)),scenario=scenario)
        results[k] = dict(score=score(nominal[k],exchange),parts=nominal[k],
            prefix_21_score=score(short,exchange),common_tail_score=score(common,exchange),
            common_tail_parts=common,extension_112_score=score(extended,exchange),
            checked_constraints=constraints,easy_floor_checks=floors,
            description_21=describe(p,history,start,21),description_56=describe(p,history,start,56),
            calendar=[dict(key=e.key,day=e.day,miles=e.miles,role=e.role) for e in p])
    winners,extended_winners,examples = Counter(),Counter(),[]
    by_volume,by_power = {},{}
    exchange_values = [region['upper_exclusive']*fraction for fraction in (.1,.5,.9)]
    for si,s in enumerate(scenario_grid):
        demands = {k:components(history,p,**kwargs,scenario=s) for k,p in plans.items()}
        for pi,pol in enumerate(policy_grid):
            values = {k:with_delivery(demands[k],p,start,end,pol) for k,p in plans.items()}
            for lam in exchange_values:
                scored = {k:score(v,lam) for k,v in values.items()}
                winner = min('ABCD',key=lambda k:(scored[k],k))
                extended = min(scored,key=lambda k:(scored[k],k))
                winners[winner]+=1
                extended_winners[extended]+=1
                by_volume.setdefault(str(pol.weekly_miles),Counter())[winner]+=1
                by_power.setdefault(str(pol.ordinary_power),Counter())[winner]+=1
                if not any(r['winner']==winner for r in examples):
                    examples.append(dict(winner=winner,scenario=asdict(s),policy=asdict(pol),exchange=lam,scores=scored))
    after = hashlib.sha256(database.read_bytes()).hexdigest()
    if after != digest:
        raise RuntimeError('Database changed during audit; results require review')
    return dict(status='offline_comparison_not_selection',as_of=as_of,start=start,end=end,
        history_count=len(history),history_keys=[e.key for e in history],unresolved=unresolved,
        database_unchanged=True,rates=rates,scales=dict(zip(('running_minutes','cardio','work_minutes'),scales)),
        nominal_policy=policy,nominal_scenario=scenario,nominal_exchange=exchange,
        region=region,results=results,sensitivity_winners=dict(winners),
        sensitivity_winners_with_neighbors=dict(extended_winners),winner_examples=examples,
        winners_by_volume={k:dict(v) for k,v in by_volume.items()},
        winners_by_ordinary_credit_power={k:dict(v) for k,v in by_power.items()},
        sensitivity_cases=sum(winners.values()),elapsed_seconds=perf_counter()-started,
        planner_influence=False,recovery_cleared=False,selected_candidate=None,
        limitations=['Role per-mile HR/duration extrema are scenarios, not forecast intervals',
            'Unknown historical fast work bounded by duration; upper case can be very pessimistic',
            'Work minutes are structured exposure, not measured muscular fatigue or intensity equivalence',
            'Fixed 56-day stage, not daily search; 21-day prefix score is boundary diagnostic only',
            'No weather/readiness/rest restrictions loaded; results are not prescriptions',
            'Ordinary credit, cadence targets, kernels and exchange remain explicit policy hypotheses',
            'No score is evidence of physiological superiority or future adaptation'])


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--database',type=Path,default=Path('data/run_analysis.sqlite'))
    parser.add_argument('--config',type=Path,default=Path('config.yaml'))
    parser.add_argument('--as-of',type=datetime.fromisoformat,default=datetime.fromisoformat('2026-09-26T04:00:00+00:00'))
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new result path')
    result = audit(args.database,load_config(args.config),args.as_of)
    with args.output.open('x') as stream:
        json.dump(result,stream,default=json_ready,allow_nan=False,indent=2)
        stream.write('\n')
    print(json.dumps({k:result[k] for k in ('status','elapsed_seconds')},indent=2))
    if 'results' in result:
        print(json.dumps(dict(scores={k:v['score'] for k,v in result['results'].items()},
            exchange=result['nominal_exchange'],winners=result['sensitivity_winners'],
            neighbors=result['sensitivity_winners_with_neighbors']),indent=2))
