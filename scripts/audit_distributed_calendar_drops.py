"""Frozen search coverage: remove one easy slot, redistribute to two others.

Same catalogue, tenth-mile totals, full scorer and admission. No live changes,
future-winner input, coaching coefficient or stability preference.
"""
from argparse import ArgumentParser
from datetime import date
from itertools import combinations
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scripts.replay_adaptive_search import load_decision
from run_analysis.adaptive.compiled_calendar_doses import compile_calendar,refine


def run(root,index,*,max_full_evaluations=48):
    case=load_decision(root,index)
    original=tuple(map(tuple,case['event']['program']))
    started=perf_counter();checks=0;cache={};raws={}
    def checked(p):
        nonlocal checks
        p=tuple(sorted(p))
        if p not in cache:
            if checks>=max_full_evaluations:raise ValueError('Frozen full-check allowance exceeded')
            raw=case['evaluate_details'](p);checks+=1;raws[p]=raw
            # Replay's public admission also includes harness funding checks.
            result=case['evaluate'](p)
            cache[p]=result
        return cache[p]
    baseline=checked(original);proposals=[];calendar_count=0;compiled_rows=0
    for removed,key in original:
        if case['catalog'][key].role!='easy':continue
        if checks>=max_full_evaluations//2:break
        base=tuple(p for p in original if p!=(removed,key))
        checked(base);compiled=compile_calendar(case['state'],base,physical=case['physical'],result=raws[base])
        calendar_count+=1
        positions=[i for i,d in enumerate(compiled.days)
            if abs((date.fromisoformat(d)-date.fromisoformat(removed)).days)<=10]
        units=np.rint(compiled.miles*10).astype(int);lookup={v:i for i,v in enumerate(units)}
        lost=round(case['catalog'][key].miles*10);x=compiled.original_levels
        allocations={}
        for a,b in combinations(positions,2):
            for first in range(lost+1):
                ua=units[x[a]]+first;ub=units[x[b]]+lost-first
                if ua not in lookup or ub not in lookup:continue
                proposed=x.copy();proposed[a]=lookup[ua];proposed[b]=lookup[ub]
                allocations.setdefault(tuple(proposed),proposed)
        if not allocations:continue
        values=np.array(list(allocations.values()));costs=compiled.cost(values)
        funding=compiled.funded(values);compiled_rows+=len(values)
        ordering=np.argsort(np.where(funding,costs,np.inf),kind='stable')[:2]
        for j in ordering:
            if np.isfinite(costs[j]) and funding[j]:
                proposals.append((float(costs[j]),compiled.physical(values[j]),removed))
    winner=original;results=[]
    for predicted,p,removed in sorted(proposals,key=lambda r:(r[0],r[1])):
        if checks>=max_full_evaluations-1:break
        full=checked(p)
        if abs(full['cost']-predicted)>1e-10:raise ValueError('Distributed compiled score mismatch')
        if not full['eligible']:continue
        remaining=max_full_evaluations-checks
        compiled=compile_calendar(case['state'],p,physical=case['physical'],result=raws[p])
        polished=refine(compiled,checked,max_full_evaluations=min(4,remaining+1))
        q=polished['program']
        if checked(q)['cost']<cache[winner]['cost']:winner=q
        results.append(dict(removed=removed,initial_cost=full['cost'],**polished))
    return dict(original_cost=baseline['cost'],cost=checked(winner)['cost'],program=winner,
        seconds=perf_counter()-started,calendar_count=calendar_count,compiled_rows=compiled_rows,
        unique_full_candidates=checks,max_unique_full_candidates=max_full_evaluations,
        actual_full_score_calls=2*checks,results=results,live_changed=False,
        scope='Frozen bounded three-to-two calendar coverage diagnostic; not global optimum or daily stability')


if __name__=='__main__':
    parser=ArgumentParser();parser.add_argument('root',type=Path);parser.add_argument('--index',type=int,default=1)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=run(args.root,args.index)
    with args.output.open('x') as stream:json.dump(result,stream,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='results'}))
