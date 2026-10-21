"""Frozen calendar moves followed by exact easy-dose allocation.

No later winner, new physiology, continuity reward or live publication. Tests
whether rebuilding distances independently hides useful calendar neighbours.
"""
from argparse import ArgumentParser
from datetime import date, timedelta
import json
from pathlib import Path
from time import perf_counter

from scripts.replay_adaptive_search import load_decision
from run_analysis.adaptive.compiled_calendar_doses import compile_calendar, refine


def run(root, index, *, max_unique_checks=240, radius=2):
    case=load_decision(root,index)
    original=tuple(map(tuple,case['event']['program']))
    started=perf_counter();cache={};raws={};checks=0
    def checked(program):
        nonlocal checks
        program=tuple(sorted(program))
        if program not in cache:
            if checks>=max_unique_checks:
                raise ValueError('Explicit frozen-check allowance exceeded')
            raws[program]=case['evaluate_details'](program)
            cache[program]=case['evaluate'](program)
            checks+=1
        return cache[program]
    baseline=checked(original);winner=original;traces=[];complete=True
    occupied={d for d,k in original}
    for old,key in original:
        for delta in range(-radius,radius+1):
            new=(date.fromisoformat(old)+timedelta(days=delta)).isoformat()
            if not delta or new in occupied or (new,key) not in case['physical']:
                continue
            if max_unique_checks-checks<6:
                complete=False;break
            candidate=tuple(sorted((new,key) if (d,k)==(old,key) else (d,k) for d,k in original))
            initial=checked(candidate)
            if not initial['eligible']:
                continue
            compiled=compile_calendar(case['state'],candidate,physical=case['physical'],result=raws[candidate])
            polished=refine(compiled,checked,max_full_evaluations=5)
            proposed=polished['program'];result=checked(proposed)
            if result['eligible'] and result['cost']<cache[winner]['cost']:
                winner=proposed
            traces.append(dict(old=old,new=new,role=case['catalog'][key].role,
                initial_cost=initial['cost'],**polished))
        if not complete:break
    return dict(original_cost=baseline['cost'],cost=checked(winner)['cost'],program=winner,
        seconds=perf_counter()-started,unique_full_candidates=checks,
        actual_full_score_calls=2*checks,enumeration_complete=complete,
        max_unique_checks=max_unique_checks,radius=radius,results=traces,live_changed=False,
        scope='Frozen single calendar moves and fixed easy-dose allocation; no global optimum or daily stability')


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path);parser.add_argument('--index',type=int,default=1)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=run(args.root,args.index)
    with args.output.open('x') as stream:json.dump(result,stream,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='results'}))
