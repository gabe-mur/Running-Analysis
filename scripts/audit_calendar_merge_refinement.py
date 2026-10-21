"""Frozen merge/calendar+distance coverage diagnostic; no live publication."""
from argparse import ArgumentParser
import json
from pathlib import Path
from time import perf_counter

from scripts.replay_adaptive_search import load_decision
from run_analysis.adaptive.decimal_dose_search import decimal_merge_neighbors
from run_analysis.adaptive.compiled_calendar_doses import compile_calendar, refine


def run(root,index,*,limit=16,full_allowance=8):
    case=load_decision(root,index)
    original=tuple(map(tuple,case['event']['program']))
    roles={k:d.role for k,d in case['catalog'].items()}
    miles={k:d.miles for k,d in case['catalog'].items()}
    positions={d:i for i,d in enumerate(case['dates'])}
    candidates=[]
    start=perf_counter()
    for p in decimal_merge_neighbors(original,roles,miles,index=positions,radius=10):
        removed=set(dict(original))-set(dict(p))
        if not any(positions[d]<7 for d in removed):
            continue
        raw=case['evaluate_details'](p)
        if not raw['decision_eligible']:
            continue
        candidates.append((p,raw))
        if len(candidates)>=limit:
            break
    results=[]
    for p,raw in candidates:
        compiled=compile_calendar(case['state'],p,physical=case['physical'],result=raw)
        polished=refine(compiled,case['evaluate'],max_full_evaluations=full_allowance)
        results.append(dict(polished,initial_merge_cost=raw['evaluation'].total_cost))
    winner=min(results,key=lambda r:r['cost'],default=None)
    return dict(source=str(root),index=index,original_cost=case['archived_cost'],
        seconds=perf_counter()-start,candidates=len(candidates),results=results,winner=winner,
        live_changed=False,scope='Frozen witnessed neighborhood only; no daily stability/global optimum claim')


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--index',type=int,default=1)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    report=run(a.root,a.index)
    with a.output.open('x') as stream:
        json.dump(report,stream,indent=2)
    print(json.dumps({k:report[k] for k in ('original_cost','seconds','candidates')}))
    if report['winner']:
        print(json.dumps({k:report['winner'][k] for k in ('cost','program','full_evaluations')}))
