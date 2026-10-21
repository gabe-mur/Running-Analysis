"""Frozen full-calendar catalogue refinement; no daily replay or publication."""
from argparse import ArgumentParser
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from scripts.replay_adaptive_search import load_decision
from scripts.research_manifest import freeze_sources
from run_analysis.adaptive.compiled_calendar_doses import compile_calendar, refine


def run(root, *, before, after, output):
    output.mkdir(parents=True,exist_ok=False)
    cases=[]
    for index in (before,after):
        case=load_decision(root,index)
        p=tuple(tuple(p) for p in case['event']['program'])
        stamp=perf_counter()
        compiled=compile_calendar(case['state'],p,physical=case['physical'],result=case['evaluate_details'](p))
        probes=[compiled.original_levels,np.zeros(len(compiled.days),dtype=int),
            np.full(len(compiled.days),len(compiled.keys)-1,dtype=int)]
        probes.extend(np.array([(n*11+i*7)%len(compiled.keys) for i in range(len(compiled.days))]) for n in range(32))
        errors=[]
        for x in probes:
            result=case['evaluate'](compiled.physical(x))
            errors.append(abs(result['cost']-compiled.cost(x)))
            if result['eligible'] and not compiled.funded(x):
                raise ValueError('Fast checkpoint check rejected an admitted full-score probe')
        if max(errors)>1e-10:
            raise ValueError('Exact catalogue compilation validation failed')
        validation_seconds=perf_counter()-stamp
        result=refine(compiled,case['evaluate'])
        cases.append(dict(index=index,as_of=case['event']['as_of'],snapshot=case['state'].fingerprint,
            validation_probes=len(probes),maximum_score_error=max(errors),validation_seconds=validation_seconds,
            original_program=p,**result))
    first,second=(dict(c['program']) for c in cases)
    original_first,original_second=(dict(c['original_program']) for c in cases)
    day=load_decision(root,after)['state'].context.decision.isoformat()
    from datetime import date,timedelta
    stop=(date.fromisoformat(day)+timedelta(days=7)).isoformat()
    comparisons=[dict(day=d,before=first.get(d),after=second.get(d),
        archived_before=original_first.get(d),archived_after=original_second.get(d))
        for d in sorted(first.keys()|second.keys()) if day<=d<stop]
    cross_starts=[]
    for i,index in enumerate((before,after)):
        case=load_decision(root,index)
        original=tuple(tuple(p) for p in case['event']['program'])
        other=dict(cases[1-i]['program'])
        # Diagnostic hindsight only: keep this state's calendar/roles and borrow
        # overlapping easy doses. Never inject a later plan into normal search.
        seed=tuple((d,other[d] if d in other and (d,other[d]) in case['physical']
            and case['physical'][(d,k)].role=='easy'
            and case['physical'][(d,other[d])].role=='easy' else k) for d,k in original)
        raw=case['evaluate_details'](seed)
        if not raw['decision_eligible']:
            cross_starts.append(dict(index=index,eligible=False,scope='Hindsight dose seed rejected'))
            continue
        compiled=compile_calendar(case['state'],seed,physical=case['physical'],result=raw)
        checked=refine(compiled,case['evaluate'])
        cross_starts.append(dict(checked,index=index,seed_program=seed,
            scope='Hindsight fixed-calendar dose seed, not a causal search or daily stability result'))
    manifest=freeze_sources(output/'sources.zip',config=dict(root=str(root),before=before,after=after),
        entrypoint='scripts.audit_coordinated_calendar',inputs=(root/f'decision-{before:04d}.json',root/f'decision-{after:04d}.json'))
    result=dict(scope='Frozen fixed-calendar refinement; not daily adherence or global optimality',
        cases=cases,visible_comparison=comparisons,cross_starts=cross_starts,manifest=manifest,live_unchanged=True)
    (output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(dict(path=str(output/'result.json'),cases=[{k:c[k] for k in
        ('index','cost','original_cost','seconds','full_evaluations','compiled_evaluations','neighborhood_complete','maximum_score_error')}
        for c in cases],visible_comparison=comparisons),indent=2),flush=True)
    return result


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('--before',type=int,required=True)
    parser.add_argument('--after',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    run(args.root,before=args.before,after=args.after,output=args.output)
