"""Cheap exact pricing of causal captured doses on a fixed winner calendar."""
from argparse import ArgumentParser
from dataclasses import replace
from datetime import timedelta
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from run_analysis.calendar_diversity import retain_diverse
from run_analysis.adaptive.compiled_calendar_doses import compile_calendar,refine
from scripts.audit_available_search_starts import carry_easy_doses
from scripts.replay_adaptive_search import load_decision
from scripts.research_manifest import freeze_sources


def mapped_pool(compiled,winner,pool,catalog):
    """Deduplicate and price every admitted captured allocation, not only finalists."""
    seeds={}
    for row in pool:
        if not row['eligible']: continue
        program=carry_easy_doses(winner,tuple(map(tuple,row['program'])),catalog)
        levels=tuple(compiled.keys.index(dict(program)[d]) for d in compiled.days)
        seeds.setdefault(levels,[]).append(row['evaluation'])
    levels=np.array(list(seeds),dtype=int)
    costs=np.concatenate([compiled.cost(levels[i:i+128]) for i in range(0,len(levels),128)])
    return tuple(dict(levels=tuple(int(v) for v in x),cost=float(cost),
        source_evaluations=seeds[tuple(x)],program=compiled.physical(x))
        for x,cost,funded in zip(levels,costs,compiled.funded(levels)) if funded)


def run(root,*,index,pool_path,output,shortlist_size=16,full_check_cap=64):
    case=load_decision(root,index)
    source=json.loads((pool_path.parent/'result.json').read_text())
    pool=json.loads(pool_path.read_text())
    if (source['snapshot_fingerprint']!=case['state'].fingerprint
            or Path(source['source_decision']).resolve()!=(root/f'decision-{index:04d}.json').resolve()
            or len(pool)!=source['evaluations']):
        raise ValueError('Captured pool must originate in this exact decision')
    output.mkdir(parents=True,exist_ok=False)
    manifest=freeze_sources(output/'sources.zip',config=dict(index=index,shortlist_size=shortlist_size,
        full_check_cap=full_check_cap),entrypoint=__name__,inputs=(pool_path,pool_path.parent/'result.json',
        root/f'decision-{index:04d}.json',root/case['event']['snapshot_file']))
    stamp=perf_counter(); cache={}; full_calls=0
    def details(p):
        nonlocal full_calls
        if p not in cache:
            cache[p]=case['evaluate_details'](p);full_calls+=1
        return cache[p]
    def evaluate(p):
        raw=details(p)
        return dict(cost=raw['evaluation'].total_cost,eligible=raw['decision_eligible'])
    winner=tuple(map(tuple,case['event']['program']))
    compiled=compile_calendar(case['state'],winner,physical=case['physical'],result=details(winner))
    rows=mapped_pool(compiled,winner,pool,case['catalog'])
    visible_end=(case['state'].context.decision+timedelta(days=7)).isoformat()
    visible=[i for i,d in enumerate(compiled.days) if d<visible_end]
    selected=retain_diverse(rows,shortlist_size,rank=lambda r:(r['cost'],r['levels']),
        identity=lambda r:r['levels'],signature=lambda r:tuple(r['levels'][i] for i in visible))
    initial=tuple(int(v) for v in compiled.original_levels)
    if all(r['levels']!=initial for r in selected):
        selected=[dict(levels=initial,cost=case['archived_cost'],source_evaluations=[],program=winner),*selected]
    pricing_seconds=perf_counter()-stamp
    results=[]
    for row in selected:
        checked=evaluate(row['program'])
        if abs(checked['cost']-row['cost'])>1e-10:
            raise ValueError('Mapped seed must reproduce its exact compiled score')
        entry=dict(row,initial_eligible=checked['eligible'])
        if checked['eligible']:
            # Identical calendar: reuse exact matrices without reanchoring any
            # constant to the alternative's score or fitting another model.
            seeded=replace(compiled,program=row['program'],original_levels=np.array(row['levels']),scored_rows=0)
            entry.update(refine(seeded,evaluate,max_full_evaluations=full_check_cap))
        results.append(entry)
    admitted=[r for r in results if r['initial_eligible']]
    best=min(admitted,key=lambda r:(r['cost'],r['program']))
    baseline=next(r for r in results if r['levels']==initial)
    result=dict(manifest=manifest,snapshot_fingerprint=case['state'].fingerprint,
        captured_evaluations=len(pool),unique_funded_allocations=len(rows),shortlisted=len(selected),
        pricing_seconds=pricing_seconds,seconds=perf_counter()-stamp,actual_full_evaluations=full_calls,
        minimum_initial_cost=min(r['cost'] for r in rows),winner_only_cost=baseline['cost'],
        best_cost=best['cost'],best_program=best['program'],best_source_evaluations=best['source_evaluations'],
        gain_over_winner_only=baseline['cost']-best['cost'],starts=results,
        all_admitted_neighborhoods_complete=all(r['neighborhood_complete'] for r in admitted),
        scope='Exact pricing of all admitted captured dose allocations; bounded score/diverse refinement, not global or daily stability',
        live_changed=False)
    (output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps({k:result[k] for k in ('captured_evaluations','unique_funded_allocations','shortlisted',
        'pricing_seconds','seconds','actual_full_evaluations','winner_only_cost','best_cost',
        'gain_over_winner_only','all_admitted_neighborhoods_complete')},indent=2),flush=True)
    return result


if __name__=='__main__':
    p=ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    p.add_argument('--index',type=int,required=True)
    p.add_argument('--pool',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.root,index=a.index,pool_path=a.pool,output=a.output)
