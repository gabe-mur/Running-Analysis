"""Finite old/new accounting comparison on the stopped replay; no search."""
from argparse import ArgumentParser
from dataclasses import replace
from datetime import timedelta
import json
from pathlib import Path
from statistics import median
from time import perf_counter

from scripts.replay_adaptive_search import load_decision
from scripts.audit_progression_rollover_instability import fixed_taper_guard,score,splice_program
from scripts.research_manifest import freeze_sources
from run_analysis.config import load_config


def run(root,output,*,phase='uniform_v1'):
    if phase not in {'uniform_v1','known_full_phase_v1'}:
        raise ValueError('Named purpose continuation phase required')
    before,after=(load_decision(root,i) for i in (6,7))  # Unmodified parity mandatory.
    output.mkdir(exist_ok=False)
    manifest=freeze_sources(output/'sources.zip',config=load_config(),entrypoint=__name__,
        inputs=tuple(root/f'{kind}-{i:04d}.json' for i in (6,7)
                     for kind in ('decision','snapshot'))+(root/'scenario.json',))
    prior=tuple(map(tuple,before['event']['program']))
    new=tuple(map(tuple,after['event']['program']))
    shared=tuple(p for p in new if p[0]<before['event']['end'])
    a,b=before['state'].context,after['state'].context
    clock=replace(a,decision=b.decision,hard_end=b.hard_end,
                  continuation_guard_days=fixed_taper_guard(a.continuation_guard_days,1))
    taper=replace(clock,continuation_guard_days=a.continuation_guard_days)
    rows=[]
    for contract in (False,True):
        for label,case,context in (('original',before,a),('clock_only',after,clock),
                                   ('taper_only',after,taper),('full_rollover',after,b)):
            context=replace(context,purpose_continuation=contract,
                purpose_continuation_phase=phase if contract else 'uniform_v1')
            old,competitor=score(case,prior,context),score(case,shared,context)
            rows.append(dict(purpose_continuation=contract,frame=label,
                prior=old,competitor_shared_dates=competitor,
                new_advantage=old['cost']-competitor['cost']))
    context=replace(b,purpose_continuation=True,purpose_continuation_phase=phase)
    winner=score(after,new,context)
    splices=[dict(cut=cut,score=score(after,splice_program(prior,new,cut),context))
             for cut in sorted({(b.decision+timedelta(days=7)).isoformat(),
                               b.hard_end.isoformat(),a.taper_start.isoformat(),b.taper_start.isoformat()})]
    timings={}
    for contract in (False,True):
        context=replace(b,purpose_continuation=contract,
            purpose_continuation_phase=phase if contract else 'uniform_v1')
        samples=[]
        for _ in range(25):
            start=perf_counter();score(after,new,context);samples.append(perf_counter()-start)
        timings[str(contract)]=dict(median_seconds=median(samples),minimum_seconds=min(samples),
                                   samples=len(samples),whole_search_measured=False)
    report=dict(manifest=manifest,purpose_continuation_phase=phase,controls=rows,new_contract_selected=winner,
        stable_prefix_splices=splices,price_timings=timings,
        scope='Fixed physical candidates only; not a new winner, daily pass or physiological clearance; live unchanged')
    with (output/'report.json').open('x') as stream:json.dump(report,stream,indent=2,allow_nan=False)
    print(json.dumps(dict(phase=phase,advantages=[dict(frame=r['frame'],continued=r['purpose_continuation'],
        new_advantage=r['new_advantage']) for r in rows],price_timings=timings)),flush=True)
    return report


if __name__=='__main__':
    parser=ArgumentParser()
    parser.add_argument('root',type=Path)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--phase',choices=('uniform_v1','known_full_phase_v1'),default='uniform_v1')
    args=parser.parse_args();run(args.root,args.output_dir,phase=args.phase)
