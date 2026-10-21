"""Frozen regression attribution; no search, live refresh or physiology refit.

Reprice live and native archived winners first. Compare whole calendars on one
native input frame, using identical assigned physical work within each calendar
while changing one quality-accounting/continuation assumption at a time.
"""
from argparse import ArgumentParser
from dataclasses import replace
from functools import partial
from pathlib import Path
import json
from time import perf_counter

from scripts.replay_adaptive_search import load_decision
from scripts.prototype_coached_quality_calendar import coached_evaluator
from scripts.research_manifest import freeze_sources
from run_analysis.adaptive.quality_coaching_policy import CoachedRecipe,QualityCoachingSequence
from run_analysis.adaptive.decision_snapshot import evaluate_snapshot
from run_analysis.adaptive.continuous_program_adapter import evaluate_continuous
from run_analysis.adaptive.program_evidence_adapter import stage_quality_credit
from run_analysis.adaptive.shared_program_evaluator import evaluate_program
from run_analysis.adaptive.compiled_calendar_doses import compile_calendar,refine
from run_analysis.adaptive.calendar_merge_refinement import refine_merges
from run_analysis.continuous_training_dose import ContinuousDoseReference
from run_analysis.workload_accounting import Quantity


def mapped_calendar(source,target,sequence):
    rows=[];rank=0
    for day,key in source['event']['program']:
        if day not in target['dates']:raise ValueError('Use a live decision with the same dated horizon')
        dose=source['catalog'][key]
        if dose.role=='quality':
            key=sequence.assignment(rank).key;rank+=1
        elif key not in target['handoffs']:
            raise ValueError('Live nonquality dose absent from native domain: '+key)
        rows.append((day,key))
    return tuple(rows)


def describe(case,program):
    from datetime import date
    keys={role:[d for d,k in program if case['catalog'][k].role==role] for role in ('long','quality')}
    return dict(miles=sum(case['catalog'][k].miles for d,k in program if d<case['state'].context.hard_end.isoformat()),
        key_dates=keys,key_gaps={r:[(date.fromisoformat(b)-date.fromisoformat(a)).days
            for a,b in zip(ds,ds[1:])] for r,ds in keys.items()})


def run(native_root,projection,live_root,live_index,output):
    started=perf_counter();case=load_decision(native_root,1);live=load_decision(live_root,live_index)
    report=json.loads((projection/'report.json').read_text())
    if report['source_snapshot']!=case['state'].fingerprint:raise ValueError('Native source changed')
    data=dict(report['coaching_contract']);data['ordinary']=tuple(CoachedRecipe(**x) for x in data['ordinary'])
    sequence=QualityCoachingSequence(**data)
    if sequence.key!=report['coaching_contract_key']:raise ValueError('Assignment changed')
    engine=coached_evaluator(case,sequence);state=case['state'];ctx=state.context
    if ctx.decision!=live['state'].context.decision or ctx.end!=live['state'].context.end:
        raise ValueError('Use live/native decisions with the same horizon')
    if state.history!=live['state'].history or ctx.policy!=live['state'].context.policy:
        raise ValueError('Physical history and stage policy must match')
    calendars=dict(live_dates=mapped_calendar(live,case,sequence),
        native_dates=engine['bind'](tuple(map(tuple,case['event']['program']))),
        compressed=tuple(map(tuple,report['search']['program'])))
    ref=ContinuousDoseReference('sandbox upload accounting')
    def original_observed(r):
        return ctx.quality_programme.delivery(r.key,r.source).opportunity if (
            ctx.quality_programme is not None and r.role=='quality') else stage_quality_credit(r,ctx.policy)
    def evaluate(program,mode):
        rows=tuple(case['physical'][p] for p in program)
        if mode=='native_exposure':return evaluate_snapshot(state,rows,tuple(k for _,k in program),reference=ref)
        cycle=mode in {'issued_cycle','shared_cycle'}
        refs=tuple(engine['quality_reference'] if r[0]=='quality' and cycle else r
            for r in ctx.continuation_boundary_references)
        changed=replace(state,context=replace(ctx,quality_programme=None,continuation_boundary_references=refs))
        def credit(r):
            if r.role!='quality':return Quantity.point(0.,'not quality')
            if r.source=='observed':return original_observed(r)
            return stage_quality_credit(r,ctx.policy) if mode=='generic_work' else Quantity.point(1.,'issued fulfillment')
        evaluator=partial(evaluate_continuous,quality_credit_provider=credit,
            program_evaluator=partial(evaluate_program,demand_interaction=(
                'running_work_shared_v1' if mode=='shared_cycle' else 'independent_v1')))
        return evaluate_snapshot(changed,rows,tuple(k for _,k in program),reference=ref,evaluator=evaluator)
    def summary(raw):
        e=raw['evaluation']
        return dict(total=e.total_cost,fatigue=e.weighted_demand_cost,delivery=e.delivery_cost,
            eligible=raw['decision_eligible'],loss=e.parts['shortfall_by_purpose'],
            continuation=e.parts['continuation_boundary']['cross_demand_cost'],
            progression=raw.get('progression_delivery'),
            constraints={k:v for k,v in raw.get('constraints',{}).items() if k!='sessions'},
            suitability={k:v for k,v in raw.get('suitability',{}).items()
                if k not in {'session_checks','responses','sequence_assessment'}})
    matrix={mode:{name:summary(evaluate(p,mode)) for name,p in calendars.items()}
        for mode in ('native_exposure','generic_work','issued_source_reference','issued_cycle','shared_cycle')}
    if abs(matrix['issued_cycle']['compressed']['total']-report['selected']['total'])>1e-12:
        raise ValueError('Current coached winner changed')
    # Inspect the omitted stable-pipeline refiners against the exact coaching
    # objective. Independent mode only: shared cross-term compiler not accepted.
    effective=replace(state,context=replace(ctx,quality_programme=None,
        continuation_boundary_references=tuple(engine['quality_reference'] if r[0]=='quality' else r
            for r in ctx.continuation_boundary_references)))
    p=calendars['compressed'];raw=engine['detail'](p)
    compiled=compile_calendar(effective,p,physical=case['physical'],result=raw)
    polished=refine(compiled,engine['evaluate'],max_full_evaluations=64)
    merged=refine_merges(effective,polished['program'],physical=case['physical'],
        evaluate_details=engine['detail'],max_full_evaluations=32)
    final=merged['program']
    controls=dict(compiled=polished,merged=merged,final=summary(engine['detail'](final)),
        original=describe(case,p),final_calendar=describe(case,final))
    # Binding a different ordinary quality dose can remove miles from an
    # otherwise funded prior. Check bounded replacement easy volume without
    # moving any dates/roles, not a new search or a relaxed funding gate.
    live_program=calendars['live_dates'];live_before=engine['detail'](live_program)
    review=state.prospective_volume_review
    additions=[]
    for day,key in live_program:
        row=case['physical'][day,key]
        if row.role!='easy' or review is None or not review.start<=row.start<review.end:continue
        options=sorted((k for k in case['handoffs'] if case['catalog'][k].role=='easy'
            and case['catalog'][k].miles>case['catalog'][key].miles),key=lambda k:(case['catalog'][k].miles,k))
        for new_key in options:
            proposed=tuple((d,new_key if d==day else k) for d,k in live_program)
            raw=engine['detail'](proposed)
            additions.append(dict(day=day,old_miles=row.miles,new_miles=case['catalog'][new_key].miles,
                program=proposed,result=summary(raw)))
    accepted=sorted((r for r in additions if r['result']['eligible']),key=lambda r:r['result']['total'])
    funded_control=dict(before=summary(live_before),evaluations=len(additions),
        accepted=len(accepted),best=accepted[0] if accepted else None)
    if accepted:
        funded_control['best_fixed_calendar_by_mode']={mode:summary(evaluate(accepted[0]['program'],mode))
            for mode in ('issued_source_reference','issued_cycle','shared_cycle')}
    result=dict(live_original_parity=live['archived_cost'],native_original_parity=case['archived_cost'],
        live_source=live['event']['as_of'],native_source=case['event']['as_of'],
        same_history=True,same_stage_policy=True,same_horizon=True,
        clocks_not_identical=True,physical_forecasts_frozen_to_native_frame=True,
        calendars={name:describe(case,p) for name,p in calendars.items()},matrix=matrix,
        omitted_refiner_controls=controls,live_calendar_funding_control=funded_control,
        seconds=perf_counter()-started,
        search_performed=False,live_changed=False,
        scope='Finite full-score attribution; whole calendars differ; no global optimality or physiology claim')
    output.mkdir(exist_ok=False)
    result['manifest']=freeze_sources(output/'sources.zip',config={},entrypoint=__name__,
        inputs=(projection/'report.json',native_root/'decision-0001.json',
            native_root/case['event']['snapshot_file'],live_root/f'decision-{live_index:04d}.json',
            live_root/live['event']['snapshot_file']))
    (output/'report.json').write_text(json.dumps(result,default=str,indent=2,allow_nan=False))
    print(json.dumps(dict(seconds=result['seconds'],live_original_parity=result['live_original_parity'],
        original_coached_cost=matrix['issued_cycle']['compressed']['total'],
        polished_coached_cost=controls['final']['total'],
        funding_deficit=live_before['progression_delivery']['shortfall_miles'],
        funded_live_best={k:v for k,v in funded_control['best'].items() if k not in {'program','result'}}
            if funded_control['best'] else None,
        funded_live_scores={mode:{k:v for k,v in scored.items() if k in {'total','fatigue','delivery','eligible'}}
            for mode,scored in funded_control.get('best_fixed_calendar_by_mode',{}).items()}),
        default=str,indent=2),flush=True)


if __name__=='__main__':
    p=ArgumentParser(description=__doc__)
    p.add_argument('native_root',type=Path);p.add_argument('projection',type=Path)
    p.add_argument('live_root',type=Path);p.add_argument('--live-index',type=int,default=12)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.native_root,a.projection,a.live_root,a.live_index,a.output)
