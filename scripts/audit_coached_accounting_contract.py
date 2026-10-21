"""Frozen physical/command controls, not measured adherence or physiology.

Verify primary-unit receipts and expected upload invariance on the real saved
catalogue and conditional forecasts. Compare nearby placements under archived
independent and explicit shared-running fatigue assumptions, without a search.
"""
from argparse import ArgumentParser
from dataclasses import replace,asdict
from pathlib import Path
import json

from scripts.replay_adaptive_search import load_decision
from scripts.prototype_coached_quality_calendar import coached_evaluator
from scripts.research_manifest import freeze_sources
from run_analysis.adaptive.quality_coaching_policy import CoachedRecipe,QualityCoachingSequence
from run_analysis.adaptive.quality_assignment_accounting import (
    issue_assignment,reconcile_assignment,advance_sequence,completion_json,read_completion)
from run_analysis.continuous_training_dose import ContinuousDoseReference


def run(root,projection,output):
    report=json.loads((projection/'report.json').read_text())
    case=load_decision(root,report['source_decision'])
    if case['state'].fingerprint!=report['source_snapshot']:raise ValueError('Input snapshot changed')
    data=dict(report['coaching_contract']);data['ordinary']=tuple(CoachedRecipe(**x) for x in data['ordinary'])
    sequence=QualityCoachingSequence(**data)
    if sequence.key!=report['coaching_contract_key']:raise ValueError('Coaching assignment changed')
    program=tuple(map(tuple,report['search']['program']))
    if report['decision']!='2026-10-08':
        raise ValueError('These placement witnesses belong to the frozen October8 opening')
    state=case['state'];results={};receipts=[]
    first=next((d,k) for d,k in program if case['catalog'][k].role=='quality')
    for recipe in sequence.ordinary:
        # Find its ordinal in one FULL cycle, rather than changing the policy.
        pos=next(i for i in range(sequence.cycle_length) if sequence.assignment(i).key==recipe.key)
        issued=issue_assignment(sequence,pos,case['catalog'])
        forecast=case['physical'][first[0],recipe.key]
        work=[p for p in case['catalog'][recipe.key].workout_definition['numeric_contract']['phases'] if p['kind']=='work']
        if len(work)!=len(forecast.bouts):raise ValueError('Command and physical work phases disagree')
        # Synthetic exact execution: measured-primary stand-in disclosed here.
        actual=replace(forecast,key='synthetic:'+forecast.key,source='observed',available_at=forecast.end,
            advisory_timezone=None,command_running_minutes=None,
            work_evidence_basis='explicit_segments',bouts=tuple(replace(b,
                distance_meters=p['meters']['estimate']) for b,p in zip(forecast.bouts,work)))
        receipt=reconcile_assignment(issued,actual,purpose_matched=True)
        if receipt.credit.low!=1 or read_completion(completion_json(receipt))!=receipt:
            raise ValueError('Exact native command did not retain full completion across restart')
        receipts.append(dict(recipe=recipe.recipe,unit=issued.primary_unit,
            required=asdict(issued.required),credit=asdict(receipt.credit),restart_exact=True))
    for mode in ('independent_v1','running_work_shared_v1'):
        engine=coached_evaluator(case,sequence,demand_interaction=mode)
        def summary(p):
            z=engine['detail'](engine['bind'](tuple(sorted(p))));e=z['evaluation']
            return dict(total=e.total_cost,fatigue=e.weighted_demand_cost,delivery=e.delivery_cost,
                eligible=z['decision_eligible'],quality=e.parts['shortfall_by_purpose']['quality'],
                long=e.parts['shortfall_by_purpose']['long'])
        contrasts={}
        for name,moves in (('selected',{}),('quality_earlier',{'2026-10-10':'2026-10-09'}),
                ('long_later',{'2026-10-11':'2026-10-12'}),
                ('second_long_later',{'2026-10-16':'2026-10-17'})):
            contrasts[name]=summary(tuple((moves.get(d,d),k) for d,k in program))
        if mode=='independent_v1' and abs(contrasts['selected']['total']-report['selected']['total'])>1e-12:
            raise ValueError('Archived coached winner no longer reproduces')
        first_run=case['physical'][first]
        work=[p for p in case['catalog'][first[1]].workout_definition['numeric_contract']['phases'] if p['kind']=='work']
        actual=replace(first_run,key='synthetic:'+first_run.key,source='observed',available_at=first_run.end,
            advisory_timezone=None,command_running_minutes=None,
            work_evidence_basis='explicit_segments',bouts=tuple(replace(b,distance_meters=p['meters']['estimate'])
                for b,p in zip(first_run.bouts,work)))
        receipt=reconcile_assignment(issue_assignment(sequence,0,case['catalog']),actual,purpose_matched=True)
        advanced=advance_sequence(sequence,receipt)
        after_engine=coached_evaluator(case,advanced,completions=(read_completion(completion_json(receipt)),),
            demand_interaction=mode)
        # Fixed endpoint/targets/forecasts: isolate completion ownership, not
        # rolling-horizon or forecast aging. Every earlier run must be observed.
        complete=[r for p,r in ((p,case['physical'][p]) for p in program) if r.start<first_run.start]
        if complete:raise ValueError('This control requires the opening quality as first future run')
        future=tuple(case['physical'][p] for p in program if p!=first)
        before=[];after=[]
        for rows in (future,tuple(case['physical'][('2026-10-12',k)] if r.start.date().isoformat()=='2026-10-11'
                else r for r in future for k in (r.key.split(':',2)[2],))):
            kwargs=dict(reference=ContinuousDoseReference('sandbox upload accounting'),timezone=state.timezone,
                purpose_fraction=state.context.scenario.bound_fraction)
            before.append(engine['connected'](state.context,state.history,(first_run,*rows),as_of=state.as_of,**kwargs)['evaluation'])
            after.append(after_engine['connected'](state.context,(*state.history,actual),rows,as_of=actual.available_at,**kwargs)['evaluation'])
        delta_before=before[0].total_cost-before[1].total_cost
        delta_after=after[0].total_cost-after[1].total_cost
        if abs(delta_before-delta_after)>1e-12:raise ValueError('Exact upload changed relative future costs')
        results[mode]=dict(contrasts=contrasts,expected_upload=dict(delta_before=delta_before,
            delta_after=delta_after,absolute_error=abs(delta_before-delta_after),
            actual_primary_credit=receipt.credit.low,physical_ledger_retained=True))
    output.mkdir(exist_ok=False)
    result=dict(source_snapshot=state.fingerprint,controls=receipts,models=results,
        scope=__doc__,synthetic_exact_execution=True,measured_hr_accuracy_established=False,
        live_changed=False,calendar_search=False)
    result['manifest']=freeze_sources(output/'sources.zip',config={},entrypoint=__name__,
        inputs=(projection/'report.json',root/'scenario.json',root/case['event']['snapshot_file']))
    (output/'report.json').write_text(json.dumps(result,default=str,indent=2,allow_nan=False))
    print(json.dumps(result,default=str,indent=2))


if __name__=='__main__':
    p=ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    p.add_argument('projection',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.root,a.projection,a.output)
