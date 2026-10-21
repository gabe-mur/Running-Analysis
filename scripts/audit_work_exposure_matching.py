"""Frozen work-only diagnostic, actual-issue contrast and counterfactual checks.

Never search a calendar, fit coefficients, alter the source database or deploy.
Recorded-duration comparisons do not establish intensity adherence.
"""
from argparse import ArgumentParser
from dataclasses import asdict, replace
from datetime import datetime, timedelta
import json
from math import fsum
from pathlib import Path
from statistics import mean
from time import perf_counter
from zipfile import ZipFile

import yaml

from run_analysis.adaptive.direct_quality_load import (
    PhaseHistory, checked_layout, forecast_direct, phase_observations, score_direct)
from run_analysis.adaptive.prescribed_hr_target import prescribed_hr_target
from run_analysis.adaptive.work_exposure_matching import (
    PhaseTrace, WorkExposure, forecast_work_exposure, forecast_work_for_ranking, requested_work_bands)
from run_analysis.adaptive.quality_dose_policy import QualityDoseStage, recipe_requests
from run_analysis.adaptive.quality_prescription_contract import causal_paces, compile_quality
from run_analysis.continuous_training_dose import ContinuousDoseReference
from run_analysis.movement import classify_movement
from run_analysis.phase_calibration import clip_trace
from run_analysis.training_signals import HeartRateReference, observed_cardio_trace
from run_analysis.work_stimulus_evidence import observed_work_signals
from run_analysis.workload_accounting import Quantity
from scripts.audit_direct_quality_load import phase_errors
from scripts.audit_issued_quality_load import whole_load_error
from scripts.audit_program_evidence_adapter import load_history
from scripts.audit_work_stimulus import encode
from scripts.compare_requested_quality_response import saved_effort
from scripts.continuous_program_adapter import with_continuous_trace
from scripts.program_evidence_adapter import QualityReference
from scripts.research_manifest import freeze_sources
from scripts.validate_quality_load import phase_layout


def saved_queries(payload, layout, config):
    effort = saved_effort(payload, 'work')
    ends = [prescribed_hr_target(c,config) for c in (effort.start_cues,effort.finish_cues)] if effort else []
    command = WorkExposure((ends[0].low,ends[0].high),(ends[1].low,ends[1].high),
        'independently saved work cues') if ends and all(p is not None for p in ends) else None
    return tuple(command if kind=='work' else None for kind,_,_ in checked_layout(layout))


def summary(cases):
    paired = [c for c in cases if c['intent_known'] and c['complete_recording']
              and c['baseline_metrics'] and c['candidate_metrics']]
    def stats(rows):
        values = [p['fraction_error'] for p in rows if p and p['fraction_error'] is not None]
        return dict(n=len(values),mae=mean(map(abs,values)) if values else None,
                    bias=mean(values) if values else None)
    return dict(inventory=len(cases),paired=len(paired),
        baseline=stats([c['baseline_metrics'] for c in paired]),
        candidate=stats([c['candidate_metrics'] for c in paired]),
        phases={kind:dict(baseline=stats([c['baseline_phase_errors'].get(kind) for c in paired]),
                         candidate=stats([c['candidate_phase_errors'].get(kind) for c in paired]))
                for kind in ('warmup','work','recovery','cooldown')})


def selection_readiness(probe, recipes):
    """An overlap-only pass cannot hide unsupported intensity or authored cues."""
    blockers = []
    if probe['supported_intensity_reversals'] or probe['duration_reversals']:
        blockers.append('supported_counterfactual_reversal')
    if any(not r['supported'] for r in probe['rows']):
        blockers.append('incomplete_counterfactual_intensity_support')
    if probe['fallback_intensity_reversals']:
        blockers.append('unsupported_fallback_can_price_harder_work_cheaper')
    if not recipes['prescriptions'] or any(not r['all_work_exposure_supported'] for r in recipes['prescriptions']):
        blockers.append('compiled_menu_contains_unresolved_or_unsupported_work_exposure')
    return dict(ready_for_selection_validation=not blockers, blockers=blockers,
        production_acceptance=False,
        interpretation='Observed overlap is reference support, not certified effort transfer or all-family accuracy')


class Inventory:
    def __init__(self, config, parents, issued, incomplete):
        self.config,self.parents,self.issued,self.incomplete = config,parents,issued,incomplete
        self.reference=ContinuousDoseReference('work-only exposure audit; unchanged accounting')
        self.hr=HeartRateReference(config['resting_hr'],config['max_hr'],'frozen configured normalization')
        self.history=[];self.original_history=[];self.profiles={};self.cases=[];self.origins=[];self.signals={}

    def __call__(self, run, activity):
        if run.role!='quality':
            return run
        intervals=classify_movement(activity.trackpoints,self.config['moving_time']).intervals
        trace=observed_cardio_trace(activity,intervals,self.hr,self.reference.policy,
            include_spans=True,compute_density=False)
        spans=trace['normalized_spans']
        updated,_=with_continuous_trace(run,spans,reference=self.reference,trace_key=run.key,
            trace_source='observed',elapsed_seconds=trace['elapsed_seconds'],
            outside_reference_seconds=trace['exclusions']['outside_reference_seconds'])
        signals=observed_work_signals(updated,activity,self.config,intervals,spans) if updated.bouts else None
        self.signals[run.key]=signals
        layout=checked_layout(phase_layout(signals.windows,trace['elapsed_seconds']) if signals else ())
        observations=phase_observations(layout,spans)
        original=forecast_direct(layout,self.original_history,as_of=run.start,accounting_key=self.reference.accounting_key)
        if json.loads(json.dumps(original,default=encode))!=self.parents[run.key]['direct_load']['forecast']:
            raise AssertionError('Original frozen direct forecast does not reproduce exactly')
        baseline=forecast_direct(layout,self.history,as_of=run.start,accounting_key=self.reference.accounting_key)
        assignment=self.issued.get(run.key)
        query=saved_queries(assignment['payload'],layout,self.config) if assignment else (None,)*len(layout)
        candidate=forecast_work_exposure(layout,self.history,self.profiles,query,
            as_of=run.start,accounting_key=self.reference.accounting_key)
        for a,b in zip(baseline['phases'],candidate['phases']):
            if a['kind']!='work' and a!=b:
                raise AssertionError('Work-only option altered a nonwork phase')
        case=dict(key=run.key,start=run.start,label=self.parents[run.key]['label'],
            complete_recording=run.key not in self.incomplete,
            recording_status='user_confirmed_incomplete_opening' if run.key in self.incomplete else 'not_flagged_incomplete_not_independent_certification',
            intent_known=any(q is not None for q in query),intensity_adherence_established=False,
            baseline=baseline,candidate=candidate,observations=observations,
            baseline_metrics=score_direct(baseline,spans),candidate_metrics=score_direct(candidate,spans),
            baseline_phase_errors=phase_errors(baseline,observations),
            candidate_phase_errors=phase_errors(candidate,observations),issued=None)
        if assignment and assignment['issued_at'] and assignment['contract']['layout']:
            at=datetime.fromisoformat(assignment['issued_at'])
            physical=assignment['contract']['layout']
            old=forecast_direct(physical,self.history,as_of=at,accounting_key=self.reference.accounting_key)
            bounds=assignment['actual']['load_bounds']
            expected=assignment['forecast']['predicted_load']
            if abs(old['predicted_load']-expected)>1e-8:
                raise AssertionError('Issue-frozen direct comparator changed')
            new=forecast_work_exposure(physical,self.history,self.profiles,
                saved_queries(assignment['payload'],physical,self.config),as_of=at,accounting_key=self.reference.accounting_key)
            case['issued']=dict(issued_at=at,baseline=old,candidate=new,
                baseline_whole_plan_vs_recorded=whole_load_error(old['predicted_load'],bounds['low'],bounds['high']),
                candidate_whole_plan_vs_recorded=whole_load_error(new['predicted_load'],bounds['low'],bounds['high']),
                execution_differences=assignment['execution_differences'],full_adherence_established=False)
        self.cases.append(case)
        if assignment and run.key not in self.incomplete:
            self.origins.append(dict(name='before_completed_'+run.key,as_of=run.start,history=tuple(self.history)))
        observed=PhaseHistory(run.key,run.start,run.end,run.available_at,self.reference.accounting_key,observations)
        self.original_history.append(observed)
        # Omit the confirmed truncated opening only from reference calibration;
        # retain its actually observed load and later supported phases unchanged.
        references=tuple(replace(p,known_seconds=0.,load=None) if i==0 and run.key in self.incomplete else p
                         for i,p in enumerate(observations))
        self.history.append(replace(observed,phases=references))
        self.profiles[run.key]=tuple(PhaseTrace(a,b,clip_trace(spans,a,b)) for _,a,b in layout)
        return updated


def probes(inventory, as_of):
    levels=[('Z2',),('Z3',),('low Z4',),('Z4',),('Z5',)]
    commands=[]
    for cues in levels:
        p=prescribed_hr_target(cues,inventory.config)
        commands.append((cues[0],p.nominal,WorkExposure((p.low,p.high),(p.low,p.high),'counterfactual '+cues[0])))
    origins=[inventory.origins[-1],dict(name='current_frozen_cutoff',as_of=as_of,history=tuple(inventory.history))]
    rows=[];reversals=[];fallbacks=[];dose_reversals=[];handoff_errors=[]
    started=perf_counter()
    for origin in origins:
        for seconds in (30,60,90,120,180,300,600,900,1200,1800):
            layout=(('warmup',0.,600.),('work',600.,600.+seconds),('cooldown',600.+seconds,1200.+seconds))
            local=[]
            for cue,nominal,command in commands:
                f=forecast_work_exposure(layout,origin['history'],inventory.profiles,(None,command,None),
                    as_of=origin['as_of'],accounting_key=inventory.reference.accounting_key)
                ranked=forecast_work_for_ranking(layout,origin['history'],inventory.profiles,(None,command,None),
                    as_of=origin['as_of'],accounting_key=inventory.reference.accounting_key)
                p=f['phases'][1]
                if (f['work_exposure_matching']['requested_intensity_rankable'] and ranked!=f or
                        not f['work_exposure_matching']['requested_intensity_rankable'] and ranked['predicted_load'] is not None):
                    handoff_errors.append(dict(origin=origin['name'],seconds=seconds,cue=cue))
                row=dict(origin=origin['name'],work_seconds=seconds,cue=cue,nominal_hrr=nominal,
                    work_rate_per_minute=p['rate']*60 if p['rate'] is not None else None,
                    whole_load=f['predicted_load'],supported=p['work_exposure']['supported'],
                    fallback=p['work_exposure']['fallback_reason'],source_keys=f['source_keys'],
                    handoff_status=ranked['status'],rankable_whole_load=ranked['predicted_load'])
                rows.append(row);local.append(row)
            for a,b in zip(local,local[1:]):
                if a['whole_load'] is not None and b['whole_load'] < a['whole_load']-1e-10:
                    (reversals if a['supported'] and b['supported'] else fallbacks).append(dict(
                        origin=origin['name'],seconds=seconds,from_cue=a['cue'],to_cue=b['cue'],
                        load_drop_fraction=1-b['whole_load']/a['whole_load'],from_rate=a['work_rate_per_minute'],
                        to_rate=b['work_rate_per_minute'],fallback=b['fallback']))
        for cue,_,_ in commands:
            chosen=[r for r in rows if r['origin']==origin['name'] and r['cue']==cue]
            for a,b in zip(chosen,chosen[1:]):
                if a['whole_load'] is not None and b['whole_load'] < a['whole_load']-1e-10:
                    dose_reversals.append(dict(origin=origin['name'],cue=cue,from_seconds=a['work_seconds'],to_seconds=b['work_seconds']))
    return dict(rows=rows,supported_intensity_reversals=reversals,fallback_intensity_reversals=fallbacks,
        duration_reversals=dose_reversals,handoff_errors=handoff_errors,
        handoff_checks_passed=not handoff_errors,
        unresolved_for_ranking=sum(r['rankable_whole_load'] is None for r in rows),
        elapsed_seconds=perf_counter()-started,
        interpretation='Same clock, prior evidence and nonwork; changing only requested work band or work duration, not a scored optimizer menu')


def menu(inventory, history, as_of):
    paces=causal_paces(history,as_of=as_of,eligible_keys={r.key for r in history})
    if not paces['easy'] or not paces['work']:
        return dict(status='missing_causal_extent_reference',prescriptions=[])
    stage=QualityDoseStage('diagnostic_not_accepted_stage',as_of,4.4,15.,7.,
        'declared shared 4.4mi/15workmin compiler diagnostic; not a live prescription')
    requests=recipe_requests(stage,issued_at=as_of,
        distance_rep_seconds={m:m*paces['work'].seconds_per_meter.estimate for m in (400,500)},
        open_recovery_seconds=90,scenario_source='frozen causal work pace and existing 90s open-recovery scenario')
    rows=[]
    for request in requests:
        prescription=compile_quality(request,easy_pace=paces['easy'],work_pace=paces['work'])
        layout=[];at=0.
        for p in prescription.phases:
            end=at+p.seconds.estimate;layout.append((p.kind,at,end));at=end
        baseline=forecast_direct(layout,inventory.history,as_of=as_of,accounting_key=inventory.reference.accounting_key)
        f=forecast_work_for_ranking(layout,inventory.history,inventory.profiles,
            requested_work_bands(prescription.phases,inventory.config),as_of=as_of,accounting_key=inventory.reference.accounting_key)
        rows.append(dict(family=request.family,variant=request.variant,prescription=prescription,
            baseline=baseline,candidate=f,primary_commands_unchanged=True,
            all_work_exposure_supported=all(p['work_exposure']['supported'] for p in f['phases'] if p['kind']=='work')))
    return dict(status='diagnostic_reprice_not_live_selection',stage=stage,extent_references=paces,
        families=sorted({r['family'] for r in rows}),prescriptions=rows)


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-root',type=Path,required=True)
    parser.add_argument('--issued-report',type=Path,required=True)
    parser.add_argument('--partial-opening-key',action='append',default=[])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    parent=json.loads((args.baseline_root/'baseline.json').read_text())
    issued=json.loads(args.issued_report.read_text())
    with ZipFile(args.baseline_root/'sources.zip') as archive:config=yaml.safe_load(archive.read('config.yaml'))
    as_of=datetime.fromisoformat(parent['protocol']['cutoff']);database=args.baseline_root/'evidence.sqlite'
    incomplete=set(args.partial_opening_key)
    if not incomplete <= {c['key'] for c in parent['cases']}:
        raise ValueError('Confirmed partial recording must resolve to an inventoried source')
    protocol=dict(version='work_only_exposure_validation_v2',
        candidate='Same full-phase band overlap, only work rates may change; default direct predictor unchanged',
        fixed='30-day availability, normalized measured HR integrals, source votes, duration similarity; no fitted weights',
        unsupported_contract='Retain timing-only diagnostic proxy; effort-aware handoff returns unresolved, never a discounted cost',
        hill_contract='Authored Z4 effort as clarified by athlete; same work cue as short intervals, no achieved-HR or tissue-cost equality claim',
        partial_opening_keys=sorted(incomplete),partial_policy='Exclude whole-session accuracy; remove first phase only from calibration, preserve observed load/later phases',
        acceptance='Preserve known-intent support and unchanged nonwork; improve conditional whole/work MAE; report every intensity/duration reversal separately',
        limits=['Four inspected saved commands; realized timing does not establish intensity adherence',
            'Two resolved complete quality commands at actual issuance; full-load errors include adherence and recording differences',
            'Counterfactual overlap support is observed HR, not proof of commanded effort; unsupported intensity retains disclosed baseline',
            'All-family menu is a shared compiler scenario, not live prescription, predictive accuracy or optimal workout selection'],
        planner_influence=False)
    with (args.output/'protocol.json').open('x') as stream:json.dump(protocol,stream,indent=2)
    manifest=freeze_sources(args.output/'sources.zip',config=config,entrypoint=__name__,
        inputs=(database,args.baseline_root/'baseline.json',args.issued_report,args.output/'protocol.json'))
    inventory=Inventory(config,{c['key']:c for c in parent['cases']},{c['key']:c for c in issued['cases']},incomplete)
    reference=QualityReference('threshold','work_seconds',Quantity.point(900,'unused comparator'),Quantity(290,310),'original source loading only')
    started=perf_counter()
    history,missing=load_history(database,config,as_of,reference,
        since=min(datetime.fromisoformat(c['start']) for c in parent['cases'])-timedelta(days=1),run_transform=inventory)
    if set(inventory.parents)!={c['key'] for c in inventory.cases}:raise AssertionError('Missing historical quality case')
    groups={name:summary(rows) for name,rows in (
        ('known_intent',inventory.cases),('intervals',[c for c in inventory.cases if c['label']=='intervals']),
        ('sustained',[c for c in inventory.cases if c['label']=='tempo_threshold']))}
    probe=probes(inventory,as_of);recipes=menu(inventory,history,as_of)
    complete=[c for c in inventory.cases if c['complete_recording'] and c['baseline_metrics']]
    conditional_gate=all(g['paired']>0 and g['candidate']['mae']<g['baseline']['mae']
        and g['phases']['work']['candidate']['mae']<=g['phases']['work']['baseline']['mae'] for g in groups.values())
    readiness=selection_readiness(probe,recipes)
    result=dict(protocol=protocol,manifest=manifest,cases=inventory.cases,summary=groups,probe=probe,menu=recipes,
        selection_readiness=readiness,
        issued_inventory=[dict(key=c['key'],role=c['current_actual_role'],contract_status=c['contract']['status'],
            current_trial=next((x['issued'] for x in inventory.cases if x['key']==c['key']),None)) for c in issued['cases']],
        completeness_screened_baseline=dict(n=len(complete),mae=mean(abs(c['baseline_metrics']['fraction_error']) for c in complete)),
        conditional_development_gate_passed=conditional_gate,
        supported_monotonicity_gate_passed=not probe['supported_intensity_reversals'] and not probe['duration_reversals'],
        baseline_exact=True,nonwork_exact=True,source_missing=missing,elapsed_seconds=perf_counter()-started,planner_influence=False)
    with (args.output/'report.json').open('x') as stream:json.dump(result,stream,indent=2,default=encode,allow_nan=False)
    print(json.dumps(dict(summary=groups,conditional_gate=conditional_gate,
        supported_intensity_reversals=len(probe['supported_intensity_reversals']),
        fallback_reversals=len(probe['fallback_intensity_reversals']),duration_reversals=len(probe['duration_reversals']),
        menu_count=len(recipes['prescriptions']),selection_readiness=readiness,
        handoff_checks_passed=probe['handoff_checks_passed'],unresolved_for_ranking=probe['unresolved_for_ranking'],
        supported_menu_count=sum(r['all_work_exposure_supported'] for r in recipes['prescriptions']),
        seconds=result['elapsed_seconds']),indent=2),flush=True)


if __name__=='__main__':main()
