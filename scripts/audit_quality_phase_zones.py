"""Describe phase HR and test one whole-phase exposure match; never select workouts."""
from argparse import ArgumentParser
from dataclasses import asdict
from datetime import datetime, timedelta
import json
from math import fsum
from pathlib import Path
import sqlite3
from statistics import mean
from time import perf_counter
from zipfile import ZipFile

import yaml

from run_analysis.adaptive.direct_quality_load import (
    POLICY, PhaseHistory, checked_layout, forecast_direct, phase_observations, score_direct, similarity)
from run_analysis.adaptive.prescribed_hr_target import prescribed_hr_target
from run_analysis.adaptive.work_exposure_matching import band_fraction
from run_analysis.continuous_training_dose import ContinuousDoseReference
from run_analysis.movement import classify_movement
from run_analysis.phase_calibration import clip_trace
from run_analysis.signal_structure import above_threshold
from run_analysis.training_signals import HeartRateReference
from run_analysis.work_stimulus_evidence import observed_work_signals
from run_analysis.workload_accounting import Quantity
from scripts.audit_direct_quality_load import phase_errors
from scripts.audit_issued_quality_load import matched_assignments, issue_boundary
from scripts.audit_program_evidence_adapter import load_history
from scripts.audit_work_stimulus import encode
from scripts.compare_requested_quality_response import saved_effort
from scripts.continuous_program_adapter import with_continuous_trace
from scripts.program_evidence_adapter import QualityReference
from scripts.research_manifest import freeze_sources
from scripts.validate_quality_load import phase_layout


def zone_profile(spans,left,right,config,reference):
    known=clip_trace(spans,left,right)
    seconds=fsum(s.end-s.start for s in known)
    # Same lower-edge convention as continuous_zone_seconds, but over all valid
    # observed wall-time HR, including stops, to match the load integrator.
    edges=[reference.reserve(config['zones']['z'+str(i)][0]) for i in range(1,6)]
    edges.append((config['zones']['z5'][1]+1-reference.resting_bpm)/
                 (reference.maximum_bpm-reference.resting_bpm))
    cumulative=[above_threshold(known,edge)['above_seconds'] for edge in edges]
    names=['below_z1',*[f'z{i}' for i in range(1,6)],'above_z5']
    amounts=[seconds-cumulative[0],*(a-b for a,b in zip(cumulative,cumulative[1:])),cumulative[-1]]
    average=fsum((s.end-s.start)*(s.first+s.last)/2 for s in known)/seconds if seconds else None
    return dict(known_seconds=seconds,unknown_seconds=right-left-seconds,
        mean_hrr=average,mean_bpm=reference.resting_bpm+average*(reference.maximum_bpm-reference.resting_bpm)
            if average is not None else None,
        first_bpm=reference.resting_bpm+known[0].first*(reference.maximum_bpm-reference.resting_bpm) if known else None,
        last_bpm=reference.resting_bpm+known[-1].last*(reference.maximum_bpm-reference.resting_bpm) if known else None,
        zone_seconds=dict(zip(names,amounts)),zone_fraction={k:v/seconds if seconds else None for k,v in zip(names,amounts)})


def targets(payload,layout,config):
    result=[]
    for kind,_,_ in checked_layout(layout):
        effort=saved_effort(payload,kind)
        ends=[prescribed_hr_target(cues,config) for cues in (effort.start_cues,effort.finish_cues)] if effort else []
        if not ends or any(p is None for p in ends):
            return None
        result.append(dict(kind=kind,effort=asdict(effort),start=asdict(ends[0]),finish=asdict(ends[1]),
            cue_provenance='saved work command' if kind=='work' else 'saved easy bookend or inherited opening-cue scenario'))
    return result


def exposure_forecast(layout,history,profiles,query,*,as_of,accounting_key):
    """One untuned trial: original timing affinity times observed band overlap.

    Phase rates remain complete-phase nonlinear load / known time. Matching an
    observed band does not prove that the source had that commanded effort.
    When all overlaps are zero, retain baseline and disclose absent matching
    evidence; never turn the phase into zero load or a workout-family ban.
    """
    layout=checked_layout(layout)
    if len(query)!=len(layout) or any(p['kind']!=q[0] for p,q in zip(query,layout)):
        raise ValueError('Aligned independent requested-effort query required')
    baseline=forecast_direct(layout,history,as_of=as_of,accounting_key=accounting_key)
    result=json.loads(json.dumps(baseline,default=encode))
    previous=None
    for (kind,left,right),command,p in zip(layout,query,result['phases']):
        sessions=[]
        bands=[(command[end]['low'],command[end]['high']) for end in ('start','finish')]
        for h in history:
            if h.key not in baseline['source_keys']:
                continue
            profile=profiles[h.key]
            indices=[i for i,o in enumerate(h.phases) if o.load is not None and
                (o.kind!='work' if p['fallback_pooled_nonwork'] else o.kind==kind)]
            weights=[];rates=[];fractions=[]
            for i in indices:
                o=h.phases[i];source=profile[i]
                fraction=band_fraction(source['spans'],source['start'],source['end'],*bands)
                weight=similarity(o,right-left,previous,True)*fraction
                weights.append(weight);rates.append(o.load/o.known_seconds);fractions.append(fraction)
            total=fsum(weights)
            if total:
                sessions.append(dict(key=h.key,rate=fsum(w*r for w,r in zip(weights,rates))/total,
                    relevance=total/len(indices),match_fractions=fractions))
        total=fsum(s['relevance'] for s in sessions)
        p['exposure_matching_sources']=sessions
        p['exposure_match_supported']=bool(total)
        p['exposure_match_fallback']=not bool(total)
        if total:
            p['rate']=fsum(s['rate']*s['relevance'] for s in sessions)/total
            p['load']=p['rate']*(right-left)
        previous=right-left
    if result['status']=='predicted':
        result['predicted_load']=fsum(p['load'] for p in result['phases'])
    return result


class Inventory:
    def __init__(self,config,assignments,parents):
        self.config,self.assignments,self.parents=config,assignments,parents
        self.reference=ContinuousDoseReference('unchanged phase-zone audit accounting')
        self.hr=HeartRateReference(config['resting_hr'],config['max_hr'],'frozen configured HR zones')
        self.history=[];self.profiles={};self.cases=[]

    def __call__(self,run,activity):
        if run.role!='quality':
            return run
        intervals=classify_movement(activity.trackpoints,self.config['moving_time']).intervals
        from run_analysis.training_signals import observed_cardio_trace
        trace=observed_cardio_trace(activity,intervals,self.hr,self.reference.policy,include_spans=True,compute_density=False)
        spans=trace['normalized_spans']
        updated,_=with_continuous_trace(run,spans,reference=self.reference,trace_key=run.key,
            trace_source='observed',elapsed_seconds=trace['elapsed_seconds'],
            outside_reference_seconds=trace['exclusions']['outside_reference_seconds'])
        signals=observed_work_signals(updated,activity,self.config,intervals,spans) if updated.bouts else None
        layout=phase_layout(signals.windows,trace['elapsed_seconds']) if signals else ()
        observations=phase_observations(layout,spans)
        baseline=forecast_direct(layout,self.history,as_of=run.start,accounting_key=self.reference.accounting_key)
        if json.loads(json.dumps(baseline,default=encode))!=self.parents[run.key]['direct_load']['forecast']:
            raise AssertionError('Phase-zone audit changed the frozen direct baseline')
        case=dict(key=run.key,start=run.start,label=self.parents[run.key]['label'],work_evidence=updated.work_evidence_basis,
            baseline=baseline,baseline_metrics=score_direct(baseline,spans),phases=[],
            intent_status='unknown_historical_intent',trial=None)
        command=self.assignments.get(run.key)
        query=targets(command['payload'],layout,self.config) if command else None
        if query:
            predicted=exposure_forecast(layout,self.history,self.profiles,query,as_of=run.start,
                accounting_key=self.reference.accounting_key)
            case.update(intent_status='saved_effort_with_realized_phase_timing',command_issued_at=command['issued_at'],
                query=query,trial=dict(forecast=predicted,metrics=score_direct(predicted,spans),
                    phase_errors=phase_errors(predicted,observations)))
        profile=[]
        for i,((kind,left,right),o) in enumerate(zip(checked_layout(layout),observations)):
            described=zone_profile(spans,left,right,self.config,self.hr)
            phase=dict(kind=kind,start=left,end=right,seconds=right-left,**described,
                whole_phase_load=o.load,load_per_minute=o.load/o.known_seconds*60 if o.load is not None else None,
                baseline_rate_per_minute=baseline['phases'][i]['rate']*60 if baseline['phases'][i]['rate'] is not None else None)
            if query:
                q=query[i]
                phase['command_band_fraction']=band_fraction(spans,left,right,
                    (q['start']['low'],q['start']['high']),(q['finish']['low'],q['finish']['high']))
            case['phases'].append(phase)
            profile.append(dict(start=left,end=right,spans=clip_trace(spans,left,right)))
        self.cases.append(case)
        # Current observations enter future history only AFTER all forecasts.
        self.history.append(PhaseHistory(run.key,run.start,run.end,run.available_at,
            self.reference.accounting_key,observations))
        self.profiles[run.key]=profile
        return run


def summarize(cases):
    paired=[c for c in cases if c['trial'] and c['trial']['metrics'] and c['baseline_metrics']]
    def stats(values):
        return dict(n=len(values),mae=mean(map(abs,values)) if values else None,bias=mean(values) if values else None)
    result=dict(total=len(cases),known_intent=sum(bool(c['trial']) for c in cases),paired=len(paired),
        baseline=stats([c['baseline_metrics']['fraction_error'] for c in paired]),
        candidate=stats([c['trial']['metrics']['fraction_error'] for c in paired]),phases={})
    for kind in ('warmup','work','recovery','cooldown'):
        both=[]
        for c in paired:
            phases=[p for p in c['phases'] if p['kind']==kind and p['whole_phase_load'] is not None]
            actual=fsum(p['whole_phase_load'] for p in phases)
            if actual:
                old=fsum(p['baseline_rate_per_minute']*p['known_seconds']/60 for p in phases)/actual-1
                new=c['trial']['phase_errors'][kind]['fraction_error']
                both.append((old,new))
        result['phases'][kind]=dict(baseline=stats([a for a,b in both]),candidate=stats([b for a,b in both]))
    return result


def main():
    p=ArgumentParser(description=__doc__)
    p.add_argument('--baseline-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    parent=json.loads((args.baseline_root/'baseline.json').read_text())
    as_of=datetime.fromisoformat(parent['protocol']['cutoff'])
    with ZipFile(args.baseline_root/'sources.zip') as archive:
        config=yaml.safe_load(archive.read('config.yaml'))
    database=args.baseline_root/'evidence.sqlite'
    protocol=dict(version='whole_phase_band_overlap_v1',
        hypothesis='Recent whole-phase load rates may become more relevant when observed HR exposure overlaps the independently saved requested band',
        match='Original duration/preceding-duration affinity times known-time requested-band overlap; one source-session vote, no coefficients',
        keep='Full nonlinear phase load including HR startup; no sample-only high-zone load substitution; unsupported overlap explicitly retains baseline',
        fixed='Existing30-day window, source ownership, HR settings, accounting, partition and observed coverage',
        limits=['Observed HR exposure is not commanded intensity or equilibrium demand',
            'Short efforts retain physiological startup lag; recovery may retain high HR from work',
            'Four known efforts and realized timing, not actual-issue or all-format accuracy',
            'Previously inspected development evidence; no parameter sweep or historical-intent reconstruction',
            'Nonwork cues can be inherited opening scenarios; not all independently prescribed',
            'Unknown gaps remain unknown; zone edges follow existing continuous lower-edge convention'],
        gate='Keep all baseline supported known-intent cases; improve whole-load MAE overall and each represented role; no worse work-phase MAE',
        planner_influence=False)
    with (args.output/'protocol.json').open('x') as stream:json.dump(protocol,stream,indent=2)
    manifest=freeze_sources(args.output/'sources.zip',config=config,entrypoint=__name__,
        inputs=(database,args.baseline_root/'baseline.json',args.output/'protocol.json'))
    assignments={}
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True) as db:
        for key,pid,saved,raw,start,*_ in matched_assignments(db,as_of):
            payload=json.loads(raw)
            try: issued_at=issue_boundary(saved,payload,datetime.fromisoformat(start))
            except ValueError:continue
            assignments[str(key)]=dict(payload=payload,issued_at=issued_at,prescription_id=pid)
    inventory=Inventory(config,assignments,{c['key']:c for c in parent['cases']})
    reference=QualityReference('threshold','work_seconds',Quantity.point(900,'unused historical comparator'),
        Quantity(290,310),'original-file evidence loading only')
    started=perf_counter()
    _,missing=load_history(database,config,as_of,reference,
        since=min(datetime.fromisoformat(c['start']) for c in parent['cases'])-timedelta(days=1),run_transform=inventory)
    if set(inventory.parents)!=set(c['key'] for c in inventory.cases):raise AssertionError('Missing quality case')
    groups={name:summarize(rows) for name,rows in (
        ('all',inventory.cases),('intervals',[c for c in inventory.cases if c['label']=='intervals']),
        ('tempo_threshold',[c for c in inventory.cases if c['label']=='tempo_threshold']))}
    gate=all(g['paired']==g['known_intent'] and g['paired']>0 and g['candidate']['mae']<g['baseline']['mae']
        and g['phases']['work']['candidate']['mae']<=g['phases']['work']['baseline']['mae'] for g in groups.values())
    result=dict(protocol=protocol,manifest=manifest,cases=inventory.cases,summary=groups,development_gate_passed=gate,
        source_missing=missing,elapsed_seconds=perf_counter()-started,baseline_exact=True,planner_influence=False,
        config_hr=dict(resting=config['resting_hr'],maximum=config['max_hr'],zones=config['zones']))
    with (args.output/'report.json').open('x') as stream:json.dump(result,stream,indent=2,default=encode,allow_nan=False)
    print(json.dumps(dict(summary=groups,gate=gate,seconds=result['elapsed_seconds']),indent=2),flush=True)


if __name__=='__main__':main()
