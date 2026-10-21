"""Q5 chronological executed-demand signal probe, not prescription acceptance.

Completed HR is deliberately a target query input for this diagnostic. The pace
models and response times are prior-only; historical intent remains unknown.
No search, coefficient sweep, live forecast or scheduler integration.
"""
from argparse import ArgumentParser
from collections import Counter
from datetime import datetime
import json
from math import fsum,log
from pathlib import Path
import sqlite3
from statistics import mean
from zipfile import ZipFile

import yaml

from run_analysis.adaptive.effort_pace_reference import exposure_band
from run_analysis.adaptive.lag_demand_reference import (
    DemandSample,duration_reference,executed_demand,fit_pace,predict_pace,
)
from run_analysis.hr_response import fit_response
from run_analysis.movement import classify_movement
from run_analysis.phase_partitions import detailed_layout
from run_analysis.projection_evidence import steady_pace_samples
from run_analysis.signal_structure import SignalSpan
from run_analysis.training_signals import HeartRateReference
from scripts.audit_direct_quality_load import DirectAudit
from scripts.audit_issued_quality_load import matched_assignments,issue_boundary
from scripts.audit_work_stimulus import encode
from scripts.compare_effort_pace_reference import assert_parent_unchanged
from scripts.compare_requested_quality_response import saved_effort
from scripts.research_manifest import freeze_sources
from scripts.validate_quality_load import audit

CONTROLS=('duration_kernel','duration_regression','mean_hrr_regression','demand_regression')


def metrics(predicted,actual):
    return dict(speed_fraction_error=predicted/actual-1,
        pace_fraction_error=actual/predicted-1,absolute_log_error=abs(log(predicted/actual)))


class Comparison:
    def __init__(self,config,assignments):
        self.config,self.assignments=config,assignments
        self.hr=HeartRateReference(config['resting_hr'],config['max_hr'],'frozen accounting')
        self.scale=config['max_hr']-config['resting_hr']
        self.history,self.sources=[],[]
        self.pending={}
        self.direct=DirectAudit()

    def observation(self,spans,left,right,speed,fit,kind,basis):
        model=fit['dynamic']
        measurement=(executed_demand(spans,left,right,tau_seconds=model['tau_seconds'],bpm_span=self.scale)
            if model is not None else dict(status='no_prior_response',demand=None,mean_hrr=None))
        return dict(start_seconds=left,end_seconds=right,seconds=right-left,kind=kind,
            speed_mps=speed,speed_basis=basis,demand_measurement=measurement,
            response_source_keys=fit['source_keys'],response_fit_origin=fit['issued_at'],
            tau_at_search_boundary=model['tau_at_search_boundary'] if model else None)

    def __call__(self,case,run,layout,actual,rows,predictions,donors,scale,response_history):
        self.direct(case,run,layout,actual,rows,predictions,donors,scale,response_history)
        models={name:fit_pace(self.history,as_of=run.start,accounting_key=run.accounting_key,channel=channel)
            for name,channel in (('duration_regression','duration'),('mean_hrr_regression','mean_hrr'),
                                 ('demand_regression','demand'))}
        if len({m['sample_keys'] for m in models.values()})!=1:
            raise AssertionError('Feature comparisons must share identical prior sample support')
        measurements=[]
        work=[(i,a,b) for i,(kind,a,b) in enumerate(layout) if kind=='work']
        if len(work)!=len(run.bouts):
            case['lag_reference_diagnostic']=dict(status='unresolved_work_alignment',records=[],models=models,
                uses_completed_hr=True,prospective=False)
            self.pending[run.key]=(case,case['fit'],measurements)
            return
        for bout,(i,a,b) in zip(run.bouts,work):
            covered=fsum(max(0.,min(b,r.end)-max(a,r.start)) for r in rows)
            speed=(bout.distance_meters/(b-a) if bout.distance_meters is not None else
                fsum(max(0.,min(b,r.end)-max(a,r.start))*r.speed for r in rows)/covered
                if covered>0 and abs(covered-(b-a))<1e-5 else None)
            observation=self.observation(actual,a,b,speed,case['fit'],'detected_work',
                'supported_bout_distance_over_duration' if bout.distance_meters is not None else
                'complete_known_work_speed_trace')
            measurement=observation['demand_measurement']
            outputs={}
            if speed is not None and speed>0 and measurement['status']=='inferred_model_demand':
                outputs['duration_kernel']=duration_reference(self.history,as_of=run.start,
                    accounting_key=run.accounting_key,seconds=b-a)
                for name,model in models.items():
                    intensity=(measurement['mean_hrr'] if model['channel']=='mean_hrr' else
                        measurement['demand'] if model['channel']=='demand' else None)
                    outputs[name]=predict_pace(model,issued_at=run.start,seconds=b-a,intensity=intensity)
                for result in outputs.values():
                    result['metrics']=metrics(result['speed_mps'],speed) if result['speed_mps'] else None
            observation['predictions']=outputs
            old=case.get('programs',{}).get('model')
            observation['original_quality_duration_reference']=(dict(speed_mps=old[i][1],
                metrics=metrics(old[i][1],speed)) if old and speed is not None and speed>0 else None)
            issued=self.assignments.get(run.key)
            observation['saved_effort_sensitivity']=dict(status='unknown_historical_intent',profiles=None)
            if issued is not None:
                effort=saved_effort(issued['payload'],'work')
                bands=[exposure_band(cues,self.config) for cues in (effort.start_cues,effort.finish_cues)] if effort else None
                model=models['demand_regression']
                if bands and all(bands):
                    profiles=[[predict_pace(model,issued_at=run.start,seconds=b-a,intensity=value)
                               for value in band] for band in bands]
                    observation['saved_effort_sensitivity']=dict(status='declared_full_band_sensitivity',
                        requested_bands=bands,profiles=profiles,saved_issue=issued['issue'],
                        assumption='Requested effort interpreted as equivalent model demand; NOT demonstrated',
                        uses_realized_duration=True,uncertainty_calibrated=False)
                else:
                    observation['saved_effort_sensitivity']=dict(status='unknown_numeric_cues',profiles=None)
            measurements.append(observation)
        case['lag_reference_diagnostic']=dict(status='measured_diagnostic' if measurements else 'no_work',
            records=measurements,models=models,uses_completed_hr=True,uses_realized_timing=True,
            prospective=False,history_source_keys=tuple(sorted({r.source_key for r in self.history})))
        self.pending[run.key]=(case,case['fit'],measurements)

    def source(self,row,activity,rows,accounting_key,eligible,prior_response_history):
        previous=self.pending.pop(row.key,None)
        if previous:
            case,fit,measurements=previous
            case['lag_reference_diagnostic']['target_eligible']=eligible
        else:
            fit=fit_response(prior_response_history,as_of=row.start,accounting_key=accounting_key,lookback_days=30.)
            intervals=classify_movement(activity.trackpoints,self.config['moving_time']).intervals
            windows=steady_pace_samples(intervals,activity.start_time_utc,self.config,self.hr,with_windows=True)
            spans=tuple(SignalSpan(r.start,r.end,r.first,r.last) for r in rows)
            measurements=[self.observation(spans,a,b,p.meters/p.seconds,fit,'steady_moving',p.basis)
                          for a,b,p in windows]
        source=dict(key=row.key,start=row.start,end=row.end,available_at=row.available_at,
            eligible=eligible,response_origin=fit['issued_at'],response_source_keys=fit['source_keys'],
            measurement_statuses=dict(Counter(m['demand_measurement']['status'] for m in measurements)),
            measurement_count=len(measurements),accepted_samples=0)
        for index,m in enumerate(measurements):
            demand=m['demand_measurement']
            if not eligible or demand['status']!='inferred_model_demand' or not m['speed_mps'] or m['speed_mps']<=0:
                continue
            self.history.append(DemandSample(f'{row.key}:{index}',row.key,row.start,row.end,row.available_at,
                accounting_key,m['seconds'],m['speed_mps'],demand['mean_hrr'],demand['demand'],m['kind']))
            source['accepted_samples']+=1
        self.sources.append(source)


def summarize(cases):
    def records(case):
        return case['lag_reference_diagnostic']['records']
    valid=[(c,[r for r in records(c) if all(r['predictions'].get(k,{}).get('metrics') for k in CONTROLS)])
           for c in cases]
    valid=[(c,rows) for c,rows in valid if rows]
    result=dict(total_cases=len(cases),common_cases=len(valid),common_bouts=sum(len(rows) for _,rows in valid),
        common_case_keys=[c['key'] for c,_ in valid],controls={},
        demand_statuses=dict(Counter(r['demand_measurement']['status'] for c in cases for r in records(c))),
        model_statuses={name:dict(Counter(c['lag_reference_diagnostic']['models'][name]['status'] for c in cases))
                        for name in CONTROLS if name!='duration_kernel'})
    for name in CONTROLS:
        run_errors=[]
        for c,rows in valid:
            seconds=fsum(r['seconds'] for r in rows)
            run_errors.append({metric:fsum(r['seconds']*(abs(r['predictions'][name]['metrics'][metric])
                if metric=='pace_fraction_error' else r['predictions'][name]['metrics'][metric]) for r in rows)/seconds
                for metric in ('pace_fraction_error','absolute_log_error')})
        result['controls'][name]=dict(n=len(run_errors),
            pace_mae=mean(r['pace_fraction_error'] for r in run_errors) if run_errors else None,
            mean_absolute_log_error=mean(r['absolute_log_error'] for r in run_errors) if run_errors else None,
            supported_bouts=sum(bool(r['predictions'].get(name,{}).get('metrics')) for c in cases for r in records(c)))
    original=[]
    for c,rows in valid:
        if all(r['original_quality_duration_reference'] for r in rows):
            seconds=fsum(r['seconds'] for r in rows)
            original.append(fsum(r['seconds']*abs(r['original_quality_duration_reference']['metrics']['pace_fraction_error'])
                                 for r in rows)/seconds)
    result['original_duration_reference_on_common_bouts']=dict(n=len(original),pace_mae=mean(original) if original else None)
    inferred=[r for c in cases for r in records(c) if r['demand_measurement'].get('demand') is not None]
    result['inferred_demand_range']=(min(r['demand_measurement']['demand'] for r in inferred),
        max(r['demand_measurement']['demand'] for r in inferred)) if inferred else None
    return result


def signal_gate(summary):
    overall=summary['all']
    controls=overall['controls']
    same_support=(controls['demand_regression']['supported_bouts']==controls['mean_hrr_regression']['supported_bouts']
                  ==controls['duration_regression']['supported_bouts']==overall['common_bouts']>0)
    def better(group):
        rows=group['controls']
        return bool(group['common_cases'] and rows['demand_regression']['mean_absolute_log_error']
            < min(rows[name]['mean_absolute_log_error'] for name in ('duration_regression','mean_hrr_regression')))
    return dict(same_target_support=same_support,
        improves_over_both_regression_controls=better(overall),
        improves_in_both_categories=all(better(summary[k]) for k in ('intervals','tempo_threshold')),
        diagnostic_signal_gate_passed=same_support and all(better(summary[k]) for k in ('all','intervals','tempo_threshold')),
        prospective_acceptance=False,planner_influence=False)


def main():
    p=ArgumentParser(description=__doc__)
    p.add_argument('--baseline-root',type=Path,required=True)
    p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--as-of',type=datetime.fromisoformat,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    database=args.baseline_root/'evidence.sqlite'
    with ZipFile(args.baseline_root/'sources.zip') as archive:
        config=yaml.safe_load(archive.read('config.yaml'))
    protocol=dict(version='q5_lag_demand_signal_v1',
        identity='Mean demand = mean measured HR reserve + preceding-fit tau * endpoint HRR change / elapsed duration',
        source_pool='All eligible nonquality steady windows plus detected quality work; no overlapping steady/work samples per source',
        models={'duration_regression':'log speed = intercept + learned log-duration coefficient',
            'mean_hrr_regression':'log speed = intercept + learned mean-HRR coefficient + learned log-duration coefficient',
            'demand_regression':'log speed = intercept + learned inferred-demand coefficient + learned log-duration coefficient',
            'duration_kernel':'Existing log-duration geometric reference, same supported source/sample cohort'},
        fixed=('Original prior30-day response fits/tau grid/HR settings, all original quality outputs exact',
            'Preceding30-day supported sample pool, each source run owns one total regression vote',
            'No regularization, family features, constrained duration slope, coefficient/window sweep or stale fallback',
            'Positive intensity slope required; unidentified/nonpositive fit remains unsupported, not clipped',
            'Complete HR interval required; gaps remain unknown; inferred demand outside [0,1] retained with a model-domain warning, not invalid actual HR',
            'Primary matched comparisons: mean absolute log pace error, elapsed-weighted within run, equal run ownership',
            'Report pace percentage error and original quality-only duration reference on identical target bouts'),
        gate='Same target support and lower absolute log pace error than mean-HR and duration regression overall and both categories',
        limits=('Completed target HR and realized timing are query inputs: pace reconstruction only, NOT a prospective forecast',
            'The demand estimate assumes the existing first-order response; it is not measured intention or a physiological threshold',
            'Demand derivation may reconstruct the prior speed/HR fit algebraically; low error alone does not establish independent physiology',
            'One-bpm perturbation sensitivity is NOT assumed device error or calibrated uncertainty',
            'Full requested-zone endpoint sensitivities on four saved intents are declared scenarios, not confidence bounds or validated execution',
            'Current labels/HR settings; source availability assumed at completion; repeatedly inspected single-athlete data',
            'Per-athlete fitted speed scale; no multi-athlete/population prior or zero-history quality forecast validated',
            'No measured-load accounting change, calendar search, recovery inference, integration or deployment'))
    with (args.output/'protocol.json').open('x') as stream:
        json.dump(protocol,stream,indent=2)
    manifest=freeze_sources(args.output/'sources.zip',config=config,entrypoint=__name__,
        inputs=(database,args.parent,args.output/'protocol.json'))
    assignments={}
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True) as db:
        for key,pid,saved,payload,started,*_ in matched_assignments(db,args.as_of):
            payload=json.loads(payload)
            assignments[str(key)]=dict(payload=payload,prescription_id=pid,
                issue=issue_boundary(saved,payload,datetime.fromisoformat(started)))
    observer=Comparison(config,assignments)
    result=audit(database,config,args.as_of,case_observer=observer,source_observer=observer.source)
    assert_parent_unchanged(result,json.loads(args.parent.read_text()))
    summary={name:summarize(rows) for name,rows in (('all',result['cases']),
        ('intervals',[c for c in result['cases'] if c['label']=='intervals']),
        ('tempo_threshold',[c for c in result['cases'] if c['label']=='tempo_threshold']))}
    result.update(protocol=protocol,manifest=manifest,lag_reference_summary=summary,signal_gate=signal_gate(summary),
        source_inventory=observer.sources,supported_samples=observer.history,original_baseline_unchanged=True)
    with (args.output/'comparison.json').open('x') as stream:
        json.dump(result,stream,default=encode,indent=2,allow_nan=False)
    print(json.dumps(dict(summary=summary,gate=result['signal_gate'],sources=len(observer.sources),
        accepted_samples=len(observer.history),seconds=result['elapsed_seconds']),indent=2))


if __name__=='__main__':
    main()
