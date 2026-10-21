"""One frozen effort contrast and exposure-reference repair, not a planner test."""
from argparse import ArgumentParser
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import sqlite3
from statistics import mean
from zipfile import ZipFile

import yaml

from run_analysis.adaptive.direct_effort_load import (
    ExposureHistory,effort_trace,forecast_effort_contrast,phase_exposures)
from run_analysis.adaptive.direct_quality_load import phase_observations,score_direct
from run_analysis.adaptive.effort_pace_reference import exposure_band
from run_analysis.adaptive.prescribed_hr_target import prescribed_hr_target
from run_analysis.adaptive.lag_demand_reference import DemandSample,fit_exposure_pace,predict_exposure_pace
from run_analysis.phase_partitions import detailed_layout
from scripts.audit_direct_quality_load import DirectAudit,phase_errors
from scripts.audit_issued_quality_load import matched_assignments,issue_boundary
from scripts.audit_work_stimulus import encode
from scripts.compare_effort_pace_reference import assert_parent_unchanged
from scripts.compare_requested_quality_response import saved_effort,trace_metrics
from scripts.research_manifest import freeze_sources
from scripts.validate_quality_load import audit,INITIAL_RESERVE


def repaired_pace_diagnostic(parent):
    samples = [DemandSample(**dict(r,**{k:datetime.fromisoformat(r[k])
        for k in ('start','end','available_at')})) for r in parent['supported_samples']]
    result=[]
    for case in parent['cases']:
        origin=datetime.fromisoformat(case['start'])
        models={channel:fit_exposure_pace(samples,as_of=origin,accounting_key=case['fit']['accounting_key'],channel=channel)
                for channel in ('mean_hrr','demand')}
        records=[]
        for record in case['lag_reference_diagnostic']['records']:
            measurement=record['demand_measurement'];predictions={}
            if measurement['status']=='inferred_model_demand' and record['speed_mps']:
                for channel,model in models.items():
                    prediction=predict_exposure_pace(model,issued_at=origin,intensity=measurement[channel])
                    actual=record['speed_mps'];speed=prediction['speed_mps']
                    prediction['pace_fraction_error']=actual/speed-1 if speed else None
                    predictions[channel]=prediction
            records.append(dict(seconds=record['seconds'],predictions=predictions))
        result.append(dict(key=case['key'],label=case['label'],models=models,records=records))
    return dict(cases=result,statuses={channel:dict(Counter(c['models'][channel]['status'] for c in result))
        for channel in ('mean_hrr','demand')},supported_bouts={channel:sum(r['predictions'].get(channel,{}).get(
            'pace_fraction_error') is not None for c in result for r in c['records']) for channel in ('mean_hrr','demand')},
        capacity_duration_inferred=False,prospective=False,used_by_load_candidate=False)


def requested_targets(effort, config, mapping):
    if effort is None:
        return None
    if mapping == 'prescription':
        targets = [prescribed_hr_target(cues, config) for cues in (effort.start_cues, effort.finish_cues)]
        return targets if all(targets) else None
    bands = [exposure_band(cues, config) for cues in (effort.start_cues, effort.finish_cues)]
    return [dict(low=b[0], nominal=sum(b)/2, high=b[1]) for b in bands] if all(bands) else None


class Comparison:
    def __init__(self,config,assignments,mapping='full-envelope'):
        self.config,self.assignments=config,assignments
        self.mapping=mapping
        self.direct=DirectAudit();self.exposures=[]

    def __call__(self,case,run,layout,actual,rows,predictions,donors,scale,response_history):
        # Save strictly prior empirical phase history; DirectAudit appends target
        # after its own prediction, before this callback resumes.
        prior=tuple(self.direct.history)
        self.direct(case,run,layout,actual,rows,predictions,donors,scale,response_history)
        issued=self.assignments.get(run.key)
        query=dict(status='unknown_historical_intent',scenarios={},nominal=None,
            uses_target_hr=False,uses_target_speed=False,uses_realized_timing=True,prospective=False)
        if issued is not None:
            query.update(status='unresolved_numeric_command',prescription_id=issued['prescription_id'],
                command_issued_at=issued['issue'])
            bands=[];targets=[];profiles=[]
            for kind,a,b in detailed_layout(layout):
                effort=saved_effort(issued['payload'],kind)
                values=requested_targets(effort,self.config,self.mapping)
                if not values:
                    break
                def field(v,name):
                    return v[name] if isinstance(v,dict) else getattr(v,name)
                bands.append([(field(v,'low'),field(v,'high')) for v in values])
                targets.append(values)
                profiles.append(dict(kind=kind,effort=effort,targets=values))
            fit=case['fit']['dynamic']
            if layout and len(bands)==len(layout) and fit is not None:
                query.update(status='evaluated',requested_bands=bands,requested_profiles=profiles,tau_seconds=fit['tau_seconds'],
                    mapping=('Prescription-specific targets; equilibrium-input interpretation unchanged and unvalidated'
                        if self.mapping=='prescription' else
                        'Full configured zone envelope as equilibrium-input scenarios; midpoint nominal is an unvalidated assumption'),
                    initial_reserve=INITIAL_RESERVE)
                observations=phase_observations(layout,actual)
                for point in ('lower','nominal','upper'):
                    name='low' if point=='lower' else 'high' if point=='upper' else 'nominal'
                    inputs=tuple((field(start,name),field(finish,name)) for start,finish in targets)
                    forecast=forecast_effort_contrast(layout,inputs,prior,self.exposures,
                        as_of=run.start,accounting_key=case['fit']['accounting_key'],
                        tau_seconds=fit['tau_seconds'],initial_reserve=INITIAL_RESERVE)
                    raw,valid=effort_trace(layout,inputs,tau_seconds=fit['tau_seconds'],initial_reserve=INITIAL_RESERVE)
                    raw_metrics=trace_metrics(raw,actual,layout,scale) if valid else None
                    scenario=dict(requested_inputs=inputs,forecast=forecast,metrics=score_direct(forecast,actual),
                        detailed_phase_errors=phase_errors(forecast,observations),
                        raw_forward_control=raw_metrics,raw_forward_hr_valid=valid)
                    query['scenarios'][point]=scenario
                    if point=='nominal':
                        query['nominal']=scenario
            elif fit is None:
                query['status']='no_prior_response_time'
        case['direct_effort_trial']=query
        # Actual moments enter history only after all target query/scoring.
        self.exposures.append(ExposureHistory(run.key,run.start,run.end,run.available_at,
            case['fit']['accounting_key'],phase_exposures(layout,actual)))


def summary(cases):
    known=[c for c in cases if c['direct_effort_trial']['status']!='unknown_historical_intent']
    paired=[c for c in known if c['direct_effort_trial']['nominal']
        and c['direct_effort_trial']['nominal']['metrics']
        and c['direct_effort_trial']['nominal']['metrics']['fraction_error'] is not None
        and c['direct_load']['metrics'] and c['direct_load']['metrics']['fraction_error'] is not None]
    def metric(rows):
        return dict(n=len(rows),mae=mean(abs(r['fraction_error']) for r in rows) if rows else None,
            bias=mean(r['fraction_error'] for r in rows) if rows else None)
    result=dict(total=len(cases),known_intent=len(known),paired=len(paired),
        statuses=dict(Counter(c['direct_effort_trial']['status'] for c in cases)),
        forecast_statuses=dict(Counter(c['direct_effort_trial']['nominal']['forecast']['status']
            for c in known if c['direct_effort_trial']['nominal'])),
        current_direct=metric([c['direct_load']['metrics'] for c in paired]),
        candidate=metric([c['direct_effort_trial']['nominal']['metrics'] for c in paired]),phases={})
    for kind in ('warmup','work','recovery','cooldown'):
        old=[c['direct_load']['detailed_phase_errors'].get(kind) for c in paired]
        new=[c['direct_effort_trial']['nominal']['detailed_phase_errors'].get(kind) for c in paired]
        pairs=[(a,b) for a,b in zip(old,new) if a and b and a['fraction_error'] is not None and b['fraction_error'] is not None]
        result['phases'][kind]=dict(current_direct=metric([a for a,b in pairs]),candidate=metric([b for a,b in pairs]))
    result['coverage_preserved']=bool(known) and len(paired)==len(known)
    result['whole_improves']=bool(paired) and result['candidate']['mae'] < result['current_direct']['mae']
    result['no_phase_worsens']=bool(paired) and all(v['candidate']['mae']<=v['current_direct']['mae']
        for v in result['phases'].values() if v['candidate']['n'])
    return result


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-root',type=Path,required=True)
    parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--as-of',type=datetime.fromisoformat,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mapping',choices=('full-envelope','prescription'),default='full-envelope')
    args=parser.parse_args();args.output.mkdir(exist_ok=False,parents=True)
    database=args.baseline_root/'evidence.sqlite'
    with ZipFile(args.baseline_root/'sources.zip') as archive:
        config=yaml.safe_load(archive.read('config.yaml'))
    protocol=dict(version='q5_direct_effort_contrast_v1',
        candidate='Existing empirical phase cost multiplied by requested/reference modeled load ratio at identical timing, tau and initial state',
        fixed=('Existing30-day policy, duration/preceding-duration similarity and equal-source ownership',
            'Reference inputs recomputed from supported prior phase HR integrals/endpoint flux at current preceding-fit tau',
            'Unknown gaps omitted from moments, not filled; partial support explicitly retained',
            'Saved numeric command envelopes as input scenarios; nominal midpoint remains an unvalidated command mapping',
            'No pace inversion, family coefficient, fitted blend, sweep or cutoff tuning',
            'All36 historical quality cases retained; unknown intent not reconstructed from actual HR',
            'Nonwork easy-opening cue is an explicit inherited scenario, not verified recovery execution',
            'Current baseline/Q4 outputs exactly unchanged; held-out moments enter history after scoring'),
        gate='Same known-command coverage and lower whole-load MAE overall AND both categories, without worsening matched phase MAE',
        scope='Development diagnostic only, not prospective/independent acceptance or new-user calibration',
        limits=('Realized timing and run-start response fit, not exact issuance-time forecast',
            'Four saved commands,32 unknown; actual execution need not match instructions',
            'Input interpretation, transfer of relative modeled contrast and unknown-time representativeness are explicit assumptions',
            'InitialHRR.45 retained; zone extrema/three scenarios are not calibrated confidence intervals',
            'Corrected descriptive pace reference is separate and cannot feed this load candidate',
            'No search, recovery/scoring weight change or deployment'))
    if args.mapping=='prescription':
        protocol['version']='q5_direct_effort_prescription_mapping_v1'
        protocol['mapping']=dict(unqualified='Full requested range midpoint',
            low_upper='Center of lower/upper half; literal bottom/top uses edge',
            adjacent_upper_low='Boundary between adjacent zones, midpoint of configured BPM gap',
            progression='Keep independently saved start/finish ordering and linear ramp',
            parameter_selection='Authored semantic conventions; no target HR, pace or residual input',
            unchanged='Same forward-input meaning, predictor, history, tau, initial state, contrast and gate')
        protocol['fixed']=tuple('Prescription-qualified numeric targets; fixed semantic mapping before scoring'
            if s.startswith('Saved numeric command') else s for s in protocol['fixed'])
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
    parent=json.loads(args.parent.read_text())
    repaired=repaired_pace_diagnostic(parent)
    result=audit(database,config,args.as_of,case_observer=Comparison(config,assignments,args.mapping))
    assert_parent_unchanged(result,parent)
    groups={name:summary(rows) for name,rows in (('all',result['cases']),
        ('intervals',[c for c in result['cases'] if c['label']=='intervals']),
        ('tempo_threshold',[c for c in result['cases'] if c['label']=='tempo_threshold']))}
    gate=all(s['coverage_preserved'] and s['whole_improves'] and s['no_phase_worsens'] for s in groups.values())
    result.update(protocol=protocol,manifest=manifest,repaired_exposure_reference=repaired,
        effort_contrast_summary=groups,development_gate_passed=gate,prospective_acceptance=False,
        planner_influence=False,original_baseline_unchanged=True)
    with (args.output/'comparison.json').open('x') as stream:
        json.dump(result,stream,default=encode,indent=2,allow_nan=False)
    print(json.dumps(dict(summary=groups,gate=gate,repair_statuses=repaired['statuses'],
        repair_supported_bouts=repaired['supported_bouts'],seconds=result['elapsed_seconds']),indent=2))


if __name__=='__main__':
    main()
