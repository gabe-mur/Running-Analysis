"""One corrected donor upload through the real parser/importer, without search.

Original failed replay is read-only. This isolated sandbox imports an unplanned
activity, independently checks measured purpose and reconciles the old issued
primary contract. Full publication/restart/calendar acceptance is separate.
"""
from argparse import ArgumentParser
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

from run_analysis.config import load_config,resolve_project_path
from run_analysis.adaptive.execution_upload_sandbox import ExecutionUploadSandbox
from run_analysis.adaptive.quality_assignment_accounting import reconcile_assignment
from run_analysis.adaptive.quality_catalogue_runtime import completion_signals,read_contract
from run_analysis.adaptive.quality_purpose_contract import assess_quality_purposes
from run_analysis.adaptive.program_evidence_adapter import read_accounted
from scripts.recorded_trace_execution import donor,transformed_payload
from scripts.replay_adaptive_search import load_decision
from scripts.research_manifest import freeze_sources


def run(root,before,after,execution,output):
    prior=load_decision(root,before); post=load_decision(root,after)
    recorded=json.loads((root/execution).read_text())
    actual_key=recorded['comparison']['actual_key']
    receipt=next(r for r in post['state'].quality_completions if r.activity_key==actual_key)
    old=next(r for r in post['state'].history if r.key==actual_key)
    dose=prior['catalog'][receipt.assignment.prescription_key]
    original=next(r for r in prior['state'].history if r.key==recorded['provenance']['source_key'])
    config=load_config()
    if (config['resting_hr'],config['max_hr'])!=(post['state'].resting_hr,post['state'].maximum_hr):
        raise ValueError('Fixture must retain archived HR references')
    output.mkdir(exist_ok=False)
    database=resolve_project_path(Path.cwd(),config['paths']['database'])
    sandbox=ExecutionUploadSandbox(output/'runtime',config,create=True)
    try:
        source=donor(database,original,config,sandbox.reference)
        if source['source_sha256']!=recorded['provenance']['source_sha256']:
            raise ValueError('Original HR donor bytes changed')
        payload,phases,end,provenance=transformed_payload(source,dose,old.start,
            distance_factor=recorded['provenance']['distance_factor'],
            duration_factor=recorded['provenance']['duration_factor'])
        imported=sandbox.upload(payload,fingerprint=None,available_at=end,
            observed_phases=phases,observed_role='quality',phase_evidence='explicit')
        actual=read_accounted(json.loads(sandbox.completion_rows(end)[0]['accounted']))
        signals,_=completion_signals(SimpleNamespace(sandbox=sandbox,config=config,
            ref=sandbox.reference),end)
        contract=read_contract(prior['event']['stage_runtime']['quality_runtime']['purpose_contract'])
        purpose=assess_quality_purposes(signals[actual.key],contract=contract,role=actual.role,as_of=end)
        q=purpose['credits'][receipt.assignment.purpose]
        matched=True if q.low>0 else None if q.high>0 else False
        corrected=reconcile_assignment(receipt.assignment,actual,purpose_matched=matched)
        corrected.for_run(actual)
        if matched is not True or corrected.credit.low<1.-1e-7:
            raise ValueError('Corrected fixture does not verify full primary completion')
        if actual.demand!=old.demand:
            raise ValueError('Distance primary correction unexpectedly changed measured physical demands')
        report=dict(status='primary_and_measured_purpose_verified',
            original_credit=asdict(receipt.credit),corrected_receipt=asdict(corrected),
            physical_demands_unchanged=True,demand=asdict(actual)['demand'],
            observed_work_meters=[b.distance_meters for b in actual.bouts],
            hr_provenance=provenance,import_status=imported['status'],
            association='unplanned import plus original issued-contract diagnostic; no calendar publication',
            live_changed=False,manifest=freeze_sources(output/'sources.zip',config=config,
                entrypoint=__name__,inputs=(root/execution,
                    root/f'decision-{before:04d}.json',root/prior['event']['snapshot_file'],
                    root/f'decision-{after:04d}.json',root/post['event']['snapshot_file'])))
        with (output/'report.json').open('x') as stream:
            json.dump(report,stream,default=str,indent=2,allow_nan=False)
        print(json.dumps(dict(status=report['status'],credit=corrected.credit.low,
            work_meters=report['observed_work_meters'],physical_demands_unchanged=True)))
    finally:sandbox.close()


if __name__=='__main__':
    p=ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path);p.add_argument('--before',required=True,type=int)
    p.add_argument('--after',required=True,type=int);p.add_argument('--execution',required=True)
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();run(a.root,a.before,a.after,a.execution,a.output)
