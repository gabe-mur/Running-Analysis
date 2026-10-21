"""All-option publication/restart/parser controls, NOT physiology or search.

Each upload is an independent synthetic structural probe, not a combined
training history or proof that automatic work detection matches a prescription.
"""
from argparse import ArgumentParser
from datetime import datetime,timedelta
import json
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

import yaml

from run_analysis.adaptive.execution_forecast_contract import issue_execution
from run_analysis.adaptive.execution_upload_sandbox import ExecutionUploadSandbox
from run_analysis.adaptive.new_model_daily_harness import DailyHarness,synthetic_payload
from run_analysis.adaptive.quality_catalogue_runtime import decision_evidence
from run_analysis.adaptive.quality_forecast_runtime import VERSION
from run_analysis.adaptive.quality_block_contract import read_quality_contract
from scripts.replay_adaptive_search import load_decision
from scripts.research_manifest import freeze_sources


def audit(seed,baseline,output):
    output.mkdir(exist_ok=False)
    started=perf_counter()
    case=load_decision(seed,1)  # Mandatory independent winner repricing.
    at=datetime.fromisoformat(case['event']['as_of'])
    with ZipFile(baseline/'sources.zip') as z:config=yaml.safe_load(z.read('config.yaml'))
    h=DailyHarness.resume(seed,config,as_of=at)
    sandbox=ExecutionUploadSandbox(output/'probes',config,create=True)
    manifest=freeze_sources(output/'sources.zip',config=config,entrypoint=__name__,
        inputs=(seed/'scenario.json',seed/'decision-0001.json',seed/case['event']['snapshot_file']))
    rows=[]
    try:
        h.quality_decision_evidence,_=decision_evidence(h,at)
        doses=[d for d in h.catalog.values() if d.workout_definition
            and d.workout_definition.get('calendar_integration')==VERSION]
        for i,dose in enumerate(doses):
            intent=h.intent(dose,h.decision(at),at,'probe:'+dose.key)
            expected=issue_execution(intent,h.phase_history)
            record=sandbox.publish(intent,h.phase_history)
            assert record==expected and record.payload['planning_handoff']['usable']
            p=read_quality_contract(record.payload['intent']['prescription_contract'])
            assert p.key==dose.key and p.miles==dose.miles
            assert p.definition()['structure']==dose.workout_definition['structure']
            # Non-overlapping ownership IDs; the probes are NOT prescribed days.
            start=at+timedelta(hours=1,minutes=i*2)
            payload,phases,end=synthetic_payload(dose,start,hr=160)
            result=sandbox.upload(payload,fingerprint=record.fingerprint,available_at=end,
                observed_role='quality',observed_phases=phases,phase_evidence='explicit')
            comparison=result['comparison']
            assert comparison['distance_adherence']=='within' and comparison['structure']=='same'
            assert comparison['actual_cardio']['estimate'] is not None
            sandbox.close();sandbox=ExecutionUploadSandbox(output/'probes',config)
            assert sandbox.record(record.fingerprint)==record
            retry=sandbox.upload(payload,fingerprint=record.fingerprint,available_at=end,
                observed_role='quality',observed_phases=phases,phase_evidence='explicit')
            assert retry['status']=='duplicate'
            rows.append(dict(key=dose.key,recipe=dose.workout_definition['recipe_key'],
                miles=dose.miles,load=record.payload['target_forecast']['predicted_load'],
                issued_fingerprint=record.fingerprint,structure=comparison['structure'],
                distance_adherence=comparison['distance_adherence']))
        report=dict(status='mechanical_controls_passed',options=len(rows),rows=rows,
            seconds=perf_counter()-started,manifest=manifest,live_changed=False,
            scope='Exact shared forecast, frozen units/steps, publication/restart, real TCX parser and duplicate upload; explicit synthetic physical phases, not automatic detection or predictive accuracy')
        (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
        print(json.dumps({k:v for k,v in report.items() if k not in ('rows','manifest')},indent=2),flush=True)
    finally:
        h.sandbox.close();sandbox.close()


if __name__=='__main__':
    p=ArgumentParser(description=__doc__)
    p.add_argument('--seed',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();audit(a.seed,a.baseline,a.output)
