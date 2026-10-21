"""Read-only replacement-measurement audit on an attributed old replay.

The saved prescriptions, completions, dates and scores are not changed. New
calendar metadata is built in memory and associated with ORIGINAL issued
fingerprints. This is a measurement diagnostic, NOT a new planner replay.
"""
from argparse import ArgumentParser
from datetime import datetime
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from scripts.replay_adaptive_search import load_decision
from scripts.research_manifest import freeze_sources
from run_analysis.adaptive.adaptive_stage_lifecycle import review_entries
from run_analysis.adaptive.execution_upload_sandbox import ExecutionUploadSandbox
from run_analysis.adaptive.realized_reference_runtime import allocate, signal


def audit(root, output):
    report=json.loads((root/'lifecycle-summary.json').read_text())
    events=report['events']; first=events[0]
    old=first['realized_reference_allocation']['config']
    config=dict(old,version='calendar_mass_reference_v2',baseline_key='calendar-mass:'+old['available_at'])
    db=sqlite3.connect((root/'sandbox.sqlite').resolve().as_uri()+'?mode=ro',uri=True)
    db.row_factory=sqlite3.Row
    journal=ExecutionUploadSandbox.__new__(ExecutionUploadSandbox);journal.db=db
    past=[json.loads(p.read_text()) for p in sorted(root.glob('decision-*.json'))]
    past=[e for e in past if e['index']<first['index']]
    translated=[]; rows=[]
    try:
        for event in events:
            case=load_decision(root,event['index']+1)
            state=case['state']
            block=allocate(state,tuple(map(tuple,event['program'])),case['physical'],config,
                reference=None,prior_events=translated)
            block['allocations']=[dict(r,issued_fingerprint=event['active'][r['opportunity'][5:]]['fingerprint'])
                for r in block['allocations']]
            translated.append(dict(event,realized_reference_allocation=block))
            clock=datetime.fromisoformat(event['as_of'])
            h=SimpleNamespace(events=past+translated,sandbox=journal,zone=ZoneInfo(state.timezone),
                catalog=case['catalog'],seed_review_entries=(),stage=SimpleNamespace(policy=SimpleNamespace(review_days=14)))
            observed=signal(h,review_entries(h,clock,original_schedule=True),config,clock)
            old_signal=event['progression_target']['reference_signal']
            rows.append(dict(event=event['index'],as_of=event['as_of'],status=observed['status'],
                old_normalized_volume=(old_signal['level_ratio']*config['volume'] if old_signal['level_ratio'] is not None else None),
                new_measured_volume=observed.get('measured_volume_per_7'),growth_per_7=observed['growth_per_7'],
                completed_spans=observed.get('closed_spans',[]),unresolved=observed['unresolved'],
                original_winner_reproduced=True))
    finally:db.close()
    output.mkdir(parents=True,exist_ok=True)
    manifest=freeze_sources(output/'measurement.sources.zip',config=config,entrypoint=__name__,
        inputs=(root/'lifecycle-summary.json',root/'sandbox.sqlite',root/'scenario.json')+
            tuple(root.glob('decision-*.json'))+tuple(root.glob('snapshot-*.json')))
    result=dict(status='measurement_only_not_reoptimized',source_status=report['status'],
        source_completed_days=report['completed_days'],rows=rows,manifest=manifest,live_changed=False,
        limitation='No future actuals manufactured; a twelve-day stopped course does not identify two full cadence spans')
    with (output/'measurement.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(dict(status=result['status'],decisions=len(rows),last=rows[-1])))


if __name__=='__main__':
    p=ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('output',type=Path)
    args=p.parse_args();audit(args.root,args.output)
