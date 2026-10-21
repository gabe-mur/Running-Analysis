"""Read-only archived shadow rates and explicitly copied V2 opening search.

Shadow rates use each original frozen dated target, not a counterfactual V2
training trajectory. Existing source sandboxes/databases are opened read-only.
Only the separate output directory may receive reports or a normal test plan.
"""
from argparse import ArgumentParser
from dataclasses import replace
from datetime import date,datetime,timedelta
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from run_analysis.adaptive.adaptive_stage_lifecycle import review_entries
from run_analysis.adaptive.execution_upload_sandbox import ExecutionUploadSandbox
from run_analysis.adaptive.smoothed_general_progression import propose,VERSION
from run_analysis.adaptive.review_seed import read_entry
from run_analysis.stage_target_history import StageTarget
from run_analysis.staged_progression import DevelopmentStage,ProgressionPolicy
from scripts.research_manifest import freeze_sources


def target(raw):
    return StageTarget(**dict(raw,available_at=datetime.fromisoformat(raw['available_at']),
        effective_on=date.fromisoformat(raw['effective_on']),
        review_anchor=datetime.fromisoformat(raw['review_anchor']) if raw.get('review_anchor') else None))


def shadow(source,output):
    output.mkdir(parents=True,exist_ok=True)
    scenario=json.loads((source/'scenario.json').read_text())
    events=[json.loads(p.read_text()) for p in sorted(source.glob('decision-*.json'))]
    db=sqlite3.connect((source/'sandbox.sqlite').resolve().as_uri()+'?mode=ro',uri=True)
    db.row_factory=sqlite3.Row
    journal=ExecutionUploadSandbox.__new__(ExecutionUploadSandbox);journal.db=db
    marker=json.loads((source/'execution-sandbox.json').read_text())
    zone=ZoneInfo(marker['timezone'])
    rows=[]
    try:
        for before,event in zip(events,events[1:]):
            runtime=before.get('stage_runtime',{})
            previous=runtime.get('general_progression_state')
            if not previous or event['reason']!='continuous_general_review':continue
            at=datetime.fromisoformat(event['as_of'])
            stage=runtime['stage']
            stage=DevelopmentStage(**dict(stage,issued_at=datetime.fromisoformat(stage['issued_at']),
                policy=ProgressionPolicy(**stage['policy'])))
            available=[e for e in events if datetime.fromisoformat(e['as_of'])<=at]
            catalog={d['key']:SimpleNamespace(role=d['role']) for e in available
                for d in e.get('stage_runtime',scenario)['catalog']}
            h=SimpleNamespace(stage=stage,general_progression_state=previous,
                target_history=tuple(map(target,runtime['target_history'])),zone=zone,
                decision=lambda _,event=event:date.fromisoformat(event['decision']),
                policy=SimpleNamespace(progression_credit='actual_distance_v2',general_progression=VERSION),
                planner_preferences=None,events=available,catalog=catalog,sandbox=journal,
                seed_review_entries=tuple(map(read_entry,scenario.get('seed_review_entries',()))))
            entries=review_entries(h,at)
            longest=max((e.completion.distance_meters.low/1609.344 for e in entries
                if e.completion and e.completion.available_at<=at and e.completion.end<=at
                and at-timedelta(days=30)<=e.completion.start<at and e.completion.distance_meters),default=None)
            result=propose(h,at,entries,longest=longest)
            e=result['smoothed_support'];p=result['growth_proposal']
            rows.append(dict(index=event['index'],at=event['as_of'],
                old_rate=event['progression_target']['growth_per_7'],
                shadow_rate=p['growth_per_7'],support_fraction=p['support_fraction'],
                actual_per_7=e['actual_per_7'],target_per_7=e['target_per_7'],
                delivery_ratio=e['delivery_ratio'],legacy_window_status=result['legacy_window_status'],
                status=result['gate_status']))
    finally:db.close()
    manifest=freeze_sources(output/'shadow.sources.zip',config={},entrypoint=__name__,
        inputs=(source/'scenario.json',source/'sandbox.sqlite',source/'execution-sandbox.json')
            +tuple(source.glob('decision-*.json')))
    report=dict(status='shadow_only',rows=rows,manifest=manifest,live_changed=False,
        limitation='Original observed training and original dated targets; not a reoptimized V2 calendar, new stability pass, or physiological validation')
    with (output/'shadow.json').open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(dict(status=report['status'],decisions=len(rows),
        old_positive=sum(r['old_rate']>0 for r in rows),
        shadow_positive=sum(r['shadow_rate']>0 for r in rows),
        shadow_rate_range=[min(r['shadow_rate'] for r in rows),max(r['shadow_rate'] for r in rows)])),flush=True)


def opening(source,output,*,historical=False,matched=False,nominal=False,paired=False,realized=False,as_of=None):
    if sum((historical,matched,nominal,paired,realized))>1:raise ValueError('Select one explicit opening policy')
    from run_analysis.config import load_config
    from run_analysis.adaptive.hr_reference import frozen_live_config
    from run_analysis.adaptive.factory import resume_service
    from scripts.run_adaptive_lifecycle import fork_seed_plan
    from scripts.replay_adaptive_search import load_decision
    output.mkdir(exist_ok=True,parents=True)
    root=output/'opening';seed_at=fork_seed_plan(source,root)
    at=datetime.fromisoformat(as_of) if as_of else seed_at+timedelta(microseconds=1)
    if at<=seed_at:raise ValueError('New opening must follow the saved decision')
    config=frozen_live_config(load_config(),root=source)
    manifest=freeze_sources(output/'opening.sources.zip',config=config,entrypoint=__name__,
        inputs=tuple(source.glob('decision-*.json'))+(source/'scenario.json',source/'sandbox.sqlite'))
    service=resume_service(root,config,at)
    try:
        from run_analysis.adaptive.smoothed_general_progression import HISTORICAL_VERSION,MATCHED_VERSION,NOMINAL_VERSION,PRESCRIPTION_VERSION,REALIZED_VERSION
        policy=REALIZED_VERSION if realized else PRESCRIPTION_VERSION if paired else NOMINAL_VERSION if nominal else MATCHED_VERSION if matched else HISTORICAL_VERSION if historical else VERSION
        h=service.harness;h.policy=replace(h.policy,general_progression=policy)
        original=h.stage,h.catalog,h.scales,h.training_policy
        result=service.refresh(at);h=service.harness;event=result['event']
        assert (h.stage,h.catalog,h.scales,h.training_policy)==original
        case=load_decision(root,len(h.events))
        assert event['stage_runtime']['general_progression_state']['version']==policy
        assert case['evaluate_details'](tuple(map(tuple,event['program'])))['decision_eligible']
        report=dict(status='opening_only',manifest=manifest,policy=policy,
            seconds=event['seconds'],churn=event['churn'],miles21=event['volume'],
            progression_target=event['progression_target'],
            support=result['stage_review']['smoothed_support'],
            initialization=event['stage_runtime']['general_progression_state'].get('historical_initialization'),
            live_changed=False,
            schedule=[dict(day=d,role=h.catalog[k].role,miles=h.catalog[k].miles)
                for d,k in event['program'] if d<event['hard_end']])
        with (output/'opening.json').open('x') as f:json.dump(report,f,indent=2)
        print(json.dumps({k:v for k,v in report.items() if k not in {'manifest','support'}}),flush=True)
    finally:service.harness.sandbox.close()


if __name__=='__main__':
    p=ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('shadow','opening','historical-opening','matched-opening','nominal-opening','prescription-opening','realized-opening'))
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--as-of',help='Explicit aware opening time; no default host-date inference')
    a=p.parse_args()
    if a.mode=='shadow':shadow(a.source,a.output)
    else:opening(a.source,a.output,historical=a.mode=='historical-opening',
        matched=a.mode=='matched-opening',nominal=a.mode=='nominal-opening',paired=a.mode=='prescription-opening',
        realized=a.mode=='realized-opening',as_of=a.as_of)
