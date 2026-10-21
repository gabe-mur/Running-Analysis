"""Read-only paired-prescription attribution on an existing archived replay."""
from argparse import ArgumentParser
from dataclasses import replace
from datetime import datetime,date,timedelta
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from run_analysis.adaptive.adaptive_stage_lifecycle import review_entries
from run_analysis.adaptive.execution_upload_sandbox import ExecutionUploadSandbox
from run_analysis.adaptive.prescription_progression import fulfillment,published_funding,growth
from run_analysis.adaptive.continuous_general_progression import propose
from run_analysis.adaptive.program_evidence_adapter import read_accounted
from run_analysis.adaptive.smoothed_general_progression import PRESCRIPTION_VERSION
from run_analysis.adaptive.review_seed import read_entry
from run_analysis.staged_progression import DevelopmentStage,ProgressionPolicy
from scripts.research_manifest import freeze_sources
from scripts.trial_smoothed_progression import target


def audit(root,output):
    output.mkdir(parents=True,exist_ok=True)
    summary=root/'resume-0001.summary.json';report=json.loads(summary.read_text())
    files=sorted(root.glob('decision-*.json'));events=[json.loads(path.read_text()) for path in files]
    scenario=json.loads((root/'scenario.json').read_text())
    reviews={row['at']:row for row in report['stage_reviews']}
    epoch=datetime.fromisoformat(report['events'][0]['as_of'])+timedelta(microseconds=1)
    zone=ZoneInfo(json.loads((root/'execution-sandbox.json').read_text())['timezone'])
    db=sqlite3.connect((root/'sandbox.sqlite').resolve().as_uri()+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
    sandbox=ExecutionUploadSandbox.__new__(ExecutionUploadSandbox);sandbox.db=db
    catalog={row['key']:SimpleNamespace(role=row['role']) for e in events
        for row in e.get('stage_runtime',scenario)['catalog']}
    rows=[];course=[]
    initial=report['events'][0];runtime=initial['stage_runtime'];raw=runtime['stage']
    physical_history=[read_accounted(row) for row in scenario['history']]
    physical_history.extend(read_accounted(json.loads(row[0]))
        for row in db.execute('SELECT accounted FROM sandbox_completions'))
    controller=SimpleNamespace(stage=DevelopmentStage(**dict(raw,
        issued_at=datetime.fromisoformat(raw['issued_at']),policy=ProgressionPolicy(**raw['policy']))),
        history=physical_history,capacity_eligibility=scenario['capacity_eligibility'],planner_preferences=None,
        zone=zone,target_history=tuple(map(target,runtime['target_history'])),
        general_progression_state=runtime['general_progression_state'],
        policy=SimpleNamespace(general_progression=PRESCRIPTION_VERSION,progression_credit='actual_distance_v2'))
    try:
        for event in report['events']:
            at=datetime.fromisoformat(event['as_of'])
            if event['index']==initial['index']:at=epoch
            h=SimpleNamespace(sandbox=sandbox,zone=zone,catalog=catalog,
                events=[e for e in events if e['index']<event['index']],
                seed_review_entries=tuple(read_entry(row) for row in scenario.get('seed_review_entries',())))
            entries=review_entries(h,at,original_schedule=True)
            prior=h.events[-1];funding=published_funding(prior,as_of=at,zone=zone)
            controller.events=h.events;controller.decision=lambda _,event=event:date.fromisoformat(event['decision'])
            longest=max(row.miles for row in physical_history if row.end<=at and row.available_at<=at
                and at-timedelta(days=30)<=row.start<at and controller.capacity_eligibility.get(row.key,True))
            updated=propose(controller,at,entries,longest=longest)
            state=updated['continuous_state']
            assert updated['candidate'] is None
            days=(date.fromisoformat(state['day'])-date.fromisoformat(initial['decision'])).days
            expected=initial['progression_target']['planning_volume_per_7']*1.04**(days/7)
            assert abs(state['volume']-expected)<1e-8
            assert updated['target'].growth_per_7==.04
            course.append(dict(at=event['as_of'],day=state['day'],volume=state['volume'],
                expected_volume=expected,rate=updated['target'].growth_per_7,
                rate_status=state['rate_status'],epoch=state['execution_epoch']))
            controller.general_progression_state=state
            controller.target_history=(*controller.target_history,updated['target'])
            if event['index']==initial['index']:continue
            physical=reviews[event['as_of']]['smoothed_support']
            volume=event['progression_target']['planning_volume_per_7']
            comparisons={}
            for basis in ('issued_nominal_v1',):
                cases={}
                for factor in (1.,.99,.95,.90):
                    # Explicit distance-only counterfactual. No altered source,
                    # physical load, HR, or claim of new prescription adherence.
                    altered=tuple(replace(e,completion=replace(e.completion,
                        distance_meters=e.completion.distance_meters.scale(factor)))
                        if e.completion and e.completion.distance_meters else e for e in entries)
                    execution=fulfillment(altered,as_of=at,epoch=epoch,zone=zone,distance_basis=basis)
                    p=growth(volume,physical,execution,funding)
                    cases[str(factor)]=dict(ratio=execution['fulfillment_ratio'],
                        growth_per_7=p['growth_per_7'],status=p['rate_status'])
                comparisons[basis]=cases
            rows.append(dict(at=event['as_of'],original_rate=event['progression_target']['growth_per_7'],
                planner_funding=funding,comparisons=comparisons))
    finally:db.close()
    manifest=freeze_sources(output/'sources.zip',config={'timezone_default':str(zone)},entrypoint=__name__,
        inputs=(summary,root/'scenario.json',root/'sandbox.sqlite',root/'execution-sandbox.json',*files))
    result=dict(status='paired_prescription_attribution_only',epoch=epoch.isoformat(),rows=rows,controller_course=course,
        manifest=manifest,planner_searches=0,live_changed=False,
        limitation='Frozen V4 decisions, original issued contracts, and already-synthetic HR replay. Counterfactual distances are not new physical activity or a V6 calendar/stability/physiology pass.')
    with (output/'report.json').open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(dict(last=rows[-1],controller_final=course[-1],rows=len(rows),planner_searches=0,live_changed=False),indent=2))


if __name__=='__main__':
    p=ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();audit(a.root,a.output)
