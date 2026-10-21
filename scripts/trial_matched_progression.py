"""Read-only V4 proposal and fixed-original-course comparison; no search/live IO."""
from argparse import ArgumentParser
from datetime import date,datetime,timedelta
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo
import json
import sqlite3

from run_analysis.adaptive.program_evidence_adapter import read_accounted
from run_analysis.adaptive.continuous_general_progression import propose
from run_analysis.adaptive.smoothed_general_progression import MATCHED_VERSION
from run_analysis.adaptive.smoothed_progression import matched_growth
from run_analysis.staged_progression import DevelopmentStage,ProgressionPolicy
from scripts.trial_smoothed_progression import target
from scripts.research_manifest import freeze_sources


def original_comparison(report):
    comparisons={}
    for name,e in report['courses'].items():
        level=report['current_forecast_level']
        evidence=dict(actual_per_7=e['actual_per_7'],delivery_ratio=e['matched_delivery_ratio'],mean_age_days=14)
        comparisons[name]=dict(original_rate=e['growth_at_levels'][str(level)]['growth_per_7'],
            matched_rate=matched_growth(level,evidence)['growth_per_7'],evidence=evidence)
    publications=[]
    for old in report['frozen_publications']:
        rows=[]
        for e in old['trace']:
            level=e['conditional_level']
            evidence=dict(actual_per_7=e['actual_per_7'],delivery_ratio=e['matched_delivery_ratio'],mean_age_days=14)
            rows.append(dict(at=e['at'],conditional_level=level,
                original_rate=e['growth_at_levels'][str(level)]['growth_per_7'],
                matched_rate=matched_growth(level,evidence)['growth_per_7']))
        publications.append(dict(index=old['index'],published_at=old['published_at'],trace=rows))
    return dict(courses=comparisons,frozen_publications=publications,
        limitation='Identical original observations/dated targets, changed volume-support formula only; no reoptimized calendar or future earned delivery')


def proposal(source):
    scenario=json.loads((source/'scenario.json').read_text())
    events=[json.loads(f.read_text()) for f in sorted(source.glob('decision-*.json'))]
    event=events[-1];runtime=event['stage_runtime']
    stage=runtime['stage']
    stage=DevelopmentStage(**dict(stage,issued_at=datetime.fromisoformat(stage['issued_at']),
        policy=ProgressionPolicy(**stage['policy'])))
    history=[read_accounted(r) for r in scenario['history']]
    db=sqlite3.connect((source/'sandbox.sqlite').resolve().as_uri()+'?mode=ro',uri=True)
    try:
        history.extend(read_accounted(json.loads(r[0])) for r in db.execute('SELECT accounted FROM sandbox_completions'))
    finally:
        db.close()
    zone=ZoneInfo(json.loads((source/'execution-sandbox.json').read_text())['timezone'])
    at=datetime.fromisoformat(event['as_of'])+timedelta(microseconds=1)
    h=SimpleNamespace(stage=stage,history=history,capacity_eligibility=scenario['capacity_eligibility'],
        planner_preferences=None,zone=zone,decision=lambda when:when.astimezone(zone).date(),
        target_history=tuple(map(target,runtime['target_history'])),
        general_progression_state=runtime['general_progression_state'],
        policy=SimpleNamespace(general_progression=MATCHED_VERSION,progression_credit='actual_distance_v2'))
    longest=max(r.miles for r in history if at-timedelta(days=30)<=r.start<at
        and r.end<=at and r.available_at<=at and h.capacity_eligibility.get(r.key,True))
    # Distance/volume-only proposal. No fabricated quality/adherence/HR review.
    result=propose(h,at,(),longest=longest)
    state=result['continuous_state'];support=result['smoothed_support']
    assert state['volume']==runtime['general_progression_state']['volume']
    assert h.general_progression_state is runtime['general_progression_state']
    return dict(at=at.isoformat(),state=state,evidence=support,growth=result['growth_proposal'],
        current_target_preserved=result['target'].volume_per_7,
        future_slope=result['target'].growth_per_7,long_miles=state['long_miles'],
        limitation='Unpublished distance-component proposal using retained history/state; no planner selection or health clearance')


def main(source,original,output):
    output.mkdir(parents=True,exist_ok=True)
    report=dict(status='matched_volume_prototype_only',policy=MATCHED_VERSION,
        proposal=proposal(source),comparison=original_comparison(json.loads(original.read_text())),
        planner_searches=0,live_changed=False,
        unchanged_assumptions=dict(mean_age_days=14,maximum_growth_per_7=.04,
            response_envelope=.0816,long_support_share=.4),
        limitations=[
            'Baseline reference, conditional planning volume and observed exposure are distinct; none is measured fitness.',
            'Same evidence memory and response envelope retained as coaching assumptions, not newly validated physiology.',
            'One matched support comparison replaces the mismatched second actual-volume ceiling.',
            'Closed-loop flat/rising numerical controls are separate unit tests, not full planner stability replays.',
            'No elapsed-time retroactive growth, automatic live opt-in or changed physical accounting.',
        ])
    report['manifest']=freeze_sources(output/'sources.zip',config={},entrypoint=__name__,
        inputs=(original,source/'scenario.json',source/'sandbox.sqlite',source/'execution-sandbox.json')
            +tuple(source.glob('decision-*.json')))
    with (output/'report.json').open('x') as stream:
        json.dump(report,stream,indent=2)
    p=report['proposal']
    print(json.dumps(dict(status=report['status'],reference=p['state']['progression_reference'],
        planning_volume=p['current_target_preserved'],observed_volume=p['state']['observed_volume_per_7'],
        slope=p['future_slope'],long_miles=p['long_miles'],comparison=report['comparison']['courses'],
        planner_searches=0,live_changed=False)),flush=True)


if __name__=='__main__':
    p=ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--original',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    main(args.source,args.original,args.output)
