"""Fixed-course V5 controller audit; no search, live writes, or new adherence claim."""
from argparse import ArgumentParser
from datetime import datetime,timedelta
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from run_analysis.adaptive.continuous_general_progression import propose
from run_analysis.adaptive.program_evidence_adapter import read_accounted
from run_analysis.adaptive.smoothed_general_progression import NOMINAL_VERSION
from run_analysis.staged_progression import DevelopmentStage,ProgressionPolicy
from scripts.research_manifest import freeze_sources
from scripts.trial_smoothed_progression import target


def run(seed,recorded,output):
    output.mkdir(parents=True,exist_ok=True)
    scenario=json.loads((seed/'scenario.json').read_text())
    event=json.loads(sorted(seed.glob('decision-*.json'))[-1].read_text())
    original=json.loads((recorded/'resume-0001.summary.json').read_text())
    zone=ZoneInfo(json.loads((seed/'execution-sandbox.json').read_text())['timezone'])
    manifest=freeze_sources(output/'sources.zip',config={'timezone_default':str(zone)},
        entrypoint=__name__,inputs=(seed/'scenario.json',seed/'execution-sandbox.json',
            seed/'sandbox.sqlite',recorded/'sandbox.sqlite',recorded/'resume-0001.summary.json')+
            tuple(seed.glob('decision-*.json')))
    history=[read_accounted(row) for row in scenario['history']]
    with sqlite3.connect((recorded/'sandbox.sqlite').resolve().as_uri()+'?mode=ro',uri=True) as db:
        history.extend(read_accounted(json.loads(row[0]))
            for row in db.execute('SELECT accounted FROM sandbox_completions'))
    raw=event['stage_runtime']['stage']
    stage=DevelopmentStage(**dict(raw,issued_at=datetime.fromisoformat(raw['issued_at']),
        policy=ProgressionPolicy(**raw['policy'])))
    h=SimpleNamespace(stage=stage,history=history,capacity_eligibility=scenario['capacity_eligibility'],
        planner_preferences=None,zone=zone,decision=lambda at:at.astimezone(zone).date(),
        target_history=tuple(map(target,event['stage_runtime']['target_history'])),
        general_progression_state=event['stage_runtime']['general_progression_state'],
        policy=SimpleNamespace(general_progression=NOMINAL_VERSION,progression_credit='actual_distance_v2'))
    at=datetime.fromisoformat(event['as_of'])+timedelta(microseconds=1)
    start=at.astimezone(zone).date();rows=[]
    for n in range(22):
        when=at if n==0 else datetime.combine(start+timedelta(days=n),datetime.min.time(),zone).replace(hour=4)
        longest=max(row.miles for row in history if row.end<=when and row.available_at<=when
            and when-timedelta(days=30)<=row.start<when and h.capacity_eligibility.get(row.key,True))
        result=propose(h,when,(),longest=longest)
        assert result['candidate'] is None
        state=result['continuous_state'];e=result['smoothed_support']
        if n==0:baseline=state['volume']
        expected=baseline*1.04**(n/7)
        assert abs(state['volume']-expected)<1e-8
        assert result['target'].growth_per_7==.04
        rows.append(dict(day=state['day'],volume=state['volume'],expected_nominal_volume=expected,
            rate=result['target'].growth_per_7,actual_per_7=e['actual_per_7'],
            matched_ratio=e['delivery_ratio'],long_miles=state['long_miles'],
            rate_status=state['rate_status'],ceiling=state['supported_volume_ceiling']))
        h.general_progression_state=state;h.target_history=(*h.target_history,result['target'])
    report=dict(status='fixed_recorded_course_controller_only',policy=NOMINAL_VERSION,
        original_final_volume=original['events'][-1]['progression_target']['planning_volume_per_7'],
        nominal_final_volume=rows[-1]['volume'],rows=rows,manifest=manifest,planner_searches=0,live_changed=False,
        limitation='Original real lead-in plus already-synthetic V4 replay actuals, causally filtered. These are NOT executions of new V5 prescriptions, a searched V5 calendar, or a fresh stability/physiology pass.')
    with (output/'report.json').open('x') as stream:json.dump(report,stream,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in report.items() if k not in {'rows','manifest'}},indent=2))


if __name__=='__main__':
    p=ArgumentParser(description=__doc__)
    p.add_argument('--seed',type=Path,required=True)
    p.add_argument('--recorded',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.seed,a.recorded,a.output)
