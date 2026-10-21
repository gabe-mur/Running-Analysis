"""Publish an EXACTLY revalidated coordinated seed into a fresh offline sandbox.

No old completions or issued calendars are copied. Real causal input history
and the declared phase remain frozen. The initial expensive search is reused
only when its snapshot identity, physical score and funding reproduce exactly.
"""
from argparse import ArgumentParser
from dataclasses import replace
from datetime import date,datetime
import json
from pathlib import Path

from scripts.search_prospective_progression import coordinated_case
from scripts.research_manifest import freeze_sources
from run_analysis.config import load_config
from run_analysis.stage_target_history import StageTarget
from run_analysis.adaptive.decision_snapshot import restore
from run_analysis.adaptive.new_model_daily_harness import DailyHarness,HarnessPolicy
from run_analysis.adaptive.stage_acceptance import StageLedger


def prepare(source,index,selection,output):
    saved=json.loads((selection/'result.json').read_text())
    state=restore((selection/'snapshot.json').read_text(),expected_fingerprint=saved['snapshot_fingerprint'])
    if saved.get('expected_empty_rollover'):
        raise ValueError('Daily seed must be the original coordinated phase decision, not a synthetic rollover')
    case=coordinated_case(source,index,review_hour=4)
    if saved.get('purpose_continuation'):
        case=dict(case,state=replace(case['state'],context=replace(
            case['state'].context,purpose_continuation=True,
            purpose_continuation_phase=saved.get('purpose_continuation_phase','uniform_v1'))))
    if case['state'].fingerprint!=state.fingerprint:
        raise ValueError('Coordinated source differs from the original captured decision')
    config=load_config()
    if (config['timezone_default'],config['resting_hr'],config['max_hr'])!=(state.timezone,state.resting_hr,state.maximum_hr):
        raise ValueError('Live configuration changed the frozen HR normalization/timezone')
    scenario=json.loads((source/'scenario.json').read_text())
    runtime=case['event'].get('stage_runtime',scenario)
    targets=tuple(StageTarget(**dict(t,available_at=datetime.fromisoformat(t['available_at']),
        effective_on=date.fromisoformat(t['effective_on']))) for t in runtime['target_history'])
    manifest=freeze_sources(output.with_suffix('.sources.zip'),config=config,entrypoint=__name__,
        inputs=(source/f'decision-{index:04d}.json',source/'scenario.json',selection/'result.json',selection/'snapshot.json'))
    h=DailyHarness(output,config,origin=state.as_of,catalog=tuple(case['catalog'].values()),
        history=state.history,phase_history=state.phase_history,scales=state.context.scales,
        training_policy=state.context.policy,
        search_policy=HarnessPolicy(**dict(scenario['search_policy'],prospective_review_hour=4,
            purpose_continuation=state.context.purpose_continuation,
            purpose_continuation_phase=state.context.purpose_continuation_phase)),
        demand_scenario=state.context.scenario,exchange=state.context.exchange,timing=state.context.timing,
        coaching_contracts=dict(state.contracts),easy_duration_policy=state.context.easy_duration_policy,
        decision_source=scenario['decision_source'],restrictions=state.restrictions,current_context=state.current,
        responses=state.responses,stage=case['stage'],target_history=targets,
        capacity_eligibility={r.key:r.capacity_eligible for r in state.athlete.sessions})
    ledger=StageLedger.for_sandbox(h.sandbox,seed=h.stage)
    try:
        event=h.replan(state.as_of,reason='exact coordinated initial search reuse',replayed_selection=saved)
        assert event['snapshot_fingerprint']==state.fingerprint
        assert tuple(map(tuple,event['program']))==tuple(map(tuple,saved['program']))
        assert not h.sandbox.completed_accounted(state.as_of)
        report=dict(manifest=manifest,selection_source=str(selection),original_source=str(source),
            score=event['cost'],snapshot_fingerprint=event['snapshot_fingerprint'],
            funded=event['selected']['progression_delivery'],
            initial_expensive_search_reused=True,old_completions_copied=False,live_changed=False)
        with (output/'seed-summary.json').open('x') as stream:json.dump(report,stream,indent=2)
        print(json.dumps(report),flush=True)
    finally:h.sandbox.close();ledger.close()


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('--source',type=Path,default=Path('output/current-v18-volume-curve-daily-2-20260930'))
    p.add_argument('--event',type=int,default=1)
    p.add_argument('--selection',type=Path,default=Path('output/prospective-progression-initial-20260930'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();prepare(a.source,a.event,a.selection,a.output)
