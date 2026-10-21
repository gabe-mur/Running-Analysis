"""Fresh general-fitness C1 sandbox with a full opening calendar search.

Preserve the causal physical evidence, HR normalization, stage, and original
search allowances. An earlier calendar is a search start, never a copied issued
promise. No old completions or post-decision evidence are imported.
"""
from argparse import ArgumentParser
from dataclasses import replace
from datetime import date, datetime
import json
from pathlib import Path

from scripts.replay_adaptive_search import load_decision
from scripts.research_manifest import freeze_sources
from run_analysis.config import load_config
from run_analysis.stage_target_history import StageTarget
from run_analysis.staged_progression import DevelopmentStage, ProgressionPolicy
from run_analysis.adaptive.new_model_daily_harness import DailyHarness, HarnessPolicy
from run_analysis.adaptive.review_seed import read_entry
from run_analysis.adaptive.stage_acceptance import StageLedger


def prepare(source, output, *, index=1):
    case = load_decision(source, index)  # Require exact original score/admission reproduction.
    state = case['state']
    if state.race_goal is not None:
        raise ValueError('This bounded seed preparer is for general fitness only')
    scenario = json.loads((source/'scenario.json').read_text())
    runtime = case['event']['stage_runtime']
    config = load_config()
    if (config['timezone_default'], config['resting_hr'], config['max_hr']) != (
            state.timezone, state.resting_hr, state.maximum_hr):
        raise ValueError('Frozen timezone and HR normalization must not change')
    raw = runtime['stage']
    stage = DevelopmentStage(**dict(raw, issued_at=datetime.fromisoformat(raw['issued_at']),
        policy=ProgressionPolicy(**raw['policy'])))
    targets = tuple(StageTarget(**dict(r, available_at=datetime.fromisoformat(r['available_at']),
        effective_on=date.fromisoformat(r['effective_on']))) for r in runtime['target_history'])
    policy = replace(HarnessPolicy(**scenario['search_policy']), continuation_distribution='c1_v1',
        coordinated_final_evaluations=64)  # Retained AS144 allowance, not an enlarged calendar search.
    manifest = freeze_sources(output.with_suffix('.sources.zip'), config=config, entrypoint=__name__,
        inputs=(source/f'decision-{index:04d}.json', source/case['event']['snapshot_file'], source/'scenario.json'))
    h = DailyHarness(output, config, origin=state.as_of, catalog=tuple(case['catalog'].values()),
        history=state.history, phase_history=state.phase_history, scales=state.context.scales,
        training_policy=state.context.policy, search_policy=policy,
        demand_scenario=state.context.scenario, exchange=state.context.exchange, timing=state.context.timing,
        coaching_contracts=dict(state.contracts), easy_duration_policy=state.context.easy_duration_policy,
        decision_source=scenario['decision_source'], restrictions=state.restrictions, current_context=state.current,
        responses=state.responses, stage=stage, target_history=targets, frequency_contract=state.frequency_contract,
        capacity_eligibility={r.key:r.capacity_eligible for r in state.athlete.sessions},
        seed_review_entries=tuple(read_entry(r) for r in scenario.get('seed_review_entries', ())))
    h.planner_preferences = state.planner_preferences
    ledger = StageLedger.for_sandbox(h.sandbox, seed=stage)
    try:
        h.previous = tuple(map(tuple, case['event']['program']))
        event = h.replan(state.as_of, reason='fresh C1 continuation opening search')
        assert event['selected']['eligible']
        assert ':c1_handoff_v1' in event['model_version']
        assert not h.sandbox.completed_accounted(state.as_of)
        report = dict(status='fresh_c1_seed_published', manifest=manifest, source=str(source), index=index,
            search_policy=policy.__dict__, cost=event['cost'], seconds=event['seconds'],
            evaluations=event['evaluations'], miles21=event['volume'], program=event['program'],
            model_version=event['model_version'], selected=event['selected'],
            snapshot_fingerprint=event['snapshot_fingerprint'], seed_history_count=len(state.history),
            seed_review_count=len(h.seed_review_entries), initial_expensive_search_reused=False,
            old_completions_copied=False, live_changed=False,
            scope='Fresh unrestricted opening search; not daily stability evidence')
        with (output/'seed-summary.json').open('x') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
        print(json.dumps({k:report[k] for k in ('status','seconds','evaluations','miles21','model_version')}), flush=True)
    finally:
        h.sandbox.close(); ledger.close()


if __name__ == '__main__':
    p = ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--event', type=int, default=1)
    a = p.parse_args(); prepare(a.source, a.output, index=a.event)
