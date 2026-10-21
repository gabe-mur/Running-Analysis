"""Fresh real-evidence sandbox with repaired frequency accounting; no deployment.

Original snapshot/forecasts and AS120 normalization are preserved. No old
completions/issued slots copied. A fresh unrestricted original search publishes
the opening candidate; prior AS120 calendar is only a revalidated search seed.
"""
from argparse import ArgumentParser
from dataclasses import asdict
from datetime import date, datetime
import json
from pathlib import Path

from scripts.replay_adaptive_search import load_decision
from scripts.research_manifest import freeze_sources
from run_analysis.config import load_config
from run_analysis.stage_target_history import StageTarget
from run_analysis.staged_progression import DevelopmentStage, ProgressionPolicy
from run_analysis.adaptive.new_model_daily_harness import DailyHarness, HarnessPolicy
from run_analysis.adaptive.frequency_range_trial import FrequencyContract
from run_analysis.adaptive.stage_acceptance import StageLedger


def prepare(source, comparator, output):
    case = load_decision(source, 1)
    state = case['state']
    scenario = json.loads((source/'scenario.json').read_text())
    runtime = case['event']['stage_runtime']
    prior = json.loads(comparator.read_text())
    if prior['snapshot_fingerprint'] != state.fingerprint:
        raise ValueError('Preference normalization must belong to this exact original decision')
    config = load_config()
    if (config['timezone_default'], config['resting_hr'], config['max_hr']) != (
            state.timezone, state.resting_hr, state.maximum_hr):
        raise ValueError('Frozen timezone and HR normalization must not change')
    raw = runtime['stage']
    stage = DevelopmentStage(**dict(raw, issued_at=datetime.fromisoformat(raw['issued_at']),
        policy=ProgressionPolicy(**raw['policy'])))
    targets = tuple(StageTarget(**dict(r, available_at=datetime.fromisoformat(r['available_at']),
        effective_on=date.fromisoformat(r['effective_on']))) for r in runtime['target_history'])
    contract = FrequencyContract.freeze(adopted_at=state.as_of, stage_key=stage.key,
        effective_on=state.context.decision, volume_per_7=stage.weekly_equivalent_miles,
        **prior['preference'])
    manifest = freeze_sources(output.with_suffix('.sources.zip'), config=config, entrypoint=__name__,
        inputs=(source/'decision-0001.json', source/case['event']['snapshot_file'],
            source/'scenario.json', comparator))
    h = DailyHarness(output, config, origin=state.as_of, catalog=tuple(case['catalog'].values()),
        history=state.history, phase_history=state.phase_history, scales=state.context.scales,
        training_policy=state.context.policy, search_policy=HarnessPolicy(**scenario['search_policy']),
        demand_scenario=state.context.scenario, exchange=state.context.exchange, timing=state.context.timing,
        coaching_contracts=dict(state.contracts), easy_duration_policy=state.context.easy_duration_policy,
        decision_source=scenario['decision_source'], restrictions=state.restrictions, current_context=state.current,
        responses=state.responses, stage=stage, target_history=targets, frequency_contract=contract,
        capacity_eligibility={r.key:r.capacity_eligible for r in state.athlete.sessions})
    ledger = StageLedger.for_sandbox(h.sandbox, seed=stage)
    try:
        h.previous = tuple(tuple(p) for p in prior['result']['program'])
        event = h.replan(state.as_of, reason='fresh repaired frequency opening search')
        assert event['selected']['eligible']
        assert not h.sandbox.completed_accounted(state.as_of)
        report = dict(status='fresh_frequency_seed_published', manifest=manifest,
            frequency_contract=asdict(contract), source=str(source), comparator=str(comparator),
            original_comparator_cost=prior['fresh_reprice']['cost'], cost=event['cost'],
            seconds=event['seconds'], evaluations=event['evaluations'], volume=event['volume'],
            program=event['program'], selected=event['selected'],
            snapshot_fingerprint=event['snapshot_fingerprint'],
            old_completions_copied=False, live_changed=False,
            scope='Fresh sandbox opening search, not daily stability or a new progression phase')
        with (output/'seed-summary.json').open('x') as stream:
            json.dump(report, stream, indent=2, default=lambda v:v.isoformat(), allow_nan=False)
        print(json.dumps(dict(status=report['status'], seconds=event['seconds'],
            miles21=event['volume'], cost=event['cost'])), flush=True)
    finally:
        h.sandbox.close(); ledger.close()


if __name__ == '__main__':
    p = ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=Path('output/v20-perfect-daily-7-20261002'))
    p.add_argument('--comparator', type=Path,
        default=Path('output/frequency-midpoint-prototype-20261003/frozen-search.json'))
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); prepare(a.source, a.comparator, a.output)
