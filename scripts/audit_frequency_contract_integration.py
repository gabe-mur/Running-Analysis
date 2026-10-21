"""Read-only saved-winner contract reprice and frequency boundary controls.

No search, uploads, publication, live DB access or physiological clearance.
Boundary examples assess frequency accounting only, not full training plans.
"""
from argparse import ArgumentParser
from dataclasses import asdict, replace
from datetime import date, datetime, time, timedelta
import json
from pathlib import Path
from zoneinfo import ZoneInfo

from scripts.replay_adaptive_search import load_decision
from scripts.research_manifest import freeze_sources
from run_analysis.config import load_config
from run_analysis.continuous_training_dose import ContinuousDoseReference
from run_analysis.adaptive.frequency_range_trial import (
    FrequencyContract, assess_frequency, compile_frequency_exposure)
from run_analysis.adaptive.decision_snapshot import serialize, restore, evaluate_snapshot
from run_analysis.adaptive.new_model_daily_harness import planned_accounted


def audit(root, prototype, output):
    output.mkdir(exist_ok=False)
    source = json.loads(prototype.read_text())
    case = load_decision(root, 1)  # Mandatory original cost and admission parity.
    state = case['state']
    stage = case['event']['stage_runtime']['stage']
    contract = FrequencyContract.freeze(adopted_at=state.as_of, stage_key=stage['key'],
        effective_on=state.context.decision, volume_per_7=stage['weekly_equivalent_miles'],
        **source['preference'], version='frequency_midpoint_v1')
    adopted = replace(state, frequency_contract=contract)
    manifest = freeze_sources(output/'sources.zip', config=load_config(), entrypoint=__name__,
        inputs=(prototype, root/'decision-0001.json', root/case['event']['snapshot_file'], root/'scenario.json'))
    reference = ContinuousDoseReference('sandbox upload accounting')
    def reprice(frozen, program):
        program = tuple(tuple(p) for p in program)
        planned = tuple(planned_accounted(f'candidate:{day}:{key}',
            datetime.combine(date.fromisoformat(day), time(18), ZoneInfo(frozen.timezone)),
            case['catalog'][key], case['handoffs'][key], reference, available=frozen.as_of)
            for day, key in program)
        r = evaluate_snapshot(frozen, planned, tuple(k for _, k in program), reference=reference)
        e = r['evaluation']
        return dict(cost=e.total_cost, fingerprint=e.fingerprint, model_version=e.model_version,
            eligible=r['decision_eligible'], model_cost=e.parts['pre_frequency_total_cost'],
            frequency=r['frequency_range'], preference=r['frequency_preference'],
            progression_delivery=r.get('progression_delivery'))
    prior = reprice(adopted, case['event']['program'])
    selected = reprice(adopted, source['result']['program'])
    restored = restore(serialize(adopted), expected_fingerprint=adopted.fingerprint)
    assert reprice(restored, source['result']['program']) == selected
    assert restored.frequency_contract == contract
    assert abs(prior['cost']-source['baseline']['cost']) < 1e-12
    assert abs(selected['cost']-source['fresh_reprice']['cost']) < 1e-12 and selected['eligible']
    revised_contract = replace(contract, version='frequency_midpoint_v2')
    revised = replace(adopted, frequency_contract=revised_contract)
    revised_prior = reprice(revised, case['event']['program'])
    revised_selected = reprice(revised, source['result']['program'])
    assert reprice(restore(serialize(revised), expected_fingerprint=revised.fingerprint),
        source['result']['program']) == revised_selected
    start = date(2026, 10, 3)
    boundary = FrequencyContract.freeze(adopted_at=datetime.combine(start, time(), ZoneInfo('UTC')),
        stage_key='accounting-only', effective_on=start, volume_per_7=22.,
        reference_fatigue=.04, reference_days=54, edge_fraction=.02)
    def counts(decision, planned, observed=(), *, legacy=False):
        a = (assess_frequency(boundary.targets, observed, planned, decision=decision,
            coverage_end=decision+timedelta(days=54), policy=boundary.policy) if legacy else
            compile_frequency_exposure(boundary, observed, decision=decision,
                coverage_end=decision+timedelta(days=54),
                taper_start=decision+timedelta(days=40)).assess(planned))
        return dict(admission=a, preference=boundary.preference.assess(a))
    tomorrow = start+timedelta(days=28)
    future = tuple(tomorrow+timedelta(days=i) for i in range(16))
    missed = dict(today_rest=counts(tomorrow-timedelta(days=1), future, (start,)),
        today_run=counts(tomorrow-timedelta(days=1), (tomorrow-timedelta(days=1), *future), (start,)),
        next_day_same_history=counts(tomorrow, future, (start,)))
    same = tuple(start+timedelta(days=i) for i in (*range(2, 18), *range(30, 50)))
    horizon = dict(initial=counts(start, same), day_two=counts(start+timedelta(days=2), same))
    assert all(r['admission']['eligible'] for r in missed.values())
    assert horizon['initial']['preference']['cost'] > 0
    assert len(horizon['initial']['admission']['blocks']) == len(horizon['day_two']['admission']['blocks']) == 1
    report = dict(status='accounting_repairs_verified', manifest=manifest,
        original_archive_cost=case['archived_cost'], contract=asdict(contract),
        integrated_prior=prior, integrated_selected=selected,
        revised_contract=asdict(revised_contract), revised_prior=revised_prior,
        revised_selected=revised_selected,
        canonical_restore_reprice_identical=True, missed_lower_floor=missed,
        horizon_activation=horizon, daily_replay_started=False, live_unchanged=True,
        scope='Exact contract reprice, not search/daily stability; synthetic boundaries are frequency-only')
    with (output/'report.json').open('x') as stream:
        json.dump(report, stream, indent=2, default=lambda v:v.isoformat(), allow_nan=False)
    return report


if __name__ == '__main__':
    p = ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('output/v20-perfect-daily-7-20261002'))
    p.add_argument('--prototype', type=Path, default=Path('output/frequency-midpoint-prototype-20261003/frozen-search.json'))
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    r = audit(args.root, args.prototype, args.output)
    print(json.dumps(dict(status=r['status'], cost=r['revised_selected']['cost'],
        restored_identical=r['canonical_restore_reprice_identical'], daily_replay_started=False)))
