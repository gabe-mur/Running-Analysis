"""Real snapshot/publication/restart controls, not daily optimizer acceptance."""
from dataclasses import asdict, replace
from datetime import date, datetime, timedelta, timezone
import json

import pytest

from test_adaptive_stage_lifecycle import completed_stage
from run_analysis.adaptive.frequency_range_trial import (
    FrequencyContract, read_frequency_contract, assess_frequency)
from run_analysis.adaptive.new_model_daily_harness import DailyHarness, planned_accounted, synthetic_payload
from run_analysis.adaptive.decision_snapshot import restore, serialize, evaluate_snapshot
from run_analysis.adaptive.adaptive_stage_lifecycle import advance_stage
from run_analysis.program_constraints import DateRestriction


def adopted(h, at):
    # Explicit synthetic normalization for lifecycle controls, NOT a calibrated
    # athlete reference. Real adoption must use its original frozen comparator.
    return FrequencyContract.freeze(adopted_at=at, stage_key=h.stage.key,
        effective_on=h.decision(at), volume_per_7=h.stage.weekly_equivalent_miles,
        reference_fatigue=.04, reference_days=54, edge_fraction=.02)


def replay(h, event):
    state = restore((h.sandbox.root/event['snapshot_file']).read_text(),
                    expected_fingerprint=event['snapshot_fingerprint'])
    planned = tuple(planned_accounted(f'candidate:{day}:{key}',
        datetime.combine(date.fromisoformat(day), datetime.min.time().replace(hour=18), h.zone),
        h.catalog[key], event['forecast_handoff_by_dose'][key], h.ref, available=state.as_of)
        for day, key in event['program'])
    result = evaluate_snapshot(state, planned, tuple(k for _, k in event['program']), reference=h.ref)
    assert result['evaluation'].total_cost == pytest.approx(event['cost'], abs=1e-12)
    assert result['evaluation'].fingerprint == event['selected']['score_fingerprint']
    assert result['decision_eligible'] == event['selected']['eligible']
    assert result['frequency_range'] == event['selected']['frequency_range']
    assert result['frequency_preference'] == event['selected']['frequency_preference']
    assert result['evaluation'].model_version == event['model_version']
    return state, planned, result


def test_contract_json_dated_stage_changes_preserve_reference_and_anchor():
    at = datetime(2026, 10, 3, tzinfo=timezone.utc)
    c = FrequencyContract.freeze(adopted_at=at, stage_key='one', effective_on=at.date(),
        volume_per_7=20., reference_fatigue=.04, reference_days=54)
    assert read_frequency_contract(json.loads(json.dumps(asdict(c), default=lambda x:x.isoformat()))) == c
    assert c.accepted_stage('same-dose', at.date()+timedelta(days=5), 20.) == c
    changed = c.accepted_stage('two', at.date()+timedelta(days=14), 21.)
    assert changed.preference == c.preference and changed.adopted_at == c.adopted_at
    assert changed.targets[0] == c.targets[0]
    assert len(changed.targets) == 2
    with pytest.raises(ValueError, match='Ordered'):
        c.accepted_stage('two', at.date(), 21.)


@pytest.mark.parametrize('edge_fraction', [.02, .03])
def test_publication_rollover_restart_and_independent_reprice(tmp_path, monkeypatch, edge_fraction):
    h, _, config, at = completed_stage(tmp_path, monkeypatch,
        growth_focus='easy_volume', continuation_boundary=True, single_volume=True,
        purpose_continuation=True)
    h.frequency_contract = adopted(h, at)
    h.frequency_contract = replace(h.frequency_contract,
        preference=replace(h.frequency_contract.preference,edge_fraction=edge_fraction))
    original = h.frequency_contract
    try:
        first = h.replan(at, reason='explicit frequency adoption fixture')
        state, _, _ = replay(h, first)
        assert state.frequency_contract == original
        assert serialize(restore(serialize(state), expected_fingerprint=state.fingerprint)) == serialize(state)
        day = min(h.active)
        dose = h.catalog[h.active[day]['dose']]
        start = datetime.combine(date.fromisoformat(day), datetime.min.time().replace(hour=16), h.zone)
        payload, phases, end = synthetic_payload(dose, start, hr=140)
        uploaded = h.upload(payload, day=day, available_at=end,
            observed_role=dose.role, observed_phases=phases)
        assert uploaded['status'] == 'consumed'
        post_at = end+timedelta(microseconds=1)
        post = h.replan(post_at, reason='post upload fixture')
        replay(h, post)
        assert post['decision'] > day
        assert h.frequency_contract == original
        assert post['selected']['frequency_range']['blocks'][0]['observed_count'] == 1
        duplicate = h.upload(payload, day=day, available_at=end+timedelta(seconds=30),
            observed_role=dose.role, observed_phases=phases)
        assert duplicate['status'] == 'duplicate'
        assert h.frequency_contract == original
        rollover = post_at+timedelta(days=1)
        second = h.replan(rollover, reason='rollover fixture')
        replay(h, second)
        assert h.frequency_contract == original
        assert second['selected']['frequency_range']['anchor'] == original.targets[0].effective_on.isoformat()
        root = h.sandbox.root
    finally:
        h.sandbox.close()
    restored = DailyHarness.resume(root, config, as_of=rollover)
    try:
        assert restored.frequency_contract == original
        replay(restored, restored.events[-1])
        event = restored.replan(rollover+timedelta(days=1), reason='restart rollover fixture')
        replay(restored, event)
        assert restored.frequency_contract == original
    finally:
        restored.sandbox.close()


def test_stage_acceptance_prices_frequency_atomically_without_rebasing(tmp_path, monkeypatch):
    h, inputs, config, at = completed_stage(tmp_path, monkeypatch,
        growth_focus='easy_volume', continuation_boundary=True, single_volume=True,
        purpose_continuation=True)
    h.frequency_contract = adopted(h, at-timedelta(days=1))
    original = h.frequency_contract
    try:
        updated, review, event = advance_stage(h, inputs, at)
        assert isinstance(event['stage_acceptance'], str) and event['stage_acceptance']
        from run_analysis.adaptive.stage_acceptance import StageLedger
        assert StageLedger.for_sandbox(updated.sandbox).current == updated.stage
        assert len(updated.frequency_contract.targets) == 2
        assert updated.frequency_contract.targets[0] == original.targets[0]
        assert updated.frequency_contract.preference == original.preference
        assert updated.frequency_contract.targets[-1].volume_per_7 == updated.stage.weekly_equivalent_miles
        replay(updated, event)
        root = updated.sandbox.root
    finally:
        h.sandbox.close()
    restored = DailyHarness.resume(root, config, as_of=at)
    try:
        assert restored.frequency_contract == updated.frequency_contract
        replay(restored, restored.events[-1])
    finally:
        restored.sandbox.close()


def test_closed_block_and_missing_runs_do_not_reset_or_renormalize_contract():
    at = datetime(2026, 10, 3, tzinfo=timezone.utc)
    c = FrequencyContract.freeze(adopted_at=at, stage_key='one', effective_on=at.date(),
        volume_per_7=22., reference_fatigue=.04, reference_days=54)
    seam = at.date()+timedelta(days=28)
    # First block is deliberately deficient; it must not demand compensation
    # in the next block. This is accounting, not clearance of a new calendar.
    planned = tuple(seam+timedelta(days=i) for i in range(16))
    audit = assess_frequency(c.targets, (at.date(),), planned, decision=seam,
        coverage_end=seam+timedelta(days=54), policy=c.policy)
    assert audit['eligible'] and len(audit['blocks']) == 1
    assert audit['blocks'][0]['required_count_min'] == 12
    assert audit['blocks'][0]['count'] == 16
    assert c.preference.assess(audit)['cost'] == 0
    missing = assess_frequency(c.targets, (at.date(),), (), decision=seam,
        coverage_end=seam+timedelta(days=54), policy=c.policy)
    assert not missing['eligible'] and missing['construction_eligible']
    assert c.preference.reference_fatigue == .04 and c.preference.reference_days == 54


def test_attributed_lower_floor_dead_end_before_seam_not_a_replay_pass():
    """Known gap reproduced: next-day block closure must not masquerade as recovery.

    This assertion records the current PROTOTYPE limitation, not an approved
    fallback. A missed-run policy repair must replace this behavior/control.
    """
    start = date(2026, 10, 3)
    c = FrequencyContract.freeze(adopted_at=datetime(2026, 10, 3, tzinfo=timezone.utc),
        stage_key='one', effective_on=start, volume_per_7=22.,
        reference_fatigue=.04, reference_days=54)
    today = start+timedelta(days=27)
    future = tuple(start+timedelta(days=28+i) for i in range(16))
    for current in ((), (today,)):
        a = assess_frequency(c.targets, (start,), (*current, *future), decision=today,
            coverage_end=today+timedelta(days=54), policy=c.policy)
        assert not a['eligible']  # Even running today cannot fund12 dates retroactively.
        assert a['blocks'][0]['deficit'] in (10, 11)
        assert a['construction_eligible']
    next_day = today+timedelta(days=1)
    b = assess_frequency(c.targets, (start,), future, decision=next_day,
        coverage_end=next_day+timedelta(days=54), policy=c.policy)
    assert b['eligible']  # Same evidence becomes admitted solely at the block seam.


def test_partial_tail_becomes_complete_on_day_two_without_new_evidence():
    start = date(2026, 10, 3)
    c = FrequencyContract.freeze(adopted_at=datetime(2026, 10, 3, tzinfo=timezone.utc),
        stage_key='one', effective_on=start, volume_per_7=22.,
        reference_fatigue=.04, reference_days=54)
    dates = tuple(start+timedelta(days=d) for d in (*range(2, 18), *range(30, 50)))
    a = assess_frequency(c.targets, (), dates, decision=start,
        coverage_end=start+timedelta(days=54), policy=c.policy)
    b = assess_frequency(c.targets, (), dates, decision=start+timedelta(days=2),
        coverage_end=start+timedelta(days=56), policy=c.policy)
    assert a['eligible'] and b['eligible']
    assert len(a['blocks']) == 1 and len(b['blocks']) == 2
    assert c.preference.assess(a)['cost'] == 0
    assert c.preference.assess(b)['cost'] == pytest.approx(.04*.02*28/54)
    # Accounting limitation documented before any daily stability run: even
    # unchanged evidence exposes a newly charged block as the horizon advances.
def test_failed_optional_growth_keeps_original_frequency_history(tmp_path, monkeypatch):
    h, inputs, _, at = completed_stage(tmp_path, monkeypatch,
        growth_focus='easy_volume', continuation_boundary=True, single_volume=True)
    h.frequency_contract = adopted(h, at-timedelta(days=1))
    original = h.frequency_contract
    import run_analysis.adaptive.new_model_daily_harness as module
    from run_analysis.adaptive.continuous_search_adapter import NoEligibleProgram
    monkeypatch.setattr(module, 'migrated_search', lambda **k: (_ for _ in ()).throw(NoEligibleProgram('fixture')))
    try:
        updated, review, event = advance_stage(h, inputs, at)
        assert updated is h and event is None and review['candidate_not_accepted']
        assert h.frequency_contract == original
    finally:
        h.sandbox.close()


def test_frequency_cannot_override_reserved_rest_or_missing_count(tmp_path, monkeypatch):
    h, _, _, at = completed_stage(tmp_path, monkeypatch,
        growth_focus='easy_volume', continuation_boundary=True, single_volume=True)
    h.frequency_contract = adopted(h, at)
    try:
        event = h.replan(at, reason='fixture')
        state, planned, result = replay(h, event)
        day = planned[0].start.astimezone(h.zone).date()
        restricted = replace(state, restrictions=(DateRestriction('off', day, day+timedelta(days=1),
            at, 'reserved_rest', 'explicit fixture'),))
        bad = evaluate_snapshot(restricted, planned, tuple(k for _, k in event['program']), reference=h.ref)
        assert not bad['decision_eligible'] and not bad['construction_eligible']
        empty = evaluate_snapshot(state, (), (), reference=h.ref)
        raw_empty = evaluate_snapshot(replace(state, frequency_contract=None), (), (), reference=h.ref)
        assert empty['decision_eligible'] == raw_empty['decision_eligible']
        assert empty['frequency_range']['lower_is_advisory']
        assert empty['frequency_range']['blocks'][0]['deficit'] > 0
        assert not result['frequency_preference']['scope'].startswith('recovery')
    finally:
        h.sandbox.close()
