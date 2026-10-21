"""No calendar deadline for ordinary training; no forecast-earned growth."""
from dataclasses import asdict, replace
from datetime import timedelta
import json

import pytest

from run_analysis.adaptive.new_model_daily_harness import HarnessPolicy, DailyHarness
from run_analysis.adaptive.decision_snapshot import restore
from run_analysis.adaptive.adaptive_stage_lifecycle import advance_stage
from run_analysis.adaptive.stage_acceptance import StageLedger
from run_analysis.adaptive.service import AdaptivePlannerService


def enable_observed(h):
    h.policy = replace(h.policy, progression_admission='observed_rolling_v2')
    path = h.sandbox.root / 'scenario.json'
    scenario = json.loads(path.read_text())
    scenario['search_policy'] = asdict(h.policy)
    path.write_text(json.dumps(scenario))


def completed(tmp_path, monkeypatch):
    from test_adaptive_stage_lifecycle import completed_stage
    return completed_stage(tmp_path, monkeypatch, growth_focus='easy_volume',
        continuation_boundary=True, single_volume=True, gradual=True,
        prospective_review_hour=4)


def test_observed_admission_is_explicit_and_keeps_historical_default():
    assert HarnessPolicy().progression_admission == 'checkpoint_v1'
    with pytest.raises(ValueError, match='Observed progression'):
        HarnessPolicy(progression_admission='unknown')
    with pytest.raises(ValueError, match='Observed progression'):
        HarnessPolicy(progression_admission='observed_rolling_v2')
    assert HarnessPolicy(progression_admission='observed_rolling_v2',
        delivery_contract='sustained_v1', continuation_boundary=True, soft_horizon_taper=True)


def test_release_changes_only_admission_not_budgets_growth_or_volume_weights():
    from types import SimpleNamespace
    from test_prospective_progression import fixture
    from run_analysis.adaptive.live_bridge import (
        continuation_release_policy, observed_progression_release_policy, runtime_version)
    stage, _ = fixture()
    policy = continuation_release_policy(HarnessPolicy(delivery_contract='sustained_v1',
        continuation_boundary=True, soft_horizon_taper=True, purpose_continuation=True,
        purpose_continuation_phase='known_full_phase_v1'))
    updated = observed_progression_release_policy(policy, stage)
    assert updated == replace(policy, progression_admission='observed_rolling_v2')
    assert policy.progression_admission == 'checkpoint_v1'
    assert runtime_version(SimpleNamespace(policy=updated)) == 207
    assert observed_progression_release_policy(updated, stage) == updated
    with pytest.raises(ValueError, match='accepted rolling-volume runtime'):
        observed_progression_release_policy(HarnessPolicy(), stage)


def test_unfunded_actual_delivery_can_replan_daily_past_old_deadline_without_growth(tmp_path, monkeypatch):
    h, inputs, config, at = completed(tmp_path, monkeypatch)
    enable_observed(h)
    original = h.stage, h.target_history, tuple(h.history)
    # Future plans are real, but this controlled review has NO completed credit.
    # Do not erase the physical history or fabricate omitted past opportunities.
    monkeypatch.setattr('run_analysis.adaptive.adaptive_stage_lifecycle.review_entries', lambda *_: ())
    service = AdaptivePlannerService(h, inputs)
    try:
        for offset in (0, 1, 2):
            result = service.refresh(at + timedelta(days=offset, hours=5))
            assert result['stage_review']['status'] == 'recent_delivery_hold_targets'
            assert result['stage_review']['credited_miles'] == 0
            event = result['event']
            assert event['selection_contract'] == 'observed_rolling_progression_v2'
            state = restore((h.sandbox.root / event['snapshot_file']).read_text(),
                expected_fingerprint=event['snapshot_fingerprint'])
            assert state.prospective_volume_review is None
            assert (service.harness.stage, service.harness.target_history,
                tuple(service.harness.history)) == original
            assert StageLedger.for_sandbox(h.sandbox).current == original[0]
        current = service.harness
        current.sandbox.close()
        # Simulate failure to export the scenario policy after journal commit.
        path = h.sandbox.root / 'scenario.json'
        scenario = json.loads(path.read_text())
        scenario['search_policy'].pop('progression_admission')
        path.write_text(json.dumps(scenario))
        resumed = DailyHarness.resume(h.sandbox.root, config,
            as_of=at + timedelta(days=2, hours=5))
        try:
            assert resumed.policy.progression_admission == 'observed_rolling_v2'
            assert resumed.stage == original[0] and resumed.target_history == original[1]
        finally:
            resumed.sandbox.close()
    finally:
        h.sandbox.close()


def test_removing_forecast_gate_leaves_identical_volume_fatigue_and_constraints(tmp_path, monkeypatch):
    h, _, _, at = completed(tmp_path, monkeypatch)
    try:
        # Before the old checkpoint both admissions can price identical work.
        # Use the already archived pre-upload snapshot, not a backdated harness.
        event = h.events[0]
        state = restore((h.sandbox.root / event['snapshot_file']).read_text(),
            expected_fingerprint=event['snapshot_fingerprint'])
        from run_analysis.adaptive.decision_snapshot import evaluate_snapshot
        from run_analysis.adaptive.new_model_daily_harness import planned_accounted
        from datetime import datetime, time
        rows = tuple(planned_accounted('candidate:' + day + ':' + key,
            datetime.combine(datetime.fromisoformat(day).date(), time(18), h.zone),
            h.catalog[key], event['forecast_handoff_by_dose'][key], h.ref,
            available=state.as_of) for day, key in event['program'])
        keys = tuple(key for _, key in event['program'])
        old = evaluate_snapshot(state, rows, keys, reference=h.ref)
        new = evaluate_snapshot(replace(state, prospective_volume_review=None), rows, keys, reference=h.ref)
        assert replace(old['evaluation'], fingerprint=new['evaluation'].fingerprint) == new['evaluation']
        assert old['constraints'] == new['constraints']
        assert old.get('frequency_range') == new.get('frequency_range')
        assert old['decision_eligible'] and new['decision_eligible']
    finally:
        h.sandbox.close()


def test_successful_actual_review_cannot_compound_on_another_same_day_review(tmp_path, monkeypatch):
    h, inputs, _, at = completed(tmp_path, monkeypatch)
    enable_observed(h)
    try:
        updated, review, event = advance_stage(h, inputs, at)
        assert event and event['stage_acceptance']
        assert review['status'] == 'conditional_development_candidate'
        stage, targets = updated.stage, updated.target_history
        again, review, event = advance_stage(updated, inputs, at + timedelta(seconds=1))
        assert event is None and review['status'] == 'continue_existing_stage'
        assert again.stage == stage and again.target_history == targets
    finally:
        h.sandbox.close()
