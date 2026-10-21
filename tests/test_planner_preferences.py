"""Dated preference wiring; not physiological or global-optimality validation."""
from dataclasses import replace
from datetime import timedelta
import json
import sqlite3
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from run_analysis.adaptive.planner_preferences import (
    PlannerPreferencesRevision, queue_request, latest_request, apply_request, progression_stage,
)
from run_analysis.adaptive.decision_snapshot import restore, serialize
from run_analysis.adaptive.new_model_daily_harness import DailyHarness,HarnessPolicy
from run_analysis.adaptive.adaptive_stage_lifecycle import advance_stage
from run_analysis.adaptive.stage_acceptance import StageLedger
from run_analysis.web.schemas import PlannerPreferencesPatch
from run_analysis.web.app import create_app
from test_adaptive_settings import adopted
from test_adaptive_stage_lifecycle import completed_stage
from test_frequency_lifecycle import adopted as frequency, replay
from test_frequency_midpoint_trial import audit
from run_analysis.adaptive.live_bridge import input_signature


def test_api_queues_request_without_overlay_or_issuance_and_retries_are_idempotent(tmp_path):
    _,runtime,state = adopted(tmp_path)
    client = TestClient(create_app(tmp_path))
    settings = client.get('/api/settings').json()['planner']
    overlay = (tmp_path/'config.local.yaml').read_bytes()
    with sqlite3.connect(runtime/'sandbox.sqlite') as db:
        old = db.execute('SELECT * FROM sandbox_decisions').fetchall()
    patch = dict(planner_preferences=dict(long_run_goal_miles=18, run_frequency='fewer',
        expected_revision=settings['preferences_revision']))
    response = client.patch('/api/settings',json=patch)
    assert response.status_code == 200
    p = response.json()['planner']
    assert p['preferences_pending']
    assert p['long_goal_miles'] == 17 and p['run_frequency'] == 'balanced'
    assert p['requested_long_goal_miles'] == 18 and p['requested_run_frequency'] == 'fewer'
    assert p['snapshot_fingerprint'] == state.fingerprint
    assert client.patch('/api/settings',json=patch).status_code == 200
    with sqlite3.connect(runtime/'sandbox.sqlite') as db:
        assert db.execute('SELECT * FROM sandbox_decisions').fetchall() == old
        assert db.execute('SELECT count(*) FROM adaptive_preference_requests').fetchone()[0] == 1
    assert (tmp_path/'config.local.yaml').read_bytes() == overlay


def test_unchanged_preferences_are_a_true_noop(tmp_path):
    _,runtime,_ = adopted(tmp_path)
    client = TestClient(create_app(tmp_path))
    p = client.get('/api/settings').json()['planner']
    before = (runtime/'sandbox.sqlite').read_bytes()
    r = client.patch('/api/settings',json={'planner_preferences':dict(
        long_run_goal_miles=17,run_frequency='balanced',expected_revision=p['preferences_revision'])})
    assert r.status_code == 200 and not r.json()['planner']['preferences_pending']
    assert r.json()['recalculation'][0]['status'] == 'unchanged'
    assert (runtime/'sandbox.sqlite').read_bytes() == before


def test_setup_confirms_progression_after_preference_save_without_reissuing_plan(tmp_path):
    _, runtime, state = adopted(tmp_path)
    client = TestClient(create_app(tmp_path))
    before = (runtime/'sandbox.sqlite').read_bytes()
    setup = client.get('/api/setup').json()
    assert not next(s for s in setup['steps'] if s['step'] == 'progression')['complete']
    p = client.get('/api/settings').json()['planner']
    saved = client.patch('/api/settings', json={'planner_preferences': dict(
        long_run_goal_miles=17, run_frequency='balanced', expected_revision=p['preferences_revision'])})
    assert saved.status_code == 200
    confirmed = client.patch('/api/settings', json={'setup': {
        'confirmed_steps': ['heart_rate', 'zones', 'goal', 'weather', 'progression'],
        # This fixture has declared zones, not an original FIT/watch-settings
        # record. Do not bypass watch provenance just to confirm progression.
        'zone_method': 'custom',
    }})
    assert confirmed.status_code == 200, confirmed.text
    assert confirmed.json()['planner']['snapshot_fingerprint'] == state.fingerprint
    assert (runtime/'sandbox.sqlite').read_bytes() == before
    after = client.get('/api/setup').json()
    assert next(s for s in after['steps'] if s['step'] == 'progression')['complete']


@pytest.mark.parametrize('edit', [dict(long_run_goal_miles=7),dict(run_frequency='every_day'),
    dict(expected_revision='stale'),dict(long_run_goal_miles=101)])
def test_invalid_requests_do_not_write_a_policy(tmp_path, edit):
    _,runtime,_ = adopted(tmp_path)
    client = TestClient(create_app(tmp_path))
    p = client.get('/api/settings').json()['planner']
    before = (runtime/'sandbox.sqlite').read_bytes()
    value = dict(long_run_goal_miles=18,run_frequency='more',expected_revision=p['preferences_revision'])
    r = client.patch('/api/settings',json={'planner_preferences':dict(value,**edit)})
    assert r.status_code == 422
    assert (runtime/'sandbox.sqlite').read_bytes() == before


def test_preferences_and_calibration_cannot_partially_save_together(tmp_path):
    _,runtime,_ = adopted(tmp_path)
    client = TestClient(create_app(tmp_path))
    p = client.get('/api/settings').json()['planner']
    before = (tmp_path/'config.local.yaml').read_bytes()
    r = client.patch('/api/settings',json=dict(target_hr=146,planner_preferences=dict(
        long_run_goal_miles=18,run_frequency='more',expected_revision=p['preferences_revision'])))
    assert r.status_code == 422
    assert (tmp_path/'config.local.yaml').read_bytes() == before


def test_missing_decision_table_returns_review_error_without_settings_write(tmp_path):
    _,runtime,_ = adopted(tmp_path)
    client = TestClient(create_app(tmp_path))
    p = client.get('/api/settings').json()['planner']
    with sqlite3.connect(runtime/'sandbox.sqlite') as db:
        db.execute('ALTER TABLE sandbox_decisions RENAME TO incomplete_decisions')
    before = (runtime/'sandbox.sqlite').read_bytes()
    overlay = (tmp_path/'config.local.yaml').read_bytes()
    r = client.patch('/api/settings',json={'planner_preferences':dict(
        long_run_goal_miles=18,run_frequency='more',expected_revision=p['preferences_revision'])})
    assert r.status_code == 422 and 'needs review' in r.json()['detail']
    assert (runtime/'sandbox.sqlite').read_bytes() == before
    assert (tmp_path/'config.local.yaml').read_bytes() == overlay


def test_concurrent_stale_request_rejected_but_new_revision_can_supersede(tmp_path):
    _,runtime,_ = adopted(tmp_path)
    client = TestClient(create_app(tmp_path))
    p = client.get('/api/settings').json()['planner']
    patch = dict(long_run_goal_miles=18,run_frequency='fewer',expected_revision=p['preferences_revision'])
    first = client.patch('/api/settings',json={'planner_preferences':patch}).json()['planner']
    assert client.patch('/api/settings',json={'planner_preferences':dict(patch,run_frequency='more')}).status_code == 422
    second = client.patch('/api/settings',json={'planner_preferences':dict(patch,
        run_frequency='more',expected_revision=first['preferences_revision'])})
    assert second.status_code == 200
    with sqlite3.connect(runtime/'sandbox.sqlite') as db:
        latest = latest_request(db)
        assert latest.parent_key == first['preferences_revision']
        assert latest_request(db,as_of=latest.available_at-timedelta(microseconds=1)).frequency_mode == 'fewer'


def test_preference_identity_and_snapshot_backward_compatibility(tmp_path):
    _,_,state = adopted(tmp_path)
    original = serialize(state)
    assert 'center_fraction' not in original and 'planner_preferences' not in original
    r = PlannerPreferencesRevision.create(available_at=state.as_of+timedelta(seconds=1),
        goal_long_miles=18,frequency_mode='more',parent_key='adopted:'+state.fingerprint)
    with pytest.raises(ValueError,match='identity changed'):
        replace(r,goal_long_miles=19)
    changed = replace(state,as_of=r.available_at,planner_preferences=r,
        frequency_contract=replace(state.frequency_contract,
            preference=replace(state.frequency_contract.preference,center_fraction=.75)))
    assert restore(serialize(changed),expected_fingerprint=changed.fingerprint) == changed
    assert serialize(restore(original,expected_fingerprint=state.fingerprint)) == original
    assert changed.context == state.context and changed.history == state.history


def test_frequency_preference_changes_utility_not_admission_or_fatigue_reference(tmp_path):
    _,_,state = adopted(tmp_path)
    p = state.frequency_contract.preference
    less = replace(p,center_fraction=.25)
    more = replace(p,center_fraction=.75)
    assert less.assess(audit(14))['cost'] == 0
    assert more.assess(audit(18))['cost'] == 0
    assert less.assess(audit(18))['cost'] > less.assess(audit(14))['cost']
    assert more.assess(audit(14))['cost'] > more.assess(audit(18))['cost']
    for shifted in (less,more):
        assert shifted.reference_fatigue == p.reference_fatigue
        assert shifted.reference_days == p.reference_days and shifted.edge_fraction == p.edge_fraction
        assert max(shifted.assess(audit(i))['cost'] for i in range(12,21)) <= p.reference_fatigue*p.edge_fraction


def test_goal_only_edit_does_not_force_calendar_search_but_frequency_edit_does(tmp_path):
    _,_,state = adopted(tmp_path)
    h = SimpleNamespace(zone=__import__('zoneinfo').ZoneInfo(state.timezone),
        config={},
        decision=lambda at:state.context.decision, history=state.history,
        frequency_contract=state.frequency_contract,planner_preferences=None,
        policy=HarnessPolicy(continuation_distribution=state.context.continuation_distribution,
            continuation_boundary=state.context.continuation_distribution!='linear_v1'))
    captured = SimpleNamespace(contexts=(),restrictions=(),responses=(),metadata=())
    before = input_signature(h,captured,state.as_of)
    h.planner_preferences = PlannerPreferencesRevision.create(available_at=state.as_of+timedelta(seconds=1),
        goal_long_miles=18,frequency_mode='balanced',parent_key='adopted:'+state.fingerprint)
    assert input_signature(h,captured,state.as_of+timedelta(seconds=2)) == before
    h.frequency_contract = replace(h.frequency_contract,
        preference=replace(h.frequency_contract.preference,center_fraction=.25))
    assert input_signature(h,captured,state.as_of+timedelta(seconds=2)) != before


def test_failed_publication_keeps_request_pending_and_old_calendar_immutable(tmp_path,monkeypatch):
    h,_,config,at = completed_stage(tmp_path,monkeypatch,continuation_boundary=True,
        single_volume=True,purpose_continuation=True)
    h.frequency_contract = frequency(h,at-timedelta(days=1))
    try:
        first = h.replan(at,reason='preference rollback fixture')
        state,_,_ = replay(h,first)
        old = h.sandbox.db.execute('SELECT * FROM sandbox_decisions').fetchall()
        request = queue_request(h.sandbox.db,state=state,stage=first['stage_runtime']['stage'],
            patch=PlannerPreferencesPatch(long_run_goal_miles=18,run_frequency='more',
                expected_revision='adopted:'+state.fingerprint),as_of=at+timedelta(seconds=1))
        apply_request(h,as_of=at+timedelta(seconds=2))
        original_publish = h.sandbox.publish
        def fail(*args,**kwargs):
            original_publish(*args,**kwargs)
            raise ValueError('simulated publication failure')
        monkeypatch.setattr(h.sandbox,'publish',fail)
        with pytest.raises(ValueError,match='simulated publication failure'):
            h.replan(at+timedelta(seconds=2),reason='failed preference fixture')
        assert h.sandbox.db.execute('SELECT * FROM sandbox_decisions').fetchall() == old
        assert latest_request(h.sandbox.db) == request
        root = h.sandbox.root
    finally:
        h.sandbox.close()
    reopened = DailyHarness.resume(root,config,as_of=at+timedelta(seconds=3))
    try:
        assert reopened.planner_preferences is None
        assert reopened.frequency_contract.preference.center_fraction == .5
        assert apply_request(reopened,as_of=at+timedelta(seconds=3))
        assert reopened.planner_preferences == request
    finally:
        reopened.sandbox.close()


def test_apply_publish_progression_and_restart_preserve_load_and_prior_decisions(tmp_path,monkeypatch):
    h,inputs,config,at = completed_stage(tmp_path,monkeypatch,continuation_boundary=True,
        single_volume=True,purpose_continuation=True)
    h.frequency_contract = frequency(h,at-timedelta(days=1))
    try:
        first = h.replan(at,reason='preference lifecycle baseline fixture')
        state,_,_ = replay(h,first)
        old_stage = h.stage
        old_state = serialize(state)
        old_history = tuple(h.history)
        queued_at = at+timedelta(seconds=1)
        patch = PlannerPreferencesPatch(long_run_goal_miles=18,run_frequency='fewer',
            expected_revision='adopted:'+state.fingerprint)
        request = queue_request(h.sandbox.db,state=state,stage=json.loads(json.dumps(first['stage_runtime']['stage'],default=str)),
            patch=patch,as_of=queued_at)
        assert not apply_request(h,as_of=at)
        assert apply_request(h,as_of=queued_at)
        assert not apply_request(h,as_of=queued_at)
        assert h.stage == old_stage and StageLedger.for_sandbox(h.sandbox).current == old_stage
        assert progression_stage(h.stage,h.planner_preferences).goal_long_miles == 18
        updated,review,event = advance_stage(h,inputs,queued_at+timedelta(seconds=1))
        assert event['stage_acceptance'] and updated.stage.goal_long_miles == 18
        applied,_,_ = replay(updated,event)
        assert applied.planner_preferences == request
        assert applied.frequency_contract.preference.center_fraction == .25
        assert applied.frequency_contract.preference.reference_fatigue == state.frequency_contract.preference.reference_fatigue
        assert tuple(updated.history) == old_history
        assert serialize(restore(old_state,expected_fingerprint=state.fingerprint)) == old_state
        root = h.sandbox.root
    finally:
        h.sandbox.close()
    reopened = DailyHarness.resume(root,config,as_of=queued_at+timedelta(seconds=2))
    try:
        assert reopened.planner_preferences == request
        assert reopened.frequency_contract.preference.center_fraction == .25
        assert reopened.stage.goal_long_miles == 18
        assert not apply_request(reopened,as_of=queued_at+timedelta(seconds=2))
        replay(reopened,reopened.events[-1])
    finally:
        reopened.sandbox.close()
