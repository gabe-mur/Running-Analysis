"""Draft settings/guidance cannot publish, reprice or mutate a live stage."""
from dataclasses import asdict, replace
from datetime import date, datetime, timedelta, timezone
import json
import sqlite3

import pytest
from fastapi.testclient import TestClient

from run_analysis.adaptive.race_settings import preview_preparation
from run_analysis.settings_service import save_settings_overlay, settings_response, recalculate_for_settings
from run_analysis.config import load_config
from run_analysis.web.app import create_app
from run_analysis.web.schemas import RacePreparationSettings, SettingsPatch
from test_adaptive_settings import adopted
from test_race_preparation import stage
from test_web_phase1 import _write_config

FUTURE = (date.today()+timedelta(days=250)).isoformat()


def test_apply_general_goal_saves_target_without_reissuing_and_retries_are_noops(tmp_path):
    _,runtime,state=adopted(tmp_path)
    client=TestClient(create_app(tmp_path))
    planner=client.get('/api/settings').json()['planner']
    goal=client.get('/api/settings/race-preparation/status').json()
    before=(tmp_path/'config.local.yaml').read_bytes()
    with sqlite3.connect(runtime/'sandbox.sqlite') as db:
        decisions=db.execute('SELECT * FROM sandbox_decisions').fetchall()
    patch=dict(goal=dict(distance='general_fitness'),expected_revision=goal['expected_revision'],
        planner_preferences=dict(long_run_goal_miles=20,run_frequency='balanced',long_run_weekday=None,
            expected_revision=planner['preferences_revision']))
    response=client.post('/api/settings/race-preparation/apply',json=patch)
    assert response.status_code==200,response.text
    assert response.json()['saved'] and not response.json()['calendar_changed']
    assert response.json()['planner']['requested_long_goal_miles']==20
    assert response.json()['planner']['long_goal_miles']==17
    assert client.post('/api/settings/race-preparation/apply',json=patch).json()['saved'] is False
    with sqlite3.connect(runtime/'sandbox.sqlite') as db:
        assert db.execute('SELECT * FROM sandbox_decisions').fetchall()==decisions
        assert db.execute('SELECT count(*) FROM adaptive_preference_requests').fetchone()[0]==1
    assert (tmp_path/'config.local.yaml').read_bytes()==before


def test_atomic_apply_rolls_back_target_when_clearing_stale_race_goal(tmp_path):
    from run_analysis.adaptive.race_goal import queue_request,latest_request
    _,runtime,state=adopted(tmp_path)
    with sqlite3.connect(runtime/'sandbox.sqlite') as db:
        race=queue_request(db,state=state,draft=RacePreparationSettings(distance='10k',race_date=date.fromisoformat(FUTURE)),
            expected_revision='adopted:'+state.fingerprint,as_of=state.as_of+timedelta(seconds=1),general_long_goal_miles=17)
    client=TestClient(create_app(tmp_path))
    planner=client.get('/api/settings').json()['planner']
    patch=dict(goal=dict(distance='general_fitness'),expected_revision='stale-race',
        planner_preferences=dict(long_run_goal_miles=20,run_frequency='balanced',expected_revision=planner['preferences_revision']))
    failed=client.post('/api/settings/race-preparation/apply',json=patch)
    assert failed.status_code==422
    with sqlite3.connect(runtime/'sandbox.sqlite') as db:
        from run_analysis.adaptive.planner_preferences import latest_request as latest_preferences
        assert latest_preferences(db) is None
        assert latest_request(db)==race
    patch['expected_revision']=race.key
    response=client.post('/api/settings/race-preparation/apply',json=patch)
    assert response.status_code==200,response.text
    with sqlite3.connect(runtime/'sandbox.sqlite') as db:
        assert latest_request(db).distance=='general_fitness'
        assert latest_request(db).general_long_goal_miles==20
        assert latest_preferences(db).goal_long_miles==20


def test_race_apply_cannot_save_hidden_general_target(tmp_path):
    adopted(tmp_path)
    response=TestClient(create_app(tmp_path)).post('/api/settings/race-preparation/apply',json=dict(
        goal=dict(distance='10k',race_date=FUTURE),expected_revision='test',
        planner_preferences=dict(long_run_goal_miles=20,run_frequency='balanced',expected_revision='test')))
    assert response.status_code==422


def test_pace_free_draft_roundtrip_clear_and_no_legacy_validation(tmp_path, monkeypatch):
    _write_config(tmp_path)
    monkeypatch.setattr('run_analysis.settings_service.assess_race_goal',
                        lambda *a: pytest.fail('Draft called retired pace predictor'))
    client = TestClient(create_app(tmp_path))
    for distance in ('5k', '10k', 'half_marathon', 'marathon'):
        draft = dict(distance=distance, race_date=FUTURE, preparation_mode='balanced')
        response = client.patch('/api/settings', json={'race_preparation': draft})
        assert response.status_code == 200, response.text
        assert response.json()['race_preparation'] == draft
        assert client.get('/api/settings').json()['race_preparation'] == draft
    r = client.patch('/api/settings', json={'race_preparation': {'distance': 'general_fitness'}})
    assert r.status_code == 200 and r.json()['race_preparation']['race_date'] is None
    assert load_config(tmp_path/'config.yaml')['race_preparation']['race_date'] is None


@pytest.mark.parametrize('draft', [dict(distance='marathon'),
    dict(distance='10k', race_date=FUTURE, preparation_mode='shorter'),
    dict(distance='general_fitness', race_date=FUTURE), None,
    dict(distance='marathon', race_date='2020-01-01')])
def test_invalid_draft_rejects_before_write(tmp_path, draft):
    _write_config(tmp_path)
    r = TestClient(create_app(tmp_path)).patch('/api/settings', json={'race_preparation': draft})
    assert r.status_code == 422
    assert not (tmp_path/'config.local.yaml').exists()


def test_preview_requires_no_save_database_or_search(tmp_path):
    _write_config(tmp_path)
    client = TestClient(create_app(tmp_path))
    r = client.post('/api/settings/race-preparation/preview', json=dict(distance='marathon', race_date=FUTURE))
    assert r.status_code == 200
    assert r.json()['status'] == 'evidence_review_required'
    assert not r.json()['accepted'] and not r.json()['may_publish_calendar']
    assert not (tmp_path/'data'/'test.sqlite').exists()
    assert not (tmp_path/'config.local.yaml').exists()
    assert client.post('/api/settings/race-preparation/preview', json=dict(
        distance='marathon', race_date='2020-01-01')).status_code == 422


def test_draft_save_does_not_invalidate_calendar_or_mutate_journal(tmp_path):
    config, runtime, state = adopted(tmp_path)
    before = (runtime/'sandbox.sqlite').read_bytes()
    patch = SettingsPatch(race_preparation=RacePreparationSettings(distance='marathon', race_date=date.fromisoformat(FUTURE)))
    candidate = save_settings_overlay(tmp_path/'config.yaml', patch)
    class NoCalendarWrites:
        def execute(self, *args): pytest.fail('Draft touched calendar')
        def commit(self): pytest.fail('Draft committed calendar')
    stages = recalculate_for_settings(NoCalendarWrites(), candidate, tmp_path, patch, previous_config=config)
    assert [s.name for s in stages] == ['race_preparation']
    assert stages[0].status == 'deferred'
    assert (runtime/'sandbox.sqlite').read_bytes() == before


def test_unchanged_legacy_goal_on_strength_save_not_revalidated(tmp_path, monkeypatch):
    _write_config(tmp_path)
    (tmp_path/'config.local.yaml').write_text('coaching:\n  training_goal: marathon\n  goal_date: "2020-01-01"\n  goal_pace_min_mile: 10.0\n')
    coaching = settings_response(load_config(tmp_path/'config.yaml')).coaching
    coaching.strength_training.enabled = True
    monkeypatch.setattr('run_analysis.settings_service.assess_race_goal',
                        lambda *a: pytest.fail('Unchanged expired legacy goal was revalidated'))
    save_settings_overlay(tmp_path/'config.yaml', SettingsPatch(coaching=coaching))


def test_expired_unchanged_draft_does_not_block_unrelated_save(tmp_path):
    _write_config(tmp_path)
    (tmp_path/'config.local.yaml').write_text('race_preparation:\n  distance: marathon\n  race_date: "2020-01-01"\n  preparation_mode: balanced\n')
    draft = settings_response(load_config(tmp_path/'config.yaml')).race_preparation
    result = save_settings_overlay(tmp_path/'config.yaml', SettingsPatch(target_hr=146,race_preparation=draft))
    assert result['target_hr'] == 146


def test_verified_preview_all_distances_read_only_and_stale_guard(tmp_path):
    config, runtime, state = adopted(tmp_path)
    # Complete stage representation, same verified snapshot identity/issuance.
    s = stage(key=state.stage_key, issued_at=state.stage_issued_at,
              weekly_equivalent_miles=20., long_miles=5.)
    with __import__('sqlite3').connect(runtime/'sandbox.sqlite') as db:
        event = json.loads(db.execute('SELECT event FROM sandbox_decisions').fetchone()[0])
        value = asdict(s); value['issued_at'] = s.issued_at.isoformat()
        event['stage_runtime']['stage'] = value
        db.execute('UPDATE sandbox_decisions SET event=?', (json.dumps(event),))
    config = dict(config, timezone_default=state.timezone)
    before = (runtime/'sandbox.sqlite').read_bytes()
    for distance in ('5k', '10k', 'half_marathon', 'marathon'):
        draft = RacePreparationSettings(distance=distance, race_date=state.as_of.date()+timedelta(days=366))
        r = preview_preparation(config, tmp_path, draft, as_of=state.as_of)
        assert r['options'], r
        assert r['snapshot_fingerprint'] == state.fingerprint
        assert not r['accepted'] and not r['calendar_changed']
        assert not r['whole_program_checked']
        stale = preview_preparation(config, tmp_path, draft, as_of=state.as_of+timedelta(days=40))
        assert stale['status'] == 'observed_session_reference_required' and not stale['options']
    assert (runtime/'sandbox.sqlite').read_bytes() == before


def test_outcome_route_uses_explicit_report_and_preserves_domain_errors(tmp_path,monkeypatch):
    _write_config(tmp_path)
    client=TestClient(create_app(tmp_path));calls=[]
    def save(config,root,**kwargs):
        calls.append(kwargs)
        if kwargs['expected_revision']=='stale':
            raise ValueError('Race outcome changed; reload the form')
        return dict(saved=True,calendar_changed=False)
    monkeypatch.setattr('run_analysis.adaptive.race_settings.save_race_outcome',save)
    body=dict(status='recorded',physical_key='original-fit',expected_revision='parent')
    response=client.post('/api/settings/race-preparation/outcome',json=body)
    assert response.status_code==200,response.text
    assert response.json()==dict(saved=True,calendar_changed=False)
    assert all(calls[0][k]==v for k,v in body.items())
    assert calls[0]['as_of'].tzinfo is not None
    response=client.post('/api/settings/race-preparation/outcome',json=dict(body,expected_revision='stale'))
    assert response.status_code==422 and 'reload' in response.json()['detail']
    assert client.post('/api/settings/race-preparation/outcome',json=dict(body,status='assumed')).status_code==422


def test_goal_status_reads_verified_physical_handoff_not_completed_execution_shape(tmp_path):
    from hashlib import sha256
    from run_analysis.adaptive.race_settings import goal_status
    from run_analysis.adaptive.new_model_daily_harness import dump
    from test_r3_decision_controls import row
    config,runtime,state=adopted(tmp_path)
    physical=row(-1,6.2,'quality',source='observed')
    encoded=dump(asdict(physical))
    with sqlite3.connect(runtime/'sandbox.sqlite') as db:
        db.execute('CREATE TABLE sandbox_completions(actual TEXT,accounted TEXT,physical_fingerprint TEXT,available_at TEXT)')
        db.execute('INSERT INTO sandbox_completions VALUES (?,?,?,?)',
            ('{"key":"execution-not-accounted"}',encoded,sha256(encoded.encode()).hexdigest(),physical.available_at.isoformat()))
    assert goal_status(config,tmp_path)['status']=='available'
    with sqlite3.connect(runtime/'sandbox.sqlite') as db:
        db.execute("UPDATE sandbox_completions SET physical_fingerprint='changed'")
    assert goal_status(config,tmp_path)['status']=='review_required'


def test_race_apply_cannot_override_a_protected_event_date(tmp_path,monkeypatch):
    from run_analysis.adaptive.race_settings import save_race_request
    from run_analysis.adaptive.decision_snapshot import serialize
    from run_analysis.program_constraints import DateRestriction
    config,runtime,state=adopted(tmp_path,race_ready=True)
    race_day=state.as_of.date()+timedelta(days=180)
    restriction=DateRestriction('reserved',race_day,race_day+timedelta(days=1),state.as_of,
        'reserved_rest','explicit athlete request')
    changed=replace(state,restrictions=(restriction,))
    with sqlite3.connect(runtime/'sandbox.sqlite') as db:
        event=json.loads(db.execute('SELECT event FROM sandbox_decisions').fetchone()[0])
        event['snapshot_fingerprint']=changed.fingerprint
        db.execute('UPDATE sandbox_decisions SET event=?,snapshot=?',(json.dumps(event),serialize(changed)))
    monkeypatch.setattr('run_analysis.adaptive.race_settings.RACE_ACTIVATION_READY',True)
    monkeypatch.setattr('run_analysis.adaptive.race_settings.preview_preparation',lambda *a,**k:
        dict(options=[dict(mode='balanced',requested_date_fits=True)]))
    before=(runtime/'sandbox.sqlite').read_bytes()
    with pytest.raises(ValueError,match='protected'):
        save_race_request(config,tmp_path,RacePreparationSettings(distance='10k',race_date=race_day),
            expected_revision='adopted:'+changed.fingerprint,as_of=state.as_of+timedelta(seconds=1))
    assert (runtime/'sandbox.sqlite').read_bytes()==before


def test_release_flag_does_not_enable_race_apply_on_old_runtime(tmp_path,monkeypatch):
    from run_analysis.adaptive.race_settings import goal_status,save_race_request
    config,runtime,state=adopted(tmp_path)
    monkeypatch.setattr('run_analysis.adaptive.race_settings.RACE_ACTIVATION_READY',True)
    monkeypatch.setattr('run_analysis.adaptive.race_settings.preview_preparation',lambda *a,**k:
        dict(options=[dict(mode='balanced',requested_date_fits=True)]))
    before=(runtime/'sandbox.sqlite').read_bytes()
    status=goal_status(config,tmp_path)
    assert status['status']=='available' and not status['race_activation_available']
    assert 'shared-search runtime' in status['activation_note']
    with pytest.raises(ValueError,match='validated shared-search'):
        save_race_request(config,tmp_path,RacePreparationSettings(distance='10k',
            race_date=state.as_of.date()+timedelta(days=180)),
            expected_revision='adopted:'+state.fingerprint,as_of=state.as_of+timedelta(seconds=1))
    assert (runtime/'sandbox.sqlite').read_bytes()==before


def test_race_activation_is_available_only_on_released_decision(tmp_path,monkeypatch):
    from run_analysis.adaptive.race_settings import goal_status
    config,runtime,state=adopted(tmp_path,race_ready=True)
    monkeypatch.setattr('run_analysis.adaptive.race_settings.RACE_ACTIVATION_READY',True)
    status=goal_status(config,tmp_path)
    assert status['race_activation_available'] and status['activation_note'] is None
    assert status['applied'] is None and not status['pending']
