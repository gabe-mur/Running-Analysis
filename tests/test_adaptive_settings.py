"""Settings cannot silently retune/reset an adopted physical ledger."""
from dataclasses import replace
import json
from pathlib import Path
import sqlite3

import pytest
from fastapi.testclient import TestClient

from run_analysis.config import load_config
from run_analysis.db import connect, initialize
from run_analysis.adaptive.decision_snapshot import serialize
from run_analysis.adaptive.frequency_range_trial import FrequencyContract
from run_analysis.adaptive.settings_view import inspect_settings
from run_analysis.settings_service import save_settings_overlay, settings_response, reset_advanced_overlay
from run_analysis.web.app import create_app
from run_analysis.web.schemas import SettingsPatch
from test_decision_snapshot import freeze
from test_web_phase1 import _write_config


def configured(tmp_path):
    _write_config(tmp_path)
    (tmp_path/'config.local.yaml').write_text('adaptive_planner:\n  engine: adaptive_frequency_v3\n')
    return load_config(tmp_path/'config.yaml')


def adopted(tmp_path, *, race_ready=False):
    config = configured(tmp_path)
    state = freeze()
    frequency = FrequencyContract.freeze(adopted_at=state.as_of, stage_key=state.stage_key,
        effective_on=state.context.decision, volume_per_7=20,
        reference_fatigue=.05, reference_days=54)
    state = replace(state, frequency_contract=frequency)
    runtime = tmp_path/'runtime'
    runtime.mkdir()
    with sqlite3.connect(runtime/'sandbox.sqlite') as db:
        db.execute('CREATE TABLE sandbox_decisions(sequence INTEGER PRIMARY KEY,event TEXT,snapshot TEXT)')
        event = dict(snapshot_fingerprint=state.fingerprint, stage_runtime=dict(stage=dict(
            key=state.stage_key, goal_long_miles=17, long_miles=8.9, weekly_equivalent_miles=20,
            policy=dict(volume_goal_miles=21, review_days=14, growth_fraction=.05))))
        if race_ready:
            policy=dict(continuation_distribution='c1_v1',
                general_progression='easy_first_coordinated_v2',evaluations=10000,
                complete_first_search=True,final_winner_evaluations=1000,
                coordinated_final_evaluations=64,calendar_merge_evaluations=32)
            (runtime/'scenario.json').write_text(json.dumps(dict(search_policy=policy)))
            event['refinement']=dict(calendar_merge_doses=dict(max_full_evaluations=32))
        db.execute('INSERT INTO sandbox_decisions VALUES (1,?,?)', (json.dumps(event),serialize(state)))
    database = tmp_path/config['paths']['database']
    with connect(database) as db:
        initialize(db)
        db.execute('INSERT INTO app_state VALUES (?,?,?)', ('adaptive_runtime',
            json.dumps(dict(engine='adaptive_frequency_v3',root=str(runtime))),state.as_of.isoformat()))
        db.commit()
    return config, runtime, state


def test_settings_read_adopted_snapshot_not_legacy_defaults_without_writes(tmp_path):
    config, runtime, state = adopted(tmp_path)
    before = (runtime/'sandbox.sqlite').read_bytes()
    response = TestClient(create_app(tmp_path)).get('/api/settings')
    assert response.status_code == 200
    p = response.json()['planner']
    assert p['status'] == 'adopted'
    assert p['snapshot_fingerprint'] == state.fingerprint
    assert p['long_goal_miles'] == 17
    assert p['long_target_miles'] == 8.9
    assert p['volume_destination_miles_per_7'] == 21
    assert p['demand_memory_days'] == state.context.scenario.memory_days
    assert p['frequency_upper_per_7'] == state.frequency_contract.targets[-1].upper_per_7
    assert (runtime/'sandbox.sqlite').read_bytes() == before


def test_missing_runtime_does_not_adopt_or_invent_values(tmp_path):
    config = configured(tmp_path)
    p = inspect_settings(config,tmp_path)
    assert p.status == 'review_required'
    assert p.long_goal_miles is None
    assert not (tmp_path/config['paths']['database']).exists()


@pytest.mark.parametrize('version',['easy_first_coordinated_v1','easy_first_coordinated_v2'])
def test_released_next_phase_detail_keeps_current_snapshot_and_stage(tmp_path,version):
    config,runtime,state=adopted(tmp_path)
    (runtime/'scenario.json').write_text(json.dumps(dict(search_policy=dict(
        general_progression=version))))
    before=(runtime/'sandbox.sqlite').read_bytes()
    p=inspect_settings(config,tmp_path)
    assert p.status=='adopted'
    assert 'adding easy volume before increasing the long run' in p.detail
    assert p.snapshot_fingerprint==state.fingerprint
    assert p.volume_destination_miles_per_7==21
    assert (runtime/'sandbox.sqlite').read_bytes()==before


def test_bad_snapshot_does_not_display_unverified_assumptions(tmp_path):
    config,runtime,_ = adopted(tmp_path)
    with sqlite3.connect(runtime/'sandbox.sqlite') as db:
        db.execute("UPDATE sandbox_decisions SET snapshot='{}'")
    p = inspect_settings(config,tmp_path)
    assert p.status == 'review_required' and p.long_goal_miles is None


@pytest.mark.parametrize('field', ['max_hr','resting_hr','moving_time','coaching'])
def test_unsupported_changes_are_rejected_before_overlay_write(tmp_path,field):
    config = configured(tmp_path)
    settings = settings_response(config)
    if field in ('max_hr','resting_hr'):
        patch = SettingsPatch(**{field:getattr(settings,field)+1})
    elif field == 'moving_time':
        patch = SettingsPatch(moving_time=settings.moving_time.model_copy(update={'maximum_interval_seconds':123}))
    else:
        patch = SettingsPatch(coaching=settings.coaching.model_copy(update={'long_run_recency_reference_days':8}))
    before = (tmp_path/'config.local.yaml').read_bytes()
    with pytest.raises(ValueError,match='No settings were saved'):
        save_settings_overlay(tmp_path/'config.yaml',patch)
    assert (tmp_path/'config.local.yaml').read_bytes() == before


def test_complete_unchanged_form_is_not_an_unsupported_change(tmp_path):
    config = configured(tmp_path)
    s = settings_response(config)
    saved = save_settings_overlay(tmp_path/'config.yaml',SettingsPatch(
        max_hr=s.max_hr,resting_hr=s.resting_hr,moving_time=s.moving_time,coaching=s.coaching))
    assert settings_response(saved).coaching == s.coaching


def test_strength_edits_are_supported_and_reset_preserves_them(tmp_path):
    config = configured(tmp_path)
    s = settings_response(config)
    strength = s.coaching.strength_training.model_copy(update={'enabled':True})
    saved = save_settings_overlay(tmp_path/'config.yaml', SettingsPatch(
        coaching=s.coaching.model_copy(update={'strength_training':strength})))
    assert settings_response(saved).coaching.strength_training.enabled
    reset = reset_advanced_overlay(tmp_path/'config.yaml')
    assert settings_response(reset).coaching.strength_training.enabled
    assert reset['adaptive_planner'] == config['adaptive_planner']


def test_reset_with_movement_difference_is_rejected_atomically(tmp_path):
    config = configured(tmp_path)
    with (tmp_path/'config.local.yaml').open('a') as f:
        f.write('moving_time:\n  maximum_interval_seconds: 123\n')
    before = (tmp_path/'config.local.yaml').read_bytes()
    response = TestClient(create_app(tmp_path)).post('/api/settings/reset-advanced')
    assert response.status_code == 422
    assert (tmp_path/'config.local.yaml').read_bytes() == before


def test_strength_only_save_does_not_touch_adaptive_journal(tmp_path):
    config,runtime,_ = adopted(tmp_path)
    client = TestClient(create_app(tmp_path))
    s = client.get('/api/settings').json()
    s['coaching']['strength_training']['enabled'] = True
    before = (runtime/'sandbox.sqlite').read_bytes()
    response = client.patch('/api/settings',json={'coaching':s['coaching']})
    assert response.status_code == 200
    assert response.json()['planner']['long_goal_miles'] == 17
    assert (runtime/'sandbox.sqlite').read_bytes() == before
