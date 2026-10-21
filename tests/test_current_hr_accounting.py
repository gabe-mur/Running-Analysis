"""Current source ownership without reinterpreting adopted physical evidence."""
from copy import deepcopy
import json
import sqlite3

import pytest

from run_analysis.adaptive.hr_reference import frozen_live_config
from run_analysis.adaptive.execution_upload_sandbox import ExecutionUploadSandbox
from run_analysis.config import load_config
from run_analysis.db import connect, initialize
from run_analysis.settings_service import save_settings_overlay
from run_analysis.web.schemas import SettingsPatch
from test_web_phase1 import _write_config
from test_watch_zones import ingest, make_watch_fit, ZONES


def pin(root, config):
    sandbox=ExecutionUploadSandbox(root,config,create=True)
    with sandbox.db:
        sandbox.db.execute('CREATE TABLE adaptive_bridge_state(key TEXT PRIMARY KEY,value_json TEXT)')
        sandbox.db.execute('INSERT INTO adaptive_bridge_state VALUES (?,?)',('normalization',
            json.dumps({k:config[k] for k in ('max_hr','resting_hr','timezone_default')})))
    sandbox.close()


def test_current_settings_are_distinct_from_verified_load_units_and_reads_do_not_write(tmp_path):
    _write_config(tmp_path)
    original=load_config(tmp_path/'config.yaml');root=tmp_path/'runtime';pin(root,original)
    current=deepcopy(original);current.update(max_hr=194,resting_hr=45,zones=ZONES)
    before=(root/'sandbox.sqlite').read_bytes(),(root/'execution-sandbox.json').read_bytes()
    frozen=frozen_live_config(current,root=root)
    assert frozen['max_hr']==original['max_hr'] and frozen['resting_hr']==original['resting_hr']
    assert frozen['zones']==ZONES and current['resting_hr']==45
    reopened=ExecutionUploadSandbox(root,frozen)
    try:
        assert json.loads(reopened.context)==dict(max_hr=original['max_hr'],resting_hr=original['resting_hr'])
    finally:reopened.close()
    assert (root/'sandbox.sqlite').read_bytes()==before[0]
    assert (root/'execution-sandbox.json').read_bytes()==before[1]


@pytest.mark.parametrize('change', ['marker','identity','journal','timezone','movement'])
def test_reference_verification_and_domain_are_not_bypassed(tmp_path,change):
    _write_config(tmp_path);config=load_config(tmp_path/'config.yaml');root=tmp_path/'runtime';pin(root,config)
    if change in {'marker','identity'}:
        marker=json.loads((root/'execution-sandbox.json').read_text())
        if change=='marker':marker['measurement_context']=json.dumps(dict(max_hr=194,resting_hr=50))
        else:marker['accounting_key']='different units'
        (root/'execution-sandbox.json').write_text(json.dumps(marker))
    elif change=='journal':
        with sqlite3.connect(root/'sandbox.sqlite') as db:
            db.execute("UPDATE adaptive_bridge_state SET value_json='{}'")
    elif change=='timezone':config['timezone_default']='Pacific/Honolulu'
    elif change=='movement':config['moving_time']=dict(config['moving_time'],maximum_interval_seconds=123)
    with pytest.raises(ValueError):frozen_live_config(config,root=root)


def test_new_watch_maximum_updates_settings_without_rebasing_the_adopted_ledger(tmp_path):
    from run_analysis.settings_service import refresh_watch_zones
    from run_analysis.training_signals import HeartRateReference
    from run_analysis.adaptive.prescribed_hr_target import prescribed_hr_target
    _write_config(tmp_path);base=load_config(tmp_path/'config.yaml');root=tmp_path/'runtime';pin(root,base)
    (tmp_path/'config.local.yaml').write_text('adaptive_planner:\n  engine: adaptive_frequency_v3\nsetup:\n  zone_method: device\n')
    with connect(tmp_path/'data/test.sqlite') as db:
        initialize(db)
        db.execute('INSERT INTO app_state VALUES (?,?,?)',('adaptive_runtime',
            json.dumps(dict(engine='adaptive_frequency_v3',root=str(root))),'now'));db.commit()
        edges=(126,140,153,167,182,196)
        ingest(db,tmp_path,make_watch_fit(tmp_path/'watch.fit',edges=edges,maximum=196,resting=45))
        before=(root/'sandbox.sqlite').read_bytes(),(root/'execution-sandbox.json').read_bytes()
        current,changed=refresh_watch_zones(tmp_path/'config.yaml',db)
        assert changed
        assert current['max_hr']==196 and current['resting_hr']==45
        assert current['zones']['z5']==[183,196]
        frozen=frozen_live_config(current,connection=db)
        assert (frozen['max_hr'],frozen['resting_hr'])==(base['max_hr'],base['resting_hr'])
        assert frozen['zones']==current['zones']
        reopened=ExecutionUploadSandbox(root,frozen);reopened.close()
        assert (root/'sandbox.sqlite').read_bytes()==before[0]
        assert (root/'execution-sandbox.json').read_bytes()==before[1]
        # A metadata update is not permission to clip or extrapolate actual HR.
        reference=HeartRateReference(frozen['resting_hr'],frozen['max_hr'],'adopted unit')
        assert reference.reserve(196) is None
        assert prescribed_hr_target(('Z4',),frozen) is not None
        with pytest.raises(ValueError,match='must fit athlete normalization'):
            prescribed_hr_target(('Z5',),frozen)
        overlay=(tmp_path/'config.local.yaml').read_bytes()
        again,changed=refresh_watch_zones(tmp_path/'config.yaml',db)
        assert not changed and again==current
        assert (tmp_path/'config.local.yaml').read_bytes()==overlay


def test_failed_hr_reference_save_has_one_clear_no_save_message(tmp_path):
    _write_config(tmp_path);base=load_config(tmp_path/'config.yaml');root=tmp_path/'runtime';pin(root,base)
    (tmp_path/'config.local.yaml').write_text('adaptive_planner:\n  engine: adaptive_frequency_v3\n')
    with connect(tmp_path/'data/test.sqlite') as db:
        initialize(db)
        db.execute('INSERT INTO app_state VALUES (?,?,?)',('adaptive_runtime',
            json.dumps(dict(engine='adaptive_frequency_v3',root=str(root))),'now'));db.commit()
        marker=json.loads((root/'execution-sandbox.json').read_text());marker['accounting_key']='broken'
        (root/'execution-sandbox.json').write_text(json.dumps(marker))
        before=(tmp_path/'config.local.yaml').read_bytes()
        with pytest.raises(ValueError) as error:
            save_settings_overlay(tmp_path/'config.yaml',SettingsPatch(resting_hr=45),connection=db)
        assert str(error.value).count('No settings were saved')==1
        assert (tmp_path/'config.local.yaml').read_bytes()==before


def test_watch_adoption_and_custom_save_keep_pins_immutable(tmp_path):
    _write_config(tmp_path);base=load_config(tmp_path/'config.yaml');root=tmp_path/'runtime';pin(root,base)
    (tmp_path/'config.local.yaml').write_text('adaptive_planner:\n  engine: adaptive_frequency_v3\n')
    with connect(tmp_path/'data/test.sqlite') as db:
        initialize(db)
        db.execute('INSERT INTO app_state VALUES (?,?,?)',('adaptive_runtime',
            json.dumps(dict(engine='adaptive_frequency_v3',root=str(root))),'now'));db.commit()
        ingest(db,tmp_path,make_watch_fit(tmp_path/'watch.fit'))
        before=(root/'sandbox.sqlite').read_bytes(),(root/'execution-sandbox.json').read_bytes()
        current=save_settings_overlay(tmp_path/'config.yaml',SettingsPatch(
            max_hr=190,resting_hr=60,setup={'zone_method':'device'},
            profile={'birth_date':'1990-01-01','sex':'male','weight_lb':150,
                'height_in':70,'max_hr_source':'measured'}),connection=db)
        assert (current['max_hr'],current['resting_hr'],current['zones'])==(194,45,ZONES)
        assert current['profile']['max_hr_source']=='watch'
        assert frozen_live_config(current,connection=db)['resting_hr']==base['resting_hr']
        custom=save_settings_overlay(tmp_path/'config.yaml',SettingsPatch(max_hr=194,resting_hr=47,
            setup={'zone_method':'custom'},profile={'birth_date':'1990-01-01','sex':'male',
                'weight_lb':150,'height_in':70,'max_hr_source':'measured'}),connection=db)
        assert custom['resting_hr']==47 and custom['profile']['max_hr_source']=='measured'
        assert (root/'sandbox.sqlite').read_bytes()==before[0]
        assert (root/'execution-sandbox.json').read_bytes()==before[1]


def test_live_resume_receives_accounting_not_current_athlete_config(tmp_path,monkeypatch):
    from run_analysis.adaptive import live_bridge
    _write_config(tmp_path);base=load_config(tmp_path/'config.yaml');root=tmp_path/'runtime';pin(root,base)
    current=dict(base,resting_hr=45,max_hr=194,zones=ZONES)
    seen=[]
    def resume(root,config,**kwargs):
        seen.append(config)
        raise ValueError('intentional stop before search')
    monkeypatch.setattr(live_bridge,'resume_service',resume)
    with connect(tmp_path/'data/test.sqlite') as db:
        initialize(db)
        db.execute('INSERT INTO app_state VALUES (?,?,?)',('adaptive_runtime',
            json.dumps(dict(engine=live_bridge.ENGINE,root=str(root))),'now'));db.commit()
        with pytest.raises(live_bridge.LivePlannerReviewRequired,match='intentional stop'):
            live_bridge.schedule(db,current,tmp_path)
    assert len(seen)==1 and seen[0]['resting_hr']==base['resting_hr']
    assert seen[0]['zones']==ZONES and current['resting_hr']==45


def test_real_committed_harness_resumes_with_identical_evidence_after_current_hr_change(tmp_path):
    from datetime import datetime
    from run_analysis.adaptive.new_model_daily_harness import demo, DailyHarness
    root=tmp_path/'replay'
    report=demo(root,days=1,budget=20,hard_days=3,tail_days=2)
    original=dict(load_config('config.yaml'),resting_hr=50,max_hr=200,timezone_default='UTC')
    with sqlite3.connect(root/'sandbox.sqlite') as db:
        db.execute('CREATE TABLE adaptive_bridge_state(key TEXT PRIMARY KEY,value_json TEXT)')
        db.execute('INSERT INTO adaptive_bridge_state VALUES (?,?)',('normalization',
            json.dumps({k:original[k] for k in ('max_hr','resting_hr','timezone_default')})))
    at=datetime.fromisoformat(report['events'][-1]['as_of'])
    prior=DailyHarness.resume(root,original,as_of=at)
    try:
        evidence=tuple(prior.history);phases=prior.phase_history
        active=deepcopy(prior.active);reference=prior.sandbox.context
    finally:prior.sandbox.close()
    current=dict(original,resting_hr=45,max_hr=194)
    resumed=DailyHarness.resume(root,frozen_live_config(current,root=root),as_of=at)
    try:
        assert tuple(resumed.history)==evidence and resumed.phase_history==phases
        assert resumed.active==active and resumed.sandbox.context==reference
    finally:resumed.sandbox.close()
