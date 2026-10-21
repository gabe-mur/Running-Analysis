"""Live ownership and menu transitions; no new physiological policy."""
from copy import deepcopy
from dataclasses import asdict,replace
from datetime import datetime,timedelta
import json
from types import SimpleNamespace as NS

import pytest

from run_analysis.adaptive.coached_runtime import revise_preferences,sync,offline_guard
from run_analysis.adaptive.quality_forecast_runtime import VERSION
from test_quality_coaching_policy import catalogue,sequence
from test_coached_runtime import fixture


def menu(h):
    h.catalog=catalogue()
    for d in h.catalog.values():
        if d.workout_definition:d.workout_definition['calendar_integration']=VERSION
    h.coaching_contracts={k:object() for k in h.catalog}
    h.config={'coaching':{'quality_sessions':{}}}
    h.quality_runtime['admission']={'stage':{'ordinary_miles':4.2,'quality_work_minutes':10.5}}
    h.coaching_epochs=(h.coaching_base,)


@pytest.mark.parametrize('family',['threshold','progression','short_intervals','long_intervals','fartlek'])
def test_menu_edit_keeps_old_receipts_and_revises_future_commands(monkeypatch,family):
    h,rows,actual,at=fixture(monkeypatch)
    menu(h);sync(h,at)
    original=h.coaching_sequence;receipts=h.coaching_receipts;base=h.coaching_base
    h.config['coaching']['quality_sessions'][family]=False
    revise_preferences(h,at)
    assert h.coaching_base is base and h.coaching_receipts==receipts
    assert all(r.family!=family for r in h.coaching_sequence.ordinary)
    assert h.coaching_sequence.assignment(0).purpose==original.assignment(0).purpose
    assert h.coaching_sequence.consumed==0 and len(h.coaching_epochs)==2
    revised=h.coaching_sequence
    sync(h,at+timedelta(seconds=1))
    assert h.coaching_receipts==receipts and h.coaching_sequence==revised
    # Retry / unchanged preferences must not create another epoch.
    revise_preferences(h,at+timedelta(seconds=2))
    assert len(h.coaching_epochs)==2 and h.coaching_sequence==revised
    h.config['coaching']['quality_sessions'][family]=True
    revise_preferences(h,at+timedelta(seconds=3))
    assert family in {r.family for r in h.coaching_sequence.ordinary}
    assert len(h.coaching_epochs)==3 and h.coaching_base is base


def test_all_sustained_disabled_is_explicit_review_not_override_or_keyerror(monkeypatch):
    h,_,_,at=fixture(monkeypatch);menu(h);sync(h,at)
    original=h.coaching_sequence;epochs=h.coaching_epochs
    h.config['coaching']['quality_sessions'].update(threshold=False,progression=False)
    with pytest.raises(ValueError,match='preference combination is not supported yet'):
        revise_preferences(h,at)
    assert h.coaching_sequence==original and h.coaching_epochs==epochs
    assert not h.config['coaching']['quality_sessions']['threshold']


def test_preference_epoch_survives_restore_and_rejects_tampered_lineage(monkeypatch):
    import sqlite3
    import run_analysis.adaptive.coached_runtime as m
    h,_,_,at=fixture(monkeypatch);menu(h);sync(h,at)
    h.config['coaching']['quality_sessions']['threshold']=False
    revise_preferences(h,at)
    db=sqlite3.connect(':memory:');db.row_factory=sqlite3.Row
    db.executescript('CREATE TABLE stages(key TEXT,parent TEXT);CREATE TABLE current_stage(singleton INTEGER,key TEXT);')
    db.execute('INSERT INTO stages VALUES (?,NULL)',(h.stage.key,))
    db.execute('INSERT INTO current_stage VALUES (1,?)',(h.stage.key,))
    restored=NS(stage=h.stage,sandbox=NS(db=db))
    runtime=dict(coaching_base=asdict(h.coaching_base),coaching_stage_keys=[h.stage.key],
        coaching_epochs=[asdict(e) for e in h.coaching_epochs])
    state=NS(quality_coaching=h.coaching_sequence,quality_completions=h.coaching_receipts,as_of=at)
    monkeypatch.setattr(m,'offline_guard',lambda h:None)
    try:
        m.restore_runtime(restored,runtime,state)
        assert restored.coaching_epochs==h.coaching_epochs
        assert restored.coaching_sequence==h.coaching_sequence
        bad=deepcopy(runtime);bad['coaching_epochs'].reverse()
        with pytest.raises(ValueError,match='lineage'):m.restore_runtime(restored,bad,state)
        with pytest.raises(ValueError,match='frozen quality policy'):
            m.restore_runtime(restored,runtime,replace_state(state,quality_coaching=h.coaching_base))
    finally:db.close()


def replace_state(state,**values):return NS(**dict(vars(state),**values))


def test_focus_edit_uses_owned_epoch_without_resetting_physical_history(monkeypatch):
    h,_,_,at=fixture(monkeypatch);menu(h);sync(h,at)
    old=h.coaching_sequence;receipts=h.coaching_receipts;base=h.coaching_base
    h.config['coaching']['quality_focus']='vo2max'
    revise_preferences(h,at)
    assert h.coaching_sequence.focus=='vo2max'
    assert h.coaching_sequence.assignment(0).purpose==old.assignment(0).purpose
    assert h.coaching_base is base and h.coaching_receipts==receipts
    assert len(h.coaching_epochs)==2
    focused=h.coaching_sequence
    sync(h,at+timedelta(seconds=1))
    revise_preferences(h,at+timedelta(seconds=2))
    assert h.coaching_sequence==focused and len(h.coaching_epochs)==2
    h.config['coaching']['quality_focus']='variety'
    revise_preferences(h,at+timedelta(seconds=3))
    assert h.coaching_sequence.focus=='variety' and len(h.coaching_epochs)==3
    assert h.coaching_base is base and h.coaching_receipts==receipts


def test_live_guard_requires_explicit_matching_release_certificate(tmp_path):
    import sqlite3
    root=tmp_path/'runtime';root.mkdir();(root/'execution-sandbox.json').write_text('{}')
    database=tmp_path/'app.sqlite'
    with sqlite3.connect(database) as db:
        db.execute('CREATE TABLE app_state(key TEXT,value_json TEXT)')
        db.execute('INSERT INTO app_state VALUES (?,?)',('adaptive_runtime',json.dumps({'root':str(root)})))
    h=NS(config={'paths':{'database':str(database)}},sandbox=NS(root=root),coaching_base=sequence())
    with pytest.raises(ValueError,match='cannot activate'):offline_guard(h)
    certificate=dict(version='coached_live_v206',database=str(database),
        coaching_base_key=h.coaching_base.key,validated_app_copy=True)
    (root/'coached-release.json').write_text(json.dumps(certificate))
    offline_guard(h)
    certificate['coaching_base_key']='unverified'
    (root/'coached-release.json').write_text(json.dumps(certificate))
    with pytest.raises(ValueError,match='certificate differs'):offline_guard(h)


@pytest.mark.parametrize('distance',['5k','10k','half_marathon','marathon'])
def test_coached_admission_retains_dated_race_rules_after_binding(monkeypatch,distance):
    from run_analysis.adaptive.coached_pipeline import coached_evaluator
    from run_analysis.adaptive.race_goal import RaceGoalRevision
    from test_decision_snapshot import freeze
    from run_analysis.workload_accounting import Quantity
    state=freeze();seq=sequence();catalog=catalogue()
    goal=RaceGoalRevision.create(available_at=state.as_of,distance=distance,
        race_date=state.as_of.date()+timedelta(days=30),preparation_mode='balanced',parent_key='test')
    state=replace(state,race_goal=goal,context=replace(state.context,race_goal=goal))
    days=[goal.taper_start-timedelta(days=1),goal.taper_start,goal.race_date,goal.race_date+timedelta(days=1)]
    physical={(day.isoformat(),r.key):NS(miles=r.miles,demand=tuple(Quantity.point(v,'test')
        for v in (35.,50.,10.))) for day in days for r in seq.ordinary}
    monkeypatch.setattr('run_analysis.adaptive.coached_pipeline.evaluate_snapshot',lambda *a,**k:dict(
        decision_eligible=True,construction_eligible=True,constraints={'status':'allowed'},
        evaluation=NS(total_cost=1.,model_version='test',fingerprint='test')))
    engine=coached_evaluator(dict(state=state,catalog=catalog,dates=tuple(d.isoformat() for d in days),
        physical=physical),seq)
    command=seq.assignment(0).key
    # Forecast support and lower fatigue cannot override a race date, taper
    # dose restriction, or post-race easy-only rule.
    assert engine['evaluate'](((days[0].isoformat(),command),))['eligible']
    for day in days[1:]:
        result=engine['evaluate'](((day.isoformat(),command),))
        assert not result['eligible'] and not result['construction_eligible']
        assert result['constraints']=='dated_race_phase_restriction'
