"""Preference wiring and immutable recipes, not physiological or stability proof."""
from dataclasses import asdict
from datetime import timedelta
from types import SimpleNamespace

import pytest

from run_analysis.adaptive.quality_choices import allowed, compile_choices, install, dose_type
from run_analysis.adaptive.continuation_boundary_trial import reference_keys
from run_analysis.adaptive.new_model_daily_harness import DailyHarness,HarnessPolicy
from run_analysis.adaptive.web_materialization import render
from run_analysis.web.schemas import QualitySessionSettings, QualitySessionType, WorkoutStep
from test_adaptive_stage_lifecycle import completed_stage


KINDS=tuple(k.value for k in QualitySessionType)


def only(config,kind):
    return {**config,'adaptive_planner':{'engine':'adaptive_frequency_v3'},
        'coaching':{**config.get('coaching',{}),'quality_sessions':{k:k==kind for k in KINDS}}}


@pytest.mark.parametrize('kind',KINDS)
def test_only_enabled_family_is_forecast_selected_rendered_and_restored(tmp_path,monkeypatch,kind):
    h,_,config,at=completed_stage(tmp_path,monkeypatch,continuation_boundary=True,
        single_volume=True,purpose_continuation=True)
    try:
        original=asdict(h.catalog['quality:ordinary'])
        h.config=only(config,kind)
        install(h)
        definitions={k:asdict(d) for k,d in h.catalog.items() if d.workout_definition}
        assert len(definitions)>=12
        assert all(d.workout_definition.get('numeric_contract') for d in h.catalog.values() if d.workout_definition)
        install(h)
        assert asdict(h.catalog['quality:ordinary'])==original
        assert definitions=={k:asdict(d) for k,d in h.catalog.items() if d.workout_definition}
        event=h.replan(at+timedelta(seconds=1),reason='quality preference wiring control')
        assert event['quality_preferences']['enabled']==(kind,)
        quality=[h.catalog[r['dose']] for r in event['active'].values() if h.catalog[r['dose']].role=='quality']
        assert quality and all(dose_type(d,h.coaching_contracts[d.key])==kind for d in quality)
        schedule=render(h,event,h.config)
        recommendations=[d.recommendation for d in schedule.planning_days
            if d.recommendation and d.recommendation.planning_role=='quality']
        assert recommendations
        assert all(r.quality_session_type.value==kind for r in recommendations)
        bases={'short_intervals':{'distance','time'},'progression':{'portion'}}.get(kind,{'time'})
        for r in recommendations:
            works=[s for s in r.structure if s.phase=='work']
            assert works and all(s.prescription_basis in bases for s in works)
        h.sandbox.close()
        reopened=DailyHarness.resume(tmp_path/'state',h.config,as_of=at+timedelta(seconds=2))
        try:
            assert definitions=={k:asdict(d) for k,d in reopened.catalog.items() if d.workout_definition}
            assert all(allowed(reopened.catalog[r['dose']],reopened.coaching_contracts[r['dose']],h.config)
                for r in reopened.active.values())
        finally:reopened.sandbox.close()
    finally:h.sandbox.close()


def test_recipes_are_deterministic_frozen_and_work_budget_is_not_multiplied(tmp_path,monkeypatch):
    h,_,_,_=completed_stage(tmp_path,monkeypatch)
    try:
        anchor=h.catalog['quality:ordinary']; contract=h.coaching_contracts[anchor.key]
        before=asdict(anchor)
        doses,contracts=compile_choices(anchor,contract)
        assert (doses,contracts)==compile_choices(anchor,contract)
        assert asdict(anchor)==before
        budget=sum(s for k,s in anchor.phases if k=='work')
        for kind in KINDS:
            pair=[d for d in doses if d.workout_definition['family']==kind]
            assert len(pair)==2
            actual=sorted(sum(s for k,s in d.phases if k=='work') for d in pair)
            assert actual[0]<actual[1]
            # Timed commands commit to clean clock increments, not hidden
            # unrounded targets. Distance-based efforts retain exact timings.
            increment={'short_intervals':0,'progression':30,'hill_repeats':5,'fartlek':15}.get(kind,30)
            for d,expected in zip(sorted(pair,key=lambda d:sum(s for k,s in d.phases if k=='work')),[.8*budget,budget]):
                bouts=[s for k,s in d.phases if k=='work']
                assert abs(sum(bouts)-expected)<=len(bouts)*increment/2+1e-8
                if increment:assert all(s/increment==round(s/increment) for s in bouts)
            for d in pair:
                assert d.miles==round(d.miles,1)
                if d.workout_definition['structure']:
                    steps=[WorkoutStep.model_validate(s) for s in d.workout_definition['structure']]
                    assert sum(s.duration_minutes for s in steps)*60==pytest.approx(sum(s for _,s in d.phases))
        available={d.key:True for d in doses if d.workout_definition['family']=='threshold'}
        catalog={d.key:d for d in doses}
        refs=reference_keys(SimpleNamespace(ordinary_miles=4.,long_miles=8.,quality_work_minutes=budget/60),
            {**catalog,'e':SimpleNamespace(role='easy',miles=4.),'l':SimpleNamespace(role='long',miles=8.)},
            {**available,'e':True,'l':True})
        assert sum(s for k,s in catalog[refs['quality']].phases if k=='work')==budget
    finally:h.sandbox.close()


def test_settings_api_saves_working_choices_and_rejects_empty_selection(tmp_path):
    from fastapi.testclient import TestClient
    from test_adaptive_settings import adopted
    from run_analysis.web.app import create_app
    adopted(tmp_path)
    client=TestClient(create_app(tmp_path))
    settings=client.get('/api/settings').json()
    coaching={**settings['coaching'],'quality_sessions':{k:k=='progression' for k in KINDS}}
    response=client.patch('/api/settings',json={'coaching':coaching})
    assert response.status_code==200,response.text
    assert client.get('/api/settings').json()['coaching']['quality_sessions']==coaching['quality_sessions']
    bad={**coaching,'quality_sessions':dict.fromkeys(KINDS,False)}
    response=client.patch('/api/settings',json={'coaching':bad})
    assert response.status_code==422
    assert client.get('/api/settings').json()['coaching']['quality_sessions']==coaching['quality_sessions']


def test_readable_clock_not_decimal_minutes():
    from run_analysis.adaptive.quality_choices import _time
    assert _time(210)=='3:30'
    assert _time(180)=='3 min'
    assert _time(45)=='45 seconds'


def test_preference_change_invalidates_cached_search_signature():
    from datetime import datetime,timezone
    from run_analysis.adaptive.live_bridge import input_signature
    at=datetime(2026,10,8,tzinfo=timezone.utc)
    h=SimpleNamespace(zone=timezone.utc,decision=lambda x:x.date(),history=(),
        policy=HarnessPolicy(),
        config=only({},'short_intervals'))
    captured=SimpleNamespace(contexts=(),restrictions=(),responses=(),metadata=())
    before=input_signature(h,captured,at)
    h.config=only({},'threshold')
    assert input_signature(h,captured,at)!=before


def test_focus_change_invalidates_cached_signature_but_default_is_exact():
    from datetime import datetime,timezone
    from run_analysis.adaptive.live_bridge import input_signature
    at=datetime(2026,10,10,tzinfo=timezone.utc)
    h=SimpleNamespace(zone=timezone.utc,decision=lambda x:x.date(),history=(),
        policy=HarnessPolicy(),config=only({},'long_intervals'))
    captured=SimpleNamespace(contexts=(),restrictions=(),responses=(),metadata=())
    before=input_signature(h,captured,at)
    h.config['coaching']['quality_focus']='variety'
    assert input_signature(h,captured,at)==before
    h.config['coaching']['quality_focus']='vo2max'
    assert input_signature(h,captured,at)!=before


def test_settings_focus_roundtrip_validation_and_reset(tmp_path):
    from fastapi.testclient import TestClient
    from test_adaptive_settings import adopted
    from run_analysis.web.app import create_app
    adopted(tmp_path)
    client=TestClient(create_app(tmp_path))
    coaching=client.get('/api/settings').json()['coaching']
    assert coaching['quality_focus']=='variety'
    choices=coaching['quality_sessions'].copy()
    coaching['quality_focus']='vo2max'
    response=client.patch('/api/settings',json={'coaching':coaching})
    assert response.status_code==200,response.text
    assert response.json()['coaching']['quality_focus']=='vo2max'
    assert response.json()['coaching']['quality_sessions']==choices
    assert client.get('/api/settings').json()['coaching']['quality_focus']=='vo2max'
    bad={**coaching,'quality_focus':'random'}
    assert client.patch('/api/settings',json={'coaching':bad}).status_code==422
    assert client.get('/api/settings').json()['coaching']['quality_focus']=='vo2max'
    response=client.post('/api/settings/reset-advanced')
    assert response.status_code==200,response.text
    assert response.json()['coaching']['quality_focus']=='vo2max'
