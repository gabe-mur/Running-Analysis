from datetime import datetime,timedelta,timezone
from dataclasses import asdict,replace
from types import SimpleNamespace
import copy
import pytest

from run_analysis.goal_journey import goal_journey,conditional_path
from run_analysis.staged_progression import DevelopmentStage,ProgressionPolicy
from run_analysis.adaptive.race_goal import RaceGoalRevision

AT=datetime(2026,10,20,12,tzinfo=timezone.utc)


def fixture():
    stage=DevelopmentStage('seed',AT,18.1,4.2,8.9,17.,7.,
        ProgressionPolicy(14,.0816,'easy_volume',19.4,'gradual'),'test')
    raw=asdict(stage);raw['issued_at']=AT.isoformat()
    state=SimpleNamespace(as_of=AT,timezone='America/New_York',planner_preferences=None,
        current=None,race_goal=None,context=SimpleNamespace(race_event=None,race_event_history=()))
    progress=dict(volume=19.4,observed_volume_per_7=17.7,growth_per_7=.04,
        growth_allowed=True,long_growth_allowed=False,long_budget=8.9,
        long_support_fraction=1.,reference_provisional=True)
    event=dict(stage_runtime=dict(stage=raw,general_progression_state=progress,catalog=[
        dict(key='long',role='long',miles=8.9)]),program=[['2026-10-26','long']],hard_end='2026-11-10')
    runs=[dict(activity_id=1,date='2026-10-05',miles=9.536,role='long'),
        dict(activity_id=2,date='2026-10-15',miles=4.2,role='easy')]
    return state,event,runs


def test_exposes_actual_prescribed_and_conditional_distances_without_mutation():
    state,event,runs=fixture();original=copy.deepcopy(event)
    out=goal_journey(state,event,as_of=AT,completed=runs)
    assert out['recent_long_run']['miles']==9.536
    assert out['current_long_miles']==8.9
    assert out['target_long_miles']==17
    assert out['volume_growth_per_7']==.04
    assert out['status']=='supporting_volume'
    assert out['supporting_volume_for_long']==pytest.approx(22.25)
    assert out['projection']['target_date']=='2027-05-19'
    assert out['projection']['points'][0]['long_miles']==8.9
    assert out['scheduled']==[dict(date='2026-10-26',miles=8.9)]
    assert event==original


@pytest.mark.parametrize('change,status',[
    (dict(growth_allowed=False,growth_per_7=0.),'holding'),
    (dict(volume=25,observed_volume_per_7=25,long_growth_allowed=True),'progressing'),
    (dict(volume=25,observed_volume_per_7=25,long_growth_allowed=False),'long_hold'),
    (dict(volume=42.5,observed_volume_per_7=42.5),'long_hold'),
])
def test_status_comes_from_committed_controller_not_chart_slope(change,status):
    state,event,runs=fixture();event['stage_runtime']['general_progression_state'].update(change)
    out=goal_journey(state,event,as_of=AT,completed=runs)
    assert out['status']==status
    if status in {'holding','long_hold'}:
        assert not out['projection']['points'] and out['projection']['target_date'] is None


def test_pending_stale_health_or_future_decision_never_produces_an_eta():
    state,event,runs=fixture()
    assert goal_journey(state,event,as_of=AT,completed=runs,needs_refresh=True)['status']=='update_pending'
    out=goal_journey(state,event,as_of=AT+timedelta(days=1),completed=runs)
    assert out['projection']['target_date'] is None
    assert out['today']=='2026-10-21'
    state.current=SimpleNamespace(state='health_concern')
    assert goal_journey(state,event,as_of=AT,completed=runs)['status']=='review'
    assert not goal_journey(state,event,as_of=AT-timedelta(days=1),completed=runs)['available']


def test_insufficient_evidence_and_recent_recording_are_not_zero_or_all_time_best():
    state,event,runs=fixture()
    old=dict(activity_id=99,date='2023-10-01',miles=26.2,role='long')
    assert goal_journey(state,event,as_of=AT,completed=[old])['recent_long_run'] is None
    event['stage_runtime'].pop('general_progression_state')
    assert not goal_journey(state,event,as_of=AT,completed=runs)['available']


def test_race_path_stops_at_taper_and_does_not_invent_a_pace_forecast():
    state,event,runs=fixture()
    state.race_goal=RaceGoalRevision.create(available_at=AT,distance='marathon',
        race_date=AT.date()+timedelta(days=70),preparation_mode='balanced',parent_key='test')
    out=goal_journey(state,event,as_of=AT,completed=runs)
    assert out['goal_kind']=='marathon'
    assert out['target_long_miles']==20
    assert out['projection']['points'][-1]['date']==state.race_goal.taper_start.isoformat()
    assert out['projection']['target_date'] is None
    assert not out['race_projection_meets_target']
    assert 'pace' not in out
    state.as_of=datetime.combine(state.race_goal.taper_start,AT.time(),timezone.utc)
    out=goal_journey(state,event,as_of=state.as_of,completed=runs)
    assert out['status']=='taper' and not out['projection']['points']


def test_local_day_and_current_zero_support_do_not_permanently_penalize_future_adherence():
    state,event,runs=fixture()
    state.as_of=datetime(2026,10,21,1,tzinfo=timezone.utc)
    event['stage_runtime']['general_progression_state']['long_support_fraction']=0.
    out=goal_journey(state,event,as_of=state.as_of,completed=runs)
    assert out['today']=='2026-10-20'
    assert out['status']=='supporting_volume'
    assert not out['long_growth_allowed']
    assert out['current_long_miles']==8.9
    assert out['projection']['target_date']=='2027-05-19'


@pytest.mark.parametrize('fraction',[0.,.2,.8963786877003586,1.])
def test_current_delivery_fraction_is_not_a_permanent_forecast_rate(fraction):
    state,event,runs=fixture()
    original=copy.deepcopy(event)
    expected=goal_journey(state,event,as_of=AT,completed=runs)
    event['stage_runtime']['general_progression_state']['long_support_fraction']=fraction
    out=goal_journey(state,event,as_of=AT,completed=runs)
    assert out['projection']==expected['projection']
    assert not out['long_growth_allowed']  # No actual permission or credit added.
    original['stage_runtime']['general_progression_state']['long_support_fraction']=fraction
    assert event==original


def test_october_20_to_21_refresh_does_not_create_a_three_week_goal_delay():
    state,event,runs=fixture()
    yesterday=goal_journey(state,event,as_of=AT,completed=runs)
    today=AT+timedelta(days=1)
    state.as_of=today
    event['stage_runtime']['general_progression_state'].update(
        volume=19.50900248810106,observed_volume_per_7=17.685750581123752,
        long_support_fraction=.8963786877003586)
    out=goal_journey(state,event,as_of=today,completed=runs)
    assert out['projection']['target_date']==yesterday['projection']['target_date']=='2027-05-19'
    assert out['projection']['support_date']==yesterday['projection']['support_date']=='2026-11-15'
    assert out['volume_growth_per_7']==.04
    assert out['current_long_miles']==yesterday['current_long_miles']==8.9
    assert out['target_long_miles']==17


def test_future_support_gate_still_prevents_unsupported_long_run_growth():
    args=dict(volume=19.4,long_miles=8.9,long_budget=8.9,
        target_volume=42.5,target_long=17,volume_rate=.04,long_fraction=.05,
        long_days=14,long_share=.4)
    path=conditional_path(AT.date(),**args)
    support=datetime.fromisoformat(path['support_date']).date()
    assert support>AT.date()
    assert all(p['long_miles']==8.9 for p in path['points'] if p['date']<support.isoformat())
    for p in path['points']:
        assert p['long_miles']<=max(8.9,p['volume_per_7']*.4)+1e-8
    insufficient=conditional_path(AT.date(),**dict(args,target_volume=20.))
    assert insufficient['support_date'] is None
    assert insufficient['target_date'] is None
    assert all(p['long_miles']==8.9 for p in insufficient['points'])
    # Slower genuine policy inputs or a larger goal must still change the date.
    slower=conditional_path(AT.date(),**dict(args,long_fraction=.04))
    assert slower['target_date']>path['target_date']
    larger=conditional_path(AT.date(),**dict(args,target_long=18.,target_volume=45.))
    assert larger['target_date']>path['target_date']
    lower_start=conditional_path(AT.date(),**dict(args,volume=17.))
    assert lower_start['support_date']>path['support_date']
    assert lower_start['target_date']>path['target_date']


def test_race_future_adherence_does_not_freeze_current_shortfall_or_extend_past_taper():
    state,event,runs=fixture()
    state.race_goal=RaceGoalRevision.create(available_at=AT,distance='marathon',
        race_date=AT.date()+timedelta(days=140),preparation_mode='balanced',parent_key='test')
    expected=goal_journey(state,event,as_of=AT,completed=runs)
    event['stage_runtime']['general_progression_state']['long_support_fraction']=.2
    out=goal_journey(state,event,as_of=AT,completed=runs)
    assert out['projection']==expected['projection']
    assert out['projection']['points'][-1]['date']==state.race_goal.taper_start.isoformat()
    assert all(p['date']<=state.race_goal.taper_start.isoformat() for p in out['projection']['points'])
    assert all(p['long_miles']<=out['target_long_miles'] and p['volume_per_7']<=out['target_volume_per_7']
        for p in out['projection']['points'])
    assert not out['long_growth_allowed']


def test_fractional_rate_scenario_is_bounded_and_never_exceeds_targets():
    state,event,runs=fixture()
    out=goal_journey(state,event,as_of=AT,completed=runs)
    assert all(p['volume_per_7']<=42.5 and p['long_miles']<=17 for p in out['projection']['points'])
    assert conditional_path(AT.date(),volume=19.4,long_miles=8.9,long_budget=8.9,
        target_volume=42.5,target_long=17,volume_rate=0,long_fraction=.05,
        long_days=14,long_share=.4)['target_date'] is None
