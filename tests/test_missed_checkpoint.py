"""Target interpretation/atomic wiring, not a missed-run behavioral pass."""
from dataclasses import replace
from datetime import timedelta,timezone

from test_prospective_progression import fixture,AT
from run_analysis.adaptive.missed_checkpoint import propose


def test_only_due_unconfirmed_executed_shortfall_can_propose_maintenance():
    stage,targets=fixture();at=stage.review_at+timedelta(hours=4)
    assert propose(stage,targets,at,'UTC',4,0.)['candidate'] is None
    assert propose(stage,targets,AT,'UTC',4,3.)['candidate'] is None
    confirmed=replace(stage,weekly_equivalent_miles=21.)
    assert propose(confirmed,targets,at,'UTC',4,3.)['candidate'] is None
    result=propose(stage,targets,at,'UTC',4,3.)
    accepted=result['candidate'];path=result['amended_target']
    assert accepted.weekly_equivalent_miles==stage.weekly_equivalent_miles
    assert accepted.long_miles==stage.long_miles and accepted.ordinary_miles==stage.ordinary_miles
    assert accepted.policy.volume_goal_miles==stage.weekly_equivalent_miles
    assert path.ramp_days==0 and path.volume_per_7==stage.weekly_equivalent_miles
    assert path.effective_on==at.date() and accepted.issued_at==at
    assert not result['confirmed_delivery']
    assert targets[-1].volume_goal==21. # historical path untouched


def test_missing_checkpoint_maintenance_is_admitted_atomic_and_restartable(tmp_path,monkeypatch):
    from test_adaptive_stage_lifecycle import completed_stage
    from run_analysis.adaptive.adaptive_stage_lifecycle import review_entries,advance_stage
    from run_analysis.adaptive.stage_acceptance import StageLedger
    from run_analysis.adaptive.new_model_daily_harness import DailyHarness
    h,inputs,config,at=completed_stage(tmp_path,monkeypatch,growth_focus='easy_volume',
        continuation_boundary=True,single_volume=True,gradual=True,prospective_review_hour=4)
    entries=review_entries(h,at);observed=tuple(h.history);before=h.stage
    # Explicit review-credit fixture: removing credit never erases physical load.
    monkeypatch.setattr('run_analysis.adaptive.adaptive_stage_lifecycle.review_entries',
        lambda *_:tuple(replace(r,completion=None) for r in entries))
    at+=timedelta(hours=4)
    try:
        updated,review,event=advance_stage(h,inputs,at)
        assert review['status']=='conditional_maintenance_candidate',review
        assert event['reason']=='accepted_checkpoint_maintenance'
        assert event['stage_acceptance']
        assert not review['confirmed_delivery']
        assert updated.stage.weekly_equivalent_miles==before.weekly_equivalent_miles
        assert updated.stage.long_miles==before.long_miles
        assert tuple(updated.history)==observed
        assert StageLedger.for_sandbox(updated.sandbox).current==updated.stage
        updated.sandbox.close()
        resumed=DailyHarness.resume(tmp_path/'state',config,as_of=at)
        try:
            assert resumed.stage==updated.stage
            preview=resumed.replan(at+timedelta(seconds=1),reason='maintenance restart fixture',publish=False)
            assert preview['state'].prospective_volume_review is None
        finally:resumed.sandbox.close()
    finally:h.sandbox.close()


def test_planner_underdelivery_is_not_mislabeled_as_a_missed_run(tmp_path,monkeypatch):
    from test_adaptive_stage_lifecycle import completed_stage
    from run_analysis.adaptive.adaptive_stage_lifecycle import advance_stage
    h,inputs,_,at=completed_stage(tmp_path,monkeypatch,growth_focus='easy_volume',
        continuation_boundary=True,single_volume=True,gradual=True,prospective_review_hour=4)
    before=h.stage
    monkeypatch.setattr('run_analysis.adaptive.adaptive_stage_lifecycle.review_recent_delivery',
        lambda *a,**k:dict(status='recent_delivery_hold_targets',volume_accounting={
            'execution_gap_miles':0.,'planner_shortfall_miles':3.}))
    try:
        updated,review,event=advance_stage(h,inputs,at+timedelta(hours=4))
        assert event is None and updated.stage==before
        assert review['status']=='recent_delivery_hold_targets'
    finally:h.sandbox.close()


def test_only_applied_resolved_taper_can_interrupt_unconfirmed_growth():
    import sqlite3
    from test_race_goal_integration import revision
    from run_analysis.adaptive.race_event import record_resolution
    from run_analysis.adaptive.race_goal import RaceGoalApplication
    stage,targets=fixture();at=stage.review_at+timedelta(days=1,hours=4)
    race=revision(AT,days=7)
    db=sqlite3.connect(':memory:')
    outcome=record_resolution(db,revision=race,history=(),timezone_name='UTC',
        status='did_not_start',physical_key=None,expected_revision=race.key,as_of=at)
    application=RaceGoalApplication(race,AT,AT.date())
    kwargs=dict(race_goal=race,race_event=outcome,race_goal_history=(application,))
    result=propose(stage,targets,at,'UTC',4,0.,**kwargs)
    assert result['candidate'] is not None and result['race_interrupted']
    assert result['candidate'].weekly_equivalent_miles==stage.weekly_equivalent_miles
    assert not result['confirmed_delivery']
    assert propose(stage,targets,at,'UTC',4,0.,race_goal=race)['candidate'] is None
    assert propose(stage,targets,at,'UTC',4,0.,race_goal=race,race_event=outcome)['candidate'] is None
    # A race applied after the checkpoint cannot explain that earlier failure.
    late=RaceGoalApplication(race,at,at.date())
    assert propose(stage,targets,at,'UTC',4,0.,**dict(kwargs,race_goal_history=(late,)))['candidate'] is None
    db.close()


def test_prior_date_unconfirmed_opportunity_pauses_optional_ramp_before_deadline():
    stage,targets=fixture();at=AT+timedelta(days=7)
    missed=(('original-issued-fingerprint',(AT+timedelta(days=5)).date().isoformat()),)
    assert propose(stage,targets,at,'UTC',4,0.)['candidate'] is None
    result=propose(stage,targets,at,'UTC',4,0.,unconfirmed_opportunities=missed)
    assert result['candidate'].weekly_equivalent_miles==stage.weekly_equivalent_miles
    assert result['candidate'].long_miles==stage.long_miles
    assert result['original_checkpoint']>at
    assert not result['confirmed_delivery'] and result['unconfirmed_opportunities']==missed
    import pytest
    with pytest.raises(ValueError,match='past-date'):
        propose(stage,targets,at,'UTC',4,0.,unconfirmed_opportunities=(('same-day',at.date().isoformat()),))


def test_maintenance_identity_survives_snapshot_utc_normalization():
    stage,targets=fixture();at=AT+timedelta(days=7)
    missed=(('original-issued-fingerprint',(AT+timedelta(days=5)).date().isoformat()),)
    local=at.astimezone(timezone(timedelta(hours=-4)))
    assert propose(stage,targets,local,'America/New_York',4,0.,unconfirmed_opportunities=missed)==propose(
        stage,targets,at,'America/New_York',4,0.,unconfirmed_opportunities=missed)


def test_cleared_goal_keeps_only_causally_applied_interrupted_phase_evidence():
    import sqlite3
    from test_race_goal_integration import revision
    from run_analysis.adaptive.race_event import record_resolution
    from run_analysis.adaptive.race_goal import RaceGoalApplication
    stage,targets=fixture();at=stage.review_at+timedelta(days=1,hours=4)
    race=revision(AT,days=10)
    db=sqlite3.connect(':memory:')
    outcome=record_resolution(db,revision=race,history=(),timezone_name='UTC',
        status='did_not_start',physical_key=None,expected_revision=race.key,as_of=AT+timedelta(days=11))
    clear=revision(AT+timedelta(days=12),distance='general_fitness')
    application=RaceGoalApplication(race,AT,AT.date())
    cleared=RaceGoalApplication(clear,clear.available_at,clear.available_at.date())
    kwargs=dict(race_goal=clear,race_goal_history=(application,cleared),race_event_history=(outcome,))
    result=propose(stage,targets,at,'UTC',4,0.,**kwargs)
    assert result['candidate'] is not None and result['race_interrupted']
    assert not result['confirmed_delivery']
    # Canceling before this goal's taper never interrupts the pending phase.
    early=revision(AT+timedelta(days=1),distance='general_fitness')
    early_clear=RaceGoalApplication(early,early.available_at,early.available_at.date())
    assert propose(stage,targets,at,'UTC',4,0.,**dict(kwargs,
        race_goal=early,race_goal_history=(application,early_clear)))['candidate'] is None
    db.close()
