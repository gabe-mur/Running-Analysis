from dataclasses import replace
from datetime import date, timedelta

import pytest

from run_analysis.race_preparation import (MARATHON_POLICIES, PreparationExposureEvidence,
    RacePreparationGoal, marathon_preparation_options, next_phase, destination)
from run_analysis.staged_progression import ProgressionPolicy
from tests.test_race_preparation import AT, stage


def options(s=None,**changes):
    args=dict(race_date=date(2027,12,1),as_of=AT,longest_prior_30d_miles=8.9,response_state='clear')
    args.update(changes)
    return marathon_preparation_options(s or stage(),**args)


def test_preparation_options_have_distinct_disclosed_doses_rates_and_allowance():
    r=options(); short,balanced,more=r['options']
    assert (short['target'].long_miles,short['target'].volume_per_7)==(17.,35.)
    assert (balanced['target'].long_miles,balanced['target'].volume_per_7)==(20.,40.)
    assert (more['target'].long_miles,more['target'].volume_per_7)==(20.,40.)
    assert more['target'].source!=balanced['target'].source
    assert short['earliest_conditional_date']<balanced['earliest_conditional_date']<more['earliest_conditional_date']
    assert [o['interruption_allowance_days'] for o in r['options']]==[0,14,28]
    assert short['status']=='conditional_limited_preparation' and short['warnings']
    assert not r['may_publish_calendar'] and not r['whole_program_checked'] and r['injury_probability'] is None


def test_existing_committed_phase_and_historical_stage_do_not_change():
    s=stage();r=options(s)
    for o in r['options']:
        first=o['phases'][0]
        assert first['end']==s.review_at and first['volume_end']==19. and first['long_miles']==8.9
        assert o['phases'][1]['end']-o['phases'][1]['start']==timedelta(days=7)
    assert s==stage() and s.policy.growth_fraction==.05 and s.goal_long_miles==17.


def test_deadline_does_not_change_doses_rates_or_compress_a_build():
    early=options(race_date=date(2026,10,20));late=options()
    assert early['status']=='no_window_under_tested_scenarios'
    for a,b in zip(early['options'],late['options']):
        assert a['phases']==b['phases'] and a['earliest_conditional_date']==b['earliest_conditional_date']
        assert not a['requested_date_fits'] and not a['accepted']


def test_each_hypothetical_long_respects_session_guard_and_existing_volume():
    for o in options()['options']:
        previous_long=8.9;available_volume=18.1
        target=o['target']
        for phase in o['phases']:
            assert phase['long_miles']<=previous_long*1.1+1e-8
            if phase['long_miles']>previous_long:
                assert phase['long_miles']<=available_volume*target.long_miles/target.volume_per_7+1e-8
            previous_long=phase['long_miles'];available_volume=phase['volume_end']


@pytest.mark.parametrize('state,expected',[('adverse','response_review_required'),('health_concern','health_review_required')])
def test_adverse_or_unresolved_growth_evidence_does_not_offer_an_approved_build(state,expected):
    r=options(response_state=state)
    assert r['status']==expected and not r['options'] and not r['may_publish_calendar']


def test_missing_optional_response_does_not_erase_conditional_guidance_or_claim_clearance():
    r=options(response_state='unknown')
    assert r['options'] and r['requires_response_review'] and not r['may_publish_calendar']
    assert r['status']=='conditional_options_pending_response_review'
    assert all(o['warnings'] for o in r['options'])


def test_actual_reference_is_required_not_invented_from_stage_or_race_goal():
    assert options(longest_prior_30d_miles=None)['status']=='observed_session_reference_required'
    assert options(longest_prior_30d_miles=6.)['status']=='held_long_requires_session_guard_review'


def test_future_evidence_or_past_race_rejects():
    e=PreparationExposureEvidence(AT+timedelta(days=1),AT,40.,20.,14,'fixture')
    with pytest.raises(ValueError):options(consolidation_evidence=e)
    with pytest.raises(ValueError):options(race_date=AT.date())


def test_established_exposure_does_not_repeat_a_full_program_or_fake_consolidation_from_age():
    s=stage(issued_at=AT-timedelta(days=60),weekly_equivalent_miles=40.,long_miles=20.,
            policy=ProgressionPolicy(14,.05))
    without=options(s,longest_prior_30d_miles=20.)['options'][1]
    assert not without['phases'] and without['consolidation_remaining_days']==14
    evidence=PreparationExposureEvidence(AT,AT,40.,20.,14,'observed-fixture:recent-maintained-exposure')
    with_credit=options(s,longest_prior_30d_miles=20.,consolidation_evidence=evidence)['options'][1]
    assert with_credit['consolidation_remaining_days']==0
    assert with_credit['earliest_conditional_date']==AT.date()+timedelta(days=21)
    assert not with_credit['whole_program_checked']


def test_new_stage_issue_does_not_discard_already_observed_consolidation():
    s=stage(weekly_equivalent_miles=40.,long_miles=20.,policy=ProgressionPolicy(14,.05))
    evidence=PreparationExposureEvidence(AT,AT,40.,20.,14,'observed-dose-before-adoption')
    r=options(s,longest_prior_30d_miles=20.,consolidation_evidence=evidence)['options'][1]
    assert r['earliest_conditional_date']==AT.date()+timedelta(days=21)


def test_old_consolidation_cannot_credit_a_future_higher_dose():
    e=PreparationExposureEvidence(AT,AT,40.,20.,60,'fixture')
    assert all(o['observed_consolidation_credit_days']==0 for o in options(consolidation_evidence=e)['options'])


def test_buffer_warning_is_separate_from_minimum_build_window():
    baseline=options()['options'][1]
    r=options(race_date=baseline['earliest_conditional_date'])['options'][1]
    assert r['requested_date_fits'] and not r['interruption_allowance_fits']
    assert r['status']=='conditional_limited_preparation' and r['warnings']


def test_bounded_failure_is_not_a_preparation_pass():
    r=options(maximum_phases=1)
    assert r['status']=='no_window_under_tested_scenarios'
    assert all(o['outlook_status']=='projection_limit' and o['earliest_conditional_date'] is None for o in r['options'])


def test_unreviewed_stage_does_not_start_new_growth_early():
    s=stage(policy=ProgressionPolicy(14,.05))
    r=options(s)
    assert all(o['phases'][0]['start']==s.review_at for o in r['options'])


@pytest.mark.parametrize('mode',['shorter','more'])
def test_other_distances_do_not_claim_marathon_options_are_validated(mode):
    with pytest.raises(ValueError):RacePreparationGoal('half_marathon',date(2027,1,1),mode)


def test_separate_policy_proposals_cannot_override_the_ten_percent_session_limit():
    s=stage(weekly_equivalent_miles=30.,long_miles=10.,policy=ProgressionPolicy(14,.05))
    target=destination(s,RacePreparationGoal('marathon',date(2027,12,1),'shorter'))
    r=next_phase(s,target,as_of=AT,longest_prior_30d_miles=10.,phase_policy=ProgressionPolicy(7,.16))
    assert r['candidate'].long_miles==11. and r['candidate'].policy.volume_goal_miles==34.8
    assert not r['accepted']


def test_insufficient_current_support_does_not_borrow_future_volume():
    s=stage(weekly_equivalent_miles=17.5)
    target=destination(s,RacePreparationGoal('marathon',date(2027,12,1)))
    r=next_phase(s,target,as_of=AT,longest_prior_30d_miles=8.9,
                 phase_policy=MARATHON_POLICIES['balanced'].progression)
    assert r['candidate'].long_miles==8.9 and r['candidate'].policy.volume_goal_miles>17.8


def test_stale_observation_cannot_become_recent_consolidation_merely_by_new_availability():
    s=stage(weekly_equivalent_miles=40.,long_miles=20.,policy=ProgressionPolicy(14,.05))
    e=PreparationExposureEvidence(AT,AT-timedelta(days=60),40.,20.,60,'old-observation')
    r=options(s,longest_prior_30d_miles=20.,consolidation_evidence=e)['options'][1]
    assert r['observed_consolidation_credit_days']==0


def test_projected_windows_do_not_restart_the_existing_phase_on_reload():
    a=options()['options'];b=options(as_of=AT+timedelta(days=2))['options']
    for first,later in zip(a,b):
        assert first['earliest_conditional_date']==later['earliest_conditional_date']
