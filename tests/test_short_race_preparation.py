"""Distance-specific numerical/contract checks, not scheduler feasibility."""
from datetime import date, timedelta

import pytest

from run_analysis.race_preparation import (PROFILES, RacePreparationGoal, destination,
    preparation_outlook, next_phase, short_preparation_options, PreparationExposureEvidence,
    SHORT_PREPARATION_ALLOWANCES)
from run_analysis.staged_progression import ProgressionPolicy
from test_race_preparation import AT, stage


@pytest.mark.parametrize('distance', ['5k', '10k', 'half_marathon'])
@pytest.mark.parametrize('volume,long', [(5.,2.), (20.,8.), (60.,18.)])
def test_low_established_and_high_volume_preserve_evidence_and_guard(distance, volume, long):
    s = stage(weekly_equivalent_miles=volume, ordinary_miles=1., long_miles=long,
              policy=ProgressionPolicy(14,.05))
    goal = RacePreparationGoal(distance, date(2028,10,1))
    target = destination(s, goal)
    assert target.volume_per_7 >= volume and target.long_miles >= long
    r = preparation_outlook(s,goal,as_of=AT,longest_prior_30d_miles=long)
    assert r['status'] == 'conditional_preparation_window'
    previous_long, previous_volume = long, volume
    for phase in r['phases']:
        assert round(phase['long_miles'],1) == phase['long_miles']
        assert round(phase['volume_end'],1) == phase['volume_end']
        assert phase['long_miles'] <= previous_long*1.1+1e-8
        if phase['long_miles'] > previous_long:
            assert phase['long_miles'] <= previous_volume*target.long_miles/target.volume_per_7+1e-8
        # Marathon8/6/4% rates were NOT copied into short-distance policy.
        assert phase['end']-phase['start'] == timedelta(days=14)
        previous_long, previous_volume = phase['long_miles'], phase['volume_end']
    assert not r['accepted'] and r['estimated_race_pace'] is None
    assert s.goal_long_miles == 17.


@pytest.mark.parametrize('distance', ['5k', '10k', 'half_marathon'])
def test_deadline_does_not_accelerate_or_shrink_committed_phase(distance):
    s = stage()
    early = preparation_outlook(s,RacePreparationGoal(distance,AT.date()+timedelta(days=1)),as_of=AT,
                                longest_prior_30d_miles=8.9)
    late = preparation_outlook(s,RacePreparationGoal(distance,date(2028,10,1)),as_of=AT,
                               longest_prior_30d_miles=8.9)
    assert early['phases'] == late['phases']
    assert early['earliest_race_date'] == late['earliest_race_date']
    assert not early['requested_date_fits']
    assert late['phases'][0]['volume_end'] == 19.


@pytest.mark.parametrize('distance', ['5k', '10k', 'half_marathon'])
def test_missing_observed_reference_requires_review(distance):
    goal=RacePreparationGoal(distance,date(2028,10,1))
    r=next_phase(stage(),destination(stage(),goal),as_of=AT,longest_prior_30d_miles=None)
    assert r['status'] == 'observed_session_reference_required' and r['candidate'] is None


def test_half_and_10k_event_not_disguised_as_guarded_training():
    # Common coach training endpoints can be below race/1.1. Race accounting
    # must remain separate for these distances too; a prototype never clears it.
    for distance in ('10k','half_marathon'):
        assert PROFILES[distance].distance_miles > PROFILES[distance].long_miles*1.1


def test_each_short_distance_exposes_only_downstream_policy_provenance():
    for distance in ('5k','10k','half_marathon'):
        assert PROFILES[distance].sources
        assert all(source.startswith('docs/training_policy_inferences.md#')
                   for source in PROFILES[distance].sources)


@pytest.mark.parametrize('distance', ['5k', '10k', 'half_marathon'])
def test_short_guidance_adds_disclosed_settling_and_interruption_room(distance):
    s=stage();goal=RacePreparationGoal(distance,date(2028,10,1))
    raw=preparation_outlook(s,goal,as_of=AT,longest_prior_30d_miles=8.9)
    r=short_preparation_options(s,goal,as_of=AT,longest_prior_30d_miles=8.9)
    o=r['options'][0];settling,buffer=SHORT_PREPARATION_ALLOWANCES[distance]
    assert o['earliest_conditional_date']==raw['earliest_race_date']+timedelta(days=settling)
    assert o['recommended_conditional_date']==o['earliest_conditional_date']+timedelta(days=buffer)
    assert o['phases']==raw['phases'] and o['volume_growth_per_review']==.05
    assert o['review_days']==14 and o['warnings']
    assert r['requires_response_review'] and not r['may_publish_calendar']
    assert r['injury_probability'] is None and not o['accepted']
    tight=short_preparation_options(s,RacePreparationGoal(distance,o['earliest_conditional_date']),
        as_of=AT,longest_prior_30d_miles=8.9,response_state='clear')['options'][0]
    assert tight['requested_date_fits'] and not tight['interruption_allowance_fits']
    assert tight['status']=='conditional_limited_preparation'
    assert tight['phases']==o['phases']  # Deadline never accelerates the ramp.


@pytest.mark.parametrize('distance', ['5k', '10k', 'half_marathon'])
def test_already_prepared_needs_actual_maintained_exposure_for_settling_credit(distance):
    s=stage(weekly_equivalent_miles=60.,long_miles=18.,policy=ProgressionPolicy(14,.05))
    goal=RacePreparationGoal(distance,date(2028,10,1))
    evidence=PreparationExposureEvidence(AT,AT,60.,18.,30,'observed maintained exposure')
    no_credit=short_preparation_options(s,goal,as_of=AT,longest_prior_30d_miles=18.)['options'][0]
    r=short_preparation_options(s,goal,as_of=AT,longest_prior_30d_miles=18.,
        consolidation_evidence=evidence)['options'][0]
    assert not r['phases'] and r['consolidation_remaining_days']==0
    assert r['earliest_conditional_date']==AT.date()+timedelta(days=goal.profile.taper_days)
    assert no_credit['earliest_conditional_date']>r['earliest_conditional_date']
    stale=PreparationExposureEvidence(AT,AT-timedelta(days=2),60.,18.,30,'stale')
    assert short_preparation_options(s,goal,as_of=AT,longest_prior_30d_miles=18.,
        consolidation_evidence=stale)['options'][0]['observed_consolidation_credit_days']==0
    future=PreparationExposureEvidence(AT+timedelta(days=1),AT,60.,18.,30,'future')
    with pytest.raises(ValueError):
        short_preparation_options(s,goal,as_of=AT,longest_prior_30d_miles=18.,consolidation_evidence=future)


@pytest.mark.parametrize('state,expected', [('adverse','response_review_required'),('health_concern','health_review_required')])
def test_short_guidance_does_not_ignore_adverse_response(state,expected):
    r=short_preparation_options(stage(),RacePreparationGoal('10k',date(2028,10,1)),
        as_of=AT,longest_prior_30d_miles=8.9,response_state=state)
    assert r['status']==expected and not r['options'] and not r['may_publish_calendar']


def test_short_guidance_missing_guard_and_bounded_outlook_remain_unresolved():
    goal=RacePreparationGoal('half_marathon',date(2028,10,1))
    r=short_preparation_options(stage(),goal,as_of=AT,longest_prior_30d_miles=None)
    assert r['status']=='observed_session_reference_required' and not r['options']
    r=short_preparation_options(stage(),goal,as_of=AT,longest_prior_30d_miles=8.9,maximum_phases=1)
    assert r['options'][0]['earliest_conditional_date'] is None
    assert not r['options'][0]['requested_date_fits']
    with pytest.raises(ValueError):
        short_preparation_options(stage(),RacePreparationGoal('marathon',date(2028,10,1)),
            as_of=AT,longest_prior_30d_miles=8.9)
