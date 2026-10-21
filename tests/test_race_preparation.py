from dataclasses import replace
from datetime import date, datetime, timedelta, timezone

import pytest

from run_analysis.race_preparation import RacePreparationGoal, destination, next_phase, preparation_outlook
from run_analysis.staged_progression import DevelopmentStage, ProgressionPolicy

AT = datetime(2026, 10, 6, 12, tzinfo=timezone.utc)


def stage(**changes):
    s = DevelopmentStage('accepted', AT, 18.1, 4.4, 8.9, 17., 7.,
                         ProgressionPolicy(14, .05, 'easy_volume', 19., 'gradual'), 'fixture')
    return replace(s, **changes)


def test_race_priority_preserves_accepted_dose_and_general_goal():
    s = stage()
    g = RacePreparationGoal('5k', date(2027, 1, 27))
    target = destination(s, g)
    assert target.long_miles == 8.9  # Not the unrelated 17-mile general goal.
    assert target.volume_per_7 == 19.  # Existing authorized ramp retained.
    assert s.goal_long_miles == 17. and s.weekly_equivalent_miles == 18.1


def test_marathon_coordinates_supporting_volume_before_long_growth():
    s = stage()
    target = destination(s, RacePreparationGoal('marathon', date(2027, 12, 1)))
    assert (target.long_miles, target.volume_per_7) == (20., 40.)
    result = next_phase(s, target, as_of=AT, longest_prior_30d_miles=8.9)
    c = result['candidate']
    assert c.long_miles == 9.0  # Existing18.1 supports at most9.0, not the19.0 future endpoint.
    assert c.weekly_equivalent_miles == 18.1 and c.policy.volume_goal_miles == 19.
    assert c.policy.volume_path == 'gradual' and not result['accepted']
    assert result['supporting_volume_required_for_current_long'] == 17.8


def test_long_growth_uses_observed_guard_not_goal_or_future_volume():
    s = stage(weekly_equivalent_miles=25., long_miles=10.,
              policy=ProgressionPolicy(14, .05, 'easy_volume', 26.3, 'gradual'))
    target = destination(s, RacePreparationGoal('marathon', date(2027, 12, 1)))
    result = next_phase(s, target, as_of=AT, longest_prior_30d_miles=9.2)
    assert result['candidate'].long_miles == 10.1  # Conservative floor of 9.2 * 1.1.
    missing = next_phase(s, target, as_of=AT, longest_prior_30d_miles=None)
    assert missing['candidate'] is None


def test_existing_phase_does_not_restart_from_outlook_reload():
    s = stage()
    g = RacePreparationGoal('marathon', date(2027, 12, 1))
    first = preparation_outlook(s, g, as_of=AT)
    later = preparation_outlook(s, g, as_of=AT + timedelta(days=3))
    assert first['phases'][0]['end'] == later['phases'][0]['end'] == s.review_at
    assert first['earliest_race_date'] == later['earliest_race_date']
    assert first['requested_date_fits'] and not first['promised_fitness_gain']
    assert first['estimated_race_pace'] is None and not first['accepted']


def test_short_deadline_does_not_change_progression_rate_or_force_faster_doses():
    s = stage()
    early = preparation_outlook(s, RacePreparationGoal('marathon', date(2027, 1, 27)), as_of=AT)
    later = preparation_outlook(s, RacePreparationGoal('marathon', date(2028, 1, 27)), as_of=AT)
    assert not early['requested_date_fits'] and later['requested_date_fits']
    assert early['phases'] == later['phases']
    assert early['earliest_race_date'] == later['earliest_race_date']
    for p in early['phases']:
        assert p['volume_end'] <= round(p['volume_start'] * 1.05, 1) + .1


def test_higher_volume_is_not_reduced_to_beginner_scenario():
    s = stage(weekly_equivalent_miles=60., long_miles=22.,
              policy=ProgressionPolicy(14, .05))
    g = RacePreparationGoal('marathon', date(2027, 1, 27))
    outlook = preparation_outlook(s, g, as_of=AT)
    assert (outlook['target'].volume_per_7, outlook['target'].long_miles) == (60., 22.)
    assert not outlook['phases']


@pytest.mark.parametrize('distance,when', [('ultra', date(2027, 1, 1)), ('5k', '2027-01-01')])
def test_invalid_goal_is_not_silently_invented(distance, when):
    with pytest.raises(ValueError):
        RacePreparationGoal(distance, when)


def test_projection_limit_is_explicit_not_claimed_feasible():
    result = preparation_outlook(stage(), RacePreparationGoal('marathon', date(2027, 12, 1)),
                                 as_of=AT, maximum_phases=1)
    assert result['status'] == 'projection_limit'
    assert result['earliest_race_date'] is None and not result['requested_date_fits']
