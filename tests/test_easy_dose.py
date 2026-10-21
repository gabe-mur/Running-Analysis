from dataclasses import replace
from datetime import datetime, timezone

import pytest

from run_analysis.easy_dose import AerobicMinimum, EasyDurationReference, assess_easy_dose, minimum_easy_miles

REF = EasyDurationReference(datetime(2026, 9, 25, tzinfo=timezone.utc), 5., 10., 'fixture')
POLICY = AerobicMinimum(5., 'user_provisional_floor')


def test_floor_is_not_target_or_benefit_reward():
    for miles in (1., 4.4, 8.):
        r = assess_easy_dose(miles, REF, POLICY)
        assert r['passes_planned_minimum']
        assert r['session_benefit_score'] is None and not r['recovery_cleared']
    assert not assess_easy_dose(.9, REF, POLICY)['passes_planned_minimum']


def test_minimum_is_duration_based_for_fast_and_slow_runners():
    assert minimum_easy_miles(replace(REF, minutes_per_mile=5), POLICY) == 2.
    assert minimum_easy_miles(replace(REF, minutes_per_mile=20), POLICY) == .5


def test_rounding_cannot_undercut_minimum_or_silently_increase_existing_command():
    ref = replace(REF, startup_minutes=5.1)
    assert minimum_easy_miles(ref, POLICY) == 1.1
    result = assess_easy_dose(1., ref, POLICY)
    assert not result['passes_planned_minimum'] and result['committed_miles'] == 1.


def test_startup_counts_in_total_load_not_a_prediction_of_lost_aerobic_benefit():
    r = assess_easy_dose(2., REF, POLICY)
    assert r['total_minutes_at_reference_pace'] == 20
    assert r['intended_aerobic_budget_minutes'] == 15
    assert not r['observed_zone_minutes_predicted']


def test_startup_sensitivity_is_explicit_not_daily_reference_refitting():
    slow_start = replace(REF, startup_minutes=10)
    assert assess_easy_dose(1.2, REF, POLICY)['passes_planned_minimum']
    assert not assess_easy_dose(1.2, slow_start, POLICY)['passes_planned_minimum']
    assert REF.startup_minutes == 5


@pytest.mark.parametrize('miles', [0, -.1, 1.01, float('nan')])
def test_invalid_committed_distance(miles):
    with pytest.raises(ValueError):
        assess_easy_dose(miles, REF, POLICY)
