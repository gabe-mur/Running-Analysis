from dataclasses import replace
from datetime import date, timedelta

import pytest

from scripts.check_frequency_preference import without_short_preference
from run_analysis.adaptive.frequency_preference_trial import (
    ANCHORS, FrequencyPreference, frequency_center, running_days_per_7,
)
from run_analysis.adaptive.program_objective_trial import (
    EasyDurationPolicy, TrialSession, easy_duration_components,
)
from run_analysis.workload_accounting import Quantity
from test_memory_covered_continuation import guarded


@pytest.mark.parametrize('volume,expected', ANCHORS)
def test_explicit_smooth_volume_centers(volume, expected):
    assert frequency_center(volume) == pytest.approx(expected)
    if volume:
        assert abs(frequency_center(volume+.001)-frequency_center(volume-.001)) < .002


def test_three_days_are_a_preference_not_a_limit_at_lower_volume():
    policy = FrequencyPreference(5, .1, .01)
    assert policy.cost(3) == 0
    # A sufficient fatigue improvement can beat the center, in either direction.
    assert .09+policy.cost(2) < .1+policy.cost(3)
    assert .09+policy.cost(4) < .1+policy.cost(3)
    assert frequency_center(1) < 3


def test_zero_coefficient_is_fatigue_only_and_bound_is_weak():
    for n in range(8):
        assert FrequencyPreference(20, .1, 0).cost(n) == 0
        assert 0 <= FrequencyPreference(20, .1, .01).cost(n) <= .004
    assert FrequencyPreference(20, .1, .01).cost(5) == pytest.approx(.001)


def test_preference_does_not_depend_on_candidate_volume_pace_role_or_doubles():
    today = date(2026, 9, 30)
    days = [today, today+timedelta(days=2), today+timedelta(days=4)]
    assert running_days_per_7(days+days, span_days=7) == 3
    for repeats in (1, 3, 4, 16):
        replicated = [day+timedelta(days=7*i+5) for i in range(repeats) for day in days]
        assert running_days_per_7(replicated, span_days=7*repeats) == 3
    # No actual dose/role argument: underfunding cannot lower the target center.
    policy = FrequencyPreference(20, .1, .01)
    assert policy.center == 4


def test_remove_only_short_selection_cost_retains_minimum_and_volume_checks():
    context = replace(guarded()[0], easy_duration_policy=EasyDurationPolicy(10, 25, .01, 'test', 7.7))
    modified = without_short_preference(context)
    assert modified.easy_duration_policy == replace(context.easy_duration_policy, maximum_penalty=0.)
    assert modified.easy_duration_policy.minimum_minutes == 10
    assert modified.easy_duration_policy.supporting_easy_miles_per_7 == 7.7
    assert modified.easy_duration_policy.penalty(12) == 0
    assert modified.policy == context.policy
    assert modified.scenario == context.scenario
    day = context.decision
    session = TrialSession('below-dose-floor', day, .5, 'easy',
        tuple(Quantity.point(v, 'test') for v in (9., 9., 0.)))
    results = [easy_duration_components((), (session,), stage_start=day, decision=day,
        delivery_end=day+timedelta(days=7), policy=c.easy_duration_policy, bound_fraction=.5)
        for c in (context, modified)]
    assert results[0]['cost'] > 0 == results[1]['cost']
    for key in ('below_minimum_keys', 'uncertain_minimum_keys', 'supporting_easy_volume'):
        assert results[0][key] == results[1][key]
    assert results[1]['below_minimum_keys'] == ('below-dose-floor',)
    assert results[1]['supporting_easy_volume']['shortfall'] > 0


@pytest.mark.parametrize('volume', [-1, float('nan'), float('inf')])
def test_invalid_volume_rejected(volume):
    with pytest.raises(ValueError):
        frequency_center(volume)


@pytest.mark.parametrize('reference,coefficient', [(0, .01), (-1, .01), (float('nan'), .01), (.1, -.01), (.1, float('inf'))])
def test_invalid_policy_rejected(reference, coefficient):
    with pytest.raises(ValueError):
        FrequencyPreference(20, reference, coefficient)


@pytest.mark.parametrize('span', [0, -1, 1.5, True])
def test_invalid_exposure_rejected(span):
    with pytest.raises(ValueError):
        running_days_per_7([], span_days=span)


def test_invalid_count_rejected():
    with pytest.raises(ValueError):
        running_days_per_7(range(8), span_days=7)
    for n in (-1, 8, float('nan')):
        with pytest.raises(ValueError):
            FrequencyPreference(20, .1, .01).cost(n)
