from dataclasses import replace
from datetime import date, timedelta
from math import exp

import pytest

from run_analysis.recency_demand import CHANNELS, DemandEvent, DemandTerm, compare_recency
from run_analysis.recovery_response import ResponseExpectation, assess_response
from run_analysis.workload_accounting import Quantity

DAY = date(2026, 9, 25)


def event(key, age=0, amount=1., *, bounds=None, variable='unit', purpose='easy', structure=None):
    q = bounds if bounds is not None else Quantity.point(1., 'fixture')
    return DemandEvent(key, DAY-timedelta(days=age),
        tuple(DemandTerm(c, variable, amount, q) for c in CHANNELS),
        'normalized_hr_reserve_v1', purpose, structure, 'planned')


def test_more_recent_identical_dose_is_no_less_demanding():
    result = compare_recency([event('a', 1)], event('next'), [event('b', 3)], event('next'))
    assert result['recency_demand_order'] == 'right_no_greater'
    assert not result['running_authorized']


def test_older_larger_and_newer_smaller_cross():
    r = compare_recency([event('a', 3, 2)], event('next'), [event('b', 1)], event('next'))
    assert set(r['channel_orders'].values()) == {'crossing_recency_tradeoff'}
    assert r['recency_demand_order'] == 'no_joint_order'


@pytest.mark.parametrize('bounds', [Quantity(0, 100, basis='unknown'), None])
def test_shared_uncertainty_cancels_before_bounds(bounds):
    h = replace(event('common', 2), terms=tuple(DemandTerm(c, 'shared', 1, bounds) for c in CHANNELS))
    r = compare_recency([h], event('next'), [h], event('next'))
    assert r['recency_demand_order'] == 'equal'
    assert all(x['left_minus_right_low'] == x['left_minus_right_high'] == 0
               for curve in r['cutoff_differences'].values() for x in curve)


@pytest.mark.parametrize('bounds', [Quantity(0, 100, basis='unknown'), None])
def test_independent_uncertainty_does_not_cancel(bounds):
    a = replace(event('a', 2), terms=tuple(DemandTerm(c, 'a', 1, bounds) for c in CHANNELS))
    b = replace(a, key='b', terms=tuple(replace(t, variable='b') for t in a.terms))
    r = compare_recency([a], event('next'), [b], event('next'))
    assert set(r['channel_orders'].values()) == {'uncertain'}


def test_shared_rate_coupling_orders_different_doses():
    q = Quantity(2, 10, basis='scenario_not_prediction')
    r = compare_recency([], event('a', amount=3, bounds=q), [], event('b', amount=4, bounds=q))
    assert r['recency_demand_order'] == 'left_no_greater'
    assert r['next_dose_changed']


def test_next_dose_is_included_at_age_zero():
    r = compare_recency([], event('a', amount=2), [], event('b'))
    assert r['recency_demand_order'] == 'right_no_greater'
    assert r['cutoff_differences']['miles'] == [dict(age_days=0, left_minus_right_low=1., left_minus_right_high=1.)]


def test_source_replacement_preserves_accounting():
    planned = event('a', 2)
    observed = replace(planned, source='observed')
    assert compare_recency([planned], event('next'), [observed], event('next'))['recency_demand_order'] == 'equal'


def test_channel_tradeoffs_are_not_collapsed_into_a_score():
    a = event('a')
    b = replace(event('b'), terms=tuple(replace(t, coefficient=2 if t.channel == 'miles' else .5) for t in a.terms))
    r = compare_recency([], a, [], b)
    assert r['channel_orders']['miles'] == 'left_no_greater'
    assert r['channel_orders']['cardio'] == 'right_no_greater'
    assert r['recency_demand_order'] == 'no_joint_order'


def test_different_structure_is_not_equivalent_training():
    a = event('a', purpose='quality', structure=(300., 300., 300.))
    b = replace(a, key='b', structure=(150.,)*6)
    r = compare_recency([], a, [], b)
    assert r['recency_demand_order'] == 'equal'
    assert not r['same_next_purpose_and_structure']
    assert not r['progression_accepted']


def test_response_concern_is_not_cancelled_by_lower_demand():
    response = assess_response(ResponseExpectation('fixture'), readiness_concern=True)
    r = compare_recency([], event('a'), [], event('b', amount=2), response_review=response)
    assert r['response_action'] == 'review_reported_concern'
    assert not r['running_authorized']


@pytest.mark.parametrize('age', [0, -1])
def test_same_day_or_future_history_rejected(age):
    with pytest.raises(ValueError, match='same-day/future'):
        compare_recency([event('past', age)], event('next'), [], event('next'))


def test_duplicate_id_rejected():
    with pytest.raises(ValueError, match='Duplicate'):
        compare_recency([event('next', 1)], event('next'), [], event('next'))


def test_units_rejected():
    with pytest.raises(ValueError, match='accounting'):
        compare_recency([], event('a'), [], replace(event('b'), accounting_key='edwards'))


def test_inconsistent_shared_bounds_rejected():
    with pytest.raises(ValueError, match='Shared variables'):
        compare_recency([], event('a'), [], event('b', bounds=Quantity.point(2, 'fixture')))


def test_all_channels_must_be_explicit():
    with pytest.raises(ValueError, match='Every channel'):
        replace(event('a'), terms=event('a').terms[:-1])


def test_date_translation_does_not_change_order():
    a, b, n = event('a', 1), event('b', 4), event('next')
    shifted = lambda e: replace(e, day=e.day+timedelta(days=365))
    assert compare_recency([a], n, [b], n)['channel_orders'] == compare_recency(
        [shifted(a)], shifted(n), [shifted(b)], shifted(n))['channel_orders']


@pytest.mark.parametrize('rate', [0., .001, .1, 1., 10.])
def test_order_agrees_with_decreasing_weight_examples(rate):
    # Demonstrate the cumulative-cutoff identity, not fit a recovery half-life.
    a, b, n = event('a', 5, 2), event('b', 1, 3), event('next')
    result = compare_recency([a], n, [b], n)
    assert result['recency_demand_order'] == 'left_no_greater'
    assert 1+2*exp(-rate*5) <= 1+3*exp(-rate*1)
