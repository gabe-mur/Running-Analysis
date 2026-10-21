from copy import deepcopy
from datetime import timedelta

import pytest

from run_analysis.workload_accounting import Quantity
from run_analysis.workload_baseline import baseline_expectation, attach_historical_context
from run_analysis.workload_expectation import forecast, compare_distribution
from test_workload_expectation import AT, POLICY, intent, profile, refs


def baseline(**assumptions):
    return baseline_expectation(intent(), issued_at=AT, assumptions=assumptions,
                                assumption_source='synthetic contract fixture, not a physiological prior')


@pytest.mark.parametrize('count', [0, 1, 2, 3, 4, 30])
def test_no_matching_history_requirement_or_three_session_switch(count):
    prior = baseline(duration_minutes=Quantity(40, 50), cardio_points=Quantity(60, 120))
    before = deepcopy(prior)
    history = forecast(intent(), refs(count=count), AT, POLICY)
    output = attach_historical_context(prior, history)
    assert output['expectation'] == before == prior
    assert not output['personalization_applied']
    assert not output['adaptation_authorized']


def test_missing_response_assumptions_remain_unknown_not_zero():
    prior = baseline()
    assert prior['expected']['distance_meters'] == intent().distance_meters
    assert prior['expected']['work_distance_meters'].low == 2400
    assert prior['expected']['cardio_points'] is None
    result = compare_distribution(prior, profile(coverage=0), {})
    assert result['features']['cardio_points']['status'] == 'unknown'
    assert not result['adaptation_authorized']


def test_conflicting_history_is_visible_without_rewriting_baseline():
    prior = baseline(cardio_points=Quantity(10, 20))
    result = attach_historical_context(prior, forecast(intent(), refs(), AT, POLICY))
    assert result['context']['cardio_points']['status'] == 'above_baseline'
    assert result['expectation']['expected']['cardio_points'] == Quantity(10, 20)
    assert not result['personalization_applied']


def test_prescription_duration_is_not_overwritten_by_response_assumption():
    from dataclasses import replace
    given = replace(intent(), duration_minutes=Quantity(40, 45))
    result = baseline_expectation(given, issued_at=AT,
        assumptions={'duration_minutes': Quantity(30, 60)}, assumption_source='fixture')
    assert result['expected']['duration_minutes'] == Quantity(40, 45)


def test_cannot_override_distance_with_baseline_guess():
    with pytest.raises(ValueError, match='Unrecognized'):
        baseline(distance_meters=Quantity(1000, 3000))


def test_different_information_cutoff_is_rejected():
    history = forecast(intent(), refs(), AT+timedelta(days=1), POLICY)
    with pytest.raises(ValueError, match='cutoff'):
        attach_historical_context(baseline(), history)


def test_missing_hrv_never_changes_baseline_comparison():
    prior = baseline(cardio_points=Quantity(60, 120))
    actual = profile()
    before = compare_distribution(prior, actual, {})
    actual['hrv'] = {'dfa': .7, 'valid': False}
    assert compare_distribution(prior, actual, {}) == before


def test_impossible_response_timing_is_not_accepted():
    with pytest.raises(ValueError, match='accommodate'):
        baseline(duration_minutes=Quantity(1, 2), work_seconds=Quantity(600, 700))


def test_shared_projected_workload_can_supply_baseline_without_history():
    from run_analysis.workload_accounting import projected_workload
    from run_analysis.workload_profile import prescribed_profile
    from run_analysis.workload_expectation import intent_from_prescription
    from test_workload_profile import CONFIG, prescription
    rec = prescription(3, 5, 2)
    projected = projected_workload(prescribed_profile(rec), duration_minutes=Quantity(40, 45),
        base_rate=Quantity(1, 2), work_rate=Quantity(3, 4))
    prior = baseline_expectation(intent_from_prescription(rec, CONFIG['zones']), issued_at=AT,
        assumptions=dict(duration_minutes=projected.running_minutes, cardio_points=projected.cardio_points),
        assumption_source='shared projection with synthetic phase-rate bounds')
    assert prior['expected']['cardio_points'] == projected.cardio_points
    assert prior['expected']['work_seconds'].low == 900
    assert prior['personalization'] == 'none'


@pytest.mark.parametrize('assumptions,source', [({}, ''), ({'cardio_points': 42}, 'fixture')])
def test_assumptions_require_named_bounded_inputs(assumptions, source):
    with pytest.raises(ValueError):
        baseline_expectation(intent(), issued_at=AT, assumptions=assumptions, assumption_source=source)
