from dataclasses import replace
from datetime import timedelta
import pytest

from run_analysis.recovery_assessment import AssessmentPolicy
from run_analysis.recovery_response import ResponseExpectation
from run_analysis.response_dose_review import DoseOption, review_next_dose
from run_analysis.workload_accounting import Quantity, Workload
from test_recovery_assessment import NOW, entry, work

P = AssessmentPolicy('arithmetic only', 60, 200, 24, 24)
EXPECTED = ResponseExpectation('issued easy exertion assumption', Quantity(2, 4))
Q = lambda value: Quantity.point(value, 'test fixture')


def option(miles=4, key='issued'):
    return DoseOption(key, 'next-session', NOW, 'easy', miles,
        Workload('projected', Q(miles*1609.344), Q(miles*10), Q(miles*20), {}), Q(0))


def review(**overrides):
    args = dict(issued=option(), alternatives=[option(3, '3'), option(3.5, '3.5'), option(3.9, '3.9')],
        history=[entry(hours=24)], policies=[P], expected_response=EXPECTED,
        expected_history=[entry(hours=24, workload=work(40, 80, 'projected'))],
        report_activity_key='run', minimum_target_miles=3, adjustment_source='explicit test bound',
        readiness='normal', effort=4, offsets_hours=(0,))
    args.update(overrides)
    return review_next_dose(**args)


def test_smallest_supported_supplied_reduction_not_rpe_multiplier():
    result = review()
    assert result['action'] == 'propose_bounded_reduction'
    assert result['target_miles'] == 3.5
    assert result['original_target_miles'] == 4
    assert not result['planner_influence']
    assert review(effort=None)['target_miles'] == 3.5
    assert review(effort=10)['target_miles'] == 3.5


def test_rpe_alone_does_not_reprice_model_load():
    supported = replace(P, running_minutes_envelope=100)
    normal = review(policies=[supported])
    hard = review(policies=[supported], effort=8)
    assert normal['action'] == hard['action'] == 'retain_issued'
    assert hard['response']['review_needed']
    unknown = review(policies=[supported], effort=8, readiness='unknown')
    assert unknown['action'] == 'request_readiness_check'
    assert unknown['target_miles'] is None


def test_expected_hard_is_not_surprise_and_low_rpe_does_not_raise_target():
    supported = replace(P, running_minutes_envelope=100)
    hard = review(policies=[supported], effort=8, readiness='unknown',
                  expected_response=ResponseExpectation('hard fixture', Quantity(7, 9)))
    assert hard['action'] == 'retain_issued'
    assert not hard['response']['review_needed']
    low = review(policies=[supported], effort=1)
    assert low['target_miles'] == 4
    assert not low['automatic_progression']


@pytest.mark.parametrize('overrides,action', [
    ({'readiness': 'concern'}, 'defer_to_existing_readiness_rules'),
    ({'feedback_conflict': True}, 'review_conflicting_feedback'),
    ({'policies': [P, replace(P, running_minutes_envelope=100)]}, 'review_model_uncertainty'),
    ({'minimum_target_miles': 3.9, 'alternatives': [option(3.9, '3.9')]}, 'review_beyond_adjustment_scope'),
])
def test_review_does_not_silently_move_delete_or_shrink_beyond_scope(overrides, action):
    result = review(**overrides)
    assert result['action'] == action
    assert result['selected_option_key'] is None


def test_reloading_cannot_compound_reduction_against_immutable_issue():
    original = option()
    assert review(issued=original) == review(issued=original)
    assert original.target_miles == 4
    assert review(alternatives=list(reversed([option(3, '3'), option(3.5, '3.5')])) )['target_miles'] == 3.5


@pytest.mark.parametrize('change', [
    {'planned_for': NOW+timedelta(days=1)}, {'workout_kind': 'long'},
    {'session_key': 'other'}, {'target_miles': 2.9, 'workload': option(2.9).workload},
    {'work_minutes': Q(5)},
    {'workload': replace(option(3).workload, cardio_points=Q(100))},
])
def test_forbidden_alternatives_are_rejected(change):
    with pytest.raises(ValueError):
        review(alternatives=[replace(option(3, 'alternative'), **change)])


def test_invalid_or_stale_reports_and_unrounded_targets():
    with pytest.raises(ValueError):
        option(3.55)
    with pytest.raises(ValueError):
        review(report_activity_key='not in ledger')
    with pytest.raises(ValueError):
        review(history=[entry(hours=24), entry(key='later', hours=12)])
    with pytest.raises(ValueError):
        review(history=[replace(entry(hours=24), workload=replace(entry().workload, source='projected'))])
    with pytest.raises(ValueError):
        review(minimum_target_miles=0)


def test_preexisting_bad_plan_is_not_mislabeled_upload_adaptation():
    result = review(expected_history=[entry(hours=24, workload=work(50, 100, 'projected'))])
    assert result['action'] == 'review_issue_baseline'
    assert result['issued_expected_status'] == 'model_contradicted'
    assert result['target_miles'] is None
    assert review(expected_history=None)['action'] == 'review_issue_baseline'


def test_perfect_execution_preserves_independent_recovery_assessment():
    expected = entry(hours=24, workload=work(40, 80, 'projected'))
    actual = replace(expected, workload=replace(expected.workload, source='observed'))
    result = review(history=[actual], expected_history=[expected])
    assert result['baseline_status'] == result['issued_expected_status'] == 'model_supported'
    assert result['action'] == 'retain_issued'
    assert result['target_miles'] == 4


def test_unrelated_history_rewrites_do_not_enter_rpe_review():
    with pytest.raises(ValueError):
        review(expected_history=[entry('other', hours=24, workload=work(40, 80, 'projected'))])
