from dataclasses import replace
import pytest

from run_analysis.recovery_assessment import AssessmentPolicy, assess_next_session, replace_expected
from run_analysis.recovery_response import ResponseExpectation, assess_response
from run_analysis.workload_accounting import Quantity
from test_recovery_assessment import NOW, work, entry

POLICY = AssessmentPolicy('three-channel arithmetic only', 150, 400, 24, 24, 20, 24)
POINT = lambda x: Quantity.point(x, 'declared fixture')


def test_structured_work_can_limit_without_repricing_cardio():
    past = replace(entry(hours=24), work_minutes=POINT(15))
    result = assess_next_session([past], work(45, 150), NOW, POLICY, proposed_work_minutes=POINT(15))
    assert result['limiting_demands'] == ['work_minutes']
    assert result['channels']['cardio_points']['total_low'] == 200
    assert result['channels']['work_minutes']['total_low'] == 22.5
    easy = assess_next_session([past], work(30, 60), NOW, POLICY, proposed_work_minutes=POINT(0))
    assert easy['status'] == 'model_supported'


def test_unknown_work_is_not_zero_and_control_is_unchanged():
    unknown = assess_next_session([entry()], work(30, 60), NOW, POLICY, proposed_work_minutes=POINT(0))
    assert unknown['status'] == 'uncertain'
    control = replace(POLICY, work_minutes_envelope=None, work_half_life_hours=None)
    assert assess_next_session([entry()], work(30, 60), NOW, control)['status'] == 'model_supported'


def test_structured_expected_actual_replacement_parity():
    expected = replace(entry(workload=work(source='projected')), work_minutes=POINT(10))
    actual = replace(expected, workload=replace(expected.workload, source='observed'))
    kwargs = dict(proposed_work_minutes=POINT(0))
    assert assess_next_session([expected], work(30, 60), NOW, POLICY, **kwargs) == assess_next_session(
        replace_expected([expected], actual), work(30, 60), NOW, POLICY, **kwargs)


def test_high_effort_can_be_expected_and_good_feeling_cannot_cancel_difficulty():
    hard = ResponseExpectation('issued example', Quantity(7, 9), Quantity(50, 100))
    assert not assess_response(hard, effort=8, feeling=75)['review_needed']
    easy = ResponseExpectation('issued example', Quantity(2, 4), Quantity(50, 100))
    result = assess_response(easy, effort=8, feeling=100)
    assert result['review_reasons'] == ['effort']
    assert not result['planner_influence']


def test_no_history_required_missing_is_unknown_and_good_report_not_progression():
    expected = ResponseExpectation('explicit starting assumption', Quantity(3, 5))
    result = assess_response(expected)
    assert result['channels']['effort']['status'] == 'unknown'
    assert not result['review_needed']
    easy = assess_response(expected, effort=2, feeling=100)
    assert not easy['automatic_progression']
    assert assess_response(expected, effort=2, readiness_concern=True)['review_needed']
    assert assess_response(expected, effort=4, feedback_conflict=True)['review_needed']


@pytest.mark.parametrize('value', [-1, 11, float('nan'), True])
def test_invalid_report(value):
    with pytest.raises(ValueError):
        assess_response(ResponseExpectation('fixture'), effort=value)


def test_policy_requires_complete_work_channel():
    with pytest.raises(ValueError):
        replace(POLICY, work_half_life_hours=None)


def test_structural_dose_cannot_exceed_total_and_more_work_does_not_reduce_cost():
    with pytest.raises(ValueError):
        replace(entry(), work_minutes=POINT(100))
    with pytest.raises(ValueError):
        assess_next_session([], work(20, 60), NOW, POLICY, proposed_work_minutes=POINT(30))
    past = replace(entry(hours=24), work_minutes=POINT(15))
    costs = [assess_next_session([past], work(45, 150), NOW, POLICY,
                                proposed_work_minutes=POINT(w))['excess_low']
             for w in (0, 10, 15, 20, 30)]
    assert costs == sorted(costs)


def test_reference_grid_is_reproducible_and_does_not_select_defaults():
    from scripts.audit_recovery_baseline import audit
    first, second = audit(), audit()
    assert first['groups'] == second['groups']
    assert not first['planner_influence']
    for result in first['groups'].values():
        assert 0 < result['accepted'] < result['tested']
        assert all(value > 0 for value in result['accepted_by_half_life'].values())
