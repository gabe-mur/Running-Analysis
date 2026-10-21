from dataclasses import replace
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from run_analysis.workload_accounting import Quantity, Workload
from run_analysis.recovery_assessment import (
    AssessmentPolicy, SessionEntry, assess_next_session, replace_expected, sensitivity_panel,
)

NOW = datetime(2026, 9, 24, 12, tzinfo=timezone.utc)
POLICY = AssessmentPolicy('arithmetic_control', 100, 250, 12, 12)


def work(minutes=50., cardio=100., source='observed'):
    return Workload(source, Quantity.point(5000, 'fixture'), Quantity.point(minutes, 'fixture'),
                    Quantity.point(cardio, 'fixture'), {})


def entry(key='run', hours=12, workload=None):
    return SessionEntry(key, NOW-timedelta(hours=hours+1), NOW-timedelta(hours=hours), workload or work())


def test_shared_expected_and_actual_evidence_has_identical_recovery():
    actual = entry(workload=work())
    expected = replace(actual, workload=replace(actual.workload, source='projected'))
    proposed = work(40, 80, 'projected')
    assert assess_next_session([actual], proposed, NOW, POLICY) == assess_next_session([expected], proposed, NOW, POLICY)


def test_completion_replaces_once_instead_of_adding():
    expected = entry(workload=work(source='projected'))
    actual = replace(expected, workload=work(60, 140))
    ledger = replace_expected([expected], actual)
    assert len(ledger) == 1
    assert ledger[0] == actual
    result = assess_next_session(ledger, work(30, 60), NOW, POLICY)
    assert result['channels']['running_minutes']['residual_low'] == 30
    assert result['channels']['cardio_points']['residual_low'] == 70
    with pytest.raises(ValueError):
        replace_expected(ledger, actual)
    with pytest.raises(ValueError):
        replace_expected([], actual)


def test_actual_upload_does_not_redefine_the_reference_envelope():
    expected = entry(workload=work(source='projected'))
    actual = replace(expected, workload=work(90, 240))
    before = assess_next_session([expected], work(60, 120), NOW, POLICY)
    after = assess_next_session(replace_expected([expected], actual), work(60, 120), NOW, POLICY)
    assert before['status'] == 'model_supported'
    assert after['status'] == 'model_contradicted'
    assert before['channels']['running_minutes']['envelope'] == after['channels']['running_minutes']['envelope']


def test_no_additional_rest_reward_inside_supported_region():
    short = assess_next_session([entry()], work(30, 60), NOW, POLICY)
    longer = assess_next_session([entry()], work(30, 60), NOW+timedelta(hours=24), POLICY)
    assert short['excess_low'] == short['excess_high'] == 0
    assert longer['excess_low'] == longer['excess_high'] == 0
    assert longer['channels']['running_minutes']['residual_low'] < short['channels']['running_minutes']['residual_low']


def test_next_demand_matters_and_channels_cannot_cancel():
    history = [entry()]
    assert assess_next_session(history, work(30, 60), NOW, POLICY)['status'] == 'model_supported'
    hard = assess_next_session(history, work(10, 220), NOW, POLICY)
    assert hard['status'] == 'model_contradicted'
    assert hard['limiting_demands'] == ['cardio_points']
    long = assess_next_session(history, work(90, 10), NOW, POLICY)
    assert long['limiting_demands'] == ['running_minutes']


def test_uncertainty_not_silent_zero_or_hard_fail():
    uncertain = replace(work(), cardio_points=Quantity(0, 500, basis='missing_hr'))
    result = assess_next_session([entry(workload=uncertain)], work(30, 100), NOW, POLICY)
    assert result['status'] == 'uncertain'
    assert result['channels']['cardio_points']['total_low'] == 100
    assert result['channels']['cardio_points']['total_high'] == 350


def test_accumulation_and_duplicate_future_rejection():
    one, two = entry('one'), entry('two', 24)
    result = assess_next_session([one, two], work(20, 40), NOW, POLICY)
    assert result['channels']['running_minutes']['residual_low'] == 37.5
    for history in ([one, one], [entry('future', -1)]):
        with pytest.raises(ValueError):
            assess_next_session(history, work(), NOW, POLICY)


def test_same_history_makes_timing_disagreement_explicit():
    result = sensitivity_panel([entry()], work(80, 100), NOW, [POLICY])
    assert result['status'] == 'uncertain'
    assert not result['agreement']
    assert len(result['scenarios']) == 3


@pytest.mark.parametrize('hours', [0, 6, 12, 24, 48, 96])
def test_more_rest_and_smaller_dose_never_increase_excess(hours):
    history = [entry()]
    large = assess_next_session(history, work(95, 240), NOW, POLICY)
    small = assess_next_session(history, work(80, 200), NOW+timedelta(hours=hours), POLICY)
    assert small['excess_high'] <= large['excess_high']


def test_elapsed_time_uses_utc_across_dst():
    zone = ZoneInfo('America/New_York')
    end = datetime(2026, 11, 1, 0, tzinfo=zone)
    run = SessionEntry('dst', end-timedelta(hours=1), end, work())
    result = assess_next_session([run], work(), datetime(2026, 11, 1, 3, tzinfo=zone), POLICY)
    assert result['channels']['running_minutes']['contributions'][0]['elapsed_hours'] == 4


def test_no_naive_clock_or_implicit_policy_constants():
    with pytest.raises(ValueError):
        assess_next_session([], work(), datetime(2026, 9, 24), POLICY)
    with pytest.raises(ValueError):
        replace(POLICY, running_half_life_hours=0)
