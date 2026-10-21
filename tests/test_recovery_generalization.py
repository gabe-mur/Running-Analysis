from dataclasses import replace
from datetime import timedelta
import pytest

from scripts.audit_recovery_baseline import NOW, EASY, LONG, QUALITY, history
from scripts.audit_recovery_generalization import recurring_requirements, REFERENCE_CYCLE, scale_case
from run_analysis.recovery_assessment import AssessmentPolicy, assess_next_session, SessionEntry
from run_analysis.workload_accounting import Quantity


POLICY = AssessmentPolicy('test arithmetic, not defaults', 140, 300, 24, 24, 25, 24)


def test_fresh_support_does_not_prove_repeatability():
    tight = replace(POLICY, running_minutes_envelope=90)
    assert assess_next_session([], LONG[0], NOW, tight, proposed_work_minutes=LONG[1])['status'] == 'model_supported'
    assert not recurring_requirements(tight)['supported']
    assert recurring_requirements(tight)['channels']['running_minutes']['margin'] < 0
    assert recurring_requirements(POLICY)['supported']


def test_infinite_cycle_sum_matches_long_explicit_history():
    exact = recurring_requirements(POLICY, offsets=(0,))
    maxima = {k: 0 for k in exact['channels']}
    for day, proposed in REFERENCE_CYCLE:
        entries = []
        for previous_day, previous in REFERENCE_CYCLE:
            first_gap = (day-previous_day) % 8 or 8
            for cycle in range(30):
                start = NOW-timedelta(days=first_gap+8*cycle)
                entries.append(SessionEntry(f'{previous_day}/{cycle}', start,
                    start+timedelta(minutes=previous[0].running_minutes.high), previous[0], previous[1]))
        result = assess_next_session(entries, proposed[0], NOW, POLICY, proposed_work_minutes=proposed[1])
        for name, channel in result['channels'].items():
            maxima[name] = max(maxima[name], channel['total_high'])
    for name, value in maxima.items():
        assert exact['channels'][name]['required_budget'] == pytest.approx(value)


def test_more_easy_days_raise_requirements_without_a_consecutive_day_ban():
    original = recurring_requirements(POLICY)
    added = recurring_requirements(POLICY, cycle=(*REFERENCE_CYCLE, (6, EASY)))
    assert added['channels']['running_minutes']['required_budget'] > original['channels']['running_minutes']['required_budget']
    assert added['channels']['work_minutes']['required_budget'] == original['channels']['work_minutes']['required_budget']


def test_single_workout_cycle_has_expected_geometric_sum():
    result = recurring_requirements(POLICY, cycle=((0, EASY),), cycle_days=1, offsets=(0,))
    # Current 30 min + earlier 30-min sessions starting every 24 h and ending 0.5 h later.
    expected = 30 + 30 * 2**(-23.5/24) / (1-2**(-24/24))
    assert result['channels']['running_minutes']['required_budget'] == pytest.approx(expected)


def test_structure_alias_is_an_explicit_model_limitation():
    a = replace(QUALITY[0], structure={'rep_seconds': [300]*3})
    b = replace(QUALITY[0], structure={'rep_seconds': [90]*10})
    kwargs = dict(proposed_work_minutes=QUALITY[1])
    assert assess_next_session(history(EASY, (24,)), a, NOW, POLICY, **kwargs) == assess_next_session(
        history(EASY, (24,)), b, NOW, POLICY, **kwargs)


def test_arithmetic_scaling_is_not_a_new_runner_calibration():
    past, proposed = history(LONG, (24,)), EASY
    original = assess_next_session(past, proposed[0], NOW, POLICY, proposed_work_minutes=proposed[1])
    for scale in (.5, 1.5):
        entries, candidate, policy = scale_case(past, proposed, POLICY, scale)
        result = assess_next_session(entries, candidate[0], NOW, policy, proposed_work_minutes=candidate[1])
        assert result['excess_high'] == pytest.approx(original['excess_high'])


def test_unmeasured_work_is_bounded_not_zero_or_permanent_uncertainty():
    unknown = tuple(replace(e, work_minutes=None) for e in history(QUALITY, (24,)))
    now = assess_next_session(unknown, EASY[0], NOW, POLICY, proposed_work_minutes=EASY[1])
    work = now['channels']['work_minutes']
    assert work['residual_low'] == 0
    assert work['residual_high'] == pytest.approx(22.5)
    assert work['contributions'][0]['basis'] == 'unknown_work_bounded_by_running_duration'
    tighter = replace(POLICY, work_minutes_envelope=20)
    assert assess_next_session(unknown, EASY[0], NOW, tighter,
                               proposed_work_minutes=EASY[1])['status'] == 'uncertain'
    later = assess_next_session(unknown, EASY[0], NOW+timedelta(days=7), tighter,
                                proposed_work_minutes=EASY[1])
    assert later['status'] == 'model_supported'
    assert later['channels']['work_minutes']['residual_high'] > 0
    unknown_candidate = assess_next_session([], QUALITY[0], NOW, tighter)
    assert unknown_candidate['status'] == 'uncertain'
    assert unknown_candidate['channels']['work_minutes']['total_high'] == 45


def test_unknown_duration_cannot_supply_a_work_bound():
    workload = replace(QUALITY[0], running_minutes=None)
    result = assess_next_session([], workload, NOW, POLICY)
    assert result['channels']['work_minutes']['total_high'] is None
    assert result['status'] == 'uncertain'


def test_explicit_work_bounds_are_preserved():
    result = assess_next_session([], QUALITY[0], NOW, POLICY,
                                 proposed_work_minutes=Quantity(10, 18, basis='caller bounds'))
    assert result['channels']['work_minutes']['total_low'] == 10
    assert result['channels']['work_minutes']['total_high'] == 18


@pytest.mark.parametrize('cycle,days', [((), 8), (REFERENCE_CYCLE, 0), (((0, EASY), (0, LONG)), 8)])
def test_invalid_cycles(cycle, days):
    with pytest.raises(ValueError):
        recurring_requirements(POLICY, cycle=cycle, cycle_days=days)
