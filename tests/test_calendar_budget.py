from datetime import date

import pytest

from scripts.benchmark_calendar_budget import investigate, prescription_difference, cross_replays


def row(day, kind='easy', miles=3.3):
    return dict(date=day, prescription=dict(workout_type=kind, distance_target_miles=miles))


def test_budget_comparison_distinguishes_date_role_and_dose_changes():
    before=[row('2026-09-22'),row('2026-09-24'),row('2026-09-25'),row('2026-09-28')]
    after=[row('2026-09-22',miles=3.5),row('2026-09-24','long',8.),row('2026-09-26')]
    actual=prescription_difference(before,after,start=date(2026,9,22),horizon=4)
    assert actual == dict(date_changes=['2026-09-25'], type_changes=['2026-09-24'],
        distance_changes=[dict(date='2026-09-22',before=3.3,after=3.5)])


@pytest.mark.parametrize('budgets', [[],[-1],[True],[1.5],[40,40]])
def test_invalid_budgets_fail_before_any_replay(budgets):
    with pytest.raises(ValueError,match='budgets'):
        investigate({},lambda item:None,budgets=budgets,indices=[0])


def test_invalid_indices_fail_before_any_replay():
    with pytest.raises(ValueError,match='indices'):
        investigate({'replans':[{}]},lambda item:None,budgets=[0,40],indices=[1])


def test_cross_replay_rejects_different_completed_work():
    baseline = dict(synthetic_sandbox={'config': {}}, realized_execution={'miles': 3.3}, replans=[])
    alternative = {**baseline, 'realized_execution': {'miles': 4.}}
    with pytest.raises(ValueError, match='matched config and completed execution'):
        cross_replays(baseline, alternative)
