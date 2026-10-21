from datetime import timedelta

import pytest

from scripts.audit_conditional_upload import path_trace, replace_path_cost, scoring, window
from scripts.audit_continuation_upload import saved_run_time_pace
from scripts.expected_completion import ExpectedCompletion
from scripts.replay_synthetic_easy_dose import synthetic_history
from run_analysis.selection_objective import EXPECTED_COMPLETION_STATE
from test_typed_continuation import fixture, CONFIG


def test_fixed_window_requires_all_dates_and_excludes_end():
    _, program = fixture()
    days, _ = program(8, {0: ('easy', 4.)})
    assert window(days, days[1].date, days[5].date) == days[1:5]
    with pytest.raises(ValueError, match='exactly'):
        window(days[:3]+days[4:], days[0].date, days[-1].date)
    with pytest.raises(ValueError, match='exactly'):
        window(days+days[-1:], days[0].date, days[-1].date+timedelta(days=1))


def test_historical_pace_is_explicit_and_restored_after_failure():
    _, program = fixture()
    days, states = program(2, {0: ('easy', 4.)})
    template, history = synthetic_history(states[0].as_of)
    model = ExpectedCompletion(template, history, CONFIG)
    item = days[0].recommendation
    original = ExpectedCompletion._pace_reference_at
    assert model._pace_reference_at(item).hour == 0
    with pytest.raises(RuntimeError):
        with saved_run_time_pace(True):
            assert model._pace_reference_at(item) == item.planned_for
            raise RuntimeError('expected')
    assert ExpectedCompletion._pace_reference_at is original


def test_scoring_context_restores_conditional_model_on_failure():
    _, program = fixture()
    _, states = program(2, {})
    template, history = synthetic_history(states[0].as_of)
    original = ExpectedCompletion._pace_reference_at
    with pytest.raises(RuntimeError):
        with scoring(dict(config=CONFIG), dict(template=template, runs=history), True):
            assert EXPECTED_COMPLETION_STATE.get() is not None
            raise RuntimeError('expected')
    assert EXPECTED_COMPLETION_STATE.get() is None
    assert ExpectedCompletion._pace_reference_at is original


def test_saved_failure_trace_is_reproduced_without_search_or_a_report_file():
    _, program = fixture()
    days, states = program(9, {0: ('intervals', 4.5), 2: ('long', 8.), 5: ('easy', 3.8)})
    for i, day in enumerate(days):
        if day.recommendation:
            days[i] = day.model_copy(update={'recommendation': day.recommendation.model_copy(
                update={'planned_for': day.recommendation.planned_for.replace(hour=7)})})
    states[0] = states[0].model_copy(update={'recent_load': states[0].recent_load.model_copy(
        update={'continuous_distance_miles': 14.316826550509477})})
    targets = ((16.4,17.4),(16.2,17.3),(16.2,17.3),(16.5,17.5),(16.5,17.5),
               (17.2,18.3),(17.4,18.4),(17.4,18.5),(17.3,18.4))
    inputs = dict(config=CONFIG, daily_states=states,
                  expected_target_projector=lambda planned,n: targets[:n])
    before = path_trace(days, inputs, length=7)
    extended = path_trace(days, inputs, length=8)
    following = states[1].model_copy(update={'recent_load': states[1].recent_load.model_copy(
        update={'continuous_distance_miles': extended['days'][0]['rate']})})
    after = dict(inputs, daily_states=[following,*states[2:]],
                 expected_target_projector=lambda planned,n: targets[1:1+n])
    shifted = path_trace(days[1:], after, length=7)
    anchored = path_trace(days[1:], after, length=7,
        anchor=dict(at=states[0].as_of,rate=before['opening_rate'],target=targets[0]))
    assert before['worst'] == 0
    assert extended['worst'] == pytest.approx(.9278552500211497)
    assert shifted['worst'] == pytest.approx(1.708729718014844)
    assert anchored['days'] == extended['days'][1:]
    assert anchored['worst'] == extended['worst']
    assert [r['rate'] for r in shifted['days']] == [r['rate'] for r in anchored['days']]
    assert shifted['days'][-1]['lower'] > anchored['days'][-1]['lower']


def test_path_accounting_substitution_preserves_all_other_costs():
    _, program = fixture()
    _, states = program(1,{})
    c=dict(total=12.,target=4.,target_violation=1.)
    result=replace_path_cost(dict(components=c),dict(daily_states=states),old_path=1.,new_path=0.)
    assert result==8.
    with pytest.raises(AssertionError):
        replace_path_cost(dict(components=c),dict(daily_states=states),old_path=2.,new_path=0.)
