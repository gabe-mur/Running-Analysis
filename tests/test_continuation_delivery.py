import pytest

from scripts.audit_continuation_delivery import variants, revised_easy, shortfall_fees, check_sequence
from scripts.typed_continuation_prototype import runs
from run_analysis import weekly_schedule as planner
from test_typed_continuation import fixture, CONFIG


def test_variants_hold_prefix_and_add_equal_mileage_without_mutation():
    _, program = fixture()
    days, states = program(49, {18: ('long', 8.), 23: ('easy', 4.), 30: ('easy', 4.), 38: ('easy', 4.)})
    before = [d.model_dump_json() for d in days]
    total = sum(planner._midpoint(r) for r in runs(days))
    choices = variants(days, states, 21)
    assert {label for label, _, _ in choices} == {'rest_control', 'insert_short', 'extend_one', 'extend_two'}
    assert len(choices) <= 100
    for label, edits, candidate in choices:
        assert [d.model_dump_json() for d in candidate[:21]] == before[:21]
        assert all(i >= 21 for i in edits)
        assert all(candidate[i] == days[i] for i in range(49) if i not in edits)
        added = sum(planner._midpoint(r) for r in runs(candidate)) - total
        assert added == pytest.approx(0 if label == 'rest_control' else 2.)
        if label == 'insert_short':
            assert days[edits[0]].recommendation is None
    assert [d.model_dump_json() for d in days] == before


def test_ordinary_insertion_is_a_separate_control_not_mislabeled_short():
    _, program = fixture()
    days, states = program(28, {23: ('easy', 4.)})
    choices = variants(days, states, 21, added_miles=4.)
    inserted = [x for x in choices if x[0] == 'insert_ordinary']
    assert inserted
    assert not any(x[0] == 'insert_short' for x in choices)
    for _, edits, candidate in inserted:
        result = candidate[edits[0]].recommendation
        assert result.distance_target_miles == 4.
        assert result.quality_session_type is None and result.structure == []


@pytest.mark.parametrize('distance', [0., .2, 2.05, float('nan')])
def test_revised_easy_requires_representable_distance(distance):
    _, program = fixture()
    days, _ = program(1, {0: ('easy', 4.)})
    r = days[0].recommendation
    with pytest.raises(ValueError, match='one-decimal'):
        revised_easy(days[0], r, r.planned_for, distance, 4., 5.)


def test_shortfall_ablation_removes_only_existing_per_run_charges():
    _, program = fixture()
    days, states = program(1, {0: ('easy', 1.5)})
    low = planner.typical_easy_distance(states[0])[0]
    missing = max(0., 1 - 1.5 / low)
    request = planner.RecommendationRequest(health_status='normal')
    assert shortfall_fees(days, states, request) == pytest.approx(32 * missing)
    full, _ = program(1, {0: ('easy', low)})
    assert shortfall_fees(full, states, request) == 0


def test_guard_check_reprojects_the_downstream_not_only_the_edit(monkeypatch):
    state_at, program = fixture()
    days, states = program(28, {22: ('easy', 2.), 24: ('long', 8.)})
    calls = []
    def select(options, planned, request, config, **kwargs):
        state = options[0]
        calls.append(tuple(planner._midpoint(r) for r in planned))
        index = (state.as_of.date() - days[0].date).days
        result = days[index].recommendation
        if planned:
            result = result.model_copy(update={'workout_type': planner.WorkoutType.EASY})
        return state, result
    monkeypatch.setattr(planner, '_select_budgeted_timed_recommendation', select)
    inputs = {'daily_states': states, 'request': planner.RecommendationRequest(health_status='normal'),
              'config': CONFIG, '_audit_targets': [(17., 18.)] * 28}
    checked = check_sequence(days, inputs, state_at, (22,))
    assert calls == [(), (2.,)]
    assert not checked['checks_pass']
    assert 24 in checked['failed_offsets']
    assert not checked['menu'][0]['within_intent_band']
