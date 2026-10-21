import pytest

from scripts import continuation_opportunity_repair as repair
from scripts.typed_continuation_prototype import compare_candidate
from run_analysis import weekly_schedule as planner
from test_typed_continuation import fixture, CONFIG


def test_common_baseline_selects_best_guard_passing_insertion():
    candidates = [(21, 'bad_guard', {}), (22, 'better', {}), (23, 'best', {})]
    seen = []
    def evaluate(trial):
        seen.append(trial)
        return {'combined_cost': {'better': 4., 'best': 3.}[trial]}
    selected, score, rows = repair.select_best_insertion('base', {'combined_cost': 5.}, candidates,
        evaluate, lambda trial, i: {'checks_pass': trial != 'bad_guard'})
    assert selected == 'best' and score['combined_cost'] == 3.
    assert seen == ['better', 'best']
    assert 'cost' not in rows[0]


@pytest.mark.parametrize('cost', [5., 6., float('nan'), float('inf')])
def test_nonimprovements_and_nonfinite_alternatives_keep_rest(cost):
    selected, score, _ = repair.select_best_insertion('base', {'combined_cost': 5.}, [(21, 'trial', {})],
        lambda _: {'combined_cost': cost}, lambda *a: {'checks_pass': True})
    assert selected == 'base' and score['combined_cost'] == 5.


def setup_polish(monkeypatch, *, blocked=()):
    state_at, program = fixture()
    days, states = program(28, {18: ('easy', 4.), 25: ('easy', 4.)})
    template = days[25].recommendation
    def select(options, planned, request, config, **kwargs):
        at = options[0].as_of
        return options[0], template.model_copy(update={'planned_for': at, 'fitness_state_as_of': at})
    monkeypatch.setattr(planner, '_select_budgeted_timed_recommendation', select)
    monkeypatch.setattr(repair, 'finalize_continuation_distance', lambda result, *a: (result, {'native': True}))
    monkeypatch.setattr(repair, 'check_sequence', lambda *a: {'checks_pass': True, 'failed_offsets': []})
    inputs = {'daily_states': states[:21], 'config': CONFIG, 'forced_rest_offsets': blocked,
              'request': planner.RecommendationRequest(health_status='normal')}
    base_count = sum(d.recommendation is not None for d in days)
    def evaluate(trial):
        count = sum(d.recommendation is not None for d in trial[0])
        return {'combined_cost': 10. - (count - base_count),
                'sessions': [(d.date.isoformat(), planner._midpoint(d.recommendation))
                             for d in trial[0] if d.recommendation]}
    initial = (days, states, [])
    return initial, inputs, state_at, evaluate


def test_polish_only_adds_one_run_skips_blocks_and_preserves_existing_workouts(monkeypatch):
    initial, inputs, state_at, evaluate = setup_polish(monkeypatch, blocked=(21, 22))
    before = [d.model_dump_json() for d in initial[0]]
    selected, scored, metrics = repair.polish(initial, evaluate(initial), inputs=inputs,
        targets=[(17., 18.)] * 28, hard_length=21, state_at=state_at, evaluate=evaluate)
    assert metrics['selected_offset'] == 23
    assert [row['offset'] for row in metrics['candidates']] == [23, 24, 26, 27]
    assert all(selected[0][i] == initial[0][i] for i in range(28) if i != 23)
    assert [d.model_dump_json() for d in initial[0]] == before
    assert selected[2][0]['source'] == 'bounded_insertion'
    assert scored['combined_cost'] == 9.


def test_downstream_failure_rejects_an_otherwise_lower_score(monkeypatch):
    initial, inputs, state_at, evaluate = setup_polish(monkeypatch)
    def check(days, inputs, state_at, edits):
        return {'checks_pass': not edits, 'failed_offsets': [25] if edits else []}
    monkeypatch.setattr(repair, 'check_sequence', check)
    selected, _, metrics = repair.polish(initial, evaluate(initial), inputs=inputs,
        targets=[(17., 18.)] * 28, hard_length=21, state_at=state_at, evaluate=evaluate)
    assert selected == initial
    assert metrics['selected_offset'] is None
    assert all(row['checks']['failed_offsets'] == [25] for row in metrics['candidates'])


def test_baseline_guard_failure_is_reported_not_ignored(monkeypatch):
    initial, inputs, state_at, evaluate = setup_polish(monkeypatch)
    monkeypatch.setattr(repair, 'check_sequence', lambda *a: {'checks_pass': False, 'failed_offsets': [18]})
    selected, _, metrics = repair.polish(initial, evaluate(initial), inputs=inputs,
        targets=[(17., 18.)] * 28, hard_length=21, state_at=state_at, evaluate=evaluate)
    assert selected == initial and metrics['status'] == 'baseline_guard_failure'
    assert metrics['candidates'] == []


def test_unexpected_error_restores_native_allocator_hooks(monkeypatch):
    initial, inputs, state_at, evaluate = setup_polish(monkeypatch)
    original = planner._easy_allocation_bounds
    def fail(trial):
        assert planner._easy_allocation_bounds is not original
        raise RuntimeError('unexpected score error')
    with pytest.raises(RuntimeError, match='unexpected score'):
        repair.polish(initial, evaluate(initial), inputs=inputs, targets=[(17., 18.)] * 28,
                      hard_length=21, state_at=state_at, evaluate=fail)
    assert planner._easy_allocation_bounds is original


def test_repair_cannot_select_with_partial_or_unfinalized_objective():
    with pytest.raises(ValueError, match='coordinated finalized'):
        compare_candidate({}, [], {}, state_at=None, continuation_days=28, repair_opportunities=True)


def test_generation_cannot_shift_into_another_day(monkeypatch):
    from datetime import timedelta
    initial, inputs, state_at, evaluate = setup_polish(monkeypatch)
    def shifted(result, *args):
        return result.model_copy(update={'planned_for': result.planned_for + timedelta(days=1)}), {}
    monkeypatch.setattr(repair, 'finalize_continuation_distance', shifted)
    selected, _, metrics = repair.polish(initial, evaluate(initial), inputs=inputs,
        targets=[(17., 18.)] * 28, hard_length=21, state_at=state_at, evaluate=evaluate)
    assert selected == initial and metrics['selected_offset'] is None
    assert metrics['candidates'] == []
    assert all(row['reason'] == 'finalization_changed_date' for row in metrics['rejected'])


def test_explicit_rest_attempt_is_replaced_not_duplicated(monkeypatch):
    initial, inputs, state_at, evaluate = setup_polish(monkeypatch)
    at = initial[1][21].as_of.replace(hour=12)
    rest = {'at': at.isoformat(), 'requested_role': 'rest', 'selected_type': 'rest'}
    initial = (initial[0], initial[1], [rest])
    selected, _, metrics = repair.polish(initial, evaluate(initial), inputs=inputs,
        targets=[(17., 18.)] * 28, hard_length=21, state_at=state_at, evaluate=evaluate)
    assert metrics['selected_offset'] == 21
    assert len(selected[2]) == 1 and selected[2][0]['source'] == 'bounded_insertion'
    assert initial[2] == [rest]


def test_native_dose_menu_is_compared_using_complete_program_score(monkeypatch):
    initial, inputs, state_at, base_evaluate = setup_polish(monkeypatch)
    calls = []
    monkeypatch.setattr(repair, 'continuation_distance_intents', lambda *args: (3.3, 3.5, 4.))
    def finalize(result, state, config, target, intent):
        calls.append(intent)
        miles = 3.3 if intent == 3.3 else 4.
        return result.model_copy(update={'distance_target_miles': miles,
            'distance_range_miles': (miles - .2, miles + .3)}), {'intent_miles': intent}
    monkeypatch.setattr(repair, 'finalize_continuation_distance', finalize)
    def evaluate(trial):
        score = base_evaluate(trial)
        if any(d.recommendation and planner._midpoint(d.recommendation) == 3.3 for d in trial[0]):
            score['combined_cost'] -= 2.
        return score
    selected, scored, metrics = repair.polish(initial, evaluate(initial), inputs=inputs,
        targets=[(17., 18.)] * 28, hard_length=21, state_at=state_at, evaluate=evaluate,
        compare_doses=True)
    assert calls and 3.3 in calls
    assert metrics['selected_offset'] == 21 and metrics['compare_doses']
    assert scored['combined_cost'] == 7.
    assert planner._midpoint(selected[0][21].recommendation) == 3.3
    # The duplicate four-mile results from several intentions are checked once.
    assert len(metrics['candidates']) == 2 * 6
    assert selected[0][:21] == initial[0][:21]


def test_history_menu_comparison_does_not_create_a_lower_floor():
    from scripts.continuation_distance import continuation_distance_intents
    from scripts.audit_easy_dose_program import history_profile_from_state
    from scripts.easy_dose_prototype import generate_intents
    _, program = fixture()
    _, states = program(1, {})
    opening = states[0]
    upper = opening.typical_easy_run_miles + .4
    profile = history_profile_from_state(opening, CONFIG)
    floor = planner._established_easy_midpoint_floor(opening)
    existing = generate_intents(profile, useful_floor_miles=floor,
        progression_allowance_miles=upper - profile.ordinary_planning_miles,
        progression_source='history_trajectory:candidate_independent_program_target')
    actual = continuation_distance_intents(opening, upper, CONFIG)
    assert actual == tuple(option.midpoint_miles for option in existing.options)
    assert min(actual) == min(floor, opening.typical_easy_run_miles)


def test_dose_comparison_requires_repair():
    with pytest.raises(ValueError, match='requires opportunity repair'):
        compare_candidate({}, [], {}, state_at=None, continuation_days=28, compare_opportunity_doses=True)
