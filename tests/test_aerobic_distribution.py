from datetime import date, timedelta
from types import SimpleNamespace

import pytest

from scripts.aerobic_distribution import distributed_support, bounded_support, continuation_support
from scripts.typed_continuation_prototype import compare_candidate
from run_analysis import weekly_schedule as planner
from test_typed_continuation import fixture


START = date(2026, 9, 22)


def assess(offsets, *, n=49, start=0, window=14, ordinary=4.):
    return distributed_support(
        [START + timedelta(days=i) for i in range(start, start + n)],
        [(START + timedelta(days=i), miles) for i, miles in offsets],
        ordinary_miles=ordinary, window_days=window)


def credit(offsets, *, n=49, start=0, ordinary=4.):
    return bounded_support([START + timedelta(days=i) for i in range(start, start + n)],
        [(START + timedelta(days=i), miles) for i, miles in offsets], ordinary_miles=ordinary)


def test_bounded_credit_does_not_regrant_allowance_every_two_weeks():
    sessions = [(i, 4.) for i in range(-7, 49, 14)]
    assert assess(sessions)['violation'] == 0
    assert credit(sessions)['violation'] > 0
    assert credit([(i, 4.) for i in range(-7, 49, 7)])['violation'] == pytest.approx(0.)


def test_bounded_credit_closes_front_loading_without_extra_weight():
    spread = credit([(-5, 4.), *[(i, 4.) for i in (2, 9, 16, 23, 30, 37)]])
    cluster = credit([(-5, 4.), *[(i, 4.) for i in (2, 5, 8, 12, 18, 44)]])
    assert spread['violation'] < cluster['violation']


def test_bounded_credit_rollover_and_split_are_additive_not_reset():
    sessions = [(-5, 4.), (2, 4.), (18, 2.)]
    whole = credit(sessions)
    prefix = credit(sessions, n=21)
    tail = credit(sessions, start=21, n=28)
    shifted = credit(sessions, start=1, n=48)
    assert whole['daily'] == prefix['daily'] + tail['daily']
    assert whole['daily'][1:] == shifted['daily']
    assert whole['violation'] == pytest.approx(prefix['violation'] + tail['violation'])


def test_bounded_short_run_credit_is_proportional_and_file_split_invariant():
    empty = credit([], n=7)['violation']
    half = credit([(0, 2.)], n=7)['violation']
    full = credit([(0, 4.)], n=7)['violation']
    assert half == pytest.approx((empty + full) / 2)
    assert credit([(0, 4.)]) == credit([(0, 2.), (0, 2.)]) == credit([(0, 8.)])
    assert credit([(0, .1)])['violation'] > credit([(0, 2.)])['violation']


def test_bounded_missing_work_is_not_a_debt_or_future_credit():
    before = credit([(20, 4.)])
    more_history = credit([(-1, 4.), (20, 4.)])
    assert before['daily'][20:] == more_history['daily'][20:]
    assert credit([])['daily'][:20] == before['daily'][:20]
    assert credit([(-20, 100.)]) == credit([])


def test_bounded_athlete_scaling_and_saturation():
    samples = [(-3, 4.), (2, 2.), (10, 4.)]
    assert credit(samples) == {**credit([(i, m / 2) for i, m in samples], ordinary=2.),
                               'ordinary_reference_miles': 4.}
    assert all(0 <= row['remaining_equivalents'] <= 1 for row in credit(samples)['daily'])


def test_bounded_history_cannot_reset_a_balance_maintained_by_small_runs():
    sessions = [(-20, 4.), *[(i, 4 / 7) for i in range(-19, 49)]]
    whole = credit(sessions)
    shifted = credit(sessions, start=1, n=48)
    assert whole['daily'][0]['remaining_equivalents'] == pytest.approx(6 / 7)
    assert whole['daily'][1:] == shifted['daily']
    assert whole['violation'] == 0


def test_equal_totals_no_longer_hide_front_loaded_support():
    # Same six ordinary runs: both satisfy the old 49/7 - 1 aggregate floor.
    distributed = assess([(-5, 4.), *[(i, 4.) for i in (2, 9, 16, 23, 30, 37)]])
    clustered = assess([(-5, 4.), *[(i, 4.) for i in (2, 5, 8, 12, 18, 44)]])
    assert distributed['violation'] == 0
    assert clustered['violation'] > 0
    assert any(row['missing_equivalents'] > 0 for row in clustered['daily'][21:])


@pytest.mark.parametrize('window', [12, 14, 16])
def test_locality_result_does_not_require_one_exact_window(window):
    spread = assess([(-5, 4.), *[(i, 4.) for i in (2, 9, 16, 23, 30, 37)]], window=window)
    clustered = assess([(-5, 4.), *[(i, 4.) for i in (2, 5, 8, 12, 18, 44)]], window=window)
    assert spread['violation'] < clustered['violation']


def test_short_easy_has_proportional_credit_not_a_whole_reset():
    empty = assess([], n=14)['violation']
    half = assess([(0, 2.)], n=14)['violation']
    full = assess([(0, 4.)], n=14)['violation']
    assert empty > half > full
    assert half == pytest.approx((empty + full) / 2)


def test_same_day_splitting_cannot_manufacture_support():
    assert assess([(0, 4.)]) == assess([(0, 2.), (0, 2.)])
    assert assess([(0, 8.)]) == assess([(0, 4.)])
    assert assess([(0, .1)])['violation'] > assess([(0, 4.)])['violation']


def test_old_credit_cannot_be_banked_or_future_credit_borrowed():
    base = assess([(0, 4.)])
    added_early = assess([(0, 4.), (1, 4.), (2, 4.)])
    assert base['daily'][16:] == added_early['daily'][16:]
    added_late = assess([(0, 4.), (30, 4.)])
    assert base['daily'][:30] == added_late['daily'][:30]
    assert assess([(-50, 100.)]) == assess([])


def test_history_contributes_then_expires_without_resetting_at_origin():
    no_history = assess([], n=14)
    history = assess([(-1, 4.)], n=14)
    assert history['daily'][0]['missing_equivalents'] == 0
    assert history['daily'][-1]['missing_equivalents'] == 1
    assert history['violation'] < no_history['violation']


def test_rollover_is_suffix_of_same_physical_accounting():
    sessions = [(-5, 4.), (2, 4.), (18, 2.), (35, 4.)]
    whole = assess(sessions)
    shifted = assess(sessions, start=1, n=48)
    assert whole['daily'][1:] == shifted['daily']
    assert whole['violation'] == pytest.approx(
        shifted['violation'] + whole['daily'][0]['missing_equivalents'] / 7)


def test_no_hard_soft_boundary_or_horizon_length_reset():
    sessions = [(-5, 4.), (2, 4.), (18, 2.)]
    whole = assess(sessions, n=49)
    prefix = assess(sessions, n=21)
    tail = assess(sessions, start=21, n=28)
    assert whole['daily'] == prefix['daily'] + tail['daily']
    assert whole['violation'] == pytest.approx(prefix['violation'] + tail['violation'])


def test_athlete_scaling_and_sparse_running_are_not_hard_failures():
    sessions = [(-5, 4.), *[(i, 4.) for i in range(5, 49, 10)]]
    assert assess(sessions)['violation'] == 0
    assert assess([(i, miles / 2) for i, miles in sessions], ordinary=2.) == {
        **assess(sessions), 'ordinary_reference_miles': 2.}


@pytest.mark.parametrize('kwargs', [{'window':7}, {'ordinary':0}, {'ordinary':float('nan')}])
def test_invalid_policy_is_not_silently_scored(kwargs):
    with pytest.raises(ValueError):
        assess([], **kwargs)


def test_local_mode_requires_explicit_history_and_full_role_accounting():
    with pytest.raises(ValueError, match='explicit completed history'):
        compare_candidate({}, [], {}, state_at=None, continuation_days=28,
                          support_policy='local', complete_role_costs=True)


@pytest.mark.parametrize('projected,future', [(True, False), (False, True)])
def test_observed_history_cannot_contain_projected_or_future_runs(projected, future):
    _, program = fixture()
    days, states = program(21, {})
    run = SimpleNamespace(projected=projected,
        start_time=states[0].as_of + timedelta(days=1 if future else -1))
    with pytest.raises(ValueError, match='precede the decision'):
        continuation_support(days, states, [run])


def test_adapter_preserves_ordinary_reference_and_ignores_easy_role_labels():
    _, program = fixture()
    days, states = program(21, {2: ('easy', 2.), 10: ('easy', 4.)})
    before = [state.model_dump_json() for state in states]
    first = continuation_support(days, states, [])
    renamed = [day.model_copy(update={'recommendation': day.recommendation.model_copy(
        update={'planning_role': 'support_easy'})}) if day.recommendation else day for day in days]
    assert continuation_support(renamed, states, []) == first
    assert [state.model_dump_json() for state in states] == before
    assert first['ordinary_reference_miles'] == planner.typical_easy_distance(states[0])[0]


def test_score_replaces_aggregate_once_and_preserves_other_role_terms(monkeypatch):
    from scripts import typed_continuation_prototype as prototype
    from test_typed_continuation import CONFIG

    state_at, program = fixture()
    combined, states = program(49, {2: ('easy', 2.), 18: ('easy', 4.)})
    prefix = combined[:21]
    aggregate = planner._rolling_aerobic_support_violation(combined, states)
    score = {'total_cost': 100., 'coaching_cost': 3., 'finalized_recovery_cost': 4.,
             'support_violation': 2., 'shape_violation': .5, 'target_violation': 1.}
    inputs = {'daily_states': states[:21], 'config': CONFIG,
              'request': planner.RecommendationRequest(health_status='normal'),
              'target_distance_range': (17., 18.),
              'expected_target_projector': lambda runs, n: [(17., 18.)] * n}
    monkeypatch.setattr(prototype, 'continued_recovery', lambda *a, **kw: (4., {}))
    monkeypatch.setattr(planner, '_finalized_program_coaching_cost',
                        lambda *a, **kw: (3., [SimpleNamespace(offset=2)]))
    monkeypatch.setattr(planner, '_joint_candidate_program_cost',
                        lambda *a, **kw: (0., aggregate + .25, .75, 0.))
    def evaluate(policy):
        return prototype.score_continuation(inputs, prefix, score, [(17., 18.)] * 49,
            (combined, states, []), state_at=state_at, continuation_days=28,
            distance_policy='stock', complete_role_costs=True,
            support_policy=policy, completed_runs=[])
    old, revised = evaluate('aggregate'), evaluate('bounded')
    local = revised['local_support']['violation']
    assert revised['combined_support_violation'] == pytest.approx(local + .25)
    assert revised['combined_shape_violation'] == old['combined_shape_violation'] == .75
    assert revised['combined_cost'] - old['combined_cost'] == pytest.approx(
        planner.PROGRAM_FIT_UNIT * planner.AEROBIC_SUPPORT_FIT_FRACTION * (local - aggregate))
    for field in ('policy_rate', 'trajectory_recovery_cost', 'combined_coaching_cost', 'sessions'):
        assert revised[field] == old[field]
