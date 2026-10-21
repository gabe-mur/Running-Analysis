from datetime import datetime, timedelta, timezone

import pytest

from scripts.coordinated_continuation import coordinate
from scripts.typed_continuation_prototype import rollout, compare_candidate
from run_analysis.prescription_distance import UnrepresentableDistanceTarget
from run_analysis import weekly_schedule as planner
from test_typed_continuation import fixture, CONFIG


def toy():
    dates = [datetime(2026,10,19,tzinfo=timezone.utc)+timedelta(days=2*i) for i in range(3)]
    calls = []
    def make(overrides):
        calls.append(dict(overrides))
        base = ['quality','long','easy'] if overrides.get(dates[0].date()) == 'quality' else ['easy','quality','long']
        sequence = [overrides.get(at.date(), default) for at,default in zip(dates,base)]
        attempts = [{'at':at.isoformat(),'requested_role':role} for at,role in zip(dates,sequence)]
        return sequence, [], attempts
    def evaluate(candidate):
        seq = candidate[0]
        # The better coordinated sequence is valued as a whole, not three
        # independently minimized per-session timing errors.
        cost = 0. if seq == ['quality','long','easy'] else 120. if seq == ['easy','quality','long'] else 150.
        return {'combined_cost':cost}
    return make, evaluate, dates, calls


def test_coordinated_sequence_beats_greedy_role_and_commits_prefix():
    make, evaluate, dates, calls = toy()
    selected, scored, audit = coordinate(make,evaluate)
    assert selected[0] == ['quality','long','easy']
    assert scored['combined_cost'] == 0
    assert audit['initial_cost'] == 120
    assert audit['evaluations'] <= 1+3*audit['opportunities']
    assert all(d['after_cost'] <= d['before_cost'] for d in audit['decisions'])
    # Later opportunities cannot rewrite the already selected first role.
    assert all(c[dates[0].date()] == 'quality' for c in calls if dates[1].date() in c)


def test_ties_keep_incumbent_without_random_choice():
    make, _, _, _ = toy()
    first, _, audit = coordinate(make,lambda _: {'combined_cost':1.})
    second, _, _ = coordinate(make,lambda _: {'combined_cost':1.})
    assert first == second
    assert first[0] == ['easy','quality','long']
    assert all(d['after_cost'] == d['before_cost'] for d in audit['decisions'])


def test_unrepresentable_option_is_rejected_not_used_to_relax_limits():
    make, evaluate, dates, _ = toy()
    def guarded(overrides):
        if overrides.get(dates[0].date()) == 'quality':
            raise UnrepresentableDistanceTarget('guarded bounds')
        return make(overrides)
    selected, _, audit = coordinate(guarded,evaluate)
    assert selected[0] == ['easy','quality','long']
    assert any(x.get('rejected') == 'guarded bounds' for x in audit['decisions'][0]['alternatives'])


def test_unexpected_error_is_not_swallowed():
    make, _, _, _ = toy()
    def fail(_):
        raise RuntimeError('diagnostic failure')
    with pytest.raises(RuntimeError,match='diagnostic failure'):
        coordinate(make,fail)


def test_nonfinite_initial_score_is_rejected():
    make, _, _, _ = toy()
    with pytest.raises(ValueError,match='finite'):
        coordinate(make,lambda _: {'combined_cost':float('nan')})


def test_nonfinite_alternative_cannot_replace_incumbent():
    make, _, _, _ = toy()
    selected, score, _ = coordinate(make, lambda candidate: {
        'combined_cost': 1. if candidate[0] == ['easy', 'quality', 'long'] else float('nan')})
    assert selected[0] == ['easy', 'quality', 'long']
    assert score['combined_cost'] == 1.


def test_role_search_cannot_silently_move_opportunities():
    make, evaluate, _, _ = toy()
    def shifted(overrides):
        candidate = make(overrides)
        if overrides:
            candidate[2][0]['at'] = '2026-10-20T00:00:00+00:00'
        return candidate
    with pytest.raises(ValueError, match='opportunity calendar'):
        coordinate(shifted, evaluate)


def test_cannot_optimize_the_incomplete_diagnostic_objective():
    state_at, program = fixture()
    days, states = program(21,{20:('long',8.)})
    inputs = {'expected_target_projector':lambda runs,n: [(17.,18.)]*n}
    with pytest.raises(ValueError,match='support/shape'):
        compare_candidate(inputs,days,{},state_at=state_at,continuation_days=28,timing_policy='coordinated')


def test_role_override_cannot_change_hard_prefix():
    state_at, program = fixture()
    days, states = program(21,{20:('long',8.)})
    with pytest.raises(ValueError,match='within the continuation'):
        rollout(days,states,planner.RecommendationRequest(health_status='normal'),CONFIG,
            [(17.,18.)]*49,state_at=state_at,continuation_days=28,
            role_overrides={days[-1].date:'rest'})


def test_conditional_roles_may_change_only_uncommitted_future_opportunities():
    first = datetime(2026, 10, 19, tzinfo=timezone.utc)
    def make(overrides):
        role = overrides.get(first.date(), 'easy')
        second = first + timedelta(days=3 if role == 'long' else 2)
        return [role], [], [dict(at=first.isoformat(), requested_role=role),
            dict(at=second.isoformat(), requested_role=overrides.get(second.date(), 'easy'))]
    def evaluate(candidate):
        return {'combined_cost': 0. if candidate[0][0] == 'long' else 10.}
    with pytest.raises(ValueError, match='opportunity calendar'):
        coordinate(make, evaluate)
    selected, scored, audit = coordinate(make, evaluate, causal_opportunities=True)
    assert scored['combined_cost'] == 0.
    assert selected[2][1]['at'] == (first + timedelta(days=3)).isoformat()
    assert audit['decisions'][1]['at'] == selected[2][1]['at']
    assert audit['evaluations'] <= 1 + 3 * audit['opportunities']
