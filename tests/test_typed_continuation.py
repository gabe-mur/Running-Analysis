from datetime import datetime, timedelta, timezone
from functools import lru_cache

import pytest

from run_analysis import adherence_projection as projection
from run_analysis import weekly_schedule as planner
from run_analysis.prescription_distance import canonical_distance_policy
from scripts.replay_synthetic_easy_dose import synthetic_history
from scripts.typed_continuation_prototype import continued_recovery, rollout, runs
from test_recommendation import CONFIG


def fixture():
    start = datetime(2026, 9, 22, tzinfo=timezone.utc)
    template, history = synthetic_history(start)

    @lru_cache(maxsize=None)
    def state_at(at):
        return projection._state_at(template, history, at, 15.75)

    def program(length, sessions):
        states = [state_at(start+timedelta(days=i)) for i in range(length)]
        days = []
        for i, state in enumerate(states):
            result = None
            if i in sessions:
                kind, miles = sessions[i]
                at = state.as_of+timedelta(hours=12)
                result = planner.RecommendationResponse(generated_at=start,
                    fitness_state_as_of=at, planned_for=at, workout_type=kind,
                    title='Boundary fixture', distance_range_miles=(miles, miles),
                    distance_target_miles=miles, confidence='moderate', readiness='ready')
            days.append(planner.WeeklyScheduleDay(date=state.as_of.date(),
                planned_at=result.planned_for if result else None, recommendation=result,
                day_role='easy_run' if result else 'rest_day', rationale='fixture'))
        return days, states
    return state_at, program


def test_terminal_long_load_is_seen_by_actual_following_session():
    state_at, program = fixture()
    with canonical_distance_policy():
        prefix = {14: ('long', 8.), 16: ('easy', 4.), 18: ('easy', 4.)}
        ordinary, states = program(28, {**prefix, 20: ('easy', 4.), 22: ('easy', 4.)})
        long, _ = program(28, {**prefix, 20: ('long', 8.), 22: ('easy', 4.)})
        ordinary_cost, ordinary_parts = continued_recovery(ordinary, states, CONFIG, (17.,18.), state_at=state_at)
        long_cost, long_parts = continued_recovery(long, states, CONFIG, (17.,18.), state_at=state_at)
        _, hard_parts = continued_recovery(long[:21], states[:21], CONFIG, (17.,18.), state_at=state_at)
    assert long_cost > ordinary_cost
    assert long_parts['short_term_bridge'] > ordinary_parts['short_term_bridge']
    assert long_parts['short_term_bridge'] > hard_parts['short_term_bridge']


def test_isolated_recovered_long_does_not_receive_blanket_penalty():
    state_at, program = fixture()
    with canonical_distance_policy():
        days, states = program(28, {20: ('long', 8.), 22: ('easy', 4.)})
        cost, _ = continued_recovery(days, states, CONFIG, (17.,18.), state_at=state_at)
    assert cost == 0


def test_terminal_compression_is_not_erased_at_hard_edge():
    state_at, program = fixture()
    with canonical_distance_policy():
        spread, states = program(35, {14: ('easy', 4.), 17: ('easy', 4.), 20: ('long', 8.), 23: ('easy', 4.)})
        compressed, _ = program(35, {18: ('easy', 4.), 19: ('easy', 4.), 20: ('long', 8.), 23: ('easy', 4.)})
        spread_cost, _ = continued_recovery(spread, states, CONFIG, (17.,18.), state_at=state_at)
        compressed_cost, _ = continued_recovery(compressed, states, CONFIG, (17.,18.), state_at=state_at)
    assert compressed_cost > spread_cost


@pytest.mark.parametrize('growth', [0., .04])
@pytest.mark.parametrize('distance_policy', ['stock', 'finalized'])
def test_promoting_same_continuation_to_hard_prefix_preserves_physical_sequence(growth, distance_policy):
    state_at, program = fixture()
    days, states = program(21, {14: ('long', 8.), 17: ('easy', 4.), 19: ('easy', 4.)})
    before = [d.model_dump_json() for d in days]
    targets = [(17.+i*growth,18.+i*growth) for i in range(49)]
    request = planner.RecommendationRequest(health_status='normal')
    with canonical_distance_policy():
        full, full_states, _ = rollout(days, states, request, CONFIG, targets,
                                      state_at=state_at, continuation_days=28,
                                      distance_policy=distance_policy)
        # Include the first actual continuation run in the promoted prefix.
        first = next(i for i in range(21,49) if full[i].recommendation)
        cut = first+1
        promoted, promoted_states, _ = rollout(full[:cut], full_states[:cut], request, CONFIG,
            targets, state_at=state_at, continuation_days=49-cut, distance_policy=distance_policy)
        assert [r.model_dump(exclude={'generated_at'}) for r in runs(promoted)] == \
            [r.model_dump(exclude={'generated_at'}) for r in runs(full)]
        assert continued_recovery(full, full_states, CONFIG, (17.,18.), state_at=state_at) == \
            continued_recovery(promoted, promoted_states, CONFIG, (17.,18.), state_at=state_at)
        assert continued_recovery(full, full_states, CONFIG, (17.,18.), state_at=state_at,
                                  targets=targets) == \
            continued_recovery(promoted, promoted_states, CONFIG, (17.,18.), state_at=state_at,
                               targets=targets)
    assert [d.model_dump_json() for d in days] == before
    assert any(r.workout_type == planner.WorkoutType.LONG for r in runs(full[21:]))
    assert any(r.workout_type in planner.QUALITY_WORKOUT_TYPES for r in runs(full[21:]))
    assert all(r.distance_target_miles*10 == pytest.approx(round(r.distance_target_miles*10))
               for r in runs(full[21:]))


@pytest.mark.parametrize('length', [0, -1, 1.5])
def test_reject_invalid_continuation_length(length):
    state_at, program = fixture()
    days, states = program(21, {20: ('easy', 4.)})
    with pytest.raises(ValueError, match='positive integer'):
        rollout(days, states, planner.RecommendationRequest(health_status='normal'),
            CONFIG, [(17.,18.)]*49, state_at=state_at, continuation_days=length)


def test_target_coverage_is_not_silently_extrapolated():
    state_at, program = fixture()
    days, states = program(21, {20: ('easy', 4.)})
    with pytest.raises(ValueError, match='trajectory'):
        rollout(days, states, planner.RecommendationRequest(health_status='normal'),
            CONFIG, [(17.,18.)]*21, state_at=state_at, continuation_days=28)


def test_flat_recovery_trajectory_exactly_preserves_legacy_score():
    state_at, program = fixture()
    days, states = program(35, {i: ('long' if i % 8 == 0 else 'easy', 8. if i % 8 == 0 else 4.)
                               for i in range(0,35,2)})
    with canonical_distance_policy():
        old = continued_recovery(days, states, CONFIG, (17.,18.), state_at=state_at)
        explicit = continued_recovery(days, states, CONFIG, (17.,18.), state_at=state_at,
                                      targets=[(17.,18.)]*35)
    assert explicit == old


def test_future_target_cannot_change_earlier_recovery_charges():
    state_at, program = fixture()
    days, states = program(35, {16: ('long',8.), 18: ('easy',4.), 20: ('long',8.), 22: ('easy',4.)})
    with canonical_distance_policy():
        _, before = continued_recovery(days, states, CONFIG, (17.,18.), state_at=state_at,
                                       targets=[(17.,18.)]*35)
        _, after = continued_recovery(days, states, CONFIG, (17.,18.), state_at=state_at,
            targets=[(17.,18.)]*23+[(20.,21.)]*12)
    assert {k:v for k,v in before.items() if k != 'soft_continuation'} == \
        {k:v for k,v in after.items() if k != 'soft_continuation'}


@pytest.mark.parametrize('targets', [[(17.,18.)], [(18.,17.)]*21, [(0.,18.)]*21, [(17.,float('nan'))]*21])
def test_invalid_recovery_trajectory_rejected(targets):
    state_at, program = fixture()
    days, states = program(21, {20: ('long',8.)})
    with pytest.raises(ValueError, match='trajectory'):
        continued_recovery(days, states, CONFIG, (17.,18.), state_at=state_at, targets=targets)


def test_finalized_policy_reuses_allocator_and_restores_hooks_on_error(monkeypatch):
    from scripts.continuation_distance import continuation_distance_policy, finalize_continuation_distance
    state_at, program = fixture()
    days, states = program(21, {20: ('easy',4.)})
    before = planner._easy_allocation_bounds
    seen = []
    def fail(*args, **kwargs):
        seen.append((args, kwargs))
        raise RuntimeError('unexpected allocation failure')
    monkeypatch.setattr(planner, '_allocate_visible_distance_ranges', fail)
    with pytest.raises(RuntimeError, match='unexpected'):
        with continuation_distance_policy(states[0], [d.date for d in days], [(17.,18.)]*21, CONFIG):
            assert planner._easy_allocation_bounds is not before
            result = days[-1].recommendation
            finalize_continuation_distance(result, state_at(result.planned_for), CONFIG, (17.,18.),4.25)
    assert planner._easy_allocation_bounds is before
    assert len(seen) == 1
    assert seen[0][1]['preserve_projected_recovery_caps']


def test_finalized_expansion_cannot_bypass_upper_envelope_recovery(monkeypatch):
    from scripts.continuation_distance import finalize_continuation_distance
    state_at, program = fixture()
    days, states = program(21, {20: ('easy',4.)})
    original = days[-1].recommendation
    def oversized(days, *args, **kwargs):
        return [days[0].model_copy(update={'recommendation': original.model_copy(update={
            'distance_range_miles': (20.,20.5), 'distance_target_miles':20.})})]
    monkeypatch.setattr(planner, '_allocate_visible_distance_ranges', oversized)
    result, facts = finalize_continuation_distance(original, state_at(original.planned_for), CONFIG,(17.,18.),4.25)
    assert result.distance_range_miles == original.distance_range_miles
    assert result.distance_target_miles <= planner._midpoint(original)
    assert facts['guarded_dose_fallback']
    assert facts['upper_envelope_load'] > facts['allowed_load']+.05


def test_future_coaching_prices_postponed_key_session():
    state_at, program = fixture()
    _, states = program(35, {})
    request = planner.RecommendationRequest(health_status='normal')
    def cost(long_offset):
        days, _ = program(35, {20: ('long',8.), long_offset: ('long',8.)})
        options = [[state_at(d.recommendation.planned_for)] if d.recommendation else [s]
                   for d,s in zip(days, states)]
        return planner._finalized_program_coaching_cost(days, states, options, request, CONFIG,(17.,18.))[0]
    assert cost(30) > cost(28)


def test_unrepresentable_singleton_fallback_never_rounds_above_guarded_center(monkeypatch):
    from scripts.continuation_distance import finalize_continuation_distance
    from run_analysis.prescription_distance import UnrepresentableDistanceTarget
    state_at, program = fixture()
    days, _ = program(21, {20: ('easy',4.)})
    original = days[-1].recommendation.model_copy(update={
        'distance_range_miles': (3.,3.5), 'distance_target_miles': None})
    def reject(*args, **kwargs):
        raise UnrepresentableDistanceTarget('fixture narrow bounds')
    monkeypatch.setattr(planner, '_allocate_visible_distance_ranges', reject)
    result, facts = finalize_continuation_distance(original, state_at(original.planned_for), CONFIG,(17.,18.),4.25)
    assert result.distance_target_miles == 3.2
    assert result.distance_range_miles == original.distance_range_miles
    assert facts['guarded_dose_fallback']


def test_continuation_and_detailed_policy_share_exact_easy_intent_bounds():
    from scripts.audit_easy_dose_program import history_profile_from_state, dose_trajectory, history_allocator
    from scripts.continuation_distance import continuation_distance_policy
    state_at, program = fixture()
    days, states = program(21, {20: ('easy',4.)})
    profile = history_profile_from_state(states[0], CONFIG)
    dates = [d.date for d in days]
    targets = [(17.,18.)]*21
    trajectory = dose_trajectory(profile, dates, targets, states[0].recent_load.capacity_reference_miles,
                                 mode='history_trajectory', fixed_fraction=0.)
    result = days[-1].recommendation
    state = state_at(result.planned_for)
    def bounds():
        return planner._easy_allocation_bounds(state, result, 6., 2, preserve_projected_recovery_caps=True)
    with history_allocator(profile, trajectory, planner._established_easy_midpoint_floor(states[0])):
        detailed = bounds()
    with continuation_distance_policy(states[0], dates, targets, CONFIG) as intentions:
        assert bounds() == detailed
        assert intentions[days[-1].date] == trajectory.upper_miles[-1]
