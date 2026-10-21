from types import SimpleNamespace

import pytest

from run_analysis import weekly_schedule as planner
from run_analysis.selection_objective import (
    SHARED_SELECTION_OBJECTIVE, PRIOR_DISTANCE_PRESCRIPTIONS,
    shared_objective_policy, scope_selection_prior,
)
from run_analysis.prescription_distance import canonical_distance_policy
from scripts.finalized_scoring_trial import finalized_scoring_policy
from test_finalized_scoring_trial import program
from test_recommendation import CONFIG
from scripts.replay_synthetic_easy_dose import shared_objective_failure


def fixture():
    states, days = program()
    prior_days = [d.model_copy(update={'recommendation': d.recommendation.model_copy(update={
        'distance_target_miles': 3.5, 'distance_range_miles': (3.2, 3.8),
    })}) if d.recommendation else d for d in days]
    candidates = [d.model_copy(update={'recommendation': d.recommendation.model_copy(update={
        'distance_target_miles': 3.8, 'distance_range_miles': (3.5, 4.),
    })}) if d.recommendation else d for d in days]
    return states, candidates, SimpleNamespace(planning_days=prior_days, days=[])


def test_existing_formula_prices_targets_not_range_midpoints():
    _, candidates, schedule = fixture()
    priors = {d.date:d.recommendation for d in schedule.planning_days if d.recommendation}
    expected = sum((planner.NEAR_TERM_STABILITY_DECISION_OFFSETS-i)
        / planner.NEAR_TERM_STABILITY_DECISION_OFFSETS
        * ((.3-planner.NEAR_TERM_DISTANCE_STABILITY_TOLERANCE_MILES)
           / planner.NEAR_TERM_DISTANCE_STABILITY_TOLERANCE_MILES)**2
        * planner.NEAR_TERM_STABILITY_ABSOLUTE_CAP for i in (0,2))
    assert planner._distance_stability_cost(candidates, priors) == pytest.approx(expected)
    assert expected > 0  # Range midpoint change alone is only .25 and would be free.
    assert planner._distance_stability_cost(candidates, {}) == 0
    assert planner._distance_stability_cost(schedule.planning_days, priors) == 0


def test_type_change_and_later_dates_keep_existing_exemptions():
    _, candidates, schedule = fixture()
    priors = {d.date:d.recommendation for d in schedule.planning_days if d.recommendation}
    changed = candidates[0].model_copy(update={'recommendation':
        candidates[0].recommendation.model_copy(update={'workout_type':'long'})})
    assert planner._distance_stability_cost([changed], priors) == 0
    late = [candidates[1]] * planner.NEAR_TERM_STABILITY_DECISION_OFFSETS + [candidates[0]]
    assert planner._distance_stability_cost(late, priors) == 0


def test_build_scopes_positional_prior_and_restores_nested_context_after_failure():
    _, _, schedule = fixture()
    @scope_selection_prior
    def build(prior_schedule=None, fail=False):
        assert len(PRIOR_DISTANCE_PRESCRIPTIONS.get()) == (2 if prior_schedule else 0)
        if fail:
            raise RuntimeError('stop')
    with shared_objective_policy(prior_schedule=schedule):
        prior = PRIOR_DISTANCE_PRESCRIPTIONS.get()
        build(None)
        with pytest.raises(RuntimeError):
            build(schedule, True)
        assert PRIOR_DISTANCE_PRESCRIPTIONS.get() is prior
        with shared_objective_policy(False):
            assert not SHARED_SELECTION_OBJECTIVE.get()
        assert SHARED_SELECTION_OBJECTIVE.get()
    assert not SHARED_SELECTION_OBJECTIVE.get()
    assert PRIOR_DISTANCE_PRESCRIPTIONS.get() is None


def test_joint_score_reports_switching_once_separately_from_training_and_recovery():
    states, candidates, schedule = fixture()
    sessions = [planner._CandidateSession(i, states[i], d.recommendation)
                for i,d in enumerate(candidates) if d.recommendation]
    def score():
        details = {}
        joint = planner._joint_candidate_program_cost((0,2), states, [[s] for s in states],
            planner.RecommendationRequest(health_status='normal'), CONFIG, (15,17),
            _allocated_override=candidates, _sessions_override=sessions, diagnostic_details=details)
        return joint, details
    with canonical_distance_policy(), finalized_scoring_policy():
        legacy, legacy_details = score()
        with shared_objective_policy(prior_schedule=schedule):
            joint, details = score()
    assert joint == legacy
    assert details['finalized_coaching_cost'] == legacy_details['finalized_coaching_cost']
    assert details['recovery_cost_components'] == legacy_details['recovery_cost_components']
    cost = details['distance_switching_cost']
    assert cost > 0
    base = planner._program_selection_cost(joint, details['finalized_coaching_cost'], 3.75)
    breakdown = planner._score_breakdown((0,2), joint, details['finalized_coaching_cost'], 3.75, details)
    assert breakdown.total_cost == pytest.approx(base + cost)
    assert breakdown.distance_switching_cost == cost
    assert 'distance_switching_cost' not in legacy_details


def test_shared_build_requires_canonical_and_finalized_prerequisites():
    states, _, _ = fixture()
    with shared_objective_policy(), pytest.raises(ValueError, match='requires canonical'):
        planner.build_weekly_schedule(states, planner.RecommendationRequest(health_status='normal'), CONFIG)


@pytest.mark.parametrize('change', ['none', 'cost', 'dose', 'missing'])
def test_replay_checks_actual_search_publication_parity(change):
    states, _, _ = fixture()
    search = dict(total_cost=1., allocation_trace=['0:easy_run:easy:3.800'])
    published = dict(search)
    if change == 'cost':
        published['total_cost'] += .01
    if change == 'dose':
        published['allocation_trace'] = ['0:easy_run:easy:4.000']
    if change == 'missing':
        published = None
    failure = shared_objective_failure(SimpleNamespace(generated_at=states[0].as_of,
        planner_diagnostics=dict(winning_candidate=search, published_candidate=published)))
    assert (failure is None) == (change == 'none')
