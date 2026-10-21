from copy import deepcopy
from datetime import datetime, timedelta
from inspect import signature
from types import SimpleNamespace

import pytest

from run_analysis import weekly_schedule as planner
from run_analysis.web.schemas import ReadinessFlag
from scripts.calendar_neighbor_prototype import calendar_neighbors, calendar_neighbor_policy, opening_opportunity_neighbors, paired_move_neighbors
from scripts.opportunity_search_repair import opportunity_search_policy
from scripts.replay_recorded_hr import inputs_for
from scripts.audit_training_opportunities import available_history
from scripts.trace_opportunity_survival import TARGET, PrefixCaptured, trace_calendar_prefix
from test_opportunity_survival import sample_report
from test_finalized_scoring_trial import program
from test_recommendation import CONFIG


def test_repair_is_nested_and_restored_on_failure():
    assert not planner._OPPORTUNITY_SEARCH_REPAIR.get()
    with opportunity_search_policy():
        with pytest.raises(RuntimeError), opportunity_search_policy(False):
            assert not planner._OPPORTUNITY_SEARCH_REPAIR.get()
            raise RuntimeError("stop")
        assert planner._OPPORTUNITY_SEARCH_REPAIR.get()
    assert not planner._OPPORTUNITY_SEARCH_REPAIR.get()


def test_combined_opening_and_tail_counterexample_is_reachable_without_intermediate_win():
    seed = (1, 2, 4, 6, 9, 12, 13, 17, 20)
    assert TARGET not in calendar_neighbors(seed, range(21))
    assert TARGET in opening_opportunity_neighbors(seed, range(21))


def test_paired_dates_escape_proxy_selected_local_minimum():
    seed = (0, 2, 4, 6, 10, 12, 14, 17, 18, 20)
    assert TARGET not in calendar_neighbors(seed, range(21))
    assert TARGET in paired_move_neighbors(seed, range(21))
    candidates = paired_move_neighbors(seed, range(21))
    assert len(candidates) <= 190
    assert all(len(candidate) == len(seed) for candidate in candidates)
    assert all(9 not in candidate for candidate in paired_move_neighbors(seed, set(range(21)) - {9}))
    assert paired_move_neighbors((0,), range(21)) == []


def test_empty_time_option_remains_invalid():
    report = sample_report()
    at = datetime.fromisoformat(report["decisions"][3]["at"])
    inputs = inputs_for(sorted(available_history(report, at), key=lambda r: r.start_time), at, CONFIG, prior=None)
    inputs["daily_state_options"][0] = []
    with opportunity_search_policy(), pytest.raises(ValueError, match="at least one timing option"):
        planner._adaptive_run_day_offsets_for_frequency(inputs["daily_states"], inputs["request"], CONFIG,
            inputs["target_run_count"], inputs["target_distance_range"], daily_state_options=inputs["daily_state_options"])


@pytest.mark.parametrize("reject_better", [False, True])
def test_stalled_search_scores_paired_moves_and_keeps_acceptance_checks(monkeypatch, reject_better):
    seed, target = (0, 3, 6, 9), (0, 2, 5, 9)
    base = planner._score_breakdown(seed, (0, 0, 0, 10), 0, 3.5)
    def search(**inputs):
        diagnostics = inputs["_planner_diagnostics"]
        diagnostics.winning_candidate = diagnostics.unconstrained_winning_candidate = base
        return list(seed)
    search.__signature__ = signature(planner.adaptive_run_day_offsets)
    monkeypatch.setattr(planner, "adaptive_run_day_offsets", search)
    monkeypatch.setattr(planner, "typical_easy_distance", lambda state: (3, 4))
    def joint(offsets, **kwargs):
        kwargs["diagnostic_details"].update(finalized_coaching_cost=0,
            aerobic_acceptance_facts={"fixture_offsets": offsets})
        return (0, 0, 0, 1 if offsets == target else 10 if offsets == seed else 20)
    monkeypatch.setattr(planner, "_joint_candidate_program_cost", joint)
    # Verify the new neighborhood goes through the same acceptance adapter.
    # Exposure arithmetic itself is covered by shared-acceptance tests.
    import scripts.calendar_neighbor_prototype as neighbors
    def assess(*, fixture_offsets, planning_mode):
        return SimpleNamespace(rejected=reject_better and fixture_offsets == target,
            reason="fixture", evidence={})
    monkeypatch.setattr(neighbors, "assess_aerobic_exposure", assess)
    with opportunity_search_policy(), calendar_neighbor_policy(max_passes=1, enforce_acceptance=True) as records:
        selected = planner.adaptive_run_day_offsets(
            [SimpleNamespace(as_of=datetime(2026, 9, 22))] * 10,
            SimpleNamespace(health_status=planner.CurrentHealthStatus.NORMAL), {}, 4, (15, 18))
    assert selected == list(seed if reject_better else target)
    assert records[0]["paired_escape_used"]
    assert records[0]["paired_escape_evaluations"] > 0


@pytest.mark.parametrize("seed", [(1, 3, 8), (0,), (), (2, 4, 6, 9)])
def test_combined_neighbors_are_bounded_deterministic_and_respect_exclusions(seed):
    allowed = set(range(21)) - {5, 10}
    result = opening_opportunity_neighbors(seed, allowed)
    assert len(result) <= 2 * len(allowed)
    assert result == sorted(set(result))
    assert result == opening_opportunity_neighbors(reversed(seed), reversed(sorted(allowed)))
    for candidate in result:
        assert candidate and tuple(sorted(set(candidate))) == candidate
        assert set(candidate) <= allowed
        assert abs(len(seed) - len(candidate)) == 1
    assert opening_opportunity_neighbors((5,), allowed) == []


@pytest.mark.parametrize("hour", [0, 12, 20])
@pytest.mark.parametrize("available", [False, True])
def test_prefilter_uses_available_time_on_every_day_and_summary_only_as_fallback(hour, available):
    report = sample_report()
    at = datetime.fromisoformat(report["decisions"][3]["at"]).replace(hour=hour)
    inputs = inputs_for(sorted(available_history(report, at), key=lambda r: r.start_time), at, CONFIG, prior=None)
    states = inputs["daily_states"]
    # Distinguish raw-summary load from timed-option load without changing the
    # planning horizon or health rules. Reverse the options to test ordering.
    states[0] = states[0].model_copy(update={"recent_load": states[0].recent_load.model_copy(
        update={"acute_distance_to_capacity_ratio": 1.8})})
    timed = states[0].model_copy(update={"as_of": at + timedelta(hours=1),
        "recent_load": states[0].recent_load.model_copy(update={"acute_distance_to_capacity_ratio": .8})})
    inputs["daily_state_options"][0] = [timed.model_copy(update={"as_of": at + timedelta(hours=2)}), timed] if available else []
    before = deepcopy(states)
    with opportunity_search_policy(), trace_calendar_prefix(TARGET, stop_after_beam=True) as trace:
        with pytest.raises(PrefixCaptured):
            planner._adaptive_run_day_offsets_for_frequency(states, inputs["request"], CONFIG,
                inputs["target_run_count"], inputs["target_distance_range"],
                daily_state_options=inputs["daily_state_options"] if available else None, horizon_run_count=len(TARGET),
                joint_program_scoring=True)
    expected = timed if available else states[0]
    weather = planner.assess_training_weather(expected.planned_weather, expected.weather_exposure_baseline)
    ratio = planner.effective_load_ratio(expected.recent_load)
    cost = weather.score * 3 + (100 if weather.extreme else 0) + max(0, ratio - 1) * 12
    assert trace[0]["weather_and_raw_load_costs"][0] == pytest.approx(cost)
    assert states == before


@pytest.mark.parametrize("prefer_shorter", [False, True])
def test_dose_alternatives_reach_final_selector_with_fixed_budget_and_preserve_roles(monkeypatch, prefer_shorter):
    states, days = program()
    before = deepcopy(days)
    # Deliberately small proxy shortlist. All doses remain actual permitted
    # options. The final objective can prefer either direction, never hardcoded.
    monkeypatch.setattr(planner, "_easy_allocation_bounds", lambda *a, **k: (3.25, 4.25, 3.75, 1.75))
    evaluated = []
    def score(candidate):
        evaluated.append(candidate)
        miles = sum(planner._midpoint(d.recommendation) for d in candidate if d.recommendation)
        return miles if prefer_shorter else -miles
    with opportunity_search_policy():
        selected = planner._allocate_visible_distance_ranges(days, states, (8, 9), CONFIG,
            allocation_selector=score, allocation_candidate_limit=1)
    assert 1 < len(evaluated) <= 7  # one finalist + at most six anchor alternatives
    costs = [sum(planner._midpoint(d.recommendation) for d in c if d.recommendation) *
             (1 if prefer_shorter else -1) for c in evaluated]
    selected_cost = sum(planner._midpoint(d.recommendation) for d in selected if d.recommendation) * (1 if prefer_shorter else -1)
    assert selected_cost == min(costs)
    assert costs[-1] != costs[0]
    for candidate in evaluated:
        assert candidate[1].recommendation is None
        for index in (0, 2):
            result = candidate[index].recommendation
            assert result.workout_type == planner.WorkoutType.EASY
            assert result.planned_for == days[index].recommendation.planned_for
            assert 3.25 <= planner._midpoint(result) <= 4.25
    assert days == before


def test_refinement_cannot_enlarge_a_health_shortened_session():
    states, days = program()
    states[0] = states[0].model_copy(update={"current_health_status": planner.CurrentHealthStatus.SICK_OR_RECOVERING})
    days[0].recommendation = days[0].recommendation.model_copy(update={
        "distance_range_miles": (2.7, 3.2), "readiness": ReadinessFlag.CAUTION})
    evaluated = []
    def prefer_more(candidate):
        evaluated.append(candidate)
        return -sum(planner._midpoint(d.recommendation) for d in candidate if d.recommendation)
    with opportunity_search_policy():
        planner._allocate_visible_distance_ranges(days, states, (15, 18), CONFIG,
            allocation_selector=prefer_more, allocation_candidate_limit=4,
            preserve_projected_recovery_caps=True)
    assert evaluated
    assert all(c[0].recommendation.distance_range_miles[1] <= 3.2 for c in evaluated)


@pytest.mark.parametrize("reference", [8.45, 12.0])
def test_refined_candidates_retain_meaningful_long_and_quality_constraints(reference):
    from test_weekly_schedule import _role_days_for_allocation, _state
    state = _state(longest_run_30d_miles=reference, retained_long_run_capacity_miles=reference,
        days_since_last_run=3., days_since_long_run=8., days_since_quality_run=3.,
        typical_easy_run_miles=3.9, running_days_28d=12)
    states, days = _role_days_for_allocation(state, ["long", "easy", "quality", "easy"])
    evaluated = []
    def prefer_more(candidate):
        evaluated.append(candidate)
        return -sum(planner._midpoint(d.recommendation) for d in candidate if d.recommendation)
    with opportunity_search_policy():
        planner._allocate_visible_distance_ranges(days, states, (24, 26), CONFIG,
            weekly_target_range=(24, 26), allocation_selector=prefer_more,
            allocation_candidate_limit=1, preserve_projected_recovery_caps=True)
    assert len(evaluated) > 1
    for candidate in evaluated:
        assert candidate[0].recommendation.workout_type == planner.WorkoutType.LONG
        assert candidate[2].recommendation.workout_type in planner.QUALITY_WORKOUT_TYPES
        assert planner._midpoint(candidate[2].recommendation) <= planner._midpoint(days[2].recommendation)
        for i in (1, 3):
            assert planner._midpoint(candidate[0].recommendation) + .1 >= 1.15 * planner._midpoint(candidate[i].recommendation)
