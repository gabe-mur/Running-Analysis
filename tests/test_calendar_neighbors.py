import pytest
from datetime import datetime, timezone
from inspect import signature
from types import SimpleNamespace

from scripts.calendar_neighbor_prototype import calendar_neighbors, calendar_neighbor_policy
from run_analysis import weekly_schedule as planner


def test_known_opening_counterexample_is_a_neighbor():
    opening = (1, 2, 4, 6, 8, 10, 12, 14, 17, 19, 20)
    neighbors = calendar_neighbors(opening, range(21))
    assert (1, 2, 4, 6, 10, 12, 14, 17, 19, 20) in neighbors
    assert len(neighbors) <= 63


def test_neighbors_include_add_remove_and_moves_across_entire_horizon():
    neighbors = calendar_neighbors((0, 10, 20), range(21))
    assert (0, 20) in neighbors
    assert (0, 10, 15, 20) in neighbors
    assert (0, 10, 19) in neighbors
    assert len(neighbors) == len(set(neighbors))
    assert neighbors == calendar_neighbors((20, 0, 10, 10), reversed(range(21)))


def test_neighbors_respect_forced_and_completed_date_exclusions():
    allowed = set(range(21)) - {1, 2, 11}
    for neighbor in calendar_neighbors((0, 10, 20), allowed):
        assert set(neighbor) <= allowed
        assert tuple(sorted(set(neighbor))) == neighbor
        assert neighbor
    assert calendar_neighbors((), allowed) == []
    assert calendar_neighbors((1, 10), allowed) == []
    assert calendar_neighbors((0,), {0}) == []


@pytest.mark.parametrize("enabled", [False, True])
def test_context_restores_hook_after_failure(enabled):
    original = planner.adaptive_run_day_offsets
    with pytest.raises(RuntimeError):
        with calendar_neighbor_policy(enabled=enabled):
            assert (planner.adaptive_run_day_offsets is not original) == enabled
            raise RuntimeError("test")
    assert planner.adaptive_run_day_offsets is original


def test_pass_budget_must_be_positive():
    with pytest.raises(ValueError, match="positive"):
        with calendar_neighbor_policy(max_passes=0):
            pass


@pytest.mark.parametrize("operation", ["remove", "add", "move", "no_improvement", "stable"])
@pytest.mark.parametrize("audit", [False, True])
@pytest.mark.parametrize("passes", [1, 4])
def test_polish_uses_full_objective_retains_baseline_and_preserves_stability(monkeypatch, operation, audit, passes):
    seed = (0, 2, 4, 8)
    targets = {"remove": (0, 2, 4), "add": (0, 2, 4, 8, 9), "move": (0, 2, 4, 9),
               "no_improvement": seed, "stable": (0, 2, 4)}
    expected = targets[operation]
    original_signature = signature(planner.adaptive_run_day_offsets)
    base = planner._score_breakdown(seed, (0, 0, 0, 10), 0, 3.5,
        {"planned_midpoint_miles": 20, "integrated_target_low_miles": 18})

    def search(**inputs):
        d = inputs["_planner_diagnostics"]
        d.winning_candidate = d.unconstrained_winning_candidate = base
        d.near_term_stable_candidate = base if operation == "stable" else None
        return list(seed)

    search.__signature__ = original_signature
    monkeypatch.setattr(planner, "adaptive_run_day_offsets", search)
    monkeypatch.setattr(planner, "typical_easy_distance", lambda state: (3, 4))
    seen = []

    def joint(offsets, **kwargs):
        assert kwargs["reconcile_allocated_prefix"] is True
        seen.append((offsets, kwargs["role_overrides_by_offset"]))
        kwargs["diagnostic_details"].update(planned_midpoint_miles=20, integrated_target_low_miles=18)
        value = 1 if offsets == expected and operation != "no_improvement" else 20
        if operation == "stable" and offsets == (2, 4, 8):
            value = 0  # Best raw score changes the protected first date.
        return (0, 0, 0, value)

    def coaching(*args, **kwargs):
        assert not kwargs["include_mileage_path"]
        assert not kwargs["include_recovery_interactions"]
        assert not kwargs["include_cadence_pressure"]
        return 0

    monkeypatch.setattr(planner, "_joint_candidate_program_cost", joint)
    monkeypatch.setattr(planner, "_adaptive_candidate_cost", coaching)
    diagnostics = planner._PlannerSearchDiagnostics()
    with calendar_neighbor_policy(audit_opening=audit, max_passes=passes) as records:
        result = planner.adaptive_run_day_offsets(
            [SimpleNamespace(as_of=datetime(2026, 9, 22, tzinfo=timezone.utc))] * 10,
            SimpleNamespace(health_status=planner.CurrentHealthStatus.NORMAL), {}, 4, (15, 18),
            prior_run_offsets=set(seed) if operation == "stable" else None,
            prior_roles_by_offset={0: "easy", 2: "easy"} if operation == "stable" else None,
            _planner_diagnostics=diagnostics)
    assert result == list(expected)
    assert records[0]["winner"]["total_cost"] <= base.total_cost
    assert records[0]["evaluated_count"] <= 60 * passes
    if passes > 1:
        assert records[0]["stop_reason"] == "unchanged_neighborhood_winners"
        assert records[0]["passes"] <= 2
        assert len(seen) == len(set((offsets, tuple(sorted((roles or {}).items()))) for offsets, roles in seen))
    if audit:
        assert records[0]["opening_neighbor_audit"]["affects_selection"] is False
    else:
        assert records[0]["opening_neighbor_audit"] is None
    if operation == "stable":
        assert diagnostics.stability_tie_break_applied
        assert diagnostics.winning_role_overrides == {0: "easy", 2: "easy"}
        assert tuple(diagnostics.unconstrained_winning_candidate.offsets) == (2, 4, 8)


@pytest.mark.parametrize("budget,expected,reason", [
    (2, (0, 2), "pass_limit"),
    (6, (0,), "unchanged_neighborhood_winners"),
])
def test_repeat_distinguishes_budget_exhaustion_from_convergence(monkeypatch, budget, expected, reason):
    original_signature = signature(planner.adaptive_run_day_offsets)
    seed = (0, 2, 4, 8)
    base = planner._score_breakdown(seed, (0, 0, 0, 10), 0, 3.5)

    def search(**inputs):
        inputs["_planner_diagnostics"].winning_candidate = base
        inputs["_planner_diagnostics"].unconstrained_winning_candidate = base
        return list(seed)

    search.__signature__ = original_signature
    monkeypatch.setattr(planner, "adaptive_run_day_offsets", search)
    monkeypatch.setattr(planner, "typical_easy_distance", lambda state: (3, 4))
    calls = []
    costs = {seed: 10, (0, 2, 4): 8, (0, 2): 6, (0,): 4}

    def joint(offsets, **kwargs):
        calls.append(offsets)
        return (0, 0, 0, costs.get(offsets, 20))

    monkeypatch.setattr(planner, "_joint_candidate_program_cost", joint)
    monkeypatch.setattr(planner, "_adaptive_candidate_cost", lambda *a, **k: 0)
    with calendar_neighbor_policy(max_passes=budget) as records:
        result = planner.adaptive_run_day_offsets(
            [SimpleNamespace(as_of=datetime(2026, 9, 22, tzinfo=timezone.utc))] * 10,
            SimpleNamespace(health_status=planner.CurrentHealthStatus.NORMAL), {}, 4, (15, 18))
    assert tuple(result) == expected
    assert records[0]["stop_reason"] == reason
    assert len(calls) == len(set(calls))
    values = [x["independent_cost"] for x in records[0]["rounds"]]
    assert values == sorted(values, reverse=True)
    assert records[0]["passes"] == min(budget, 4)
