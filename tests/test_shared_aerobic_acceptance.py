from dataclasses import replace
from datetime import timedelta
from inspect import signature
from types import SimpleNamespace

import pytest

from run_analysis.aerobic_acceptance import assess_aerobic_exposure, cadence_exceptions
from run_analysis import weekly_schedule as planner
from scripts.calendar_neighbor_prototype import calendar_neighbor_policy, NoAcceptableCalendarError
from test_aerobic_acceptance_case import gap_fixture, gap_failures, ORIGIN, OFFSETS


@pytest.mark.parametrize("change", [{}, {"cadence_exception_reasons": ("recovery guardrail",)},
                                   {"planning_mode": "baseline_building"}])
def test_shared_assessment_matches_gate(change):
    snapshot = replace(gap_fixture(), **change)
    result = assess_aerobic_exposure(
        sessions=[(s.planned_for.date(), s.midpoint_miles) for s in snapshot.planned_sessions],
        target_trajectory=snapshot.target_trajectory, ordinary_miles=snapshot.ordinary_easy_midpoint_miles,
        planning_mode=snapshot.planning_mode, exception_reasons=snapshot.cadence_exception_reasons)
    failures = gap_failures(snapshot)
    assert result.rejected == bool(failures)
    if failures:
        assert result.evidence == failures[0].evidence


def test_exception_extraction_preserves_existing_markers_and_order():
    reasons = ["Ordinary rest", "Readiness guardrail delayed timing", "Readiness guardrail delayed timing",
               "A taxing run was replaced with aerobic work"]
    assert cadence_exceptions(reasons) == (reasons[1], reasons[3])


@pytest.mark.parametrize("scenario", ["alternative", "all_rejected", "exception", "missing", "stable_only"])
def test_both_lanes_filter_before_ranking_and_do_not_return_invalid_fallback(monkeypatch, scenario):
    fixture = gap_fixture()
    seed = OFFSETS
    good = tuple(sorted((*seed, 7)))
    base = planner._score_breakdown(seed, (0, 0, 0, .7), 0, 3.5,
        {"planned_midpoint_miles": 48.45, "integrated_target_low_miles": 51.17})
    original_signature = signature(planner.adaptive_run_day_offsets)

    def search(**inputs):
        diagnostics = inputs["_planner_diagnostics"]
        diagnostics.winning_candidate = diagnostics.unconstrained_winning_candidate = base
        diagnostics.near_term_stable_candidate = base
        return list(seed)

    search.__signature__ = original_signature
    monkeypatch.setattr(planner, "adaptive_run_day_offsets", search)
    monkeypatch.setattr(planner, "typical_easy_distance", lambda state: (3, 4))

    def joint(offsets, **kwargs):
        protected = bool(kwargs.get("role_overrides_by_offset"))
        accepted = offsets == good and scenario in {"alternative", "stable_only"}
        accepted = accepted and (scenario != "stable_only" or protected)
        runs = [(s.planned_for.date(), s.midpoint_miles) for s in fixture.planned_sessions]
        if accepted:
            runs.append(((ORIGIN + timedelta(days=7)).date(), 4.25))
        facts = dict(sessions=runs, target_trajectory=fixture.target_trajectory, ordinary_miles=3.75,
                     exception_reasons=("recovery guardrail",) if scenario == "exception" else ())
        kwargs["diagnostic_details"].update(
            aerobic_acceptance_facts=facts, planned_midpoint_miles=sum(x[1] for x in runs),
            integrated_target_low_miles=51.17)
        return (0, 0, 0, 1.7 if accepted else .7)

    monkeypatch.setattr(planner, "_joint_candidate_program_cost", joint)
    monkeypatch.setattr(planner, "_adaptive_candidate_cost", lambda *args, **kwargs: 0)
    diagnostics = planner._PlannerSearchDiagnostics()
    inputs = dict(daily_states=[SimpleNamespace(as_of=ORIGIN)] * 21,
                  request=SimpleNamespace(health_status=planner.CurrentHealthStatus.NORMAL), config={},
                  target_run_count=9, target_distance_range=(16.5, 17.5),
                  prior_run_offsets=set(seed), prior_roles_by_offset={0: "easy", 2: "quality"},
                  _planner_diagnostics=diagnostics,
                  _acceptance_planning_mode=None if scenario == "missing" else "established")
    with calendar_neighbor_policy(enforce_acceptance=True) as records:
        if scenario in {"all_rejected", "missing"}:
            with pytest.raises(NoAcceptableCalendarError):
                planner.adaptive_run_day_offsets(**inputs)
        else:
            result = planner.adaptive_run_day_offsets(**inputs)
            if scenario != "exception":
                assert result == list(good)
                assert records[0]["acceptance_rejected"] > 0
            else:
                assert records[0]["acceptance_rejected"] == 0
            if scenario == "stable_only":
                assert diagnostics.unconstrained_winning_candidate is None
                assert diagnostics.winning_role_overrides == {0: "easy", 2: "quality"}
