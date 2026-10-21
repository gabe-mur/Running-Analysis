from datetime import datetime, timedelta, timezone

import pytest

from scripts.issued_plan_execution import OpeningPlanExecution
from scripts.replay_synthetic_easy_dose import synthetic_history
from run_analysis import adherence_projection as projection
from run_analysis.web.schemas import RecommendationResponse, WeeklyScheduleDay, WorkoutType
from test_prescription_matching import _schedule
from test_recommendation import CONFIG


START = datetime(2026, 9, 22, tzinfo=timezone.utc)


def schedule(generated, *, low=3.2, high=3.8, planned_day=2):
    when = START + timedelta(days=planned_day, hours=7)
    rec = RecommendationResponse(generated_at=generated, fitness_state_as_of=generated,
        planned_for=when, workout_type=WorkoutType.EASY, title="Issued fixture",
        distance_range_miles=(low, high), readiness="ready", confidence="moderate")
    result = _schedule(generated, rec)
    result.days = [WeeklyScheduleDay(date=(START + timedelta(days=i)).date(),
        recommendation=rec if i == planned_day else None, day_role="easy_run" if i == planned_day else "rest_day",
        rationale="fixture") for i in range(7)]
    result.planning_days = result.days
    return result


def test_opening_execution_preserves_issued_range_and_does_not_mutate_new_plan():
    contract = OpeningPlanExecution()
    try:
        assert contract.select(START, START + timedelta(days=1), schedule(START)) == []
        latest = schedule(START + timedelta(days=2), low=3.7, high=4.2)
        chosen = contract.select(START + timedelta(days=2), START + timedelta(days=3), latest)
        assert chosen[0].distance_range_miles == (3.2, 3.8)
        assert latest.days[2].recommendation.distance_range_miles == (3.7, 4.2)
        chosen[0].distance_range_miles = (1, 2)
        assert contract.opening.days[2].recommendation.distance_range_miles == (3.2, 3.8)
    finally:
        contract.close()


def test_production_match_and_original_contract_are_reported_separately():
    contract = OpeningPlanExecution()
    try:
        contract.select(START, START + timedelta(days=1), schedule(START))
        latest = schedule(START + timedelta(days=2), low=3.7, high=4.2)
        item = contract.select(START + timedelta(days=2), START + timedelta(days=3), latest)[0]
        run = projection.ProjectionRun(item.planned_for, 3.5, 38.5, WorkoutType.EASY,
            projection._projected_difficulty(item, 3.5, 11), prescribed_low_miles=3.2, prescribed_high_miles=3.8)
        contract.observe(run, latest.days[2].recommendation)
        record = contract.executions[0]
        assert record["within_issued_range"] is True
        assert record["within_latest_range"] is False
        assert record["production_match"]["distance_low_miles"] == 3.7
        assert record["production_match"]["distance_delta_miles"] == pytest.approx(.2)
    finally:
        contract.close()


@pytest.mark.parametrize("edge,actual_miles", [(None, 3.5), ("in_range_low", 3.2), ("in_range_high", 3.8)])
def test_simulator_keeps_forecast_separate_and_replans_after_old_contract_upload(monkeypatch, edge, actual_miles):
    template, history = synthetic_history(START)
    builds = []

    def build(states, *args, **kwargs):
        builds.append(kwargs.get("prior_schedule"))
        return schedule(states[0].as_of, low=3.2 if len(builds) == 1 else 3.7,
                        high=3.8 if len(builds) == 1 else 4.2)

    monkeypatch.setattr(projection, "build_weekly_schedule", build)
    snapshots = []
    contract = OpeningPlanExecution()
    try:
        result = projection.simulate_adherence(template, history, CONFIG, START,
            simulation_days=3, replan_observer=snapshots.append,
            execution_schedule_selector=contract.select, execution_observer=contract.observe,
            deterministic_profile=projection.DeterministicAdherenceProfile(
                projection.DeterministicAdherenceScenario(edge)) if edge else None)
        assert len(builds) == 4  # Three daily decisions plus the actual upload.
        assert builds[0] is None
        assert builds[2].days[2].recommendation.distance_range_miles == (3.7, 4.2)
        assert result[0].actual_sessions[0].distance_miles == actual_miles
        assert snapshots[2].planned_sessions[0].midpoint_miles == 3.95
        assert snapshots[2].committed_sessions[0].midpoint_miles == 3.5
        upload = snapshots[-1]
        assert upload.trigger == "post_upload"
        assert upload.expectation_reference == "issued_execution_contract"
        if edge is None:
            assert upload.continuous_distance_surprise_miles == pytest.approx(0)
        assert upload.recovery_surprise_units == pytest.approx(0)
        assert upload.published_forecast_comparison["actual_minus_published"]["continuous_distance_miles"] < 0
        assert upload.material_evidence_reasons == ()  # No athlete deviation from the issued contract.
        from dataclasses import replace
        from run_analysis.projection_gate import evaluate_replan_regressions, ProjectionGateConfig
        assert "unexpected_recovery_change" not in {
            f.code for f in evaluate_replan_regressions(upload, ProjectionGateConfig())}
        assert "unexpected_recovery_change" in {
            f.code for f in evaluate_replan_regressions(replace(upload, recovery_surprise_units=.05), ProjectionGateConfig())}
    finally:
        contract.close()


def test_opening_rest_day_does_not_execute_newly_inserted_run():
    contract = OpeningPlanExecution()
    try:
        contract.select(START, START + timedelta(days=1), schedule(START))
        assert contract.select(START + timedelta(days=1), START + timedelta(days=2),
                               schedule(START + timedelta(days=1), planned_day=1)) == []
    finally:
        contract.close()


def test_deleted_original_run_is_executed_but_not_resurrected_by_matcher():
    contract = OpeningPlanExecution()
    try:
        contract.select(START, START + timedelta(days=1), schedule(START))
        latest = schedule(START + timedelta(days=2), planned_day=3)
        item = contract.select(START + timedelta(days=2), START + timedelta(days=3), latest)[0]
        run = projection.ProjectionRun(item.planned_for, 3.5, 38.5, WorkoutType.EASY,
            projection._projected_difficulty(item, 3.5, 11), prescribed_low_miles=3.2, prescribed_high_miles=3.8)
        contract.observe(run, None)
        assert contract.executions[0]["within_issued_range"]
        assert contract.executions[0]["production_match"] is None
    finally:
        contract.close()


def test_selector_cannot_execute_future_window_or_duplicate_dates(monkeypatch):
    template, history = synthetic_history(START)
    monkeypatch.setattr(projection, "build_weekly_schedule", lambda *args, **kwargs: schedule(START))
    for chosen in ([schedule(START).days[2].recommendation],
                   [schedule(START, planned_day=0).days[0].recommendation] * 2):
        with pytest.raises(ValueError, match="unique run dates inside"):
            projection.simulate_adherence(template, history, CONFIG, START, simulation_days=1,
                execution_schedule_selector=lambda *args: chosen)


def test_next_refresh_preserves_both_forecasts_without_upload_replan(monkeypatch):
    template, history = synthetic_history(START)
    calls = []
    def build(states, *args, **kwargs):
        calls.append(1)
        return schedule(states[0].as_of, low=3.2 if len(calls) == 1 else 3.7,
                        high=3.8 if len(calls) == 1 else 4.2)
    monkeypatch.setattr(projection, "build_weekly_schedule", build)
    contract = OpeningPlanExecution()
    snapshots = []
    try:
        projection.simulate_adherence(template, history, CONFIG, START, simulation_days=4,
            emit_post_upload_replans=False, execution_schedule_selector=contract.select,
            replan_observer=snapshots.append)
        assert snapshots[-1].recovery_surprise_units == pytest.approx(0)
        assert snapshots[-1].published_forecast_comparison["actual_minus_published"]["recovery_residual_load"] < 0
    finally:
        contract.close()


def test_published_forecast_does_not_compare_different_decision_times():
    state, _ = synthetic_history(START)
    expected = projection._ExpectedOpeningState(
        START + timedelta(days=1), state.recovery_residual_load,
        state.recent_load.continuous_distance_miles,
        state.recent_load.continuous_short_term_distance_miles)
    comparison = projection._published_forecast_comparison(expected, state)
    assert comparison["comparable"] is False
    assert all(value is None for value in comparison["actual_minus_published"].values())
    assert projection._published_forecast_comparison(None, state) is None
