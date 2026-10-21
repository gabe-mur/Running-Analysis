from datetime import datetime, timedelta, timezone
import json
from types import SimpleNamespace

import pytest

from scripts import audit_capped_calendar as audit
from scripts.audit_funding_candidates import finalized_coaching_probe
from scripts.replay_synthetic_easy_dose import synthetic_history
from run_analysis.web.schemas import (RecommendationRequest, RecommendationResponse,
    WeeklyScheduleDay, WeeklyScheduleResponse, WeeklyTargetEvidence, PlannerDiagnostics)


def test_frozen_build_inputs_round_trip_without_callable_or_type_loss():
    state, _ = synthetic_history(datetime(2026, 9, 22, tzinfo=timezone.utc))
    bound = audit.BUILD_SIGNATURE.bind([state], RecommendationRequest(health_status="normal"), {},
        forced_rest_offsets={0}, completed_activities_by_offset={0: []},
        expected_target_projector=lambda candidate, n: ((16.5, 17.5),) * n)
    bound.apply_defaults()
    frozen = json.loads(json.dumps(audit.freeze_inputs(bound.arguments)))
    restored = audit.thaw_inputs(frozen)
    assert restored["daily_states"] == [state]
    assert restored["forced_rest_offsets"] == {0}
    assert restored["completed_activities_by_offset"] == {0: []}
    assert restored["request"] == bound.arguments["request"]
    assert restored["expected_target_projector"]([], 1) == ((16.5, 17.5),)
    with pytest.raises(ValueError, match="too short"):
        restored["expected_target_projector"]([], 1000)


def test_inspection_restores_search_hook_on_error():
    original = audit.planner.adaptive_run_day_offsets
    with pytest.raises(RuntimeError):
        with audit.inspect_winner():
            assert audit.planner.adaptive_run_day_offsets is not original
            raise RuntimeError("stop")
    assert audit.planner.adaptive_run_day_offsets is original


def test_report_refuses_overwrite(tmp_path):
    path = tmp_path / "result.json"
    audit.write_new(path, {"value": 1})
    with pytest.raises(FileExistsError):
        audit.write_new(path, {"value": 2})
    assert json.loads(path.read_text()) == {"value": 1}


def test_freezing_preserves_api_excluded_full_horizon_and_diagnostics():
    now = datetime(2026, 9, 22, tzinfo=timezone.utc)
    state, _ = synthetic_history(now)
    evidence = WeeklyTargetEvidence(recent_7d_miles=15, chronic_42d_weekly_miles=15,
        best_sustained_28d_weekly_miles=15, peak_7d_miles=15,
        demonstrated_run_days_per_week=3, capacity_reference_miles=15, rationale="fixture")
    days = [WeeklyScheduleDay(date=(now + timedelta(days=i)).date(), day_role="rest_day", rationale="fixture")
            for i in range(21)]
    days[10] = days[10].model_copy(update={"recommendation": RecommendationResponse(
        generated_at=now, fitness_state_as_of=now, planned_for=now + timedelta(days=10),
        workout_type="easy", title="Beyond visible week", distance_range_miles=(3.7, 4.2),
        confidence="moderate", readiness="ready")})
    prior = WeeklyScheduleResponse(generated_at=now, start_date=now.date(),
        end_date=(now + timedelta(days=6)).date(), target_run_count=3, target_distance_range_miles=(15,16),
        target_evidence=evidence, run_count=0, projected_distance_range_miles=(0,0), summary="fixture",
        days=days[:7], planning_days=days, planner_diagnostics=PlannerDiagnostics(candidate_count=19))
    bound = audit.BUILD_SIGNATURE.bind([state], RecommendationRequest(health_status="normal"), {},
        prior_schedule=prior, expected_target_projector=lambda candidate, n: ((15,16),) * n)
    bound.apply_defaults()
    frozen = json.loads(json.dumps(audit.freeze_inputs(bound.arguments)))
    assert "planning_days" not in prior.model_dump()
    restored = audit.thaw_inputs(frozen)["prior_schedule"]
    assert restored == prior
    assert len(restored.planning_days) == 21
    assert restored.planning_days[10].recommendation.distance_range_miles == (3.7, 4.2)
    del frozen["inputs"]["prior_schedule"]["planning_days"]
    with pytest.raises(ValueError, match="lost internal"):
        audit.thaw_inputs(frozen)


@pytest.mark.parametrize("mismatch", [False, True])
def test_finalized_coaching_probe_rejects_mismatch_and_restores_hooks(monkeypatch, mismatch):
    now = datetime(2026, 9, 22, tzinfo=timezone.utc)
    rec = RecommendationResponse(generated_at=now, fitness_state_as_of=now, planned_for=now,
        workout_type="easy", title="fixture", distance_range_miles=(3,4), confidence="moderate", readiness="ready")
    materialize = lambda *a, **kw: [SimpleNamespace(recommendation=rec)]
    monkeypatch.setattr(audit.planner, "_materialize_candidate_sessions", materialize)
    def fail(*a, **kw):
        raise RuntimeError("coaching failure")
    monkeypatch.setattr(audit.planner, "_adaptive_candidate_cost", fail)
    inputs = dict(daily_states=[SimpleNamespace(as_of=now)], daily_state_options=[], request=None,
                  config={}, target_distance_range=(15,16), completed_miles_by_offset={})
    scored = {"sessions": [dict(date=now.date(), type="easy", range=(4,5) if mismatch else (3,4), at=now)]}
    if mismatch:
        assert not finalized_coaching_probe(inputs, dict(offsets=[0], roles={}), scored)["sessions_match"]
    else:
        with pytest.raises(RuntimeError, match="coaching failure"):
            finalized_coaching_probe(inputs, dict(offsets=[0], roles={}), scored)
    assert audit.planner._materialize_candidate_sessions is materialize
