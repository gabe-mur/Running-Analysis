from datetime import timedelta

import pytest

from scripts.finalized_scoring_trial import finalized_scoring_policy
from run_analysis import weekly_schedule as planner
from run_analysis.web.schemas import RecommendationRequest, RecommendationResponse, WeeklyScheduleDay
from test_recommendation import _state, CONFIG


def test_trial_is_opt_in_nested_and_restored_on_failure():
    assert planner._FINALIZED_SCORING_TRIAL.get() is None
    with finalized_scoring_policy() as outer:
        assert planner._FINALIZED_SCORING_TRIAL.get() is outer
        with pytest.raises(RuntimeError):
            with finalized_scoring_policy(False):
                assert planner._FINALIZED_SCORING_TRIAL.get() is None
                raise RuntimeError("stop")
        assert planner._FINALIZED_SCORING_TRIAL.get() is outer
    assert planner._FINALIZED_SCORING_TRIAL.get() is None


def program():
    base = _state()
    states = [base.model_copy(update={"as_of": base.as_of + timedelta(days=i)}) for i in range(3)]
    days = []
    for index, state in enumerate(states):
        result = None if index == 1 else RecommendationResponse(
            generated_at=base.as_of, fitness_state_as_of=state.as_of, planned_for=state.as_of,
            workout_type="easy", title="Exact final dose", distance_range_miles=(3.7,4.2),
            confidence="moderate", readiness="ready")
        days.append(WeeklyScheduleDay(date=state.as_of.date(), planned_at=state.as_of,
            recommendation=result, day_role="easy_run" if result else "rest_day", rationale="fixture"))
    return states, days


def test_finalized_projection_uses_exact_previous_prescriptions_and_no_new_recommendations(monkeypatch):
    states, days = program()
    calls = []
    original = planner._project_state
    def project(raw, preceding, config, **kwargs):
        calls.append((raw, list(preceding)))
        return original(raw, preceding, config, **kwargs)
    monkeypatch.setattr(planner, "_project_state", project)
    def forbidden(*a, **kw):
        raise AssertionError("Final scoring must not regenerate workouts")
    monkeypatch.setattr(planner, "_materialize_candidate_sessions", forbidden)
    with finalized_scoring_policy() as metrics:
        cost, sessions = planner._finalized_program_coaching_cost(days, states,
            [[s] for s in states], RecommendationRequest(health_status="normal"), CONFIG, (15,17))
    assert [s.offset for s in sessions] == [0,2]
    assert sessions[0].recommendation is days[0].recommendation
    assert sessions[1].recommendation is days[2].recommendation
    assert calls[0] == (states[0], [])
    assert calls[1] == (states[2], [days[0].recommendation])
    assert cost >= 0 and metrics["evaluations"] == 1
    assert metrics["projection_seconds"] >= 0


def test_finalized_projection_requires_actual_timed_state():
    states, days = program()
    days[0].recommendation.planned_for += timedelta(hours=3)
    with pytest.raises(ValueError, match="matching timed"):
        planner._finalized_program_coaching_cost(days, states, [[s] for s in states],
            RecommendationRequest(health_status="normal"), CONFIG, (15,17))


def test_coaching_diagnostics_preserve_exact_cost_and_account_for_every_charge():
    states,days=program()
    args=(days,states,[[s] for s in states],RecommendationRequest(health_status='normal'),CONFIG,(15,17))
    original,_=planner._finalized_program_coaching_cost(*args)
    details={}
    measured,_=planner._finalized_program_coaching_cost(*args,diagnostic_components=details)
    assert original==measured
    assert sum(details.values())==pytest.approx(measured)
    assert set(details)=={'rest_selection','recovery_substitution','readiness','recovery_interaction',
        'weather','load_ratio','mileage_path','cadence_idle','key_session_cadence'}
    assert details['recovery_interaction']==details['mileage_path']==details['cadence_idle']==0


def test_explicit_empty_program_does_not_fall_back_to_provisional_sessions(monkeypatch):
    def forbidden(*a, **kw):
        raise AssertionError("Empty finalized program is not a missing override")
    monkeypatch.setattr(planner, "_materialize_candidate_sessions", forbidden)
    state = _state()
    assert planner._adaptive_candidate_cost((), [state], [[state]],
        RecommendationRequest(health_status="normal"), CONFIG, (15,17),
        include_mileage_path=False, include_recovery_interactions=False,
        include_cadence_pressure=False, _sessions_override=[]) == 0


@pytest.mark.parametrize("enabled", [False, True])
def test_fixed_allocation_exposes_coaching_only_when_trial_enabled(monkeypatch, enabled):
    states, days = program()
    sessions = [planner._CandidateSession(i, states[i], day.recommendation)
                for i, day in enumerate(days) if day.recommendation]
    calls = []
    def finalized(allocated, *args, **kwargs):
        calls.append(allocated)
        return 42.0, sessions
    monkeypatch.setattr(planner, "_finalized_program_coaching_cost", finalized)
    details = {}
    with finalized_scoring_policy(enabled):
        planner._joint_candidate_program_cost((0,2), states, [[s] for s in states],
            RecommendationRequest(health_status="normal"), CONFIG, (15,17),
            _allocated_override=days, _sessions_override=sessions, diagnostic_details=details)
    if enabled:
        assert calls == [days]
        assert details["finalized_coaching_cost"] == 42
    else:
        assert calls == [] and "finalized_coaching_cost" not in details
