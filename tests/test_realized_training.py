from copy import deepcopy

import pytest

from scripts.audit_realized_training import audit


def session(day, miles=4, kind="easy", *, planned=False):
    item = {"workout_type": kind}
    if planned:
        item.update(planned_for=f"2026-09-{day:02d}T07:00:00+00:00", midpoint_miles=miles,
                    distance_low_miles=miles - .25, distance_high_miles=miles + .25)
    else:
        item.update(occurred_at=f"2026-09-{day:02d}T07:00:00+00:00", distance_miles=miles)
    return item


def snapshot(day, planned, *, hour=0, decision=None):
    return {"generated_at": f"2026-09-{day:02d}T{hour:02d}:00:00+00:00",
            "decision_start_date": f"2026-09-{decision or day:02d}",
            "planned_sessions": planned, "material_evidence_reasons": []}


def report(actual, snapshots, until="2026-09-29T00:00:00+00:00"):
    return {"mode": "perfect", "replans": snapshots,
            "realized_execution": {"start_at": "2026-09-22T00:00:00+00:00",
                                   "observed_until": until, "complete": True,
                                   "historical_sessions": [session(20, 7.5, "long")],
                                   "sessions": actual}}


def test_seed_history_fills_windows_without_becoming_future_delivery():
    result = audit(report([session(23)], [snapshot(22, [session(23, planned=True)])]))
    assert result["total_completed_miles"] == 4
    assert result["rolling_daily"][0]["windows"]["7"]["miles"] == 7.5
    assert result["rolling_daily"][-1]["windows"]["7"]["miles"] == 4
    assert result["rolling_daily"][-1]["windows"]["7"]["wholly_simulated"]
    assert not result["rolling_daily"][-1]["windows"]["14"]["wholly_simulated"]
    cohort = result["forecast_delivery_cohorts"]["7"][0]
    assert cohort["delivered_miles"] == 4


@pytest.mark.parametrize("miles", [3.75, 4.25])
def test_entire_original_range_is_compliant(miles):
    result = audit(report([session(23, miles)], [snapshot(22, [session(23, planned=True)])]))
    assert result["commitments"]["7"]["outcomes"]["within_initial_range"] == 1


def test_rewrite_followed_perfectly_does_not_erase_original_commitment():
    before = snapshot(22, [session(25, planned=True)])
    after = snapshot(24, [session(26, 5, planned=True)])
    data = report([session(26, 5)], [before, after])
    original = deepcopy(data)
    result = audit(data)
    counts = result["commitments"]["7"]["outcomes"]
    assert counts["forecast_run_not_completed_on_date"] == 1
    assert counts["added_run"] == 1
    assert result["commitments"]["7"]["matured_dates_with_revisions"] == 2
    assert data == original


def test_post_upload_consumption_is_not_churn_and_tail_is_pending():
    result = audit(report([session(23)], [snapshot(22, [session(23, planned=True)]),
                                         snapshot(23, [session(29, planned=True)], hour=9, decision=24)]))
    assert result["commitments"]["7"]["matured_dates_with_revisions"] == 0
    assert any(x["date"] == "2026-09-29" for x in result["commitments"]["7"]["pending"])
    assert len(result["forecast_delivery_cohorts"]["7"]) == 1


def test_rolling_window_has_exclusive_upper_boundary_and_no_partial_day():
    data = report([session(23)], [snapshot(22, [])], until="2026-09-23T09:00:00+00:00")
    data["realized_execution"]["complete"] = False
    result = audit(data)
    assert len(result["rolling_daily"]) == 1
    assert result["rolling_daily"][0]["windows"]["7"]["miles"] == 7.5
    assert result["total_completed_miles"] == 4
    assert result["forecast_delivery_cohorts"]["4"] == []


def test_quality_gaps_use_actual_schema_types_and_historical_boundary():
    data = report([session(23, kind="tempo_threshold"), session(28, kind="race")], [snapshot(22, [])])
    data["realized_execution"]["historical_sessions"].append(session(19, kind="intervals"))
    result = audit(data)
    assert [x["days"] for x in result["session_gaps"]["quality"]["completed_intervals"]] == [4, 5]


def test_changes_that_revert_are_retained_even_if_execution_matches():
    result = audit(report([session(25)], [snapshot(22, [session(25, planned=True)]),
                                         snapshot(23, [session(25, 5, planned=True)]),
                                         snapshot(24, [session(25, planned=True)])]))
    rows = result["commitments"]["7"]
    assert rows["outcomes"]["within_initial_range"] == 1
    assert rows["matured_dates_with_revisions"] == 1
    assert len(next(x for x in rows["matured"] if x["date"] == "2026-09-25")["revisions"]) == 2


def test_role_only_revision_is_not_hidden_by_same_type_and_mileage():
    initial, changed = session(25, planned=True), session(25, planned=True)
    initial["planning_role"], changed["planning_role"] = "ordinary_easy", "medium_long"
    result = audit(report([session(25)], [snapshot(22, [initial]), snapshot(24, [changed])]))
    assert result["commitments"]["7"]["matured_dates_with_revisions"] == 1


def test_cumulative_small_edits_survive_pairwise_check_but_not_original_range_audit():
    """Oct 9 drift reduced to a small date-translated measurement fixture.

    This records the current pairwise blind spot; it does not certify stability
    or redefine compliant execution as matching a midpoint.
    """
    from datetime import datetime
    from run_analysis.adherence_projection import ProjectionPlanSession, ProjectionReplan
    from run_analysis.projection_gate import _future_distance_changes
    from run_analysis.web.schemas import WorkoutType

    snapshots = []
    for day, hour, bounds in ((22, 9, (3., 3.5)), (23, 0, (3.2, 3.8)),
                              (24, 0, (3.5, 4.)), (24, 9, (3.7, 4.2))):
        item = session(25, sum(bounds)/2, planned=True)
        item.update(distance_low_miles=bounds[0], distance_high_miles=bounds[1])
        snapshots.append(snapshot(day, [item], hour=hour))

    replans = [ProjectionReplan(
        generated_at=datetime.fromisoformat(s['generated_at']), opening_load_ratio=None,
        target_low_miles=17., target_high_miles=18., committed_sessions=(),
        planned_sessions=(ProjectionPlanSession(
            planned_for=datetime.fromisoformat(s['planned_sessions'][0]['planned_for']),
            workout_type=WorkoutType.EASY,
            midpoint_miles=s['planned_sessions'][0]['midpoint_miles']),)) for s in snapshots]
    assert all(not _future_distance_changes(a, b, horizon_days=4, tolerance_miles=.5)
               for a, b in zip(replans, replans[1:]))
    assert len(_future_distance_changes(replans[0], replans[-1],
                                        horizon_days=4, tolerance_miles=.5)) == 1
    actual = session(25, 3.95)
    result = audit(report([actual], snapshots))
    commitment = next(x for x in result['commitments']['4']['matured'] if x['date']=='2026-09-25')
    assert commitment['outcome'] == 'outside_initial_range'
    assert len(commitment['revisions']) == 3
    assert all(not x['material_evidence'] for x in commitment['revisions'])
    assert commitment['revisions'][-1]['plan']['midpoint_miles'] - commitment['initial']['midpoint_miles'] == pytest.approx(.7)
    assert 3.7 <= actual['distance_miles'] <= 4.2  # Latest plan was followed.


def test_runner_preserves_actual_work_when_gate_stops_before_weekly_return(monkeypatch, tmp_path):
    import json
    import sys
    from datetime import datetime, timezone
    from scripts import replay_synthetic_easy_dose as runner
    from run_analysis.adherence_projection import ProjectionReplan
    from run_analysis.projection_gate import ProjectionGateFailure, ProjectionGateTriggered

    when = datetime(2026, 9, 22, 9, tzinfo=timezone.utc)
    failure = ProjectionGateFailure("fixture_failure", when, "Intentional stop", {})

    def stop(gate, snap):
        gate.snapshots.append(snap)
        gate.failures.append(failure)
        raise ProjectionGateTriggered(failure)

    def simulate(template, runs, config, start, **kwargs):
        # Reuse a valid run without invoking the expensive scheduler.
        from dataclasses import replace
        run = replace(runs[-1], start_time=when, distance_miles=4,
                      prescribed_low_miles=3.75, prescribed_high_miles=4.25)
        kwargs["execution_observer"](run, None)
        kwargs["replan_observer"](ProjectionReplan(
            generated_at=when, opening_load_ratio=None, target_low_miles=15,
            target_high_miles=18, planned_sessions=(), committed_sessions=(), planning_seconds=0.0))

    output = tmp_path / "partial.json"
    monkeypatch.setattr(runner.ProjectionGate, "observe", stop)
    monkeypatch.setattr(runner.projection, "simulate_adherence", simulate)
    monkeypatch.setattr(sys, "argv", ["runner", "--output-json", str(output)])
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 1
    saved = json.loads(output.read_text())
    assert saved["status"] == "failed" and saved["weeks"] == []
    assert saved["realized_execution"]["sessions"][0]["distance_miles"] == 4
    assert saved["realized_execution"]["observed_until"] == when.isoformat()
    assert not saved["realized_execution"]["complete"]


@pytest.mark.parametrize("scenario", ["below_range_easy", "above_range_or_harder"])
def test_deterministic_scenario_does_not_change_evidence_mode_on_compliant_followup(monkeypatch, scenario):
    from run_analysis import adherence_projection as projection
    from scripts.replay_synthetic_easy_dose import synthetic_history
    from test_issued_plan_execution import schedule, START
    from test_recommendation import CONFIG

    def build(states, *args, **kwargs):
        opening = schedule(states[0].as_of, planned_day=0, low=3.7, high=4.2)
        opening.days[2] = schedule(states[0].as_of, planned_day=2, low=3.7, high=4.2).days[2]
        opening.planning_days = opening.days
        return opening

    monkeypatch.setattr(projection, "build_weekly_schedule", build)
    template, history = synthetic_history(START)
    actual, snapshots = [], []
    projection.simulate_adherence(template, history, CONFIG, START, simulation_days=3,
        replan_observer=snapshots.append, execution_observer=lambda run, _: actual.append(run),
        deterministic_profile=projection.DeterministicAdherenceProfile(
            projection.DeterministicAdherenceScenario(scenario)))
    assert len(actual) == 2
    uploads = [snap for snap in snapshots if snap.trigger == "post_upload"]
    assert len(uploads) == 2
    assert uploads[0].material_evidence_reasons
    if scenario == "below_range_easy":
        assert actual[0].distance_miles == pytest.approx(2.775)
        assert actual[0].difficulty.zone_load is None
        assert actual[0].difficulty.zone_breakdown.hard_minutes == 0
        assert uploads[0].recovery_surprise_units < 0
    else:
        assert actual[0].distance_miles > 4.2
        assert actual[0].difficulty.zone_load > 0
        assert actual[0].difficulty.zone_breakdown.hard_minutes > 0
        assert uploads[0].recovery_surprise_units > 0
    assert actual[1].distance_miles == pytest.approx(3.95)
    assert actual[1].difficulty.zone_load is None
    assert actual[1].difficulty.difficulty_flags == ["adherence_projection_no_observed_response"]
    assert actual[1].adherence_note == "within prescribed margin"
    assert snapshots[-1].material_evidence_reasons == ()
    assert snapshots[-1].recovery_surprise_units == pytest.approx(0, abs=.01)
