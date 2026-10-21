from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from run_analysis.adherence_projection import (
    ProjectionActualSession,
    ProjectionPlanSession,
    ProjectionReplan,
    ProjectionWeek,
)
from run_analysis.projection_gate import (
    ProjectionGate,
    ProjectionGateConfig,
    ProjectionGateTriggered,
    build_projection_report,
    evaluate_replan_regressions,
)
from run_analysis.web.schemas import WorkoutType


OPENING = datetime(2026, 9, 19, 7, tzinfo=timezone.utc)


def _session(
    day: int,
    workout_type: WorkoutType = WorkoutType.EASY,
    *,
    readiness: str | None = None,
    midpoint_miles: float = 4.0,
):
    return ProjectionPlanSession(
        planned_for=OPENING + timedelta(days=day),
        workout_type=workout_type,
        midpoint_miles=midpoint_miles,
        readiness=readiness,
    )


def _replan(
    *sessions: ProjectionPlanSession,
    generated_at: datetime = OPENING,
    target: tuple[float, float] = (17.0, 18.0),
    planning_mode: str = "established",
    material_evidence_reasons: tuple[str, ...] = (),
    recovery_surprise_units: float | None = None,
    days_since_long_run: float | None = None,
    cadence_exception_reasons: tuple[str, ...] = (),
    trigger: str = "scheduled_refresh",
    boundary_session_miles: float | None = None,
    peak_projected_continuous_mileage_rate: float | None = None,
    target_trajectory: tuple[tuple[float, float], ...] = (),
) -> ProjectionReplan:
    return ProjectionReplan(
        generated_at=generated_at,
        opening_load_ratio=1.0,
        target_low_miles=target[0],
        target_high_miles=target[1],
        planned_sessions=tuple(sessions),
        committed_sessions=(),
        capacity_reference_miles=16.0,
        planning_mode=planning_mode,
        ordinary_easy_midpoint_miles=4.0,
        boundary_session_miles=boundary_session_miles,
        peak_projected_continuous_mileage_rate=(
            peak_projected_continuous_mileage_rate
        ),
        planning_seconds=1.25,
        target_trajectory=target_trajectory,
        material_evidence_reasons=material_evidence_reasons,
        opening_recovery_residual_load=(
            1.0 + recovery_surprise_units
            if recovery_surprise_units is not None
            else None
        ),
        expected_opening_recovery_residual_load=(
            1.0 if recovery_surprise_units is not None else None
        ),
        recovery_surprise_units=recovery_surprise_units,
        decision_start_date=generated_at.date(),
        days_since_long_run=days_since_long_run,
        long_cadence_reference_days=7.0,
        cadence_exception_reasons=cadence_exception_reasons,
        trigger=trigger,
    )


def test_hard_horizon_gate_uses_observed_boundary_session_size() -> None:
    sessions = tuple(_session(day) for day in range(12))
    trajectory = tuple((17.7, 18.7) for _ in range(21))

    ordinary_only = evaluate_replan_regressions(
        _replan(*sessions, target_trajectory=trajectory),
        ProjectionGateConfig(),
    )
    after_long_run = evaluate_replan_regressions(
        _replan(
            *sessions,
            target_trajectory=trajectory,
            boundary_session_miles=8.25,
        ),
        ProjectionGateConfig(),
    )

    assert "hard_horizon_underfunded" in {
        item.code for item in ordinary_only
    }
    assert "hard_horizon_underfunded" not in {
        item.code for item in after_long_run
    }


def test_peak_load_gate_allows_only_prescription_rounding_tolerance() -> None:
    within_resolution = evaluate_replan_regressions(
        _replan(peak_projected_continuous_mileage_rate=22.49),
        ProjectionGateConfig(),
    )
    beyond_resolution = evaluate_replan_regressions(
        _replan(peak_projected_continuous_mileage_rate=22.51),
        ProjectionGateConfig(),
    )
    after_medium_long = evaluate_replan_regressions(
        _replan(
            peak_projected_continuous_mileage_rate=22.66,
            boundary_session_miles=4.75,
        ),
        ProjectionGateConfig(),
    )

    assert "projected_load_corridor" not in {
        item.code for item in within_resolution
    }
    assert "projected_load_corridor" in {
        item.code for item in beyond_resolution
    }
    assert "projected_load_corridor" not in {
        item.code for item in after_medium_long
    }


def test_gate_records_streaks_without_treating_three_days_as_unsafe() -> None:
    gate = ProjectionGate(ProjectionGateConfig())
    gate.observe(
        _replan(_session(1), _session(2), _session(3))
    )
    report = build_projection_report(
        mode="perfect",
        seed=1,
        replan_interval_days=1,
        weeks=[],
        gate=gate,
        stopped_early=False,
    )

    assert not gate.failures
    assert report["streaks"]["maximum_in_one_live_plan"] == 3


def test_gate_rejects_a_run_selected_against_not_ready_recovery() -> None:
    failures = evaluate_replan_regressions(
        _replan(_session(1, readiness="not_ready")),
        ProjectionGateConfig(),
    )

    assert [item.code for item in failures] == ["selected_not_ready_run"]
    assert failures[0].evidence["sessions"][0]["date"] == (
        OPENING + timedelta(days=1)
    ).date().isoformat()


def test_recovery_surprise_requires_material_evidence() -> None:
    unexplained = evaluate_replan_regressions(
        _replan(_session(1), recovery_surprise_units=0.02),
        ProjectionGateConfig(),
    )
    explained = evaluate_replan_regressions(
        _replan(
            _session(1),
            recovery_surprise_units=0.20,
            material_evidence_reasons=("deviation: harder than prescribed",),
        ),
        ProjectionGateConfig(),
    )

    assert [item.code for item in unexplained] == [
        "unexpected_recovery_change"
    ]
    assert not explained


def test_key_session_cadence_uses_elapsed_gaps_and_recovery_exceptions() -> None:
    config = ProjectionGateConfig(enforce_key_session_cadence=True)
    supported = evaluate_replan_regressions(
        _replan(
            _session(2, WorkoutType.LONG),
            _session(10, WorkoutType.LONG),
            _session(18, WorkoutType.LONG),
            days_since_long_run=6.0,
        ),
        config,
    )
    overdue = evaluate_replan_regressions(
        _replan(
            _session(3, WorkoutType.LONG),
            _session(11, WorkoutType.LONG),
            _session(19, WorkoutType.LONG),
            days_since_long_run=6.0,
        ),
        config,
    )
    recovery_rejected = evaluate_replan_regressions(
        _replan(
            _session(3, WorkoutType.LONG),
            days_since_long_run=6.0,
            cadence_exception_reasons=(
                "A taxing session was replaced with aerobic running.",
            ),
        ),
        config,
    )

    assert not supported
    assert [item.code for item in overdue] == ["long_cadence_gap"]
    assert overdue[0].evidence["observed_gaps_days"][0] == 9.0
    assert not recovery_rejected


def test_key_session_cadence_anchors_recency_to_decision_boundary() -> None:
    failures = evaluate_replan_regressions(
        _replan(
            _session(2, WorkoutType.LONG),
            _session(10, WorkoutType.LONG),
            _session(18, WorkoutType.LONG),
            days_since_long_run=6.2,
        ),
        ProjectionGateConfig(enforce_key_session_cadence=True),
    )

    assert [item.code for item in failures] == ["long_cadence_gap"]
    assert failures[0].evidence["observed_gaps_days"][0] == 9.0


def test_underfunded_established_plan_rejects_unexplained_aerobic_gap() -> None:
    trajectory = tuple((17.7, 18.7) for _ in range(21))
    sessions = tuple(
        _session(day, midpoint_miles=5.8)
        for day in (1, 3, 5, 10, 12, 14, 17, 19)
    )

    failures = evaluate_replan_regressions(
        _replan(
            *sessions,
            boundary_session_miles=8.45,
            target_trajectory=trajectory,
        ),
        ProjectionGateConfig(enforce_aerobic_exposure_cadence=True),
    )

    # The completed long run is large enough to satisfy the legacy aggregate
    # boundary allowance. It must not hide an internal five-day hole while
    # the new horizon remains materially below its integrated target.
    assert "hard_horizon_underfunded" not in {item.code for item in failures}
    aerobic = [
        item for item in failures
        if item.code == "underfunded_aerobic_exposure_gap"
    ]
    assert len(aerobic) == 1
    assert aerobic[0].evidence["maximum_date_separation_days"] == 5
    assert aerobic[0].evidence["allowed_date_separation_days"] == 4


def test_aerobic_gap_gate_is_load_aware_and_accepts_recovery_exceptions() -> None:
    sparse_sessions = tuple(
        _session(day) for day in (0, 5, 10, 15, 20)
    )
    low_volume_trajectory = tuple((8.0, 9.0) for _ in range(21))
    high_volume_trajectory = tuple((17.7, 18.7) for _ in range(21))
    config = ProjectionGateConfig(enforce_aerobic_exposure_cadence=True)

    low_volume = evaluate_replan_regressions(
        _replan(
            *sparse_sessions,
            target=(8.0, 9.0),
            target_trajectory=low_volume_trajectory,
        ),
        config,
    )
    recovery_limited = evaluate_replan_regressions(
        _replan(
            *sparse_sessions,
            target_trajectory=high_volume_trajectory,
            cadence_exception_reasons=(
                "Recovery screening deferred the next aerobic exposure.",
            ),
        ),
        config,
    )

    assert "underfunded_aerobic_exposure_gap" not in {
        item.code for item in low_volume
    }
    assert "underfunded_aerobic_exposure_gap" not in {
        item.code for item in recovery_limited
    }


def test_gate_distinguishes_zero_target_from_empty_positive_plan() -> None:
    zero_target = evaluate_replan_regressions(
        _replan(target=(0.0, 0.0)),
        ProjectionGateConfig(),
    )
    empty_positive = evaluate_replan_regressions(
        _replan(target=(17.0, 18.0)),
        ProjectionGateConfig(),
    )

    assert [item.code for item in zero_target] == ["established_zero_target"]
    assert [item.code for item in empty_positive] == [
        "empty_positive_target_plan"
    ]


@pytest.mark.parametrize("ending,failed", [(15.0, True), (16.0, True), (17.0, False)])
def test_progression_uses_last_decision_not_weekly_row(ending, failed) -> None:
    gate = ProjectionGate(ProjectionGateConfig(require_capacity_progression=True))
    gate.snapshots = [
        _replan(),
        replace(
            _replan(generated_at=OPENING + timedelta(days=89)),
            capacity_reference_miles=ending,
        ),
    ]
    gate.finalize([])
    assert bool(gate.failures) is failed
    if failed:
        assert gate.failures[0].code == "no_capacity_progression"
        assert gate.failures[0].evidence["final_capacity_miles"] == ending


def test_missing_capacity_endpoint_cannot_pass_progression() -> None:
    gate = ProjectionGate(ProjectionGateConfig(require_capacity_progression=True))
    gate.snapshots = [_replan(), replace(_replan(), capacity_reference_miles=None)]
    gate.finalize([])
    assert gate.failures[0].code == "missing_capacity_endpoint"


def test_capacity_audit_reports_intermediate_drawdown() -> None:
    gate = ProjectionGate(ProjectionGateConfig())
    gate.snapshots = [
        replace(_replan(), capacity_reference_miles=value)
        for value in (16.0, 20.0, 15.0, 21.0)
    ]
    report = build_projection_report(
        mode="perfect", seed=1, replan_interval_days=1,
        weeks=[], gate=gate, stopped_early=False,
    )
    assert report["capacity_audit"] == {
        "opening_decision_miles": 16.0, "final_decision_miles": 21.0,
        "minimum_miles": 15.0, "maximum_miles": 21.0,
        "largest_drawdown_miles": 5.0, "observations": 4,
    }


def test_advisory_cadence_is_reported_without_disabling_other_gates() -> None:
    gate = ProjectionGate(ProjectionGateConfig(audit_key_session_cadence=True))
    # Enough work to fund the horizon; long exposures are intentionally late.
    sessions = tuple(
        _session(
            day, WorkoutType.LONG if day in {6, 14} else WorkoutType.EASY,
            midpoint_miles=4.75,
        )
        for day in range(0, 21, 2)
    )
    gate.observe(_replan(*sessions, days_since_long_run=6.0))
    assert any(item.code == "long_cadence_gap" for item in gate.advisories)
    assert not any(item.code == "long_cadence_gap" for item in gate.failures)
    assert not gate.failures
    gate.observe(_replan(days_since_long_run=6.0))
    assert any(item.code == "empty_positive_target_plan" for item in gate.failures)
    report = build_projection_report(
        mode="perfect", seed=1, replan_interval_days=1,
        weeks=[], gate=gate, stopped_early=False,
    )
    assert report["advisories"]
    assert report["gate_policy"]["audit_key_session_cadence"] is True
    assert report["status"] == "failed"


def test_gate_excludes_consumed_date_but_catches_future_date_move() -> None:
    gate = ProjectionGate(
        ProjectionGateConfig(enforce_stability=True),
    )
    gate.observe(_replan(_session(1), _session(3)))
    gate.observe(
        replace(
            _replan(_session(3), generated_at=OPENING + timedelta(days=1)),
            decision_start_date=(OPENING + timedelta(days=2)).date(),
        )
    )
    assert not gate.failures

    gate.observe(
        _replan(
            _session(4),
            generated_at=OPENING + timedelta(days=2),
        )
    )
    assert gate.failures[-1].code == "no_evidence_refresh_churn"
    assert gate.failures[-1].evidence["date_changes"] == [
        (OPENING + timedelta(days=3)).date().isoformat(),
        (OPENING + timedelta(days=4)).date().isoformat(),
    ]


@pytest.mark.parametrize("remove_today", [True, False])
def test_refresh_catches_unconsumed_today_date_and_distance_churn(remove_today) -> None:
    gate = ProjectionGate(ProjectionGateConfig(enforce_stability=True))
    gate.observe(_replan(_session(0), _session(3)))
    after = [_session(3)]
    if not remove_today:
        after.append(_session(0, midpoint_miles=2.0))
    gate.observe(_replan(*after, generated_at=OPENING + timedelta(hours=1)))
    expected_code = (
        "no_evidence_refresh_churn" if remove_today else "no_evidence_distance_churn"
    )
    assert any(item.code == expected_code for item in gate.failures)


def test_checking_today_does_not_drop_fourth_future_day_from_gate() -> None:
    gate = ProjectionGate(ProjectionGateConfig(enforce_stability=True))
    gate.observe(_replan(_session(4)))
    gate.observe(_replan(_session(5), generated_at=OPENING + timedelta(hours=1)))
    assert any(item.code == "no_evidence_refresh_churn" for item in gate.failures)


def test_distance_only_edit_is_reported_but_does_not_fail_stability() -> None:
    gate = ProjectionGate(
        ProjectionGateConfig(enforce_stability=True),
    )
    opening = _replan(_session(2))
    resized = _session(2)
    resized = ProjectionPlanSession(
        planned_for=resized.planned_for,
        workout_type=resized.workout_type,
        midpoint_miles=4.5,
    )

    gate.observe(opening)
    gate.observe(
        _replan(
            resized,
            generated_at=OPENING + timedelta(days=1),
        )
    )
    report = build_projection_report(
        mode="perfect",
        seed=1,
        replan_interval_days=1,
        weeks=[],
        gate=gate,
        stopped_early=False,
    )

    assert not gate.failures
    assert report["churn"]["4"]["distance_only_changed_comparisons"] == 1
    assert report["churn"]["4"]["distance_changes"] == 1


def test_material_distance_edit_fails_without_new_evidence() -> None:
    gate = ProjectionGate(
        ProjectionGateConfig(enforce_stability=True),
    )
    opening = _replan(_session(2))
    resized = _session(2)
    resized = ProjectionPlanSession(
        planned_for=resized.planned_for,
        workout_type=resized.workout_type,
        midpoint_miles=2.75,
    )

    gate.observe(opening)
    gate.observe(
        _replan(
            resized,
            generated_at=OPENING + timedelta(days=1),
        )
    )

    assert gate.failures[-1].code == "no_evidence_distance_churn"
    assert gate.failures[-1].evidence["distance_changes"] == [
        {
            "date": (OPENING + timedelta(days=2)).date().isoformat(),
            "before_midpoint_miles": 4.0,
            "after_midpoint_miles": 2.75,
            "movement_miles": 1.25,
        }
    ]


def test_same_date_workout_type_change_fails_compliant_stability() -> None:
    gate = ProjectionGate(
        ProjectionGateConfig(enforce_stability=True),
    )
    gate.observe(_replan(_session(2, WorkoutType.EASY)))
    gate.observe(
        _replan(
            _session(2, WorkoutType.INTERVALS),
            generated_at=OPENING + timedelta(days=1),
        )
    )

    assert gate.failures[-1].code == "no_evidence_refresh_churn"
    assert gate.failures[-1].evidence["date_changes"] == []
    assert gate.failures[-1].evidence["workout_type_changes"] == [
        (OPENING + timedelta(days=2)).date().isoformat()
    ]


def test_gate_labels_compliant_upload_churn_separately() -> None:
    gate = ProjectionGate(
        ProjectionGateConfig(enforce_stability=True),
    )
    gate.observe(_replan(_session(1), _session(3)))
    gate.observe(
        _replan(
            _session(2),
            generated_at=OPENING + timedelta(hours=3),
            trigger="post_upload",
        )
    )

    assert gate.failures[-1].code == "compliant_upload_schedule_churn"
    assert gate.failures[-1].evidence["transition"] == (
        "scheduled_refresh->post_upload"
    )


def test_material_evidence_permits_near_term_churn_and_is_serialized() -> None:
    gate = ProjectionGate(
        ProjectionGateConfig(enforce_stability=True),
    )
    gate.observe(_replan(_session(2)))
    gate.observe(
        _replan(
            _session(3),
            generated_at=OPENING + timedelta(days=1),
            material_evidence_reasons=(
                "overload: completed above prescribed range",
            ),
        )
    )
    report = build_projection_report(
        mode="human",
        seed=1,
        replan_interval_days=1,
        weeks=[],
        gate=gate,
        stopped_early=False,
    )

    assert not gate.failures
    assert report["replans"][1]["material_evidence_reasons"] == [
        "overload: completed above prescribed range"
    ]


def test_churn_outside_the_configured_window_does_not_fail() -> None:
    gate = ProjectionGate(
        ProjectionGateConfig(
            enforce_stability=True,
            stability_horizon_days=4,
        ),
    )
    gate.observe(_replan(_session(6)))
    gate.observe(
        _replan(
            _session(7),
            generated_at=OPENING + timedelta(days=1),
        )
    )

    assert not gate.failures


def test_fail_fast_gate_raises_on_first_bad_daily_plan() -> None:
    gate = ProjectionGate(ProjectionGateConfig(), fail_fast=True)

    with pytest.raises(ProjectionGateTriggered) as raised:
        gate.observe(_replan(_session(1, readiness="not_ready")))

    assert raised.value.failure.code == "selected_not_ready_run"
    assert len(gate.snapshots) == 1
    assert len(gate.failures) == 1


def test_projection_report_is_json_ready_and_retains_daily_timing() -> None:
    gate = ProjectionGate(ProjectionGateConfig())
    gate.observe(_replan(_session(1), _session(3)))

    report = build_projection_report(
        mode="perfect",
        seed=20260902,
        replan_interval_days=1,
        weeks=[],
        gate=gate,
        stopped_early=False,
    )

    assert report["status"] == "passed"
    assert report["daily_replan_count"] == 1
    assert report["planner_timing_seconds"] == {
        "median": 1.25,
        "maximum": 1.25,
    }
    assert report["replans"][0]["generated_at"] == OPENING.isoformat()
    assert report["replans"][0]["planned_sessions"][0][
        "workout_type"
    ] == "easy"


def test_projection_report_separates_upload_and_refresh_churn() -> None:
    gate = ProjectionGate(ProjectionGateConfig())
    gate.observe(_replan(_session(1), _session(3)))
    gate.observe(
        _replan(
            _session(2),
            generated_at=OPENING + timedelta(hours=3),
            trigger="post_upload",
        )
    )
    gate.observe(
        _replan(
            _session(2),
            generated_at=OPENING + timedelta(days=1),
        )
    )

    report = build_projection_report(
        mode="perfect",
        seed=1,
        replan_interval_days=1,
        weeks=[],
        gate=gate,
        stopped_early=False,
    )

    assert report["daily_replan_count"] == 2
    assert report["post_upload_replan_count"] == 1
    assert report["replan_count"] == 3
    assert report["churn_by_trigger"]["post_upload"]["4"][
        "comparison_count"
    ] == 1
    assert report["churn_by_trigger"]["scheduled_refresh"]["4"][
        "comparison_count"
    ] == 1


def test_report_separates_planner_and_athlete_created_streaks() -> None:
    gate = ProjectionGate(ProjectionGateConfig())
    gate.observe(
        _replan(
            _session(1),
            _session(2),
        )
    )
    week = ProjectionWeek(
        week=1,
        start_date=OPENING.date(),
        target_low_miles=17.0,
        target_high_miles=18.0,
        prescribed_low_miles=8.0,
        prescribed_high_miles=8.0,
        assumed_completed_miles=12.0,
        run_count=3,
        capacity_reference_miles=16.0,
        opening_acute_ratio=1.0,
        workouts=(),
        actual_sessions=(
            ProjectionActualSession(
                occurred_at=OPENING + timedelta(days=1),
                workout_type=WorkoutType.EASY,
                distance_miles=4.0,
                was_prescribed=True,
            ),
            ProjectionActualSession(
                occurred_at=OPENING + timedelta(days=2),
                workout_type=WorkoutType.EASY,
                distance_miles=4.0,
                was_prescribed=True,
            ),
            ProjectionActualSession(
                occurred_at=OPENING + timedelta(days=3),
                workout_type=WorkoutType.EASY,
                distance_miles=4.0,
                was_prescribed=False,
            ),
        ),
    )

    report = build_projection_report(
        mode="human",
        seed=1,
        replan_interval_days=1,
        weeks=[week],
        gate=gate,
        stopped_early=False,
    )

    assert report["streaks"] == {
        "maximum_in_one_live_plan": 2,
        "committed_prescribed": 0,
        "actual_all": 3,
        "actual_prescribed": 2,
        "actual_involving_unscheduled_athlete_run": 3,
    }
