from __future__ import annotations

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
):
    return ProjectionPlanSession(
        planned_for=OPENING + timedelta(days=day),
        workout_type=workout_type,
        midpoint_miles=4.0,
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


def test_gate_excludes_consumed_date_but_catches_future_date_move() -> None:
    gate = ProjectionGate(
        ProjectionGateConfig(enforce_stability=True),
    )
    gate.observe(_replan(_session(1), _session(3)))
    gate.observe(
        _replan(
            _session(3),
            generated_at=OPENING + timedelta(days=1),
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
