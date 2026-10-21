"""Small acceptance fixture from the stopped four-pass synthetic replay.

No optimizer invocation or saved output file is required. These checks preserve
the existing gate contract while candidate eligibility is being investigated.
They are not a universal maximum-gap policy or an integration fix.
"""

from dataclasses import replace
from datetime import datetime, timedelta, timezone

from run_analysis.adherence_projection import ProjectionPlanSession, ProjectionReplan
from run_analysis.projection_gate import ProjectionGateConfig, evaluate_replan_regressions
from run_analysis.web.schemas import WorkoutType
from run_analysis.weekly_schedule import _has_underfunded_aerobic_exposure_gap


ORIGIN = datetime(2026, 9, 24, 7, tzinfo=timezone.utc)
OFFSETS = (0, 2, 4, 10, 12, 15, 16, 18, 20)
MILES = (3.95, 4.5, 7.95, 4.5, 7.95, 3.95, 3.95, 3.75, 7.95)
TYPES = ("easy", "intervals", "long", "tempo_threshold", "long", "easy", "easy", "tempo_threshold", "long")
TARGETS = ((16.5, 17.5), (16.5, 17.5), (16.5, 17.5), (16.2, 17.3),
           (16.2, 17.3), (16.5, 17.5), (16.5, 17.5), (17.2, 18.3),
           (17.4, 18.4), (17.4, 18.5), (17.3, 18.4), (17.4, 18.4),
           (17.4, 18.5), (17.4, 18.4), (17.4, 18.4), (17.4, 18.5),
           (17.4, 18.4), (17.4, 18.5), (17.4, 18.5), (17.4, 18.4), (17.4, 18.5))


def gap_fixture():
    return ProjectionReplan(
        generated_at=datetime(2026, 9, 23, 7, 43, 27, tzinfo=timezone.utc),
        opening_load_ratio=None, target_low_miles=16.5, target_high_miles=17.5,
        planned_sessions=tuple(ProjectionPlanSession(
            planned_for=ORIGIN + timedelta(days=offset), workout_type=WorkoutType(kind),
            midpoint_miles=miles, readiness="ready")
            for offset, miles, kind in zip(OFFSETS, MILES, TYPES, strict=True)),
        committed_sessions=(), target_trajectory=TARGETS,
        planning_mode="established", ordinary_easy_midpoint_miles=3.75,
        decision_start_date=ORIGIN.date())


def gap_failures(snapshot):
    return [failure for failure in evaluate_replan_regressions(
        snapshot, ProjectionGateConfig(enforce_aerobic_exposure_cadence=True))
        if failure.code == "underfunded_aerobic_exposure_gap"]


def legacy_gap(snapshot, offsets=OFFSETS):
    return _has_underfunded_aerobic_exposure_gap(
        offsets, planned_midpoint_miles=sum(s.midpoint_miles for s in snapshot.planned_sessions),
        integrated_target_low_miles=sum(low for low, _ in snapshot.target_trajectory) / 7,
        ordinary_easy_midpoint=3.5,
        target_distance_range=(snapshot.target_low_miles, snapshot.target_high_miles))


def test_actual_failure_is_rejected_by_both_gap_calculations():
    snapshot = gap_fixture()
    assert legacy_gap(snapshot)
    failure, = gap_failures(snapshot)
    assert failure.evidence["maximum_date_separation_days"] == 6
    assert failure.evidence["allowed_date_separation_days"] == 4


def test_already_scored_funded_alternative_passes_gap_contract():
    snapshot = gap_fixture()
    restored = ProjectionPlanSession(planned_for=ORIGIN + timedelta(days=7),
        workout_type=WorkoutType.EASY, midpoint_miles=4.25, readiness="ready")
    snapshot = replace(snapshot, planned_sessions=tuple(sorted(
        (*snapshot.planned_sessions, restored), key=lambda x: x.planned_for)))
    assert not gap_failures(snapshot)
    assert not legacy_gap(snapshot, tuple(sorted((*OFFSETS, 7))))


def test_gate_preserves_recovery_exception_and_non_established_scope():
    snapshot = gap_fixture()
    assert not gap_failures(replace(snapshot,
        cadence_exception_reasons=("Recovery guardrail delayed a session.",)))
    assert not gap_failures(replace(snapshot, planning_mode="baseline_building"))


def test_gate_uses_full_target_trajectory_not_only_opening_target():
    # Synthetic input contrast, not a second real-planner reproduction.
    snapshot = replace(gap_fixture(), target_low_miles=9, target_high_miles=10)
    assert not legacy_gap(snapshot)
    assert gap_failures(snapshot)


def test_gate_uses_actual_sessions_not_requested_dates():
    snapshot = gap_fixture()
    # A requested date with no actual run must not hide the real six-day gap.
    assert not legacy_gap(snapshot, tuple(sorted((*OFFSETS, 8))))
    assert gap_failures(snapshot)
