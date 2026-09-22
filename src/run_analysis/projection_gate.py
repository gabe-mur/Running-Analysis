"""Executable regression gates for closed-loop planner projections."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from datetime import date, datetime, timedelta
from enum import Enum
from statistics import median
from typing import Any

from .adherence_projection import (
    ProjectionReplan,
    ProjectionWeek,
    summarize_replan_churn,
)


@dataclass(frozen=True, slots=True)
class ProjectionGateConfig:
    """Pass/fail policy for one simulation run.

    These are regression-fixture constraints, not universal coaching rules.
    Consecutive dates are diagnostic rather than intrinsically invalid. A
    sequence fails only through the actual recovery, readiness, load, or
    stability invariants below.
    """

    boundary_allowance_sessions: float = 1.0
    funding_rounding_tolerance_miles: float = 0.5
    peak_load_rounding_tolerance_miles: float = 0.5
    stability_horizon_days: int = 4
    distance_stability_tolerance_miles: float = 0.5
    recovery_surprise_tolerance_units: float = 0.01
    enforce_stability: bool = False
    enforce_key_session_cadence: bool = False
    require_capacity_progression: bool = False


@dataclass(frozen=True, slots=True)
class ProjectionGateFailure:
    code: str
    occurred_at: datetime
    detail: str
    evidence: dict[str, Any]


class ProjectionGateTriggered(RuntimeError):
    """Raised when a fail-fast simulation reaches its first bad replan."""

    def __init__(self, failure: ProjectionGateFailure):
        super().__init__(failure.detail)
        self.failure = failure


def _longest_planned_streak(replan: ProjectionReplan) -> int:
    return _longest_date_streak(
        {item.planned_for.date() for item in replan.planned_sessions}
    )


def _longest_date_streak(run_dates: set[date]) -> int:
    ordered_dates = sorted(run_dates)
    longest = 0
    current = 0
    previous: date | None = None
    for run_date in ordered_dates:
        current = (
            current + 1
            if previous is not None and run_date == previous + timedelta(days=1)
            else 1
        )
        longest = max(longest, current)
        previous = run_date
    return longest


def _future_schedule_changes(
    previous: ProjectionReplan,
    current: ProjectionReplan,
    *,
    horizon_days: int,
) -> tuple[list[str], list[str]]:
    """Return date-slot and workout-type changes after consumed work."""

    window_start = (
        current.decision_start_date
        if current.trigger == "post_upload"
        and current.decision_start_date is not None
        else current.generated_at.date() + timedelta(days=1)
    )
    window_end = window_start + timedelta(days=horizon_days)

    def by_date(replan: ProjectionReplan) -> dict[date, str]:
        return {
            item.planned_for.date(): item.workout_type.value
            for item in replan.planned_sessions
            if window_start <= item.planned_for.date() < window_end
        }

    before = by_date(previous)
    after = by_date(current)
    date_changes = sorted(value.isoformat() for value in set(before) ^ set(after))
    type_changes = sorted(
        value.isoformat()
        for value in set(before) & set(after)
        if before[value] != after[value]
    )
    return date_changes, type_changes


def _future_distance_changes(
    previous: ProjectionReplan,
    current: ProjectionReplan,
    *,
    horizon_days: int,
    tolerance_miles: float,
) -> list[dict[str, float | str]]:
    """Return material same-date, same-workout dose rewrites."""

    window_start = (
        current.decision_start_date
        if current.trigger == "post_upload"
        and current.decision_start_date is not None
        else current.generated_at.date() + timedelta(days=1)
    )
    window_end = window_start + timedelta(days=horizon_days)

    def by_date(replan: ProjectionReplan) -> dict[date, Any]:
        return {
            item.planned_for.date(): item
            for item in replan.planned_sessions
            if window_start <= item.planned_for.date() < window_end
        }

    before = by_date(previous)
    after = by_date(current)
    changes: list[dict[str, float | str]] = []
    for plan_date in sorted(set(before) & set(after)):
        old = before[plan_date]
        new = after[plan_date]
        if old.workout_type != new.workout_type:
            continue
        movement = abs(new.midpoint_miles - old.midpoint_miles)
        if movement <= max(0.0, tolerance_miles) + 1e-9:
            continue
        changes.append(
            {
                "date": plan_date.isoformat(),
                "before_midpoint_miles": old.midpoint_miles,
                "after_midpoint_miles": new.midpoint_miles,
                "movement_miles": movement,
            }
        )
    return changes


def evaluate_replan_regressions(
    replan: ProjectionReplan,
    config: ProjectionGateConfig,
) -> tuple[ProjectionGateFailure, ...]:
    """Evaluate invariants that are knowable from one daily planner output."""

    failures: list[ProjectionGateFailure] = []
    target = (replan.target_low_miles, replan.target_high_miles)
    if (
        replan.planning_mode == "established"
        and (replan.capacity_reference_miles or 0.0) > 0
        and replan.target_high_miles <= 0
    ):
        failures.append(
            ProjectionGateFailure(
                code="established_zero_target",
                occurred_at=replan.generated_at,
                detail="Established athlete received a zero mileage target.",
                evidence={
                    "target": target,
                    "capacity_reference_miles": replan.capacity_reference_miles,
                },
            )
        )
    if replan.target_low_miles > 0 and not replan.planned_sessions:
        failures.append(
            ProjectionGateFailure(
                code="empty_positive_target_plan",
                occurred_at=replan.generated_at,
                detail="Positive target produced an empty 21-day plan.",
                evidence={"target": target},
            )
        )

    not_ready_sessions = [
        item
        for item in replan.planned_sessions
        if item.readiness == "not_ready"
    ]
    if not_ready_sessions:
        failures.append(
            ProjectionGateFailure(
                code="selected_not_ready_run",
                occurred_at=replan.generated_at,
                detail=(
                    "Planner selected running on a date its own recovery "
                    "model marked not ready."
                ),
                evidence={
                    "sessions": [
                        {
                            "date": item.planned_for.date().isoformat(),
                            "workout_type": item.workout_type.value,
                            "readiness": item.readiness,
                        }
                        for item in not_ready_sessions
                    ],
                },
            )
        )

    ordinary = replan.ordinary_easy_midpoint_miles
    peak = replan.peak_projected_continuous_mileage_rate
    if ordinary is not None and peak is not None:
        boundary_session = max(
            ordinary,
            replan.boundary_session_miles or 0.0,
        )
        allowed_peak = (
            replan.target_high_miles
            + boundary_session
            * max(0.0, config.boundary_allowance_sessions)
        )
        if peak > (
            allowed_peak
            + max(0.0, config.peak_load_rounding_tolerance_miles)
            + 1e-9
        ):
            failures.append(
                ProjectionGateFailure(
                    code="projected_load_corridor",
                    occurred_at=replan.generated_at,
                    detail=(
                        f"Projected continuous load {peak:.1f} exceeds the "
                        f"{allowed_peak:.1f} target-plus-boundary corridor."
                    ),
                    evidence={
                        "peak_projected_continuous_mileage_rate": peak,
                        "target": target,
                        "ordinary_easy_midpoint_miles": ordinary,
                        "boundary_session_miles": boundary_session,
                        "allowed_peak": allowed_peak,
                        "rounding_tolerance_miles": (
                            config.peak_load_rounding_tolerance_miles
                        ),
                    },
                )
            )

    if replan.target_trajectory and ordinary is not None:
        integrated_low = sum(item[0] for item in replan.target_trajectory) / 7.0
        integrated_high = sum(item[1] for item in replan.target_trajectory) / 7.0
        planned_miles = sum(item.midpoint_miles for item in replan.planned_sessions)
        boundary_session = max(
            ordinary,
            replan.boundary_session_miles or 0.0,
        )
        allowance = boundary_session * max(
            0.0, config.boundary_allowance_sessions
        )
        if planned_miles < (
            integrated_low
            - allowance
            - max(0.0, config.funding_rounding_tolerance_miles)
            - 1e-9
        ):
            failures.append(
                ProjectionGateFailure(
                    code="hard_horizon_underfunded",
                    occurred_at=replan.generated_at,
                    detail=(
                        f"Plan funds {planned_miles:.1f} miles against a "
                        f"{integrated_low:.1f}-mile integrated minimum."
                    ),
                    evidence={
                        "planned_midpoint_miles": planned_miles,
                        "integrated_target": (integrated_low, integrated_high),
                        "boundary_allowance_miles": allowance,
                        "rounding_tolerance_miles": (
                            config.funding_rounding_tolerance_miles
                        ),
                    },
                )
            )
    recovery_surprise = replan.recovery_surprise_units
    if (
        recovery_surprise is not None
        and not replan.material_evidence_reasons
        and abs(recovery_surprise)
        > max(0.0, config.recovery_surprise_tolerance_units)
    ):
        failures.append(
            ProjectionGateFailure(
                code="unexpected_recovery_change",
                occurred_at=replan.generated_at,
                detail=(
                    "Compliant execution changed opening recovery load by "
                    f"{recovery_surprise:+.3f} units versus the pre-upload "
                    "projection."
                ),
                evidence={
                    "expected_opening_recovery_residual_load": (
                        replan.expected_opening_recovery_residual_load
                    ),
                    "opening_recovery_residual_load": (
                        replan.opening_recovery_residual_load
                    ),
                    "recovery_surprise_units": recovery_surprise,
                    "tolerance_units": (
                        config.recovery_surprise_tolerance_units
                    ),
                },
            )
        )
    if (
        config.enforce_key_session_cadence
        and replan.planning_mode == "established"
        and replan.decision_start_date is not None
        and not replan.cadence_exception_reasons
    ):
        cadence_lanes = (
            (
                "long",
                {"long"},
                replan.days_since_long_run,
                replan.long_cadence_reference_days,
            ),
            (
                "quality",
                {"intervals", "tempo_threshold", "race"},
                replan.days_since_quality_run,
                replan.quality_cadence_reference_days,
            ),
        )
        for lane, workout_types, opening_age, reference in cadence_lanes:
            if opening_age is None or reference is None:
                continue
            allowed_gap = max(1.0, reference + 1.0)
            offsets = sorted(
                (
                    item.planned_for.date()
                    - replan.decision_start_date
                ).days
                for item in replan.planned_sessions
                if item.workout_type.value in workout_types
            )
            observed_gaps: list[float] = []
            if offsets:
                observed_gaps.append(opening_age + offsets[0])
                observed_gaps.extend(
                    float(current - previous)
                    for previous, current in zip(offsets, offsets[1:])
                )
                observed_gaps.append(float(20 - offsets[-1]))
            else:
                observed_gaps.append(opening_age + 20.0)
            maximum_gap = max(observed_gaps)
            if maximum_gap <= allowed_gap + 1e-9:
                continue
            failures.append(
                ProjectionGateFailure(
                    code=f"{lane}_cadence_gap",
                    occurred_at=replan.generated_at,
                    detail=(
                        f"{lane.title()}-session cadence reaches "
                        f"{maximum_gap:.1f} days; allowed fixture gap is "
                        f"{allowed_gap:.1f} days without a recovery exception."
                    ),
                    evidence={
                        "lane": lane,
                        "opening_age_days": opening_age,
                        "planned_offsets": offsets,
                        "observed_gaps_days": observed_gaps,
                        "allowed_gap_days": allowed_gap,
                    },
                )
            )
    return tuple(failures)


class ProjectionGate:
    """Collect daily replans and optionally stop at the first regression."""

    def __init__(
        self,
        config: ProjectionGateConfig,
        *,
        fail_fast: bool = False,
    ) -> None:
        self.config = config
        self.fail_fast = fail_fast
        self.snapshots: list[ProjectionReplan] = []
        self.failures: list[ProjectionGateFailure] = []

    def observe(self, replan: ProjectionReplan) -> None:
        previous = self.snapshots[-1] if self.snapshots else None
        self.snapshots.append(replan)
        new_failures = list(evaluate_replan_regressions(replan, self.config))
        if previous is not None and self.config.enforce_stability:
            date_changes, type_changes = _future_schedule_changes(
                previous,
                replan,
                horizon_days=self.config.stability_horizon_days,
            )
            if (
                date_changes or type_changes
            ) and not replan.material_evidence_reasons:
                new_failures.append(
                    ProjectionGateFailure(
                        code=(
                            "compliant_upload_schedule_churn"
                            if replan.trigger == "post_upload"
                            else "no_evidence_refresh_churn"
                        ),
                        occurred_at=replan.generated_at,
                        detail=(
                            "Compliant upload changed near-term dates or "
                            "workout types."
                            if replan.trigger == "post_upload"
                            else
                            "No-new-evidence refresh changed near-term dates "
                            "or workout types."
                        ),
                        evidence={
                            "transition": (
                                f"{previous.trigger}->{replan.trigger}"
                            ),
                            "horizon_days": self.config.stability_horizon_days,
                            "date_changes": date_changes,
                            "workout_type_changes": type_changes,
                            "material_evidence_reasons": list(
                                replan.material_evidence_reasons
                            ),
                        },
                    )
                )
            distance_changes = _future_distance_changes(
                previous,
                replan,
                horizon_days=self.config.stability_horizon_days,
                tolerance_miles=(
                    self.config.distance_stability_tolerance_miles
                ),
            )
            if distance_changes and not replan.material_evidence_reasons:
                new_failures.append(
                    ProjectionGateFailure(
                        code=(
                            "compliant_upload_distance_churn"
                            if replan.trigger == "post_upload"
                            else "no_evidence_distance_churn"
                        ),
                        occurred_at=replan.generated_at,
                        detail=(
                            "Compliant upload materially changed near-term "
                            "run distance."
                            if replan.trigger == "post_upload"
                            else "No-new-evidence refresh materially changed "
                            "near-term run distance."
                        ),
                        evidence={
                            "transition": (
                                f"{previous.trigger}->{replan.trigger}"
                            ),
                            "horizon_days": (
                                self.config.stability_horizon_days
                            ),
                            "distance_tolerance_miles": (
                                self.config.distance_stability_tolerance_miles
                            ),
                            "distance_changes": distance_changes,
                            "material_evidence_reasons": list(
                                replan.material_evidence_reasons
                            ),
                        },
                    )
                )
        self.failures.extend(new_failures)
        if self.fail_fast and new_failures:
            raise ProjectionGateTriggered(new_failures[0])

    def finalize(self, weeks: list[ProjectionWeek]) -> None:
        if not self.config.require_capacity_progression or not weeks:
            return
        opening = weeks[0].capacity_reference_miles
        final = weeks[-1].capacity_reference_miles
        if final <= opening + 1e-9:
            failure = ProjectionGateFailure(
                code="no_capacity_progression",
                occurred_at=self.snapshots[-1].generated_at,
                detail=(
                    f"Capacity did not progress across the projection: "
                    f"{opening:.1f} to {final:.1f} miles/week."
                ),
                evidence={
                    "opening_capacity_miles": opening,
                    "final_capacity_miles": final,
                },
            )
            self.failures.append(failure)
            if self.fail_fast:
                raise ProjectionGateTriggered(failure)


def _json_ready(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_ready(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    return value


def build_projection_report(
    *,
    mode: str,
    seed: int,
    replan_interval_days: int,
    weeks: list[ProjectionWeek],
    gate: ProjectionGate,
    stopped_early: bool,
) -> dict[str, Any]:
    """Return one stable JSON-ready projection artifact."""

    timings = [
        item.planning_seconds
        for item in gate.snapshots
        if item.planning_seconds is not None
    ]
    churn = {
        str(horizon): _json_ready(
            summarize_replan_churn(gate.snapshots, horizon_days=horizon)
        )
        for horizon in (3, 4, 7)
    }
    churn_by_trigger = {
        trigger: {
            str(horizon): _json_ready(
                summarize_replan_churn(
                    gate.snapshots,
                    horizon_days=horizon,
                    transition_trigger=trigger,
                )
            )
            for horizon in (3, 4, 7)
        }
        for trigger in ("post_upload", "scheduled_refresh")
    }
    scheduled_refresh_count = sum(
        item.trigger == "scheduled_refresh" for item in gate.snapshots
    )
    post_upload_replan_count = sum(
        item.trigger == "post_upload" for item in gate.snapshots
    )
    planned_streak = max(
        (_longest_planned_streak(item) for item in gate.snapshots),
        default=0,
    )
    committed_dates = {
        session.planned_for.date()
        for item in gate.snapshots
        for session in item.committed_sessions
    }
    actual_sessions = [
        session for week in weeks for session in week.actual_sessions
    ]
    actual_dates = {session.occurred_at.date() for session in actual_sessions}
    prescribed_actual_dates = {
        session.occurred_at.date()
        for session in actual_sessions
        if session.was_prescribed
    }
    unscheduled_dates = {
        session.occurred_at.date()
        for session in actual_sessions
        if not session.was_prescribed
    }
    athlete_involved_streak = 0
    if unscheduled_dates:
        for run_date in actual_dates:
            if run_date not in unscheduled_dates:
                continue
            streak_dates = {run_date}
            earlier = run_date - timedelta(days=1)
            while earlier in actual_dates:
                streak_dates.add(earlier)
                earlier -= timedelta(days=1)
            later = run_date + timedelta(days=1)
            while later in actual_dates:
                streak_dates.add(later)
                later += timedelta(days=1)
            athlete_involved_streak = max(
                athlete_involved_streak,
                len(streak_dates),
            )
    return {
        "status": "failed" if gate.failures else "passed",
        "stopped_early": stopped_early,
        "mode": mode,
        "seed": seed,
        "replan_interval_days": replan_interval_days,
        "daily_replan_count": scheduled_refresh_count,
        "post_upload_replan_count": post_upload_replan_count,
        "replan_count": len(gate.snapshots),
        "failures": _json_ready(gate.failures),
        "planner_timing_seconds": {
            "median": median(timings) if timings else None,
            "maximum": max(timings) if timings else None,
        },
        "churn": churn,
        "churn_by_trigger": churn_by_trigger,
        "streaks": {
            "maximum_in_one_live_plan": planned_streak,
            "committed_prescribed": _longest_date_streak(committed_dates),
            "actual_all": _longest_date_streak(actual_dates),
            "actual_prescribed": _longest_date_streak(
                prescribed_actual_dates
            ),
            "actual_involving_unscheduled_athlete_run": (
                athlete_involved_streak
            ),
        },
        "replans": _json_ready(gate.snapshots),
        "weeks": _json_ready(weeks),
    }
