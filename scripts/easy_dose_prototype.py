"""Offline, history-led easy-dose candidates. Not imported by the application.

This generates training intentions, not clearance to run them. Recovery and
whole-program selection are deliberately separate; none of the options is a
prescription until those downstream checks have been applied.
"""

from dataclasses import asdict, dataclass
from datetime import datetime
from math import ceil, floor, isfinite

from run_analysis.easy_baseline import (
    ordinary_easy_sample_distance, recency_weighted_easy_distance,
)


@dataclass(frozen=True)
class Observation:
    activity_id: str
    occurred_at: datetime
    actual_miles: float
    moving_minutes: float | None = None
    workout_type: str = "easy"
    health: str = "normal"
    planning_role: str | None = None
    prescribed_range: tuple[float, float] | None = None
    perceived_exertion: float | None = None
    # An explicit separate observation, never inferred from completion or RPE.
    recovery_response: str | None = None
    projected: bool = False

    def __post_init__(self):
        if self.occurred_at.utcoffset() is None:
            raise ValueError("Observation needs a timezone")
        if not isfinite(self.actual_miles) or self.actual_miles <= 0:
            raise ValueError("Actual distance must be positive and finite")
        if self.prescribed_range is not None:
            low, high = self.prescribed_range
            if not (isfinite(low) and isfinite(high) and 0 < low <= high):
                raise ValueError("Invalid prescription range")


@dataclass(frozen=True)
class HistoryProfile:
    as_of: datetime
    half_life_days: float
    ordinary_planning_miles: float | None
    ordinary_actual_miles: float | None
    support_planning_miles: float | None
    ordinary_count: int
    support_count: int
    recovery_response_count: int
    newest_observation: datetime | None
    observations: tuple[Observation, ...]
    ordinary_sources: tuple[str, ...]
    exclusions: tuple[tuple[str, str], ...]


def build_profile(observations, as_of: datetime, *, half_life_days: float) -> HistoryProfile:
    """Keep raw observations separate from the existing compliance-normalized reference.

    Existing half-life and role eligibility are retained as explicit provisional
    assumptions. No statistical confidence or physiological tolerance is inferred.
    A weighted median is robust to some outliers, but not guaranteed continuous
    when new observations arrive; upload tests remain necessary.
    """
    if as_of.utcoffset() is None or not isfinite(half_life_days) or half_life_days < 1:
        raise ValueError("Require timezone-aware boundary and half-life >= 1 day")
    seen, retained, sources, excluded = set(), [], [], []
    ordinary, actual, support = [], [], []
    for item in sorted(observations, key=lambda x: (x.occurred_at, x.activity_id)):
        if item.activity_id in seen:
            raise ValueError(f"Duplicate activity {item.activity_id}")
        seen.add(item.activity_id)
        reason = (
            "projected_not_observed" if item.projected else
            "after_decision_boundary" if item.occurred_at > as_of else
            "non_easy_workout" if item.workout_type != "easy" else
            "health_caution" if item.health != "normal" else None
        )
        if reason:
            excluded.append((item.activity_id, reason))
            continue
        retained.append(item)
        sample = ordinary_easy_sample_distance(
            item.actual_miles,
            "ordinary_easy" if item.planning_role == "medium_long" else item.planning_role,
            item.prescribed_range,
        )
        if sample is not None:
            ordinary.append((item.occurred_at, sample))
            actual.append((item.occurred_at, item.actual_miles))
            sources.append(item.activity_id)
        elif item.planning_role == "support_easy":
            # Same compliance normalization, but never teaches ordinary size.
            support.append((item.occurred_at, ordinary_easy_sample_distance(
                item.actual_miles, "ordinary_easy", item.prescribed_range,
            )))
        else:
            excluded.append((item.activity_id, "role_not_ordinary_or_support"))

    def median(samples):
        return recency_weighted_easy_distance(samples, as_of, half_life_days=half_life_days)

    return HistoryProfile(
        as_of=as_of, half_life_days=half_life_days,
        ordinary_planning_miles=median(ordinary), ordinary_actual_miles=median(actual),
        support_planning_miles=median(support), ordinary_count=len(ordinary),
        support_count=len(support),
        recovery_response_count=sum(x.recovery_response is not None for x in retained),
        newest_observation=max((x.occurred_at for x in retained), default=None),
        observations=tuple(retained), ordinary_sources=tuple(sources), exclusions=tuple(excluded),
    )


@dataclass(frozen=True)
class DoseIntent:
    midpoint_miles: float
    execution_range: tuple[float, float]
    purposes: tuple[str, ...]
    sources: tuple[str, ...]


@dataclass(frozen=True)
class IntentMenu:
    ordinary_miles: float | None
    options: tuple[DoseIntent, ...]
    notes: tuple[str, ...]


def generate_intents(
    profile: HistoryProfile, *, useful_floor_miles: float,
    progression_allowance_miles: float, progression_source: str,
    prior_range: tuple[float, float] | None = None,
    quantum_miles: float = .25, execution_margin_miles: float = .25,
) -> IntentMenu:
    """Generate a connected distance menu independently of current recovery.

    The caller must supply a useful-dose floor and a program progression
    allowance with provenance. Neither is learned from additional rest or
    remaining mileage shortfall. Resolution/margin reuse current display
    conventions, not physiological thresholds. Retaining a prior option does
    not make it preferred, feasible or a new progression reference.
    """
    if not all(isfinite(x) for x in (
        useful_floor_miles, progression_allowance_miles, quantum_miles, execution_margin_miles,
    )) or useful_floor_miles <= 0 or progression_allowance_miles < 0 or quantum_miles <= 0 or execution_margin_miles < 0:
        raise ValueError("Invalid dose-generation input")
    if not progression_source.strip():
        raise ValueError("Progression needs an explicit source, including when disabled")
    ordinary = profile.ordinary_planning_miles
    if ordinary is None:
        return IntentMenu(None, (), ("Insufficient ordinary history; use the existing baseline-acquisition workflow.",))
    lower = min(ordinary, useful_floor_miles)
    upper = ordinary + progression_allowance_miles
    if not isfinite(upper) or (upper - lower) / quantum_miles > 95:
        raise ValueError("Offline menu exceeds bounded search budget")
    centers = {ordinary, lower, upper}
    centers.update(i * quantum_miles for i in range(
        ceil(lower / quantum_miles), floor(upper / quantum_miles) + 1,
    ))
    notes = ["Options are intentions, not recovery-cleared prescriptions."]
    support = profile.support_planning_miles
    if support is not None and lower <= support <= ordinary:
        centers.add(support)
    elif support is None:
        notes.append("No support-run history; shorter options use the supplied legacy useful-dose floor.")
    if profile.recovery_response_count == 0:
        notes.append("Recovery response unobserved; completion does not establish tolerance.")
    prior_mid = None
    if prior_range is not None:
        low, high = prior_range
        if not (isfinite(low) and isfinite(high) and 0 < low <= high):
            raise ValueError("Invalid prior range")
        prior_mid = (low + high) / 2
        centers.add(prior_mid)
    # A malformed caller budget must not silently produce an unbounded search.
    if len(centers) > 100:
        raise ValueError("Offline menu exceeds 100 options; inspect the supplied budget")
    options = []
    for center in sorted(centers):
        purposes, sources = [], []
        if center < ordinary - 1e-9 and center >= lower - 1e-9:
            purposes.append("shorter_useful")
            sources.append("supplied_useful_floor")
        elif abs(center - ordinary) < 1e-9:
            purposes.append("ordinary")
            sources.append("compliance_normalized_observed_history")
        elif ordinary < center <= upper + 1e-9:
            purposes.append("progression")
            sources.append(progression_source)
        if support is not None and abs(center - support) < 1e-9:
            sources.append("observed_support_session")
        is_prior = prior_mid is not None and abs(center - prior_mid) < 1e-9
        if is_prior:
            purposes.append("prior_prescription")
            sources.append("prior_plan_not_new_capacity_evidence")
        margin = min(execution_margin_miles, center / 2)
        execution = prior_range if is_prior else (center - margin, center + margin)
        options.append(DoseIntent(center, execution, tuple(purposes), tuple(sources)))
    return IntentMenu(ordinary, tuple(options), tuple(notes))


def restrict_intents(menu: IntentMenu, *, maximum_midpoint_miles: float | None,
                     reason: str | None = None, no_run: bool = False):
    """Separate evidence-based screening; never calls waiting a training benefit.

    Diagnostic primitive, NOT the complete production recovery validator. Does
    not clip an excluded dose into a new dose or manufacture a tiny filler run.
    """
    if maximum_midpoint_miles is not None and (
        not isfinite(maximum_midpoint_miles) or maximum_midpoint_miles < 0
    ):
        raise ValueError("Invalid maximum")
    if (maximum_midpoint_miles is not None or no_run) and not reason:
        raise ValueError("A restriction needs an explanation")
    eligible, rejected = [], []
    for option in menu.options:
        if no_run or (maximum_midpoint_miles is not None and option.midpoint_miles > maximum_midpoint_miles + 1e-9):
            rejected.append({"option": asdict(option), "reason": reason})
        else:
            eligible.append(option)
    return tuple(eligible), tuple(rejected)
