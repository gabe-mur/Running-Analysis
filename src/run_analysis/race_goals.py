"""Inspectable race-goal guardrails derived from recent running evidence."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from math import ceil
from statistics import median
import sqlite3


@dataclass(frozen=True, slots=True)
class RaceGoalProfile:
    label: str
    distance_miles: float
    ready_minimum_weeks: int
    development_weeks: int
    absolute_fastest_pace: float
    prediction_penalty_minutes: float
    long_run_bias: float
    quality_bias: float
    taper_days: int
    preferred_quality: tuple[str, ...]


RACE_GOALS: dict[str, RaceGoalProfile] = {
    "5k": RaceGoalProfile("5K", 3.10686, 2, 9, 4.0, 0.0, 0.0, 1.5, 7, ("short_intervals", "long_intervals", "threshold")),
    "10k": RaceGoalProfile("10K", 6.21371, 3, 12, 4.15, 0.0, 0.25, 1.25, 7, ("long_intervals", "threshold", "short_intervals")),
    "half_marathon": RaceGoalProfile("Half marathon", 13.1094, 4, 10, 4.3, 0.0, 1.25, 0.75, 10, ("threshold", "progression", "long_intervals")),
    "marathon": RaceGoalProfile("Marathon", 26.2188, 8, 18, 4.45, 10.0, 2.0, 0.5, 14, ("progression", "threshold", "long_intervals")),
}


def format_pace(pace_min_mile: float) -> str:
    """Render decimal minutes without ever producing a ``:60`` value."""
    total_seconds = round(pace_min_mile * 60)
    minutes, seconds = divmod(total_seconds, 60)
    return f"{minutes}:{seconds:02d}/mi"


@dataclass(frozen=True, slots=True)
class RaceGoalAssessment:
    goal: str
    race_date: date
    goal_pace_min_mile: float
    evidence_runs: int
    recent_fast_training_pace: float
    supported_goal_pace: float
    fastest_allowed_goal_pace: float
    weeks_remaining: float
    minimum_weeks: int
    rationale: str


def configured_race_goal(config: dict, *, on_date: date | None = None) -> tuple[RaceGoalProfile, date, float] | None:
    coaching = config.get("coaching", {})
    goal = str(coaching.get("training_goal", "general_fitness"))
    if goal not in RACE_GOALS:
        return None
    raw_date = coaching.get("goal_date")
    raw_pace = coaching.get("goal_pace_min_mile")
    if not raw_date or raw_pace is None:
        return None
    race_date = date.fromisoformat(str(raw_date))
    if on_date is not None and race_date < on_date:
        return None
    return RACE_GOALS[goal], race_date, float(raw_pace)


def _recent_training_paces(connection: sqlite3.Connection) -> list[tuple[float, float]]:
    rows = connection.execute(
        """
        SELECT am.moving_pace_min_mile, am.analysis_distance_m
        FROM activities a
        JOIN activity_metrics am ON am.activity_id=a.id
        LEFT JOIN run_overrides ro ON ro.activity_id=a.activity_id
        WHERE a.sport='Running'
          AND am.moving_pace_min_mile BETWEEN 3 AND 25
          AND am.analysis_distance_m >= 2414.016
          AND COALESCE(ro.include_in_model, 1) != 0
          AND COALESCE(ro.workout_type, 'easy') NOT IN ('hike', 'bike', 'run_walk')
          AND COALESCE(ro.health_tag, 'normal') = 'normal'
        ORDER BY a.start_time_utc_epoch DESC
        LIMIT 10
        """
    ).fetchall()
    return [(float(row[0]), float(row[1]) / 1609.344) for row in rows]


def assess_race_goal(
    connection: sqlite3.Connection | None,
    config: dict,
    *,
    as_of: date | None = None,
) -> RaceGoalAssessment | None:
    today = as_of or date.today()
    configured = configured_race_goal(config)
    if configured is None:
        return None
    profile, race_date, goal_pace = configured
    if race_date <= today:
        raise ValueError("Goal date must be in the future")
    days_remaining = (race_date - today).days
    if days_remaining > 366:
        raise ValueError("Choose a goal date within the next 12 months so the plan can remain responsive")
    if goal_pace < profile.absolute_fastest_pace or goal_pace > 20:
        raise ValueError(
            f"{profile.label} goal pace must be between {profile.absolute_fastest_pace:.2f} and 20.00 min/mi"
        )
    if connection is None:
        raise ValueError("Import and process at least 10 usable runs before setting a race goal")
    performances = _recent_training_paces(connection)
    if len(performances) < 10:
        raise ValueError(
            f"A race goal requires 10 usable normal-health runs; {len(performances)} are currently available"
        )
    fast_training_pace = median(sorted(pace for pace, _ in performances)[:3])
    equivalent_paces = []
    for pace, distance in performances:
        source_time_minutes = pace * distance
        target_time_minutes = source_time_minutes * (profile.distance_miles / distance) ** 1.06
        target_time_minutes += profile.prediction_penalty_minutes
        equivalent_paces.append(target_time_minutes / profile.distance_miles)
    supported = median(sorted(equivalent_paces)[:3])
    weeks = days_remaining / 7.0
    ambitious = goal_pace < supported * 0.99
    minimum_weeks = profile.development_weeks if ambitious else profile.ready_minimum_weeks
    if weeks < minimum_weeks:
        earliest = today + timedelta(weeks=minimum_weeks)
        mode = "development" if ambitious else "race-specific preparation"
        raise ValueError(
            f"The {profile.label} goal needs at least {minimum_weeks} weeks of {mode}; choose {earliest.isoformat()} or later"
        )
    # A developing goal must leave room for improvement rather than requiring
    # the athlete to be race-ready before training begins. This is a disclosed
    # planning guardrail, not a promised adaptation rate: one quarter-percent
    # per available week, capped at eight percent.
    improvement_allowance = min(0.08, weeks * 0.0025)
    fastest_allowed = supported * (1.0 - improvement_allowance)
    if goal_pace < fastest_allowed:
        required_improvement = 1.0 - goal_pace / supported
        if required_improvement <= 0.08:
            required_weeks = max(
                profile.development_weeks,
                ceil(required_improvement / 0.0025),
            )
            timing = (
                f", or move the race to {today + timedelta(weeks=required_weeks):%Y-%m-%d} or later"
            )
        else:
            timing = "; a later date is not enough until newer runs support a faster baseline"
        raise ValueError(
            f"The latest 10 runs support roughly {supported:.2f} min/mi for this goal; "
            f"for this date choose {format_pace(fastest_allowed)} or slower{timing}"
        )
    longest = max(distance for _, distance in performances)
    if profile.label == "Marathon" and longest < 6 and weeks < 26:
        raise ValueError(
            "A marathon inside 26 weeks requires at least one recent 6-mile run; build durability before setting this date"
        )
    return RaceGoalAssessment(
        goal=str(config["coaching"]["training_goal"]),
        race_date=race_date,
        goal_pace_min_mile=goal_pace,
        evidence_runs=len(performances),
        recent_fast_training_pace=fast_training_pace,
        supported_goal_pace=supported,
        fastest_allowed_goal_pace=fastest_allowed,
        weeks_remaining=weeks,
        minimum_weeks=minimum_weeks,
        rationale=(
            f"Validated from Riegel-equivalent performances across the latest 10 usable runs; "
            f"{profile.label} composition will influence quality, long-run, and taper scoring without overriding load or health guardrails."
        ),
    )
