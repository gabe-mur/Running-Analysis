"""Existing coaching inputs for the shared V7 progression controller.

No new rates, fit, scoring weights, or preparation destinations live here.
Race preparation changes inputs, not the volume-credit algorithm.
"""
from dataclasses import dataclass, replace

from run_analysis.general_progression import (destination as general_destination,
    VOLUME_GROWTH_PER_7, LONG_SHARE, LONG_GROWTH_PER_REVIEW)
from run_analysis.race_preparation import destination as race_destination, MARATHON_POLICIES
from .race_event import preparation_active


@dataclass(frozen=True)
class ProgressionInputs:
    target: object
    volume_growth_per_7: float
    long_growth_fraction: float
    long_growth_days: int
    long_share: float
    growth_allowed: bool
    phase: str


def resolve(h, as_of):
    day = h.decision(as_of)
    race = getattr(h, 'race_goal', None)
    if preparation_active(h, day):
        target = race_destination(h.stage, race.goal)
        policy = (MARATHON_POLICIES[race.preparation_mode].progression
                  if race.distance == 'marathon' else h.stage.policy)
        return ProgressionInputs(target, (1 + policy.growth_fraction) ** (7 / policy.review_days) - 1,
            policy.growth_fraction, policy.review_days,
            target.long_miles / target.volume_per_7, race.phase(day) == 'build', race.phase(day))
    stage = h.stage
    preferences = getattr(h, 'planner_preferences', None)
    if preferences is not None:
        stage = replace(stage, goal_long_miles=preferences.goal_long_miles)
    elif race is not None and race.general_long_goal_miles is not None:
        stage = replace(stage, goal_long_miles=max(stage.long_miles, race.general_long_goal_miles))
    return ProgressionInputs(general_destination(stage), VOLUME_GROWTH_PER_7,
        LONG_GROWTH_PER_REVIEW, 14, LONG_SHARE, True, 'general_fitness')
