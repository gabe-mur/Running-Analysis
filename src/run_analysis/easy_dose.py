"""Provisional planned aerobic-dose floor, separate from benefit and recovery."""
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, ROUND_CEILING
from math import isfinite

from .athlete_projection import aware


@dataclass(frozen=True)
class AerobicMinimum:
    minimum_aerobic_minutes: float
    source: str

    def __post_init__(self):
        if not self.source or not isfinite(self.minimum_aerobic_minutes) or self.minimum_aerobic_minutes <= 0:
            raise ValueError('Explicit positive aerobic block and policy source required')


@dataclass(frozen=True)
class EasyDurationReference:
    issued_at: datetime
    startup_minutes: float
    minutes_per_mile: float
    source: str

    def __post_init__(self):
        aware(self.issued_at)
        if (not self.source or not isfinite(self.startup_minutes) or self.startup_minutes < 0
                or not isfinite(self.minutes_per_mile) or self.minutes_per_mile <= 0):
            raise ValueError('Explicit nonnegative startup, positive pace and reference provenance required')


def minimum_easy_miles(reference, policy):
    """Round upward only to obtain the smallest one-decimal *planned* command."""
    minutes = Decimal(str(reference.startup_minutes))+Decimal(str(policy.minimum_aerobic_minutes))
    return float((minutes/Decimal(str(reference.minutes_per_mile))).quantize(
        Decimal('.1'), rounding=ROUND_CEILING))


def assess_easy_dose(miles, reference, policy):
    """Evaluate an issued command, never round/modify it or erase actual load.

    The post-startup budget is intended aerobic phase time at the supplied pace,
    NOT a prediction that every second will fall in the observed HR band. Actual
    warm-up may itself include aerobic time; that remains in recorded accounting.
    A five-minute block is a user-selected provisional floor, not an optimum.
    """
    if not isfinite(miles) or miles <= 0 or abs(miles*10-round(miles*10)) > 1e-8:
        raise ValueError('Positive one-decimal committed distance required')
    total = miles*reference.minutes_per_mile
    block = max(0., total-reference.startup_minutes)
    return dict(committed_miles=miles, total_minutes_at_reference_pace=total,
        startup_allowance_minutes=reference.startup_minutes,
        intended_aerobic_budget_minutes=block,
        minimum_aerobic_minutes=policy.minimum_aerobic_minutes,
        passes_planned_minimum=block+1e-8 >= policy.minimum_aerobic_minutes,
        minimum_command_miles=minimum_easy_miles(reference, policy),
        reference_source=reference.source, policy_source=policy.source,
        observed_zone_minutes_predicted=False, recovery_cleared=False,
        session_benefit_score=None, planner_influence=False)
