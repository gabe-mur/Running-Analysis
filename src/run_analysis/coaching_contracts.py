"""Compile frozen endurance prescriptions; explicit assumptions, not recovery fits.

Stage distances remain authoritative. Original paired distance/time observations
describe their duration; they do not silently replace the stage with today's
rolling median. Quality purpose and full-program suitability are separate inputs.
"""
from dataclasses import dataclass
from math import isfinite

from .demonstrated_training import demonstrated_baseline
from .session_decision import CoachingContract, DoseLimits
from .staged_progression import MILE, tenth
from .workload_accounting import Quantity


@dataclass(frozen=True)
class ReductionPolicy:
    duration_fraction: float
    minimum_minutes: float
    source: str

    def __post_init__(self):
        if not self.source or not isfinite(self.duration_fraction) or not 0 < self.duration_fraction < 1:
            raise ValueError('Explicit fractional reduction and coaching provenance required')
        if not isfinite(self.minimum_minutes) or self.minimum_minutes <= 0:
            raise ValueError('Explicit meaningful-duration floor required')


def compile_endurance_contract(history, stage, *, purpose, intensity_key,
                              intents=(), reduction=None):
    """One-time issue-time compiler, not a daily re-anchoring or acceptance gate.

    A point limit is a *planned command*, not a zero-tolerance adherence contract.
    Duration transfer assumes the reference pace; it is not a forecast interval.
    An absent justified reduction permits ordinary continuation but not an
    invented reduced dose. The caller must review phase and surrounding program.
    """
    if purpose not in {'easy', 'long'}:
        raise ValueError('Quality needs explicit adaptation purpose and work/recovery structure')
    history, intents = tuple(history), tuple(intents)
    baseline = demonstrated_baseline(history, stage.issued_at, intents=intents, windows=(28,))
    recent = baseline['windows']['28']
    result = dict(contract=None, stage_key=stage.key, issued_at=stage.issued_at,
        planner_influence=False, original_program_justified=False,
        physiological_recovery_established=False, limitations=[
            'Stage dose is a declared provisional prescription, not demonstrated sustainable capacity',
            '28-day reference window is a summary convention, not a recovery deadline',
            'Paired reference pace transfer is descriptive, not a calibrated duration forecast',
            'Reduction fraction and meaningful-duration floor, if supplied, are coaching assumptions',
            'Point commands do not replace delivered-dose adherence tolerances',
            'Phase suitability, quality contracts and surrounding structure require separate review'])
    # Do not erase an adverse/excluded recent session by finding an older donor.
    if recent['reference_excluded_keys']:
        return dict(result, status='recent_exclusions_require_review')
    eligible = set(recent['reference_eligible']['keys'])
    candidates = [s for s in history if s.key in eligible and s.role == purpose
                  and s.meters > 0 and s.moving_seconds > 0]
    intent_known = {i.key: i for i in intents if i.available_at <= stage.issued_at}
    provisional_intent = False
    if purpose == 'easy':
        candidates = [s for s in candidates if s.key not in intent_known or
                      intent_known[s.key].purpose != 'support']
        ordinary = [s for s in candidates if s.key in intent_known and
                    intent_known[s.key].purpose in {'ordinary', 'medium_long'}]
        provisional_intent = not bool(ordinary)
        candidates = ordinary or candidates
    if not candidates:
        return dict(result, status='missing_paired_reference')
    target = stage.ordinary_miles if purpose == 'easy' else stage.long_miles
    # Stable deterministic tie break; never splice one run's distance to another's time.
    anchor = (min(candidates, key=lambda s: (abs(s.meters/MILE-target), -s.end.timestamp(), s.key))
              if purpose == 'easy' else max(candidates, key=lambda s: (s.end, s.key)))
    minutes_per_mile = anchor.moving_seconds/60/(anchor.meters/MILE)
    ordinary_minutes = target*minutes_per_mile
    reduced, reduced_minutes, reduction_status = None, None, 'not_declared'
    if reduction is not None:
        miles = tenth(target*reduction.duration_fraction)
        reduced_minutes = miles*minutes_per_mile
        if 0 < miles < target and reduced_minutes >= reduction.minimum_minutes:
            reduced = DoseLimits(Quantity.point(miles, 'explicit_duration_reduction_rounded_distance'))
            reduction_status = 'declared_trial_not_validated'
        else:
            reduction_status = 'no_meaningful_reduction_after_rounding'
    contract = CoachingContract(stage.issued_at,
        f'stage:{stage.key};paired_reference:{anchor.key};reduction:{reduction.source if reduction else "none"}',
        purpose, intensity_key, DoseLimits(Quantity.point(target, 'frozen_stage_command_not_adherence_band')),
        reduced)
    return dict(result, status='provisional_contract', contract=contract,
        source_key=anchor.key, source_miles=anchor.meters/MILE,
        source_minutes=anchor.moving_seconds/60, reference_count=len(candidates),
        ordinary_intent_provisional=provisional_intent,
        ordinary_minutes_at_reference_pace=ordinary_minutes,
        reduced_minutes_at_reference_pace=reduced_minutes,
        realized_reduction_fraction=(reduced.miles.estimate/target if reduced else None),
        reduction_policy=reduction, reduction_status=reduction_status)
