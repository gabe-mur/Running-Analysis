"""Experimental next-session demand assessment; no live planner consumers.

This is an inspectable two/three-demand sensitivity model, not measured recovery
or injury clearance. Policy envelopes and half-lives must be supplied explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from math import isfinite

from .workload_accounting import Quantity, Workload


def _utc(value):
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError('Timezone-aware times are required')
    return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class SessionEntry:
    key: str
    start: datetime
    end: datetime
    workload: Workload
    work_minutes: Quantity | None = None

    def __post_init__(self):
        if not self.key or _utc(self.end) < _utc(self.start):
            raise ValueError('Session key and ordered timestamps required')
        if (self.work_minutes is not None and self.workload.running_minutes is not None
                and self.work_minutes.low > self.workload.running_minutes.high):
            raise ValueError('Work duration exceeds session duration')


@dataclass(frozen=True)
class AssessmentPolicy:
    name: str
    running_minutes_envelope: float
    cardio_points_envelope: float
    running_half_life_hours: float
    cardio_half_life_hours: float
    work_minutes_envelope: float | None = None
    work_half_life_hours: float | None = None

    def __post_init__(self):
        values = (self.running_minutes_envelope, self.cardio_points_envelope,
                  self.running_half_life_hours, self.cardio_half_life_hours)
        if not self.name or any(not isfinite(x) or x <= 0 for x in values):
            raise ValueError('Named policy with finite positive parameters required')
        if (self.work_minutes_envelope is None) != (self.work_half_life_hours is None):
            raise ValueError('Structured work requires both budget and half-life')
        if self.work_minutes_envelope is not None and any(
            not isfinite(x) or x <= 0
            for x in (self.work_minutes_envelope, self.work_half_life_hours)
        ):
            raise ValueError('Structured work parameters must be finite and positive')


def _unique(entries):
    if len({x.key for x in entries}) != len(entries):
        raise ValueError('Duplicate session keys would double-count workload')


def _bounded_work_minutes(workload, supplied):
    """Unknown work may occupy any of the known running duration, not infinity.

    This is a logical upper bound, not a statistical interval or inferred rep
    detector. Unknown structure must not imply either zero work or permanent
    uncertainty long after a finite session's modelled residue has diminished.
    """
    if supplied is not None or workload.running_minutes is None:
        return supplied
    return Quantity(0, workload.running_minutes.high,
                    basis='unknown_work_bounded_by_running_duration')


def replace_expected(entries, completion):
    """Replace exactly one projected ledger item, retaining actual measurements."""
    entries = tuple(entries)
    _unique(entries)
    matches = [x for x in entries if x.key == completion.key]
    if (len(matches) != 1 or matches[0].workload.source != 'projected'
            or completion.workload.source != 'observed'):
        raise ValueError('Completion must replace a matching projected entry exactly once')
    return tuple(completion if x.key == completion.key else x for x in entries)


def assess_next_session(history, proposed: Workload, at: datetime, policy: AssessmentPolicy,
                        *, proposed_work_minutes: Quantity | None = None):
    history = tuple(history)
    _unique(history)
    at = _utc(at)
    if any(_utc(x.end) > at for x in history):
        raise ValueError('History contains unfinished/future sessions')
    if (proposed_work_minutes is not None and proposed.running_minutes is not None
            and proposed_work_minutes.low > proposed.running_minutes.high):
        raise ValueError('Proposed work duration exceeds session duration')
    channels = {}
    specs = [
        ('running_minutes', policy.running_minutes_envelope, policy.running_half_life_hours),
        ('cardio_points', policy.cardio_points_envelope, policy.cardio_half_life_hours),
    ]
    if policy.work_minutes_envelope is not None:
        specs.append(('work_minutes', policy.work_minutes_envelope, policy.work_half_life_hours))
    for name, envelope, half_life in specs:
        low, high, contributions = 0., 0., []
        for entry in sorted(history, key=lambda x: (_utc(x.end), x.key)):
            elapsed = (at-_utc(entry.end)).total_seconds()/3600
            factor = 2**(-elapsed/half_life)
            dose = (_bounded_work_minutes(entry.workload, entry.work_minutes)
                    if name == 'work_minutes' else getattr(entry.workload, name))
            contribution_low = dose.low*factor if dose is not None else 0.
            contribution_high = dose.high*factor if dose is not None else None
            low += contribution_low
            high = high+contribution_high if high is not None and contribution_high is not None else None
            contributions.append(dict(key=entry.key, elapsed_hours=elapsed,
                                      basis=dose.basis if dose is not None else 'unknown',
                                      low=contribution_low, high=contribution_high))
        dose = (_bounded_work_minutes(proposed, proposed_work_minutes)
                if name == 'work_minutes' else getattr(proposed, name))
        total_low = low+(dose.low if dose is not None else 0.)
        total_high = high+dose.high if high is not None and dose is not None else None
        status = ('model_contradicted' if total_low > envelope+1e-8 else
                  'model_supported' if total_high is not None and total_high <= envelope+1e-8
                  else 'uncertain')
        channels[name] = dict(status=status, envelope=envelope, half_life_hours=half_life,
            proposed_basis=dose.basis if dose is not None else 'unknown',
            residual_low=low, residual_high=high, total_low=total_low, total_high=total_high,
            margin_low=envelope-total_high if total_high is not None else None,
            margin_high=envelope-total_low,
            excess_low=max(0., total_low/envelope-1),
            excess_high=max(0., total_high/envelope-1) if total_high is not None else None,
            contributions=contributions)
    states = [x['status'] for x in channels.values()]
    status = ('model_contradicted' if 'model_contradicted' in states else
              'model_supported' if all(x == 'model_supported' for x in states) else 'uncertain')
    # Limits do not trade off: take the largest excess, not a weighted sum of
    # correlated exposure measures. Every supported choice has exactly zero cost.
    excess_low = max(x['excess_low'] for x in channels.values())
    highs = [x['excess_high'] for x in channels.values()]
    excess_high = max(highs) if all(x is not None for x in highs) else None
    limiting = [name for name, x in channels.items() if x['status'] != 'model_supported']
    return dict(version=('structured_demand_recovery_shadow_v1' if policy.work_minutes_envelope is not None
                         else 'two_demand_recovery_shadow_v1'), policy=policy.name,
        status=status, limiting_demands=limiting, channels=channels,
        excess_low=excess_low, excess_high=excess_high, planner_influence=False,
        limitations=['Model compatibility only; not physiological recovery or safety clearance',
                     'Work minutes are a descriptor, not a measurement of anaerobic load or muscular recovery',
                     'No terrain, novelty or automatic subjective-response multiplier',
                     'Envelopes and decay rates are explicit uncalibrated hypotheses'])


def sensitivity_panel(history, proposed, at, policies, offsets_hours=(-6., 0., 6.),
                      *, proposed_work_minutes=None):
    """Same history and candidate dose for every parameter/time scenario."""
    history, policies, offsets_hours = tuple(history), tuple(policies), tuple(offsets_hours)
    if not policies or not offsets_hours or any(not isfinite(x) for x in offsets_hours):
        raise ValueError('Nonempty finite sensitivity scenarios required')
    rows = []
    for policy in policies:
        for offset in offsets_hours:
            result = assess_next_session(history, proposed, _utc(at)+timedelta(hours=offset), policy,
                                         proposed_work_minutes=proposed_work_minutes)
            rows.append(dict(offset_hours=offset, **result))
    statuses = {r['status'] for r in rows}
    return dict(status=next(iter(statuses)) if len(statuses) == 1 else 'uncertain',
                agreement=len(statuses) == 1, scenarios=rows, planner_influence=False)
