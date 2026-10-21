"""Unconnected Q5 references: measured steady exposure != commanded execution.

No equilibrium inversion, workout-family coefficient, current-target measurement,
or confidence claim. A supplied scenario remains available when history cannot
identify execution. Historical intent is never reconstructed from measured HR.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta
from math import fsum, isfinite, log
import re

from run_analysis.athlete_projection import aware
from run_analysis.performance_reference import geometric_mean
from run_analysis.training_signals import HeartRateReference
from run_analysis.workload_accounting import Quantity
from .quality_history_policy import QualityHistoryPolicy
from .quality_prescription_contract import Effort, PaceReference
from .requested_quality_forecast import PaceProfile

VERSION = 'effort_pace_reference_v1'


@dataclass(frozen=True)
class ExecutionPace:
    seconds: float
    speed_mps: float
    effort: Effort | None

    def __post_init__(self):
        if not all(isfinite(x) and x > 0 for x in (self.seconds, self.speed_mps)):
            raise ValueError('Positive measured duration and speed required')


@dataclass(frozen=True)
class EffortEvidence:
    key: str
    start: datetime
    end: datetime
    available_at: datetime
    accounting_key: str
    steady_samples: tuple = ()
    executions: tuple[ExecutionPace, ...] = ()
    command_issued_at: datetime | None = None
    eligible: bool = True

    def __post_init__(self):
        for at in (self.start, self.end, self.available_at):
            aware(at)
        if not self.key or not self.accounting_key or self.end <= self.start or self.available_at < self.end:
            raise ValueError('Observed identity and completed evidence availability required')
        if self.command_issued_at is not None:
            aware(self.command_issued_at)
            if self.command_issued_at >= self.start:
                raise ValueError('Known command must precede execution')
        if any(x.effort is not None for x in self.executions) and self.command_issued_at is None:
            raise ValueError('Known execution effort needs independent issued provenance')


def _signature(effort):
    # Keep upper/low and effort qualifiers: full-zone envelopes cannot prove
    # these commands are equivalent. Ignore order/case/whitespace only.
    def cues(values):
        return tuple(sorted({' '.join(v.lower().split()) for v in values}))
    return cues(effort.start_cues), cues(effort.finish_cues), effort.purpose


def exposure_band(cues, config):
    """Full numeric envelope, not a band center or a fitted threshold.

    Upper/low qualifiers remain in the execution signature; measured exposure
    scenarios disclose that they use the whole zones rather than invented halves.
    """
    zones = {int(z) for z in re.findall(r'\bZ([1-5])\b', ' '.join(cues), re.I)}
    if not zones:
        return None
    hr = HeartRateReference(config['resting_hr'], config['max_hr'], 'frozen configured exposure bands')
    low = hr.reserve(min(config['zones']['z'+str(z)][0] for z in zones))
    high = hr.reserve(max(config['zones']['z'+str(z)][1] for z in zones))
    if low is None or high is None or low > high:
        raise ValueError('Numeric exposure must fit athlete normalization')
    return low, high


def effort_reference(history, effort, *, mode, config, as_of, accounting_key,
                     bout_seconds, supplied=None, policy=QualityHistoryPolicy()):
    """One query, explicitly chosen evidence semantics, no automatic fallback.

    commanded_execution transfers independently saved effort across structures
    using the existing log-duration relevance; actual HR never selects reps.
    steady_exposure uses valid measured moving windows in the full HR band. It
    describes pace at measured exposure, NOT execution of a short-effort command.
    supplied preserves an explicitly authored causal pace scenario unchanged.
    """
    aware(as_of)
    if mode not in {'commanded_execution', 'steady_exposure', 'supplied'}:
        raise ValueError('Explicit evidence semantics required')
    if not accounting_key or not isfinite(bout_seconds) or bout_seconds <= 0:
        raise ValueError('Accounting identity and positive queried duration required')
    result = dict(version=VERSION, mode=mode, status='unresolved_effort_pace', pace=None,
        profile=None, source_keys=(), session_estimates=(), excluded=(),
        uncertainty_calibrated=False, planner_influence=False,
        duration_transfer=False, effort=effort)
    if mode == 'supplied':
        if supplied is None or supplied.issued_at > as_of:
            raise ValueError('Explicit causal supplied pace required')
        return dict(result, status='supplied_execution_scenario', pace=supplied,
                    source_keys=supplied.source_keys)
    history = tuple(history)
    if len({r.key for r in history}) != len(history):
        raise ValueError('Duplicate history identity')
    selected, excluded = [], []
    for row in sorted(history, key=lambda r: (r.start, r.key)):
        reason = ('ineligible' if not row.eligible else 'not_completed' if row.end >= as_of else
            'not_available' if row.available_at > as_of else
            'accounting_mismatch' if row.accounting_key != accounting_key else
            'outside_current_window' if row.start < as_of-timedelta(days=policy.current_days) else None)
        if reason:
            excluded.append((row.key, reason))
        else:
            selected.append(row)
    if mode == 'commanded_execution':
        if effort.purpose == 'gradual':
            return dict(result, status='explicit_progression_profile_needed', excluded=tuple(excluded))
        values, durations = [], []
        for row in selected:
            samples = [x for x in row.executions if x.effort is not None
                       and _signature(x.effort) == _signature(effort)]
            if not samples:
                excluded.append((row.key, 'no_independently_known_matching_effort'))
                continue
            weights = [1/(1+log(x.seconds/bout_seconds)**2) for x in samples]
            values.append((row.key, geometric_mean([x.speed_mps for x in samples], weights),
                           fsum(weights)/len(weights)))
            durations.extend(x.seconds for x in samples)
        endpoints = (values, values)
        transfer = bool(durations and not min(durations) <= bout_seconds <= max(durations))
        status = 'prior_known_effort_execution_scenario'
    else:
        bands = [exposure_band(cues, config) for cues in (effort.start_cues, effort.finish_cues)]
        if any(b is None for b in bands):
            return dict(result, status='unknown_numeric_exposure', excluded=tuple(excluded))
        endpoints = []
        for low, high in bands:
            values = []
            for row in selected:
                samples = [x for x in row.steady_samples if x.mean_hr_reserve is not None
                    and low <= x.mean_hr_reserve <= high and x.seconds > 0 and x.meters > 0]
                if samples:
                    values.append((row.key, geometric_mean([x.meters/x.seconds for x in samples]), 1.))
            endpoints.append(values)
        transfer = False
        status = 'measured_steady_exposure_scenario_not_command_execution'
        result['requested_bands'] = bands
    if not all(endpoints):
        return dict(result, excluded=tuple(excluded),
                    status='no_recent_effort_evidence_explicit_scenario_needed')
    def speed(values):
        return Quantity(min(v for _, v, _ in values), max(v for _, v, _ in values),
            geometric_mean([v for _, v, _ in values], [w for _, _, w in values]),
            'between_session_dispersion_not_prediction_interval')
    first, last = map(speed, endpoints)
    if (mode == 'steady_exposure' and effort.purpose == 'gradual'
            and bands[1][0] > bands[0][0] and bands[1][1] > bands[0][1]
            and last.estimate <= first.estimate):
        # An empirical HR-band association need not imply faster execution.
        # Do not sort endpoints, alter the measurements or issue a backwards
        # increasing-effort progression. Keep the conflicting evidence visible.
        return dict(result, status='measured_endpoints_do_not_support_increasing_progression',
            observed_endpoint_speeds=(first,last),
            source_keys=tuple(sorted({k for values in endpoints for k,_,_ in values})),
            session_estimates=tuple(tuple(v) for v in endpoints), excluded=tuple(excluded))
    mean = Quantity((first.low+last.low)/2, (first.high+last.high)/2,
        (first.estimate+last.estimate)/2, 'declared_linear_endpoint_mean')
    keys = tuple(sorted({k for values in endpoints for k, _, _ in values}))
    profile = PaceProfile(first, last, as_of, keys, status)
    pace = PaceReference(Quantity(1/mean.high, 1/mean.low, 1/mean.estimate, mean.basis),
                         as_of, keys, status)
    return dict(result, status=status, pace=pace, profile=profile, source_keys=keys,
        session_estimates=tuple(tuple(v) for v in endpoints), excluded=tuple(excluded),
        duration_transfer=transfer,
        progression_profile_is_declared=effort.purpose == 'gradual')
