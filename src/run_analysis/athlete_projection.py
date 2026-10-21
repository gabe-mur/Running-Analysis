"""Chronological exposure and explicit phase forecasts, offline model candidate.

No fitted result caches, planner writes, injury bands or implicit cold-start load.
Empirical pace ranges are descriptive support, not calibrated prediction limits.
HR equilibrium ranges/kinetics are declared assumptions, never rep-average HR.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import product
from math import ceil, exp, isfinite
from statistics import median
import json

from .response_model import Policy, ewma
from .signal_structure import SignalSpan
from .training_signals import integrate_cardio_trace
from .workload_accounting import Quantity
from .performance_reference import freeze_aerobic_reference, compare_aerobic_completion


def accounting_signature(resting_hr, maximum_hr, policy=Policy()):
    from .training_signals import HeartRateReference
    HeartRateReference(resting_hr, maximum_hr, 'snapshot_settings')
    return json.dumps(dict(method='normalized_hrr_trace_v1', resting_hr=resting_hr,
        maximum_hr=maximum_hr, exponent=policy.intensity_exponent,
        reserve_reference=policy.reserve_reference), sort_keys=True)


def aware(at):
    if at.tzinfo is None or at.utcoffset() is None:
        raise ValueError('Timezone-aware evidence and issue times required')
    return at


@dataclass(frozen=True)
class PaceSample:
    phase: str
    seconds: float
    meters: float
    basis: str
    mean_hr_reserve: float | None = None

    def __post_init__(self):
        if self.phase not in {'easy', 'work', 'recovery'} or not self.basis:
            raise ValueError('Named pace evidence required')
        if any(not isfinite(x) or x <= 0 for x in (self.seconds, self.meters)):
            raise ValueError('Positive finite movement evidence required')
        if self.mean_hr_reserve is not None and not 0 <= self.mean_hr_reserve <= 1:
            raise ValueError('Supported HR reserve required')


@dataclass(frozen=True)
class SessionEvidence:
    key: str
    start: datetime
    end: datetime
    available_at: datetime
    role: str
    meters: float
    moving_seconds: float
    cardio: Quantity | None
    accounting_key: str
    pace_samples: tuple[PaceSample, ...] = ()
    capacity_eligible: bool = True
    ordinary_reference_eligible: bool = False

    def __post_init__(self):
        for at in (self.start, self.end, self.available_at):
            aware(at)
        if not self.key or not self.accounting_key or self.end <= self.start or self.available_at < self.end:
            raise ValueError('Completed session identity, chronology and units required')
        if any(not isfinite(x) or x < 0 for x in (self.meters, self.moving_seconds)):
            raise ValueError('Nonnegative finite session exposure required')
        if self.moving_seconds > (self.end-self.start).total_seconds()+1e-6:
            raise ValueError('Moving duration exceeds elapsed duration')


@dataclass(frozen=True)
class AthleteSnapshot:
    as_of: datetime
    accounting_key: str
    sessions: tuple[SessionEvidence, ...]
    excluded: tuple[tuple[str, str], ...]


def snapshot(history, as_of, accounting_key):
    """Cut on BOTH completion and information availability, before inference.

    Duplicate actual/projected identities are rejected rather than double counted.
    Late uploads do not change an already-issued frozen snapshot.
    """
    aware(as_of)
    if not accounting_key:
        raise ValueError('An accounting signature is required')
    history = tuple(history)
    if len({s.key for s in history}) != len(history):
        raise ValueError('Duplicate session identity')
    accepted, excluded = [], []
    for s in sorted(history, key=lambda s: (s.start, s.key)):
        reason = ('not_completed' if s.end > as_of else
                  'not_available' if s.available_at > as_of else
                  'accounting_mismatch' if s.accounting_key != accounting_key else None)
        if reason:
            excluded.append((s.key, reason))
        else:
            accepted.append(s)
    if any(b.start < a.end for a, b in zip(accepted, accepted[1:])):
        raise ValueError('Overlapping sessions need source reconciliation')
    return AthleteSnapshot(as_of, accounting_key, tuple(accepted), tuple(excluded))


def exposure_summary(state, *, days=42.):
    """Recorded exposure, not a claim that missing dates were observed rest."""
    if not isfinite(days) or days <= 0:
        raise ValueError('Positive exposure window required')
    cutoff = state.as_of-timedelta(days=days)
    recent = [s for s in state.sessions if s.end > cutoff]
    # Whole-session counts/doses by completion; explicitly not a boundary-clipped rate.
    ordinary = [s.moving_seconds/60 for s in recent
                if s.role == 'easy' and s.capacity_eligible and s.ordinary_reference_eligible]
    return dict(window_days=days, convention='completed_sessions_ending_in_window',
        session_keys=[s.key for s in recent], run_count=len(recent),
        distance_meters=sum(s.meters for s in recent),
        moving_minutes=sum(s.moving_seconds for s in recent)/60,
        ordinary_easy_minutes=median(ordinary) if ordinary else None,
        longest_distance_meters=max((s.meters for s in recent), default=None),
        long_count=sum(s.role == 'long' for s in recent),
        quality_count=sum(s.role in {'intervals', 'tempo_threshold', 'race'} for s in recent),
        recorded_cardio_low=sum(s.cardio.low for s in recent if s.cardio is not None),
        recorded_cardio_high=(sum(s.cardio.high for s in recent)
                             if all(s.cardio is not None for s in recent) else None),
        history_completeness='not_inferred_from_uploads',
        ordinary_reference_note='Requires explicit ordinary-dose eligibility; an easy label alone does not distinguish support runs')


def pace_reference(state, phase, *, seconds=None, meters=None, lookback_days=84.,
                   maximum_work_ratio=2., reserve_band=None):
    """Original bounded reference retained for diagnostic/control comparisons.

    New forecasts use performance_reference.phase_pace_reference, which does
    not turn this matching neighborhood into a workout acceptance gate.

    Equal session votes prevent six repetitions being six independent runners.
    No easy-to-fast extrapolation or hard minimum history count. One source is
    labelled sparse, not promoted into certainty. Bounds are observed dispersion.
    """
    if phase not in {'easy', 'work', 'recovery'}:
        raise ValueError('Unknown phase')
    if (not isfinite(lookback_days) or lookback_days <= 0
            or not isfinite(maximum_work_ratio) or maximum_work_ratio < 1):
        raise ValueError('Valid explicit reference policy required')
    for x in (seconds, meters):
        if x is not None and (not isfinite(x) or x <= 0):
            raise ValueError('Positive requested work dose required')
    cutoff = state.as_of-timedelta(days=lookback_days)
    speeds, keys, durations = [], [], []
    for session in state.sessions:
        if session.end <= cutoff or not session.capacity_eligible:
            continue
        matching = []
        for p in session.pace_samples:
            if p.phase != phase:
                continue
            if reserve_band is not None and (p.mean_hr_reserve is None or not
                    reserve_band.low <= p.mean_hr_reserve <= reserve_band.high):
                continue
            if phase == 'work' and any(target is not None and not
                    target/maximum_work_ratio <= actual <= target*maximum_work_ratio
                    for actual, target in ((p.seconds, seconds), (p.meters, meters))):
                continue
            matching.append(p)
        if matching:
            speeds.append(median(p.meters/p.seconds for p in matching))
            keys.append(session.key)
            durations.extend(p.seconds for p in matching)
    return dict(speed_mps=(Quantity(min(speeds), max(speeds), median(speeds),
                         'prior_session_pace_dispersion_not_prediction_interval') if speeds else None),
        phase=phase, source_keys=keys, source_count=len(keys),
        status='unsupported' if not speeds else 'single_session' if len(keys) == 1 else 'empirical_support',
        observed_duration_seconds=(min(durations), max(durations)) if durations else None,
        lookback_days=lookback_days, maximum_work_ratio=maximum_work_ratio,
        context_adjusted=False)


@dataclass(frozen=True)
class LoadSeed:
    at: datetime
    atl: Quantity
    ctl: Quantity
    source: str
    accounting_key: str

    def __post_init__(self):
        aware(self.at)
        if not self.source or not self.accounting_key:
            raise ValueError('Initial load-state provenance required')


def project_load_history(state, *, seed=None, complete_recording_since=None):
    """Continuous-time rate convention; session load spread over its duration.

    This is a declared coarse approximation, not precise within-session kinetics.
    Seed and completeness are explicit: unknown prior history is never zero.
    """
    if seed is None or complete_recording_since is None:
        return dict(status='unknown_initial_state_or_history', atl=None, ctl=None, ratio=None)
    if any(reason in {'not_available', 'accounting_mismatch'} for _, reason in state.excluded):
        return dict(status='incomplete_available_ledger', atl=None, ctl=None, ratio=None)
    aware(complete_recording_since)
    if seed.accounting_key != state.accounting_key:
        raise ValueError('Load seed and history must use the same accounting')
    if complete_recording_since > seed.at or seed.at > state.as_of:
        raise ValueError('Seed must be inside declared coverage and before cutoff')
    if any(s.start < seed.at < s.end for s in state.sessions):
        raise ValueError('Seed cannot split a session')
    sessions = [s for s in state.sessions if s.start >= seed.at]
    if any(s.cardio is None for s in sessions):
        return dict(status='unbounded_session_load', atl=None, ctl=None, ratio=None)
    atl, ctl = [seed.atl.low, seed.atl.high], [seed.ctl.low, seed.ctl.high]
    previous = seed.at
    for session in sessions:
        gap = (session.start-previous).total_seconds()/86400
        duration = (session.end-session.start).total_seconds()/86400
        for i, load in enumerate((session.cardio.low, session.cardio.high)):
            atl[i] = ewma(ewma(atl[i], 0, gap, 7), load/duration, duration, 7)
            ctl[i] = ewma(ewma(ctl[i], 0, gap, 42), load/duration, duration, 42)
        previous = session.end
    tail = (state.as_of-previous).total_seconds()/86400
    atl = [ewma(x, 0, tail, 7) for x in atl]
    ctl = [ewma(x, 0, tail, 42) for x in ctl]
    return dict(status='conditional_on_declared_history', atl=atl, ctl=ctl,
        tsb=[ctl[0]-atl[1], ctl[1]-atl[0]],
        ratio=[atl[0]/ctl[1], atl[1]/ctl[0]] if ctl[0] > 0 else None,
        seed_source=seed.source, convention='uniform_session_rate_continuous_elapsed_time',
        injury_risk_interpretation=False)


@dataclass(frozen=True)
class ForecastPhase:
    name: str
    seconds: float | None
    meters: float | None
    speed_mps: Quantity | None
    equilibrium_reserve: Quantity
    assumption_source: str
    finish_total_distance: bool = False

    def __post_init__(self):
        if not self.name or not self.assumption_source or (self.seconds is None) == (self.meters is None):
            raise ValueError('Phase requires one timing/distance prescription and provenance')
        for value in (self.seconds, self.meters):
            if value is not None and (not isfinite(value) or value <= 0):
                raise ValueError('Positive prescribed dose required')
        if self.equilibrium_reserve.high > 1:
            raise ValueError('Supported HR reserve range is [0,1]')
        if self.speed_mps is not None and self.speed_mps.low <= 0:
            raise ValueError('Positive speed bounds required')
        if self.meters is not None and self.speed_mps is None:
            raise ValueError('Distance phase needs a supported or explicitly supplied pace')
        if self.finish_total_distance and self.meters is None:
            raise ValueError('Remaining-distance phase requires a session distance target')


def forecast_trace(phases, *, initial_reserve, tau_seconds, branch='estimate', step_seconds=.5):
    """First-order HR assumption with carried state across phase boundaries.

    Branch low/high refers to supplied speed AND HR choices, not a guaranteed
    monotone load bound: faster speed also shortens a distance-based phase.
    """
    if branch not in {'low', 'high', 'estimate'}:
        raise ValueError('Unknown forecast branch')
    if not 0 <= initial_reserve <= 1 or not isfinite(tau_seconds) or tau_seconds <= 0:
        raise ValueError('Valid initial HR and positive HR time constant required')
    if not isfinite(step_seconds) or step_seconds <= 0:
        raise ValueError('Positive integration step required')
    spans, rows = [], []
    at, reserve = 0., initial_reserve
    def select(q):
        value = getattr(q, branch)
        return value if value is not None else (q.low+q.high)/2
    phases = tuple(phases)
    for index, p in enumerate(phases):
        speed = select(p.speed_mps) if p.speed_mps is not None else None
        meters = p.meters
        if p.finish_total_distance:
            if index != len(phases)-1 or any(r['meters'] is None for r in rows):
                raise ValueError('Distance completion needs known preceding distance and must be last')
            meters -= sum(r['meters'] for r in rows)
            if meters <= 0:
                raise ValueError('Structured phases exhaust total distance; do not inflate the target')
        duration = p.seconds if p.seconds is not None else meters/speed
        meters = meters if meters is not None else duration*speed if speed is not None else None
        target = select(p.equilibrium_reserve)
        count = max(1, ceil(duration/step_seconds))
        width = duration/count
        first = at
        for _ in range(count):
            end_reserve = target+(reserve-target)*exp(-width/tau_seconds)
            spans.append(SignalSpan(at, at+width, reserve, end_reserve))
            reserve = end_reserve
            at += width
        rows.append(dict(name=p.name, start_seconds=first, end_seconds=at,
                         meters=meters, speed_mps=speed, equilibrium_reserve=target))
    return tuple(spans), tuple(rows), at


def issue_forecast(state, phases, *, initial_reserve, initial_source,
                   tau_seconds=(20., 45., 90.), policy=Policy()):
    """Freeze declared prescription and numeric scenarios before observing outcome.

    Enumerate HR and pace corners independently (opposite effects on distance
    load). This is a finite sensitivity panel, not a confidence interval or proof
    of global extrema over every independent phase combination.
    """
    phases = tuple(phases)
    basis = json.loads(state.accounting_key)
    if (basis['exponent'] != policy.intensity_exponent
            or basis['reserve_reference'] != policy.reserve_reference):
        raise ValueError('Forecast policy does not match the history accounting')
    if not phases or not initial_source or initial_reserve.high > 1 or not tau_seconds:
        raise ValueError('Explicit phases, initial HR assumptions and kinetics required')
    scenarios, evaluated = [], {}
    from dataclasses import replace
    for speed_side, hr_side, tau, initial_side in product(('low', 'high'), ('low', 'high'),
                                                        tau_seconds, ('low', 'high')):
        selected = tuple(replace(p,
            speed_mps=Quantity.point(getattr(p.speed_mps, speed_side), p.speed_mps.basis) if p.speed_mps else None,
            equilibrium_reserve=Quantity.point(getattr(p.equilibrium_reserve, hr_side),
                                                p.equilibrium_reserve.basis)) for p in phases)
        initial = getattr(initial_reserve, initial_side)
        cache_key = (tuple((p.seconds, p.meters, p.speed_mps.low if p.speed_mps else None,
                           p.equilibrium_reserve.low, p.finish_total_distance) for p in selected), tau, initial)
        if cache_key not in evaluated:
            spans, rows, duration = forecast_trace(selected, initial_reserve=initial, tau_seconds=tau)
            cardio = integrate_cardio_trace(spans, duration, policy)
            distance = sum(p['meters'] for p in rows) if all(p['meters'] is not None for p in rows) else None
            evaluated[cache_key] = (duration, distance, cardio['total_load'], cardio['response_load_candidate'])
        duration, distance, load, response = evaluated[cache_key]
        scenarios.append(dict(speed_side=speed_side, hr_side=hr_side, tau_seconds=tau,
            initial_side=initial_side, elapsed_seconds=duration,
            distance_meters=distance, cardio_load=load, response_load_candidate=response))
    central_tau = median(tau_seconds)
    central_initial = (initial_reserve.estimate if initial_reserve.estimate is not None
                       else (initial_reserve.low+initial_reserve.high)/2)
    trace, phase_rows, central_duration = forecast_trace(phases, initial_reserve=central_initial,
                                                        tau_seconds=central_tau)
    central_cardio = integrate_cardio_trace(trace, central_duration, policy)
    central = dict(tau_seconds=central_tau, initial_reserve=central_initial,
        elapsed_seconds=central_duration, cardio_load=central_cardio['total_load'],
        response_load_candidate=central_cardio['response_load_candidate'], phases=phase_rows,
        basis='Supplied estimates or midpoints; median kinetics scenario, not a fitted prediction')
    return dict(version='issued_phase_forecast_v1', issued_at=state.as_of,
        history_keys=tuple(s.key for s in state.sessions), accounting_key=state.accounting_key,
        aerobic_reference=freeze_aerobic_reference(state),
        phases=phases, initial_reserve=initial_reserve, initial_source=initial_source,
        policy=policy, scenarios=scenarios, central=central,
        duration_scenario_range=(min(s['elapsed_seconds'] for s in scenarios), max(s['elapsed_seconds'] for s in scenarios)),
        cardio_scenario_range=(min(s['cardio_load'] for s in scenarios), max(s['cardio_load'] for s in scenarios)),
        forecast_calibrated=False, planner_influence=False)


def compare_completion(issued, actual):
    if actual.start < issued['issued_at'] or actual.key in issued['history_keys']:
        raise ValueError('Completion must be held out from the issued forecast')
    if actual.accounting_key != issued['accounting_key']:
        raise ValueError('Cannot compare mixed HR settings or load units')
    low, high = issued['cardio_scenario_range']
    q = actual.cardio
    status = ('unknown' if q is None else 'above_scenarios' if q.low > high else
              'below_scenarios' if q.high < low else 'within_scenarios' if low <= q.low <= q.high <= high
              else 'overlapping_uncertainty')
    return dict(cardio_status=status, actual_cardio=q, scenario_range=(low, high),
        aerobic_performance=compare_aerobic_completion(issued['aerobic_reference'], actual),
        central_cardio_load=issued['central']['cardio_load'],
        difference_from_central=(q.low-issued['central']['cardio_load'], q.high-issued['central']['cardio_load'])
            if q is not None else None,
        actual_elapsed_seconds=(actual.end-actual.start).total_seconds(),
        central_elapsed_seconds=issued['central']['elapsed_seconds'],
        expected_duration_scenarios=issued['duration_scenario_range'],
        interpretation='Forecast discrepancy, not proof of noncompliance, fatigue or permission to reschedule',
        adaptation_authorized=False)
