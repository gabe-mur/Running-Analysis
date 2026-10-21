"""Chronological empirical execution corrections, offline candidate only.

Learn relative forecast errors, not absolute historical fitness. Phase HR/load
and pace errors stay separate; observed data and the shared HRR accounting never
change. This is calibration of the declared forecast, not validated physiology.
"""
from collections import defaultdict
from dataclasses import dataclass, replace
from math import isfinite, log
from datetime import datetime, timedelta
import json

from .athlete_projection import ForecastPhase, aware, forecast_trace
from .performance_reference import geometric_mean, phase_pace_reference
from .response_model import Policy
from .signal_structure import SignalSpan
from .training_signals import integrate_cardio_trace
from .workload_accounting import Quantity


def validate_accounting(state):
    key=json.loads(state.accounting_key)
    policy=Policy()
    if (key['method']!='normalized_hrr_trace_v1' or key['exponent']!=policy.intensity_exponent
            or key['reserve_reference']!=policy.reserve_reference):
        raise ValueError('Calibration requires matching shared accounting policy')


def phase_kind(name):
    if name.startswith('work_') or name == 'work':
        return 'work'
    if name.startswith('recovery_') or name == 'recovery':
        return 'recovery'
    if name in {'finish_easy', 'cooldown'}:
        return 'cooldown'
    if name in {'warmup', 'easy'}:
        return name
    raise ValueError('Explicit phase identity required')


def clip_trace(spans, start, end):
    result = []
    for s in spans:
        lo, hi = max(start, s.start), min(end, s.end)
        if hi > lo:
            scale = (s.last-s.first)/(s.end-s.start)
            result.append(SignalSpan(lo, hi, s.first+scale*(lo-s.start), s.first+scale*(hi-s.start)))
    return tuple(result)


def trace_load(spans):
    if not spans:
        return 0.
    return integrate_cardio_trace(spans, max(s.end for s in spans))['observed_load']


def mask_trace(spans, masks):
    result, index = [], 0
    for mask in masks:
        while index < len(spans) and spans[index].end <= mask.start:
            index += 1
        j = index
        while j < len(spans) and spans[j].start < mask.end:
            result.extend(clip_trace((spans[j],), mask.start, mask.end))
            j += 1
    return tuple(result)


@dataclass(frozen=True)
class PhaseResidual:
    key: str
    phase: str
    available_at: datetime
    accounting_key: str
    seconds: float
    pace_ratio: float | None
    load_ratio: float | None
    hr_coverage: float
    origin: str
    completed_at: datetime | None = None

    def __post_init__(self):
        aware(self.available_at)
        if self.completed_at is not None:
            aware(self.completed_at)
            if self.completed_at > self.available_at:
                raise ValueError('Evidence cannot be available before completion')
        if phase_kind(self.phase) != self.phase or not self.key or not self.origin:
            raise ValueError('Canonical phase and source identity required')
        if not isfinite(self.seconds) or self.seconds <= 0 or not 0 <= self.hr_coverage <= 1:
            raise ValueError('Positive duration and fractional coverage required')
        if any(x is not None and (not isfinite(x) or x <= 0) for x in (self.pace_ratio, self.load_ratio)):
            raise ValueError('Finite positive residual ratios required')


def correction(history, *, as_of, accounting_key, phase, seconds, lookback_days=84.):
    """One vote per source run; duration proximity is continuous, not eligibility.

    Use the same recent window as capacity references. Older runs are historical
    test origins, not corrections applied to today's run. None enables the
    explicitly rejected pooled-history diagnostic for reproducibility only.
    No fitted significance cutoff or rejection of unfamiliar formats. Unknown
    load does not become a zero sample. HR coverage reduces its contribution.
    """
    aware(as_of)
    phase = phase_kind(phase)
    if not isfinite(seconds) or seconds <= 0:
        raise ValueError('Positive target duration required')
    if lookback_days is not None and (not isfinite(lookback_days) or lookback_days <= 0):
        raise ValueError('Positive calibration window required')
    cutoff=as_of-timedelta(days=lookback_days) if lookback_days is not None else None
    accepted = [s for s in history if s.available_at <= as_of and
                (cutoff is None or (s.completed_at or s.available_at) > cutoff) and
                s.accounting_key == accounting_key and s.phase == phase]
    result = {}
    for field in ('pace_ratio', 'load_ratio'):
        by_session = defaultdict(list)
        for s in accepted:
            value = getattr(s, field)
            if value is None:
                continue
            w = 1/(1+log(s.seconds/seconds)**2)
            if field == 'load_ratio':
                w *= s.hr_coverage
            if w > 0:
                by_session[s.key].append((value, w))
        values, weights, keys = [], [], []
        for key, rows in sorted(by_session.items()):
            values.append(geometric_mean((v for v, _ in rows), (w for _, w in rows)))
            weights.append(sum(w for _, w in rows)/len(rows))
            keys.append(key)
        result[field] = dict(factor=geometric_mean(values, weights) if values else 1.,
            source_keys=tuple(keys), source_count=len(keys),
            session_range=(min(values), max(values)) if values else None,
            status='empirical_correction' if values else 'unchanged_fallback',
            uncertainty_calibrated=False)
    return result


def learn_phase_residuals(state, actual, observations, *, initial_reserve=.45, tau_seconds=45.):
    """Observed phase timing conditions this diagnostic, not a prospective plan.

    observations: nonoverlapping wall-time segments with raw HR spans, measured
    moving distance/time and provenance. Unknown intervals never receive HR.
    References are cut before the completion; resulting residuals become available
    only at actual.available_at. Fixed-duration HR prediction is separated from
    pace error so an incorrect duration cannot hide an intensity error.
    """
    validate_accounting(state)
    if actual.start < state.as_of or any(s.key == actual.key for s in state.sessions):
        raise ValueError('Completion must be held out from its pace references')
    if actual.accounting_key != state.accounting_key:
        raise ValueError('Mixed accounting')
    if not actual.capacity_eligible:
        return ()
    phases = [ForecastPhase(o['phase'], o['end']-o['start'], None, None,
        o['equilibrium_reserve'], o['origin']) for o in observations]
    if not phases:
        return ()
    if (any(abs(a['end']-b['start']) > 1e-6 for a,b in zip(observations,observations[1:]))
            or observations[0]['start'] != 0
            or abs(observations[-1]['end']-(actual.end-actual.start).total_seconds())>1e-6):
        raise ValueError('Diagnostic phases must partition elapsed time')
    predicted, _, _ = forecast_trace(phases, initial_reserve=initial_reserve, tau_seconds=tau_seconds)
    rows = []
    for o in observations:
        phase, seconds = o['phase'], o['end']-o['start']
        actual_spans = o['spans']
        # Same known timestamps on both sides: gaps cannot manufacture errors.
        pred = mask_trace(predicted, actual_spans)
        expected_load, observed_load = trace_load(pred), trace_load(actual_spans)
        known = sum(s.end-s.start for s in actual_spans)
        rate_ratio = (observed_load/expected_load if actual.cardio is not None
                      and expected_load > 0 and observed_load > 0 else None)
        role = phase if phase in {'work','recovery'} else 'easy'
        ref = phase_pace_reference(state, role, equilibrium_reserve=o['equilibrium_reserve'],
            **({'seconds': seconds} if role == 'work' else {}))
        q = ref['speed_mps']
        ratio = (o['meters']/o['moving_seconds']/q.estimate if q and o['movement_reliable']
                 and o['moving_seconds'] > 0 and o['meters'] > 0 else None)
        rows.append(PhaseResidual(actual.key, phase, actual.available_at, actual.accounting_key,
            seconds, ratio, rate_ratio, min(1., known/seconds), o['origin'],actual.end))
    return tuple(rows)


def calibrated_forecast(state, phases, history, *, initial_reserve=.45, tau_seconds=45., lookback_days=84.):
    """Central execution forecast with independently reported phase corrections.

    No pause-duration invention, injury score, calendar policy, or alteration of
    actual TRIMP. Result is a conditional uninterrupted-execution estimate.
    Empirical load factors are not a newly validated HR trajectory or EPOC model.
    """
    validate_accounting(state)
    phases = tuple(phases)
    base_trace, base_rows, base_duration = forecast_trace(phases,
        initial_reserve=initial_reserve, tau_seconds=tau_seconds)
    adjusted = []
    pace_details = []
    for p,row in zip(phases,base_rows):
        c = correction(history, as_of=state.as_of, accounting_key=state.accounting_key,
            phase=p.name, seconds=row['end_seconds']-row['start_seconds'],lookback_days=lookback_days)
        factor = c['pace_ratio']['factor']
        adjusted.append(replace(p, speed_mps=p.speed_mps.scale(factor) if p.speed_mps else None))
        pace_details.append(c['pace_ratio'])
    trace, rows, duration = forecast_trace(adjusted, initial_reserve=initial_reserve, tau_seconds=tau_seconds)
    output = []
    for p,row,pace_c in zip(adjusted,rows,pace_details):
        c = correction(history, as_of=state.as_of, accounting_key=state.accounting_key,
            phase=p.name, seconds=row['end_seconds']-row['start_seconds'],lookback_days=lookback_days)['load_ratio']
        base = trace_load(clip_trace(trace,row['start_seconds'],row['end_seconds']))
        output.append(dict(**row, pace_correction=pace_c, load_correction=c,
            uncalibrated_phase_load=base, cardio_load=base*c['factor']))
    return dict(issued_at=state.as_of, accounting_key=state.accounting_key,
        baseline_seconds=base_duration, baseline_cardio_load=trace_load(base_trace),
        execution_seconds=duration, cardio_load=sum(p['cardio_load'] for p in output),
        phases=tuple(output), distance_meters=sum(p['meters'] for p in rows) if all(p['meters'] is not None for p in rows) else None,
        duration_convention='Uninterrupted execution; not a forecast of unplanned pauses',
        calibration_status='Empirical relative-error candidate, not calibrated uncertainty', planner_influence=False)


def observed_phases(activity, evidence, config, profile):
    """Reuse existing phase boundaries; never infer quality from ordinary jitter.

    Easy runs get the existing configured warmup analysis window, explicitly not
    a recovered prescription. Recorded lap *starts* partition time without the
    fractional overlaps/gaps of rounded lap starts plus separate timer totals.
    """
    from dataclasses import asdict
    from .movement import classify_movement
    from .training_signals import HeartRateReference
    from .workout_detection import usable_recorded_laps, recorded_interval_work_positions
    origin = activity.start_time_utc
    elapsed = (evidence.end-origin).total_seconds()
    if elapsed <= 0:
        return ()
    if evidence.role in {'easy','long'}:
        warmup = min(elapsed, float(config.get('model', {}).get('minimum_reliable_segment_minutes',5))*60)
        layout = [('warmup',0.,warmup)]
        if warmup < elapsed:
            layout.append(('easy',warmup,elapsed))
        basis = 'configured_warmup_analysis_window_not_verified_prescription'
    elif evidence.role == 'intervals':
        laps = usable_recorded_laps([asdict(l) for l in activity.laps])
        positions = sorted(recorded_interval_work_positions(laps))
        if len(positions) >= 2 and all(l['start_time_utc'] is not None for l in laps):
            starts = [max(0.,(l['start_time_utc']-origin).total_seconds()) for l in laps]
            if starts != sorted(set(starts)) or starts[-1] >= elapsed:
                return ()
            starts[0] = 0.
            ends = starts[1:]+[elapsed]
            layout = [('work' if i in positions else 'warmup' if i<positions[0] else
                       'cooldown' if i == len(laps)-1 else 'recovery', start, end)
                      for i,(start,end) in enumerate(zip(starts,ends))]
            basis = 'recorded_phase_starts'
        else:
            inferred = profile['intensity_structure'].get('pace_inferred_work')
            if not inferred or not inferred['usable_for_comparison']:
                return ()
            work = [((datetime.fromisoformat(w['start'])-origin).total_seconds(),
                     (datetime.fromisoformat(w['end'])-origin).total_seconds()) for w in inferred['windows']]
            cuts = sorted({0.,elapsed,*(t for pair in work for t in pair)})
            if cuts[0] < 0 or cuts[-1] > elapsed:
                return ()
            layout = [('work' if any(a<=start and end<=b for a,b in work) else
                       'warmup' if end<=work[0][0] else 'cooldown' if start>=work[-1][1] else 'recovery',start,end)
                      for start,end in zip(cuts,cuts[1:]) if end>start]
            basis = 'existing_inferred_work_boundaries_unquantified_uncertainty'
    else:
        return ()  # Unresolved tempo/race boundaries are not fabricated.
    intervals = classify_movement(activity.trackpoints,config['moving_time']).intervals
    ref = HeartRateReference(config['resting_hr'],config['max_hr'],'fixed_audit_configuration')
    spans = []
    for i in intervals:
        first,last = ref.reserve(i.start.heart_rate_bpm),ref.reserve(i.end.heart_rate_bpm)
        if first is not None and last is not None and 'large_recording_gap' not in i.flags:
            spans.append(SignalSpan((i.start.timestamp_utc-origin).total_seconds(),
                (i.end.timestamp_utc-origin).total_seconds(),first,last))
    result = []
    for kind,start,end in layout:
        distance = moving = covered = 0.
        reliable = True
        for i in intervals:
            low,high=(i.start.timestamp_utc-origin).total_seconds(),(i.end.timestamp_utc-origin).total_seconds()
            overlap=max(0.,min(end,high)-max(start,low))
            if overlap <= 0:
                continue
            part=overlap/i.elapsed_s
            covered+=overlap
            moving+=part*i.moving_time_s
            distance+=part*i.distance_m
            reliable &= not bool({'large_recording_gap','distance_reset','implausible_distance_speed'} & set(i.flags)) and i.distance_source!='unknown'
        reliable &= abs(covered-(end-start)) < 1e-6
        lo,hi=(config['zones']['z4'] if kind=='work' else
               (config['zones']['z1'][0],config['zones']['z2'][1]) if kind=='recovery' else config['zones']['z2'])
        result.append(dict(phase=kind,start=start,end=end,meters=distance,moving_seconds=moving,
            movement_reliable=reliable,spans=clip_trace(spans,start,end),origin=basis,
            equilibrium_reserve=Quantity(ref.reserve(lo),ref.reserve(hi),basis='original_trial_zone_assumption')))
    return tuple(result)
