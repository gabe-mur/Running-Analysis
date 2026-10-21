"""Offline direct phase-cost contrast, no pace inversion or live integration.

Keep observed phase costs, adjusting only the relative modeled effect of an
explicit requested-input scenario versus prior executed-input references under
the SAME timing/state/tau. Command HR bands as equilibrium input are an assumption,
not identified intention. No coefficient fitting, blending or family multiplier.
"""
from dataclasses import dataclass
from math import exp, fsum, isfinite

from run_analysis.athlete_projection import aware
from run_analysis.phase_calibration import clip_trace
from run_analysis.signal_structure import SignalSpan
from run_analysis.training_signals import integrate_cardio_trace
from .direct_quality_load import POLICY, checked_layout, forecast_direct, similarity


@dataclass(frozen=True)
class PhaseExposure:
    kind: str
    seconds: float  # Actual phase elapsed time, not measurement window width.
    previous_seconds: float | None
    known_seconds: float
    mean_hrr: float | None
    endpoint_change_sum: float | None


@dataclass(frozen=True)
class ExposureHistory:
    key: str
    start: object
    end: object
    available_at: object
    accounting_key: str
    phases: tuple[PhaseExposure, ...]

    def __post_init__(self):
        for at in (self.start, self.end, self.available_at):
            aware(at)
        if not self.key or not self.accounting_key or not self.start < self.end <= self.available_at:
            raise ValueError('Causal observed identity required')
        for p in self.phases:
            if (p.kind not in {'warmup','work','recovery','cooldown'} or not isfinite(p.seconds) or p.seconds <= 0
                    or not isfinite(p.known_seconds) or not 0 <= p.known_seconds <= p.seconds+1e-6
                    or p.previous_seconds is not None and (not isfinite(p.previous_seconds) or p.previous_seconds <= 0)
                    or (p.known_seconds == 0) != (p.mean_hrr is None)
                    or (p.known_seconds == 0) != (p.endpoint_change_sum is None)
                    or p.mean_hrr is not None and (not isfinite(p.mean_hrr) or not 0 <= p.mean_hrr <= 1
                        or not isfinite(p.endpoint_change_sum))):
                raise ValueError('Supported phase HR moments required')


def phase_exposures(layout, spans):
    """Coextensive supported HR integral/endpoint flux; never bridge a gap.

    Summed endpoint changes telescope within a contiguous segment. Partial
    support describes input over known time only, not the unrecorded intervals.
    Retain that distinction instead of discarding every partly recorded bout.
    """
    spans = tuple(spans)
    if any(b.start < a.end for a,b in zip(spans,spans[1:])):
        raise ValueError('Ordered nonoverlapping HR support required')
    result, previous = [], None
    for kind, a, b in checked_layout(layout):
        known = clip_trace(spans, a, b)
        seconds = fsum(s.end-s.start for s in known)
        mean = fsum((s.end-s.start)*(s.first+s.last)/2 for s in known)/seconds if seconds else None
        flux = fsum(s.last-s.first for s in known) if seconds else None
        result.append(PhaseExposure(kind,b-a,previous,seconds,mean,flux))
        previous = b-a
    return tuple(result)


def input_reference(layout, history, *, as_of, accounting_key, tau_seconds):
    """Recompute every prior input using the current preceding-fit tau.

    Same direct-load duration similarity and source ownership. Measurement
    support normalizes the integral; actual phase time determines similarity.
    Model input can be outside [0,1]; forecast HR validity is checked separately.
    """
    from datetime import timedelta
    aware(as_of)
    history = tuple(history)
    if (not isfinite(tau_seconds) or tau_seconds <= 0 or not accounting_key
            or len({h.key for h in history}) != len(history)):
        raise ValueError('Positive response time, units and unique sources required')
    accepted = sorted((h for h in history if h.accounting_key == accounting_key
        and as_of-timedelta(days=POLICY.current_days) <= h.start < h.end < as_of
        and h.available_at <= as_of), key=lambda h:(h.start,h.key))
    phases, previous = [], None
    for kind, a, b in checked_layout(layout):
        matching = {h.key:[p for p in h.phases if p.kind == kind and p.known_seconds > 0] for h in accepted}
        fallback = kind != 'work' and not any(matching.values())
        if fallback:
            matching = {h.key:[p for p in h.phases if p.kind != 'work' and p.known_seconds > 0] for h in accepted}
        sessions = []
        for h in accepted:
            obs = matching[h.key]
            if not obs:
                continue
            weights = [similarity(p,b-a,previous,True) for p in obs]
            total = fsum(weights)
            inputs = [p.mean_hrr+tau_seconds*p.endpoint_change_sum/p.known_seconds for p in obs]
            sessions.append(dict(key=h.key, demand=fsum(w*u for w,u in zip(weights,inputs))/total,
                relevance=total/len(obs), partial_support=any(p.known_seconds < p.seconds-1e-6 for p in obs)))
        weight = fsum(s['relevance'] for s in sessions)
        demand = fsum(s['relevance']*s['demand'] for s in sessions)/weight if weight else None
        phases.append(dict(kind=kind,start=a,end=b,demand=demand,sources=sessions,
            fallback_pooled_nonwork=fallback))
        previous = b-a
    return phases


def effort_trace(layout, inputs, *, tau_seconds, initial_reserve, step_seconds=.5):
    """Shared first-order response to explicit HRR-input ramps, exact endpoints.

    State carries through phases. Half-second linear spans use the existing load
    integrator; neither pace nor phase-specific physiological coefficient needed.
    Finite signed/above-one inputs are allowed, but invalid predicted HR fails.
    """
    layout = checked_layout(layout)
    if (len(layout) != len(inputs) or not all(isfinite(v) and v > 0 for v in (tau_seconds,step_seconds))
            or not isfinite(initial_reserve) or not 0 <= initial_reserve <= 1
            or any(len(p) != 2 or not all(isfinite(v) for v in p) for p in inputs)):
        raise ValueError('Aligned numeric inputs, positive response/grid, supported initial HR required')
    spans, state = [], initial_reserve
    for (_,a,b), (first,last) in zip(layout,inputs):
        slope = (last-first)/(b-a)
        at, start_state = a, state
        def value(elapsed):
            return first+slope*(elapsed-tau_seconds)+(start_state-first+slope*tau_seconds)*exp(-elapsed/tau_seconds)
        while at < b:
            end = min(b,at+step_seconds)
            if end <= at:
                raise ValueError('Integration step below time precision')
            state = value(end-a)
            spans.append(SignalSpan(at,end,value(at-a),state))
            at = end
    return tuple(spans), all(0 <= s.first <= 1 and 0 <= s.last <= 1 for s in spans)


def forecast_effort_contrast(layout, requested_inputs, phase_history, exposure_history, *,
                             as_of, accounting_key, tau_seconds, initial_reserve):
    """One fixed contrast, no optimized blend or actual target measurements.

    Empirical cost * requested-scenario load / reference-scenario load. This
    preserves the direct baseline when input scenarios match. Without phase
    history the forward curve is an explicitly unvalidated prior scenario, not
    a fitted forecast or zero-load result. No exact-workout/family gate.
    """
    layout = checked_layout(layout)
    baseline = forecast_direct(layout,phase_history,as_of=as_of,accounting_key=accounting_key)
    result = dict(version='direct_effort_contrast_v1',status='unresolved_reference_input',
        baseline=baseline,phases=[],predicted_load=None,uncertainty_calibrated=False,
        planner_influence=False, command_mapping_calibrated=False)
    requested, valid = effort_trace(layout,requested_inputs,tau_seconds=tau_seconds,initial_reserve=initial_reserve)
    if not valid:
        return dict(result,status='requested_hr_outside_reference')
    inputs = input_reference(layout,exposure_history,as_of=as_of,
        accounting_key=accounting_key,tau_seconds=tau_seconds)
    if baseline['status'] != 'predicted':
        # Caller must distinguish this prior mode from empirical adaptation.
        if not layout:
            return result
        for kind,a,b in layout:
            load = integrate_cardio_trace(clip_trace(requested,a,b),b,compute_density=False)['observed_load']
            result['phases'].append(dict(kind=kind,start=a,end=b,load=load,rate=load/(b-a),
                adjustment=None,reference_sources=[]))
        return dict(result,status='predicted',mode='declared_forward_prior_without_phase_history',
            predicted_load=fsum(p['load'] for p in result['phases']))
    if any(p['demand'] is None for p in inputs):
        return result
    reference, valid = effort_trace(layout,[(p['demand'],p['demand']) for p in inputs],
        tau_seconds=tau_seconds,initial_reserve=initial_reserve)
    if not valid:
        return dict(result,status='reference_hr_outside_reference')
    for p,exposure in zip(baseline['phases'],inputs):
        a,b = p['start'],p['end']
        numerator = integrate_cardio_trace(clip_trace(requested,a,b),b,compute_density=False)['observed_load']
        denominator = integrate_cardio_trace(clip_trace(reference,a,b),b,compute_density=False)['observed_load']
        if denominator <= 0:
            return dict(result,status='unsupported_reference_load',phases=[])
        adjustment = numerator/denominator
        amount = p['load']*adjustment
        result['phases'].append(dict(kind=p['kind'],start=a,end=b,load=amount,rate=amount/(b-a),
            adjustment=adjustment,reference_demand=exposure['demand'],reference_sources=exposure['sources']))
    return dict(result,status='predicted',mode='empirical_phase_cost_relative_effort_contrast',
        predicted_load=fsum(p['load'] for p in result['phases']))
