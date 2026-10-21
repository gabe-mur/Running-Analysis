"""Offline direct phase-load forecast, not a synthetic HR curve or recovery model.

Nonlinear HR load is measured before averaging. One vote per source session,
with existing log-duration similarity applied to phase and preceding-phase time.
This similarity is an explicit engineering assumption, not validated physiology.
"""
from dataclasses import dataclass
from datetime import timedelta
from math import fsum, isfinite, log

from run_analysis.athlete_projection import aware
from run_analysis.phase_calibration import clip_trace
from run_analysis.training_signals import integrate_cardio_trace
from run_analysis.adaptive.quality_history_policy import QualityHistoryPolicy
from run_analysis.phase_partitions import detailed_layout


POLICY = QualityHistoryPolicy()
KINDS = {'work','warmup','recovery','cooldown'}


@dataclass(frozen=True)
class PhaseDose:
    kind: str
    seconds: float
    previous_seconds: float | None
    known_seconds: float
    load: float | None

    def __post_init__(self):
        if (self.kind not in KINDS or not isfinite(self.seconds) or self.seconds <= 0
                or not isfinite(self.known_seconds) or not 0 <= self.known_seconds <= self.seconds+1e-6
                or self.previous_seconds is not None and (not isfinite(self.previous_seconds) or self.previous_seconds <= 0)
                or (self.known_seconds == 0) != (self.load is None)
                or self.load is not None and (not isfinite(self.load) or self.load < 0)):
            raise ValueError('Finite phase duration and explicit observed load coverage required')


@dataclass(frozen=True)
class PhaseHistory:
    key: str
    start: object
    end: object
    available_at: object
    accounting_key: str
    phases: tuple[PhaseDose,...]

    def __post_init__(self):
        for at in (self.start,self.end,self.available_at):
            aware(at)
        if not self.key or not self.accounting_key or not self.start < self.end <= self.available_at:
            raise ValueError('Observed completed source and accounting identity required')


def checked_layout(layout):
    explicit = bool(layout) and any(kind in {'warmup','recovery','cooldown'} for kind,_,_ in layout)
    at = 0.
    for kind,a,b in layout:
        if kind not in (KINDS if explicit else {'work','nonwork'}) or not all(isfinite(x) for x in (a,b)) or a != at or b <= a:
            raise ValueError('Continuous ordered supported phase layout required')
        at = b
    # Issued prescriptions can identify final recovery separately from cooldown.
    # Preserve it; inferred boundary-only histories retain the existing behavior.
    return tuple(layout) if explicit else detailed_layout(layout)


def phase_observations(layout, spans):
    """Record observed load per known second, never g(mean HR) or missing=zero."""
    phases,previous = [],None
    for kind,a,b in checked_layout(layout):
        known = clip_trace(spans,a,b)
        seconds = fsum(s.end-s.start for s in known)
        amount = integrate_cardio_trace(known,b,compute_density=False)['observed_load']
        phases.append(PhaseDose(kind,b-a,previous,seconds,amount))
        previous = b-a
    return tuple(phases)


def similarity(observation, seconds, previous_seconds, conditioned):
    if not conditioned:
        return 1.
    distance = log(observation.seconds/seconds)**2
    if previous_seconds is not None and observation.previous_seconds is not None:
        distance += log(observation.previous_seconds/previous_seconds)**2
    return 1/(1+distance)


def forecast_direct(layout, history, *, as_of, accounting_key, conditioned=True,
                    sequence_context=False):
    """Timing-only query: neither held-out HR nor speed is an accepted input.

    No label/template gate. Novel phase durations use continuous similarity.
    Missing non-work phase evidence can use pooled non-work with explicit fallback;
    missing work evidence never borrows easy load or becomes zero.
    """
    aware(as_of)
    layout = checked_layout(layout)
    if not accounting_key:
        raise ValueError('Accounting identity required')
    history = tuple(history)
    if len({h.key for h in history}) != len(history):
        raise ValueError('Duplicate source session')
    accepted = sorted((h for h in history if h.accounting_key == accounting_key
        and as_of-timedelta(days=POLICY.current_days) <= h.start < h.end < as_of
        and h.available_at <= as_of and h.phases),key=lambda h:(h.start,h.key))
    result = dict(status='unresolved_work_alignment' if not layout else 'no_recent_phase_evidence',
        source_keys=[h.key for h in accepted],phases=[],conditioned=conditioned,
        issued_at=as_of,accounting_key=accounting_key,policy=POLICY,
        predicted_load=None,uncertainty_calibrated=False)
    if not layout:
        return result
    previous = previous_kind = None
    for kind,a,b in layout:
        matching = {h.key:[p for p in h.phases if p.kind == kind and p.load is not None] for h in accepted}
        fallback = kind != 'work' and not any(matching.values())
        if fallback:
            matching = {h.key:[p for p in h.phases if p.kind != 'work' and p.load is not None] for h in accepted}
        # Offline ablation only: a first effort after warm-up need not have the
        # same rate as an effort after recovery of identical duration. Match
        # physical phase adjacency, never workout names or held-out HR. Retain
        # the original estimate if that context has no preceding evidence.
        if sequence_context:
            context = {h.key:[p for i,p in enumerate(h.phases)
                if p.kind == kind and p.load is not None
                and (h.phases[i-1].kind if i else None) == previous_kind] for h in accepted}
            context_supported = not fallback and any(context.values())
            if context_supported:
                matching = context
        sessions = []
        for h in accepted:
            obs = matching[h.key]
            if not obs:
                continue
            weights = [similarity(p,b-a,previous,conditioned) for p in obs]
            total = fsum(weights)
            sessions.append(dict(key=h.key,rate=fsum(w*p.load/p.known_seconds for w,p in zip(weights,obs))/total,
                relevance=total/len(obs),known_seconds=fsum(p.known_seconds for p in obs),
                phase_seconds=fsum(p.seconds for p in obs)))
        weight = fsum(s['relevance'] for s in sessions)
        rate = fsum(s['rate']*s['relevance'] for s in sessions)/weight if weight else None
        result['phases'].append(dict(kind=kind,start=a,end=b,previous_seconds=previous,
            rate=rate,load=rate*(b-a) if rate is not None else None,
            source_sessions=sessions,fallback_pooled_nonwork=fallback,
            source_rate_dispersion=(min(s['rate'] for s in sessions),max(s['rate'] for s in sessions)) if sessions else None))
        if sequence_context:
            result['phases'][-1].update(previous_kind=previous_kind,
                sequence_context_supported=context_supported,
                sequence_context_fallback=not context_supported)
        previous = b-a
        previous_kind = kind
    if all(p['load'] is not None for p in result['phases']):
        result.update(status='predicted',predicted_load=fsum(p['load'] for p in result['phases']))
    return result


def score_direct(prediction, actual, *, kind=None):
    """Compare load on identical known-HR time; no claimed HR-curve accuracy."""
    if prediction['status'] != 'predicted':
        return None
    expected = observed = known = 0.
    for p in prediction['phases']:
        if kind is not None and (('work' if p['kind']=='work' else 'nonwork') != kind):
            continue
        spans = clip_trace(actual,p['start'],p['end'])
        seconds = fsum(s.end-s.start for s in spans)
        if seconds:
            observed += integrate_cardio_trace(spans,p['end'],compute_density=False)['observed_load']
            expected += seconds*p['rate']
            known += seconds
    if not known:
        return None
    return dict(seconds=known,observed_load=observed,predicted_load=expected,
        signed_load_error=expected-observed,fraction_error=expected/observed-1 if observed > 0 else None)
