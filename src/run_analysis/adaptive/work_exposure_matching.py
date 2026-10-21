"""Offline work-only relevance trial around the unchanged direct predictor.

Requested bands select earlier whole-phase measurements; they are not latent
effort or equilibrium HR. Nonwork rates, accounting and source-session ownership
are unchanged. This module is deliberately not selected by the live forecaster.
Single-query diagnostics can retain an unconditioned proxy; effort-aware ranking
must use forecast_work_for_ranking, which never promotes that proxy to a supported cost.
"""
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from math import fsum, isfinite

from run_analysis.phase_calibration import clip_trace
from run_analysis.signal_structure import SignalSpan
from .direct_quality_load import checked_layout, forecast_direct, similarity
from .prescribed_hr_target import prescribed_hr_target


VERSION = 'work_only_exposure_overlap_v1'


@dataclass(frozen=True)
class WorkExposure:
    start_band: tuple[float, float]
    finish_band: tuple[float, float]
    basis: str

    def __post_init__(self):
        if (not self.basis or any(len(b) != 2 or
                not all(isfinite(v) for v in b) or not 0 <= b[0] <= b[1] <= 1
                for b in (self.start_band, self.finish_band))):
            raise ValueError('Explicit normalized requested work bands and provenance required')


@dataclass(frozen=True)
class PhaseTrace:
    start: float
    end: float
    spans: tuple[SignalSpan, ...]

    def __post_init__(self):
        if (not all(isfinite(v) for v in (self.start, self.end)) or not 0 <= self.start < self.end
                or any(s.start < self.start or s.end > self.end or
                       not 0 <= s.first <= 1 or not 0 <= s.last <= 1 for s in self.spans)
                or any(b.start < a.end for a, b in zip(self.spans, self.spans[1:]))):
            raise ValueError('Ordered coextensive observed HR support required')


def requested_work_bands(phases, config):
    """Use each compiled phase's independent effort cues, not workout names.

    Unresolved cues remain unresolved. Native distance/time extent belongs to
    the compiler; this adapter does not reconstruct timing from a target run.
    """
    result = []
    for phase in phases:
        if phase.kind != 'work':
            result.append(None)
            continue
        ends = tuple(prescribed_hr_target(cues, config) for cues in
                     (phase.effort.start_cues, phase.effort.finish_cues))
        result.append(WorkExposure((ends[0].low, ends[0].high), (ends[1].low, ends[1].high),
            'compiled phase effort cues; '+ends[0].basis+' -> '+ends[1].basis)
            if all(p is not None for p in ends) else None)
    return tuple(result)


def band_fraction(spans, left, right, start_band, finish_band):
    """Exact known-time overlap; preserve gaps and lower-HR effort startup.

    Both bands follow normalized phase time, including when durations differ.
    This describes exposure, not whether the earlier command was adhered to.
    """
    spans=tuple(spans)
    if all(type(s) is SignalSpan for s in spans):
        return _band_fraction_cached(spans,left,right,tuple(start_band),tuple(finish_band))
    return _band_fraction_uncached(spans,left,right,start_band,finish_band)


@lru_cache(maxsize=256)
def _band_fraction_cached(spans,left,right,start_band,finish_band):
    return _band_fraction_uncached(spans,left,right,start_band,finish_band)


def _band_fraction_uncached(spans, left, right, start_band, finish_band):
    if not all(isfinite(v) for v in (left, right)) or not 0 <= left < right:
        raise ValueError('Finite positive phase interval required')
    WorkExposure(tuple(start_band), tuple(finish_band), 'overlap query')
    spans = tuple(spans)
    if any(b.start < a.end for a, b in zip(spans, spans[1:])):
        raise ValueError('Overlapping observed HR support')
    known = matched = 0.
    for s in clip_trace(spans, left, right):
        seconds = s.end-s.start
        def bounds(at):
            fraction = (at-left)/(right-left)
            return tuple(a+(b-a)*fraction for a,b in zip(start_band, finish_band))
        low_a, high_a = bounds(s.start)
        low_b, high_b = bounds(s.end)
        low_diff = (s.first-low_a, s.last-low_b)
        high_diff = (high_a-s.first, high_b-s.last)
        cuts = [0., 1.]
        for a,b in (low_diff, high_diff):
            if a != b and 0 < -a/(b-a) < 1:
                cuts.append(-a/(b-a))
        cuts.sort()
        for a,b in zip(cuts, cuts[1:]):
            mid = (a+b)/2
            if all(x+(y-x)*mid >= 0 for x,y in (low_diff, high_diff)):
                matched += seconds*(b-a)
        known += seconds
    return matched/known if known else None


def forecast_work_exposure(layout, history, profiles, requested, *, as_of, accounting_key):
    """One untuned work-only candidate; no target-run measurements are inputs.

    All nonwork phase dictionaries are exactly the baseline. An unresolved cue,
    missing source trace or absent band support retains the timing-only estimate
    with explicit diagnostics, not an intensity-conditioned or zero-cost claim.
    Source rates include every known second of the observed phase, not just the
    seconds overlapping the band. No exact format or matching workout required.
    """
    layout = checked_layout(layout)
    history = tuple(history)
    requested = tuple(requested)
    if (len(layout) != len(requested) or any(
            q is not None and (kind != 'work' or not isinstance(q, WorkExposure))
            for (kind,_,_),q in zip(layout, requested))):
        raise ValueError('Work-only query must align with the complete phase sequence')
    baseline = forecast_direct(layout, history, as_of=as_of, accounting_key=accounting_key)
    result = deepcopy(baseline)
    result['work_exposure_matching'] = dict(version=VERSION, planner_influence=False,
        uncertainty_calibrated=False, observed_exposure_is_commanded_effort=False)
    accepted = {h.key:h for h in history if h.key in baseline['source_keys']}
    previous = None
    for (kind,left,right),command,p in zip(layout, requested, result['phases']):
        if kind != 'work':
            previous = right-left
            continue
        reason = 'unknown_requested_work_exposure' if command is None else 'no_observed_band_overlap'
        sessions = []
        missing = []
        if command is not None:
            for h in accepted.values():
                indices = [i for i,o in enumerate(h.phases) if o.kind == 'work' and o.load is not None]
                if not indices:
                    continue
                profile = profiles.get(h.key)
                if profile is None:
                    missing.append(h.key)
                    continue
                if len(profile) != len(h.phases):
                    raise ValueError('Phase trace and source observations must align')
                weights, rates, fractions = [], [], []
                at = 0.
                for i,o in enumerate(h.phases):
                    source = profile[i]
                    if (not isinstance(source, PhaseTrace) or abs(source.start-at) > 1e-6
                            or abs(source.end-source.start-o.seconds) > 1e-6):
                        raise ValueError('Observed phase trace boundaries do not match phase cost')
                    at += o.seconds
                    if i not in indices:
                        continue
                    known = fsum(s.end-s.start for s in source.spans)
                    if abs(known-o.known_seconds) > 1e-6:
                        raise ValueError('Exposure must use the same known-time mask as measured phase load')
                    fraction = band_fraction(source.spans, source.start, source.end,
                                             command.start_band, command.finish_band)
                    weights.append(similarity(o, right-left, previous, True)*fraction)
                    rates.append(o.load/o.known_seconds)
                    fractions.append(fraction)
                total = fsum(weights)
                if total:
                    sessions.append(dict(key=h.key, rate=fsum(w*r for w,r in zip(weights,rates))/total,
                        relevance=total/len(indices), match_fractions=fractions))
        total = fsum(s['relevance'] for s in sessions)
        supported = bool(total) and not missing
        if missing:
            reason = 'missing_observed_source_trace'
        if supported:
            p['rate'] = fsum(s['rate']*s['relevance'] for s in sessions)/total
            p['load'] = p['rate']*(right-left)
        p['work_exposure'] = dict(supported=supported, fallback_reason=None if supported else reason,
            command=command, missing_trace_sources=missing, sources=sessions)
        previous = right-left
    if result['status'] == 'predicted':
        result['predicted_load'] = fsum(p['load'] for p in result['phases'])
    work = [p for p in result['phases'] if p['kind'] == 'work']
    result['work_exposure_matching'].update(
        requested_intensity_rankable=result['status'] == 'predicted' and bool(work)
            and all(p['work_exposure']['supported'] for p in work),
        timing_only_proxy_load=baseline['predicted_load'])
    return result


def forecast_work_for_ranking(layout, history, profiles, requested, *, as_of, accounting_key):
    """Effort-aware handoff, not an invented out-of-support transfer model.

    Preserve the complete diagnostic and original timing-only estimate, but do
    not expose an unsupported intensity as a comparable predicted load. Novel
    durations/formats remain eligible when their work cues have observed support.
    Unresolved status is a forecast limitation, not a medical or workout ban.
    All-supported predictions are bit-for-bit the diagnostic candidate.
    """
    result = forecast_work_exposure(layout, history, profiles, requested,
        as_of=as_of, accounting_key=accounting_key)
    if result['work_exposure_matching']['requested_intensity_rankable']:
        return result
    result['work_exposure_matching'].update(
        diagnostic_predicted_load=result['predicted_load'],
        diagnostic_work_costs=[dict(index=i,rate=p['rate'],load=p['load'])
            for i,p in enumerate(result['phases']) if p['kind']=='work'],
        unresolved_phases=[dict(index=i,reason=p['work_exposure']['fallback_reason'])
            for i,p in enumerate(result['phases'])
            if p['kind']=='work' and not p['work_exposure']['supported']])
    if result['status'] == 'predicted':
        result['status'] = 'unresolved_requested_work_exposure'
    result['predicted_load'] = None
    for p in result['phases']:
        if p['kind']=='work' and not p['work_exposure']['supported']:
            p.update(rate=None,load=None)
    return result
