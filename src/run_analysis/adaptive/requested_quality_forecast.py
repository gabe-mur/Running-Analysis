"""Unconnected Q5 physical query: exact command, explicit pace, shared HR response.

No family factors or residual correction. Caller freezes a causal response fit
and pace scenarios; this module cannot turn workout names into precise HR. A
linear progression shape is a declared mean-preserving scenario, not a fitted
athlete behavior. Evaluated scenario spread is NOT a confidence interval.
"""
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import fsum, isfinite
import re

from run_analysis.athlete_projection import aware
from run_analysis.hr_response import predict_speed_program
from run_analysis.phase_calibration import clip_trace
from run_analysis.training_signals import integrate_cardio_trace
from run_analysis.training_signals import HeartRateReference
from run_analysis.workload_accounting import Quantity
from .quality_prescription_contract import PaceReference

VERSION = 'requested_quality_response_v1'


@dataclass(frozen=True)
class PaceProfile:
    first_speed: Quantity
    last_speed: Quantity
    issued_at: object
    source_keys: tuple[str, ...]
    basis: str

    def __post_init__(self):
        aware(self.issued_at)
        if (not self.basis or not self.source_keys or not isinstance(self.source_keys, tuple)
                or len(set(self.source_keys)) != len(self.source_keys)
                or any(q.estimate is None or q.low <= 0 for q in (self.first_speed, self.last_speed))):
            raise ValueError('Explicit positive causal pace/profile scenario required')


def effort_pace(effort, fit, config, *, issued_at):
    """Interpret numeric zone cues as broad coaching scenarios, never target HR.

    Invert the SAME prior athlete speed/HR equilibrium for start/finish bands.
    Band centers are declared scenario points, not physiological thresholds or
    predicted executed HR. No template/label gate or pooled short-rep pace.
    Words without numeric zones require a separate declared pace scenario.
    """
    aware(issued_at)
    result = dict(status='no_supported_response', pace=None, profile=None,
        uncertainty_calibrated=False, planner_influence=False)
    if fit is None or fit.get('dynamic') is None:
        return result
    if fit['issued_at'] > issued_at:
        raise ValueError('Future response cannot determine requested pace')
    model = fit['dynamic']
    if (not fit['source_keys'] or not isfinite(model['slope']) or model['slope'] <= 0
            or not isfinite(model['intercept'])):
        raise ValueError('Causal positive response and source identities required')
    hr = HeartRateReference(config['resting_hr'], config['max_hr'], 'frozen coaching-zone scenario')
    bands = []
    for cues in (effort.start_cues, effort.finish_cues):
        zones = {int(z) for z in re.findall(r'\bZ([1-5])\b', ' '.join(cues))}
        if not zones:
            return dict(result, status='unresolved_requested_effort_numerics')
        low = hr.reserve(min(config['zones']['z'+str(z)][0] for z in zones))
        high = hr.reserve(max(config['zones']['z'+str(z)][1] for z in zones))
        if low is None or high is None or low > high:
            raise ValueError('Configured effort bands must fit athlete HR normalization')
        bands.append(Quantity(low, high, (low+high)/2, 'AS163 full zone envelope; center is a coaching scenario'))
    raw_speeds = [((q.low-model['intercept'])/model['slope'],
        (q.high-model['intercept'])/model['slope'],
        (q.estimate-model['intercept'])/model['slope']) for q in bands]
    if min(q[0] for q in raw_speeds) <= 0:
        return dict(result, status='unsupported_effort_response_range', requested_bands=bands,
            speed_scenarios=raw_speeds, reason='Nonpositive inferred speed; no clipping or invented finite pace bound')
    speeds = [Quantity(*q, 'inverted prior speed/HR equilibrium; not maximal pace or confidence') for q in raw_speeds]
    mean = Quantity((speeds[0].low+speeds[1].low)/2, (speeds[0].high+speeds[1].high)/2,
        (speeds[0].estimate+speeds[1].estimate)/2, 'explicit linear effort profile mean')
    keys = tuple(fit['source_keys'])
    profile = PaceProfile(*speeds, issued_at, keys,
        'AS163 declared zone-band profile via shared prior response; not verified execution')
    pace = PaceReference(Quantity(1/mean.high, 1/mean.low, 1/mean.estimate,
        'inverted coaching scenario envelope; not prediction/confidence interval'), issued_at, keys, profile.basis)
    domain = model['speed_domain']
    return dict(result, status='declared_coaching_scenario', pace=pace, profile=profile,
        requested_bands=bands, normalization=asdict(hr), response_source_keys=keys,
        speed_extrapolation=any(q.low < domain[0] or q.high > domain[1] for q in speeds),
        calibrated_effort_execution=False)


@dataclass(frozen=True)
class SpeedPhase:
    kind: str
    seconds: float
    first_speed: float
    last_speed: float
    basis: str

    def __post_init__(self):
        if (self.kind not in {'warmup', 'work', 'recovery', 'cooldown'} or not self.basis
                or not all(isfinite(x) for x in (self.seconds, self.first_speed, self.last_speed))
                or self.seconds <= 0 or min(self.first_speed, self.last_speed) < 0):
            raise ValueError('Explicit finite physical phase/profile required')

    @property
    def meters(self):
        return self.seconds*(self.first_speed+self.last_speed)/2


def command_scenario(prescription, *, point='nominal', profiles=None):
    """Resolve primary units and correlated pace/duration before forecasting.

    A distance command lasts distance/mean speed, not an independent duration
    box endpoint. Timed commands retain their authored time/range. Gradual work
    starts at the easy scenario and finishes at 2*work_mean-easy, preserving the
    compiler's mean pace, numerical duration and distance. This shape is explicit.
    """
    if point not in {'lower_speed', 'nominal', 'higher_speed'}:
        raise ValueError('Declared scenario point required')
    def speed(reference):
        q = reference.seconds_per_meter
        return 1/(q.high if point == 'lower_speed' else q.low if point == 'higher_speed' else q.estimate)
    easy, work = speed(prescription.easy_pace), speed(prescription.work_pace)
    profiles = profiles or {}
    if any(type(i) is not int or not 0 <= i < len(prescription.phases) for i in profiles):
        raise ValueError('Profile must identify an exact compiled phase')
    phases = []
    for i, p in enumerate(prescription.phases):
        first = last = work if p.kind == 'work' else easy
        basis = 'frozen phase mean-speed scenario'
        if p.effort.purpose == 'gradual' and i not in profiles:
            first, last = easy, 2*work-easy
            if last < first:
                raise ValueError('Gradual faster finish needs a compatible explicit pace/profile scenario')
            basis = 'AS163 declared linear profile; mean-preserving, not measured or fitted'
        if i in profiles:
            profile = profiles[i]
            if not isinstance(profile, PaceProfile) or profile.issued_at > prescription.request.issued_at:
                raise ValueError('Causal explicit profile required')
            attr = 'low' if point == 'lower_speed' else 'high' if point == 'higher_speed' else 'estimate'
            first, last = (getattr(q, attr) for q in (profile.first_speed, profile.last_speed))
            expected = work if p.kind == 'work' else easy
            if abs((first+last)/2-expected) > 1e-9:
                raise ValueError('Profile cannot change frozen compiled mean speed/dose')
            basis = profile.basis
        mean = (first+last)/2
        if p.primary.unit == 'time':
            q = p.primary.amount
            # Coupled scenario: slower execution with the longer allowed effort,
            # faster with the shorter. All three are examples, not extrema.
            seconds = q.high if point == 'lower_speed' else q.low if point == 'higher_speed' else q.estimate
        else:
            seconds = p.meters.estimate/mean
        phases.append(SpeedPhase(p.kind, seconds, first, last, basis))
    return tuple(phases)


def forecast_physical(phases, fit, *, issued_at, accounting_key, initial_reserve, step_seconds=.5):
    """One label-independent forward model, carrying state through every phase.

    Initial HR is explicitly declared; no target observations enter this API.
    No invalid HR clipping, hidden fallback, fabricated zero or phase reset.
    """
    aware(issued_at)
    phases = tuple(phases)
    result = dict(version=VERSION, issued_at=issued_at, accounting_key=accounting_key,
        status='no_supported_response', predicted_load=None, phases=[],
        source_keys=(), uncertainty_calibrated=False, planner_influence=False)
    if not accounting_key or not phases or any(not isinstance(p, SpeedPhase) for p in phases):
        raise ValueError('Identified nonempty physical query required')
    if not isfinite(initial_reserve) or not 0 <= initial_reserve <= 1:
        raise ValueError('Explicit valid initial HR reserve required')
    if fit is None:
        return result
    if (fit['issued_at'] > issued_at or fit['accounting_key'] != accounting_key
            or len(set(fit['source_keys'])) != len(fit['source_keys'])):
        raise ValueError('Causal compatible response fit required')
    result['source_keys'] = tuple(fit['source_keys'])
    model = fit.get('dynamic')
    if model is None:
        return result
    if (not all(isfinite(model[k]) for k in ('intercept', 'slope', 'tau_seconds'))
            or model['slope'] <= 0 or model['tau_seconds'] <= 0):
        raise ValueError('Positive finite shared response relation required')
    if not isfinite(step_seconds) or step_seconds <= 0:
        raise ValueError('Positive response integration step required')
    program = []
    for p in phases:
        at = 0.
        while at < p.seconds:
            end = min(p.seconds, at+step_seconds)
            if end <= at:
                raise ValueError('Integration step below time precision')
            v = p.first_speed+(p.last_speed-p.first_speed)*(at+end)/(2*p.seconds)
            program.append((end-at, v))
            at = end
    spans, diagnostics = predict_speed_program(program, model,
        initial_reserve=initial_reserve, step_seconds=step_seconds)
    result.update(response_fit=fit, initial_reserve=initial_reserve,
        diagnostics=diagnostics, normalized_spans=spans)
    if diagnostics['hr_outside_reference']:
        return dict(result, status='predicted_hr_outside_reference')
    at = 0.
    for p in phases:
        end = at+p.seconds
        known = clip_trace(spans, at, end)
        load = integrate_cardio_trace(known, end, compute_density=False)['observed_load']
        result['phases'].append(dict(kind=p.kind, start=at, end=end,
            load=load, rate=load/p.seconds, first_speed=p.first_speed, last_speed=p.last_speed,
            meters=p.meters, basis=p.basis))
        at = end
    return dict(result, status='predicted', predicted_load=fsum(p['load'] for p in result['phases']))


def forecast_prescription(prescription, fit, *, accounting_key, initial_reserve, profiles=None):
    """Exact issued identity and three explicit physical scenarios, no repricing.

    The scenario envelope contains only these evaluations, not every possible
    execution or response-fit parameter. Do not expose it as calibrated bounds.
    A missing fit stays unresolved; it is not an automatic family ban.
    """
    scenarios = {point: forecast_physical(command_scenario(prescription, point=point, profiles=profiles), fit,
        issued_at=prescription.request.issued_at, accounting_key=accounting_key,
        initial_reserve=initial_reserve) for point in ('lower_speed', 'nominal', 'higher_speed')}
    nominal = scenarios['nominal']
    amounts = [r['predicted_load'] for r in scenarios.values() if r['status'] == 'predicted']
    all_supported = len(amounts) == len(scenarios)
    fingerprint = sha256(json.dumps(dict(version=VERSION, command=prescription.key,
        fit=fit, accounting_key=accounting_key, initial_reserve=initial_reserve,
        profiles={str(i): asdict(p) for i, p in (profiles or {}).items()}),
        sort_keys=True, default=str, allow_nan=False).encode()).hexdigest()
    return dict(version=VERSION, prescription_key=prescription.key, fingerprint=fingerprint,
        issued_at=prescription.request.issued_at, nominal=nominal, scenarios=scenarios,
        scenario_envelope=Quantity(min(amounts), max(amounts), nominal['predicted_load'],
            'three evaluated execution scenarios only; not confidence or exhaustive bounds') if all_supported else None,
        all_scenarios_supported=all_supported, uncertainty_calibrated=False, planner_influence=False,
        limits=('Declared pace inputs are not verified requested-effort execution predictions',
            'Gradual profile is a mean-preserving linear scenario, not athlete-calibrated shape',
            'Raw pace is not adjusted for hills, terrain, wind or weather',
            'HR/cardio model does not estimate tendon or muscular readiness',
            'No residual correction, family coefficient, load-to-recovery claim or calendar acceptance'))
