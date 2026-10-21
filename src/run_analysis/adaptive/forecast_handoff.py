"""Explicit provisional forecast scope; no new predictor or calibrated intervals."""
from dataclasses import asdict, dataclass
from datetime import datetime
from math import fsum

from run_analysis.athlete_projection import aware
from run_analysis.workload_accounting import Quantity
from run_analysis.adaptive.direct_quality_load import KINDS, forecast_direct


@dataclass(frozen=True)
class PhaseRateScenario:
    kind: str
    per_minute: Quantity
    available_at: datetime
    accounting_key: str
    intensity_key: str
    source: str

    def __post_init__(self):
        aware(self.available_at)
        if (self.kind not in KINDS or self.per_minute.estimate is None
                or self.per_minute.estimate <= 0 or not self.accounting_key
                or not self.intensity_key or not self.source.strip()):
            raise ValueError('Typed sourced rate scenario with positive nominal and units required')


@dataclass(frozen=True)
class ForecastScope:
    intensity_key: str
    duration_source: str
    movement_source: str
    mode: str = 'recent_execution_proxy'
    rate_scenarios: tuple[PhaseRateScenario, ...] = ()
    movement_assumption: str = 'continuous_running'

    def __post_init__(self):
        if (not self.intensity_key or not self.duration_source.strip() or not self.movement_source.strip()
                or self.mode not in {'recent_execution_proxy', 'declared_phase_rates'}
                or self.movement_assumption != 'continuous_running'
                or not isinstance(self.rate_scenarios, tuple)
                or len({r.kind for r in self.rate_scenarios}) != len(self.rate_scenarios)):
            raise ValueError('Explicit effort scope, duration/movement provenance and unique phase rates required')


def read_scope(value):
    if value is None:
        return None
    return ForecastScope(**dict(value,rate_scenarios=tuple(PhaseRateScenario(**dict(r,
        per_minute=Quantity(**r['per_minute']),available_at=datetime.fromisoformat(r['available_at'])))
        for r in value['rate_scenarios'])))


def scoped_forecast(layout, history, *, as_of, accounting_key, scope=None):
    """Keep the phase predictor unchanged; fill unknowns only from explicit inputs."""
    result = forecast_direct(layout,history,as_of=as_of,accounting_key=accounting_key)
    if scope is None:
        return result
    rates = {r.kind:r for r in scope.rate_scenarios}
    for rate in rates.values():
        if rate.available_at > as_of or (rate.accounting_key,rate.intensity_key) != (accounting_key,scope.intensity_key):
            raise ValueError('Rate scenario must be available and match units/declared effort')
    for p in result['phases']:
        declared = rates.get(p['kind'])
        if scope.mode == 'declared_phase_rates' or p['rate'] is None:
            p['rate'] = declared.per_minute.estimate/60 if declared else None
            p['load'] = p['rate']*(p['end']-p['start']) if p['rate'] is not None else None
            p['declared_rate_scenario'] = asdict(declared) if declared else None
            # Evidence exists, but it did not determine this phase's selected rate.
            p['source_sessions'] = []
            p['fallback_pooled_nonwork'] = False
        else:
            p['declared_rate_scenario'] = None
    if result['phases'] and all(p['load'] is not None for p in result['phases']):
        result.update(status='predicted',predicted_load=fsum(p['load'] for p in result['phases']))
    else:
        result.update(status='unresolved_phase_rate',predicted_load=None)
    return result


def planning_handoff(phases, target, history, scope):
    """Declared command stays separate from workload scenarios and response error."""
    base = dict(version='forecast_handoff_v1',status='unresolved',usable=False,
        scope=asdict(scope) if scope else None,uncertainty_calibrated=False,
        confidence_interval=None,response_allowance_derived=False,
        limits=['Recent execution is a proxy, not requested-effort/route/weather conditioning',
                'Duration boxes and source extrema are scenarios, not predictive confidence',
                'Running duration assumes running/jogging throughout; planned standing/walking needs a separate movement contract',
                'No implicit rate prior; unsupported candidates remain explicit'])
    if scope is None:
        return dict(base,reason='forecast_scope_not_declared')
    if target['predicted_load'] is None:
        return dict(base,reason='phase_rate_unavailable',unresolved_phases=[p['kind'] for p in target['phases'] if p['rate'] is None])
    if any(p.seconds.low is None for p in phases):
        return dict(base,reason='duration_scenarios_not_declared')
    accepted = [h for h in history if h.key in target['source_keys']]
    bounds, diagnostics = [], []
    for command, p in zip(phases,target['phases']):
        scenario = p.get('declared_rate_scenario')
        if scenario:
            q = scenario['per_minute']
            low,high = q['low']/60,q['high']/60
            source = 'declared_rate_scenario'
            extrapolated = None
        else:
            observations = [o for h in accepted for o in h.phases if o.load is not None and
                (o.kind != 'work' if p['fallback_pooled_nonwork'] else o.kind == p['kind'])]
            rates = [o.load/o.known_seconds for o in observations]
            if not rates:
                return dict(base,reason='phase_rate_bounds_unavailable')
            low,high = min(rates),max(rates)
            source = 'recent_observed_rate_extrema'
            extrapolated = not min(o.seconds for o in observations) <= command.seconds.target <= max(o.seconds for o in observations)
        bounds.append((command.seconds.low*low,command.seconds.high*high))
        diagnostics.append(dict(kind=p['kind'],source=source,nominal_rate_per_minute=p['rate']*60,
            pooled_nonwork=p['fallback_pooled_nonwork'],outside_duration_support=extrapolated))
    nominal = target['predicted_load']
    # Protect only floating roundoff; the extrema enclose every convex rate estimate.
    cardio = Quantity(min(nominal,fsum(b[0] for b in bounds)),max(nominal,fsum(b[1] for b in bounds)),
                      nominal,'frozen_phase_duration_and_rate_scenarios_not_confidence')
    duration = Quantity(fsum(p.seconds.low for p in phases)/60,fsum(p.seconds.high for p in phases)/60,
                        fsum(p.seconds.target for p in phases)/60,'declared_running_duration_scenarios')
    return dict(base,status='provisional',usable=True,reason=None,cardio=asdict(cardio),
                running_minutes=asdict(duration),phase_diagnostics=diagnostics,
                scope_basis=scope.mode,nominal_is_not_range_midpoint=True)
