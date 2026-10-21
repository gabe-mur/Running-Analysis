"""Unconnected block contract; one unit compiler and one physical forecast.

Family is metadata, not an effort lookup. Author every phase with its own causal
pace, cues and optional varying-speed profile. Old v2 contracts stay readable
and unchanged; explicit v3 adaptation creates a NEW identity, not a migration
of an already issued workout. No recipe allocation or predictor calibration.
"""
from dataclasses import asdict, dataclass
from datetime import datetime
from hashlib import sha256
import json
from math import fsum, isfinite

from run_analysis.athlete_projection import aware
from run_analysis.staged_progression import MILE, tenth
from run_analysis.web.schemas import WorkoutStep, WorkoutType
from run_analysis.workload_accounting import Quantity
from .quality_prescription_contract import (
    Effort, Measure, NumericPhase, PaceReference, clock, compile_numeric_phase, read_prescription,
)
from .requested_quality_forecast import PaceProfile, SpeedPhase, command_scenario, forecast_physical

VERSION = 'quality_block_contract_v3'
POINTS = ('lower_speed', 'nominal', 'higher_speed')


def _speed(pace, point):
    q = pace.seconds_per_meter
    return 1/(q.high if point == 'lower_speed' else q.low if point == 'higher_speed' else q.estimate)


@dataclass(frozen=True)
class BlockCommand:
    kind: str
    primary: Measure
    effort: Effort
    pace: PaceReference
    instruction: str
    profile: PaceProfile | None = None
    open_recovery: bool = False

    def __post_init__(self):
        if (self.kind not in {'warmup', 'work', 'recovery', 'cooldown'}
                or not isinstance(self.primary, Measure) or not isinstance(self.effort, Effort)
                or not isinstance(self.pace, PaceReference) or not isinstance(self.instruction, str)
                or not self.instruction.strip() or type(self.open_recovery) is not bool
                or self.open_recovery and self.kind != 'recovery'):
            raise ValueError('Complete explicit block instruction, effort, units and pace required')
        if self.effort.purpose == 'gradual' and self.profile is None:
            raise ValueError('Gradual block requires an explicit sourced profile, not a family-based guess')
        if self.profile is not None:
            if not isinstance(self.profile, PaceProfile):
                raise ValueError('Typed explicit speed profile required')
            for point, attr in zip(POINTS, ('low', 'estimate', 'high')):
                mean = (getattr(self.profile.first_speed, attr)+getattr(self.profile.last_speed, attr))/2
                if abs(mean-_speed(self.pace, point)) > 1e-9:
                    raise ValueError('Block profile must preserve its own frozen mean pace/dose')


@dataclass(frozen=True)
class BlockRequest:
    family: str
    variant: str
    title: str
    workout_type: str
    issued_at: datetime
    source: str
    blocks: tuple[BlockCommand, ...]
    total_miles: float | None = None

    def __post_init__(self):
        aware(self.issued_at)
        if (any(not isinstance(s, str) or not s.strip() for s in
                (self.family, self.variant, self.title, self.source))
                or self.workout_type not in {WorkoutType.INTERVALS.value, WorkoutType.TEMPO_THRESHOLD.value}
                or not isinstance(self.blocks, tuple) or not self.blocks
                or any(not isinstance(b, BlockCommand) for b in self.blocks)):
            raise ValueError('Identified immutable block request required')
        if (self.blocks[0].kind != 'warmup' or not any(b.kind == 'work' for b in self.blocks)
                or any(b.kind == 'warmup' for b in self.blocks[1:])
                or any(b.kind == 'cooldown' for b in self.blocks[:-1])):
            raise ValueError('Ordered warm-up, work, recovery and optional final cooldown required')
        if any(b.pace.issued_at > self.issued_at or b.profile is not None
               and b.profile.issued_at > self.issued_at for b in self.blocks):
            raise ValueError('Future block pace/profile cannot alter the command')
        portions = any(b.primary.unit == 'portion' for b in self.blocks)
        if portions:
            if (self.total_miles is None or not isfinite(self.total_miles) or self.total_miles <= 0
                    or tenth(self.total_miles) != self.total_miles
                    or any(b.primary.unit != 'portion' or b.primary.amount.low != b.primary.amount.high
                           for b in self.blocks)
                    or abs(fsum(b.primary.amount.estimate for b in self.blocks)-1) > 1e-12):
                raise ValueError('Exact portion commands must partition a committed tenth-mile total')
        elif self.total_miles is not None:
            raise ValueError('Do not add an independent exact total to timed/distance blocks')


def _primary_text(m):
    q = m.amount
    if m.unit == 'portion':
        return f'{q.estimate*100:g}% of the run'
    if q.low == q.high:
        return clock(q.estimate) if m.unit == 'time' else f'{q.estimate:g} m'
    if m.unit == 'distance':
        return f'{q.low:g}–{q.high:g} m'
    return f'{q.low:g}–{q.high:g} seconds' if q.high < 120 else f'{clock(q.low)}–{clock(q.high)}'


@dataclass(frozen=True)
class BlockPrescription:
    request: BlockRequest
    phases: tuple[NumericPhase, ...]
    miles: float
    distance_basis: str
    version: str = VERSION

    def __post_init__(self):
        expected = tuple(compile_numeric_phase(b.kind, b.primary, b.pace, b.effort,
            total_miles=self.request.total_miles, open_recovery=b.open_recovery) for b in self.request.blocks)
        if (self.version != VERSION or self.phases != expected or not self.distance_basis
                or not isfinite(self.miles) or self.miles <= 0
                or self.miles != tenth(fsum(p.meters.estimate for p in expected)/MILE)):
            raise ValueError('Compiled blocks must agree exactly with their own frozen commands')

    @property
    def key(self):
        encoded = json.dumps(asdict(self), sort_keys=True, default=lambda v:v.isoformat(), allow_nan=False)
        return 'quality:'+self.request.family+':'+sha256(encoded.encode()).hexdigest()[:16]

    def steps(self):
        """Compact identical repeats; keep heterogeneous blocks in exact order.

        No family-based intensity sentence, machine-phase rewrite or averaging
        that turns two different efforts into one misleading instruction.
        """
        blocks, phases, result, i = self.request.blocks, self.phases, [], 0
        while i < len(blocks):
            start, count, recoveries = i, 1, 0
            if blocks[i].kind == 'work' and i+1 < len(blocks) and blocks[i+1].kind == 'recovery':
                work, recovery = blocks[i], blocks[i+1]
                end = i
                while end+1 < len(blocks) and blocks[end] == work and blocks[end+1] == recovery:
                    recoveries += 1
                    end += 2
                    if end < len(blocks) and blocks[end] == work:
                        count += 1
                    else:
                        break
                # Include the final work when there are only between-rep rests.
                i = end+1 if end < len(blocks) and blocks[end] == work else end
            else:
                i += 1
            chosen, body = blocks[start], phases[start:i]
            instruction = f'{count} × ' if count > 1 else ''
            instruction += f'{_primary_text(chosen.primary)} · {chosen.instruction}'
            if recoveries:
                recovery = blocks[start+1]
                where = 'after each effort' if recoveries == count else 'between efforts'
                instruction += (f' Recovery by feel {where}: {recovery.instruction} {_primary_text(recovery.primary)} is the forecast scenario.'
                    if recovery.open_recovery else
                    f' {_primary_text(recovery.primary)} · {recovery.instruction} {where}.')
            result.append(WorkoutStep(instruction=instruction, phase=chosen.kind,
                prescription_basis=chosen.primary.unit,
                duration_minutes=fsum(p.seconds.estimate for p in body)/60,
                distance_miles=fsum(p.meters.estimate for p in body)/MILE,
                repetitions=count if count > 1 else None,
                work_duration_minutes=phases[start].seconds.estimate/60 if count > 1 else None,
                work_duration_range_minutes=(chosen.primary.amount.low/60, chosen.primary.amount.high/60)
                    if chosen.primary.unit == 'time' and chosen.primary.amount.low != chosen.primary.amount.high else None,
                recovery_duration_minutes=phases[start+1].seconds.estimate/60 if recoveries else None,
                target_zones=list(dict.fromkeys((*chosen.effort.start_cues, *chosen.effort.finish_cues)))))
        return tuple(result)

    def definition(self):
        return dict(version=self.version, family=self.request.family, variant=self.request.variant,
            title=self.request.title, workout_type=self.request.workout_type,
            zones=list(dict.fromkeys(c for b in self.request.blocks for c in
                (*b.effort.start_cues, *b.effort.finish_cues))),
            structure=[s.model_dump(mode='json') for s in self.steps()],
            numeric_contract=json.loads(json.dumps(asdict(self), default=lambda v:v.isoformat())),
            distance_basis=self.distance_basis)


def compile_blocks(request):
    phases = tuple(compile_numeric_phase(b.kind, b.primary, b.pace, b.effort,
        total_miles=request.total_miles, open_recovery=b.open_recovery) for b in request.blocks)
    return BlockPrescription(request, phases, tenth(fsum(p.meters.estimate for p in phases)/MILE),
        'exact committed total; explicitly partitioned portions' if request.total_miles is not None
        else 'committed tenth-mile extent estimate; primary block commands govern execution')


def block_scenario(prescription, *, point='nominal'):
    if point not in POINTS:
        raise ValueError('Declared scenario point required')
    phases = []
    for b, p in zip(prescription.request.blocks, prescription.phases):
        first = last = _speed(b.pace, point)
        basis = 'frozen phase mean-speed scenario'
        if b.profile is not None:
            attr = dict(lower_speed='low', nominal='estimate', higher_speed='high')[point]
            first, last = (getattr(q, attr) for q in (b.profile.first_speed, b.profile.last_speed))
            basis = b.profile.basis
        q = b.primary.amount
        seconds = (q.high if point == 'lower_speed' else q.low if point == 'higher_speed' else q.estimate
            ) if b.primary.unit == 'time' else p.meters.estimate/((first+last)/2)
        phases.append(SpeedPhase(b.kind, seconds, first, last, basis))
    return tuple(phases)


def forecast_blocks(prescription, fit, *, accounting_key, initial_reserve):
    scenarios = {point:forecast_physical(block_scenario(prescription, point=point), fit,
        issued_at=prescription.request.issued_at, accounting_key=accounting_key,
        initial_reserve=initial_reserve) for point in POINTS}
    nominal = scenarios['nominal']
    loads = [s['predicted_load'] for s in scenarios.values() if s['status'] == 'predicted']
    fingerprint = sha256(json.dumps(dict(version=VERSION, command=prescription.key,
        fit=fit, accounting_key=accounting_key, initial_reserve=initial_reserve),
        sort_keys=True, default=str, allow_nan=False).encode()).hexdigest()
    return dict(version=VERSION, prescription_key=prescription.key, fingerprint=fingerprint,
        issued_at=prescription.request.issued_at, nominal=nominal, scenarios=scenarios,
        all_scenarios_supported=len(loads)==3,
        scenario_envelope=Quantity(min(loads), max(loads), nominal['predicted_load'],
            'three evaluated execution scenarios only; not confidence or exhaustive bounds') if len(loads)==3 else None,
        uncertainty_calibrated=False, planner_influence=False,
        limits=('Explicit block pace/profile is not a validated effort-to-execution predictor',
            'Same shared raw-speed response; terrain, drift, muscle/tendon readiness remain unresolved',
            'No recipe allocation, quality quota, calendar admission or stability acceptance'))


def read_blocks(value):
    def pace(p):
        return PaceReference(Quantity(**p['seconds_per_meter']), datetime.fromisoformat(p['issued_at']),
            tuple(p['source_keys']), p['basis'])
    def effort(e):
        return Effort(tuple(e['start_cues']), tuple(e['finish_cues']), e['purpose'])
    blocks = []
    r = value['request']
    for b in r['blocks']:
        profile = b['profile']
        profile = PaceProfile(Quantity(**profile['first_speed']), Quantity(**profile['last_speed']),
            datetime.fromisoformat(profile['issued_at']), tuple(profile['source_keys']), profile['basis']) if profile else None
        blocks.append(BlockCommand(b['kind'], Measure(b['primary']['unit'], Quantity(**b['primary']['amount'])),
            effort(b['effort']), pace(b['pace']), b['instruction'], profile, b['open_recovery']))
    request = BlockRequest(**dict(r, issued_at=datetime.fromisoformat(r['issued_at']), blocks=tuple(blocks)))
    phases = tuple(NumericPhase(**dict(p, primary=Measure(p['primary']['unit'], Quantity(**p['primary']['amount'])),
        seconds=Quantity(**p['seconds']), meters=Quantity(**p['meters']), effort=effort(p['effort']))) for p in value['phases'])
    return BlockPrescription(request, phases, value['miles'], value['distance_basis'], value['version'])


def read_quality_contract(value):
    """Read frozen versions explicitly; never reissue or silently upgrade v2."""
    if value.get('version') == VERSION:
        return read_blocks(value)
    if value.get('version') == 'quality_prescription_contract_v2':
        return read_prescription(value)
    raise ValueError('Unsupported quality contract version')


def adapt_v2(prescription):
    """Explicit comparator, NOT a rewrite of old issued bytes or identities."""
    scenarios = {point:command_scenario(prescription, point=point) for point in POINTS}
    blocks = []
    for i, p in enumerate(prescription.phases):
        reference = prescription.work_pace if p.kind == 'work' else prescription.easy_pace
        profile = None
        if p.effort.purpose == 'gradual':
            def speeds(which):
                values = [getattr(scenarios[point][i], which) for point in POINTS]
                return Quantity(values[0], values[2], values[1], 'AS163 preserved v2 scenario, not calibrated execution')
            profile = PaceProfile(speeds('first_speed'), speeds('last_speed'), prescription.request.issued_at,
                tuple(dict.fromkeys((*prescription.easy_pace.source_keys, *prescription.work_pace.source_keys))),
                'AS163 declared linear profile; mean-preserving, not measured or fitted')
        cues = ' → '.join(dict.fromkeys((*p.effort.start_cues, *p.effort.finish_cues)))
        terrain = ' uphill' if prescription.request.family == 'hill_repeats' and p.kind == 'work' else ''
        instruction = ('Gradually build' if p.effort.purpose == 'gradual' else 'Run')+terrain+' at '+cues+'.'
        if prescription.request.family == 'hill_repeats':
            if p.kind == 'recovery':
                instruction = 'Jog gently downhill until breathing and form are composed.'
            elif p.kind == 'warmup':
                instruction = 'Warm up easily on flat terrain.'
        elif prescription.request.family == 'fartlek' and p.kind == 'work':
            instruction = 'Use landmarks at '+cues+'.'
        blocks.append(BlockCommand(p.kind, p.primary, p.effort, reference,
            instruction, profile, p.open_recovery))
    d = prescription.definition()
    return compile_blocks(BlockRequest(prescription.request.family, prescription.request.variant,
        d['title'], d['workout_type'], prescription.request.issued_at,
        'explicit v2 comparator:'+prescription.key, tuple(blocks), prescription.request.total_miles))
