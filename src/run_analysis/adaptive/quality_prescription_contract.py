"""Unit-aware quality commands; deliberately not installed in the live calendar.

Allocation belongs to the accepted stage, forecasting to the shared predictor.
This compiler freezes supplied commands, causal pace scenarios and authored
effort cues. It does not assert that equal work minutes have equal adaptations.
"""
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from hashlib import sha256
import json
from math import isfinite
from statistics import median

from run_analysis.athlete_projection import aware
from run_analysis.recommendation import _quality_prescription
from run_analysis.staged_progression import MILE, tenth
from run_analysis.web.schemas import QualitySessionType, WorkoutStep
from run_analysis.workload_accounting import Quantity

VERSION = 'quality_prescription_contract_v2'


def clock(seconds):
    seconds = round(seconds)
    minutes, remainder = divmod(seconds, 60)
    return f'{minutes}:{remainder:02d}' if remainder and minutes else (
        f'{minutes} min' if minutes else f'{remainder} seconds')


@dataclass(frozen=True)
class Measure:
    unit: str  # time=seconds, distance=metres, portion=fraction of total distance
    amount: Quantity

    def __post_init__(self):
        if (self.unit not in {'time', 'distance', 'portion'} or self.amount.estimate is None
                or self.amount.low <= 0 or self.unit == 'portion' and self.amount.high > 1):
            raise ValueError('A positive primary target and explicit units are required')
        if self.unit in {'time', 'distance'} and self.amount.estimate != round(self.amount.estimate):
            raise ValueError('Commit readable whole-second or whole-metre commands before prediction')


@dataclass(frozen=True)
class PaceReference:
    seconds_per_meter: Quantity
    issued_at: datetime
    source_keys: tuple[str, ...]
    basis: str

    def __post_init__(self):
        aware(self.issued_at)
        if (self.seconds_per_meter.low <= 0 or self.seconds_per_meter.estimate is None
                or not self.source_keys or not isinstance(self.source_keys, tuple)
                or len(set(self.source_keys)) != len(self.source_keys) or not self.basis):
            raise ValueError('Positive causal pace evidence and provenance are required')


def causal_paces(history, *, as_of, eligible_keys, history_days=30):
    """Reuse recent observed bout and aerobic pace, never the target run's pace.

    Whole-quality pace is a disclosed fallback, not an interval-pace claim.
    No work history returns no work reference; the caller must declare a prior.
    """
    aware(as_of)
    eligible = [r for r in history if r.key in eligible_keys and as_of-timedelta(days=history_days) <= r.start
                and r.end <= as_of and r.available_at <= as_of and r.miles > 0]
    if len({r.key for r in eligible}) != len(eligible):
        raise ValueError('Duplicate pace source session')
    aerobic, work, fallback = [], [], []
    for run in eligible:
        seconds = run.demand[0].estimate*60 if run.demand[0].estimate is not None else None
        if seconds is None or seconds <= 0:
            continue
        # These durations use the same continuous-running assumption as the
        # existing phase adapter; stops/standing require a separate contract.
        if run.role in {'easy', 'long'}:
            aerobic.append((run.key, seconds/(run.miles*MILE)))
        elif run.role == 'quality':
            fallback.append((run.key, seconds/(run.miles*MILE)))
            bouts = getattr(run, 'bouts', ())
            valid = [b for b in bouts if b.distance_meters and b.distance_meters > 0]
            if valid and getattr(run, 'work_evidence_basis', 'unknown') != 'unknown':
                # One vote per run, not one vote per repetition.
                work.append((run.key, sum(b.work_seconds for b in valid)/sum(b.distance_meters for b in valid)))
    def reference(rows, basis):
        if not rows:
            return None
        values = [value for _, value in rows]
        return PaceReference(Quantity(min(values), max(values), median(values), basis),
            as_of, tuple(key for key, _ in rows), basis)
    return dict(easy=reference(aerobic, 'recent continuous aerobic pace'),
        work=reference(work or fallback, 'recent measured work-bout pace' if work
            else 'whole-quality pace fallback; work-speed uncertainty unresolved'))


@dataclass(frozen=True)
class Effort:
    start_cues: tuple[str, ...]
    finish_cues: tuple[str, ...]
    purpose: str  # easy, purposeful, or gradual (not a flat hard block)

    def __post_init__(self):
        if (not self.start_cues or not self.finish_cues
                or not isinstance(self.start_cues, tuple) or not isinstance(self.finish_cues, tuple)
                or self.purpose not in {'easy', 'purposeful', 'gradual'}):
            raise ValueError('Explicit effort cues and purpose are required')


@dataclass(frozen=True)
class NumericPhase:
    kind: str
    primary: Measure
    seconds: Quantity
    meters: Quantity
    effort: Effort
    extent_basis: str
    open_recovery: bool = False

    def __post_init__(self):
        if (self.kind not in {'warmup', 'work', 'recovery', 'cooldown'}
                or any(q.estimate is None or q.low <= 0 for q in (self.seconds, self.meters))
                or not self.extent_basis or self.open_recovery and self.kind != 'recovery'):
            raise ValueError('Complete positive phase extents and provenance are required')


@dataclass(frozen=True)
class QualityRequest:
    family: str
    variant: str
    issued_at: datetime
    source: str
    work: tuple[Measure, ...]
    recoveries: tuple[Measure, ...] = ()
    warmup: Measure | None = None
    cooldown: Measure | None = None
    total_miles: float | None = None
    open_recovery: bool = False

    def __post_init__(self):
        aware(self.issued_at)
        family = QualitySessionType(self.family)
        if (not self.variant or not self.source or not self.work
                or not isinstance(self.work, tuple) or not isinstance(self.recoveries, tuple)):
            raise ValueError('Identified prescription and explicit work targets are required')
        if family == QualitySessionType.PROGRESSION:
            if (self.variant != 'second_half' or len(self.work) != 1
                    or self.work[0].unit != 'portion' or self.work[0].amount.estimate != .5
                    or self.work[0].amount.low != .5 or self.work[0].amount.high != .5
                    or self.total_miles is None or self.recoveries or self.warmup or self.cooldown):
                raise ValueError('Progression preserves first/second halves of the committed total distance')
        elif (any(m.unit == 'portion' for m in (*self.work, *self.recoveries,
                    *((self.warmup,) if self.warmup else ()), *((self.cooldown,) if self.cooldown else ())))
                or len(self.recoveries) not in {len(self.work)-1, len(self.work)}):
            raise ValueError('Supply every between-work recovery, with optional final recovery')
        if family == QualitySessionType.THRESHOLD and (len(self.work) != 1 or self.recoveries):
            raise ValueError('Threshold is continuous work, not arbitrary repeated intervals')
        if self.total_miles is not None and (not isfinite(self.total_miles)
                or self.total_miles <= 0 or tenth(self.total_miles) != self.total_miles):
            raise ValueError('Total mileage must be committed to one decimal')
        if self.open_recovery and not self.recoveries:
            raise ValueError('Open recovery still needs an explicit forecast scenario')


@dataclass(frozen=True)
class QualityPrescription:
    request: QualityRequest
    phases: tuple[NumericPhase, ...]
    easy_pace: PaceReference
    work_pace: PaceReference
    miles: float
    distance_basis: str
    version: str = VERSION

    def __post_init__(self):
        if (self.version != VERSION or not self.phases or not isinstance(self.phases, tuple)
                or self.phases[0].kind != 'warmup' or not any(p.kind == 'work' for p in self.phases)
                or not isfinite(self.miles) or self.miles <= 0 or self.miles != tenth(self.miles)
                or not self.distance_basis or any(p.issued_at > self.request.issued_at
                    for p in (self.easy_pace, self.work_pace))):
            raise ValueError('Valid immutable versioned prescription and causal references required')
        estimated = sum(p.meters.estimate for p in self.phases)/MILE
        if self.miles != tenth(estimated):
            raise ValueError('Committed total must agree with the compiled phase extents')

    @property
    def key(self):
        payload = json.dumps(asdict(self), sort_keys=True, default=lambda v: v.isoformat(), allow_nan=False)
        return 'quality:'+self.request.family+':'+sha256(payload.encode()).hexdigest()[:16]

    def definition(self):
        original = _quality_prescription(QualitySessionType(self.request.family))
        steps = self.steps()
        return dict(version=self.version, family=self.request.family, variant=self.request.variant,
            title=original['title'], workout_type=original['workout_type'].value, zones=original['zones'],
            structure=[s.model_dump(mode='json') for s in steps],
            numeric_contract=json.loads(json.dumps(asdict(self), default=lambda v: v.isoformat())),
            distance_basis=self.distance_basis)

    def steps(self):
        """Authored cues plus exact supplied targets; the existing UI renders these."""
        def measure(m):
            q = m.amount
            if q.low != q.high:
                if m.unit != 'time':
                    return f'{q.low:g}–{q.high:g} m'
                if q.high < 120:
                    return f'{q.low:g}–{q.high:g} seconds'
                if q.low % 60 == q.high % 60 == 0:
                    return f'{q.low/60:g}–{q.high/60:g} min'
                def mmss(seconds):
                    minutes, remainder = divmod(round(seconds), 60)
                    return f'{minutes}:{remainder:02d}'
                return mmss(q.low)+'–'+mmss(q.high)
            return clock(q.estimate) if m.unit == 'time' else f'{q.estimate:g} m'
        def step(phase, instruction):
            return WorkoutStep(instruction=instruction, phase=phase.kind,
                prescription_basis=phase.primary.unit, duration_minutes=phase.seconds.estimate/60,
                distance_miles=phase.meters.estimate/MILE, target_zones=list(phase.effort.finish_cues))
        if self.request.family == 'progression':
            return (step(self.phases[0], 'Run the first half relaxed in Z1/Z2.'),
                step(self.phases[1], 'Gradually increase through the second half, finishing controlled in Z3 without sprinting.'))
        work = [p for p in self.phases if p.kind == 'work']
        recovery = [p for p in self.phases if p.kind == 'recovery']
        uniform = all(p.primary == work[0].primary for p in work)
        targets = (f'{len(work)} × {measure(work[0].primary)}' if len(work) > 1 and uniform
            else ', '.join(measure(p.primary) for p in work))
        cue = dict(short_intervals='at controlled fast effort; do not sprint or chase HR lag',
            long_intervals='at controlled hard effort; finish as smoothly as you start',
            threshold='continuously at threshold effort; start conservatively',
            fartlek='using landmarks at controlled strong effort',
            hill_repeats='uphill at strong controlled effort; stop early if form changes')[self.request.family]
        instruction = f'Run {targets} {cue}.'
        if recovery:
            uniform_recovery = all(p.primary == recovery[0].primary for p in recovery)
            timing = (measure(recovery[0].primary) if uniform_recovery
                      else ', '.join(measure(p.primary) for p in recovery))
            location = 'after each effort' if len(recovery) == len(work) else 'between efforts'
            if self.request.open_recovery:
                instruction += ' Jog gently downhill' if self.request.family == 'hill_repeats' else ' Run easily'
                instruction += ' until breathing and form are composed; recovery timing is a forecast scenario, not a stopwatch requirement.'
            else:
                instruction += f' Jog easily for {timing} {location}'
                instruction += '.' if uniform_recovery else ', in that order.'
        body = (*work, *recovery)
        main = WorkoutStep(instruction=instruction, phase='work', prescription_basis=work[0].primary.unit,
            duration_minutes=sum(p.seconds.estimate for p in body)/60,
            distance_miles=sum(p.meters.estimate for p in body)/MILE,
            repetitions=len(work) if len(work) > 1 else None,
            work_duration_minutes=work[0].seconds.estimate/60 if uniform else None,
            work_duration_range_minutes=(work[0].seconds.low/60, work[0].seconds.high/60)
                if uniform and work[0].primary.unit == 'time' and
                work[0].seconds.low != work[0].seconds.high else None,
            recovery_duration_minutes=recovery[0].seconds.estimate/60 if recovery and
                all(p.primary == recovery[0].primary for p in recovery) else None,
            target_zones=list(work[0].effort.finish_cues))
        warm = 'Easy warm-up on flat terrain' if self.request.family == 'hill_repeats' else 'Easy warm-up'
        return (step(self.phases[0], f'{warm} · {measure(self.phases[0].primary)}.'), main,
            step(self.phases[-1], f'Easy cool-down · {measure(self.phases[-1].primary)}.'))


def compile_numeric_phase(kind, primary, reference, effort, *, total_miles=None, open_recovery=False):
    """Shared unit accounting for old recipes and explicit per-block commands."""
    if not isinstance(reference, PaceReference):
        raise ValueError('Explicit sourced pace reference required')
    pace = reference.seconds_per_meter
    q = primary.amount
    if primary.unit == 'time':
        seconds = q
        meters = Quantity(q.low/pace.high, q.high/pace.low, q.estimate/pace.estimate,
            'distance estimate from frozen pace; time is the command')
    else:
        if primary.unit == 'portion' and total_miles is None:
            raise ValueError('Portion target requires an explicit total distance')
        meters = q if primary.unit == 'distance' else q.scale(total_miles*MILE)
        seconds = Quantity(meters.low*pace.low, meters.high*pace.high, meters.estimate*pace.estimate,
            'duration estimate from frozen pace; distance/portion is the command')
    return NumericPhase(kind, primary, seconds, meters, effort, reference.basis, open_recovery)


def compile_quality(request, *, easy_pace, work_pace):
    """Compile explicit stage allocation; never derive units from the family."""
    if not isinstance(easy_pace, PaceReference) or not isinstance(work_pace, PaceReference):
        raise ValueError('Missing pace evidence needs an explicit sourced scenario, not an implicit guess')
    if any(p.issued_at > request.issued_at for p in (easy_pace, work_pace)):
        raise ValueError('Future pace references cannot alter a prescription')
    original = _quality_prescription(QualitySessionType(request.family))
    main = next(s for s in original['structure'] if s.phase == 'work')
    easy = Effort(('Z1', 'Z2'), ('Z1', 'Z2'), 'easy')
    effort = Effort(tuple(main.target_zones), tuple(main.target_zones), 'purposeful')
    def phase(kind, primary, reference, effort, *, open_recovery=False):
        return compile_numeric_phase(kind, primary, reference, effort,
            total_miles=request.total_miles, open_recovery=open_recovery)
    if request.family == 'progression':
        first = Measure('portion', Quantity.point(.5, 'authored first-half instruction'))
        phases = [phase('warmup', first, easy_pace, easy),
            phase('work', request.work[0], work_pace, Effort(('Z2',), ('Z3',), 'gradual'))]
    else:
        def bookend(provided, index):
            if provided is not None:
                return provided
            authored = original['structure'][index].duration_minutes
            if authored is None:
                raise ValueError('Open warm-up/cooldown needs an explicit sourced scenario')
            return Measure('time', Quantity.point(authored*60, 'existing authored warm-up/cooldown'))
        phases = [phase('warmup', bookend(request.warmup, 0), easy_pace, easy)]
        for i, work in enumerate(request.work):
            phases.append(phase('work', work, work_pace, effort))
            if i < len(request.recoveries):
                phases.append(phase('recovery', request.recoveries[i], easy_pace, easy, open_recovery=request.open_recovery))
        phases.append(phase('cooldown', bookend(request.cooldown, -1), easy_pace, easy))
    estimated = sum(p.meters.estimate for p in phases)/MILE
    miles = request.total_miles if request.total_miles is not None else tenth(estimated)
    if miles <= 0 or request.total_miles is not None and request.family != 'progression':
        raise ValueError('Do not impose an independent exact total on timed/distance recipe phases')
    return QualityPrescription(request, tuple(phases), easy_pace, work_pace, miles,
        'exact committed total; first/second-half portions' if request.family == 'progression'
        else 'committed tenth-mile extent estimate; primary step commands govern execution')


def read_prescription(value):
    """Restore the exact issued command; no query of newer history or recipe."""
    def measure(value):
        return Measure(value['unit'], Quantity(**value['amount'])) if value else None
    r = value['request']
    request = QualityRequest(**dict(r, issued_at=datetime.fromisoformat(r['issued_at']),
        work=tuple(measure(m) for m in r['work']), recoveries=tuple(measure(m) for m in r['recoveries']),
        warmup=measure(r['warmup']), cooldown=measure(r['cooldown'])))
    def pace(p):
        return PaceReference(**dict(p, issued_at=datetime.fromisoformat(p['issued_at']),
            seconds_per_meter=Quantity(**p['seconds_per_meter']), source_keys=tuple(p['source_keys'])))
    phases = tuple(NumericPhase(**dict(p, primary=measure(p['primary']),
        seconds=Quantity(**p['seconds']), meters=Quantity(**p['meters']), effort=Effort(**dict(p['effort'],
            start_cues=tuple(p['effort']['start_cues']), finish_cues=tuple(p['effort']['finish_cues'])))))
        for p in value['phases'])
    return QualityPrescription(request, phases, pace(value['easy_pace']), pace(value['work_pace']),
        value['miles'], value['distance_basis'], value['version'])
