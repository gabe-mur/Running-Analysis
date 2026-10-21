"""Frozen Q3 coaching choices; no search, physiological fit or live installation.

Native recipe structures have different doses. One existing stage scale limits
quality growth; purpose coverage is not a claim of interchangeable adaptations.
All proposals still require Q5 forecast and Q7 whole-program admission checks.
"""
from dataclasses import asdict, dataclass
from datetime import datetime
from hashlib import sha256
import json
from math import ceil, floor, isfinite

from run_analysis.athlete_projection import aware
from run_analysis.staged_progression import tenth
from run_analysis.training_signals import HeartRateReference
from run_analysis.work_stimulus import CappedStimulusTarget
from run_analysis.workload_accounting import Quantity
from .quality_prescription_contract import Measure, QualityRequest, compile_quality
from .quality_purpose_contract import PurposeRequirement, QualityPurposeContract

VERSION = 'quality_dose_policy_v1'


def _allocation_source(stage, policy):
    frozen = json.dumps(dict(stage=asdict(stage), policy=asdict(policy)),
        sort_keys=True, default=str, allow_nan=False)
    return policy.source+';stage:'+stage.source+';allocation:'+sha256(frozen.encode()).hexdigest()


@dataclass(frozen=True)
class QualityDosePolicy:
    source: str = 'AS162 provisional coaching contract; not calibrated physiology'
    sustained_share: float = .5
    sustained_bout_seconds: float = 120.
    reduced_fraction: float = .8
    reference_work_minutes: float = 15.
    faster_reference_minutes: float = 8.
    time_increment_seconds: int = 30

    def __post_init__(self):
        values = (self.sustained_share, self.sustained_bout_seconds,
                  self.reduced_fraction, self.reference_work_minutes, self.faster_reference_minutes)
        if (not self.source or any(not isfinite(v) for v in values)
                or not 0 < self.sustained_share < 1 or not 0 < self.sustained_bout_seconds <= 300
                or round(self.sustained_bout_seconds) != self.sustained_bout_seconds
                or not 0 < self.reduced_fraction < 1
                or min(self.reference_work_minutes, self.faster_reference_minutes) <= 0
                or type(self.time_increment_seconds) is not int or self.time_increment_seconds <= 0):
            raise ValueError('Explicit finite coaching policy required')


@dataclass(frozen=True)
class QualityDoseStage:
    """Caller supplies the current accepted stage; never infer it on refresh."""
    key: str
    issued_at: datetime
    ordinary_miles: float
    quality_work_minutes: float
    cadence_days: float
    source: str

    def __post_init__(self):
        aware(self.issued_at)
        if (not self.key or not self.source or any(not isfinite(v) or v <= 0 for v in
                (self.ordinary_miles, self.quality_work_minutes, self.cadence_days))
                or tenth(self.ordinary_miles) != self.ordinary_miles):
            raise ValueError('Frozen accepted stage dose/cadence and provenance required')


def purpose_contract(stage, *, config, issued_at, easy_pace, faster_pace, policy=QualityDosePolicy()):
    """Freeze common targets from configured zones and accepted stage, not winners.

    Zone3→Zone4 is a coaching exposure ramp, not an inferred lactate threshold.
    Faster exposure ramps from the frozen easy command pace to the separately
    declared faster command pace. Neither an arbitrary easy-speed multiplier nor
    the fastest historical performance becomes a physiological threshold.
    """
    aware(issued_at)
    if stage.issued_at > issued_at:
        raise ValueError('Stage not yet available')
    if any(p.issued_at > issued_at for p in (easy_pace, faster_pace)):
        raise ValueError('Future pace scenarios cannot define current targets')
    hr = HeartRateReference(config['resting_hr'], config['max_hr'], 'frozen current athlete settings')
    onset, full = (hr.reserve(config['zones'][z][0]) for z in ('z3', 'z4'))
    if onset is None or full is None or not onset < full:
        raise ValueError('Configured sustained-effort zones must fit the HR reference')
    scale = min(1., stage.quality_work_minutes/policy.reference_work_minutes)
    source = _allocation_source(stage, policy)
    sustained = CappedStimulusTarget('sustained work exposure', 'hrr', onset, full,
        policy.reference_work_minutes*scale, source+';configured Z3/Z4 onset anchors')
    easy_speed, fast_speed = (1/p.seconds_per_meter.estimate for p in (easy_pace, faster_pace))
    if not easy_speed < fast_speed:
        raise ValueError('Faster purpose needs a distinct explicitly sourced faster-effort scenario')
    refs = json.dumps(dict(easy=asdict(easy_pace), faster=asdict(faster_pace)), sort_keys=True, default=str)
    faster = CappedStimulusTarget('faster work exposure', 'speed_mps', easy_speed, fast_speed,
        policy.faster_reference_minutes*scale, source+';frozen coaching command pace anchors:'+refs)
    return QualityPurposeContract(issued_at, stage.key, stage.cadence_days, (
        PurposeRequirement('sustained', sustained, policy.sustained_share,
            policy.sustained_bout_seconds, source),
        PurposeRequirement('faster', faster, 1-policy.sustained_share, 0., source),
    ), source)


def _time(seconds, source):
    return Measure('time', Quantity.point(seconds, source))


def recipe_requests(stage, *, issued_at, policy=QualityDosePolicy(),
                    distance_rep_seconds=None, open_recovery_seconds=None, scenario_source=None):
    """Ordinary/reduced authored commands; do not give every recipe equal minutes.

    Distance variants keep 400/500m commands; the caller supplies a frozen causal
    duration scenario to allocate rep count. Open recovery requires an explicit
    scenario and remains open to the runner. No pooled all-quality pace is fitted
    here. Progression uses ordinary stage distance, not a generic work clock.
    """
    aware(issued_at)
    if stage.issued_at > issued_at:
        raise ValueError('Stage not yet available')
    if ((distance_rep_seconds is not None or open_recovery_seconds is not None) and not scenario_source
            or distance_rep_seconds is not None and (set(distance_rep_seconds) != {400, 500}
                or any(not isfinite(v) or v <= 0 for v in distance_rep_seconds.values()))
            or open_recovery_seconds is not None and (not isfinite(open_recovery_seconds) or open_recovery_seconds <= 0
                or round(open_recovery_seconds) != open_recovery_seconds)):
        raise ValueError('Explicit sourced duration scenarios required')
    scale = min(1., stage.quality_work_minutes/policy.reference_work_minutes)
    source = _allocation_source(stage, policy)
    requests = []
    def add(family, variant, work, recovery, *, final=False, opened=False, scenario=False):
        rests = len(work) if final else len(work)-1
        recoveries = (_time(recovery, source),)*rests if rests else ()
        requests.append(QualityRequest(family, variant, issued_at,
            source+(';scenario:'+scenario_source if scenario else ''), tuple(work),
            recoveries,
            _time(720, source+';authored easy warm-up'), _time(600, source+';authored cool-down'),
            open_recovery=opened and bool(rests)))
    def rounded(seconds):
        step = policy.time_increment_seconds
        return max(step, floor(seconds/step+.5)*step)
    for reduced in (False, True):
        fraction = scale*(policy.reduced_fraction if reduced else 1.)
        suffix = 'reduced' if reduced else 'ordinary'
        # Keep native one-minute fast reps. Integer-count quantization is
        # exposed, not corrected by pretending the display/scored doses differ.
        count = max(1, ceil(policy.faster_reference_minutes*fraction))
        add('short_intervals', 'timed_pickups_'+suffix, (_time(60, source),)*count, 90, final=True)
        if distance_rep_seconds is not None:
            for metres in (400, 500):
                count = max(1, ceil(policy.faster_reference_minutes*60*fraction/distance_rep_seconds[metres]))
                add('short_intervals', f'{metres}m_'+suffix,
                    (Measure('distance', Quantity.point(metres, source+';explicit distance variant')),)*count,
                    90, final=True, scenario=True)
        # Fewer long reps at smaller stages; never shrink every rep to a tiny
        # interval solely to preserve the authored repetition count.
        budget = policy.reference_work_minutes*60*fraction
        if budget >= policy.sustained_bout_seconds:
            count = min(3, max(1, floor(budget/policy.sustained_bout_seconds)))
            seconds = max(policy.sustained_bout_seconds, rounded(budget/count))
            add('long_intervals', 'timed_long_'+suffix, (_time(seconds, source),)*count, 120)
        seconds = rounded(18*60*fraction)
        if seconds >= policy.sustained_bout_seconds:
            add('threshold', 'continuous_'+suffix, (_time(seconds, source),), 0)
        miles = tenth(stage.ordinary_miles*(policy.reduced_fraction if reduced else 1.))
        if miles > 0:
            requests.append(QualityRequest('progression', 'second_half', issued_at,
                source+';'+suffix, (Measure('portion', Quantity.point(.5, 'authored second-half progression')),),
                total_miles=miles))
        if open_recovery_seconds is not None:
            count = max(1, ceil(policy.faster_reference_minutes*fraction))
            surge = Measure('time', Quantity(45, 90, 60, source+';authored45-90s range, nominal forecast60s'))
            add('fartlek', 'surges_'+suffix, (surge,)*count,
                open_recovery_seconds, final=True, opened=True, scenario=True)
            add('hill_repeats', 'uphill_'+suffix, (_time(45, source),)*count,
                open_recovery_seconds, opened=True, scenario=True)
    # At a very small stage rounding can produce identical ordinary/reduced
    # commands. Keep one rather than claiming a fake meaningful reduction.
    unique = {}
    for r in requests:
        identity = (r.family, tuple(r.work), tuple(r.recoveries), r.total_miles, r.open_recovery,
                    r.variant.split('_')[0])
        unique.setdefault(identity, r)
    return tuple(unique.values())


def compile_dose_menu(stage, *, contract, easy_pace, purpose_paces, policy=QualityDosePolicy(),
                      open_recovery_seconds=None, scenario_source=None):
    """Freeze candidate byte identities; separate purpose pace inputs, not one pool.

    Q5 must validate or qualify these supplied effort scenarios. This does not
    admit candidates to live scheduling or assert equal cost for equal credit.
    """
    if (contract.stage_key != stage.key or contract.cadence_days != stage.cadence_days
            or contract.source != _allocation_source(stage, policy)
            or set(purpose_paces) != {'sustained', 'faster', 'gradual'}):
        raise ValueError('Common stage contract and distinct declared effort scenarios required')
    faster = purpose_paces['faster']
    durations = {metres: metres*faster.seconds_per_meter.estimate for metres in (400, 500)}
    requests = recipe_requests(stage, issued_at=contract.issued_at, policy=policy,
        distance_rep_seconds=durations, open_recovery_seconds=open_recovery_seconds,
        scenario_source=scenario_source or faster.basis)
    compiled = []
    for request in requests:
        purpose = ('gradual' if request.family == 'progression' else 'sustained'
                   if request.family in {'long_intervals', 'threshold'} else 'faster')
        compiled.append(compile_quality(request, easy_pace=easy_pace, work_pace=purpose_paces[purpose]))
    key = sha256(json.dumps(dict(stage=asdict(stage), contract=contract.key, policy=asdict(policy),
        commands=[p.key for p in compiled]), sort_keys=True, default=str, allow_nan=False).encode()).hexdigest()
    families = {'short_intervals', 'long_intervals', 'threshold', 'progression', 'fartlek', 'hill_repeats'}
    unresolved = () if open_recovery_seconds is not None else ('fartlek', 'hill_repeats')
    return dict(version=VERSION, key=key, prescriptions=tuple(compiled), contract=contract,
        unresolved_families=unresolved,
        stage_unsupported_families=tuple(sorted(families-{p.request.family for p in compiled}-set(unresolved))),
        forecasting_validated=False, candidate_admission='requires_shared_forecast_and_existing_stage_checks',
        planner_influence=False)
