"""Frozen, athlete-relative candidate commands. No calendar or recovery clock.

Numerical defaults are provisional coaching/engineering choices, not fitted
physiology. Persist the result for its stage; do not rebase doses on each upload.
"""
from dataclasses import dataclass, replace
from datetime import timedelta
from math import ceil, floor
from statistics import median

from .session_decision import CoachingContract, DoseLimits
from .staged_progression import MILE, tenth
from .workload_accounting import Quantity


@dataclass(frozen=True)
class PrescriptionPolicy:
    history_days: int = 30
    duration_margin: float = .10
    reduction_fraction: float = .8
    minimum_minutes: float = 10.
    penalty_free_minutes: float = 25.
    source: str = 'provisional adaptive prescription policy v1; assumptions AS37-AS40'

    def __post_init__(self):
        if (self.history_days < 1 or not 0 < self.duration_margin < 1 or
                not 0 < self.reduction_fraction < 1 or
                not 0 < self.minimum_minutes < self.penalty_free_minutes or not self.source):
            raise ValueError('Explicit positive prescription policy required')


@dataclass(frozen=True)
class CandidatePrescription:
    key: str
    role: str
    miles: float
    phases: tuple[tuple[str,float], ...]
    phase_scenarios: tuple[tuple[float,float], ...]
    contract: CoachingContract
    intensity_key: str
    duration_source: str
    source_keys: tuple[str, ...]
    trial: bool = False


def _quantile(values, fraction):
    values = sorted(values)
    pos = (len(values)-1)*fraction
    return values[floor(pos)]+(values[ceil(pos)]-values[floor(pos)])*(pos-floor(pos))


def compile_prescriptions(history, stage, *, physical=(), policy=PrescriptionPolicy()):
    """SessionEvidence supplies duration, observed physical bouts supply structure.

    A tiny run cannot redefine ordinary size: the accepted stage owns that size.
    Quality purpose is a declared general aerobic-work objective, not inferred
    threshold/VO2max physiology from a session title or high HR.
    """
    recent = sorted((s for s in history if stage.issued_at-timedelta(days=policy.history_days) <= s.start
        and s.end <= stage.issued_at and s.available_at <= stage.issued_at and s.capacity_eligible),key=lambda s:(s.start,s.key))
    aerobic = [s for s in recent if s.role in {'easy','long'} and s.meters > 0 and s.moving_seconds > 0]
    if not aerobic:
        return dict(status='missing_duration_evidence',prescriptions=(),source_keys=(),policy=policy)
    eligible_keys = {s.key for s in recent}
    physical = [r for r in physical if r.key in eligible_keys and r.role=='quality' and r.bouts
                and r.work_evidence_basis!='unknown' and r.available_at<=stage.issued_at]
    pace = [s.moving_seconds/60/(s.meters/MILE) for s in aerobic]
    nominal_pace = median(pace)
    low_pace = min(nominal_pace*(1-policy.duration_margin),_quantile(pace,.1))
    high_pace = max(nominal_pace*(1+policy.duration_margin),_quantile(pace,.9))
    source = policy.source+';paired_aerobic:'+','.join(s.key for s in aerobic)
    recipes = []
    minimum_miles = ceil(policy.minimum_minutes/low_pace*10)/10
    # Short options are a menu, not a rewarded count. Sizes have a fixed stage origin.
    maximum_easy = min(stage.long_miles,tenth(stage.ordinary_miles*1.5),stage.weekly_equivalent_miles)
    if not minimum_miles <= stage.ordinary_miles <= maximum_easy:
        return dict(status='stage_duration_envelope_conflict',prescriptions=(),
                    source_keys=tuple(s.key for s in recent),policy=policy)
    distances = {minimum_miles,tenth(stage.ordinary_miles*policy.reduction_fraction),stage.ordinary_miles}
    # Every publishable tenth is a real command with its own duration/forecast.
    # Calendar exploration can use representatives; dose refinement cannot be
    # restricted to the old half-mile menu.
    distances.update(i/10 for i in range(ceil(minimum_miles*10),floor(maximum_easy*10)+1))
    for role, values in (('easy',sorted(d for d in distances if minimum_miles<=d<=maximum_easy)),
                         ('long',sorted({tenth(stage.long_miles*policy.reduction_fraction),stage.long_miles}))):
        for miles in values:
            if miles<=0:
                continue
            minutes = miles*nominal_pace
            key = f'{role}:{miles:.1f}'
            ordinary = Quantity.point(stage.long_miles,'stage long target') if role=='long' else Quantity(minimum_miles,maximum_easy,stage.ordinary_miles,'stage easy command envelope')
            reduced = DoseLimits(Quantity.point(tenth(stage.long_miles*policy.reduction_fraction),'declared reduced long')) if role=='long' else None
            contract = CoachingContract(stage.issued_at,source,role,'controlled_easy',DoseLimits(ordinary),reduced)
            recipes.append(CandidatePrescription(key,role,miles,(('warmup',minutes*60),),
                ((miles*low_pace*60,miles*high_pace*60),),contract,'controlled_easy',source,
                tuple(s.key for s in aerobic)))
    # Keep quality work separate from volume progression; do not compound both.
    quality_source = max(physical,key=lambda r:(r.start,r.key),default=None)
    if quality_source:
        work = [round(b.work_seconds/5)*5 for b in quality_source.bouts]
        rests = [b.recovery_after_seconds for b in quality_source.bouts[:-1]]
        work = [max(10.,v) for v in work]
        rests = [v if v is not None and v>0 else min(120.,work[i]) for i,v in enumerate(rests)]
        keys = (quality_source.key,)
    else:
        # Small controlled introduction, not all-out repetitions or full threshold work.
        work,rests,keys = [60.,60.,60.],[90.,90.],()
    quality_recipes = []
    for reduced in (False,True):
        chosen = work[:max(1,len(work)-1)] if reduced and len(work)>1 else (
            [max(5.,round(work[0]*policy.reduction_fraction/5)*5)] if reduced else work)
        warm,cool = 600.,600.
        phases = [('warmup',warm)]
        for i,seconds in enumerate(chosen):
            phases.append(('work',seconds))
            if i<len(chosen)-1:
                phases.append(('recovery',rests[i]))
        phases.append(('cooldown',cool))
        # Extent uses earlier whole quality pace when available, not the target run.
        qpace = [s.moving_seconds/60/(s.meters/MILE) for s in recent
                 if s.role in {'intervals','tempo_threshold','quality'} and s.meters>0 and s.moving_seconds>0]
        minutes_per_mile = median(qpace) if qpace else nominal_pace
        miles = tenth(sum(s for _,s in phases)/60/minutes_per_mile)
        seconds_total = sum(chosen)
        ratios = [rests[i]/chosen[i] for i in range(len(chosen)-1)] or [0.]
        limits = DoseLimits(Quantity.point(miles,'committed extent at prior quality pace'),
            Quantity.point(seconds_total,'declared work'),Quantity(min(chosen),max(chosen)),Quantity(min(ratios),max(ratios)))
        contract = CoachingContract(stage.issued_at,source+';quality_structure:'+(','.join(keys) or 'introductory_trial'),
            'high_aerobic','controlled_aerobic_work',limits,None)
        quality_recipes.append(CandidatePrescription('quality:reduced' if reduced else 'quality:ordinary','quality',miles,
            tuple(phases),tuple((s*(1-policy.duration_margin),s*(1+policy.duration_margin)) for _,s in phases),
            contract,'controlled_aerobic_work',source,keys,not bool(quality_source)))
    ordinary, reduced = quality_recipes
    # Both alternatives are judged against the SAME predeclared purpose contract.
    # A novel ordinary dose is explicitly a trial; the smaller option remains a
    # meaningful reduction, including when current context reports mild fatigue.
    shared = replace(ordinary.contract, reduced=reduced.contract.ordinary,
        trial=ordinary.contract.ordinary if ordinary.trial else None,
        trial_target=sum(work) if ordinary.trial else None)
    recipes.extend(replace(p,contract=shared) for p in quality_recipes)
    return dict(status='provisional_prescriptions',prescriptions=tuple(recipes),source_keys=tuple(s.key for s in recent),
        policy=policy,nominal_minutes_per_mile=nominal_pace,duration_pace_scenarios=(low_pace,high_pace),
        quality_work_minutes=sum(work)/60,minimum_easy_miles=minimum_miles,
        limitations=('No exact template required; general aerobic quality purpose is provisional',
            'Time-based quality extent is a prior-pace estimate, not a second independent distance requirement',
            'Warmup category stands for continuous easy running in this adapter',
            'Rates and suitability must be attached and evaluated before selection'))
