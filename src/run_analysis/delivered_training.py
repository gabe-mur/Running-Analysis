"""Purpose-aware numeric delivery, offline; no training credit or cadence reset."""
from dataclasses import dataclass
from datetime import datetime
from math import isfinite
import sqlite3

from .athlete_projection import aware
from .workload_accounting import Quantity, observed_workload
from .workload_profile import WorkBout, bout_summary
from .quality_phases import detect_continuous_quality_phase

PURPOSES = {'easy_volume', 'long_endurance', 'threshold', 'high_aerobic', 'speed_skill', 'race_specific'}
QUALITY = PURPOSES-{'easy_volume', 'long_endurance'}
METRICS = {'distance_meters', 'moving_seconds', 'work_seconds', 'work_meters',
           'bout_count', 'between_recovery_seconds'}


@dataclass(frozen=True)
class DoseTarget:
    metric: str
    quantity: Quantity

    def __post_init__(self):
        if self.metric not in METRICS or self.quantity.low <= 0:
            raise ValueError('Named metric and positive issued target required')


@dataclass(frozen=True)
class DoseContract:
    issued_at: datetime
    purpose: str
    targets: tuple[DoseTarget, ...]
    source: str
    bout_seconds: Quantity | None = None
    bout_meters: Quantity | None = None

    def __post_init__(self):
        aware(self.issued_at)
        names = {t.metric for t in self.targets}
        if not self.source or self.purpose not in PURPOSES or not names or len(names) != len(self.targets):
            raise ValueError('Unique explicit dose targets, purpose and source required')
        if self.purpose in QUALITY:
            if not names & {'work_seconds', 'work_meters'} or not (self.bout_seconds or self.bout_meters):
                raise ValueError('Quality needs total work and per-bout dose, not a label')
        elif not names & {'distance_meters', 'moving_seconds'}:
            raise ValueError('Aerobic/endurance needs a session dose')
        if any(q is not None and q.low <= 0 for q in (self.bout_seconds, self.bout_meters)):
            raise ValueError('Positive per-bout target required')


@dataclass(frozen=True)
class PurposeEvidence:
    key: str
    purpose: str
    available_at: datetime
    source: str

    def __post_init__(self):
        aware(self.available_at)
        if not self.key or not self.source or self.purpose not in PURPOSES:
            raise ValueError('Independent purpose evidence with provenance required')


@dataclass(frozen=True)
class DeliveredDose:
    key: str
    start: datetime
    end: datetime
    available_at: datetime
    distance_meters: Quantity | None
    moving_seconds: Quantity | None
    bouts: tuple[WorkBout, ...]
    structure_basis: str
    role: str | None = None

    def __post_init__(self):
        for at in (self.start, self.end, self.available_at):
            aware(at)
        if not self.key or self.end <= self.start or self.available_at < self.end:
            raise ValueError('Single completed session identity and chronology required')
        if self.structure_basis not in {'recorded_lap_contrast', 'recorded_continuous_lap', 'pace_inferred', 'explicit_segments', 'unknown'}:
            raise ValueError('Known structure provenance required')
        if self.bouts and self.structure_basis == 'unknown':
            raise ValueError('Work observations require provenance')
        bout_summary(self.bouts)
        for bout in self.bouts:
            if bout.distance_meters is not None and (not isfinite(bout.distance_meters) or bout.distance_meters < 0):
                raise ValueError('Nonnegative finite work distance required')
        elapsed = (self.end-self.start).total_seconds()
        if self.moving_seconds is not None and self.moving_seconds.high > elapsed+1e-6:
            raise ValueError('Moving duration exceeds elapsed duration')
        if sum(b.work_seconds for b in self.bouts) > elapsed+1e-6:
            raise ValueError('Work exceeds session duration')
        known_block = sum(b.work_seconds for b in self.bouts)+sum(b.recovery_after_seconds or 0 for b in self.bouts[:-1])
        if known_block > elapsed+1e-6:
            raise ValueError('Work and recovery exceed session duration')
        if self.distance_meters and sum(b.distance_meters or 0 for b in self.bouts) > self.distance_meters.high+1e-6:
            raise ValueError('Work exceeds session distance')


def continuous_phase_from_activity(activity, config):
    """Reuse the app's continuous-phase rule on ORIGINAL laps, not cached fits.

    Existing selector needs SQL rows. Adapt in memory, without rewriting its
    8–40 minute / configured-HR heuristic or modifying the athlete's database.
    Its result is a candidate work segment, not proof of threshold physiology.
    """
    with sqlite3.connect(':memory:') as db:
        db.row_factory = sqlite3.Row
        db.execute('CREATE TABLE laps (activity_id,lap_index,total_time_s,distance_m,average_hr_bpm,maximum_hr_bpm,intensity,trigger_method,start_time_utc)')
        db.executemany('INSERT INTO laps VALUES (1,?,?,?,?,?,?,?,?)',
            [(lap.lap_index, lap.total_time_s, lap.distance_m, lap.average_hr_bpm, lap.maximum_hr_bpm,
              getattr(lap,'intensity',None),getattr(lap,'trigger_method',None),
              lap.start_time_utc.isoformat() if getattr(lap,'start_time_utc',None) else None)
             for lap in activity.laps])
        return detect_continuous_quality_phase(db, config, 1)


def dose_from_profile(profile, *, key, start, end, available_at, continuous_phase=None):
    """Reuse observed exposure and existing detector; no role/HR reclassification."""
    load = observed_workload(profile)
    shape = profile['intensity_structure']
    role = shape.get('session_workout_type')
    selected, basis = None, 'unknown'
    # Match the existing observed-feature gate; contrast is not absolute effort.
    if role in {None, 'intervals'} and shape['bout_count']:
        selected, basis = shape, 'recorded_lap_contrast'
    elif role == 'intervals':
        inferred = shape.get('pace_inferred_work')
        if inferred and inferred['usable_for_comparison'] and inferred['bout_count']:
            selected, basis = inferred, 'pace_inferred'
    bouts = tuple(WorkBout(b['work_seconds'], b['recovery_after_seconds'], b.get('distance_meters'))
                  for b in selected['bouts']) if selected else ()
    if continuous_phase is not None:
        if role != 'tempo_threshold':
            raise ValueError('Continuous phase requires the existing eligible role')
        if continuous_phase.start_offset_seconds < 0 or continuous_phase.end_offset_seconds > (end-start).total_seconds()+1e-6:
            raise ValueError('Continuous phase lies outside session')
        bouts = (WorkBout(continuous_phase.duration_seconds, None,
                         continuous_phase.distance_miles*1609.344 if continuous_phase.distance_miles is not None else None),)
        basis = 'recorded_continuous_lap'
    return DeliveredDose(str(key), start, end, available_at, load.distance_meters,
        load.running_minutes.scale(60) if load.running_minutes else None, bouts, basis, role)


def dose_metrics(dose):
    shape = bout_summary(dose.bouts)
    point = lambda x: Quantity.point(x, dose.structure_basis) if x is not None else None
    known = bool(dose.bouts)
    distances = [b.distance_meters for b in dose.bouts]
    return dict(distance_meters=dose.distance_meters, moving_seconds=dose.moving_seconds,
        work_seconds=point(shape['total_work_seconds']) if known else None,
        work_meters=point(sum(distances)) if known and all(d is not None for d in distances) else None,
        bout_count=point(len(dose.bouts)) if known else None,
        between_recovery_seconds=point(shape['between_work_recovery_seconds']) if known else None)


def compare_quantity(target, done):
    if done is None:
        return dict(status='unknown', actual=None, target=target, ratio_to_target_range=None)
    status = ('below' if done.high < target.low-1e-8 else
              'above' if done.low > target.high+1e-8 else
              'within' if done.low >= target.low-1e-8 and done.high <= target.high+1e-8 else
              'overlapping_uncertainty')
    return dict(status=status, actual=done, target=target,
                ratio_to_target_range=(done.low/target.high, done.high/target.low))


def assess_delivery(contract, dose, *, as_of, purpose_evidence=None):
    """Arithmetic fit is not equivalent stimulus, confirmed adherence or readiness."""
    aware(as_of)
    if contract.issued_at > dose.start:
        raise ValueError('Contract must be frozen before the run')
    if dose.end > as_of or dose.available_at > as_of:
        raise ValueError('Completion not yet available')
    if purpose_evidence is not None and purpose_evidence.key != dose.key:
        raise ValueError('Purpose evidence belongs to another session')
    metrics = dose_metrics(dose)
    rows = {t.metric: compare_quantity(t.quantity, metrics[t.metric]) for t in contract.targets}
    bouts = []
    for index, bout in enumerate(dose.bouts):
        row = dict(index=index)
        for name, target, actual in (
            ('seconds', contract.bout_seconds, bout.work_seconds),
            ('meters', contract.bout_meters, bout.distance_meters),
        ):
            if target is not None:
                row[name] = compare_quantity(target, Quantity.point(actual, dose.structure_basis) if actual is not None else None)
        bouts.append(row)
    statuses = [r['status'] for r in rows.values()]
    statuses += [r['status'] for b in bouts for k, r in b.items() if k != 'index']
    if (contract.bout_seconds or contract.bout_meters) and not bouts:
        statuses.append('unknown')
    fit = ('within_all_issued_bands' if all(s == 'within' for s in statuses) else
           'measured_difference' if any(s in {'below', 'above'} for s in statuses) else 'unresolved')
    known_purpose = purpose_evidence if purpose_evidence and purpose_evidence.available_at <= as_of else None
    purpose = ('unverified' if known_purpose is None else
               'consistent' if known_purpose.purpose == contract.purpose else 'different_purpose')
    return dict(key=dose.key, purpose=contract.purpose, contract_source=contract.source,
        issued_at=contract.issued_at, numerical_fit=fit, dimensions=rows,
        per_bout=bouts, observed_metrics=metrics, observed_bouts=dose.bouts,
        structure_basis=dose.structure_basis,
        structure_certainty='provisional_unquantified_boundaries' if dose.structure_basis == 'pace_inferred' else
            'unknown' if dose.structure_basis == 'unknown' else 'observed_structure_not_verified_intensity',
        purpose_evidence=purpose, purpose_source=known_purpose.source if known_purpose else None,
        physiological_equivalence_established=False, adaptation_authorized=False, planner_influence=False,
        limitations=['Numerical fit does not establish training adaptation, fatigue or recovery',
            'Detector output may not cover all work; missing bouts are not zero',
            'Recovery activity/intensity and uninterrupted endurance are not established',
            'No cadence reset, catch-up debt, or automatic schedule changes'])
