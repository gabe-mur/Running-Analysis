"""Read-only real-HR donor bank and explicitly synthetic workload transformations.

No learning, outcome-based donor choice, new interval detector, source rewrites,
or claim that stretching a recorded HR pattern predicts a runner's response.
The recovery prototype consumes integrated workloads, so phase HR integrals are
rescaled directly; this does not manufacture a new FIT/TCX file or beat stream.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from math import isfinite, log

from run_analysis.movement import classify_movement
from run_analysis.workout_detection import usable_recorded_laps, recorded_interval_work_positions, inferred_work_groups
from run_analysis.workload_accounting import Quantity, Workload, continuous_zone_seconds
from run_analysis.workload_profile import prescribed_profile
from run_analysis.prescription_load import projected_running_minutes
from run_analysis.recovery_assessment import SessionEntry
from run_analysis.training_load import ZONE_WEIGHTS
from scripts.audit_workload_contract import load_sessions
from scripts.audit_workload_expectation import app_session_types
from scripts.sandbox_recovery_adapter import session_key


def phase_rate(intervals, zones):
    seconds = continuous_zone_seconds(intervals, zones)
    total = sum(seconds.values())/60
    if total <= 0:
        return None
    known = sum(seconds.get(k, 0)*v/60 for k,v in ZONE_WEIGHTS.items())
    unknown = seconds.get('unknown', 0)/60
    return Quantity(known/total, (known+5*unknown)/total,
                    basis='original HR integral per moving minute; missing HR bounded'), seconds


@dataclass(frozen=True)
class HRDonor:
    key: str
    role: str
    recorded_start: datetime
    original_miles: float
    original_minutes: float
    total_rate: Quantity
    work_rate: Quantity | None
    other_rate: Quantity | None
    source_work_minutes: float | None
    source_work_bouts: int
    hr_coverage: float
    phase_evidence: str

    @property
    def pace(self):
        return self.original_minutes/self.original_miles


def donor_from_activity(activity, entry, profile, config):
    from dataclasses import asdict
    role = profile['intensity_structure']['session_workout_type']
    intervals = classify_movement(activity.trackpoints, config['moving_time']).intervals
    whole = phase_rate(intervals, config['zones'])
    if whole is None or entry.workload.distance_meters is None or entry.workload.distance_meters.low <= 0:
        return None
    groups, evidence = [], 'whole-session HR'
    # Preserve production eligibility; easy pace fluctuations never become reps.
    if role == 'intervals':
        laps = [asdict(lap) for lap in activity.laps]
        usable = usable_recorded_laps(laps)
        positions = sorted(recorded_interval_work_positions(laps))
        if len(positions) >= 2:
            for pos in positions:
                lap = usable[pos]
                if lap['start_time_utc'] is not None and lap['total_time_s']:
                    a = lap['start_time_utc']; b = a+timedelta(seconds=lap['total_time_s'])
                    groups.append([i for i,x in enumerate(intervals)
                        if a <= x.start.timestamp_utc+(x.end.timestamp_utc-x.start.timestamp_utc)/2 < b])
            evidence = 'recorded lap contrasts; boundary intervals assigned by midpoint'
        else:
            groups = [list(range(a,b)) for a,b in inferred_work_groups(intervals)]
            evidence = 'existing app-gated pace inference; boundaries not validated'
    indexes = {i for group in groups for i in group}
    work = [x for i,x in enumerate(intervals) if i in indexes]
    other = [x for i,x in enumerate(intervals) if i not in indexes]
    work_result, other_result = phase_rate(work, config['zones']), phase_rate(other, config['zones'])
    return HRDonor(entry.key, role, entry.start,
        entry.workload.distance_meters.low/1609.344, sum(x.moving_time_s for x in intervals)/60,
        whole[0], work_result[0] if work_result else None, other_result[0] if other_result else None,
        sum(x.moving_time_s for x in work)/60 if work else None, len(groups),
        profile['cardiovascular']['hr_coverage'], evidence)


def load_bank(database, config):
    types = app_session_types(database)
    bank = []
    def collect(activity, entry, profile):
        donor = donor_from_activity(activity, entry, profile, config)
        if donor is not None:
            bank.append(donor)
    load_sessions(database, config, datetime(2026,1,1,tzinfo=timezone.utc),
        datetime(2026,9,25,tzinfo=timezone.utc), session_types=types, activity_observer=collect)
    return tuple(bank)


def choose_donor(bank, rec, *, variant=0, exclude=()):
    kind = rec.workout_type.value
    role = 'long' if kind == 'long' else 'quality' if kind in {'intervals','tempo_threshold','race'} else 'easy'
    profile = prescribed_profile(rec)
    shape = profile['intensity_structure']
    pool = [d for d in bank if d.key not in exclude and d.hr_coverage >= .95 and
        (d.role == role if role != 'quality' else
         d.role == 'intervals' and d.work_rate is not None and d.other_rate is not None)]
    # Tempo also borrows measured high-effort and non-work HR phases; it is an
    # explicit constructed scenario, not a claim these sessions are identical.
    if not pool:
        raise ValueError(f'No HR donor for {role}; do not substitute missing HR')
    def mismatch(d):
        work = shape['total_work_seconds']/60
        source = d.source_work_minutes or 0
        return ((abs(log(max(work,.1)/max(source,.1))) if role == 'quality' else 0),
                abs(log(rec.planned_distance_miles/d.original_miles)), d.key)
    return sorted(pool,key=mismatch)[variant % len(pool)]


def transformed_entry(donor, rec, *, miles=None, at=None, source='observed'):
    """Resize observed phase integrals to an explicitly prescribed synthetic run.

    Actual HR amplitudes/rates are not reduced to meet a recovery threshold.
    Work duration follows numeric prescription; non-work time uses the donor's
    measured non-work HR (including delayed HR recovery). Phase sequencing and
    hypothetical adaptation are not physiological predictions.
    """
    miles = rec.planned_distance_miles if miles is None else miles
    at = rec.planned_for if at is None else at
    if not isfinite(miles) or miles <= 0:
        raise ValueError('Positive transformed distance required')
    minutes = projected_running_minutes(rec, miles, donor.pace)
    profile = prescribed_profile(rec)
    if 'work_duration_missing' in profile['limitations']:
        raise ValueError('This scenario adapter requires a numeric work prescription')
    bouts = profile['intensity_structure']['bouts']
    work = sum(b['work_seconds'] for b in bouts)/60
    if work and (donor.work_rate is None or donor.other_rate is None):
        raise ValueError('Numeric quality work needs a measured HR-phase donor')
    rates = ((work, donor.work_rate), (minutes-work, donor.other_rate)) if work else ((minutes, donor.total_rate),)
    low = sum(t*r.low for t,r in rates)
    high = sum(t*r.high for t,r in rates)
    structure = dict(profile['intensity_structure'], evidence='synthetic prescribed phases from recorded HR',
        donor_key=donor.key, donor_recorded_start=donor.recorded_start.isoformat(),
        donor_phase_evidence=donor.phase_evidence,
        distance_scale=miles/donor.original_miles, total_time_scale=minutes/donor.original_minutes,
        work_time_scale=work/donor.source_work_minutes if work and donor.source_work_minutes else None,
        other_time_scale=(minutes-work)/(donor.original_minutes-(donor.source_work_minutes or 0)),
        donor_total_hr_rate=donor.total_rate, donor_work_hr_rate=donor.work_rate,
        donor_other_hr_rate=donor.other_rate)
    workload = Workload(source, Quantity.point(miles*1609.344, 'synthetic transformed distance'),
        Quantity.point(minutes,'synthetic duration from recorded pace and prescribed work'),
        Quantity(low,high,basis='rescaled observed phase HR integrals; no invented HR'), structure,
        ('Transformed test fixture, not an original activity or physiological prediction',
         'No new independent athlete evidence from repeated use of a donor',))
    return SessionEntry(session_key(rec.planned_for), at, at+timedelta(minutes=minutes), workload,
                        Quantity.point(work,'synthetic prescribed work duration'))


def issued_envelope(donor, rec):
    """Pre-execution range from a committed donor and the prescribed distance band."""
    nominal = transformed_entry(donor,rec,source='projected')
    ends = [transformed_entry(donor,rec,miles=x,source='projected') for x in rec.distance_range_miles]
    from dataclasses import replace
    def bounded(name):
        a,b = (getattr(e.workload,name) for e in ends)
        n = getattr(nominal.workload,name)
        return Quantity(min(a.low,b.low),max(a.high,b.high), n.estimate,
                        'precommitted donor over issued distance band')
    return replace(nominal,workload=replace(nominal.workload,
        distance_meters=bounded('distance_meters'),running_minutes=bounded('running_minutes'),
        cardio_points=bounded('cardio_points')))
