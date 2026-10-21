"""Explicit synthetic adherence files built from prior original HR/pace traces.

Not a forecast of physiological response. No future donor or invented HR stream;
phase time/distance transformations are recorded, while source files stay intact.
"""
from bisect import bisect_right
from datetime import timedelta
from hashlib import sha256
from math import isfinite
from pathlib import Path
import sqlite3

from run_analysis.importer import parser_for
from run_analysis.movement import classify_movement
from run_analysis.training_signals import HeartRateReference,observed_cardio_trace
from run_analysis.work_stimulus_evidence import observed_work_signals
from run_analysis.web.upload_service import UploadPayload
from run_analysis.phase_partitions import phase_layout,detailed_layout


def recorded_phase_bounds(phases,first,last):
    """Trim <=1s timer/trackpoint rounding only; never extrapolate HR/distance."""
    aligned=[]; trimmed=0.
    for kind,a,b in phases:
        left=max(a,first); right=min(b,last)
        trim=max(0.,left-a)+max(0.,b-right)
        if trim>1.+1e-8 or left>=right:
            raise ValueError('Donor phase must be covered by recorded source timestamps')
        trimmed+=trim
        aligned.append((kind,left,right))
    return tuple(aligned),trimmed


def original_activity_id(db,run):
    """Resolve a recorded seed alias, never a transformed future upload.

    Physical start/distance/duration must uniquely identify the original. No
    nearest-time matching, fallback HR, or rewriting the planner's physical key.
    """
    if run.key.isdecimal():return int(run.key)
    rows=db.execute('''SELECT id,start_time_utc,total_distance_m,total_elapsed_time_s
        FROM activities WHERE start_time_utc_epoch=?''',(run.start.timestamp(),)).fetchall()
    from datetime import datetime
    exact=[r[0] for r in rows if datetime.fromisoformat(r[1])==run.start
        and abs(r[2]-run.miles*1609.344)<=1e-6
        and abs(r[3]-(run.end-run.start).total_seconds())<=1e-6]
    if len(exact)!=1:raise ValueError('Recorded seed alias needs one exact original activity identity')
    return exact[0]


def donor(database,run,config,reference):
    with sqlite3.connect(Path(database).resolve().as_uri()+'?mode=ro',uri=True) as db:
        activity_id=original_activity_id(db,run)
        path=Path(db.execute('''SELECT s.path FROM source_files s JOIN activity_sources l ON l.source_file_id=s.id
            WHERE l.activity_id=? AND l.is_primary=1''',(activity_id,)).fetchone()[0])
    parsed=parser_for(path)(path,default_timezone=config['timezone_default'])
    activity=next(a for a in parsed.activities if a.start_time_utc==run.start)
    intervals=classify_movement(activity.trackpoints,config['moving_time']).intervals
    trace=observed_cardio_trace(activity,intervals,HeartRateReference(config['resting_hr'],config['max_hr'],'frozen donor settings'),
        reference.policy,include_spans=True,compute_density=False)
    signals=observed_work_signals(run,activity,config,intervals,trace['normalized_spans']) if run.bouts else None
    phases=detailed_layout(phase_layout(signals.windows,trace['elapsed_seconds'])) if signals else (('warmup',0.,trace['elapsed_seconds']),)
    if run.role=='quality' and not signals:
        raise ValueError('Quality donor requires supported source phase alignment')
    raw_points=[((p.timestamp_utc-run.start).total_seconds(),p.distance_m,p.heart_rate_bpm)
                  for p in activity.trackpoints if p.timestamp_utc is not None and p.distance_m is not None]
    # TCX lap boundaries can repeat the exact final point as the next first
    # point. Merge only identical records, not conflicting HR or distance resets.
    points=sorted(dict.fromkeys(raw_points),key=lambda p:p[:2])
    if not points or not any(p[2] is not None for p in points):
        raise ValueError('Original timed distance and HR required')
    phases,trimmed=recorded_phase_bounds(phases,points[0][0],points[-1][0])
    return dict(key=run.key,source_activity_id=activity_id,role=run.role,phases=phases,points=points,
        source_boundary_trim_seconds=trimmed,
        identical_boundary_points_removed=len(raw_points)-len(points),source_sha256=sha256(path.read_bytes()).hexdigest())


def transformed_payload(donor,dose,start,*,distance_factor=1.,duration_factor=1.):
    if start.tzinfo is None or any(not isfinite(v) or v<=0 for v in (distance_factor,duration_factor)):
        raise ValueError('Aware execution time and finite positive transformation factors required')
    source=donor['points']
    if len(source)<2 or any(not all(isfinite(v) for v in p[:2]) for p in source) or any(
            b[0]<=a[0] or b[1]<a[1] for a,b in zip(source,source[1:])):
        raise ValueError('Strictly ordered timed monotone source distance required')
    times=[p[0] for p in source]
    def sample(t):
        i=max(0,min(len(source)-2,bisect_right(times,t)-1))
        a,b=source[i],source[i+1]
        fraction=max(0.,min(1.,(t-a[0])/max(1e-9,b[0]-a[0])))
        distance=a[1]+fraction*(b[1]-a[1])
        hr=round(a[2]+fraction*(b[2]-a[2])) if a[2] is not None and b[2] is not None else None
        return distance,hr
    pools={k:[(a,b) for kind,a,b in donor['phases'] if kind==k] for k,_ in dose.phases}
    used={k:0 for k in pools}
    phases=tuple((k,s*duration_factor) for k,s in dose.phases)
    segments=[]
    elapsed=distance=0.
    for kind,seconds in phases:
        pool=pools[kind]
        if not pool:
            raise ValueError(f'No original {kind} phase; do not substitute unknown work')
        a,b=pool[min(used[kind],len(pool)-1)]
        if not times[0]<=a<b<=times[-1]:
            raise ValueError('Donor phase must be covered by recorded source timestamps')
        used[kind]+=1
        initial,_=sample(a)
        endpoint,_=sample(b)
        if b<=a or endpoint<=initial:
            raise ValueError('Donor phase has no positive timed distance')
        # Preserve within-phase pace shape and mean pace before one final
        # distance scaling; simply scaling phase length cannot speed up warmup.
        samples=sorted({a,b,*(t for t in times if a<t<b)})
        points=[]
        for t in samples:
            meters,hr=sample(t)
            points.append((elapsed+(t-a)/(b-a)*seconds,(meters-initial)*seconds/(b-a),hr))
        elapsed+=seconds
        extent=(endpoint-initial)*seconds/(b-a)
        distance+=extent
        segments.append((points,extent))
    total=dose.miles*distance_factor*1609.344
    numeric=(dose.workout_definition or {}).get('numeric_contract')
    committed=[None]*len(phases)
    if numeric:
        commands=numeric['phases']
        if len(commands)!=len(phases) or any(p['kind']!=kind for p,(kind,_) in zip(commands,phases)):
            raise ValueError('Numeric primary contract must match every execution phase')
        for i,p in enumerate(commands):
            primary=p['primary']; unit=primary['unit']; amount=primary['amount']['estimate']
            if unit=='distance':committed[i]=amount*distance_factor
            elif unit=='portion':committed[i]=amount*total
            elif unit!='time':raise ValueError('Unsupported execution primary unit')
    # Commit authored distance/portion phases separately. Time-primary phases
    # retain source-relative pace, sharing only the remaining total distance.
    # Forecast meters for a TIME command are not an observed distance target.
    residual=total-sum(v for v in committed if v is not None)
    flexible=sum(extent for (_,extent),target in zip(segments,committed) if target is None)
    if residual<0 or (flexible==0 and abs(residual)>1e-6) or (flexible>0 and residual<=0):
        raise ValueError('Primary distances cannot fit the complete execution distance')
    points=[];distance=0.
    for (part,extent),target in zip(segments,committed):
        assigned=target if target is not None else extent*residual/flexible
        points.extend((t,distance+m*assigned/extent,hr) for t,m,hr in part)
        distance+=assigned
    # Shared boundary points merge, retaining the single physical timestamp.
    unique={round(t,6):(m,hr) for t,m,hr in points}
    track=''.join(f'<Trackpoint><Time>{(start+timedelta(seconds=t)).isoformat()}</Time>'
        f'<DistanceMeters>{m}</DistanceMeters>'+(f'<HeartRateBpm><Value>{hr}</Value></HeartRateBpm>' if hr is not None else '')+
        '</Trackpoint>' for t,(m,hr) in sorted(unique.items()))
    content=(f'<TrainingCenterDatabase><Activities><Activity Sport="Running"><Id>{start.isoformat()}</Id>'
        f'<Lap StartTime="{start.isoformat()}"><TotalTimeSeconds>{elapsed}</TotalTimeSeconds>'
        f'<DistanceMeters>{dose.miles*distance_factor*1609.344}</DistanceMeters><Track>{track}</Track></Lap>'
        '</Activity></Activities></TrainingCenterDatabase>').encode()
    return UploadPayload('transformed-prior-trace.tcx',content),phases,start+timedelta(seconds=elapsed),dict(
        source_key=donor['key'],source_sha256=donor['source_sha256'],distance_factor=distance_factor,
        duration_factor=duration_factor,source_role=donor['role'],synthetic_not_physiological_prediction=True,
        execution_contract='primary_units_v2' if numeric else 'legacy_whole_distance_phase_time_v1',
        committed_phase_meters=committed,
        source_boundary_trim_seconds=donor.get('source_boundary_trim_seconds',0.),
        **({'source_activity_id':donor['source_activity_id']} if 'source_activity_id' in donor else {}),
        identical_boundary_points_removed=donor.get('identical_boundary_points_removed',0))


def compatible_donor(database,runs,dose,config,reference,cache):
    """Choose by ORIGINAL supported phase kinds before nearest distance.

    A progression donor is not an interval donor merely because both have the
    quality role. Missing evidence is reported, never manufactured.
    """
    required={kind for kind,_ in dose.phases}
    rejected=[]
    for run in sorted(runs,key=lambda r:(abs(r.miles-dose.miles),-r.start.timestamp(),r.key)):
        if run.key not in cache:
            cache[run.key]=donor(database,run,config,reference)
        original=cache[run.key]
        available={kind for kind,_,_ in original['phases']}
        if required<=available:
            return run,original
        rejected.append(dict(key=run.key,missing=sorted(required-available)))
    raise ValueError(f'No original donor covers the prescribed phases: {rejected}')
