"""Read-only original-signal and reference-support audit, not run rejection.

Audit every included 2026 run; rank both signed leave-one-out trend residuals.
July 7 is explicit, not selected because a desired result improves. Raw GPS,
HR and distance summaries do not establish sensor accuracy or fitness truth.
"""
from argparse import ArgumentParser
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sqlite3
from urllib.request import urlopen

import numpy as np

from run_analysis import model_windows as mw, pace_comparability as pc
from run_analysis.aerobic_candidate import measurement_inputs
from run_analysis.config import load_config, resolve_project_path
from run_analysis.movement import classify_movement
from run_analysis.objective_modeling import _primary_window_weight, _pre_quality_indexes, _reference_time_support
from run_analysis.privacy import private_file
from run_analysis.processing import _load_points
from run_analysis.quality_phases import detect_continuous_quality_phase
from run_analysis.read_cache import invariant_snapshot
from run_analysis.tcx import parse_tcx
from scripts.aerobic_local_trend import LocalPaceTrend


def weighted(values,weights):
    valid=[(v,w) for v,w in zip(values,weights) if v is not None and math.isfinite(v) and w>0]
    return float(sum(v*w for v,w in valid)/sum(w for _,w in valid)) if valid else None


def intervals_summary(intervals):
    seconds=sum(i.moving_time_s for i in intervals);distance=sum(i.distance_m for i in intervals)
    hrs=[np.mean([v for v in (i.start.heart_rate_bpm,i.end.heart_rate_bpm) if v is not None])
        if i.start.heart_rate_bpm is not None or i.end.heart_rate_bpm is not None else None for i in intervals]
    pairs=[i for i in intervals if i.gps_distance_m is not None and i.device_distance_m is not None and i.device_distance_m>=0]
    device=sum(i.device_distance_m for i in pairs);gps=sum(i.gps_distance_m for i in pairs)
    return dict(moving_seconds=seconds,distance_m=distance,
        pace_min_mile=seconds/60/(distance/1609.344) if distance>0 else None,
        hr_bpm=weighted(hrs,[i.moving_time_s for i in intervals]),
        gps_device_distance_ratio=gps/device if device>0 else None,
        maximum_recording_gap_seconds=max((i.elapsed_s for i in intervals),default=None))


def raw_audit(connection,identity,config):
    activity=dict(connection.execute('SELECT id,start_time_utc,total_elapsed_time_s,total_distance_m,average_hr_bpm,maximum_hr_bpm,creator,trackpoint_count,distance_source FROM activities WHERE id=?',(identity,)).fetchone())
    points=_load_points(connection,identity);movement=classify_movement(points,config['moving_time'])
    time0=datetime.fromisoformat(activity['start_time_utc'])
    timeline=[];moving=0.
    for interval in movement.intervals:
        timeline.append((moving,interval));moving+=interval.moving_time_s
    max_stable_hr=0.;stable=0.;last_hr=None
    for interval in movement.intervals:
        if interval.start.heart_rate_bpm==interval.end.heart_rate_bpm and interval.start.heart_rate_bpm is not None:
            stable=(stable if last_hr==interval.start.heart_rate_bpm else 0.)+interval.elapsed_s
            max_stable_hr=max(max_stable_hr,stable)
        else:stable=0.
        last_hr=interval.end.heart_rate_bpm
    bins=[]
    for minute in range(0,math.ceil(moving/300)*5,5):
        selected=[i for position,i in timeline if minute*60<=position<(minute+5)*60]
        if selected:bins.append(dict(minutes=[minute,minute+5],**intervals_summary(selected)))
    gps_points=[p for p in points if p.gps_valid and p.timestamp_utc is not None]
    hr_points=[p for p in points if p.heart_rate_bpm is not None and p.timestamp_utc is not None]
    half=(min(300.,moving)+moving)/2
    postwarmup_halves=[intervals_summary([i for position,i in timeline if lower<=position<upper])
        for lower,upper in ((min(300.,moving),half),(half,moving+1))]
    jumps=[abs(i.end.heart_rate_bpm-i.start.heart_rate_bpm) for i in movement.intervals
        if i.start.heart_rate_bpm is not None and i.end.heart_rate_bpm is not None]
    short_jumps=[abs(i.end.heart_rate_bpm-i.start.heart_rate_bpm) for i in movement.intervals
        if i.elapsed_s<=10 and i.start.heart_rate_bpm is not None and i.end.heart_rate_bpm is not None]
    events=[dict(moving_minute=position/60,elapsed_seconds=i.elapsed_s,moving_seconds=i.moving_time_s,
        stopped_seconds=i.stopped_time_s,start_hr=i.start.heart_rate_bpm,end_hr=i.end.heart_rate_bpm,
        flags=i.flags) for position,i in timeline if i.elapsed_s>30 or
        (i.start.heart_rate_bpm is not None and i.end.heart_rate_bpm is not None
            and abs(i.end.heart_rate_bpm-i.start.heart_rate_bpm)>25)]
    metadata=dict(connection.execute('SELECT moving_pace_min_mile,moving_average_hr_bpm,model_eligible,exclusion_reason,standardized_pace_at_target_hr_min_mile,selected_model_name FROM activity_metrics WHERE activity_id=?',(identity,)).fetchone())
    return dict(activity=activity,metrics=metadata,movement=movement.diagnostics,
        whole_run=intervals_summary(movement.intervals),
        after_five_minutes=intervals_summary([i for position,i in timeline if position>=300]),
        postwarmup_halves=postwarmup_halves,
        near_twenty_minutes=intervals_summary([i for position,i in timeline if 17*60<=position<23*60]),
        gps_first_seconds=(gps_points[0].timestamp_utc-time0).total_seconds() if gps_points else None,
        hr_first_seconds=(hr_points[0].timestamp_utc-time0).total_seconds() if hr_points else None,
        point_hr_coverage=len(hr_points)/max(1,len(points)),gps_point_coverage=len(gps_points)/max(1,len(points)),
        hr_range=[min(p.heart_rate_bpm for p in hr_points),max(p.heart_rate_bpm for p in hr_points)] if hr_points else None,
        largest_hr_sample_step=max(jumps,default=None),largest_hr_step_within_ten_seconds=max(short_jumps,default=None),
        longest_identical_hr_seconds=max_stable_hr,recording_events=events,
        five_minute_bins=bins,
        parsed_flags=dict(Counter(flag for p in points for flag in p.parse_flags)))


def window_audit(windows,canonical,config,calibration):
    weights=[_primary_window_weight(w,config) for w in windows]
    base=pc.compare_run_windows(windows,weights,config,calibration)
    if not math.isclose(base['pace_min_mile'],canonical['pace_min_mile'],abs_tol=1e-9):
        raise ValueError('Canonical and locally reconstructed comparable run disagree')
    times=[w['moving_minutes_into_run'] for w in windows]
    support=_reference_time_support(times,config['reference_conditions']['within_run_minutes'],config)
    no_time=deepcopy(config);no_time['model']['primary_time_weight_scale_minutes']=1e12
    no_drift=deepcopy(calibration);no_drift['parameters'][1]=0.
    def aggregate(selected,weights,cal=calibration):
        if not selected:return None
        value=pc.compare_run_windows(selected,weights,config,cal)
        return dict(pace_min_mile=value['pace_min_mile'],observed_pace_min_mile=value['observed_pace_min_mile'],
            sampled_hr_bpm=value['sampled_hr_bpm'])
    near=[w for w in windows if 17<=w['moving_minutes_into_run']<=23]
    midpoint=(min(times)+max(times))/2
    first=[w for w in windows if w['moving_minutes_into_run']<=midpoint]
    last=[w for w in windows if w['moving_minutes_into_run']>midpoint]
    return dict(usable_windows=len(windows),center_minutes_range=[min(times),max(times)],
        sampled_mean_minutes=weighted(times,weights),
        reference_support=dict(supported=support[0],uncertainty_multiplier=support[1] if math.isfinite(support[1]) else None,
            gap_minutes=support[2],basis=support[3]),
        weather=dict(temperature_f=weighted([w.get('temperature_f') for w in windows],weights),
            dewpoint_f=weighted([w.get('dewpoint_f') for w in windows],weights)),
        scenarios=dict(current=aggregate(windows,weights),uniform_windows=aggregate(windows,[1.]*len(windows)),
            no_time_preference=aggregate(windows,[_primary_window_weight(w,no_time) for w in windows]),
            no_duration_adjustment=aggregate(windows,weights,no_drift),
            near_twenty=aggregate(near,[_primary_window_weight(w,config) for w in near]),
            earlier_half=aggregate(first,[_primary_window_weight(w,config) for w in first]),
            later_half=aggregate(last,[_primary_window_weight(w,config) for w in last])),
        window_profile=[dict(minutes=w['moving_minutes_into_run'],pace=w['moving_pace_min_mile'],hr=w['average_hr_bpm'],
            relative_weight=weight/sum(weights),hr_change=w['heart_rate_change_bpm'],speed_change=w['speed_change_fraction'])
            for w,weight in zip(windows,weights)])


def source_check(connection,identity,config):
    checks=[]
    points=_load_points(connection,identity)
    for row in connection.execute('SELECT f.path FROM source_files f JOIN activity_sources s ON s.source_file_id=f.id WHERE s.activity_id=?',(identity,)):
        path=Path(row['path'])
        if not path.is_absolute():path=resolve_project_path(Path.cwd(),str(path))
        item=dict(format=path.suffix.lower(),exists=path.exists())
        if item['exists'] and path.suffix.lower()=='.tcx':
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            parsed=parse_tcx(path,config['timezone_default'])
            activity=next((a for a in parsed.activities if a.start_time_utc==points[0].timestamp_utc),None)
            if activity:
                original=activity.trackpoints
                # Coordinate equality is tested but never emitted in the report.
                fields=('timestamp_utc','distance_m','heart_rate_bpm','latitude','longitude','speed_mps')
                item.update(trackpoint_count=len(original),imported_count=len(points),
                    source_matches_imported=len(original)==len(points) and all(
                        all(getattr(a,key)==getattr(b,key) for key in fields) for a,b in zip(original,points)),
                    source_unchanged=hashlib.sha256(path.read_bytes()).hexdigest()==digest)
        checks.append(item)
    return checks


def audit(payload,config,database):
    report=payload['pace_comparison'];calibration=report['calibration']
    rows=sorted([r for r in report['runs'] if r['included_in_trend'] and r['trend_weight']>0 and r['start_time']>='2026-03-01'],
        key=lambda r:r['start_time'])
    cov=np.zeros((4,4));cov[:3,:3]=calibration['covariance'];cov[3,3]=calibration['heat']['sd']**2
    m,g=measurement_inputs(rows,cov);end=datetime.fromisoformat(rows[-1]['start_time'])
    times=np.array([(datetime.fromisoformat(r['start_time'])-end).total_seconds()/86400 for r in rows]);y=np.log([r['pace_min_mile'] for r in rows])
    residuals={}
    for i,row in enumerate(rows):
        mask=np.arange(len(rows))!=i
        point=LocalPaceTrend(y[mask],m[mask],g[mask],cov,times[mask]).level(times[i])
        residuals[row['activity_id']]=60*(row['pace_min_mile']-point['pace_min_mile']) if point['available'] else None
    valid=sorted([r for r in rows if residuals[r['activity_id']] is not None],key=lambda r:residuals[r['activity_id']])
    early=sorted([r for r in rows if r['start_time']<'2026-05-01'],key=lambda r:r['pace_min_mile'])[:3]
    detailed={34}|{r['activity_id'] for r in [*valid[:3],*valid[-3:],*early]}
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True) as connection:
        connection.row_factory=sqlite3.Row
        with invariant_snapshot(connection):
            raw,_=mw.load_overlapping_model_windows(connection,config,
                window_seconds=int(config['model']['primary_window_seconds']),
                stride_seconds=int(config['model']['primary_window_stride_seconds']),check_elevation=True)
            grouped=defaultdict(list)
            for i,w in enumerate(raw):grouped[w['activity_id']].append(i)
            results=[]
            for row in rows:
                identity=row['activity_id'];indexes=grouped[identity]
                phase=detect_continuous_quality_phase(connection,config,identity) if raw[indexes[0]].get('workout_type')=='tempo_threshold' else None
                windows=[raw[i] for i in _pre_quality_indexes(raw,indexes,phase)]
                detail=window_audit(windows,row,config,calibration)
                sensors=raw_audit(connection,identity,config)
                result=dict(activity_id=identity,date=row['start_time'],loo_trend_residual_seconds_per_mile=residuals[identity],
                    comparable=dict(pace=row['pace_min_mile'],observed_pace=row['observed_pace_min_mile'],sampled_hr=row['sampled_hr_bpm'],
                        contributions_seconds_per_mile=row['contributions_seconds_per_mile'],terrain=row['terrain_quality']),
                    sensors=sensors,windows=detail)
                if identity in detailed:result['source_check']=source_check(connection,identity,config)
                results.append(result)
            assert connection.total_changes==0
    return dict(version='comparable-run-outlier-audit-v1',live_changed=False,source_database_read_only=True,
        cohort=len(rows),runs=results,detailed_ids=sorted(detailed),
        residual_rank=[dict(activity_id=r['activity_id'],date=r['start_time'],seconds_per_mile=residuals[r['activity_id']]) for r in valid],
        reference_support_counts=dict(Counter(r['windows']['reference_support']['basis'] for r in results)),
        assumptions=['Signed residual ranking uses leave-one-run-out retrospective local fits, not known physiological truth.',
            'All included 2026 runs from March onward are audited, not just favorable outliers; no run is deleted or downweighted.',
            'Raw interval sensor screens are descriptive, not proof of chest-strap or GPS accuracy.',
            'Uniform/time-preference/half-run/near-20-minute alternatives are diagnostic sensitivity, not replacements or selected best scores.',
            '20-minute reference-support classification reuses the original scorer helper; its multiplier is not applied to the current comparison estimate.',
            'Terrain is checked from existing local cache only; no new external elevation or weather requests.',
            'Current normalization and metadata are fixed; cached older scores may use a different model or reference and are not treated as ground truth.'])


def main():
    p=ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():p.error('New output path required')
    with urlopen('http://127.0.0.1:8000/api/progress-wip?window_days=365&include_exploratory=false',timeout=120) as response:
        payload=json.load(response)
    config=load_config('config.yaml');database=resolve_project_path(Path.cwd(),config['paths']['database'])
    result=audit(payload,config,database)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');private_file(args.output)
    print(json.dumps(dict(cohort=result['cohort'],reference_support=result['reference_support_counts'],
        detailed_ids=result['detailed_ids'],strongest=result['residual_rank'][:4],weakest=result['residual_rank'][-4:]),indent=2))


if __name__=='__main__':main()
