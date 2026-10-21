"""Descriptive recorded-versus-map terrain audit, never an automatic veto.

Ground maps are a comparator, not bridge-deck truth. Remove one constant
vertical offset before assessing profile shape; compare grade effects too,
because equal total ascent or absolute altitude need not imply equal grades.
"""
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone, timedelta
from pathlib import Path
from statistics import median

import numpy as np

from .elevation_quality import map_elevation, tile_directory, recorded_elevation_source, assess_recorded_elevation
from .model_windows import _overlapping_windows_for_intervals
from .movement import classify_movement, attach_elevation_deltas, smooth_elevation
from .objective_modeling import _primary_window_weight
from .pace_comparability import compare_run_windows, fit_comparison_calibration
from .processing import _load_points


def profile_comparison(distances, recorded, mapped, *, sample_m=20., maximum_gap_m=120.):
    """Uniform-distance sampling avoids weighting stationary GPS fixes as hills."""
    def samples(values, grid):
        by_distance=defaultdict(list)
        for distance,value in zip(distances,values):
            if value is not None and np.isfinite(value) and np.isfinite(distance):
                by_distance[float(distance)].append(float(value))
        if len(by_distance)<2:return np.full(len(grid),np.nan)
        x=np.array(sorted(by_distance));y=np.array([median(by_distance[d]) for d in x])
        result=np.interp(grid,x,y)
        left=np.clip(np.searchsorted(x,grid,side='right')-1,0,len(x)-1)
        right=np.clip(np.searchsorted(x,grid,side='left'),0,len(x)-1)
        result[(grid<x[0])|(grid>x[-1])|(x[right]-x[left]>maximum_gap_m)]=np.nan
        return result
    if not distances or max(distances)<=0:return dict(status='insufficient_overlap')
    grid=np.arange(0,max(distances)+1e-8,sample_m)
    raw,reference=samples(recorded,grid),samples(mapped,grid)
    keep=np.isfinite(raw)&np.isfinite(reference)
    if keep.sum()<3:return dict(status='insufficient_overlap',samples=int(keep.sum()))
    raw,reference=raw[keep],reference[keep]
    difference=raw-reference;offset=float(np.median(difference));residual=difference-offset
    return dict(status='compared',samples=len(raw),sample_spacing_m=sample_m,
        common_sample_fraction=float(keep.mean()),constant_offset_m=offset,
        median_absolute_residual_m=float(np.median(abs(residual))),
        p90_absolute_residual_m=float(np.percentile(abs(residual),90)),
        recorded_range_m=float(np.ptp(raw)),map_range_m=float(np.ptp(reference)),
        profile_correlation=float(np.corrcoef(raw,reference)[0,1]) if np.std(raw)>1e-6 and np.std(reference)>1e-6 else None)


def _summaries(rows):
    compared=[r for r in rows if r['profile']['status']=='compared']
    effects=[abs(r['recorded_terrain_seconds_per_mile']-r['map_terrain_seconds_per_mile'])
        for r in compared if r['recorded_terrain_seconds_per_mile'] is not None and r['map_terrain_seconds_per_mile'] is not None]
    errors=[r['profile']['p90_absolute_residual_m'] for r in compared]
    return dict(runs=len(rows),compared_runs=len(compared),
        median_p90_profile_disagreement_m=median(errors) if errors else None,
        profile_p90_within_2m=sum(v<=2 for v in errors),profile_p90_within_5m=sum(v<=5 for v in errors),
        terrain_effect_comparisons=len(effects),
        median_absolute_terrain_effect_difference_seconds=median(effects) if effects else None,
        terrain_effect_within_5_seconds=sum(v<=5 for v in effects),
        contradictory_revisits=sum(r['recorded_quality']['status']=='rejected' for r in compared))


def audit_elevation_profiles(connection,config,*,as_of=None):
    at=as_of or datetime.now(timezone.utc)
    if at.tzinfo is None:raise ValueError('Timezone-aware audit time required')
    directory=tile_directory(connection,config)
    activities=connection.execute('''SELECT a.id,a.activity_id,a.creator,a.start_time_utc,
        (SELECT s.path FROM activity_sources src JOIN source_files s ON s.id=src.source_file_id
         WHERE src.activity_id=a.id AND src.is_primary=1 ORDER BY s.id DESC LIMIT 1) AS primary_path,
        m.previous_7d_miles,m.previous_28d_miles,m.days_since_previous_run,m.days_since_previous_hard_run,
        m.moving_pace_min_mile AS run_moving_pace,m.moving_average_hr_bpm,
        COALESCE(o.health_tag,'normal') AS health_tag,
        COALESCE(o.workout_type,m.detected_workout_type,'unknown') AS workout_type
        FROM activities a JOIN activity_metrics m ON m.activity_id=a.id
        LEFT JOIN run_overrides o ON o.activity_id=a.activity_id
        WHERE m.model_eligible=1 ORDER BY a.start_time_utc_epoch''').fetchall()
    calibration=fit_comparison_calibration([],config)
    results=[]
    for activity in activities:
        start=datetime.fromisoformat(activity['start_time_utc'])
        if start>at:continue
        points=_load_points(connection,activity['id'])
        movement=classify_movement(points,config['moving_time'])
        indexed={i.index:i.distance_m for i in movement.intervals}
        distances=[0.]
        for i in range(len(points)-1):distances.append(distances[-1]+indexed.get(i,0.))
        mapped=[map_elevation(p.latitude,p.longitude,directory) if p.gps_valid else None for p in points]
        profiles=[];effects=[];gains=[]
        for selected in (points,[replace(p,altitude_m=value) for p,value in zip(points,mapped)]):
            profiles.append(smooth_elevation(selected,movement.intervals,float(config['elevation']['smoothing_window_meters'])))
            for interval in movement.intervals:interval.elevation_delta_m=None
            attach_elevation_deltas(selected,movement.intervals,float(config['elevation']['smoothing_window_meters']))
            for interval in movement.intervals:
                if selected[interval.index].altitude_m is None or selected[interval.index+1].altitude_m is None:
                    interval.elevation_delta_m=None
            gains.append(sum(max(0.,i.elevation_delta_m or 0.) for i in movement.intervals))
            windows,_=_overlapping_windows_for_intervals(movement.intervals,activity,{},
                int(config['model'].get('primary_window_seconds',120)),
                int(config['model'].get('primary_window_stride_seconds',60)),config)
            if windows:
                estimate=compare_run_windows(windows,[_primary_window_weight(w,config) for w in windows],config,calibration)
                effects.append(estimate['contributions_seconds_per_mile']['terrain'])
            else:effects.append(None)
        # A missing DEM sample stays missing despite the smoothing helper's
        # neighbouring-point fill, so the audit cannot invent reference coverage.
        profiles[1]=[value if original is not None else None for value,original in zip(profiles[1],mapped)]
        results.append(dict(activity_id=activity['id'],start_time=activity['start_time_utc'],creator=activity['creator'],
            source_format=Path(activity['primary_path']).suffix.lower().lstrip('.') if activity['primary_path'] else 'unknown',
            source=recorded_elevation_source(activity['creator']),
            profile=profile_comparison(distances,*profiles),
            recorded_quality=assess_recorded_elevation(points,activity['creator']),
            recorded_smoothed_ascent_m=gains[0],map_smoothed_ascent_m=gains[1],
            recorded_terrain_seconds_per_mile=effects[0],map_terrain_seconds_per_mile=effects[1]))
    groups={source:_summaries([r for r in results if r['source']==source]) for source in sorted({r['source'] for r in results})}
    recent=[r for r in results if datetime.fromisoformat(r['start_time'])>=at-timedelta(days=90)]
    formats=sorted({(r['source'],r['source_format']) for r in results})
    return dict(as_of=at.isoformat(),runs=results,by_source=groups,
        by_source_and_format={f'{source}/{kind}':_summaries([r for r in results if r['source']==source and r['source_format']==kind]) for source,kind in formats},
        recent_90_days_by_source={source:_summaries([r for r in recent if r['source']==source]) for source in groups},
        scope='Recorded-versus-bare-earth-map disagreement, not a claim that every map or watch reading is true.',
        limitations=['No bridge-deck ground truth or route-level road/bridge labels.',
            'Unknown exports may already have elevation correction; they are not labelled watch GPS.',
            'Two/five metre and five-second bins are descriptive audit summaries, not automatic selection rules.',
            'Profile shape removes a constant vertical offset, but not time-varying drift.'])
