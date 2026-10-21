"""Rejected offline surrounding-context exclusion experiment, not live scoring.

It discarded too much usable evidence and must not feed the Progress reader.
Retained for reproducing the failed audit, not as a recommended fallback.
"""
from collections import defaultdict
from datetime import datetime
import math

import numpy as np

from .physiology import grade_energy_ratio

VERSION='settled-effort-context-v1'
BIN_SECONDS=30.
BEFORE_SECONDS=120.
AFTER_SECONDS=60.


def independent_sections(rows):
    """Count disjoint evidence, not overlapping sliding windows."""
    count=0;end=-math.inf
    for row in sorted(rows,key=lambda r:r['window_end_time_utc']):
        low=datetime.fromisoformat(row['window_start_time_utc']).timestamp()
        high=datetime.fromisoformat(row['window_end_time_utc']).timestamp()
        if low>=end-1e-6:
            count+=1;end=high
    return count


def cached_settled_comparability(connection,report,config):
    """Offline experiment only; no production reader calls this function."""
    import json
    from .read_cache import invariant_read
    inputs={key:config.get(key) for key in ('model','moving_time','elevation','target_hr','max_hr','resting_hr','zones','reference_conditions')}
    return invariant_read(connection,(VERSION,json.dumps(report,sort_keys=True),json.dumps(inputs,sort_keys=True,default=str)),
        lambda:build_settled_comparability(connection,report,config))


def build_settled_comparability(connection,report,config):
    from collections import Counter
    from .model_windows import load_overlapping_model_windows
    from .objective_modeling import _primary_window_weight
    from .pace_comparability import compare_run_windows
    from .quality_phases import detect_continuous_quality_phase
    from .objective_modeling import _pre_quality_indexes
    windows,_=load_overlapping_model_windows(connection,config,window_seconds=120,stride_seconds=60,
        check_elevation=True,settled_context=True)
    groups=defaultdict(list)
    for w in windows:groups[w['activity_id']].append(w)
    results=[];excluded=[];audit=[]
    for ordinal,original in enumerate(report['runs']):
        if not original['included_in_trend'] or original['trend_weight']<=0:
            continue
        identity=original['activity_id'];rows=groups[identity]
        if rows and rows[0].get('workout_type')=='tempo_threshold':
            phase=detect_continuous_quality_phase(connection,config,identity)
            rows=[rows[i] for i in _pre_quality_indexes(rows,list(range(len(rows))),phase)]
        selected=[w for w in rows if w['settled_context']['usable']]
        count=independent_sections(selected)
        reasons=Counter(reason for w in rows if not w['settled_context']['usable'] for reason in w['settled_context']['reasons'])
        screen=dict(activity_id=identity,start_time=original['start_time'],original_pace_min_mile=original['pace_min_mile'],
            original_windows=len(rows),settled_windows=len(selected),independent_sections=count,
            rejected_reasons=dict(reasons),used=count>=2,
            reason='settled_sections' if count>=2 else 'insufficient_settled_sections')
        audit.append(screen)
        if count<2:
            excluded.append(screen);continue
        value=compare_run_windows(selected,[_primary_window_weight(w,config) for w in selected],config,report['calibration'],
                                  precision_seed=20261015+ordinal)
        results.append({**original,**value,'section_screen':screen})
    return {**report,'runs':results,'section_screen':dict(version=VERSION,
        before_seconds=BEFORE_SECONDS,after_seconds=AFTER_SECONDS,bin_seconds=BIN_SECONDS,
        minimum_independent_sections=2,audit=audit,excluded_runs=excluded,
        assumptions='Existing steady-benchmark tolerances applied to surrounding effort/HR; two disjoint sections required. Engineering comparability checks, not physiological truth.')}


def annotate_settled_context(intervals, rows, config):
    if not intervals or not rows:
        return
    origin=intervals[0].start.timestamp_utc.timestamp()
    end=intervals[-1].end.timestamp_utc.timestamp()
    bins=defaultdict(lambda:dict(seconds=0.,effort=0.,hr=0.,hr_seconds=0.,stopped=0.,unknown=0.))
    max_grade=float(config['elevation']['maximum_plausible_grade_percent'])
    for interval in intervals:
        low=interval.start.timestamp_utc.timestamp()-origin
        high=interval.end.timestamp_utc.timestamp()-origin
        if high<=low:
            continue
        speed=interval.distance_m/(high-low)
        ratio=1.
        if interval.elevation_delta_m is not None and interval.distance_m>=1.:
            grade=100*interval.elevation_delta_m/interval.distance_m
            if abs(grade)<=max_grade:
                ratio=grade_energy_ratio(grade,max_grade)
        hrs=[v for v in (interval.start.heart_rate_bpm,interval.end.heart_rate_bpm) if v is not None]
        for index in range(math.floor(low/BIN_SECONDS),math.ceil(high/BIN_SECONDS)):
            seconds=max(0.,min(high,(index+1)*BIN_SECONDS)-max(low,index*BIN_SECONDS))
            if seconds<=0:
                continue
            b=bins[index];b['seconds']+=seconds;b['effort']+=speed*ratio*seconds
            b['stopped']+=seconds*interval.stopped_time_s/(high-low)
            if hrs:
                b['hr']+=sum(hrs)/len(hrs)*seconds;b['hr_seconds']+=seconds
            if high-low>30 and interval.moving_time_s>0:
                b['unknown']+=seconds
    settings=config['model']
    for row in rows:
        start=datetime.fromisoformat(row['window_start_time_utc']).timestamp()
        finish=datetime.fromisoformat(row['window_end_time_utc']).timestamp()
        low=start-BEFORE_SECONDS;high=finish+AFTER_SECONDS
        clipped_low=max(origin,low);clipped_high=min(end,high)
        selected=[b for index,b in sorted(bins.items()) if origin+(index+.5)*BIN_SECONDS>=clipped_low
                  and origin+(index+.5)*BIN_SECONDS<clipped_high and b['seconds']>=BIN_SECONDS*.8]
        reasons=[]
        if low<origin or high>end:
            reasons.append('incomplete_surrounding_context')
        effort=[b['effort']/b['seconds'] for b in selected]
        hr=[b['hr']/b['hr_seconds'] for b in selected if b['hr_seconds']>=b['seconds']*.9]
        span=None;hr_change=None;hr_range=None
        if not effort or np.median(effort)<=0 or len(hr)!=len(selected):
            reasons.append('missing_context_signal')
        else:
            # Bins already average 30 seconds: do not trim away the very brief
            # slowdown whose delayed HR benefit we need to detect.
            span=float((max(effort)-min(effort))/np.median(effort))
            changes=[float(np.mean(hr[-2:])-np.mean(hr[:2]))]
            changes.extend(float(np.mean(hr[i+2:i+4])-np.mean(hr[i:i+2])) for i in range(max(0,len(hr)-3)))
            hr_change=max(changes,key=abs)
            hr_range=float(max(hr)-min(hr))
            if span>float(settings['steady_benchmark_maximum_speed_change_fraction']):
                reasons.append('effort_transition')
            if abs(hr_change)>float(settings['steady_benchmark_maximum_hr_change_bpm']):
                reasons.append('hr_still_changing')
            if hr_range>float(settings['steady_benchmark_maximum_hr_range_bpm']):
                reasons.append('hr_context_range')
        stopped=sum(b['stopped'] for b in selected)
        if stopped>float(settings['steady_benchmark_maximum_stopped_seconds']):
            reasons.append('recent_or_following_stop')
        if any(b['unknown']>0 for b in selected):
            reasons.append('unknown_moving_gap')
        row['settled_context']=dict(usable=not reasons,reasons=reasons,
            effort_span_fraction=span,hr_change_bpm=hr_change,hr_range_bpm=hr_range,
            stopped_seconds=stopped,before_seconds=BEFORE_SECONDS,after_seconds=AFTER_SECONDS,
            context_seconds=sum(b['seconds'] for b in selected),version=VERSION)
