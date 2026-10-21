"""Read-only exercise-HRV validation and matched-segment pilot.

No DB import, scheduler call, VO2 fitting or automatic threshold recalibration.
Optional nolds wheel is a downloaded reference only, not an app dependency.
"""
from argparse import ArgumentParser
import ast
from collections import Counter
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
from time import perf_counter
import warnings
from zipfile import ZipFile

import numpy as np

from run_analysis.config import load_config
from run_analysis.exercise_hrv import (
    SCALES, dfa_alpha1, exercise_rr_windows, screen_rr, smoothness_detrend,
)
from run_analysis.model_windows import load_overlapping_model_windows
from run_analysis.pace_comparability import cached_comparability, compare_run_windows, _features
from run_analysis.read_cache import invariant_snapshot


def reference_dfa(wheel):
    """Evaluate only the inspected mathematical functions from the MIT wheel.

    Explicit parameters avoid nolds' sklearn/RANSAC or plotting dependencies;
    no archive extraction and no package installation into the application.
    """
    with ZipFile(wheel) as archive:
        source = archive.read('nolds/measures.py').decode()
    tree = ast.parse(source)
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {'dfa', 'poly_fit'}]
    if len(selected) != 2:
        raise ValueError('Expected nolds DFA reference functions')
    namespace = dict(np=np, warnings=warnings)
    exec(compile(ast.Module(body=selected, type_ignores=[]), 'nolds-reference', 'exec'), namespace)
    return lambda rr: float(namespace['dfa'](rr, nvals=SCALES, overlap=True, order=1,
                                             fit_trend='poly', fit_exp='poly'))


def numerical_reference_check(reference):
    cases = []
    for n in (180, 257, 300, 1000):
        for seed in (17, 812):
            noise = np.random.default_rng(seed).normal(0, .01, n)
            for kind, rr in (('white', .45+noise), ('correlated', .45+np.cumsum(noise)*.01)):
                for preprocessing, values in (('unfiltered', rr), ('smoothness_500', smoothness_detrend(rr))):
                    actual, expected = dfa_alpha1(values)['alpha1'], reference(values)
                    cases.append(dict(n=n, seed=seed, signal=kind, preprocessing=preprocessing,
                                      ours=actual, reference=expected, error=abs(actual-expected)))
    maximum = max(r['error'] for r in cases)
    if maximum > 1e-9:
        raise AssertionError('DFA arithmetic disagrees with reference')
    return dict(reference='nolds 0.6.2, overlap=True, order=1, scales=4..16, both fits=poly',
        cases=cases, maximum_absolute_error=maximum,
        validates='DFA arithmetic only, not FIT beat timing or physiological interpretation')


def matched_checks(rows):
    """Match on specified context, never alpha; each run contributes once."""
    rows=sorted(rows,key=lambda r:(r['run_start'],r['activity_id'],r['start_seconds']))
    result = {}
    for mode in ('similar_hr', 'similar_output'):
        candidates = []
        for i, a in enumerate(rows):
            for b in rows[i+1:]:
                if a['activity_id'] == b['activity_id']:
                    continue
                gap = abs((datetime.fromisoformat(b['run_start'])-datetime.fromisoformat(a['run_start'])).total_seconds())/86400
                if gap > 21 or abs(a['minutes_into_run']-b['minutes_into_run']) > 5:
                    continue
                if a['heat_c'] is None or b['heat_c'] is None or abs(a['heat_c']-b['heat_c']) > 3:
                    continue
                if abs(a['wind_log']-b['wind_log']) > .02 or abs(a['grade_ratio']-b['grade_ratio']) > .05:
                    continue
                if mode == 'similar_hr' and abs(a['hr_bpm']-b['hr_bpm']) > 3:
                    continue
                if mode == 'similar_output' and abs(np.log(a['terrain_wind_speed_mps']/b['terrain_wind_speed_mps'])) > np.log(1.03):
                    continue
                candidates.append((gap, abs(a['minutes_into_run']-b['minutes_into_run']), a, b))
        used, pairs = set(), []
        for gap, _, a, b in sorted(candidates, key=lambda p:(p[0],p[1],p[2]['activity_id'],p[3]['activity_id'],p[2]['start_seconds'],p[3]['start_seconds'])):
            if a['activity_id'] in used or b['activity_id'] in used:
                continue
            used.update((a['activity_id'],b['activity_id']))
            if a['run_start'] > b['run_start']:
                a, b = b, a
            pairs.append(dict(earlier=a,later=b,gap_days=gap,
                alpha1_change=b['alpha1']-a['alpha1'],hr_change_bpm=b['hr_bpm']-a['hr_bpm'],
                adjusted_speed_change_percent=100*(a['adjusted_pace_min_mile']/b['adjusted_pace_min_mile']-1)))
        result[mode] = dict(candidate_pairs=len(candidates), independent_run_pairs=len(pairs), pairs=pairs,
            incremental_predictive_value_established=False,
            limitation='Descriptive pilot; shared full-history pace calibration is not a prospective validation')
    return result


def audit(database, config, reference=None):
    started = perf_counter()
    with sqlite3.connect(f'{database.resolve().as_uri()}?mode=ro',uri=True) as connection:
        connection.row_factory = sqlite3.Row
        with invariant_snapshot(connection):
            sources = connection.execute('''SELECT a.id,a.start_time_utc,a.start_time_local,
                a.total_elapsed_time_s,r.payload_json,s.sha256,s.path
                FROM activity_sensor_streams r JOIN activities a ON a.id=r.activity_id
                JOIN source_files s ON s.id=r.source_file_id ORDER BY a.start_time_utc,s.id DESC''').fetchall()
            selected = {}
            for row in sources:
                payload = json.loads(row['payload_json'])
                if payload.get('rr_batches') and row['id'] not in selected:
                    selected[row['id']] = (row,payload)
            report = cached_comparability(connection,config)
            windows,_ = load_overlapping_model_windows(connection,config,
                window_seconds=120,stride_seconds=30,check_elevation=True)
            assert connection.total_changes == 0
    runs, comparable, real_reference = [], [], []
    for identity, (source,payload) in selected.items():
        analysis = exercise_rr_windows(payload,source['start_time_utc'],source['total_elapsed_time_s'])
        if reference:
            candidate=next((w for w in analysis['windows'] if w['usable_for_comparison']),None)
            if candidate:
                rr=np.array([value for batch in payload['rr_batches']
                    if batch.get('preceding_record_utc') and batch.get('following_record_utc')
                    and candidate['start_seconds'] <= (datetime.fromisoformat(batch['preceding_record_utc'])-datetime.fromisoformat(source['start_time_utc'])).total_seconds()
                    and (datetime.fromisoformat(batch['following_record_utc'])-datetime.fromisoformat(source['start_time_utc'])).total_seconds() <= candidate['end_seconds']
                    for value in batch['intervals_seconds'] if value is not None])
                flagged=screen_rr(rr,.20);cleaned=rr.copy()
                indexes=np.flatnonzero(flagged)
                if len(indexes):cleaned[indexes]=np.interp(indexes,np.flatnonzero(~flagged),rr[~flagged])
                expected=reference(smoothness_detrend(cleaned))
                error=abs(candidate['alpha1']-expected)
                if error>1e-9:raise AssertionError('Real-window DFA disagrees with reference')
                real_reference.append(dict(activity_id=identity,start_seconds=candidate['start_seconds'],
                    ours=candidate['alpha1'],reference=expected,error=error))
        candidates = []
        raw_start = datetime.fromisoformat(source['start_time_utc'])
        for estimate in analysis['windows']:
            if not estimate['usable_for_comparison']:
                continue
            matches = []
            for w in windows:
                if w['activity_id'] != identity:
                    continue
                a = (datetime.fromisoformat(w['window_start_time_utc'])-raw_start).total_seconds()
                b = (datetime.fromisoformat(w['window_end_time_utc'])-raw_start).total_seconds()
                overlap = max(0.,min(b,estimate['end_seconds'])-max(a,estimate['start_seconds']))
                if overlap >= .9*120:
                    matches.append((abs((a+b)/2-(estimate['start_seconds']+60)),w))
            if not matches:
                continue
            w = min(matches,key=lambda x:x[0])[1]
            if (not 5 <= w['moving_minutes_into_run'] <= 45
                    or w.get('heart_rate_change_bpm') is None or abs(w['heart_rate_change_bpm']) > 5
                    or w.get('speed_change_fraction') is None or abs(w['speed_change_fraction']) > .05
                    or (w.get('terrain_quality') or {}).get('calibration_eligible') is False):
                continue
            f = _features(w,config)
            pace = compare_run_windows([w],[1.],config,report['calibration'])
            candidates.append(dict(activity_id=identity,run_start=source['start_time_utc'],
                start_seconds=estimate['start_seconds'],end_seconds=estimate['end_seconds'],
                alpha1=estimate['alpha1'],preprocessing_range=estimate['preprocessing_range'],
                hr_bpm=w['average_hr_bpm'],minutes_into_run=w['moving_minutes_into_run'],
                heat_c=f['heat'],wind_log=f['wind_log'],grade_ratio=f['grade'],
                terrain_wind_speed_mps=1609.344/60/pace['stages_min_mile']['wind'],
                adjusted_pace_min_mile=pace['pace_min_mile']))
        comparable.extend(candidates)
        runs.append(dict(activity_id=identity,date=source['start_time_local'][:10],source_sha256=source['sha256'],
            rr_analysis=analysis,matched_segment_candidates=len(candidates),
            median_alpha1=float(np.median([w['alpha1'] for w in analysis['windows'] if w['usable_for_comparison']]))
                if any(w['usable_for_comparison'] for w in analysis['windows']) else None))
    return dict(version='exercise-hrv-pilot-v1',runs=runs,matched_segments=matched_checks(comparable),
        numerical_reference=dict(**numerical_reference_check(reference),real_windows=real_reference) if reference else dict(status='not_run'),
        available_rr_runs=len(runs),comparable_runs=len({r['activity_id'] for r in comparable}),
        source_database_read_only=True,planner_influence=False,fitness_influence=False,
        elapsed_seconds=perf_counter()-started,
        limitations=['Approximate record brackets, not measured beat timestamps.',
            'Sparse correction is an engineering screen, not Kubios-validated artifact repair.',
            'Matched context is incomplete: breathing, surface, fatigue and device effects remain.',
            'Overlapping windows are not independent runs; no fitted personal recovery or fitness law.'])


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference-wheel',type=Path)
    args=parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output path')
    result=audit(Path('data/run_analysis.sqlite'),load_config('config.yaml'),
        reference_dfa(args.reference_wheel) if args.reference_wheel else None)
    if args.reference_wheel:
        result['numerical_reference']['wheel_sha256']=sha256(args.reference_wheel.read_bytes()).hexdigest()
    from run_analysis.privacy import private_file
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as out:
        json.dump(result,out,indent=2,allow_nan=False)
    private_file(args.output)
    print(json.dumps(dict(runs=[dict(id=r['activity_id'],date=r['date'],counts=r['rr_analysis']['status_counts'],
        matched_candidates=r['matched_segment_candidates'],alpha1=r['median_alpha1']) for r in result['runs']],
        matched_checks={k:{f:v for f,v in r.items() if f!='pairs'} for k,r in result['matched_segments'].items()},
        reference_error=result['numerical_reference'].get('maximum_absolute_error'),seconds=result['elapsed_seconds']),indent=2))


if __name__=='__main__':main()
