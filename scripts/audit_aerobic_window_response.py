"""Isolate HR timing in the existing comparable-pace scores without exclusions."""
from argparse import ArgumentParser
from collections import defaultdict
from dataclasses import replace
import json
import os
from pathlib import Path
import sqlite3

import numpy as np

from run_analysis.config import load_config, resolve_project_path
from run_analysis.model_windows import load_overlapping_model_windows
from run_analysis.movement import classify_movement
from run_analysis.objective_modeling import _primary_window_weight, _pre_quality_indexes
from run_analysis.pace_comparability import build_pace_comparability, compare_run_windows
from run_analysis.processing import _load_points
from run_analysis.quality_phases import detect_continuous_quality_phase
from run_analysis.read_cache import invariant_snapshot
from run_analysis.aerobic_response_evidence import BalanceBlock, TAUS
from run_analysis.aerobic_hr_timing import statistics, fit, response_corrections, aligned_windows


def audit(connection, config, trace_report):
    base = build_pace_comparability(connection, config)
    if base['calibration'] != trace_report['calibration']:
        raise ValueError('Trace report does not match current calibration')
    source = {r['activity_id']: r for r in base['runs'] if r['included_in_trend'] and r['trend_weight'] > 0}
    traced = {r['activity_id']: r for r in trace_report['runs']}
    if source.keys() != traced.keys() or any(source[k]['pace_min_mile'] != traced[k]['canonical_pace'] for k in source):
        raise ValueError('Trace report does not match current run evidence')
    micros = {r['activity_id']: [BalanceBlock(**b) for b in r['blocks']] for r in trace_report['microblocks']}
    windows, _ = load_overlapping_model_windows(connection, config, window_seconds=120, stride_seconds=60, check_elevation=True)
    groups = defaultdict(list)
    for w in windows:
        groups[w['activity_id']].append(w)
    results = []
    for k, original in source.items():
        rows = groups[k]
        if rows and rows[0].get('workout_type') == 'tempo_threshold':
            phase = detect_continuous_quality_phase(connection, config, k)
            rows = [rows[i] for i in _pre_quality_indexes(rows, list(range(len(rows))), phase)]
        weights = [_primary_window_weight(w, config) for w in rows]
        check = compare_run_windows(rows, weights, config, base['calibration'])
        np.testing.assert_allclose(check['pace_min_mile'], original['pace_min_mile'], rtol=1e-12)
        intervals = classify_movement(_load_points(connection, k), config['moving_time']).intervals
        origin = intervals[0].start.timestamp_utc.timestamp()
        fitted = fit(statistics(micros[k])); i = TAUS.index(traced[k]['tau'])
        corrections = response_corrections(micros[k], float(fitted['inverse_reference_speed'][i]), TAUS[i])
        aligned = aligned_windows(rows, corrections, origin_epoch=origin, config=config, calibration=base['calibration'])
        after = compare_run_windows(aligned, weights, config, base['calibration'])
        # Control uses tau=0, and must reproduce every original window score.
        null = aligned_windows(rows, response_corrections(micros[k], float(fitted['inverse_reference_speed'][0]), 0.),
            origin_epoch=origin, config=config, calibration=base['calibration'])
        np.testing.assert_allclose(compare_run_windows(null, weights, config, base['calibration'])['pace_min_mile'],
            original['pace_min_mile'], rtol=1e-12)
        results.append(dict(activity_id=k, date=original['start_time'], before=original['pace_min_mile'],
            after=after['pace_min_mile'], delta_seconds=60*(after['pace_min_mile']-original['pace_min_mile']),
            source_windows=len(rows), retained_windows=len(aligned),
            adjusted_windows=sum(w['hr_timing']['available'] for w in aligned), tau=TAUS[i],
            sections=[dict(minute=a['moving_minutes_into_run'], raw_pace=a['moving_pace_min_mile'],
                before_hr=a['average_hr_bpm'], after_hr=b['average_hr_bpm'], **b['hr_timing']) for a, b in zip(rows, aligned)]))
    return dict(version='existing-window-hr-response-v1', live_changed=False, source_read_only=True, runs=results,
        runs_retained=len(results), windows_retained=sum(r['retained_windows'] for r in results),
        windows_adjusted=sum(r['adjusted_windows'] for r in results),
        median_absolute_change_seconds=float(np.median([abs(r['delta_seconds']) for r in results])),
        assumptions=[
            'Original windows, original relevance weights, original raw pace, original environment/HR/drift calibration.',
            'Observed HR effort plus modeled instantaneous-minus-delayed HR effort; measurement residuals retained.',
            'Response time chosen without the target run; full recorded moving HR fits its nuisance level.',
            'Missing context retains original estimate with an explicit reason. No dropped runs or windows.',
            'Zero-response control reproduces every canonical score.',
            'This isolates a timing correction, not a new full-run normalization or independently measured fitness.'
        ])


def main():
    p = ArgumentParser(description=__doc__); p.add_argument('--trace-report', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); args = p.parse_args()
    if args.output.exists():
        p.error('Use a new output path')
    report = json.loads(args.trace_report.read_text())
    c = load_config('config.yaml'); database = resolve_project_path(Path.cwd(), c['paths']['database'])
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        with invariant_snapshot(db):
            result = audit(db, c, report)
            assert db.total_changes == 0
    fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as out:
        json.dump(result, out, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in result.items() if k != 'runs'}, indent=2))
    print(json.dumps([{k: v for k, v in r.items() if k != 'sections'} for r in result['runs']
                      if r['activity_id'] in (352, 353, 356, 34, 373, 390)], indent=2))


if __name__ == '__main__':
    main()
