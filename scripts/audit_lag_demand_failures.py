"""Explain frozen lag-reference failures; no replacement fit or live integration.

Reconstruct exactly the old design for rank/covariance/leverage diagnostics.
Reduced design rank is NOT a fitted alternative forecast or accepted transfer.
"""
from argparse import ArgumentParser
from collections import Counter
from datetime import datetime
from hashlib import sha256
import json
from math import exp, log
from pathlib import Path
import sqlite3
from time import perf_counter
from zipfile import ZipFile

import numpy as np
import yaml

from run_analysis.fit import parse_fit
from run_analysis.tcx import parse_tcx
from run_analysis.movement import classify_movement
from run_analysis.training_signals import HeartRateReference


def design_diagnostic(rows, channel):
    if not rows:
        return dict(samples=0, sources=0, rank=0, reduced_intensity_rank=0,
                    constant_duration=False, coefficients=None)
    counts = Counter(r['source_key'] for r in rows)
    w = np.array([1/counts[r['source_key']] for r in rows])
    x = np.array([[1., r[channel], log(r['seconds'])] for r in rows])
    y = np.log([r['speed_mps'] for r in rows])
    matrix = x*np.sqrt(w)[:, None]
    coefficients, _, rank, _ = np.linalg.lstsq(matrix, y*np.sqrt(w), rcond=None)
    # Frisch-Waugh-Lovell decomposition: intensity/pace residuals after duration.
    z = x[:, [0, 2]]
    nuisance = z*np.sqrt(w)[:, None]
    rx = x[:, 1]-z@np.linalg.lstsq(nuisance, x[:, 1]*np.sqrt(w), rcond=None)[0]
    ry = y-z@np.linalg.lstsq(nuisance, y*np.sqrt(w), rcond=None)[0]
    between = within = 0.
    leverage = np.sum(matrix*np.linalg.pinv(matrix).T, axis=1)
    sources = []
    for key in sorted(counts):
        indices = np.array([r['source_key'] == key for r in rows])
        mx, my = (float(np.average(v[indices], weights=w[indices])) for v in (rx, ry))
        between += float(sum(w[indices])*mx*my)
        within += float(np.sum(w[indices]*(rx[indices]-mx)*(ry[indices]-my)))
        sources.append(dict(key=key, samples=counts[key],
            kinds=dict(Counter(r['kind'] for r in rows if r['source_key'] == key)),
            design_leverage=float(sum(leverage[indices]))))
    variance = float(np.sum(w*rx*rx))
    covariance = float(np.sum(w*rx*ry))
    if abs(covariance-between-within) > 1e-10:
        raise AssertionError('Covariance attribution identity failed')
    return dict(samples=len(rows), sources=len(counts), rank=int(rank),
        reduced_intensity_rank=int(np.linalg.matrix_rank(matrix[:, :2])),
        constant_duration=len({r['seconds'] for r in rows}) == 1,
        kinds=dict(Counter(r['kind'] for r in rows)),
        durations=sorted({r['seconds'] for r in rows}),
        coefficients=[float(v) for v in coefficients] if rank == 3 else None,
        intensity_residual_variance=variance, conditional_covariance=covariance,
        between_source_covariance=between, within_source_covariance=within,
        source_diagnostics=sources)


def coverage_diagnostic(database, config, cases):
    reference = HeartRateReference(config['resting_hr'], config['max_hr'], 'frozen audit')
    results = []
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro', uri=True) as db:
        for case in cases:
            bouts = [r for r in case['lag_reference_diagnostic']['records']
                     if r['demand_measurement']['status'] == 'incomplete_hr']
            if not bouts:
                continue
            start, source = db.execute('''SELECT a.start_time_utc,s.path FROM activities a
                JOIN activity_sources l ON l.activity_id=a.id
                JOIN source_files s ON s.id=l.source_file_id
                WHERE a.id=? AND l.is_primary=1''', (case['key'],)).fetchone()
            path = Path(source)
            parsed = {'.fit': parse_fit, '.tcx': parse_tcx}[path.suffix.lower()](
                path, default_timezone=config['timezone_default'])
            matches = [a for a in parsed.activities if a.start_time_utc == datetime.fromisoformat(start)]
            if len(matches) != 1:
                raise AssertionError('Original activity identity mismatch')
            activity = matches[0]
            intervals = classify_movement(activity.trackpoints, config['moving_time']).intervals
            for bout in bouts:
                left, right = bout['start_seconds'], bout['end_seconds']
                reasons = Counter()
                for interval in intervals:
                    a, b = ((p.timestamp_utc-activity.start_time_utc).total_seconds()
                            for p in (interval.start, interval.end))
                    seconds = max(0., min(right, b)-max(left, a))
                    if not seconds:
                        continue
                    h = (interval.start.heart_rate_bpm, interval.end.heart_rate_bpm)
                    reason = ('recording_gap' if 'large_recording_gap' in interval.flags else
                        'missing_hr' if any(v is None or v <= 0 for v in h) else
                        'outside_hr_reference' if any(reference.reserve(v) is None for v in h) else 'valid')
                    reasons[reason] += seconds
                if abs(reasons['valid']-bout['demand_measurement']['known_seconds']) > 1e-6:
                    raise AssertionError('Original valid HR support does not reproduce')
                results.append(dict(key=case['key'], source_sha256=sha256(path.read_bytes()).hexdigest(),
                    start=left, end=right, seconds_by_reason=dict(reasons),
                    unrepresented_seconds=right-left-sum(reasons.values())))
    return results


def analyze(comparison):
    samples = {r['key']: r for r in comparison['supported_samples']}
    cases = []
    for case in comparison['cases']:
        diagnostic = case['lag_reference_diagnostic']
        models = diagnostic['models']
        details = {}
        for name, channel in (('demand_regression', 'demand'), ('mean_hrr_regression', 'mean_hrr')):
            model = models[name]
            rows = [samples[k] for k in model['sample_keys']]
            d = design_diagnostic(rows, channel)
            if model['coefficients'] is not None and not np.allclose(
                    d['coefficients'], model['coefficients'], atol=1e-11, rtol=1e-11):
                raise AssertionError('Original coefficients changed')
            details[name] = dict(d, original_status=model['status'])
        records = diagnostic['records']
        common = [r for r in records if r['predictions'].get('demand_regression', {}).get('metrics')]
        errors = {}
        if common:
            total = sum(r['seconds'] for r in common)
            for name in ('duration_regression', 'mean_hrr_regression', 'demand_regression'):
                errors[name] = sum(r['seconds']*abs(r['predictions'][name]['metrics']['pace_fraction_error'])
                                  for r in common)/total
            old = [r for r in common if r['original_quality_duration_reference']]
            if len(old) == len(common):
                errors['original_quality_duration_reference'] = sum(r['seconds']*abs(
                    r['original_quality_duration_reference']['metrics']['pace_fraction_error'])
                    for r in old)/total
        coefficient = models['demand_regression']['coefficients']
        perturbations = [(exp(abs(coefficient[1])*r['demand_measurement']['demand_change_bound_per_one_bpm'])-1)
            for r in common] if coefficient else []
        cases.append(dict(key=case['key'], label=case['label'], designs=details,
            work_pace_mae=errors, extrapolated_bouts=sum(bool(
                r['predictions']['demand_regression']['extrapolation']) for r in common),
            one_bpm_speed_perturbation_bound=(min(perturbations), max(perturbations)) if perturbations else None))
    recoverable = [c['key'] for c in cases if c['designs']['demand_regression']['constant_duration']
        and c['designs']['demand_regression']['rank'] < 3
        and c['designs']['demand_regression']['reduced_intensity_rank'] == 2]
    negative = [c['key'] for c in cases if c['designs']['demand_regression']['original_status']
                == 'nonpositive_intensity_relation']
    return dict(cases=cases, constant_duration_intensity_identifiable_cases=recoverable,
        nonpositive_cases=negative, sample_kinds=dict(Counter(r['kind'] for r in samples.values())),
        limits=['Design/covariance/leverage attribution is not a new accepted predictor',
            'No new forecast, coefficient sweep or target-informed history',
            'One-bpm sensitivity is not actual sensor error or an uncertainty interval',
            'Between-source covariance does not identify weather, health, route or fatigue causes'],
        planner_influence=False)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--baseline-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    started = perf_counter()
    database = args.baseline_root/'evidence.sqlite'
    before = {str(p): sha256(p.read_bytes()).hexdigest() for p in (args.comparison, database)}
    comparison = json.loads(args.comparison.read_text())
    with ZipFile(args.baseline_root/'sources.zip') as archive:
        config = yaml.safe_load(archive.read('config.yaml'))
    result = analyze(comparison)
    result['incomplete_hr_attribution'] = coverage_diagnostic(database, config, comparison['cases'])
    if before != {str(p): sha256(p.read_bytes()).hexdigest() for p in (args.comparison, database)}:
        raise AssertionError('Frozen input changed')
    result.update(input_sha256=before, frozen_inputs_unchanged=True, seconds=perf_counter()-started)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({k: result[k] for k in ('constant_duration_intensity_identifiable_cases',
        'nonpositive_cases', 'sample_kinds', 'incomplete_hr_attribution', 'seconds', 'frozen_inputs_unchanged')}, indent=2))


if __name__ == '__main__':
    main()
