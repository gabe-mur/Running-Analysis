"""Read-only shared-state and individual-run influence audit, not live fitting."""
from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime, timedelta
import hashlib
import json
import math
from pathlib import Path
import sqlite3

import numpy as np

from run_analysis import aerobic_progress as ap, pace_comparability as pc
from run_analysis.config import load_config, resolve_project_path
from run_analysis.read_cache import invariant_snapshot
from scripts.aerobic_shared_state import fit_shared_state, state_contrast
from scripts.audit_aerobic_three_layers import inputs, CONTRASTS
from scripts.audit_progress_responsiveness import at, build_from_windows, covariance
from scripts.audit_small_aerobic_changes import describe


def robust(rows, end, cov):
    current = ap._period(rows, end, cov)
    result = dict(current_pace_min_mile=current['pace_min_mile'])
    for target, days in (('monthly', 28), ('ninety_day', 90)):
        prior = ap._period(rows, end-timedelta(days=days), cov)
        result[target] = dict(estimate=math.log(prior['pace_min_mile']/current['pace_min_mile']))
    return result


def point(rows, y, m, g, cov, groups, times, end):
    state = fit_shared_state(y, m, g, cov, groups, times=times)
    return dict(shared=state, contrasts={name: state_contrast(state, v) for name, v in CONTRASTS.items()},
                robust=robust(rows, end, cov))


def difference(value, baseline):
    return dict(current_shift_seconds_per_mile=60*(value['shared']['current_pace_min_mile']-baseline['shared']['current_pace_min_mile']),
        robust_current_shift_seconds_per_mile=60*(value['robust']['current_pace_min_mile']-baseline['robust']['current_pace_min_mile']),
        contrast_shift_log={name: value['contrasts'][name]['estimate']-baseline['contrasts'][name]['estimate'] for name in CONTRASTS},
        robust_contrast_shift_log={name: value['robust'][name]['estimate']-baseline['robust'][name]['estimate'] for name in CONTRASTS},
        kernel=value['shared']['kernel'], unconverged_kernels=value['shared']['unconverged_kernels'])


def audit(database, config, end, precision_file, frozen_file, *, everyday=(.03,.05), extreme=(.1,.2)):
    precision = json.loads(precision_file.read_text())
    frozen = json.loads(frozen_file.read_text())['statistical_inputs']
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro', uri=True) as connection:
        connection.row_factory = sqlite3.Row
        with invariant_snapshot(connection):
            sets, _ = pc.load_model_window_sets(connection, config,
                (int(config['model'].get('window_seconds', 300)),), check_elevation=True)
            windows, _ = pc.load_overlapping_model_windows(connection, config,
                window_seconds=int(config['model']['primary_window_seconds']),
                stride_seconds=int(config['model']['primary_window_stride_seconds']), check_elevation=True)
            sets = {k: [r for r in v if datetime.fromisoformat(r['start_time_utc']) <= end] for k, v in sets.items()}
            windows = [r for r in windows if datetime.fromisoformat(r['start_time_utc']) <= end]
            report = build_from_windows(connection, config, sets, windows); before = deepcopy(report)
            rows, m, g, groups, latest, _, _ = inputs(report, precision, end)
            cov = covariance(report); y = np.log([r['pace_min_mile'] for r in rows])
            times = np.array([(at(r)-latest).total_seconds()/86400 for r in rows])
            for name, value in dict(actual_log_pace=y, measurement_variance=m, gradients=g,
                shared_covariance=cov, groups=groups, times=times).items():
                np.testing.assert_allclose(value, frozen[name], rtol=1e-12, atol=1e-12)
            baseline = point(rows, y, m, g, cov, groups, times, latest)
            leave_one_out = []; contaminated = []
            for i, row in enumerate(rows):
                mask = np.arange(len(rows)) != i
                subset = [r for j, r in enumerate(rows) if j != i]
                fit = point(subset, y[mask], m[mask], g[mask], cov, groups[mask], times[mask], latest)
                leave_one_out.append(dict(activity_id=row['activity_id'], period=int(groups[i]),
                    start_time=row['start_time'], **difference(fit, baseline)))
            # Every current-period run is stressed, never just a handpicked
            # convenient activity. Hold all precision and exposure inputs fixed.
            for i in np.flatnonzero(groups == 0):
                for category, magnitudes in (('everyday_sensitivity',everyday),('extreme_data_stress',extreme)):
                  for change in [sign*value for value in magnitudes for sign in (-1,1)]:
                    changed_y = y.copy(); changed_y[i] += math.log1p(change)
                    changed = deepcopy(rows); changed[i]['pace_min_mile'] *= 1+change
                    fit = point(changed, changed_y, m, g, cov, groups, times, latest)
                    contaminated.append(dict(activity_id=rows[i]['activity_id'], category=category, pace_change_percent=change*100,
                        **difference(fit, baseline)))
            assert report == before and connection.total_changes == 0
    summary = {}
    sections=dict(leave_one_out=leave_one_out,current_run_leave_one_out=[v for v in leave_one_out if v['period']==0])
    sections.update({category:[v for v in contaminated if v['category']==category]
        for category in ('everyday_sensitivity','extreme_data_stress')})
    sections.update({f'pace_change_{100*magnitude:g}_percent':[v for v in contaminated if math.isclose(abs(v['pace_change_percent']),100*magnitude)]
        for magnitude in everyday+extreme})
    for name, samples in sections.items():
        summary[name] = dict(current_absolute_shift_seconds_per_mile=describe([abs(v['current_shift_seconds_per_mile']) for v in samples]),
            robust_current_absolute_shift_seconds_per_mile=describe([abs(v['robust_current_shift_seconds_per_mile']) for v in samples]),
            contrast_absolute_shift_log={target: describe([abs(v['contrast_shift_log'][target]) for v in samples]) for target in CONTRASTS},
            optimizer_warning_cases=sum(bool(v['unconverged_kernels']) for v in samples))
    root = Path(__file__).resolve().parents[1]
    current=np.exp(y[groups==0]);log_current=y[groups==0]
    return dict(version='aerobic-shared-influence-v2-ordinary-and-extreme', deployed=False, source_database_read_only=True,
        frozen_cohort_matched=True, as_of=end.isoformat(), baseline=baseline, summary=summary,
        observed_current_variation=dict(run_count=len(current),sample_pace_cv_percent=100*float(np.std(current,ddof=1)/np.mean(current)),
            absolute_log_deviation_from_median_percent={str(q):100*math.expm1(float(np.quantile(abs(log_current-np.median(log_current)),q))) for q in (.5,.9)},
            scope='Descriptive variation in ten adjusted runs, not a measurement-error distribution or pure physiological day SD.'),
        leave_one_out=leave_one_out, single_run_stress=contaminated,
        source_sha256={name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in (
            'scripts/aerobic_shared_state.py', 'scripts/audit_aerobic_shared_influence.py',
            'scripts/aerobic_joint_profile.py', 'src/run_analysis/aerobic_progress.py')},
        frozen_sha256=hashlib.sha256(frozen_file.read_bytes()).hexdigest(), assumptions=[
            'Existing full ML likelihood and unchanged nine kernels; one fitted mean vector for every contrast.',
            'Leave-one-out keeps original comparison dates and period membership fixed, including when the latest run is removed.',
            'Everyday sensitivity uses plus/minus 3/5 percent in every current-period run; larger 10/20 percent cases are separately labeled extreme data stresses.',
            'These test magnitudes are not error-rate estimates or acceptance thresholds; observed current-run variation provides descriptive context only.',
            'Measurement precision, shared calibration and context remain fixed to isolate estimator influence.',
            'Conditional standard errors are scale diagnostics, not calibrated confidence intervals or posterior probabilities.',
            'No observation is rejected or edited in the source; no live signal, correction, VO2 or planner change.'])


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--precision', type=Path, default=Path('output/run-pace-precision-coverage-20261014.json'))
    parser.add_argument('--frozen', type=Path, default=Path('output/aerobic-joint-bootstrap-validation-20261014.json'))
    parser.add_argument('--as-of', type=datetime.fromisoformat, default=datetime.fromisoformat('2026-10-14T22:30:00+00:00'))
    parser.add_argument('--everyday-percent',type=float,nargs='+',default=[3,5])
    parser.add_argument('--extreme-percent',type=float,nargs='+',default=[10,20])
    args = parser.parse_args()
    if args.output.exists() or args.as_of.tzinfo is None or not all(np.isfinite(v) and 0<v<100 for v in args.everyday_percent+args.extreme_percent):
        parser.error('New output and timezone-aware date required')
    root = Path(__file__).resolve().parents[1]; config = load_config(root/'config.yaml')
    result = audit(resolve_project_path(root, config['paths']['database']), config,
        args.as_of, args.precision, args.frozen,
        everyday=tuple(v/100 for v in args.everyday_percent),extreme=tuple(v/100 for v in args.extreme_percent))
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(current_pace_min_mile=result['baseline']['shared']['current_pace_min_mile'],
        observed_current_variation=result['observed_current_variation'],summary=result['summary']), indent=2))


if __name__ == '__main__':
    main()
