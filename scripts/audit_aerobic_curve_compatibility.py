"""Read-only comparison of existing robust curves and offline profile points.

This attributes point-estimate changes; it does not transplant confidence
intervals between estimands or convert bootstrap p-values to probabilities.
"""
from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sqlite3

import numpy as np

from run_analysis import aerobic_progress as ap, pace_comparability as pc
from run_analysis.config import load_config, resolve_project_path
from run_analysis.read_cache import invariant_snapshot
from scripts.aerobic_joint_profile import GaussianChangeProfile
from scripts.audit_aerobic_three_layers import CONTRASTS, inputs
from scripts.audit_progress_responsiveness import at, build_from_windows, covariance


def conditional_rows(rows, measurement, shared_covariance):
    """Replace only per-run precision; retain the production context policy.

    `measurement` already includes the context divisor. Undo that divisor
    before supplying sigma to _period, which applies context itself. Shared
    parameter error remains separate and retains its original gradient.
    """
    result = deepcopy(rows)
    for row, variance in zip(result, measurement, strict=True):
        gradient = np.array(row['shared_log_speed_gradient'])
        shared = max(0., float(gradient @ shared_covariance @ gradient))
        independent = variance * min(1., row['trend_weight'])
        row['uncertainty_95_min_mile'] = 1.96 * row['pace_min_mile'] * math.sqrt(independent + shared)
    return result


def robust_point(rows, end, cov):
    state = ap.build_aerobic_progress(rows, as_of=end, covariance=cov, history_days=1)
    longer = ap.build_aerobic_progress(rows, as_of=end, covariance=cov, history_days=180)
    for key in ('current', 'change', 'change_90d'):
        assert state[key] == longer[key], 'Viewport changed current state'
    return dict(current_pace_min_mile=state['current']['pace_min_mile'],
                monthly_percent=state['change']['speed_change_percent'],
                ninety_day_percent=state['change_90d']['speed_change_percent'],
                history_selector_invariant=True)


def profile_points(y, measurement, gradients, cov, groups, times):
    results = {}
    for mode in ('ml', 'nuisance_residual'):
        model = GaussianChangeProfile(y, measurement, gradients, cov, groups,
                                      times=times, future_time=2., mode=mode)
        fitted = {}
        for target, contrast in CONTRASTS.items():
            fit = model.fit(contrast)
            fitted[target] = dict(current_pace_min_mile=math.exp(fit['beta'][0]),
                change_percent=100*math.expm1(fit['estimate']), kernel=fit['kernel'],
                period_day_log_sd=fit['period_day_log_sd'],
                unconverged_kernels=fit['unconverged_kernels'])
        results[mode] = dict(comparisons=fitted,
            current_level_disagreement_seconds_per_mile=60*abs(
                fitted['monthly']['current_pace_min_mile']-fitted['ninety_day']['current_pace_min_mile']))
    return results


def audit(database, config, end, precision_file, frozen_file):
    frozen = json.loads(frozen_file.read_text())['statistical_inputs']
    precision = json.loads(precision_file.read_text())
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
            report = build_from_windows(connection, config, sets, windows)
            before = deepcopy(report)
            rows, m, g, groups, latest, _, _ = inputs(report, precision, end)
            cov = covariance(report)
            y = np.log([r['pace_min_mile'] for r in rows])
            times = np.array([(at(r)-latest).total_seconds()/86400 for r in rows])
            for name, value in dict(actual_log_pace=y, measurement_variance=m, gradients=g,
                shared_covariance=cov, groups=groups, times=times).items():
                np.testing.assert_allclose(value, frozen[name], rtol=1e-12, atol=1e-12)
            original = robust_point(rows, end, cov)
            conditioned = robust_point(conditional_rows(rows, m, cov), end, cov)
            profiles = profile_points(y, m, g, cov, groups, times)
            # Diagnose a nonlinear pace-to-log transformation separately from
            # changed weights, covariance fitting, or outlier handling.
            equal_arithmetic = np.array([np.exp(y[groups == i]).mean() for i in range(4)])
            equal_geometric = np.array([np.exp(y[groups == i].mean()) for i in range(4)])
            equal = {}
            for name, means in (('arithmetic_pace', equal_arithmetic), ('geometric_pace', equal_geometric)):
                equal[name] = dict(current_pace_min_mile=float(means[0]),
                    monthly_percent=100*float(means[1]/means[0]-1),
                    ninety_day_percent=100*float(means[2]/means[0]-1))
            assert report == before and connection.total_changes == 0
    root = Path(__file__).resolve().parents[1]
    return dict(version='aerobic-curve-compatibility-v1', deployed=False,
        source_database_read_only=True, frozen_cohort_matched=True, run_count=len(rows),
        as_of=end.isoformat(), original_robust_curve=original,
        robust_curve_with_conditional_precision=conditioned, equal_weights=equal, profiles=profiles,
        frozen_input_sha256=hashlib.sha256(frozen_file.read_bytes()).hexdigest(),
        source_sha256={name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in (
            'scripts/audit_aerobic_curve_compatibility.py', 'scripts/aerobic_joint_profile.py',
            'src/run_analysis/aerobic_progress.py')},
        interpretation=[
            'Same physical runs, adjusted pace values, dates and shared calibration in every comparison.',
            'Conditional-precision variant changes per-run uncertainty only; production Huber and context treatment remain.',
            'Equal-weight arithmetic/geometric comparison isolates the unit-transform effect, not an alternative production policy.',
            'Gaussian profile points are not Huber robust points; new intervals cannot silently be attached to the existing curve.',
            'Nuisance-residual profiles can have different current-level points by contrast and are not a shared-state replacement.',
            'A bootstrap p-value is not a posterior probability of improvement.',
            'No live graph, headline, normalization fit, source evidence or planner change.'])


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--precision', type=Path, default=Path('output/run-pace-precision-coverage-20261014.json'))
    parser.add_argument('--frozen', type=Path, default=Path('output/aerobic-joint-bootstrap-validation-20261014.json'))
    parser.add_argument('--as-of', type=datetime.fromisoformat, default=datetime.fromisoformat('2026-10-14T22:30:00+00:00'))
    args = parser.parse_args()
    if args.output.exists() or args.as_of.tzinfo is None:
        parser.error('New output and timezone-aware as-of required')
    root = Path(__file__).resolve().parents[1]
    config = load_config(root/'config.yaml')
    result = audit(resolve_project_path(root, config['paths']['database']), config,
                   args.as_of, args.precision, args.frozen)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: result[k] for k in ('original_robust_curve',
        'robust_curve_with_conditional_precision', 'equal_weights', 'profiles')}, indent=2))


if __name__ == '__main__':
    main()
