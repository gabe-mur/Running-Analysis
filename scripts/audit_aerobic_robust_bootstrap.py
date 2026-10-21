"""Read-only calibration screen for the unchanged robust Progress statistic."""
from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sqlite3

import numpy as np

from run_analysis import pace_comparability as pc
from run_analysis.config import load_config, resolve_project_path
from run_analysis.read_cache import invariant_snapshot
from scripts.aerobic_robust_bootstrap import RobustCurveBootstrap
from scripts.aerobic_three_layer_model import fit_period_layers
from scripts.audit_aerobic_three_layers import inputs, simulations, CONTRASTS, rate
from scripts.audit_progress_responsiveness import at, build_from_windows, covariance
from scripts.audit_small_aerobic_changes import describe


def audit(database, config, end, precision_file, frozen_file, *, draws, inner, seed, scenarios, gains, targets,
          actual_changes=(), grid_draws=127, direction_tests=False):
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
            rows, m, g, groups, latest, _, eligible = inputs(report, precision, end)
            cov = covariance(report)
            y = np.log([r['pace_min_mile'] for r in rows])
            times = np.array([(at(r)-latest).total_seconds()/86400 for r in rows])
            for name, value in dict(actual_log_pace=y, measurement_variance=m, gradients=g,
                shared_covariance=cov, groups=groups, times=times).items():
                np.testing.assert_allclose(value, frozen[name], rtol=1e-12, atol=1e-12)
            model = RobustCurveBootstrap(rows, m, g, cov, groups, times=times)
            actual = {target: model.test(target, 0., draws=inner, seed=seed+900000) for target in targets}
            actual_grid = {}
            for target in targets:
                actual_grid[target] = []
                for proposed in sorted(set(actual_changes)):
                    test = model.test(target, math.log1p(proposed/100), draws=grid_draws, seed=seed+910000)
                    actual_grid[target].append(dict(proposed_change_percent=proposed, **test))
            cells = {}
            def observe(scenario, gain, simulated_y, _, truths):
                simulated = [dict(row, pace_min_mile=float(np.exp(value)))
                    for row, value in zip(rows, simulated_y, strict=True)]
                model = RobustCurveBootstrap(simulated, m, g, cov, groups, times=times)
                for target in targets:
                    key = (scenario, gain, target); index = len(cells.get(key, []))
                    number = int.from_bytes(hashlib.sha256(scenario.encode()).digest()[:4])
                    truth = truths[target][1]
                    result = model.test(target, truth, draws=inner,
                        seed=seed+number*10000+index, decision_only=True)
                    if direction_tests:
                        zero = result if truth == 0 else model.test(target, 0., draws=inner,
                            seed=seed+number*10000+index, decision_only=True)
                        result['zero_test'] = zero.copy() if zero is result else zero
                    cells.setdefault(key, []).append(dict(truth=truth, **result))
                    if len(cells[key]) == draws:
                        print(f'{scenario} gain={gain*100:g}% {target}: '
                            f'{sum(v["covered_95"] for v in cells[key])}/{draws} robust bootstrap covered; '
                            f'{sum(bool(v["null_optimizer_warnings"]) for v in cells[key])} optimizer warnings', flush=True)
            simulations(rows, m, g, cov, groups, latest, eligible, draws, seed, fit_period_layers,
                serial_controls=True, only_scenarios=scenarios, gains=gains, observer=observe)
            assert report == before and connection.total_changes == 0
    summary = []
    for (scenario, gain, target), values in cells.items():
        cell = dict(scenario=scenario, gain_percent=gain*100, target=target,
            bootstrap_coverage_95=rate([v['covered_95'] for v in values]),
            original_normal_coverage_95=rate([v['statistic'] <= 1.96 for v in values]),
            bias_log=float(np.mean([v['estimate']-v['truth'] for v in values])),
            evaluated_draws=describe([v['evaluated_draws'] for v in values]),
            optimizer_warning_draws=sum(bool(v['null_optimizer_warnings']) for v in values),
            null_kernel_counts={k: sum(v['null_kernel'] == k for v in values)
                for k in sorted({v['null_kernel'] for v in values})})
        if direction_tests:
            cell['zero_change_rejection'] = rate([not v['zero_test']['covered_95'] for v in values])
            cell['positive_direction_rejection'] = rate([not v['zero_test']['covered_95'] and v['estimate'] > 0 for v in values])
            cell['zero_test_optimizer_warnings'] = sum(bool(v['zero_test']['null_optimizer_warnings']) for v in values)
        summary.append(cell)
    root = Path(__file__).resolve().parents[1]
    return dict(version='aerobic-robust-bootstrap-audit-v2-grid-and-direction', deployed=False,
        source_database_read_only=True, frozen_cohort_matched=True, as_of=end.isoformat(),
        scope=dict(draws=draws, inner_draws=inner, seed=seed, scenarios=scenarios,
            gains_percent=[v*100 for v in gains], targets=list(targets), direction_tests=direction_tests,
            actual_changes_percent=list(actual_changes), grid_draws=grid_draws),
        actual=actual, actual_grid=actual_grid, coverage=summary,
        frozen_input_sha256=hashlib.sha256(frozen_file.read_bytes()).hexdigest(),
        source_sha256={name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in (
            'scripts/aerobic_robust_bootstrap.py', 'scripts/audit_aerobic_robust_bootstrap.py',
            'scripts/aerobic_joint_profile.py', 'scripts/audit_aerobic_three_layers.py',
            'src/run_analysis/aerobic_progress.py')},
        assumptions=[
            'Existing production pace-domain Huber statistic, context weights and absolute per-run uncertainties retained.',
            'Same known precision/shared covariance and nine predeclared null covariance kernels as the joint-profile prototype.',
            'Only the constrained Gaussian nuisance fit is needed for null simulation; no Gaussian point replaces the robust curve.',
            'Statistic is absolute robust log change minus proposed change, divided by its original conditional standard error.',
            'Every replicate recomputes Huber location, weighting and standard error; no inner covariance fit is required for this statistic.',
            'A fitted Gaussian null remains an assumption; successful LR-bootstrap controls do not validate this different test.',
            'Early decisions exactly match the fixed seeded budget; actual-data p-values use every replicate.',
            'Known truth is latent period log-mean change, not the larger endpoint gain; gain cases share noise draws.',
            'Optional tests of zero change measure conditional false alarms or detection, separately from coverage of known truth.',
            'Actual-data grid tests are not interpolated confidence limits or proof of a connected confidence set.',
            'P-values are tests of proposed changes, not posterior directional probabilities.',
            'No production graph, headline, correction, VO2, planner or canonical evidence change.'])


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--precision', type=Path, default=Path('output/run-pace-precision-coverage-20261014.json'))
    parser.add_argument('--frozen', type=Path, default=Path('output/aerobic-joint-bootstrap-validation-20261014.json'))
    parser.add_argument('--draws', type=int, default=64)
    parser.add_argument('--bootstrap-draws', type=int, default=63)
    parser.add_argument('--seed', type=int, default=20261117)
    parser.add_argument('--scenarios', nargs='+', required=True)
    parser.add_argument('--gains-percent', type=float, nargs='+', default=[0., 2.])
    parser.add_argument('--targets', nargs='+', choices=list(CONTRASTS), default=list(CONTRASTS))
    parser.add_argument('--actual-changes-percent', type=float, nargs='+', default=[])
    parser.add_argument('--grid-draws', type=int, default=127)
    parser.add_argument('--direction-tests', action='store_true')
    parser.add_argument('--as-of', type=datetime.fromisoformat, default=datetime.fromisoformat('2026-10-14T22:30:00+00:00'))
    args = parser.parse_args()
    if (args.output.exists() or args.draws < 2 or args.bootstrap_draws < 19 or args.grid_draws < 19
        or args.as_of.tzinfo is None or not all(np.isfinite(v) and v >= 0 for v in args.gains_percent)
        or not all(np.isfinite(v) and v > -100 for v in args.actual_changes_percent)):
        parser.error('New output, valid draw budgets, timezone-aware date and nonnegative finite gains required')
    root = Path(__file__).resolve().parents[1]; config = load_config(root/'config.yaml')
    result = audit(resolve_project_path(root, config['paths']['database']), config, args.as_of,
        args.precision, args.frozen, draws=args.draws, inner=args.bootstrap_draws, seed=args.seed,
        scenarios=args.scenarios, gains=tuple(v/100 for v in args.gains_percent), targets=tuple(args.targets),
        actual_changes=tuple(args.actual_changes_percent), grid_draws=args.grid_draws, direction_tests=args.direction_tests)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result['actual'], indent=2))


if __name__ == '__main__':
    main()
