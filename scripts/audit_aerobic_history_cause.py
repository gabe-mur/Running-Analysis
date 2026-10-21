"""Read-only covariance ablations and new-upload attribution, not a model fix."""
from argparse import ArgumentParser
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3

import numpy as np

from run_analysis import pace_comparability as pc
from run_analysis.config import load_config, resolve_project_path
from run_analysis.objective_modeling import _primary_window_weight
from run_analysis.quality_phases import detect_continuous_quality_phase
from run_analysis.read_cache import invariant_snapshot
from scripts.aerobic_joint_profile import GaussianChangeProfile
from scripts.audit_aerobic_shared_history import frame, snapshot, comparisons
from scripts.audit_aerobic_three_layers import inputs, measurement_inputs
from scripts.audit_progress_responsiveness import at, build_from_windows, covariance
from scripts.audit_run_pace_precision import sources, sampling_precision


def inventory(connection, config, windows, report):
    """Same sampling seeds/precision as the existing audit, at held calibration."""
    by_id = {r['activity_id']: r for r in report['runs']}
    grouped = defaultdict(list)
    for row in windows:
        grouped[row['activity_id']].append(row)
    result = []
    for ordinal, (identity, rr) in enumerate(grouped.items()):
        ref = by_id.get(identity)
        if ref is None or not ref['included_in_trend']:
            continue
        phase = detect_continuous_quality_phase(connection, config, identity) if rr[0].get('workout_type') == 'tempo_threshold' else None
        selected = pc._pre_quality_indexes(rr, list(range(len(rr))), phase)
        rr = [rr[i] for i in selected]
        weights = [_primary_window_weight(r, config) for r in rr]
        speeds, parts = sources(rr, weights, config, report['calibration'])
        center, boot = sampling_precision(rr, weights, speeds, 256, 20261014+ordinal)
        assert math.isclose(center, ref['pace_min_mile'], rel_tol=1e-10)
        result.append(dict(activity_id=identity, uncertainty_95_min_mile=ref['uncertainty_95_min_mile'],
            log_variance_parts=parts, sampling_precision=boot))
    return dict(runs=result)


def fitted_geometry(rows, m, g, cov, end, *, shared=True, serial=True):
    indices, observed, groups = frame(rows, end)
    if 0 not in observed or len(indices) <= len(observed):
        raise ValueError('Observed current data and residual df required')
    y = np.log([rows[i]['pace_min_mile'] for i in indices])
    times = np.array([(at(rows[i])-end).total_seconds()/86400 for i in indices])
    used_cov = cov if shared else np.zeros_like(cov)
    model = GaussianChangeProfile(y, m[indices], g[indices], used_cov, groups,
        times=times, future_time=2., mode='ml')
    if not serial:
        model.kernels = {'calendar_half_life_0d': model.kernels['calendar_half_life_0d']}
    fit = model.fit(np.eye(model.p)[0])
    sd = np.array(fit['period_day_log_sd'])[groups]
    shared_v = g[indices]@used_cov@g[indices].T
    day_v = sd[:, None]*model.kernels[fit['kernel']]*sd
    return dict(indices=indices, observed=observed, groups=groups, y=y, x=model.x,
        m=m[indices], shared=shared_v, day=day_v, fit=fit,
        gradients=g[indices], calibration_covariance=used_cov)


def gls(y, x, v):
    inv_x = np.linalg.solve(v, x)
    weights = np.linalg.solve(x.T@inv_x, inv_x.T)
    return weights@y, weights


def cross_period_controls(geometry):
    """Remove only cross-period terms; preserve every marginal covariance."""
    y, x, groups = (geometry[k] for k in ('y', 'x', 'groups'))
    same = groups[:, None] == groups[None, :]
    results = {}
    for shared, serial in ((True, True), (True, False), (False, True), (False, False)):
        v = np.diag(geometry['m'])+geometry['shared']*(1 if shared else same)+geometry['day']*(1 if serial else same)
        beta, weights = gls(y, x, v)
        # A X = I: outside-period weights sum to zero within each group,
        # but their weighted residuals need not cancel.
        np.testing.assert_allclose(weights@x, np.eye(x.shape[1]), atol=1e-10)
        results[f'shared_cross_{shared}_serial_cross_{serial}'] = dict(
            period_log_pace=beta.tolist(), minimum_covariance_eigenvalue=float(np.linalg.eigvalsh(v).min()))
    np.testing.assert_allclose(results['shared_cross_True_serial_cross_True']['period_log_pace'],
        geometry['fit']['beta'], atol=1e-10)
    return results


def calibration_controls(geometry):
    """Separate independent heat uncertainty from the correlated core block.

    All diagonal period blocks, noise amplitudes and serial covariance stay
    fixed. Core coordinates are not individually zeroed inside their correlated
    covariance block, which would change more than the named component.
    """
    y, x, groups, g, c = (geometry[k] for k in ('y', 'x', 'groups', 'gradients', 'calibration_covariance'))
    if c.shape != (4, 4) or not np.allclose(c[3, :3], 0.):
        raise ValueError('Independent heat coordinate and three-parameter core required')
    heat = np.outer(g[:, 3], g[:, 3])*c[3, 3]
    core = g[:, :3]@c[:3, :3]@g[:, :3].T
    np.testing.assert_allclose(heat+core, geometry['shared'], atol=1e-12)
    same = groups[:, None] == groups[None, :]
    result = {}
    for core_cross, heat_cross in ((True, True), (True, False), (False, True), (False, False)):
        v = np.diag(geometry['m'])+geometry['day']+core*(1 if core_cross else same)+heat*(1 if heat_cross else same)
        beta, _ = gls(y, x, v)
        result[f'core_cross_{core_cross}_heat_cross_{heat_cross}'] = beta.tolist()
    full_v = np.diag(geometry['m'])+geometry['day']+geometry['shared']
    beta, _ = gls(y, x, full_v)
    # Conditional random calibration error: marginalizing G C G' and fitting
    # beta implicitly uses this residual-based correction, despite fixed input
    # normalized paces. Reconstruct beta to verify this interpretation.
    error = c@g.T@np.linalg.solve(full_v, y-x@beta)
    reconstructed, _ = gls(y-g@error, x, np.diag(geometry['m'])+geometry['day'])
    np.testing.assert_allclose(reconstructed, beta, atol=1e-10)
    return dict(controls=result, conditional_calibration_error=error.tolist(),
        conditional_error_in_marginal_sd_units=np.divide(error, np.sqrt(np.diag(c)),
            out=np.zeros_like(error), where=np.diag(c) > 0).tolist(),
        conditional_correction_reconstructs_joint_mean=True)


def paired_cause(rows, m, g, cov, end, days):
    prior = end-timedelta(days=days)
    current = fitted_geometry(rows, m, g, cov, end)
    previous = fitted_geometry(rows, m, g, cov, prior)
    period = 1 if days == 28 else 2
    k = current['observed'].index(period)
    current_ids = current['indices'][current['groups'] == k]
    previous_ids = previous['indices'][previous['groups'] == 0]
    np.testing.assert_array_equal(current_ids, previous_ids)
    frozen = {}
    a = cross_period_controls(current); b = cross_period_controls(previous)
    for key in a:
        frozen[key] = dict(reference_disagreement_seconds_per_mile=60*(
            math.exp(a[key]['period_log_pace'][k])-math.exp(b[key]['period_log_pace'][0])),
            minimum_covariance_eigenvalue=min(a[key]['minimum_covariance_eigenvalue'],b[key]['minimum_covariance_eigenvalue']))
    ac = calibration_controls(current); bc = calibration_controls(previous)
    components = {key:60*(math.exp(ac['controls'][key][k])-math.exp(bc['controls'][key][0])) for key in ac['controls']}
    refitted = {}
    for shared, serial in ((True, False), (False, True), (False, False)):
        aa = fitted_geometry(rows, m, g, cov, end, shared=shared, serial=serial)
        bb = fitted_geometry(rows, m, g, cov, prior, shared=shared, serial=serial)
        refitted[f'shared_{shared}_serial_{serial}'] = dict(reference_disagreement_seconds_per_mile=60*(
            math.exp(aa['fit']['beta'][k])-math.exp(bb['fit']['beta'][0])),
            current_kernel=aa['fit']['kernel'], prior_kernel=bb['fit']['kernel'],
            unconverged_kernels=sorted(set(aa['fit']['unconverged_kernels']+bb['fit']['unconverged_kernels'])))
    return dict(anchor=end.isoformat(), days=days, identical_reference_observations=True,
        reference_run_count=len(current_ids), current_kernel=current['fit']['kernel'],
        prior_kernel=previous['fit']['kernel'], frozen_covariance_controls=frozen, refitted_controls=refitted,
        calibration_component_disagreement_seconds_per_mile=components,
        current_conditional_calibration_error=ac['conditional_calibration_error'],
        prior_conditional_calibration_error=bc['conditional_calibration_error'],
        conditional_correction_reconstructs_joint_mean=ac['conditional_correction_reconstructs_joint_mean'] and bc['conditional_correction_reconstructs_joint_mean'],
        unconverged_kernels=sorted(set(current['fit']['unconverged_kernels']+previous['fit']['unconverged_kernels'])))


def overview(rows, m, g, cov, end):
    current = snapshot(rows, m, g, cov, end)
    result = dict(anchor=end.isoformat(), current_pace_min_mile=current['state']['current_pace_min_mile'],
        kernel=current['state']['kernel'], observed_periods=current['observed_periods'], counts=current['counts'], comparisons=[])
    for days in (28, 90):
        previous = snapshot(rows, m, g, cov, end-timedelta(days=days))
        result['comparisons'].append(comparisons(current, previous, days, 0., 0.))
    return result


def audit(database, config, end, old_precision_file, new_precision_file, frozen_file, history_file):
    old_end = datetime.fromisoformat('2026-10-14T22:30:00+00:00')
    old_precision = json.loads(old_precision_file.read_text())
    new_precision = json.loads(new_precision_file.read_text())
    frozen = json.loads(frozen_file.read_text())['statistical_inputs']
    history = json.loads(history_file.read_text())
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro', uri=True) as connection:
        connection.row_factory = sqlite3.Row
        with invariant_snapshot(connection):
            sets, _ = pc.load_model_window_sets(connection, config,
                (int(config['model'].get('window_seconds', 300)),), check_elevation=True)
            windows, _ = pc.load_overlapping_model_windows(connection, config,
                window_seconds=int(config['model']['primary_window_seconds']),
                stride_seconds=int(config['model']['primary_window_stride_seconds']), check_elevation=True)
            def through(anchor):
                return ({k:[r for r in rr if datetime.fromisoformat(r['start_time_utc']) <= anchor] for k,rr in sets.items()},
                    [r for r in windows if datetime.fromisoformat(r['start_time_utc']) <= anchor])
            old_sets, old_windows = through(old_end); new_sets, new_windows = through(end)
            old_report = build_from_windows(connection, config, old_sets, old_windows)
            new_report = build_from_windows(connection, config, new_sets, new_windows)
            held_report = build_from_windows(connection, config, new_sets, new_windows, calibration=old_report['calibration'])
            before = deepcopy((old_report, new_report, held_report))
            selected, fm, fg, groups, old_latest, _, old_rows = inputs(old_report, old_precision, old_end)
            old_cov = covariance(old_report)
            for key, value in dict(actual_log_pace=np.log([r['pace_min_mile'] for r in selected]),
                measurement_variance=fm, gradients=fg, shared_covariance=old_cov, groups=groups,
                times=[(at(r)-old_latest).total_seconds()/86400 for r in selected]).items():
                np.testing.assert_allclose(value, frozen[key], rtol=1e-12, atol=1e-12)
            new_rows = sorted([r for r in new_report['runs'] if r['included_in_trend'] and r['trend_weight'] > 0 and at(r) <= end], key=at)
            held_rows = sorted([r for r in held_report['runs'] if r['included_in_trend'] and r['trend_weight'] > 0 and at(r) <= end], key=at)
            assert [r['activity_id'] for r in new_rows] == [r['activity_id'] for r in held_rows]
            new_m, new_g, _ = measurement_inputs(new_report, new_precision, new_rows)
            old_m, old_g, _ = measurement_inputs(old_report, old_precision, old_rows)
            held_m, held_g, _ = measurement_inputs(held_report, inventory(connection, config, new_windows, held_report), held_rows)
            new_cov = covariance(new_report); latest = at(new_rows[-1])
            stages = dict(original=overview(old_rows, old_m, old_g, old_cov, old_latest),
                recalibration_only=overview(new_rows, new_m, new_g, new_cov, old_latest),
                clock_only=overview(old_rows, old_m, old_g, old_cov, latest),
                uploaded_run_fixed_old_anchor=overview(held_rows, held_m, held_g, old_cov, old_latest),
                clock_and_uploaded_run=overview(held_rows, held_m, held_g, old_cov, latest),
                current=overview(new_rows, new_m, new_g, new_cov, latest))
            # The later run cannot enter a fixed earlier anchor; held
            # calibration and identical inputs must reproduce it exactly.
            assert stages['original'] == stages['uploaded_run_fixed_old_anchor']
            cases = []
            for days in (28, 90):
                cases.append(dict(label='latest_with_upload', **paired_cause(new_rows, new_m, new_g, new_cov, latest, days)))
                worst = max((v for v in history['comparisons'] if v['days'] == days and v['available']),
                    key=lambda v:abs(v['change_disagreement_percentage_points']))
                cases.append(dict(label='original_worst_historical', **paired_cause(old_rows, old_m, old_g, old_cov,
                    datetime.fromisoformat(worst['anchor']), days)))
            old_by_id = {r['activity_id']:r for r in old_rows}
            adjustment_changes = [60*(r['pace_min_mile']-old_by_id[r['activity_id']]['pace_min_mile'])
                for r in new_rows if r['activity_id'] in old_by_id]
            assert (old_report, new_report, held_report) == before and connection.total_changes == 0
    root = Path(__file__).resolve().parents[1]
    return dict(version='aerobic-history-cause-v1', deployed=False, source_database_read_only=True,
        as_of=end.isoformat(), old_frozen_cohort_matched=True,
        latest_run={k:new_rows[-1][k] for k in ('activity_id', 'start_time', 'distance_miles', 'pace_min_mile', 'sampled_hr_bpm', 'workout_type', 'trend_weight')},
        eligible_before=len(old_rows), eligible_after=len(new_rows), stages=stages, cause_cases=cases,
        normalization_revision_seconds_per_mile=dict(max_absolute=float(max(abs(v) for v in adjustment_changes)),
            median_absolute=float(np.median(np.abs(adjustment_changes)))),
        source_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in (
            'scripts/audit_aerobic_history_cause.py', 'scripts/audit_aerobic_shared_history.py',
            'scripts/aerobic_joint_profile.py', 'scripts/aerobic_shared_state.py', 'src/run_analysis/pace_comparability.py',
            'scripts/audit_run_pace_precision.py', 'scripts/audit_aerobic_three_layers.py')},
        input_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in (
            old_precision_file, new_precision_file, frozen_file, history_file)},
        assumptions=[
            'Diagnostic controls are not replacement estimators, chosen weights or release recommendations.',
            'Full covariance controls preserve all within-period covariance and remove cross-period shared/serial terms separately; every control remains PSD.',
            'Refitted controls relearn nuisance variances and, when enabled, search the unchanged kernel grid; removal of shared covariance retains the same measured independent variance.',
            'New run, calendar rollover and normalization refitting are separated with fixed-anchor and held-calibration controls.',
            'Held calibration uses existing block precision arithmetic and seeds, not stale current-calibration precision.',
            'Worst historical cases are selected by previously recorded discrepancy magnitude, never inference outcomes.',
            'No coverage or physiological-accuracy claim; no source, live UI, model, adjustment coefficient or planner changes.'])


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--as-of', type=datetime.fromisoformat, default=datetime.now(timezone.utc))
    args = parser.parse_args()
    if args.output.exists() or args.as_of.tzinfo is None:
        parser.error('New output and timezone-aware date required')
    root = Path(__file__).resolve().parents[1]; config = load_config(root/'config.yaml')
    result = audit(resolve_project_path(root, config['paths']['database']), config, args.as_of,
        root/'output/run-pace-precision-coverage-20261014.json',
        root/'output/run-pace-precision-current-20261015.json',
        root/'output/aerobic-joint-bootstrap-validation-20261014.json',
        root/'output/aerobic-shared-history-alignment-20261015.json')
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('latest_run', 'eligible_before', 'eligible_after', 'stages',
        'normalization_revision_seconds_per_mile', 'cause_cases')}, indent=2))


if __name__ == '__main__':
    main()
