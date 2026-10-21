"""Read-only historical/reference alignment check for the offline shared point.

Normalization is frozen. This is not a prospective calibration replay or a
confidence-coverage experiment, and it never changes the production graph.
"""
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
from scripts.aerobic_serial_model import kernel_grid
from scripts.aerobic_shared_state import fit_shared_state
from scripts.audit_aerobic_three_layers import inputs, measurement_inputs
from scripts.audit_progress_responsiveness import at, build_from_windows, covariance
from scripts.audit_small_aerobic_changes import describe


def frame(rows, end):
    """Exact calendar anchor, not snapped to the last run in that frame."""
    indices = [i for i, row in enumerate(rows) if end-timedelta(days=118) < at(row) <= end]
    labels = []
    for i in indices:
        days = (end-at(rows[i])).total_seconds()/86400
        labels.append(0 if days < 28 else 1 if days < 56 else 2 if days >= 90 else 3)
    observed = sorted(set(labels))
    groups = np.array([observed.index(v) for v in labels], dtype=int)
    return np.array(indices, dtype=int), observed, groups


def snapshot(rows, m, g, cov, end):
    indices, observed, groups = frame(rows, end)
    if 0 not in observed or len(indices) <= len(observed):
        return dict(anchor=end.isoformat(), available=False, observed_periods=observed,
                    reason='Current observations and positive residual degrees of freedom required')
    y = np.log([rows[i]['pace_min_mile'] for i in indices])
    times = np.array([(at(rows[i])-end).total_seconds()/86400 for i in indices])
    state = fit_shared_state(y, m[indices], g[indices], cov, groups, times=times)
    # Attribution only: marginal GLS within each period at THIS fit's fixed
    # covariance. Difference from joint beta measures cross-period borrowing.
    sd = np.asarray(state['period_day_log_sd'])[groups]
    r = kernel_grid(times, 2.)[state['kernel']][:-1, :-1]
    v = np.diag(m[indices])+g[indices]@cov@g[indices].T+sd[:, None]*r*sd
    local = []
    for k in range(len(observed)):
        mask = groups == k
        weights = np.linalg.solve(v[np.ix_(mask, mask)], np.ones(int(mask.sum())))
        local.append(float(weights@y[mask]/weights.sum()))
    return dict(anchor=end.isoformat(), available=True, observed_periods=observed,
        counts=[int(sum(groups == k)) for k in range(len(observed))],
        state=state, marginal_period_log_pace=local)


def crop(curve, end, days):
    """Presentation only; one predecessor supports the visible left edge."""
    cutoff = end-timedelta(days=days)
    before = [v for v in curve if datetime.fromisoformat(v['anchor']) < cutoff]
    return before[-1:]+[v for v in curve if cutoff <= datetime.fromisoformat(v['anchor']) <= end]


def comparisons(current, previous, days, robust_current, robust_previous):
    period = 1 if days == 28 else 2
    if not previous['available'] or period not in current['observed_periods']:
        return dict(days=days, available=False)
    k = current['observed_periods'].index(period)
    now = current['state']['period_log_pace'][0]
    joint_prior = current['state']['period_log_pace'][k]
    curve_prior = previous['state']['period_log_pace'][0]
    joint_local = current['marginal_period_log_pace'][k]
    curve_local = previous['marginal_period_log_pace'][0]
    joint_change = 100*math.expm1(joint_prior-now)
    curve_change = 100*math.expm1(curve_prior-now)
    terms = dict(current_frame_borrowing=joint_prior-joint_local,
        marginal_covariance_refit=joint_local-curve_local,
        previous_frame_borrowing=curve_local-curve_prior)
    assert math.isclose(sum(terms.values()), joint_prior-curve_prior, abs_tol=1e-12)
    return dict(days=days, available=True, prior_anchor=previous['anchor'],
        joint_prior_pace_min_mile=math.exp(joint_prior), curve_prior_pace_min_mile=math.exp(curve_prior),
        prior_disagreement_seconds_per_mile=60*(math.exp(joint_prior)-math.exp(curve_prior)),
        joint_change_percent=joint_change, curve_endpoint_change_percent=curve_change,
        change_disagreement_percentage_points=joint_change-curve_change,
        log_pace_attribution=terms,
        robust_control_disagreement_seconds_per_mile=60*(robust_current-robust_previous))


def audit(database, config, end, precision_file, frozen_file):
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
            selected, fm, fg, groups, latest, _, rows = inputs(report, precision, end)
            cov = covariance(report)
            for name, value in dict(actual_log_pace=np.log([r['pace_min_mile'] for r in selected]),
                measurement_variance=fm, gradients=fg, shared_covariance=cov, groups=groups,
                times=[(at(r)-latest).total_seconds()/86400 for r in selected]).items():
                np.testing.assert_allclose(value, frozen[name], rtol=1e-12, atol=1e-12)
            m, g, sources = measurement_inputs(report, precision, rows)
            cached = {}
            def get(anchor):
                if anchor not in cached:
                    cached[anchor] = snapshot(rows, m, g, cov, anchor)
                return cached[anchor]
            # Every observed date in the recent 90-day curve, not selected by
            # outcomes. Exact earlier anchors are evaluated even without a run.
            anchors = [at(r) for r in rows if at(r) >= latest-timedelta(days=90)]
            curve = [get(anchor) for anchor in anchors]
            checks = []
            for current in curve:
                anchor = datetime.fromisoformat(current['anchor'])
                if not current['available']:
                    continue
                for days in (28, 90):
                    prior_anchor = anchor-timedelta(days=days)
                    previous = get(prior_anchor)
                    # Production _period is local: a later evidence frame may
                    # contain extra rows, but not extra data INSIDE this period.
                    frame_indices, _, _ = frame(rows, anchor)
                    local_rows = [rows[i] for i in frame_indices]
                    a = ap._period(local_rows, prior_anchor, cov)
                    b = ap._period(rows, prior_anchor, cov)
                    if a is None or b is None:
                        checks.append(dict(anchor=anchor.isoformat(), days=days, available=False))
                        continue
                    checks.append(dict(anchor=anchor.isoformat(), **comparisons(current, previous, days,
                        a['pace_min_mile'], b['pace_min_mile'])))
            for days in (28, 90, 180):
                visible = crop(curve, latest, days)
                assert all(v == cached[datetime.fromisoformat(v['anchor'])] for v in visible)
            # Adding the last run must not revise an earlier independently
            # anchored point when pace normalization itself is frozen.
            preceding = anchors[-2]
            without_latest = snapshot(rows[:-1], m[:-1], g[:-1], cov, preceding)
            assert without_latest == get(preceding)
            assert report == before and connection.total_changes == 0
    summary = {}
    for days in (28, 90):
        available = [v for v in checks if v['days'] == days and v['available']]
        summary[str(days)] = dict(available=len(available), unavailable=sum(
            v['days'] == days and not v['available'] for v in checks),
            absolute_prior_disagreement_seconds_per_mile=describe([abs(v['prior_disagreement_seconds_per_mile']) for v in available]),
            absolute_change_disagreement_percentage_points=describe([abs(v['change_disagreement_percentage_points']) for v in available]),
            robust_max_disagreement_seconds_per_mile=max((abs(v['robust_control_disagreement_seconds_per_mile']) for v in available), default=None))
    root = Path(__file__).resolve().parents[1]
    return dict(version='aerobic-shared-history-alignment-v1', deployed=False, source_database_read_only=True,
        frozen_cohort_matched=True, as_of=end.isoformat(), latest_anchor=latest.isoformat(), eligible_runs=len(rows),
        recent_curve_points=len(curve), fitted_snapshots=len(cached), viewport_invariant=True,
        earlier_snapshot_unchanged_by_new_run=True,
        latest_comparisons=[v for v in checks if v['anchor'] == latest.isoformat()],
        summary=summary, comparisons=checks, curve=curve,
        optimizer_warning_snapshots=sum(bool(v.get('state', {}).get('unconverged_kernels')) for v in cached.values()),
        source_counts={name:sources.count(name) for name in set(sources)},
        source_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in (
            'scripts/audit_aerobic_shared_history.py', 'scripts/audit_aerobic_three_layers.py',
            'scripts/aerobic_shared_state.py', 'scripts/aerobic_joint_profile.py', 'src/run_analysis/aerobic_progress.py')},
        input_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in (precision_file, frozen_file)},
        assumptions=[
            'Same frozen normalized paces, run precision and shared calibration at every anchor; not an as-known-then normalization replay.',
            'Each point independently fits its own trailing 118-day evidence frame with the existing full ML estimator and kernel grid.',
            'Missing historical periods are omitted, never fabricated; current data and positive residual df are required.',
            'Historical comparison dates are exact calendar anchors, not silently snapped to the nearest activity.',
            'Marginal GLS decomposition holds each fit covariance fixed for attribution; it is not a replacement estimator.',
            'Descriptive dependent historical checks, not a coverage experiment or new physiological validation.',
            'No live graph, headline, original VO2, adjustments, source activity evidence or scheduler changes.'])


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--precision', type=Path, default=Path('output/run-pace-precision-coverage-20261014.json'))
    parser.add_argument('--frozen', type=Path, default=Path('output/aerobic-joint-bootstrap-validation-20261014.json'))
    parser.add_argument('--as-of', type=datetime.fromisoformat, default=datetime.fromisoformat('2026-10-14T22:30:00+00:00'))
    args = parser.parse_args()
    if args.output.exists() or args.as_of.tzinfo is None:
        parser.error('New output and timezone-aware as-of required')
    root = Path(__file__).resolve().parents[1]; config = load_config(root/'config.yaml')
    result = audit(resolve_project_path(root, config['paths']['database']), config,
        args.as_of, args.precision, args.frozen)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('latest_comparisons', 'summary', 'recent_curve_points',
        'fitted_snapshots', 'optimizer_warning_snapshots', 'viewport_invariant',
        'earlier_snapshot_unchanged_by_new_run')}, indent=2))


if __name__ == '__main__':
    main()
