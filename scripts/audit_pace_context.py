"""Bounded conditional weather/terrain test. No cached fit or production writes.

Predict held-out steady windows conditional on recorded HR/position/context.
This tests incremental explanatory prediction, NOT pre-run forecast accuracy.
All coefficients use earlier runs only; every run receives equal fit weight.
"""
from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timedelta
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import sqlite3
from time import perf_counter

import numpy as np

from run_analysis.config import load_config
from run_analysis.model_windows import load_model_window_sets
from run_analysis.modeling import FeatureBuilder, fit_ridge
from scripts.audit_workload_expectation import app_session_types


MODELS = ('B', 'C', 'D', 'D+grade')


def chronological_cases(rows, config, *, lookback_days=84, minimum_prior_runs=6):
    """Fixed settings, not a search over the best window/penalty/features."""
    grouped = {}
    for row in rows:
        grouped.setdefault(row['activity_id'], []).append(row)
    ordered = sorted(grouped, key=lambda key: (grouped[key][0]['start_time_utc'], key))
    cases, skipped = [], []
    for key in ordered:
        heldout = grouped[key]
        at = datetime.fromisoformat(heldout[0]['start_time_utc'])
        prior_keys = [k for k in ordered if k != key
            and at-timedelta(days=lookback_days) <= datetime.fromisoformat(grouped[k][0]['start_time_utc']) < at
            and max(datetime.fromisoformat(r['activity_end_time_utc']) for r in grouped[k]) <= at]
        if len(prior_keys) < minimum_prior_runs:
            skipped.append(key)
            continue
        train = [r for k in prior_keys for r in grouped[k]]
        counts = Counter(r['activity_id'] for r in train)
        weights = np.array([1/counts[r['activity_id']] for r in train])
        results = {}
        for name in MODELS:
            builder = FeatureBuilder(name, config).fit(train)
            model = fit_ridge(builder.transform(train)*np.sqrt(weights)[:, None],
                              builder.response(train)*np.sqrt(weights), config['model']['ridge_alpha'])
            predicted = builder.predicted_raw_paces(model, heldout)
            actual = np.array([r['moving_pace_min_mile'] for r in heldout])
            if not np.isfinite(predicted).all():
                raise ValueError('Nonpositive predicted speed; do not drop a losing forecast')
            error = (predicted-actual)*60
            results[name] = dict(mae_seconds_per_mile=float(np.mean(np.abs(error))),
                bias_seconds_per_mile=float(np.mean(error)),
                mean_predicted_pace_min_mile=float(np.mean(predicted)),
                mean_absolute_percent_error=float(np.mean(np.abs(predicted/actual-1))*100))
        cases.append(dict(key=key, start=at.isoformat(), prior_keys=prior_keys,
                          heldout_windows=len(heldout), results=results))
    return cases, skipped


def summarize(cases, threshold):
    if len(cases) < 2:
        return dict(status='insufficient_heldout_runs', cases=len(cases))
    parts = {'all': cases, 'early': cases[:len(cases)//2], 'late': cases[len(cases)//2:]}
    summary = {}
    for name in MODELS:
        groups = {}
        for group, rows in parts.items():
            errors = [r['results'][name]['mae_seconds_per_mile'] for r in rows]
            gains = [r['results']['B']['mae_seconds_per_mile']-r['results'][name]['mae_seconds_per_mile'] for r in rows]
            groups[group] = dict(runs=len(rows), mean_run_mae_seconds_per_mile=float(np.mean(errors)),
                median_run_mae_seconds_per_mile=float(np.median(errors)),
                gain_over_baseline_seconds_per_mile=float(np.mean(gains)),
                runs_better_than_baseline=sum(g > 0 for g in gains))
        summary[name] = dict(groups=groups, consistent_incremental_gain=(name != 'B' and
            all(groups[group]['gain_over_baseline_seconds_per_mile'] >= threshold for group in ('early', 'late'))))
    return dict(status='conditional_context_comparison', models=summary,
                improvement_requirement_each_half_seconds_per_mile=threshold)


def audit(database, config):
    started = perf_counter()
    digest = sha256(database.read_bytes()).hexdigest()
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro', uri=True) as connection:
        connection.row_factory = sqlite3.Row
        sets, diagnostics = load_model_window_sets(connection, config, (120,))
        ends = {r['id']: (datetime.fromisoformat(r['start_time_utc'])+
            timedelta(seconds=r['total_elapsed_time_s'])).isoformat() for r in connection.execute(
                'SELECT id,start_time_utc,total_elapsed_time_s FROM activities WHERE total_elapsed_time_s>0')}
    original = sets[120]
    roles = app_session_types(database)
    rows, exclusions = [], Counter()
    for row in original:
        # Match the preceding original-file audits: cached detected_workout_type
        # is often 'unknown' while app-resolved history identifies easy/long.
        if roles[str(row['activity_id'])] not in {'easy', 'recovery', 'medium_long', 'long'}:
            exclusions['not_resolved_endurance_context'] += 1
            continue
        if not row['gps_sufficient_for_shared_parameters'] or row['weather_location_estimated']:
            exclusions['incomparable_or_estimated_location'] += 1
            continue
        if (row['grade_complete_fraction'] < .95 or row['grade_energy_ratio'] is None
                or row['activity_id'] not in ends or any(row.get(k) is None or not isfinite(row[k])
                    for k in ('temperature_f', 'dewpoint_f', 'moving_pace_min_mile'))):
            exclusions['incomplete_common_context'] += 1
            continue
        rows.append(dict(row, workout_type=roles[str(row['activity_id'])],
                         activity_end_time_utc=ends[row['activity_id']]))
    cases, skipped = chronological_cases(rows, config)
    if sha256(database.read_bytes()).hexdigest() != digest:
        raise AssertionError('Source database changed during context audit')
    return dict(version='conditional_pace_context_v1', elapsed_seconds=perf_counter()-started,
        database_unchanged=True, planner_influence=False, parameters=dict(window_seconds=120,
            lookback_days=84, minimum_prior_runs_for_diagnostic=6, ridge_alpha=config['model']['ridge_alpha'],
            models=MODELS, minimum_grade_coverage=.95, fit_weight='equal total weight per run'),
        eligible_windows=len(rows), eligible_runs=len({r['activity_id'] for r in rows}),
        excluded_windows=dict(exclusions), loader_diagnostics=diagnostics[120],
        skipped_insufficient_prior_runs=skipped, cases=cases,
        summary=summarize(cases, config['model']['minimum_cv_improvement_seconds_per_mile']),
        limitations=['Conditional on held-out recorded HR, position, weather and route grade; not a pre-run forecast',
            'Historical metadata/role resolution frozen at audit time, not a full app replay',
            'No fitted-cache coefficients or future-run responses used in training',
            'Point-prediction error comparison does not calibrate uncertainty intervals',
            'Six-run minimum is diagnostic sample sufficiency, not a planner eligibility gate',
            'No feature/regularization tuning after looking at results; no production promotion'])


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = audit(Path('data/run_analysis.sqlite'), load_config(Path('config.yaml')))
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({k: result[k] for k in ('elapsed_seconds', 'eligible_runs', 'eligible_windows', 'summary')}))


if __name__ == '__main__':
    main()
