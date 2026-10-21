"""Read-only humidity addition test; no live correction or activity changes."""
from argparse import ArgumentParser
from datetime import datetime, timedelta
import json
import math
import os
from pathlib import Path
import sqlite3

import numpy as np

from run_analysis.config import load_config, resolve_project_path
from run_analysis.read_cache import invariant_snapshot
from scripts.audit_thermal_hr_trial import collect, paired_difference
from scripts.thermal_hr_trial import forward
from scripts.humidity_hr_trial import window_features, fit_joint, forward_joint, summarize


def audit(rows):
    eligible = [r for r in rows if r['available'] and r['humidity_available']]
    models = dict(current=forward(eligible), none=forward(eligible, model='none'),
        wbgt=forward(eligible, model='personal'), air=forward(eligible, model='air'),
        separated=forward_joint(eligible))
    # Diagnostics and grouping use identical physical runs, never extra votes per window.
    lookup = {r['activity_id']: r for r in eligible}
    for value in models.values():
        for case in value['cases']:
            case['humidity_percent'] = lookup[case['activity_id']]['humidity_percent']
        value['groups'] = {key: summarize([c for c in value['cases'] if condition(c)]) for key, condition in (
            ('cool_below_55f', lambda c: c['temperature_f'] < 55),
            ('warm_at_least_70f', lambda c: c['temperature_f'] >= 70),
            ('humid_at_least_70rh', lambda c: c['humidity_percent'] >= 70),
            ('drier_below_70rh', lambda c: c['humidity_percent'] < 70),
            ('warm_and_humid', lambda c: c['temperature_f'] >= 70 and c['humidity_percent'] >= 70),
            ('study_temperature_at_least_87_8f', lambda c: c['temperature_f'] >= 87.8))}
    fitted = fit_joint(eligible)
    sensitivity = []
    for period, sd in ((28., 1.), (84., 1.), (56., .5), (56., 2.)):
        candidate = forward_joint(eligible, period=period, prior_sd=sd)
        control = forward_joint(eligible, period=period, humidity=False)
        sensitivity.append(dict(calendar_days=period, humidity_prior_sd=sd,
            fit=fit_joint(eligible, period=period, prior_sd=sd),
            summary=candidate['summary'], paired_vs_air=paired_difference(control, candidate)))
    loo = [dict(activity_id=r['activity_id'], scales=fit_joint([v for v in eligible if v is not r])['scales']) for r in eligible]
    planted = []; latest = max(datetime.fromisoformat(r['start_time_utc']) for r in eligible)
    for recent_only in (False, True):
        def affected(r):
            return not recent_only or datetime.fromisoformat(r['start_time_utc']) >= latest-timedelta(days=28)
        changed = [dict(r, pace_no_weather=r['pace_no_weather']/(1.02 if affected(r) else 1.)) for r in eligible]
        after = fit_joint(changed)
        recovered = [100*(1.02*math.exp(sum((a-b)*r[k] for k, a, b in zip(fitted['keys'], after['scales'], fitted['scales'])))-1)
            for r in eligible if affected(r)]
        planted.append(dict(recent_only=recent_only, recovered_mean_percent=float(np.mean(recovered)),
            minimum_percent=min(recovered), maximum_percent=max(recovered), scales=after['scales']))
    return dict(version='humidity-hr-trial-v1', live_changed=False, source_read_only=True,
        original_runs=len(rows), weather_supported_runs=len(eligible),
        missing_context_originals_retained=[r['activity_id'] for r in rows if r not in eligible],
        models=models, fit=fitted, sensitivity=sensitivity, leave_one_out=loo, planted=planted,
        paired_vs_current={k: paired_difference(models['current'], v) for k, v in models.items()},
        paired_vs_air=paired_difference(models['air'], models['separated']),
        quality=dict(humidity_min=min(r['humidity_percent'] for r in eligible), humidity_max=max(r['humidity_percent'] for r in eligible),
            runs_at_least_87_8f=sum(r['temperature_f'] >= 87.8 for r in eligible),
            original_windows=sum(r['original_windows'] for r in rows), retained_windows=sum(r['windows_retained'] for r in rows),
            max_rh_dewpoint_disagreement_points=max(r['rh_dewpoint_max_disagreement_points'] or 0 for r in eligible),
            median_rh_dewpoint_disagreement_points=float(np.median([r['rh_dewpoint_max_disagreement_points'] for r in eligible
                if r['rh_dewpoint_max_disagreement_points'] is not None])),
            dewpoint_above_air_windows=sum(r['dewpoint_above_air_windows'] for r in eligible)),
        observations=rows,
        assumptions=[
            'Every original run and selected window retained. Four missing-weather runs are not used as known-neutral weather examples.',
            'Weather from existing hourly cache interpolated at each scored running window, not daily weather summaries.',
            'Humidity increment is shade WBGT at actual RH minus shade WBGT at fixed reference RH, at identical air temperature and elapsed time.',
            'Air HR shift plus humidity increment is converted through existing HR normalization with legacy heat disabled.',
            'Joint log-pace burdens are additive approximations; temperature prior 1 +/- 1, humidity prior 0 +/- 1 in transferred WBGT units.',
            'Humidity scale is not sourced from a controlled humidity dose study. Null and negative fits allowed, not forced into desired physiological direction.',
            'Earlier-run-only weather fits, frozen retrospective HR timing and other calibration. One athlete and conditional prediction, not causal fitness validation.',
            'Local calendar controls and nuisance variables same as previous thermal trial. Preset sensitivities retained regardless of outcome.',
            'Reference RH held constant isolates humidity from temperature; RH is not interpreted as independent of temperature without joint adjustment.'
        ])


def main():
    p = ArgumentParser(description=__doc__); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--reuse-observations', type=Path); args = p.parse_args()
    if args.output.exists():
        p.error('Use a new output path')
    if args.reuse_observations:
        rows = json.loads(args.reuse_observations.read_text())['observations']
    else:
        config = load_config('config.yaml'); database = resolve_project_path(Path.cwd(), config['paths']['database'])
        with sqlite3.connect(database.resolve().as_uri()+'?mode=ro', uri=True) as db:
            db.row_factory = sqlite3.Row
            with invariant_snapshot(db):
                rows, _ = collect(db, config, window_diagnostic=window_features)
                assert db.total_changes == 0
    result = audit(rows)
    fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as out:
        json.dump(result, out, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in result.items() if k not in ('observations', 'leave_one_out', 'sensitivity', 'models')}, indent=2))
    print(json.dumps({k: {a: b for a, b in v.items() if a in ('summary', 'groups')} for k, v in result['models'].items()}, indent=2))
    print(json.dumps([dict(calendar_days=r['calendar_days'], humidity_prior_sd=r['humidity_prior_sd'], scales=r['fit']['scales'],
        summary=r['summary'], paired=r['paired_vs_air']) for r in result['sensitivity']], indent=2))


if __name__ == '__main__':
    main()
