"""Offline separation of air temperature from same-temperature humidity.

Humidity has a zero-centered engineering prior, not a published dose estimate.
The scale unit is the previous shade-WBGT transfer, evaluated at constant air
temperature. No production imports this module.
"""
import math

import numpy as np

from run_analysis.pace_comparability import compare_run_windows
from run_analysis.physiology import relative_humidity_from_dewpoint
from scripts.thermal_hr_trial import NUISANCE, design, day, hr_shift, thermal_windows


def window_features(rows, weights, config, neutral):
    ref = config['reference_conditions']
    ref_rh = relative_humidity_from_dewpoint(ref['temperature_f'], ref['dewpoint_f'])
    air, _ = thermal_windows(rows, config, exposure='air')
    air_pace = compare_run_windows(air, weights, config, neutral)['pace_min_mile']
    shifted = []; conditions = []; complete = True
    for row, baseline in zip(rows, air):
        out = dict(baseline)
        try:
            temperature = float(row['temperature_f'])
            if not math.isfinite(temperature):
                raise ValueError('Invalid temperature')
            dew = row.get('dewpoint_f'); rh = row.get('relative_humidity_percent')
            derived = relative_humidity_from_dewpoint(temperature, dew) if dew is not None else None
            if rh is None:
                rh = derived
            if rh is None or not math.isfinite(rh) or not 0 <= rh <= 100:
                raise ValueError('Invalid humidity')
            observed = dict(row, relative_humidity_percent=rh)
            constant = dict(row, relative_humidity_percent=ref_rh)
            # Reference humidity gives zero increment at EVERY air temperature.
            delta = hr_shift(observed, config)-hr_shift(constant, config)
            out['average_hr_bpm'] -= delta
            if not math.isfinite(out['average_hr_bpm']) or out['average_hr_bpm'] <= config['resting_hr']:
                raise ValueError('Unsupported HR mapping')
            conditions.append(dict(temperature_f=temperature, dewpoint_f=dew,
                humidity_percent=rh, derived_humidity_percent=derived,
                humidity_hr_increment_bpm=delta))
        except (TypeError, ValueError, KeyError):
            complete = False
        shifted.append(out)
    result = dict(humidity_available=complete, humidity_windows=len(conditions),
        humidity_reference_percent=ref_rh, humidity_burden=None)
    if complete:
        pace = compare_run_windows(shifted, weights, config, neutral)['pace_min_mile']
        result['humidity_burden'] = math.log(air_pace/pace)
        for key in ('humidity_percent', 'dewpoint_f', 'humidity_hr_increment_bpm'):
            values = [r[key] for r in conditions]
            result[key] = float(np.average(values, weights=weights)) if all(v is not None for v in values) else None
        diffs = [r['humidity_percent']-r['derived_humidity_percent'] for r in conditions if r['derived_humidity_percent'] is not None]
        result['rh_dewpoint_max_disagreement_points'] = max(map(abs, diffs)) if diffs else None
        result['dewpoint_above_air_windows'] = sum(r['dewpoint_f'] is not None and r['dewpoint_f'] > r['temperature_f']+.2 for r in conditions)
    return result


def fit_joint(rows, *, prior_sd=1., period=56., humidity=True):
    if not rows or len({r['activity_id'] for r in rows}) != len(rows):
        raise ValueError('Unique nonempty runs required')
    if prior_sd <= 0 or period <= 0:
        raise ValueError('Positive uncertainty and period required')
    x, spec = design(rows, NUISANCE)
    t = np.array([day(r) for r in rows]); blocks = np.floor(t/period)
    z = np.column_stack([col for k in np.unique(blocks) for col in (
        (blocks == k).astype(float), np.where(blocks == k, (t-t[blocks == k].mean())/period, 0.))])
    def residual(a):
        return a-z@np.linalg.lstsq(z, a, rcond=None)[0]
    keys = ['air_burden']+(['humidity_burden'] if humidity else [])
    b = np.array([[r[k] for k in keys] for r in rows])
    prior = np.array([1.]+([0.] if humidity else []))
    rx = residual(x); rb = residual(b); ry = residual(np.log([r['pace_no_weather'] for r in rows])-b@prior)
    ridge = 4.*np.eye(x.shape[1])
    beta = np.linalg.solve(rx.T@rx+ridge, rx.T@ry)
    sigma = max(1e-6, float(np.sqrt(np.mean((ry-rx@beta)**2))))
    a = np.column_stack((rx, rb)); p = np.zeros((a.shape[1], a.shape[1]))
    p[:x.shape[1], :x.shape[1]] = ridge
    # Same temperature prior and ridge as the previous air-only trial.
    p[x.shape[1]:, x.shape[1]:] = np.diag([sigma**2]+([sigma**2/prior_sd**2] if humidity else []))
    matrix = a.T@a+p; coef = np.linalg.solve(matrix, a.T@ry)
    covariance = np.linalg.inv(matrix)[x.shape[1]:, x.shape[1]:]*sigma**2
    return dict(scales=(prior+coef[x.shape[1]:]).tolist(), keys=keys,
        conditional_scale_sd=np.sqrt(np.maximum(0, np.diag(covariance))).tolist(),
        scale_correlation=float(covariance[0, 1]/math.sqrt(covariance[0, 0]*covariance[1, 1])) if humidity else None,
        nuisance_beta=coef[:x.shape[1]].tolist(), nuisance_spec=spec,
        residual_burden_sd=np.std(rb, axis=0).tolist(), sigma=sigma,
        source_ids=[r['activity_id'] for r in rows], physiological_uncertainty_calibrated=False)


def predict(fit, anchor, target):
    x, _ = design([anchor, target], NUISANCE, fit['nuisance_spec'])
    burden = sum(scale*(target[key]-anchor[key]) for key, scale in zip(fit['keys'], fit['scales']))
    return anchor['pace_no_weather']*math.exp(burden+float((x[1]-x[0])@fit['nuisance_beta']))


def forward_joint(rows, **kwargs):
    ordered = sorted(rows, key=day); cases = []
    for i, target in enumerate(ordered):
        train = ordered[:i]
        if i < 24 or day(train[-1])-day(train[0]) < 56 or day(target)-day(train[-1]) > 14:
            continue
        fitted = fit_joint(train, **kwargs); value = predict(fitted, train[-1], target)
        cases.append(dict(activity_id=target['activity_id'], date=target['start_time_utc'],
            prediction=value, actual=target['pace_no_weather'], error_seconds=60*(value-target['pace_no_weather']),
            temperature_f=target['temperature_f'], humidity_percent=target['humidity_percent'],
            scales=fitted['scales'], prior_runs=i))
    return dict(summary=summarize(cases), cases=cases)


def summarize(cases):
    return dict(cases=len(cases), mae_seconds=float(np.mean([abs(r['error_seconds']) for r in cases])) if cases else None,
        rmse_seconds=float(np.sqrt(np.mean([r['error_seconds']**2 for r in cases]))) if cases else None)
