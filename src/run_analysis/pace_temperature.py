"""Signed, duration-dependent temperature HR comparison, not training load.

Gagnon et al. 2013 (PMC3650516) running group means at 0/22 C and
40% RH inform a weak transferred shape. The personal scale is an association,
not physiological calibration. Humidity is deliberately not fitted here.
"""
from functools import lru_cache
import math

import numpy as np

VERSION = 'signed-air-hr-v1'
NUISANCE = ('sampled_hr_bpm', 'sampled_minutes', 'duration_minutes', 'device', 'area',
            'hour_sin', 'hour_cos', 'stop_fraction', 'previous_7d_miles', 'previous_28d_miles')


@lru_cache(maxsize=1)
def study_shape():
    minutes = np.array([15., 30., 45., 60.]); difference = np.array([7., 11., 14., 15.])
    candidates = []
    for tau in np.linspace(1., 120., 1191):
        f = -np.expm1(-minutes/tau); amplitude = float(f@difference/(f@f))
        candidates.append((float(np.sum((amplitude*f-difference)**2)), float(tau), amplitude))
    _, tau, amplitude = min(candidates)
    return dict(tau_minutes=tau, amplitude_bpm=amplitude,
        source='https://pmc.ncbi.nlm.nih.gov/articles/PMC3650516/',
        transfer='Fitted running group means; linear air-temperature transfer, not a published universal correction')


def hr_shift(row, config):
    """Positive thermal HR contribution in warmth, negative in cool conditions."""
    temperature, minute = row.get('temperature_f'), row.get('moving_minutes_into_run')
    if any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) for v in (temperature, minute)) or minute < 0:
        return None
    shape = study_shape()
    return shape['amplitude_bpm']*(-math.expm1(-minute/shape['tau_minutes']))*(
        temperature-config['reference_conditions']['temperature_f'])*5/9/22.


def design(rows, keys, spec=None):
    """Training-derived numeric scaling and categories; no target-fit leakage."""
    from collections import Counter
    def finite(v):
        return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
    if spec is None:
        spec = []
        for key in keys:
            values = [r.get(key) for r in rows]
            if key in {'device', 'area'}:
                levels = Counter(str(v or 'missing') for v in values); reference = levels.most_common(1)[0][0]
                spec.extend(dict(key=key, kind='category', level=level) for level in sorted(levels) if level != reference)
            else:
                known = [v for v in values if finite(v)]
                if len(known) < 2 or np.std(known) < 1e-9:
                    continue
                spec.append(dict(key=key, kind='numeric', center=float(np.mean(known)), scale=float(np.std(known))))
                if len(known) != len(rows):
                    spec.append(dict(key=key, kind='missing'))
    columns = []
    for f in spec:
        if f['kind'] == 'numeric':
            columns.append([(r[f['key']]-f['center'])/f['scale'] if finite(r.get(f['key'])) else 0. for r in rows])
        elif f['kind'] == 'missing':
            columns.append([float(not finite(r.get(f['key']))) for r in rows])
        else:
            columns.append([float(str(r.get(f['key']) or 'missing') == f['level']) for r in rows])
    return np.asarray(columns).T.reshape(len(rows), len(spec)), spec


def fit_temperature(rows, *, period=56., prior_sd=1.):
    """Same one-vote/run local-calendar fit evaluated in the offline trial.

    Sparse/no independent contrasts retain the weak study shape. Nuisance
    terms explain associations during fitting, but never become pace credits.
    """
    if not math.isfinite(period) or not math.isfinite(prior_sd) or period <= 0 or prior_sd <= 0:
        raise ValueError('Positive finite calendar period and uncertainty required')
    if len({r['activity_id'] for r in rows}) != len(rows):
        raise ValueError('Unique physical runs required')
    result = dict(version=VERSION, scale=1., conditional_scale_sd=prior_sd,
        prior_scale=1., prior_sd=prior_sd, runs=len(rows), source=study_shape(),
        humidity_fitted=False, physiological_uncertainty_calibrated=False,
        calendar_days=period, independent_contrast_sd=0., personal_information_fraction=0.)
    if not rows:
        return result
    x, spec = design(rows, NUISANCE)
    from datetime import datetime
    t = np.array([datetime.fromisoformat(r['start_time_utc']).timestamp()/86400 for r in rows])
    blocks = np.floor(t/period).astype(int); columns = []
    for key in np.unique(blocks):
        mask = blocks == key
        columns.extend((mask.astype(float), np.where(mask, (t-t[mask].mean())/period, 0.)))
    z = np.column_stack(columns)
    def residual(v):
        return v-z@np.linalg.lstsq(z, v, rcond=None)[0]
    burden = np.array([r['air_burden'] for r in rows]); rb = residual(burden); rx = residual(x)
    ry = residual(np.log([r['pace_no_weather'] for r in rows])-burden)
    ridge = 4.*np.eye(rx.shape[1]); beta = np.linalg.solve(rx.T@rx+ridge, rx.T@ry)
    sigma = max(1e-6, float(np.sqrt(np.mean((ry-rx@beta)**2))))
    a = np.column_stack((rx, rb)); penalty = np.zeros((a.shape[1], a.shape[1])); penalty[:-1, :-1] = ridge
    penalty[-1, -1] = sigma**2/prior_sd**2; matrix = a.T@a+penalty
    coefficient = np.linalg.solve(matrix, a.T@ry)
    variance = float(np.linalg.inv(matrix)[-1, -1]*sigma**2)
    # An inverse association is not turned into a cooling/heating physiology.
    # Keep the weak prior, and make that fallback visible instead.
    estimate = 1.+float(coefficient[-1])
    result.update(scale=estimate if estimate >= 0 else 1.,
        conditional_scale_sd=math.sqrt(max(0., variance)) if estimate >= 0 else prior_sd,
        personal_estimate=estimate, independent_contrast_sd=float(np.std(rb)),
        personal_information_fraction=1.-variance/prior_sd**2,
        fallback='inverse_temperature_association_prior_retained' if estimate < 0 else None)
    return result


def comparison_covariance(calibration):
    legacy = calibration.get('comparison_legacy_heat_sd')
    covariance = np.zeros((5 if legacy is not None else 4,)*2); covariance[:3, :3] = calibration['covariance']
    weather = calibration.get('temperature_hr')
    covariance[3, 3] = (weather['conditional_scale_sd'] if weather else calibration['heat']['sd'])**2
    if legacy is not None:
        covariance[4, 4] = legacy**2
    return covariance
