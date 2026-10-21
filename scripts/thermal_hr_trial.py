"""Offline replacement weather model, never an extra weather credit.

Gagnon 2013 supplies a deliberately weak transferred shape, not individual
paired-data precision. Primary exposure is signed shade WBGT, with an air-only
sensitivity. No athlete score or outlier determines the population shape.
"""
from functools import lru_cache
import math

import numpy as np

from run_analysis.objective_modeling import _wbgt_c, _reference_wbgt_c
from scripts.audit_additional_pace_signals import design, day


NUISANCE = ('sampled_hr_bpm', 'sampled_minutes', 'duration_minutes', 'device', 'area',
            'hour_sin', 'hour_cos', 'stop_fraction', 'previous_7d_miles', 'previous_28d_miles')


@lru_cache(maxsize=1)
def study_shape():
    """Fit ONLY published running group means, 0 vs 22 C, RH 40%, n=10.

    Four repeated group means are not four independent studies. Their supplied
    SDs are not paired-difference SEs, so no pseudo-precise confidence is fitted.
    """
    minutes = np.array([15., 30., 45., 60.])
    difference = np.array([159.-152., 170.-159., 176.-162., 178.-163.])
    candidates = []
    for tau in np.linspace(1., 120., 1191):
        f = -np.expm1(-minutes/tau)
        amplitude = float(f@difference/(f@f))
        candidates.append((float(np.sum((amplitude*f-difference)**2)), float(tau), amplitude))
    loss, tau, amplitude = min(candidates)
    return dict(tau_minutes=tau, amplitude_bpm=amplitude, group_mean_squared_loss=loss,
        temperatures_c=[0., 22.], relative_humidity_percent=40., participants=10,
        measurement_minutes=minutes.tolist(), observed_hr_difference_bpm=difference.tolist(),
        source='https://pmc.ncbi.nlm.nih.gov/articles/PMC3650516/',
        transfer='Saturating interpolation in time and linear exposure contrast; not a published universal correction equation')


def hr_shift(row, config, *, exposure='wbgt'):
    """Positive = warmer conditions add HR; negative = cooler conditions."""
    shape = study_shape()
    minute = float(row['moving_minutes_into_run'])
    amplitude = shape['amplitude_bpm']*(-math.expm1(-minute/shape['tau_minutes']))
    if exposure == 'wbgt':
        observed, reference = _wbgt_c(row), _reference_wbgt_c(config)
        cold = _wbgt_c(dict(temperature_f=32., relative_humidity_percent=40.))
        warm = _wbgt_c(dict(temperature_f=71.6, relative_humidity_percent=40.))
    elif exposure == 'air':
        observed = (row['temperature_f']-32)*5/9
        reference = (config['reference_conditions']['temperature_f']-32)*5/9
        cold, warm = 0., 22.
    else:
        raise ValueError('Known thermal exposure required')
    return amplitude*(observed-reference)/(warm-cold)


def thermal_windows(rows, config, *, exposure='wbgt'):
    """Keep all windows; invalid/missing context is explicit and unchanged.

    Caller MUST disable the legacy heat transform when using these HR values.
    No HR clipping, new fitness gate, source rewrite or fast-run penalty.
    """
    result = []; diagnostics = []
    for r in rows:
        copy = dict(r); reason = None; delta = None
        try:
            delta = hr_shift(r, config, exposure=exposure)
            adjusted = float(r['average_hr_bpm'])-delta
            if not math.isfinite(adjusted) or adjusted <= config['resting_hr']:
                reason = 'unsupported_thermal_hr_mapping'
            else:
                copy['average_hr_bpm'] = adjusted
        except (TypeError, ValueError, KeyError):
            reason = 'missing_thermal_context'
        diagnostics.append(dict(applied=reason is None, reason=reason, hr_shift_bpm=delta))
        result.append(copy)
    return result, diagnostics


def fit_weather(rows, *, model='current', prior_sd=1., period=56.):
    """Calendar-adjusted one-vote-per-run fit; only prior rows supplied by caller.

    Temperature-scale prior is 1 +/- 1, an engineering transfer uncertainty,
    not a study-derived CI. Nuisance ridge 4 and calendar controls reuse the
    existing signal audit. No coefficient is forced positive after seeing data.
    """
    if not rows or len({r['activity_id'] for r in rows}) != len(rows):
        raise ValueError('Unique nonempty physical runs required')
    if prior_sd <= 0 or period <= 0:
        raise ValueError('Positive uncertainty and calendar period required')
    if model not in {'none', 'current', 'population', 'personal', 'air', 'air_population'}:
        raise ValueError('Known weather trial model required')
    x, spec = design(rows, NUISANCE)
    t = np.array([day(r) for r in rows]); blocks = np.floor(t/period).astype(int)
    columns = []
    for key in np.unique(blocks):
        mask = blocks == key
        columns.extend((mask.astype(float), np.where(mask, (t-t[mask].mean())/period, 0.)))
    z = np.column_stack(columns)
    def residual(v):
        return v-z@np.linalg.lstsq(z, v, rcond=None)[0]
    y = np.log([r['pace_no_weather'] for r in rows]); rx = residual(x)
    burden_key = 'current_burden' if model in {'none', 'current'} else ('air_burden' if model.startswith('air') else 'thermal_burden')
    burden = np.array([r[burden_key] for r in rows]) if model != 'none' else np.zeros(len(rows))
    rb = residual(burden)
    ry = residual(y-burden)
    ridge = 4.*np.eye(rx.shape[1])
    beta = np.linalg.solve(rx.T@rx+ridge, rx.T@ry)
    sigma = max(1e-6, float(np.sqrt(np.mean((ry-rx@beta)**2))))
    scale = 1.; conditional_sd = prior_sd; data_weight = 0.
    learned = model not in ('none', 'current', 'population', 'air_population')
    if learned:
        a = np.column_stack((rx, rb))
        penalty = np.zeros((a.shape[1], a.shape[1])); penalty[:-1, :-1] = ridge
        penalty[-1, -1] = sigma**2/prior_sd**2
        # Last coefficient is the update from scale 1, not scale itself.
        matrix = a.T@a+penalty
        coefficient = np.linalg.solve(matrix, a.T@ry)
        beta = coefficient[:-1]; scale += float(coefficient[-1])
        variance = float(np.linalg.inv(matrix)[-1, -1]*sigma**2)
        conditional_sd = math.sqrt(max(0., variance))
        data_weight = 1.-variance/prior_sd**2
    return dict(scale=scale if model != 'none' else 0., conditional_scale_sd=conditional_sd, prior_sd=prior_sd,
        personal_information_fraction=data_weight, model=model, burden_key=burden_key,
        nuisance_beta=beta.tolist(), nuisance_spec=spec, calendar_blocks=len(np.unique(blocks)),
        remaining_burden_sd=float(np.std(rb)), residual_log_pace_sd=sigma,
        source_ids=[r['activity_id'] for r in rows], physiological_uncertainty_calibrated=False)


def forward(rows, *, model='current', prior_sd=1., period=56.):
    """Predict unchanged weather-unadjusted target, never corrected target truth."""
    ordered = sorted(rows, key=day); cases = []; not_evaluated = []
    for i, target in enumerate(ordered):
        train = ordered[:i]
        if i < 24 or day(train[-1])-day(train[0]) < 56:
            not_evaluated.append(dict(activity_id=target['activity_id'], reason='insufficient_prior_history')); continue
        anchor = train[-1]
        if day(target)-day(anchor) > 14:
            not_evaluated.append(dict(activity_id=target['activity_id'], reason='no_recent_anchor')); continue
        fitted = fit_weather(train, model=model, prior_sd=prior_sd, period=period)
        x, _ = design([anchor, target], NUISANCE, fitted['nuisance_spec'])
        burden = fitted['scale']*(target[fitted['burden_key']]-anchor[fitted['burden_key']])
        prediction = anchor['pace_no_weather']*math.exp(burden+float((x[1]-x[0])@fitted['nuisance_beta']))
        cases.append(dict(activity_id=target['activity_id'], date=target['start_time_utc'], anchor_id=anchor['activity_id'],
            temperature_f=target.get('temperature_f'), prediction=prediction, actual=target['pace_no_weather'],
            error_seconds=60*(prediction-target['pace_no_weather']), scale=fitted['scale'], prior_runs=i))
    def summary(selected):
        return dict(cases=len(selected), mae_seconds=float(np.mean([abs(r['error_seconds']) for r in selected])) if selected else None,
                    rmse_seconds=float(np.sqrt(np.mean([r['error_seconds']**2 for r in selected]))) if selected else None)
    return dict(summary=summary(cases), recent=summary([r for r in cases if r['date'] >= '2026-03-01']),
        cool=summary([r for r in cases if r['temperature_f'] is not None and r['temperature_f'] < 55]),
        warm=summary([r for r in cases if r['temperature_f'] is not None and r['temperature_f'] >= 70]),
        cases=cases, not_evaluated=not_evaluated)
