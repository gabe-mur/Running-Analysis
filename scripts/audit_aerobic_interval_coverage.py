"""Conditional coverage audit, without changing production intervals or data.

Truth is the production robust estimate on noise-free latent run paces, not the
instantaneous endpoint gain. Empirical residuals may already include sensor and
calibration errors: do not add independent measurement noise a second time.
"""
from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import sqlite3
from statistics import NormalDist
from time import perf_counter

import numpy as np

from run_analysis.aerobic_progress import _period, _change, build_aerobic_progress
from run_analysis.analytics import MIN_MEASUREMENT_SIGMA_MIN_MILE
from run_analysis.config import load_config, resolve_project_path
from run_analysis import pace_comparability as pc
from run_analysis.read_cache import invariant_snapshot
from scripts.audit_progress_responsiveness import at, covariance, build_from_windows
from scripts.audit_small_aerobic_changes import (
    GAINS, DURATIONS, SEED, describe, noise_library, draw_noise, simulate_rows,
)


def period_parts(rows, end, cov, *, include_design=False):
    """Decompose exactly the returned variance, using the shipped location."""
    p = _period(rows, end, cov)
    selected = [r for r in rows if end-timedelta(days=28) < at(r) <= end]
    pace = np.array([r['pace_min_mile'] for r in selected])
    sigma = np.array([max(MIN_MEASUREMENT_SIGMA_MIN_MILE,
                         r['uncertainty_95_min_mile']/1.96) for r in selected])
    base = np.array([min(1., r['trend_weight']) for r in selected])/sigma**2
    weight = base.copy()
    level = float(np.average(pace, weights=weight))
    for _ in range(8):
        residual = pace-level
        scale = max(5/60, 1.4826*float(np.median(abs(residual-np.median(residual)))))
        weight = base*np.minimum(1., 1.345*scale/np.maximum(abs(residual), 1e-12))
        level = float(np.average(pace, weights=weight))
    weight /= weight.sum()
    assert math.isclose(level, p['pace_min_mile'], rel_tol=1e-12)
    gradients = -pace[:, None]*np.array([r['shared_log_speed_gradient'] for r in selected])
    shared = np.einsum('ij,jk,ik->i', gradients, cov, gradients)
    measurement = float(np.sum(weight**2*np.maximum(sigma**2-shared, 0.)))
    # Keep the pre-change additive baseline reproducible after deployment.
    day = p['observed_variance']
    parts = dict(measurement=measurement, day=day, shared=p['shared_calibration_variance'])
    if include_design:
        return p, parts, dict(rows=selected, pace=pace, weight=weight,
            independent=np.maximum(sigma**2-shared, 0.), gradients=gradients)
    return p, parts


def evaluate(rows, end, cov):
    current, cp = period_parts(rows, end, cov)
    out = dict(level=dict(estimate=current['pace_min_mile'],
        measurement=cp['measurement'], day=cp['day'], shared=cp['shared'],
        reported=current['random_variance'], established=current['evidence']=='established'))
    for name, days in (('monthly', 28), ('ninety_day', 90)):
        prior, pp = period_parts(rows, end-timedelta(days=days), cov)
        change = _change(current, prior, cov, comparison_days=days)
        c, p = current['pace_min_mile'], prior['pace_min_mile']
        out[name] = dict(estimate=math.log(p/c),
            measurement=cp['measurement']/c**2+pp['measurement']/p**2,
            day=cp['day']/c**2+pp['day']/p**2,
            shared=change['shared_calibration_log_variance'],
            reported=prior['random_variance']/p**2+current['random_variance']/c**2,
            established=change['direction']!='provisional')
        lo, hi = change['estimated_change_range_95_percent']
        se = math.sqrt(out[name]['reported']+out[name]['shared'])
        assert math.isclose(100*math.expm1(out[name]['estimate']-1.96*se), lo, abs_tol=1e-9)
        assert math.isclose(100*math.expm1(out[name]['estimate']+1.96*se), hi, abs_tol=1e-9)
    return out


def wilson(successes, count):
    p = successes/count
    denominator = 1+1.96**2/count
    center = (p+1.96**2/(2*count))/denominator
    half = 1.96*math.sqrt(p*(1-p)/count+1.96**2/(4*count**2))/denominator
    return [center-half, center+half]


def coverage(samples, truth, components):
    errors = np.array([r['estimate']-truth for r in samples])
    variances = np.array([sum(r[k] for k in components) for r in samples])
    se = np.sqrt(variances)
    z = errors/np.maximum(se, 1e-15)
    rates = {}
    for nominal in (.80, .95):
        cutoff = NormalDist().inv_cdf((1+nominal)/2)
        count = int(np.sum(abs(z)<=cutoff))
        rates[str(nominal)] = dict(covered=count, rate=count/len(z),
                                  monte_carlo_wilson_95=wilson(count, len(z)))
    return dict(draws=len(samples), established=sum(r['established'] for r in samples),
        coverage=rates, bias=float(errors.mean()), rmse=float(np.sqrt(np.mean(errors**2))),
        empirical_error_sd=float(errors.std(ddof=1)), rms_reported_se=float(np.sqrt(variances.mean())),
        error_sd_to_reported_se=float(errors.std(ddof=1)/max(np.sqrt(variances.mean()), 1e-15)),
        absolute_standardized_error=describe(abs(z)),
        variance_fractions={k:float(np.mean([r[k] for r in samples])/max(variances.mean(), 1e-30))
                            for k in components})


def simulate_measurement(latent, cov, rng, shared_draw):
    """Gaussian log errors at the claimed per-run SD, covariance held coherent."""
    result = []
    delta = rng.multivariate_normal(np.zeros(len(cov)), cov) if shared_draw else np.zeros(len(cov))
    for row in latent:
        p = row['pace_min_mile']; g = np.array(row['shared_log_speed_gradient'])
        sigma = max(MIN_MEASUREMENT_SIGMA_MIN_MILE, row['uncertainty_95_min_mile']/1.96)
        independent = max(0., sigma**2-p*p*float(g@cov@g))
        error = rng.normal(0., math.sqrt(independent)/p)-float(g@delta)
        result.append(dict(row, pace_min_mile=p*math.exp(error)))
    return result


def checks(rows, cov, end, draws=512, seed=SEED, *, evaluator=evaluate, extra_variants=()):
    eligible = sorted([r for r in rows if r.get('included_in_trend') and r['trend_weight']>0 and at(r)<=end], key=at)
    end = at(eligible[-1])
    donors = [r for r in eligible if at(r)>=end-timedelta(days=180)]
    template = [r for r in eligible if at(r)>end-timedelta(days=118)]
    base = float(np.median([r['pace_min_mile'] for r in donors]))
    libraries = []
    for detrend in (False, True):
        for days in (7, 14):
            blocks, info = noise_library(donors, days, detrend)
            libraries.append((blocks, info))
    modes = [('claimed_measurement_gaussian', None, None),
             *[(info['residual_method'], info['block_days'], blocks) for blocks, info in libraries]]
    cells = []
    for mode_index, (mode, days, blocks) in enumerate(modes):
        for shared_draw in (False, True):
            for duration in DURATIONS:
                onset = end-timedelta(days=duration)
                for gain in GAINS:
                    latent = simulate_rows(template, np.zeros(len(template)), base, onset, gain, duration)
                    truth = evaluator(latent, end, cov)
                    samples = {k:[] for k in truth}
                    # Common draws across gains and ramp lengths, never selected by outcomes.
                    rng = np.random.default_rng(seed+mode_index*100+int(shared_draw))
                    for _ in range(draws):
                        if blocks is None:
                            observed = simulate_measurement(latent, cov, rng, shared_draw)
                        else:
                            noise = draw_noise(template, blocks, rng)
                            delta = rng.multivariate_normal(np.zeros(len(cov)), cov) if shared_draw else np.zeros(len(cov))
                            noise -= np.array([np.array(r['shared_log_speed_gradient'])@delta for r in template])
                            observed = simulate_rows(template, noise, base, onset, gain, duration)
                        evaluated = evaluator(observed, end, cov)
                        for key in samples: samples[key].append(evaluated[key])
                    cells.append(dict(noise=mode, block_days=days, shared_calibration_draw=shared_draw,
                        ramp_days=duration, endpoint_gain_percent=100*gain,
                        truth={k:v['estimate'] for k,v in truth.items()},
                        results={k:dict(full=coverage(v, truth[k]['estimate'], ('measurement','day','shared')),
                            shipped=coverage(v, truth[k]['estimate'], ('reported','shared')),
                            # Attribution only; these are not proposed replacement intervals.
                            no_measurement_term=coverage(v, truth[k]['estimate'], ('day','shared')),
                            conditional_random_only=coverage(v, truth[k]['estimate'], ('measurement','day')),
                            **{name:coverage(v, truth[k]['estimate'], (name, 'shared')) for name in extra_variants})
                            for k,v in samples.items()}))
    oracle = build_aerobic_progress(latent, as_of=end, covariance=cov, history_days=1)
    assert math.isclose(truth['level']['estimate'], oracle['current']['pace_min_mile'], rel_tol=1e-12)
    assert math.isclose(100*math.expm1(truth['monthly']['estimate']), oracle['change']['speed_change_percent'], abs_tol=1e-9)
    return dict(last_run=end.isoformat(), template_runs=len(template), seed=seed, draws_per_cell=draws,
        libraries=[info for _,info in libraries], cells=cells,
        estimand='Noise-free production robust trailing 28-day pace and its log contrasts, not instantaneous endpoint fitness.')


def audit(database, config, end, draws=512, *, evaluator=evaluate, extra_variants=(), source_diagnostic=None):
    started = perf_counter()
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro', uri=True) as c:
        c.row_factory = sqlite3.Row
        with invariant_snapshot(c):
            size = int(config['model'].get('window_seconds', 300))
            sets, _ = pc.load_model_window_sets(c, config, (size,), check_elevation=True)
            overlapping, _ = pc.load_overlapping_model_windows(c, config,
                window_seconds=int(config['model']['primary_window_seconds']),
                stride_seconds=int(config['model']['primary_window_stride_seconds']), check_elevation=True)
            sets = {k:[r for r in v if datetime.fromisoformat(r['start_time_utc'])<=end] for k,v in sets.items()}
            overlapping = [r for r in overlapping if datetime.fromisoformat(r['start_time_utc'])<=end]
            report = build_from_windows(c, config, sets, overlapping); before = deepcopy(report)
            simulation = checks(report['runs'], covariance(report), end, draws,
                evaluator=evaluator, extra_variants=extra_variants)
            real = evaluator(sorted([r for r in report['runs'] if r.get('included_in_trend')
                and r['trend_weight']>0],key=at),
                max(at(r) for r in report['runs'] if r.get('included_in_trend') and r['trend_weight']>0), covariance(report))
            source_checks = source_diagnostic(report['runs'], end) if source_diagnostic else None
            assert report==before and c.total_changes==0
    return dict(version='aerobic-interval-coverage-audit-v1', as_of=end.isoformat(),
        source_database_read_only=True, production_model_changed=False, simulation=simulation,
        elapsed_seconds=perf_counter()-started, real_history_variance_diagnostic=real,
        source_diagnostic=source_checks, assumptions=[
            'Truth is the shipped estimator on noise-free latent paces with real dates and fixed uncertainty/context weights.',
            'Endpoint gains 0/0.5/1/2 percent over 56/84 days are not the trailing-window comparison truth.',
            'Gaussian controls use claimed independent per-run measurement variance, without adding empirical noise.',
            'Empirical 7/14-day block controls reuse observed residual variation; no extra independent measurement noise is added.',
            'Shared-calibration controls draw one coherent coefficient error per complete history using supplied covariance.',
            'Empirical noise may already contain calibration error; added shared draws are a conservative sensitivity, not identified extra error.',
            'Variance-removal controls diagnose attribution; they are not candidate production fixes or calibrated physiological intervals.',
            'Wilson intervals describe Monte Carlo precision conditional on this one-athlete resampling model, not population uncertainty.',
            'Absolute pace is tested, not VO2-equivalent scale accuracy or laboratory fitness.',
            'No model selection, coefficient tuning, raw sensor gate validation, planner calls, or activity writes.'])


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--as-of', type=datetime.fromisoformat, default=datetime.now(timezone.utc))
    p.add_argument('--draws', type=int, default=512)
    args = p.parse_args()
    if args.output.exists(): p.error('Choose a new output path')
    if args.as_of.tzinfo is None: p.error('Timezone-aware --as-of required')
    if args.draws<2: p.error('At least two draws required')
    root = Path(__file__).resolve().parents[1]; config = load_config(root/'config.yaml')
    result = audit(resolve_project_path(root, config['paths']['database']), config, args.as_of, args.draws)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(cells=len(result['simulation']['cells']), draws_per_cell=args.draws,
                         elapsed_seconds=result['elapsed_seconds']), indent=2))


if __name__=='__main__': main()
