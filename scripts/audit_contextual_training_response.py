"""Offline contextual-response experiment, with chronological error updates.

Predeclared alternatives to the existing constant calendar/mileage response.
No May output participates in model fitting or selection. No live code, DB,
planner, corrections, settings or existing forecast is modified.
"""
from argparse import ArgumentParser
from collections import Counter
import json
import math
from pathlib import Path

import numpy as np

from run_analysis.privacy import private_file
from run_analysis.training_response import fit_response
from scripts.audit_empirical_training_response import disjoint, score


# The existing dose/clock fits are controls. Added terms test changes in dose
# and quality relative to the runner's starting training, not starting pace.
# Scaling is numerical only; the fits are unpenalized and sign-unconstrained.
MODELS = {
    'calendar': ('elapsed',),
    'miles': ('miles',),
    'volume_context': ('elapsed', 'miles', 'usual_miles'),
    'quality_context': ('elapsed', 'miles', 'quality_change'),
    'volume_quality_context': ('elapsed', 'miles', 'usual_miles', 'quality_change'),
}
COHORTS = ('pooled', 'no_documented_illness')


def context(row):
    """Zero-duration -> zero change. Never regress absolute pace to a mean."""
    return {**row, 'usual_miles': row['prior_volume'] * row['elapsed'] * 4 / 100,
        'quality_change': (row['delivered_quality'] - row['prior_quality']) * row['elapsed']}


def weights(rows):
    origins = Counter(r['origin_id'] for r in rows)
    outcomes = Counter(i for r in rows for i in r['outcome_ids'])
    w = np.array([sum(1/outcomes[i] for i in r['outcome_ids']) / len(r['outcome_ids']) /
        origins[r['origin_id']] / r['response_variance'] for r in rows])
    return w / w.sum()


def fit(rows, keys):
    if len({r['origin_id'] for r in rows}) < 8:
        return None
    x = np.array([[r[k] for k in keys] for r in rows], float)
    y = np.array([r['change'] for r in rows], float)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Finite aligned inputs required')
    w = weights(rows)
    scale = np.sqrt(w @ (x*x))
    scale[scale < 1e-12] = 1
    design = np.sqrt(w)[:, None] * x / scale
    if np.linalg.matrix_rank(design) < len(keys):
        return None
    beta = np.linalg.lstsq(design, np.sqrt(w)*y, rcond=None)[0] / scale
    return dict(coefficients=dict(zip(keys, map(float, beta))), origins=len({r['origin_id'] for r in rows}),
        latest_outcome=max(r['outcome_end'] for r in rows), condition_number=float(np.linalg.cond(design)))


def predict(fitted, row):
    return sum(value * row[key] for key, value in fitted['coefficients'].items())


def correction(prior):
    """Learn bias per elapsed time from completed OUT-OF-SAMPLE errors only.

    No fixed decay, lookback, sign constraint or hand-picked correction size.
    This is a testable bias correction, not a measured physiological state.
    """
    if len({r['origin_id'] for r in prior}) < 8:
        return None
    w = weights(prior)
    x = np.array([r['elapsed'] for r in prior])
    y = np.array([-r['log_error'] for r in prior])
    return dict(coefficient=float((w@(x*y))/(w@(x*x))),
        latest_outcome=max(r['outcome_end'] for r in prior))


def choose(prior, available):
    """Same-case earlier errors only, horizons balanced as in prior audits."""
    eligible = [m for m in sorted(available) if len({r['origin'] for r in prior if r['method'] == m}) >= 8]
    if not eligible:
        return None
    common = set.intersection(*[{(r['origin'], r['target_id']) for r in prior if r['method'] == m} for m in eligible])
    if len({o for o, t in common}) < 8:
        return None
    losses = {}
    for method in eligible:
        rows = [r for r in prior if r['method'] == method and (r['origin'], r['target_id']) in common]
        losses[method] = float(np.mean([score([r for r in rows if r['horizon'] == h])['log_mae']
            for h in sorted({r['horizon'] for r in rows})]))
    return dict(method=min(losses, key=losses.get), losses=losses, common_cases=len(common),
        latest_outcome=max(r['outcome_end'] for r in prior if (r['origin'], r['target_id']) in common))


def evaluate(examples):
    examples = [context(r) for r in examples]
    predictions = []
    for origin in sorted({r['origin'] for r in examples}):
        completed = [r for r in examples if r['outcome_end'] < origin]
        previous = [r for r in predictions if r['outcome_end'] < origin and r['method'] != 'selected']
        fits = {}
        for cohort in COHORTS:
            training = [r for r in completed if cohort == 'pooled' or not r['consistency']['health_events']]
            for model, keys in MODELS.items():
                fitted = fit(training, keys)
                if fitted is None:
                    continue
                method = f'{cohort}/{model}'
                fits[method] = (fitted, None)
                errors = [r for r in previous if r['method'] == method]
                adjust = correction(errors)
                if adjust:
                    fits[method + '/error_update'] = (fitted, adjust)
                clean_adjust = correction([r for r in errors if r['regime'] == 'consistent'])
                if clean_adjust:
                    fits[method + '/consistent_error_update'] = (fitted, clean_adjust)
        # Forecast condition is continued consistent training. Selection can
        # therefore use previously COMPLETED consistent outcomes, never the
        # unknown future cohort of the current test case.
        selection = choose([r for r in previous if r['regime'] == 'consistent'], fits)
        # Exact response forms underlying the active v3 scenario envelope.
        controls = {f'current/{name}': fit_response(
            [r for r in completed if r['consistency']['regime'] == 'consistent'], name)
            for name in ('calendar', 'log_miles')}
        for row in (r for r in examples if r['origin'] == origin):
            common = dict(origin=origin, origin_id=row['origin_id'], target_id=row['target_id'],
                outcome_ids=row['outcome_ids'], outcome_end=row['outcome_end'], horizon=row['horizon'],
                elapsed=row['elapsed'], response_variance=row['response_variance'],
                regime=row['consistency']['regime'], actual=row['actual'], persistence=row['start_pace'])
            produced = {}
            for method, (fitted, adjust) in fits.items():
                change = predict(fitted, row) + (adjust['coefficient'] * row['elapsed'] if adjust else 0)
                predicted = row['start_pace'] * math.exp(change)
                p = dict(**common, method=method, predicted=predicted,
                    log_error=math.log(predicted/row['actual']), latest_fit=fitted['latest_outcome'],
                    latest_update=adjust['latest_outcome'] if adjust else None,
                    coefficients=fitted['coefficients'], correction=adjust['coefficient'] if adjust else 0,
                    condition_number=fitted['condition_number'])
                predictions.append(p)
                produced[method] = p
            if selection:
                predictions.append({**produced[selection['method']], 'method': 'selected',
                    'selected_method': selection['method'], 'latest_selection': selection['latest_outcome']})
            for method, fitted in controls.items():
                if fitted:
                    predicted = row['start_pace'] * math.exp(predict(fitted, row))
                    predictions.append(dict(**common, method=method, predicted=predicted,
                        log_error=math.log(predicted/row['actual']), latest_fit=fitted['latest_outcome'],
                        latest_update=None, coefficients=fitted['coefficients'], correction=0.,
                        condition_number=fitted['condition_number']))
    return predictions


def sequential_blocks(rows):
    """Disjoint full origin-to-outcome intervals, ordered without error sign."""
    end = ''
    result = []
    for row in sorted(rows, key=lambda r: (r['outcome_end'], r['origin'], r['target_id'])):
        if row['origin'] >= end:
            result.append(row)
            end = row['outcome_end']
    return result


def compare(predictions, method, baseline, *, horizon=None, regime=None, origin_after=None):
    subset = [r for r in predictions if (horizon is None or r['horizon'] == horizon)
        and (regime is None or r['regime'] == regime) and (origin_after is None or r['origin'] >= origin_after)]
    lookup = {(r['origin'], r['target_id']): r for r in subset if r['method'] == baseline}
    candidate = [r for r in subset if r['method'] == method and (r['origin'], r['target_id']) in lookup]
    def pair(rows):
        other = [lookup[r['origin'], r['target_id']] for r in rows]
        return dict(candidate=score(rows), baseline=score(other))
    return dict(all=pair(candidate), outcome_disjoint=pair(disjoint(candidate)),
        sequential=pair(sequential_blocks(candidate)))


def report(predictions):
    predictions = [*predictions, *[{**r, 'method': 'unchanged', 'predicted': r['persistence'],
        'log_error': math.log(r['persistence']/r['actual'])} for r in predictions if r['method'] == 'pooled/calendar']]
    methods = sorted({r['method'] for r in predictions} - {'unchanged'})
    return {label: {m: {b: compare(predictions, m, b, **options)
        for b in ('pooled/calendar', 'pooled/miles', 'current/v3_midpoint', 'unchanged')}
        for m in methods} for label, options in (
            ('all', {}), ('consistent', dict(regime='consistent')),
            ('consistent_14', dict(regime='consistent', horizon=14)),
            ('consistent_28', dict(regime='consistent', horizon=28)),
            ('consistent_56', dict(regime='consistent', horizon=56)),
            ('latest_consistent', dict(regime='consistent', origin_after='2026-09-01')))}


def attach_current_midpoint(results):
    """Exact min/max midpoint of all eight active v3 forms, same cases.

    A control is only available when every policy/model had a historical fit.
    No reduced two-scenario substitute is silently compared with production.
    """
    lookups = [{(r['origin'], r['target_id']): r for r in data['predictions'] if r['method'] == method}
        for data in results.values() for method in ('current/calendar', 'current/log_miles')]
    common = set.intersection(*(set(v) for v in lookups))
    for data in results.values():
        bases = {(r['origin'], r['target_id']): r for r in data['predictions'] if r['method'] == 'pooled/calendar'}
        for key in sorted(common):
            values = [lookup[key]['predicted'] for lookup in lookups]
            predicted = (min(values) + max(values))/2
            base = bases[key]
            data['predictions'].append({**base, 'method': 'current/v3_midpoint',
                'predicted': predicted, 'log_error': math.log(predicted/base['actual']),
                'latest_fit': max(lookup[key]['latest_fit'] for lookup in lookups),
                'coefficients': None, 'scenario_count': len(values)})


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=Path('output/consistent-training-response-20261020.json'))
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    source = json.loads(args.input.read_text())
    results = {}
    for policy, data in source['policies'].items():
        predictions = evaluate(data['examples'])
        results[policy] = dict(predictions=predictions)
    attach_current_midpoint(results)
    for data in results.values():
        data['summary'] = report(data['predictions'])
    output = dict(version='contextual-training-response-v1', as_of=source['as_of'],
        primary_policy=source['primary_policy'], policies=results,
        protocol=['Same frozen causal benchmark and physical exposure as previous audits',
            'Fit only outcomes completed strictly before the forecast origin',
            'Context models and error correction fixed before inspecting May projections',
            'Unpenalized signed fits; no starting-pace regression, imposed taper or forced positive growth',
            'Corrections use matured out-of-sample errors, never in-sample residuals',
            'Selection uses common earlier consistent cases and equal horizon weights',
            'Report identical matched cases, disjoint outcomes and disjoint full intervals'],
        limitations=['Actual future exposure and consistency are conditional, not known at forecast origin',
            'No consistent-error update or automatic selector is available before eight matured earlier consistent forecast origins',
            'Quality session counts are not measured quality dose',
            'Recent usual volume is a training context, not a physiological fatigue measurement',
            'Overlapping causal benchmarks retain shared history and retrospective corrections',
            'Very few distinct consistent stretches; no seven-month prediction interval validated'],
        live_forecast_changed=False, planner_changed=False, database_changed=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
    private_file(args.output)
    primary = results[source['primary_policy']]['summary']
    compact = {scope: {m: dict(pairs=comparisons['pooled/miles']['all']['candidate']['pairs'],
        mae=comparisons['pooled/miles']['all']['candidate'].get('mae_seconds'),
        matched_miles_mae=comparisons['pooled/miles']['all']['baseline'].get('mae_seconds'))
        for m, comparisons in methods.items()} for scope, methods in primary.items()
        if scope in {'consistent_28', 'latest_consistent'}}
    print(json.dumps(compact, indent=2))


if __name__ == '__main__':
    main()
