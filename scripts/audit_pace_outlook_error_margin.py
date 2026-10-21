"""Empirical forecast-error scaling, not a scenario-envelope confidence band.

Offline audit of the exact active v3 midpoint's chronological predictions.
Both continued-consistency and all-recorded-context errors remain visible.
Observed residuals, not alternate forecast equations, determine the spread.
Extrapolation from shorter horizons is explicit and never called calibrated
May coverage. No live model, database, settings or planner writes.
"""
from argparse import ArgumentParser
import json
import math
from pathlib import Path

import numpy as np

from run_analysis.privacy import private_file
from run_analysis.pace_forecast_error import weights,weighted_quantile
from scripts.audit_contextual_training_response import sequential_blocks
from scripts.audit_empirical_training_response import disjoint


QUANTILES = (.1, .9)
SCALINGS = {'constant_error': 0., 'independent_increments': .5, 'persistent_rate_error': 1.}


def scaled_errors(rows, days, exponent):
    if days <= 0 or not math.isfinite(days) or not math.isfinite(exponent):
        raise ValueError('Positive finite horizon and finite exponent required')
    # log(actual/predicted): positive means actual pace was slower.
    return np.array([-r['log_error']*(days/(28*r['elapsed']))**exponent for r in rows])


def interval(rows, *, days, pace, exponent):
    if not rows:
        return dict(available=False, pairs=0)
    if pace <= 0 or not math.isfinite(pace):
        raise ValueError('Finite positive central pace required')
    w = weights(rows)
    e = scaled_errors(rows, days, exponent)
    lo, hi = weighted_quantile(e, w, QUANTILES)
    radius = float(weighted_quantile(abs(e), w, [.8])[0])
    return dict(available=True, pairs=len(rows), origins=len({r['origin'] for r in rows}),
        outcome_runs=len({i for r in rows for i in r['outcome_ids']}),
        weighted_mean_log_error=float(w@e), q10_log_error=float(lo), q90_log_error=float(hi),
        low_pace=float(pace*math.exp(lo)), high_pace=float(pace*math.exp(hi)),
        absolute_error_80_log=radius,
        centered_low_pace=float(pace*math.exp(-radius)),
        centered_high_pace=float(pace*math.exp(radius)),
        latest_error_outcome=max(r['outcome_end'] for r in rows))


def summary(rows):
    if not rows:
        return dict(pairs=0)
    w = weights(rows)
    e = np.array([-r['log_error'] for r in rows])
    return dict(pairs=len(rows), origins=len({r['origin'] for r in rows}),
        outcome_runs=len({i for r in rows for i in r['outcome_ids']}),
        actual_days=float(w@np.array([r['elapsed']*28 for r in rows])),
        rms_log_error=float(np.sqrt(w@(e*e))), mean_log_error=float(w@e),
        mae_seconds=float(w@np.abs([(r['actual']-r['predicted'])*60 for r in rows])),
        error_80_log=float(weighted_quantile(abs(e), w, [.8])[0]))


def horizon_growth(rows):
    """Descriptive power law across horizon RMS errors, not a causal model.

    Each horizon has one vote. All fits retain the sign of the estimated
    exponent; no forced sqrt(time), rate persistence or upper/lower clamp.
    """
    groups = {h: [r for r in rows if r['horizon'] == h] for h in sorted({r['horizon'] for r in rows})}
    stats = {h: summary(v) for h, v in groups.items()}
    if len(stats) < 2:
        return None
    x = np.log([v['actual_days'] for v in stats.values()])
    y = np.log([v['rms_log_error'] for v in stats.values()])
    beta = np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]), y, rcond=None)[0]
    return dict(exponent=float(beta[1]), intercept=float(beta[0]), horizon_count=len(stats),
        max_observed_horizon=max(stats), note='Different horizons may contain different origins and health contexts')


def matched_growth(rows, first=14, second=28):
    """Same origins at both horizons, avoiding changes in case composition."""
    a = {r['origin']: r for r in rows if r['horizon'] == first}
    b = {r['origin']: r for r in rows if r['horizon'] == second}
    common = sorted(set(a)&set(b))
    if len(common) < 2:
        return dict(available=False, pairs=len(common))
    sa, sb = summary([a[o] for o in common]), summary([b[o] for o in common])
    power = math.log(sb['rms_log_error']/sa['rms_log_error']) / math.log(sb['actual_days']/sa['actual_days'])
    return dict(available=True, pairs=len(common), exponent=power, earlier=sa, later=sb,
        note='Diagnostic only: too few same-origin pairs to certify long-horizon scaling')


def transfer_test(rows, source_horizon, target_horizon, exponent):
    """Transfer shorter-horizon residual bands using only already known errors.

    Same current method. Target residual and future calibration residuals may
    not enter the calibration. Compare nominal 10th–90th residual band coverage;
    do not interpret overlapping forecasts as independent trials.
    """
    predictions = []
    for target in (r for r in rows if r['horizon'] == target_horizon):
        prior = [r for r in rows if r['horizon'] == source_horizon and r['outcome_end'] < target['origin']]
        # Need more than two distinct disjoint error outcomes to estimate tails;
        # this is a minimum audit support check, not a coverage guarantee.
        calibration = disjoint(prior)
        if len(calibration) < 3:
            continue
        band = interval(calibration, days=target['elapsed']*28, pace=target['predicted'], exponent=exponent)
        predictions.append({**target, 'covered': band['low_pace'] <= target['actual'] <= band['high_pace'],
            'centered_covered': band['centered_low_pace'] <= target['actual'] <= band['centered_high_pace'],
            'band': band, 'calibration_cases': len(calibration)})
    def stats(chosen):
        if not chosen:
            return dict(pairs=0)
        w = weights(chosen)
        return dict(pairs=len(chosen), coverage=float(w@np.array([r['covered'] for r in chosen])),
            centered_coverage=float(w@np.array([r['centered_covered'] for r in chosen])),
            average_width_seconds=float(w@np.array([(r['band']['high_pace']-r['band']['low_pace'])*60 for r in chosen])))
    return dict(all=stats(predictions), sequential=stats(sequential_blocks(predictions)), predictions=predictions)


def audit(rows, days, pace):
    output = {}
    broader_growth = matched_growth(rows)
    # Borrowing this descriptive scaling from mixed histories to continued
    # consistency is explicit: clean-history pairs alone cannot establish it.
    empirical_exponent = broader_growth['exponent'] if broader_growth['available'] else None
    for scope in ('continued_consistency', 'all_recorded_contexts'):
        chosen = [r for r in rows if scope == 'all_recorded_contexts' or r['regime'] == 'consistent']
        horizon_stats = {}
        projections = {}
        for horizon in sorted({r['horizon'] for r in chosen}):
            group = [r for r in chosen if r['horizon'] == horizon]
            horizon_stats[horizon] = dict(all=summary(group), outcome_disjoint=summary(disjoint(group)),
                sequential=summary(sequential_blocks(group)))
            projections[horizon] = {method: dict(
                all=interval(group, days=days, pace=pace, exponent=exponent),
                outcome_disjoint=interval(disjoint(group), days=days, pace=pace, exponent=exponent),
                sequential=interval(sequential_blocks(group), days=days, pace=pace, exponent=exponent))
                for method, exponent in SCALINGS.items()}
        output[scope] = dict(horizon_stats=horizon_stats, projections=projections,
            descriptive_horizon_growth=horizon_growth(chosen), matched_14_28_growth=matched_growth(chosen),
            combined_short_error_projection={method: interval(
                [r for r in chosen if r['horizon'] in {14, 28}], days=days, pace=pace, exponent=exponent)
                for method, exponent in {**SCALINGS,
                    **({'empirical_matched_power': empirical_exponent} if empirical_exponent is not None else {})}.items()},
            shorter_to_longer_transfer={f'{a}_to_{b}': {method: transfer_test(chosen, a, b, exponent)
                for method, exponent in SCALINGS.items()} for a, b in ((14,28),(28,56),(56,84),(56,112))})
    return output


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=Path('output/contextual-training-response-20261020.json'))
    p.add_argument('--benchmark', type=Path, default=Path('output/empirical-training-response-benchmark-20261020.json'))
    p.add_argument('--consistent', type=Path, default=Path('output/consistent-training-response-20261020.json'))
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    source = json.loads(args.input.read_text())
    base = json.loads(args.benchmark.read_text())
    consistent = json.loads(args.consistent.read_text())
    if len({source['as_of'], base['as_of'], consistent['as_of']}) != 1:
        raise ValueError('Forecast, benchmark and consistent-fit snapshots must share one as_of')
    # Reconstruct the current v3 midpoint from its saved exact fits, not the
    # rounded UI number and not the spread between scenario equations.
    scenario_values = []
    for data in consistent['policies'].values():
        for model in ('calendar', 'log_miles'):
            fitted = data['final_consistent_fits'][model]
            exposure = base['may_horizon_days']/28 if model == 'calendar' else math.log1p(
                base['scenarios']['calendar']['goal'][-1]['completed_miles']/base['era_miles'])
            key = 'elapsed' if model == 'calendar' else model
            scenario_values.append(base['current_pace']*math.exp(fitted['coefficients'][key]*exposure))
    pace = (min(scenario_values)+max(scenario_values))/2
    rows = [r for r in source['policies'][source['primary_policy']]['predictions'] if r['method'] == 'current/v3_midpoint']
    output = dict(version='pace-outlook-error-margin-v1', as_of=source['as_of'],
        target_date=base['scenarios']['calendar']['goal'][-1]['date'], target_days=base['may_horizon_days'],
        central_pace=pace, method='Empirical held-out log(actual/predicted) residuals of the exact current v3 midpoint',
        results=audit(rows, base['may_horizon_days'], pace),
        protocol=['No scenario spread used to estimate error',
            'Signed residual bands and centered absolute-error margins both reported',
            'Actual weighted elapsed days, not nominal horizon labels, scale each error',
            'No bias correction silently changes the central forecast',
            'Constant, sqrt-time and persistent-rate error assumptions shown separately, not averaged',
            'Continued-consistency calibration and interrupted/illness context stress errors remain separate',
            'Short-to-long transfer tests embargo calibration errors after each test origin'],
        limitations=['211-day errors are extrapolated, not observed or calibrated May coverage',
            'Empirical 10th/90th percentiles describe historical residuals, not a certified probability interval',
            'Few independent consistent outcomes; shared benchmark/correction history remains',
            'Error persistence and future training beyond observed volumes are not identified',
            'Empirical power uses matched mixed-context errors; transferring it to consistent-training errors is an extrapolation assumption',
            'Single-run variability is additional; this predicts the comparable-HR aerobic benchmark'],
        live_forecast_changed=False, planner_changed=False, database_changed=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')
    private_file(args.output)
    compact = dict(central_pace=pace, target_days=output['target_days'],
        results={scope: dict(matched=data['matched_14_28_growth'],
            combined_margin=data['combined_short_error_projection'],
            transfers={pair: {m: v['all'] for m,v in methods.items()} for pair,methods in data['shorter_to_longer_transfer'].items()})
            for scope,data in output['results'].items()})
    print(json.dumps(compact, indent=2))


if __name__ == '__main__':
    main()
