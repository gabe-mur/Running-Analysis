"""Test short-horizon error scaling against longer actual-run forecasts.

No live changes. Same-origin descriptive horizon holdouts are distinct from
strict chronological calibration: the latter uses only errors available before
the original forecast origin, and estimates its own exponent from that prefix.
The previously quoted 0.21 is a hypothesis control, not a pre-registered or
historically available estimate. Interrupted and consistent cases stay separate.
"""
from argparse import ArgumentParser
from collections import Counter
import json
import math
from pathlib import Path

import numpy as np

from run_analysis.privacy import private_file
from scripts.audit_contextual_training_response import sequential_blocks
from scripts.audit_pace_outlook_error_margin import interval, matched_growth, summary, weights, weighted_quantile

FIXED_POWERS = {'constant':0., 'quoted_slow_hypothesis':.21, 'sqrt_time':.5, 'persistent_rate':1.}
LONG_HORIZONS = (56,84,112)


def descriptive(rows):
    """Matched origin composition. Long outcomes never fit the short power."""
    result = {}
    for horizon in LONG_HORIZONS:
        by_h = {h:{r['origin']:r for r in rows if r['horizon']==h} for h in (14,28,horizon)}
        common = set.intersection(*(set(v) for v in by_h.values()))
        if len(common)<2:
            result[str(horizon)] = dict(available=False,origins=len(common))
            continue
        short = [by_h[h][o] for o in sorted(common) for h in (14,28)]
        future = [by_h[horizon][o] for o in sorted(common)]
        learned = matched_growth(short)['exponent']
        stats = {str(h):summary([by_h[h][o] for o in sorted(common)]) for h in (14,28,horizon)}
        # Scale each same-origin four-week residual to its own actual longer
        # elapsed time. This predicts error magnitude, not calibrated coverage.
        errors = np.array([abs(r['log_error']) for r in future])
        predicted = {}
        for method,power in {**FIXED_POWERS,'matched_short_fit':learned}.items():
            scaled = np.array([abs(by_h[28][r['origin']]['log_error']) *
                (r['elapsed']/by_h[28][r['origin']]['elapsed'])**power for r in future])
            w = weights(future)
            predicted[method] = dict(exponent=power,predicted_rms=float(np.sqrt(w@(scaled*scaled))),
                actual_rms=float(np.sqrt(w@(errors*errors))),
                predicted_to_actual_rms=float(np.sqrt((w@(scaled*scaled))/(w@(errors*errors)))),
                predicted_absolute_80=float(weighted_quantile(scaled,w,[.8])[0]),
                actual_absolute_80=float(weighted_quantile(errors,w,[.8])[0]))
        observed = matched_growth([*short,*future],first=28,second=horizon)
        result[str(horizon)] = dict(available=True,origins=len(common),stats=stats,
            short_fitted_power=learned,observed_28_to_long_power=observed['exponent'],
            horizon_predictions=predicted,regimes=dict(Counter(r['regime'] for r in future)),
            sequential_long_intervals=len(sequential_blocks(future)))
    return result


def coverage_summary(rows):
    if not rows:
        return dict(pairs=0)
    w = weights(rows)
    misses = np.array([max(0.,abs(r['log_error'])-r['radius']) for r in rows])
    return dict(pairs=len(rows),origins=len({r['origin'] for r in rows}),
        outcome_runs=len({i for r in rows for i in r['outcome_ids']}),
        centered_coverage=float(w@np.array([r['covered'] for r in rows])),
        mean_radius_log=float(w@np.array([r['radius'] for r in rows])),
        mean_width_seconds=float(w@np.array([r['width_seconds'] for r in rows])),
        mean_excess_error_log=float(w@misses),
        # Proper central 80% interval score: width plus 10x outside miss.
        mean_interval_score_log=float(w@(2*np.array([r['radius'] for r in rows])+10*misses)))


def chronological(rows,minimum_origins):
    predictions = []
    for target in (r for r in rows if r['horizon'] in LONG_HORIZONS):
        prior = [r for r in rows if r['horizon'] in {14,28} and r['outcome_end']<target['origin']]
        fit = matched_growth(prior)
        if not fit['available'] or fit['pairs']<minimum_origins:
            continue
        matched = set(r['origin'] for r in prior if r['horizon']==14)&set(r['origin'] for r in prior if r['horizon']==28)
        # Equal calibration cases for every power, paired horizons only.
        calibration = [r for r in prior if r['origin'] in matched]
        latest = max(r['outcome_end'] for r in calibration)
        for method,power in {**FIXED_POWERS,'prior_fitted':fit['exponent']}.items():
            band = interval(calibration,days=target['elapsed']*28,pace=target['predicted'],exponent=power)
            radius = band['absolute_error_80_log']
            predictions.append({**target,'method':method,'exponent':power,'radius':radius,
                'covered':abs(target['log_error'])<=radius,
                'width_seconds':60*(band['centered_high_pace']-band['centered_low_pace']),
                'latest_calibration':latest,'calibration_origins':fit['pairs'],
                'calibration_outcome_ids':sorted({i for r in calibration for i in r['outcome_ids']})})
    report = {}
    for label,h in [('all_long',None),*[(str(h),h) for h in LONG_HORIZONS]]:
        report[label] = {}
        for method in (*FIXED_POWERS,'prior_fitted'):
            chosen = [r for r in predictions if r['method']==method and (h is None or r['horizon']==h)]
            report[label][method] = dict(all=coverage_summary(chosen),sequential=coverage_summary(sequential_blocks(chosen)))
    return dict(predictions=predictions,summary=report)


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=Path('output/future-run-smoothing-20261020.json'))
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    source = json.loads(args.input.read_text())
    result = {}
    for days in (28,42,56):
        rows = [r for r in source['predictions'] if r['bandwidth']==days and r['mode']=='joint']
        result[str(days)] = {}
        for scope in ('consistent_only','all_recorded_contexts'):
            chosen = [r for r in rows if scope=='all_recorded_contexts' or r['regime']=='consistent']
            result[str(days)][scope] = dict(descriptive_horizon_holdouts=descriptive(chosen),
                chronological={str(n):chronological(chosen,n) for n in (2,4,8)})
    output = dict(version='long-horizon-error-growth-v1',as_of=source['as_of'],results=result,
        primary_target=source['primary_target'],
        protocol=['Same corrected actual future-run means; no smoothed benchmark outcome',
            'Same-origin horizon holdouts separate changing origin composition from actual error growth',
            'Every chronological power fit and residual calibration finishes strictly before the forecast origin',
            'All powers use identical shorter-error calibration cases and longer test cases',
            'Report minimum 2/4/8 matched-origin sensitivities, not just the most favorable support choice',
            'Quoted 0.21 is a post-hoc hypothesis control; prior_fitted estimates only from earlier errors',
            'Score coverage and width jointly; wider intervals do not automatically win',
            'No May projection or desired band width chooses the exponent'],
        limitations=['Few original long-forecast origins have enough matured calibration history',
            'All long outcomes are health affected under the existing cohort definition',
            'Same-origin longer errors are horizon-held-out but not an independent temporal validation',
            'Outcome overlap and shared retrospective corrections remain',
            'No 211-day error or healthy long-horizon coverage is directly observed'],
        live_forecast_changed=False,planner_changed=False,database_changed=False)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    private_file(args.output)
    print(json.dumps({d:{scope:dict(descriptive={h:dict(origins=v.get('origins'),
        short_power=v.get('short_fitted_power'),long_power=v.get('observed_28_to_long_power'),
        slow_ratio=v.get('horizon_predictions',{}).get('quoted_slow_hypothesis',{}).get('predicted_to_actual_rms'))
        for h,v in info['descriptive_horizon_holdouts'].items()},
        chronological={n:{m:v['all'] for m,v in r['summary']['all_long'].items()} for n,r in info['chronological'].items()})
        for scope,info in data.items()} for d,data in result.items()},indent=2))


if __name__=='__main__':
    main()
