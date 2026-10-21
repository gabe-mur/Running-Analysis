"""Re-score smoothing forecasts against actual corrected FUTURE runs first.

Primary target is the same weighted mean of actual future-run log pace for
every window, not any candidate's fitted benchmark. Also reconstruct forecasts
at each future run's exact date/dose, removing repeats across horizon windows.
No forecast is selected by its May pace or interval width. Offline only.
"""
from argparse import ArgumentParser
import json
import math
from pathlib import Path

from run_analysis.privacy import private_file
from scripts.audit_contextual_training_response import sequential_blocks
from scripts.audit_empirical_training_response import disjoint, score
from scripts.audit_pace_outlook_error_margin import interval, matched_growth, transfer_test
from scripts.audit_pace_smoothing_forecast import WINDOWS, fit_at, forecast, prefix_levels, rebuild


def actual_target(row):
    """Old benchmark cannot enter primary scores or residual calibration."""
    return {**row, 'benchmark_actual': row['fixed_benchmark'], 'actual': row['raw_target'],
        'log_error': math.log(row['predicted']/row['raw_target'])}


def subsets(rows):
    return {'all': rows, 'outcome_disjoint': disjoint(rows), 'sequential': sequential_blocks(rows)}


def summarize(rows):
    result = {}
    baseline = {(r['origin'],r['target_id']):r for r in rows if r['bandwidth']==28 and r['mode']=='joint'}
    for scope,horizon,regime,latest in (
            ('consistent_all',None,'consistent',False), ('consistent_14',14,'consistent',False),
            ('consistent_28',28,'consistent',False), ('latest_consistent',None,'consistent',True),
            ('all_contexts',None,None,False), ('all_56',56,None,False), ('all_84',84,None,False)):
        result[scope] = {}
        for days in WINDOWS:
            result[scope][str(days)] = {}
            for mode in ('joint','rate_only','anchor_only','unchanged_anchor'):
                chosen = [r for r in rows if r['bandwidth']==days and r['mode']==mode
                    and (r['origin'],r['target_id']) in baseline
                    and (horizon is None or r['horizon']==horizon) and (regime is None or r['regime']==regime)
                    and (not latest or r['origin']>='2026-09-01')]
                result[scope][str(days)][mode] = {}
                for subset,items in subsets(chosen).items():
                    b = [baseline[r['origin'],r['target_id']] for r in items]
                    unchanged = [{**r,'predicted':r['persistence'],
                        'log_error':math.log(r['persistence']/r['actual'])} for r in items]
                    result[scope][str(days)][mode][subset] = dict(candidate=score(items),baseline=score(b),unchanged=score(unchanged))
    return result


def individual_runs(policies, snapshot):
    """Exact-date/dose predictions; each origin/run pair appears only once.

    Cohort/horizon labels come from the nearest parent horizon's unchanged
    training/health classification. They are conditional diagnostics, not
    contemporaneously known future health. Parent-window future-run means
    remain the primary target for the typical-pace May forecast.
    """
    raw = {r['activity_id']:r['pace_min_mile'] for r in snapshot['rows']}
    pairs = {}
    for row in policies[28]['gap7_fraction0.500']:
        for run_id,exposure in zip(row['outcome_ids'],row['training_by_outcome']):
            key = (row['origin'],run_id)
            rank = (abs(exposure['days']-row['horizon']),row['horizon'])
            if key not in pairs or rank<pairs[key][0]:
                pairs[key] = (rank,row,exposure)
    output = []
    indexed = {d:{(r['origin'],r['target_id']):r for r in policies[d]['gap7_fraction0.500']} for d in WINDOWS}
    for origin in sorted({o for o,i in pairs}):
        fits = {d:fit_at(policies[d],origin) for d in WINDOWS}
        if any(f is None for f in fits.values()):
            continue
        for (o,run_id),(rank,parent,exposure) in pairs.items():
            if o!=origin:
                continue
            direct = dict(elapsed=exposure['days']/28,
                log_miles=math.log1p(exposure['total_miles']/parent['origin_cumulative_miles']))
            for days in WINDOWS:
                row = indexed[days][origin,parent['target_id']]
                for mode in ('joint','rate_only','anchor_only','unchanged_anchor'):
                    start = parent['start_pace'] if mode=='rate_only' else row['start_pace']
                    prediction = start if mode=='unchanged_anchor' else forecast(
                        fits[28] if mode=='anchor_only' else fits[days],direct,start)
                    output.append(dict(origin=origin,origin_id=parent['origin_id'],target_id=str(run_id),
                        outcome_ids=[run_id],outcome_end=exposure['at'],horizon=parent['horizon'],
                        elapsed=direct['elapsed'],regime=parent['consistency']['regime'],bandwidth=days,mode=mode,
                        actual=raw[run_id],predicted=prediction,persistence=parent['start_pace'],
                        log_error=math.log(prediction/raw[run_id]),
                        latest_training_outcome=max(f['latest_outcome'] for p in fits[days].values() for f in p.values())))
    return output


def margins(rows, final, horizon):
    baseline = [r for r in rows if r['bandwidth']==28 and r['mode']=='joint']
    observed = matched_growth(baseline)
    exponents = {'sqrt_time': .5,'constant_error': 0.,'persistent_rate_error': 1.}
    if observed['available']:
        exponents['actual_future_matched_power'] = observed['exponent']
    output = {}
    for scope in ('continued_consistency','all_contexts'):
        output[scope] = {}
        for days in WINDOWS:
            output[scope][str(days)] = {}
            for mode in ('joint','rate_only','anchor_only'):
                chosen = [r for r in rows if r['bandwidth']==days and r['mode']==mode and r['horizon'] in {14,28}
                    and (scope=='all_contexts' or r['regime']=='consistent')]
                center = final[str(days)]['may_rate_only' if mode=='rate_only' else 'may_joint']
                if mode=='anchor_only':
                    center = final['28']['may_joint']*final[str(days)]['current_pace']/final['28']['current_pace']
                output[scope][str(days)][mode] = {method:dict(
                    same_center=interval(chosen,days=horizon,pace=final['28']['may_joint'],exponent=p),
                    candidate_center=interval(chosen,days=horizon,pace=center,exponent=p),
                    outcome_disjoint=interval(disjoint(chosen),days=horizon,pace=center,exponent=p))
                    for method,p in exponents.items()}
    return dict(observed_actual_future_error_growth=observed,fixed_exponents=exponents,projections=output,
        transfer_checks={f'{a}_to_{b}':{label:transfer_test(baseline,a,b,p) for label,p in exponents.items()
            if label in {'sqrt_time','constant_error','persistent_rate_error'}} for a,b in ((14,28),(28,56))})


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=Path('output/pace-smoothing-forecast-20261020.json'))
    p.add_argument('--snapshot',type=Path,default=Path('output/pace-smoothing-input-20261020.json'))
    p.add_argument('--consistent',type=Path,default=Path('output/consistent-training-response-20261020.json'))
    p.add_argument('--benchmark',type=Path,default=Path('output/empirical-training-response-benchmark-20261020.json'))
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    source,snapshot,consistent,base = (json.loads(path.read_text()) for path in (args.input,args.snapshot,args.consistent,args.benchmark))
    if len({x['as_of'] for x in (source,snapshot,consistent,base)})!=1:
        raise ValueError('Input snapshots must share as_of')
    rows = [actual_target(r) for r in source['predictions']]
    policies = {d:{policy:rebuild(data['examples'],prefix_levels(snapshot,d))
        for policy,data in consistent['policies'].items()} for d in WINDOWS}
    individual = individual_runs(policies,snapshot)
    output = dict(version='future-run-smoothing-v1',as_of=source['as_of'],
        primary_target='Weighted mean of actual corrected future-run log pace, identical observations and weights for every candidate',
        secondary_target='Each actual corrected future run at its exact elapsed time and completed exposure; unique origin/run pair',
        predictions=rows,summary=summarize(rows),individual_predictions=individual,individual_summary=summarize(individual),
        final_sensitivity=source['final_sensitivity'],target_date=base['scenarios']['calendar']['goal'][-1]['date'],
        target_days=base['may_horizon_days'],margins=margins(rows,source['final_sensitivity'],base['may_horizon_days']),
        primary_margin_assumption='Fixed sqrt-time error growth for comparable working margins; actual-run-derived exponent is separate unvalidated sensitivity',
        protocol=['Same existing forecasts, genuinely different primary target and error calibration',
            'Actual future-run observations have fixed corrections but no temporal benchmark smoothing',
            'Parent outcome weights and completed-dose alignment are unchanged across windows',
            'No window selected by May point forecast or interval width',
            'Counterfactual smoothing and fit prefixes never receive later outcome information',
            'Hold error-growth exponents fixed across windows; actual-run-derived growth is separate sensitivity',
            'No benchmark error enters primary scores or May residual calibration'],
        limitations=['Training/health cohort is conditional on future training, not known at prediction origin',
            'Outcome windows and historical corrections overlap; disjoint runs do not make benchmark histories independent',
            'Only two disjoint consistent four-week intervals; long-horizon cases are health affected',
            'Individual-run error contains day-to-day variation, separate from typical-pace forecast uncertainty',
            'Every May band extrapolates shorter-horizon error behavior; no calibrated May coverage',
            'Empirical error-power fitted to mixed contexts cannot establish consistent-training long-term error persistence'],
        live_forecast_changed=False,planner_changed=False,database_changed=False)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    private_file(args.output)
    compact = dict(growth=output['margins']['observed_actual_future_error_growth'],
        primary={s:{d:{m:round(v['all']['candidate']['mae_seconds'],3) for m,v in methods.items()}
            for d,methods in output['summary'][s].items()} for s in ('consistent_all','consistent_28','all_56','all_84')},
        individual={s:{d:round(methods['joint']['all']['candidate']['mae_seconds'],3)
            for d,methods in output['individual_summary'][s].items()} for s in ('consistent_all','all_contexts')})
    print(json.dumps(compact,indent=2))


if __name__=='__main__':
    main()
