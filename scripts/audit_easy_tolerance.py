"""Bounded dose-policy falsification panel; never approves a whole program."""
from collections import Counter
from argparse import ArgumentParser
from dataclasses import replace
from datetime import date, timedelta
from itertools import product
from pathlib import Path
from time import perf_counter
import json

from scripts.audit_easy_distribution import exposures, panel as distribution_panel, program_option
from scripts.audit_easy_placement import ORIGIN, records
from scripts.audit_recency_demand import json_ready
from scripts.easy_tolerance_trial import EasyTolerancePolicy, PriorExposureReference, assess, quantity
from run_analysis.program_constraints import assess_program_constraints
from run_analysis.recency_demand import CHANNELS, DemandEvent, DemandTerm, compare_recency
from run_analysis.session_decision import CurrentContext
from run_analysis.workload_accounting import Quantity

FRACTIONS = (.5, .65, .8)


def inputs(directory=Path('output')):
    base = distribution_panel(directory)
    raw = json.loads((directory/'recency-demand-panel-20260925.json').read_text())
    history = tuple(DemandEvent('actual:'+r['key'], date.fromisoformat(r['day']), tuple(
        DemandTerm(c, f'actual:{r["key"]}:{c}', 1., Quantity(**r['quantities'][c])
                   if r['quantities'][c] else None) for c in CHANNELS),
        r['accounting_key'], r['role'], tuple(r['structure']) if r['structure'] else None, 'observed')
        for r in raw['sessions'])
    refs = []
    for role in ('easy', 'long', 'quality'):
        rows = [r for r in raw['sessions'] if r['eligible'] and r['role'] == role
                and ORIGIN-timedelta(days=28) <= date.fromisoformat(r['day']) < ORIGIN]
        ceilings = []
        for channel in CHANNELS:
            eligible = [r for r in rows if r['quantities'][channel] is not None
                        and (channel != 'work_minutes' or r['structure'])]
            if eligible:
                ceilings.append((channel, max(r['quantities'][channel]['high'] for r in eligible)))
        if ceilings:
            refs.append(PriorExposureReference(role, tuple(ceilings), tuple(r['key'] for r in rows)))
    return base, raw['rates'], history, tuple(refs)


def policy_for(base, refs, fraction=.65, startup='central'):
    # The 4.4 command was frozen before this experiment, not inferred from candidates.
    return EasyTolerancePolicy(base['references'][startup], base['policy'], 4.4,
                               fraction, refs, 'declared_reduction_sensitivity_not_fitted')


def easy_checks(option, history, policy, as_of):
    return [assess(e, (*history, *(p for p in option.demands if p.day < e.day)), policy,
                   as_of=as_of) for e in option.demands if e.purpose == 'easy']


def describe(rows):
    main = [e for e in rows if ORIGIN <= e.day < ORIGIN+timedelta(days=28)]
    def peak(span, value):
        return max(sum(value(e) for e in main if day-timedelta(days=span-1) <= e.day <= day)
                   for day in (ORIGIN+timedelta(days=d) for d in range(28)))
    days = {e.day for e in main}
    longest = streak = 0
    for d in range(28):
        streak = streak+1 if ORIGIN+timedelta(days=d) in days else 0
        longest = max(longest, streak)
    return dict(miles=sum(e.miles for e in main), counts=dict(Counter(e.role for e in main)),
        maximum_consecutive_run_dates=longest,
        rolling_peaks={str(span):dict(miles=peak(span, lambda e: e.miles),
            quality_work_minutes=peak(span, lambda e: sum(e.work_bout_seconds or ())/60)) for span in (3,7)},
        interpretation='projected_main_interval_descriptions_not_recovery_limits')


def key_comparisons(reference, candidate, history):
    comparisons = []
    for target in candidate.demands:
        if target.purpose not in {'long', 'threshold'}:
            continue
        before = next(e for e in reference.demands if e.key == target.key)
        result = compare_recency((*history, *(e for e in reference.demands if e.day < before.day)), before,
            (*history, *(e for e in candidate.demands if e.day < target.day)), target)
        comparisons.append(dict(key=target.key, reference_day=before.day, candidate_day=target.day,
                                order=result['recency_demand_order'], channels=result['channel_orders']))
    return comparisons


def prefix_peaks(history, option):
    """Include actual lead-in; do not treat the first planned date as a reset."""
    ledger = (*history, *option.demands)
    peaks = {}
    for span in (3,7):
        rows = []
        for offset in range(28):
            day = ORIGIN+timedelta(days=offset)
            recent = [e for e in ledger if day-timedelta(days=span-1) <= e.day <= day]
            miles = [quantity(e,'miles') for e in recent]
            if any(q is None for q in miles):
                raise ValueError('This frozen descriptive panel needs known distance bounds')
            rows.append(dict(end=day, miles_low=sum(q.low for q in miles),
                miles_high=sum(q.high for q in miles),
                detected_or_prescribed_quality_work_minutes=sum(sum(e.structure or ())/60
                    for e in recent if e.purpose in {'quality','threshold'})))
        peaks[str(span)] = dict(miles_low=max(r['miles_low'] for r in rows),
            miles_high=max(r['miles_high'] for r in rows),
            detected_or_prescribed_quality_work_minutes=max(
                r['detected_or_prescribed_quality_work_minutes'] for r in rows))
    return dict(peaks=peaks, includes_original_preplan_runs=True,
        work_interpretation='known_quality_bouts_only_not_all_high_intensity_or_muscular_demand')


def stacked_records():
    """Same volume and session doses, intentionally compressed quality placement."""
    destinations = {6: 1, 13: 2, 20: 4}
    return tuple(sorted((replace(e, day=ORIGIN+timedelta(days=destinations[(e.day-ORIGIN).days]))
        if e.role == 'quality' and (e.day-ORIGIN).days in destinations else e
        for e in exposures(8, ORIGIN)), key=lambda e: e.day))


def panel(directory=Path('output')):
    started = perf_counter()
    base, rates, history, refs = inputs(directory)
    as_of = base['as_of']
    original = program_option('original_eight', exposures(8, ORIGIN), history, rates)
    policy = policy_for(base, refs)
    witness_days = (0,4,7,12,14,19,21,26)
    witness = program_option('witness', records(8,witness_days), history, rates)
    scenarios = {}
    for startup in base['references']:
        for fraction in FRACTIONS:
            p = policy_for(base, refs, fraction, startup)
            key = f'{startup}:{fraction}'
            improvements = []
            for moves in product((0,1), repeat=4):
                if not any(moves):
                    continue
                days = tuple(d for cycle in range(4) for d in (cycle*7, cycle*7+5-moves[cycle]))
                option = program_option(str(days), records(8,days), history, rates)
                checks = easy_checks(option, history, p, as_of)
                # Common tail contains ordinary easy runs the day after quality.
                # Report it separately; never hide its policy mismatch.
                improvements.append(dict(days=days,
                    main_statuses=dict(Counter(c['status'] for c in checks if c['day'] < ORIGIN+timedelta(days=28))),
                    tail_statuses=dict(Counter(c['status'] for c in checks if c['day'] >= ORIGIN+timedelta(days=28)))))
            distributions = {}
            for count in (2,4,8,12,16,20):
                option = program_option(str(count), exposures(count,ORIGIN), history,rates)
                checks = easy_checks(option, history, p, as_of)
                distributions[str(count)] = dict(main_statuses=dict(Counter(c['status'] for c in checks
                    if c['day'] < ORIGIN+timedelta(days=28))), description=describe(option.exposures))
            scenarios[key] = dict(improvements=improvements, distributions=distributions)

    # Today's easy immediately follows the original Sept 24 interval workout.
    today = next(e for e in original.demands if e.day == ORIGIN)
    def today_dose(miles):
        return next(e for e in program_option('today', (replace(original.exposures[0], miles=miles,
            easy_miles=miles),), history, rates).demands)
    controls = {}
    for name, context, unexpected in (
            ('unknown_response', None, False),
            ('usual', CurrentContext(ORIGIN,as_of,'usual','synthetic_control'),False),
            ('mild_fatigue',CurrentContext(ORIGIN,as_of,'little_tired','synthetic_control'),False),
            ('health_concern',CurrentContext(ORIGIN,as_of,'health_concern','synthetic_control'),False),
            ('unexpected_response_without_current_check',None,True),
            ('future_report_ignored',CurrentContext(ORIGIN,as_of+timedelta(hours=1),'health_concern','synthetic_control'),False)):
        controls[name] = assess(today,history,policy,as_of=as_of,context=context,
                                unexpected_previous_response=unexpected)
    for name,miles in [('token',.8),('ordinary_after_quality',4.4),('large_easy_after_quality',8.8)]:
        controls[name] = assess(today_dose(miles),history,policy,as_of=as_of)
    # Policy fractions change the boundary, not the committed distances.
    dose_sensitivity = {str(f):{str(m):assess(today_dose(m),history,
        policy_for(base,refs,f),as_of=as_of)['status'] for m in (2.2,2.3,2.8,2.9,3.5,3.6,4.4)} for f in FRACTIONS}
    excessive = replace(history[-1], terms=tuple(replace(t,coefficient=t.coefficient*2) for t in history[-1].terms))
    controls['double_previous_dose'] = assess(today,(*history[:-1],excessive),policy,as_of=as_of)
    controls['ordinary_after_one_rest_date'] = assess(replace(today_dose(4.4),day=ORIGIN+timedelta(days=1)),
        history,policy,as_of=as_of)
    old_health = CurrentContext(ORIGIN-timedelta(days=1),as_of-timedelta(days=1),'health_concern','synthetic_control')
    controls['unresolved_health_after_rollover'] = assess(today,history,policy,as_of=as_of,context=old_health)

    stacked = program_option('stacked_quality',stacked_records(),history,rates)
    reports = {}
    for option in (original,witness,stacked):
        checks = easy_checks(option,history,policy,as_of)
        reports[option.key] = dict(easy_checks=checks,description=describe(option.exposures),
            actual_plus_planned_rolling_peaks=prefix_peaks(history,option),
            constraints=assess_program_constraints(option.demands,history,as_of=as_of,plan_complete=True),
            key_comparisons=key_comparisons(original,option,history),
            calendar=[dict(key=e.key,day=e.day,miles=e.miles,role=e.role) for e in option.exposures])
    return dict(protocol='dose_specific_easy_permission_falsification_v1',as_of=as_of,
        original_history_count=len(history), references=refs, policy=policy, scenarios=scenarios,
        controls=controls,dose_sensitivity=dose_sensitivity,programs=reports,
        elapsed_seconds=perf_counter()-started, planner_influence=False,program_approved=False,
        findings_are_policy_sensitivity_not_physiological_validation=True,
        unresolved=['Local dose permission cannot determine whole-program recovery or best frequency',
            'Next-date adjacency is a policy boundary, not a measured recovery transition',
            'Key session suitability and accumulated multi-session response remain unresolved',
            'No daily/upload replay, prospective outcomes, or selected reduction fraction'])


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = panel()
    encoded = json.dumps(report,default=json_ready,allow_nan=False,indent=2)
    if args.output:
        args.output.write_text(encoded+'\n')
        print(json.dumps(dict(output=str(args.output),elapsed_seconds=report['elapsed_seconds'],
                              program_approved=report['program_approved'])))
    else:
        print(encoded)
