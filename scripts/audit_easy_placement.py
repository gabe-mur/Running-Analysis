"""Exhaustive fixed-dose easy-date census, not a physiological optimizer."""
from dataclasses import replace
from datetime import date, timedelta
from itertools import combinations, islice
from math import comb
from pathlib import Path
from time import perf_counter
import json

import numpy as np

from scripts.audit_easy_distribution import exposures, panel as distribution_panel, program_option
from scripts.audit_recency_demand import json_ready
from run_analysis.easy_dose import assess_easy_dose
from run_analysis.program_constraints import assess_program_constraints
from run_analysis.recency_demand import CHANNELS, DemandEvent, DemandTerm, compare_recency
from run_analysis.workload_accounting import Quantity

ORIGIN = date(2026,9,25)
TARGETS = tuple(sorted([(7*c+o, role) for c in range(6) for o,role in ((3,'long'),(6,'quality'))]))
COORDINATES = tuple((target, start) for target,_ in TARGETS for start in range(target+1))
ENDS = np.array([min(t,28) for t,_ in COORDINATES])
STARTS = np.array([min(s,28) for _,s in COORDINATES])
GROUPS = {name: np.array([test(t) for t,_ in COORDINATES]) for name,test in {
    'all': lambda t: True, 'main_long': lambda t: t < 28 and t % 7 == 3,
    'main_quality': lambda t: t < 28 and t % 7 == 6, 'tail': lambda t: t >= 28}.items()}
LABELS = ('equal','candidate_no_greater','candidate_no_less','crossing')


def curves(doses):
    """Exact cumulative easy mileage in integer tenths, at fixed receiving dates."""
    doses = np.asarray(doses)
    if doses.ndim != 2 or doses.shape[1] != 28 or np.any(doses < 0) or np.any(doses != doses.astype(int)):
        raise ValueError('Nonnegative integer-tenth exposure over exactly 28 days required')
    prefix = np.concatenate((np.zeros((len(doses),1),dtype=np.int64),np.cumsum(doses,axis=1)),axis=1)
    return prefix[:,ENDS]-prefix[:,STARTS]


def classify(delta):
    lower, higher = np.any(delta < 0,axis=1), np.any(delta > 0,axis=1)
    return lower.astype(int)+2*higher.astype(int)


def dose_vector(days, count):
    if count not in (4,8) or len(days) != count or len(set(days)) != count or any(
            type(d) is not int or not 0 <= d < 28 or d % 7 in (3,6) for d in days):
        raise ValueError('Unique declared-family easy dates outside fixed key dates required')
    values = np.zeros(28,dtype=np.int64)
    values[list(days)] = 176//count
    return values


def records(count, days):
    dose_vector(days,count)
    fixed = [e for e in exposures(count,ORIGIN) if e.role != 'easy' or e.day >= ORIGIN+timedelta(days=28)]
    templates = [e for e in exposures(count,ORIGIN) if e.role == 'easy' and e.day < ORIGIN+timedelta(days=28)]
    moved = [replace(e,key=f'easy:{day}',day=ORIGIN+timedelta(days=day)) for e,day in zip(templates,sorted(days))]
    return tuple(sorted((*fixed,*moved),key=lambda e:(e.day,e.key)))


def census(count, *, batch_size=2048):
    if count not in (4,8) or type(batch_size) is not int or batch_size <= 0:
        raise ValueError('Declared family and positive batch size required')
    allowed = [d for d in range(28) if d % 7 not in (3,6)]
    baselines = {n:tuple((e.day-ORIGIN).days for e in exposures(n,ORIGIN)
        if e.role == 'easy' and e.day < ORIGIN+timedelta(days=28)) for n in (4,8)}
    baseline_curves = {n:curves([dose_vector(days,n)])[0] for n,days in baselines.items()}
    histograms = {n:{g:np.zeros(4,dtype=np.int64) for g in GROUPS} for n in baselines}
    witness, witness_rank = {}, {}
    total = comb(len(allowed),count)
    sample_indices = {0,total//4,total//2,3*total//4,total-1}
    samples, processed = [], 0
    iterator = combinations(allowed,count)
    while batch := list(islice(iterator,batch_size)):
        doses = np.zeros((len(batch),28),dtype=np.int64)
        doses[np.arange(len(batch))[:,None],np.asarray(batch)] = 176//count
        values = curves(doses)
        for n,reference in baseline_curves.items():
            delta = values-reference
            for group,mask in GROUPS.items():
                codes = classify(delta[:,mask])
                histograms[n][group] += np.bincount(codes,minlength=4)
                if group != 'all':
                    continue
                for i in np.flatnonzero(codes == 1):
                    days = batch[i]
                    rank = (len(set(days)^set(baselines[n])),days)
                    if n not in witness_rank or rank < witness_rank[n]:
                        witness_rank[n] = rank
                        witness[n] = days
        for i,days in enumerate(batch):
            if processed+i in sample_indices:
                samples.append(days)
        processed += len(batch)
    assert processed == total
    return dict(easy_count=count,calendars_enumerated=processed,
        comparisons={str(n):{g:dict(zip(LABELS,map(int,values))) for g,values in groups.items()}
                     for n,groups in histograms.items()},
        nearest_demand_dominance_witness={str(n):days for n,days in witness.items()},
        canonical_validation_samples=samples)


def panel(directory=Path('output'), *, batch_size=2048):
    started = perf_counter()
    baseline = distribution_panel(directory)
    recency = json.loads((directory/'recency-demand-panel-20260925.json').read_text())
    q = lambda value: Quantity(**value) if value is not None else None
    history = tuple(DemandEvent('actual:'+r['key'],date.fromisoformat(r['day']),tuple(
        DemandTerm(c,f'actual:{r["key"]}:{c}',1.,q(r['quantities'][c])) for c in CHANNELS),
        r['accounting_key'],r['role'],tuple(r['structure']) if r['structure'] else None,'observed')
        for r in recency['sessions'])
    # Reduction is valid only with shared, finite positive changing demand rates.
    for channel in ('running_minutes','cardio'):
        bounds = q(recency['rates']['easy'][channel]['bounds'])
        if bounds is None or bounds.low <= 0:
            raise ValueError('Exact common-rate sign reduction is unsupported for these inputs')
    refs = {n:program_option(n,exposures(n,ORIGIN),history,recency['rates']) for n in (4,8)}
    results = {str(n):census(n,batch_size=batch_size) for n in (4,8)}
    checks = []
    def validate(count,days,reference_count):
        candidate = program_option('candidate',records(count,days),history,recency['rates'])
        reference = refs[reference_count]
        ref_days = tuple((e.day-ORIGIN).days for e in reference.exposures if e.role == 'easy' and e.day < ORIGIN+timedelta(days=28))
        delta = curves([dose_vector(days,count)])[0]-curves([dose_vector(ref_days,reference_count)])[0]
        outcomes = []
        for day,role in TARGETS:
            target = next(e for e in candidate.demands if e.day == ORIGIN+timedelta(days=day))
            before = next(e for e in reference.demands if e.key == target.key)
            a = (*history,*(e for e in reference.demands if e.day < before.day))
            b = (*history,*(e for e in candidate.demands if e.day < target.day))
            slow = compare_recency(a,before,b,target)
            mask = np.array([t == day for t,_ in COORDINATES])
            code = int(classify(delta[None,mask])[0])
            expected = ('equal','right_no_greater','left_no_greater','no_joint_order')[code]
            if slow['recency_demand_order'] != expected:
                raise AssertionError('Reduced comparison disagrees with full original-history accounting')
            outcomes.append(dict(day=day,role=role,tail=day >= 28,order=LABELS[code]))
        constraints = assess_program_constraints(candidate.demands,history,
            as_of=baseline['as_of'],plan_complete=True)
        policy = baseline['policy']
        floors = {name:all(assess_easy_dose(e.miles,ref,policy)['passes_planned_minimum']
                    for e in candidate.exposures if e.role == 'easy') for name,ref in baseline['references'].items()}
        easy_comparisons = []
        if count == reference_count:
            # Equal-dose, equal-count opportunities paired in chronological order.
            # Receiving dates can change here: use the full canonical comparator,
            # NOT the fixed-target cancellation used in the exhaustive census.
            def easy_main(option):
                return sorted((e for e in option.demands if e.purpose == 'easy'
                    and e.day < ORIGIN+timedelta(days=28)),key=lambda e:e.day)
            for before,target in zip(easy_main(reference),easy_main(candidate)):
                a = (*history,*(e for e in reference.demands if e.day < before.day))
                b = (*history,*(e for e in candidate.demands if e.day < target.day))
                report = compare_recency(a,before,b,target)
                label = {'equal':'equal','left_no_greater':'candidate_no_less',
                         'right_no_greater':'candidate_no_greater','no_joint_order':'crossing'}[report['recency_demand_order']]
                easy_comparisons.append(dict(previous_day=(before.day-ORIGIN).days,
                    candidate_day=(target.day-ORIGIN).days,order=label,
                    same_count_ordered_pairing_not_cross_frequency_equivalence=True))
        return dict(easy_count=count,easy_days=days,reference_count=reference_count,
            canonical_comparison_matches=True,constraints=constraints['status'],aerobic_floor_passes=floors,
            receiving_opportunities=outcomes,easy_opportunity_comparisons=easy_comparisons)
    for n,result in results.items():
        requested = {(tuple(days),ref) for days in result['canonical_validation_samples'] for ref in (4,8)}
        requested.update((tuple(days),int(ref)) for ref,days in result['nearest_demand_dominance_witness'].items())
        for days,ref in sorted(requested):
            checks.append(validate(int(n),days,ref))
    return dict(as_of=baseline['as_of'],protocol='fixed_key_doses_all_easy_date_placements_v1',
        original_history_count=len(history),batch_size=batch_size,results=results,
        canonical_checks=checks,canonical_opportunities_checked=sum(len(c['receiving_opportunities']) for c in checks),
        canonical_easy_opportunities_checked=sum(len(c['easy_opportunity_comparisons']) for c in checks),
        elapsed_seconds=perf_counter()-started,planner_influence=False,physiological_optimum_established=False,
        limitations=['Exact enumeration only within the four- and eight-easy-session equal-dose families',
            'Fixed key dates/doses and shared role-rate scenario are necessary for this reduction',
            'Demand dominance is not training benefit, own-easy recovery or whole-program suitability',
            'Same-day double runs excluded; no protected dates supplied in this experiment',
            'Common tail bounds the audit but does not establish infinite-horizon or daily-replanning stability'])


if __name__ == '__main__':
    print(json.dumps(panel(),default=json_ready,allow_nan=False,indent=2))
