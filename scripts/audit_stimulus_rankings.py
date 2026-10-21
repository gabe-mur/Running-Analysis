"""Fixed-calendar candidate ranking sensitivity, not scheduler/forecast validation.

Recorded responses supply five counterfactual quality-session shapes. Current
reference values are shared by ALL future sessions/alternatives in each scenario.
No independent best-case credit, re-search, adaptive weights or live writes.
"""
from argparse import ArgumentParser
from collections import Counter
from dataclasses import replace
from datetime import datetime, timedelta, time
from hashlib import sha256
from itertools import product
import json
from math import isfinite
from pathlib import Path
from statistics import median
from time import perf_counter
from zoneinfo import ZoneInfo

from run_analysis.config import load_config
from run_analysis.continuous_training_dose import ContinuousDoseReference
from run_analysis.movement import classify_movement
from run_analysis.performance_reference import AerobicReference, aerobic_speed
from run_analysis.projection_evidence import steady_pace_samples
from run_analysis.training_signals import HeartRateReference, observed_cardio_trace
from run_analysis.work_stimulus import StimulusTarget
from run_analysis.work_stimulus_evidence import observed_work_signals
from run_analysis.workload_accounting import Quantity
from scripts.audit_program_evidence_adapter import load_history
from scripts.audit_work_stimulus import encode
from scripts.continuous_program_adapter import with_continuous_trace
from scripts.program_evidence_adapter import QualityReference
from scripts.program_objective_trial import TrialPolicy, DemandScenario
from scripts.shared_program_evaluator import ProgramContext
from scripts.stimulus_program_adapter import evaluate_stimulus_program
from scripts.timely_purpose_trial import PurposeTimingPolicy


def ranking_summary(cases, *, tolerance=1e-10):
    if not isfinite(tolerance) or tolerance < 0:
        raise ValueError('Finite nonnegative tie tolerance required')
    if not cases:
        raise ValueError('Nonempty scenario set required')
    names = set(cases[0]['scores'])
    if len(names) < 2 or any(set(c['scores']) != names for c in cases):
        raise ValueError('Compare the same candidates in every scenario')
    winners, possible, gaps = Counter(), set(), []
    regret = {name: 0. for name in sorted(names)}
    for case in cases:
        scores = case['scores']
        if any(not isfinite(v) for v in scores.values()):
            raise ValueError('Finite comparable scores required')
        best = min(scores.values())
        tied = sorted(name for name, score in scores.items() if score-best <= tolerance)
        possible.update(tied)
        winners[' | '.join(tied)] += 1
        gaps.append(sorted(scores.values())[1]-best)
        for name in names:
            regret[name] = max(regret[name], scores[name]-best)
    return dict(scenarios=len(cases), observed_winner_sets=dict(winners),
        possible_winners=sorted(possible), winner_unchanged_on_sampled_scenarios=len(possible)==1,
        minimum_top_two_gap=min(gaps), maximum_top_two_gap=max(gaps),
        maximum_regret_by_candidate=regret,
        interpretation='Scenario counts are not probabilities; sampled stability is not a continuum guarantee')


def load_inventory(database, config, as_of, reference):
    hr = HeartRateReference(config['resting_hr'], config['max_hr'], 'configured audit reference')
    old = QualityReference('threshold', 'work_seconds', Quantity.point(900, 'discarded comparator'),
                           Quantity(290, 310), 'physical bridge validation only')
    originals = {}
    def collect(row, activity):
        intervals = classify_movement(activity.trackpoints, config['moving_time']).intervals
        trace = observed_cardio_trace(activity, intervals, hr, reference.policy,
                                     include_spans=True, compute_density=False)
        originals[row.key] = activity, intervals, trace
        return with_continuous_trace(row, trace['normalized_spans'], reference=reference,
            trace_key=row.key, trace_source='observed', elapsed_seconds=trace['elapsed_seconds'],
            outside_reference_seconds=trace['exclusions']['outside_reference_seconds'])[0]
    history, unresolved = load_history(database, config, as_of, old, run_transform=collect)
    if unresolved:
        raise ValueError(f'Unresolved inventory: {unresolved}')
    easy, signals = [], {}
    def freeze(at):
        prior = tuple((key, knots) for key, end, knots in easy if at-timedelta(days=84) <= end < at)
        return aerobic_speed(AerobicReference(at, reference.accounting_key, prior, 84., .25), .70)
    for row in sorted(history, key=lambda r:r.start):
        activity, intervals, trace = originals[row.key]
        if row.role == 'quality':
            prior = freeze(row.start)
            signal = observed_work_signals(row, activity, config, intervals, trace['normalized_spans'],
                easy_speed_mps=prior['speed_mps'], reference_at=row.start,
                reference_source='historical prior-only reference held fixed across future scenarios')
            if signal is None:
                raise ValueError('Quality donor lacks aligned original work evidence')
            signals[row.key] = signal
        if row.role == 'easy':
            knots = {}
            for s in steady_pace_samples(intervals, row.start, config, hr):
                knots.setdefault(s.mean_hr_reserve, []).append(s.meters/s.seconds)
            if knots:
                easy.append((row.key, row.end, tuple((r, median(v)) for r, v in sorted(knots.items()))))
    return history, signals, freeze(as_of)


def transplant(row, signals, *, key, start, issued_at, speed):
    future = replace(row, key=key, start=start, end=start+(row.end-row.start),
        available_at=issued_at, source='planned', miles=round(row.miles, 1),
        work_evidence_basis='prescribed' if row.bouts else row.work_evidence_basis)
    signal = signals.get(row.key)
    if signal is not None:
        signal = replace(signal, key=key, start=future.start, end=future.end, source='planned',
            available_at=issued_at, basis=future.work_evidence_basis,
            easy_speed_mps=Quantity.point(speed, 'one_frozen_reference_for_all_future_alternatives'),
            reference_at=issued_at, reference_source='current snapshot scenario; raw response is replayed, not predicted')
    return future, signal


def audit(database, config, as_of, *, grid_points=201):
    started = perf_counter()
    digest = sha256(database.read_bytes()).hexdigest()
    reference = ContinuousDoseReference('unchanged offline work-coverage reference')
    history, historical_signals, prior = load_inventory(database, config, as_of, reference)
    bounds = prior['speed_mps']
    if bounds is None or grid_points < 3:
        raise ValueError('Current reference and at least three grid points required')
    zone = ZoneInfo(config['timezone_default'])
    origin = datetime.combine(as_of.astimezone(zone).date(), time(), zone)
    donors = {role: max((r for r in history if r.role==role), key=lambda r:r.start)
              for role in ('easy','long','quality')}
    quality = tuple(r for r in history if r.role=='quality')
    easy = [r for r in history if r.role=='easy' and r.end >= as_of-timedelta(days=28)]
    scales = tuple(4.4*median((r.demand[i].low+r.demand[i].high)/2/r.miles for r in easy)
                   for i in (0,1))+(15.,)
    context = ProgramContext((), origin.date(), origin.date(), (origin+timedelta(days=21)).date(),
        (origin+timedelta(days=35)).date(), scales, TrialPolicy(), DemandScenario(), .01,
        PurposeTimingPolicy(1.), demand_accounting_key=reference.accounting_key)
    targets = (
        StimulusTarget('elevated cardiovascular work','hrr',(.7,.8,.9,1.),15.,'unchanged illustrative target'),
        StimulusTarget('fast running work','speed_over_easy',(1.05,1.25,1.6,1.8),15.,'unchanged illustrative target'),
    )
    cache = {}
    def compare(target, speed, demand_fraction=.5, purpose_fraction=.5):
        cache_key=(target.channel, speed, demand_fraction, purpose_fraction)
        if cache_key in cache:
            return cache[cache_key]
        scores, details = {}, {}
        for donor in quality:
            planned, signals = [], dict(historical_signals)
            for offset in range(35):
                role={1:'long',3:'quality',5:'easy'}.get(offset%7)
                if role is None:
                    continue
                chosen = donor if offset==3 else donors[role]
                row, signal = transplant(chosen, historical_signals, key=f'future:{offset}',
                    start=origin+timedelta(days=offset,hours=7), issued_at=as_of, speed=speed)
                planned.append(row)
                if signal is not None:
                    signals[row.key]=signal
            result = evaluate_stimulus_program(replace(context, scenario=DemandScenario(bound_fraction=demand_fraction)),
                history, tuple(planned), signals, reference=reference, target=target, as_of=as_of,
                timezone=config['timezone_default'], purpose_fraction=purpose_fraction,
                history_complete_since=min(r.start.astimezone(zone).date() for r in history))
            scores[donor.key]=result['evaluation'].total_cost
            q=result['quality_credit']['future:3']
            details[donor.key]=dict(target_credit=(q.low,q.high),
                delivery_cost=result['evaluation'].delivery_cost,
                weighted_demand_cost=result['evaluation'].weighted_demand_cost,
                constraint_status=result['constraints']['status'])
        case=dict(reference_mps=speed, demand_fraction=demand_fraction, purpose_fraction=purpose_fraction,
                  scores=scores, details=details)
        cache[cache_key]=case
        return case
    panels = []
    def winner(case):
        best = min(case['scores'].values())
        # Preserve ties; insertion order is not evidence of a different winner.
        return ' | '.join(sorted(key for key, value in case['scores'].items()
                                 if value-best <= 1e-10))
    for target in targets:
        values = sorted({bounds.low, bounds.high, bounds.estimate,
            *([bounds.low+(bounds.high-bounds.low)*i/(grid_points-1) for i in range(grid_points)]
               if target.channel=='speed_over_easy' else [])})
        primary = [compare(target, speed) for speed in values]
        transitions=[]
        for left, right in zip(primary, primary[1:]):
            if winner(left)==winner(right):
                continue
            low, high=left['reference_mps'],right['reference_mps']
            original_winner=winner(left)
            while high-low>1e-6:
                middle=(low+high)/2
                if winner(compare(target,middle))==original_winner:
                    low=middle
                else:
                    high=middle
            transitions.append(dict(reference_bracket_mps=(low,high),
                from_winner=original_winner,to_winner=winner(right)))
        control_speeds=sorted({bounds.low,bounds.high,bounds.estimate,
                              *[bounds.low+(bounds.high-bounds.low)*f for f in (.25,.5,.75)]})
        controls=[compare(target,s,d,p) for s,d,p in product(control_speeds,(0.,.5,1.),(0.,.5,1.))]
        panels.append(dict(target=target, primary_summary=ranking_summary(primary),
            uncertainty_control_summary=ranking_summary(controls), transitions=transitions,
            nominal=compare(target,bounds.estimate), primary_cases=primary, uncertainty_cases=controls))
    if sha256(database.read_bytes()).hexdigest()!=digest:
        raise AssertionError('Source database changed')
    return dict(version='coherent_stimulus_ranking_audit_v1', elapsed_seconds=perf_counter()-started,
        database_unchanged=True, planner_influence=False, source_reference=prior,
        context=context, candidates=[dict(key=r.key, local_date=r.start.astimezone(zone).date(),
            bouts=len(r.bouts), miles=r.miles, work_minutes=r.demand[2], demand=r.demand) for r in quality],
        panels=panels, evaluations=len(cache)*len(quality),
        limitations=['Five recorded response shapes, not an exhaustive candidate/calendar search',
            'Recorded weather/effort differ between donors; responses are replayed, not forecast for today',
            'One quality slot varied; common 21-day calendar plus 14-day continuation held fixed',
            'Same reference scenario for all future runs/alternatives; historical references frozen',
            'Grid counts are not probabilities; sampled stability does not prove no narrow intervening reversal',
            'Constraints checked separately; no physiological clearance, deployment or daily churn claim'])


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--grid-points',type=int,default=201)
    args=parser.parse_args()
    result=audit(Path('data/run_analysis.sqlite'),load_config(Path('config.yaml')),
        datetime.fromisoformat('2026-09-26T04:00:00+00:00'),grid_points=args.grid_points)
    with args.output.open('x') as stream:
        json.dump(result,stream,default=encode,indent=2,allow_nan=False)
    print(json.dumps(dict(seconds=result['elapsed_seconds'],evaluations=result['evaluations'],
        panels=[dict(target=p['target'].name,summary=p['primary_summary'],
                     controls=p['uncertainty_control_summary'],transitions=p['transitions']) for p in result['panels']]),indent=2))


if __name__=='__main__':
    main()
