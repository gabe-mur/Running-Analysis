"""Read-only original-file signal audit; not an optimizer or forecast validation."""
from argparse import ArgumentParser
from dataclasses import asdict, replace
from datetime import datetime, timedelta, time
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
from statistics import median
from time import perf_counter
from zoneinfo import ZoneInfo

from run_analysis.config import load_config
from run_analysis.continuous_training_dose import ContinuousDoseReference
from run_analysis.movement import classify_movement
from run_analysis.signal_structure import SignalSpan
from run_analysis.training_signals import HeartRateReference, observed_cardio_trace
from run_analysis.workload_accounting import Quantity
from scripts.audit_program_evidence_adapter import load_history, json_ready
from scripts.continuous_program_adapter import with_continuous_trace, evaluate_continuous
from scripts.program_evidence_adapter import QualityReference, reconcile_completion
from scripts.program_objective_trial import TrialPolicy, DemandScenario
from scripts.shared_program_evaluator import ProgramContext
from scripts.timely_purpose_trial import PurposeTimingPolicy


def audit(database, config, as_of):
    started = perf_counter()
    digest = sha256(database.read_bytes()).hexdigest()
    reference = ContinuousDoseReference('provisional sensitivity scenario, not fitted to outcomes')
    hr = HeartRateReference(config['resting_hr'], config['max_hr'], 'configured athlete reference')
    # Old reference exists ONLY to reuse the established physical bridge checks.
    # Every old cardiovascular value and template credit is discarded below.
    old_reference = QualityReference('threshold', 'work_seconds', Quantity.point(900, 'old comparator'),
        Quantity(290, 310, basis='old comparator'), 'discarded bridge comparator')
    reports, traces, original, rejected, padding_checks = {}, {}, {}, [], []

    def convert(row, activity):
        intervals = classify_movement(activity.trackpoints, config['moving_time']).intervals
        trace = observed_cardio_trace(activity, intervals, hr, reference.policy,
                                     include_spans=True, compute_density=False)
        traces[row.key], original[row.key] = trace, row
        # Fail the audit explicitly rather than omit an out-of-reference session.
        result, report = with_continuous_trace(row, trace['normalized_spans'], reference=reference,
            trace_key=row.key, trace_source='observed', elapsed_seconds=trace['elapsed_seconds'],
            outside_reference_seconds=trace['exclusions']['outside_reference_seconds'])
        reports[row.key] = report
        return result

    history, unresolved = load_history(database, config, as_of, old_reference, run_transform=convert)
    if unresolved:
        raise ValueError(f'Unresolved original sessions: {unresolved}')
    for row in history:
        trace = traces[row.key]
        # Counterfactual easy padding holds work evidence fixed. It must add
        # physical load, never additional quality credit (including bounds).
        elapsed = trace['elapsed_seconds']
        padded = replace(original[row.key], end=row.end+timedelta(minutes=20),
            available_at=row.available_at+timedelta(minutes=20),
            demand=(Quantity(row.demand[0].low+20, row.demand[0].high+20, basis='padding_counterfactual'),
                    original[row.key].demand[1], row.demand[2]))
        adjusted, _ = with_continuous_trace(padded,
            (*trace['normalized_spans'], SignalSpan(elapsed, elapsed+1200, .7, .7)),
            reference=reference, trace_key=row.key, trace_source='observed', elapsed_seconds=elapsed+1200)
        if adjusted.quality != row.quality or adjusted.demand[1].low <= row.demand[1].low:
            raise AssertionError('Non-work padding changed quality credit or lost physical load')
        padding_checks.append(dict(key=row.key, unchanged_quality_credit=True,
                                   added_cardio=adjusted.demand[1].low-row.demand[1].low))
    zone = ZoneInfo(config['timezone_default'])
    sensitivities = []
    for quality_hrr, quality_minutes in product((.8, .85, .9), (10., 15., 20.)):
        ref = replace(reference, quality_hrr=quality_hrr, quality_minutes=quality_minutes)
        cases = []
        for row in history:
            trace = traces[row.key]
            adjusted, report = with_continuous_trace(original[row.key], trace['normalized_spans'],
                reference=ref, trace_key=row.key, trace_source='observed',
                elapsed_seconds=trace['elapsed_seconds'])
            if row.role == 'quality':
                if adjusted.quality.low <= 0:
                    rejected.append((row.key, quality_hrr, quality_minutes))
                cases.append(dict(key=row.key, credit=asdict(adjusted.quality)))
            elif row.role in {'easy', 'long'} and adjusted.quality.high != 0:
                raise AssertionError('HR-only relabeling of easy/long')
        sensitivities.append(dict(quality_hrr=quality_hrr, quality_minutes=quality_minutes, cases=cases))

    # Known physical trace copied to expected event ONLY for the identity seam.
    # This is deliberately not a claimed out-of-sample HR forecast.
    actual = max(history, key=lambda r: r.start)
    past = tuple(r for r in history if r.key != actual.key)
    origin = datetime.combine(actual.start.astimezone(zone).date(), time(), zone)
    expected = replace(actual, key='identity_forecast', source='planned', available_at=origin,
                       miles=round(actual.miles, 1),
                       work_evidence_basis='prescribed' if actual.bouts else actual.work_evidence_basis)
    easy = [r for r in past if r.role == 'easy']
    scales = tuple(4.4*median((r.demand[i].low+r.demand[i].high)/2/r.miles for r in easy)
                   for i in (0, 1))+(15.,)
    context = ProgramContext((), origin.date(), origin.date(), (origin+timedelta(days=21)).date(),
        (origin+timedelta(days=35)).date(), scales, TrialPolicy(), DemandScenario(), .01,
        PurposeTimingPolicy(1.), demand_accounting_key=reference.accounting_key)
    donors = {role: max((r for r in past if r.role == role), key=lambda r: r.start)
              for role in ('easy', 'long', 'quality')}
    transitions = []
    for demand_fraction, purpose_fraction in product((0., .5, 1.), repeat=2):
        ctx = replace(context, scenario=DemandScenario(bound_fraction=demand_fraction))
        costs = []
        for shift in (0, 1):
            remaining = []
            for offset, role in ((3, 'easy'), (7, 'long'), (10+shift, 'quality'), (23, 'quality')):
                donor = donors[role]
                start = origin+timedelta(days=offset, hours=7)
                remaining.append(replace(donor, key=f'fixture:{offset}', source='planned',
                    start=start, end=start+(donor.end-donor.start), available_at=origin,
                    miles=round(donor.miles, 1),
                    work_evidence_basis='prescribed' if donor.bouts else donor.work_evidence_basis))
            def evaluate(h, p, at):
                return evaluate_continuous(ctx, h, p, reference=reference, as_of=at,
                    timezone=config['timezone_default'], purpose_fraction=purpose_fraction)['evaluation'].total_cost
            before = evaluate(past, (expected, *remaining), origin)
            h, p = reconcile_completion(past, (expected, *remaining), actual,
                                       replaces_key=expected.key, as_of=actual.available_at)
            after = evaluate(h, p, actual.available_at)
            rolled = evaluate(h, p, origin+timedelta(days=1))
            if abs(after-rolled) > 1e-10:
                raise AssertionError('Identical upload/rollover changed remaining score')
            costs.append((before, after))
        error = (costs[0][0]-costs[1][0])-(costs[0][1]-costs[1][1])
        if abs(error) > 1e-10:
            raise AssertionError('Expected completion changed relative future score')
        transitions.append(dict(demand_fraction=demand_fraction, purpose_fraction=purpose_fraction,
                                relative_score_error=error))
    if sha256(database.read_bytes()).hexdigest() != digest:
        raise ValueError('Source database changed during audit')
    return dict(status='offline_signal_and_accounting_audit', database_unchanged=True,
        elapsed_seconds=perf_counter()-started, reference=asdict(reference), hr_reference=asdict(hr),
        history_count=len(history), roles={role: sum(r.role == role for r in history)
            for role in ('easy', 'long', 'quality', 'unknown')},
        sessions=[dict(key=r.key, day=r.start.astimezone(zone).date(), role=r.role,
            bouts=len(r.bouts), work_minutes=asdict(r.demand[2]), credit=asdict(r.quality),
            previous_hr_or_work_credit_low=(min(1., max(reports[r.key]['high_effort_minutes'].low,
                r.demand[2].low)/reference.quality_minutes) if r.role == 'quality' else 0.),
            **{k: asdict(v) if isinstance(v, Quantity) else v for k, v in reports[r.key].items()})
            for r in history], sensitivities=sensitivities, rejected_quality_cases=rejected,
        transitions=transitions, padding_checks=padding_checks, normalization=scales, planner_influence=False,
        limitations=['General coverage is not stimulus equivalence or measured adaptation',
            'Work extent relies on the existing detector; it does not verify absolute work intensity',
            'The duration reference remains a policy assumption, not a physiological exchange rate',
            'Sensitivity grid is not fitting or physiological validation',
            'Trace-copy transition checks accounting, not forecast accuracy or daily search stability'])


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(Path('data/run_analysis.sqlite'), load_config(Path('config.yaml')),
                   datetime.fromisoformat('2026-09-26T04:00:00+00:00'))
    with args.output.open('x') as handle:
        json.dump(result, handle, default=json_ready, indent=2, allow_nan=False)
    print(json.dumps({k: result[k] for k in ('status', 'history_count', 'roles', 'elapsed_seconds',
                                           'database_unchanged', 'rejected_quality_cases')}))


if __name__ == '__main__':
    main()
