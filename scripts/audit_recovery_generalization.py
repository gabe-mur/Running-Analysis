"""Probe survivors on unlabelled cases, without tuning on their outcomes.

This is model sensitivity, not validation against measured recovery outcomes.
No scenario below becomes a new 'must pass' physiological label.
"""
from collections import Counter
from dataclasses import replace
from datetime import timedelta
from math import expm1, log
import argparse
import json
from pathlib import Path
from time import perf_counter

from run_analysis.recovery_assessment import assess_next_session, sensitivity_panel, SessionEntry
from run_analysis.workload_accounting import Quantity
from scripts.audit_recovery_baseline import NOW, EASY, LONG, QUALITY, history, select_policies


def mixed_history(days):
    """Explicit day-level program; index zero is the most recent completed day."""
    entries = []
    for i, item in enumerate(days):
        if item is not None:
            entries.extend(replace(e, key=f'day-{i+1}') for e in history(item, (24*(i+1),)))
    return tuple(entries)


def probes():
    cases = []
    for previous_name, previous in (('long', LONG), ('quality', QUALITY)):
        for gap in (18, 24, 36, 48, 72):
            for next_name, proposed in (('easy', EASY), ('long', LONG), ('quality', QUALITY)):
                cases.append((f'{previous_name}_to_{next_name}_{gap}h', history(previous, (gap,)), proposed))
    # Complete preceding 14 days; this is not a calendar optimization.
    programs = {
        'mixed_with_rests': (EASY, None, QUALITY, EASY, None, LONG, None)*2,
        'mixed_more_easy': (EASY, EASY, QUALITY, EASY, None, LONG, None)*2,
        'mixed_no_rests': (EASY, QUALITY, EASY, LONG, EASY, QUALITY, EASY)*2,
        'daily_short_easy': (EASY,)*14,
    }
    for name, days in programs.items():
        for next_name, proposed in (('easy', EASY), ('long', LONG), ('quality', QUALITY)):
            cases.append((f'{name}_to_{next_name}', mixed_history(days), proposed))
    # Work is unmeasured but cannot exceed total running duration. This bound is
    # physical/accounting support, NOT an invented confidence interval.
    past = history(QUALITY, (24,))
    cases.append(('uncertain_previous_work', tuple(replace(e, work_minutes=Quantity(
        0, e.workload.running_minutes.high, basis='unmeasured work bounded by duration')) for e in past), EASY))
    return cases


def evaluate_case(past, proposed, policy):
    return sensitivity_panel(past, proposed[0], NOW, (policy,), proposed_work_minutes=proposed[1])


def base_key(policy):
    return (policy.running_minutes_envelope, policy.cardio_points_envelope,
            policy.running_half_life_hours, policy.cardio_half_life_hours)


def scale_case(past, proposed, policy, factor):
    def scale_workload(workload):
        return replace(workload,
                       running_minutes=workload.running_minutes.scale(factor),
                       cardio_points=workload.cardio_points.scale(factor))
    scaled_history = tuple(replace(e, workload=scale_workload(e.workload),
        start=e.end-timedelta(minutes=e.workload.running_minutes.high*factor),
        work_minutes=e.work_minutes.scale(factor) if e.work_minutes is not None else None) for e in past)
    scaled_proposed = (scale_workload(proposed[0]), proposed[1].scale(factor))
    scaled_policy = replace(policy, running_minutes_envelope=policy.running_minutes_envelope*factor,
        cardio_points_envelope=policy.cardio_points_envelope*factor,
        work_minutes_envelope=policy.work_minutes_envelope*factor if policy.work_minutes_envelope is not None else None)
    return scaled_history, scaled_proposed, scaled_policy


REFERENCE_CYCLE = ((0, LONG), (2, EASY), (4, QUALITY), (5, EASY))


def recurring_requirements(policy, *, cycle=REFERENCE_CYCLE, cycle_days=8, offsets=(-6, 0, 6)):
    """Exact geometric sum of all earlier cycles, not a finite-horizon tail.

    This is a declared training-stage example, NOT an imposed eight-day schedule.
    Starts are day-spaced; finish times include each session's running duration.
    Require the familiar example to be repeatable without headroom exhaustion.
    """
    if not cycle or cycle_days <= 0 or not offsets:
        raise ValueError('Nonempty positive cycle and timing scenarios required')
    days = [day for day, _ in cycle]
    if len(set(days)) != len(days) or any(not 0 <= d < cycle_days for d in days):
        raise ValueError('Unique session days within cycle required')
    specs = [('running_minutes', 0, policy.running_half_life_hours, policy.running_minutes_envelope),
             ('cardio_points', 0, policy.cardio_half_life_hours, policy.cardio_points_envelope)]
    if policy.work_minutes_envelope is not None:
        specs.append(('work_minutes', 1, policy.work_half_life_hours, policy.work_minutes_envelope))
    result = {}
    for name, index, half_life, budget in specs:
        denominator = -expm1(-log(2)*cycle_days*24/half_life)
        worst, limiting = 0., None
        for day, proposed in cycle:
            next_dose = proposed[1] if index else getattr(proposed[0], name)
            if next_dose is None:
                raise ValueError('Reference cycles must have explicit bounded doses')
            for offset in offsets:
                demand = next_dose.high
                for previous_day, previous in cycle:
                    gap_days = (day-previous_day) % cycle_days or cycle_days
                    elapsed = gap_days*24-previous[0].running_minutes.high/60+offset
                    if elapsed < 0:
                        raise ValueError('Reference sessions overlap under timing scenario')
                    dose = previous[1] if index else getattr(previous[0], name)
                    demand += dose.high * 2**(-elapsed/half_life) / denominator
                if demand > worst:
                    worst, limiting = demand, dict(day=day, offset_hours=offset)
        result[name] = dict(required_budget=worst, budget=budget, margin=budget-worst,
                            supported=worst <= budget+1e-8, limiting=limiting)
    return dict(supported=all(v['supported'] for v in result.values()), channels=result)


def audit():
    started = perf_counter()
    families = {name: select_policies(structured)[0]
                for name, structured in (('two_channels', False), ('three_channels', True))}
    cases = probes()
    rows = []
    for name, past, proposed in cases:
        row = dict(name=name, families={})
        for family, candidates in families.items():
            counts = Counter(evaluate_case(past, proposed, policy)['status'] for policy in candidates)
            row['families'][family] = dict(counts=counts,
                unanimous=len(counts) == 1, candidate_count=len(candidates))
        rows.append(row)

    # Ablation holds the two existing budgets and decays fixed. Comparing raw
    # pass percentages from unequal families would confound this question.
    controls = {base_key(p): p for p in families['two_channels']}
    paired = [(p, controls[base_key(p)]) for p in families['three_channels'] if base_key(p) in controls]
    matched = []
    for name, past, proposed in cases:
        counts = Counter()
        for candidate, control in paired:
            a = evaluate_case(past, proposed, control)['status']
            b = evaluate_case(past, proposed, candidate)['status']
            counts[f'{a} -> {b}'] += 1
        matched.append(dict(name=name, paired_extensions=len(paired), transitions=counts))

    # Different rep splits with identical aggregate T/C/W are observationally
    # identical to this model. These are not asserted equal-fatigue sessions.
    alias_checks = 0
    for policy in families['three_channels']:
        for durations in ((900,), (300, 300, 300), (90,)*10):
            proposed = replace(QUALITY[0], structure={'work_rep_seconds': durations})
            result = assess_next_session(history(EASY, (24,)), proposed, NOW, policy,
                                         proposed_work_minutes=QUALITY[1])
            baseline = assess_next_session(history(EASY, (24,)), QUALITY[0], NOW, policy,
                                           proposed_work_minutes=QUALITY[1])
            if result != baseline:
                raise AssertionError('Aggregate model unexpectedly consumed unmodelled rep structure')
            alias_checks += 1

    # Similarity is an arithmetic invariant, not evidence that half-size budgets
    # describe a new runner. No cold-start capability is inferred here.
    scale_checks = 0
    for policy in families['two_channels'] + families['three_channels']:
        past, proposed = mixed_history((EASY, QUALITY, None, LONG)), EASY
        baseline = evaluate_case(past, proposed, policy)['status']
        for factor in (.5, 1.5):
            scaled = scale_case(past, proposed, policy, factor)
            if evaluate_case(*scaled)['status'] != baseline:
                raise AssertionError('Dose/budget scale invariance failed')
            scale_checks += 1
    recurring = {}
    for name, candidates in families.items():
        retained = [p for p in candidates if recurring_requirements(p)['supported']]
        recurring[name] = dict(retained=len(retained), rejected=len(candidates)-len(retained),
            retained_by_half_life={str(h): sum(p.running_half_life_hours == h for p in retained)
                                   for h in (12, 24, 36)},
            remaining_disagreement_cases=sum(len({evaluate_case(past, proposed, p)['status']
                for p in retained}) > 1 for _, past, proposed in cases),
            zero_long_duration_headroom=sum(p.running_minutes_envelope <= LONG[0].running_minutes.high
                                            for p in candidates))
    return dict(elapsed_seconds=perf_counter()-started, scenario_count=len(cases),
        accepted_counts={k: len(v) for k, v in families.items()},
        disagreement_cases={k: sum(not row['families'][k]['unanimous'] for row in rows) for k in families},
        cases=rows, matched_ablation=matched, aggregate_alias_checks=alias_checks,
        recurring_reference_screen=recurring,
        scale_invariance_checks=scale_checks, planner_influence=False,
        limitations=['Candidate fractions are not probabilities or confidence levels',
                     'Unlabelled probes are not additional calibration constraints',
                     'Equal aggregate scores do not establish equal fatigue',
                     'Scaling checks do not validate new-runner capacity or recovery'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = audit()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k not in {'cases', 'matched_ablation'}}, indent=2))
