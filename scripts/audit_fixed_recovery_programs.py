"""Matched-work fixed-program comparison. No search, live scoring or calibration."""
from collections import Counter
from dataclasses import replace
from datetime import timedelta
from itertools import combinations
import argparse
import json
from pathlib import Path
from time import perf_counter

from scripts.audit_recovery_baseline import NOW, EASY, LONG, QUALITY, select_policies
from scripts.audit_recovery_generalization import REFERENCE_CYCLE, recurring_requirements
from run_analysis.recovery_assessment import SessionEntry, assess_next_session

DOSES = {'E': EASY, 'L': LONG, 'Q': QUALITY}
PROGRAMS = {
    'spaced': 'L.E.QE..L.E.QE..L.EQE',
    'easy_after_long': 'LE..QE..LE..QE..LE.QE',
    'easy_before_quality': 'L..EQ.E.L..EQ.E.L.EQE',
    'quality_before_long': 'Q.EL.E..Q.EL.E..Q.ELE',
    'compressed_blocks': 'LQEE....LQEE....LQEE.',
}
# Every candidate faces identical subsequent work. Tail costs are reported
# separately and together, not silently dropped or labelled a clinical risk.
TAIL = '.E.QE..L.E.QE.'
TAILS = {f'phase_{shift}': TAIL[shift:]+TAIL[:shift] for shift in (0, 2, 4)}
TIMINGS = {'nominal': (0,), 'alternating_early': (-6, 6), 'alternating_late': (6, -6)}


def history_before_plan():
    result = []
    for cycle in range(1, 5):
        for day, (workload, work) in REFERENCE_CYCLE:
            start = NOW + timedelta(days=day-8*cycle)
            result.append(SessionEntry(f'history/{cycle}/{day}', start,
                start+timedelta(minutes=workload.running_minutes.high),
                replace(workload, source='observed'), work))
    return tuple(result)


def events(program, timing='nominal', tail=TAIL):
    offsets = TIMINGS[timing]
    result = []
    for day, label in enumerate(program+tail):
        if label == '.':
            continue
        workload, work = DOSES[label]
        start = NOW+timedelta(days=day, hours=offsets[day % len(offsets)])
        result.append((day, label, SessionEntry(f'planned/{day}', start,
            start+timedelta(minutes=workload.running_minutes.high), workload, work)))
    return result


def demand_trace(program, timing, policy, tail=TAIL):
    history = list(history_before_plan())
    trace = []
    for day, label, event in events(program, timing, tail):
        assessment = assess_next_session(history, event.workload, event.start, policy,
                                         proposed_work_minutes=event.work_minutes)
        trace.append(dict(day=day, role=label, totals={k: v['total_high']
                         for k, v in assessment['channels'].items()}))
        history.append(event)
    return trace


def score_trace(trace, policy):
    budgets = {'running_minutes': policy.running_minutes_envelope,
               'cardio_points': policy.cardio_points_envelope}
    if policy.work_minutes_envelope is not None:
        budgets['work_minutes'] = policy.work_minutes_envelope
    costs = [(row['day'], max(max(0., row['totals'][k]/budget-1) for k, budget in budgets.items()))
             for row in trace]
    return dict(plan_excess=sum(cost for day, cost in costs if day < 21),
                tail_excess=sum(cost for day, cost in costs if day >= 21),
                combined_excess=sum(cost for _, cost in costs),
                peak_excess=max(cost for _, cost in costs),
                violating_plan_sessions=sum(cost > 1e-8 for day, cost in costs if day < 21))


def metrics(program):
    days = [i for i, label in enumerate(program) if label != '.']
    totals = {name: sum(getattr(DOSES[label][0], name).high for label in program if label != '.')
              for name in ('running_minutes', 'cardio_points')}
    totals['work_minutes'] = sum(DOSES[label][1].high for label in program if label != '.')
    streak = longest = 0
    for label in program:
        streak = streak+1 if label != '.' else 0
        longest = max(longest, streak)
    return dict(counts=dict(Counter(label for label in program if label != '.')), totals=totals,
                largest_start_gap_days=max(b-a for a, b in zip(days, days[1:])),
                longest_running_streak=longest,
                full_seven_day_run_count_range=[min(v := [sum(c != '.' for c in program[i:i+7])
                                                       for i in range(15)]), max(v)])


def audit():
    started = perf_counter()
    if any(len(program) != 21 for program in PROGRAMS.values()):
        raise AssertionError('All prescriptions must span 21 days')
    program_metrics = {name: metrics(program) for name, program in PROGRAMS.items()}
    reference = next(iter(program_metrics.values()))
    if any(m['counts'] != reference['counts'] or m['totals'] != reference['totals']
           for m in program_metrics.values()):
        raise AssertionError('Programs must match aggregate work')
    families = {name: [p for p in select_policies(structured)[0] if recurring_requirements(p)['supported']]
                for name, structured in (('two_channels', False), ('three_channels', True))}
    trace_cache = {}
    results = {}
    for family, policies in families.items():
        values = {name: [] for name in PROGRAMS}
        tail_values = {tail_name: {name: [] for name in PROGRAMS} for tail_name in TAILS}
        for policy in policies:
            for name, program in PROGRAMS.items():
                for tail_name, tail in TAILS.items():
                    scenarios = []
                    for timing in TIMINGS:
                        # Demand depends on decay, not on the budget; reuse the exact
                        # assessor's totals across identical decay parameters.
                        key = (name, timing, tail_name, policy.running_half_life_hours, policy.cardio_half_life_hours,
                               policy.work_half_life_hours)
                        if key not in trace_cache:
                            trace_cache[key] = demand_trace(program, timing, policy, tail)
                        scenarios.append(score_trace(trace_cache[key], policy))
                    result = {metric: max(row[metric] for row in scenarios) for metric in scenarios[0]}
                    tail_values[tail_name][name].append(result)
                    if tail_name == 'phase_0':
                        values[name].append(result)
        comparisons = []
        for left, right in combinations(PROGRAMS, 2):
            counts = Counter()
            for a, b in zip(values[left], values[right]):
                difference = a['combined_excess']-b['combined_excess']
                counts['tie' if abs(difference) <= 1e-8 else 'left_lower' if difference < 0 else 'right_lower'] += 1
            comparisons.append(dict(left=left, right=right, counts=counts,
                                    preference_reversal=counts['left_lower'] > 0 and counts['right_lower'] > 0))
        tail_comparisons = []
        for left, right in combinations(PROGRAMS, 2):
            reversals = changes = 0
            for i in range(len(policies)):
                signs = set()
                for by_program in tail_values.values():
                    delta = by_program[left][i]['combined_excess']-by_program[right][i]['combined_excess']
                    signs.add(0 if abs(delta) <= 1e-8 else -1 if delta < 0 else 1)
                reversals += -1 in signs and 1 in signs
                changes += len(signs) > 1
            tail_comparisons.append(dict(left=left, right=right, strict_reversals=reversals,
                                         changed_including_ties=changes))
        results[family] = dict(candidate_count=len(policies),
            programs={name: {metric: [min(row[metric] for row in rows), max(row[metric] for row in rows)]
                             for metric in rows[0]} | {'zero_combined_excess': sum(row['combined_excess'] <= 1e-8 for row in rows)}
                      for name, rows in values.items()},
            comparisons=comparisons, continuation_sensitivity=tail_comparisons,
            tail_changes_comparison=sum(
                (a['plan_excess']-b['plan_excess'] > 1e-8) != (a['combined_excess']-b['combined_excess'] > 1e-8)
                or (a['plan_excess']-b['plan_excess'] < -1e-8) != (a['combined_excess']-b['combined_excess'] < -1e-8)
                for left, right in combinations(PROGRAMS, 2) for a, b in zip(values[left], values[right])))
    return dict(elapsed_seconds=perf_counter()-started, programs=PROGRAMS,
                metrics=program_metrics, shared_tail=TAIL, alternative_shared_tails=TAILS, timing_scenarios=TIMINGS,
                history_days=32, results=results, trace_cache_entries=len(trace_cache), planner_influence=False,
                limitations=['Matched aggregate dose does not establish equal training benefit',
                    'Excess sums are diagnostic, not a new live objective or injury probability',
                    'Worst timing path is evaluated consistently over the whole program',
                    'Fourteen-day shared continuation exposes downstream effects but is not an infinite-tail proof',
                    'No calendar search or daily-replanning stability test'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = audit()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
