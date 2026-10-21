"""Bounded rest vs short insertion vs equal-mileage easy extension audit.

Offline only. Hold the hard program and all unedited soft workouts fixed;
reproject every following session. Separately report recommendation/recovery
checks, existing dose-menu coverage and the unchanged complete objective.
Removing shortfall fees is an explicit diagnostic ablation, never activation.
"""

from datetime import datetime
from functools import lru_cache
from itertools import combinations
from math import isfinite
from time import perf_counter

from run_analysis import adherence_projection as projection
from run_analysis import weekly_schedule as planner
from scripts.typed_continuation_prototype import compare_candidate, rollout, runs, score_continuation
from scripts.continuation_checks import check_sequence
from scripts.aerobic_distribution import per_run_shortfall_components


def revised_easy(day, template, at, miles, ordinary, expansion_reference):
    """Canonical tenth-mile target with the existing quarter-mile margin."""
    if not isfinite(miles) or miles <= .25 or abs(miles * 10 - round(miles * 10)) > 1e-8:
        raise ValueError('Require a positive one-decimal target above the execution margin')
    medium = miles > expansion_reference + 1e-9
    result = template.model_copy(update={
        'planned_for': at, 'fitness_state_as_of': at, 'workout_type': planner.WorkoutType.EASY,
        'distance_target_miles': miles, 'distance_range_miles': (miles - .25, miles + .25),
        'duration_range_minutes': None, 'quality_session_type': None, 'structure': [],
        'planning_role': 'medium_long' if medium else 'support_easy' if miles < ordinary else 'ordinary_easy',
        'title': 'Offline dose comparison — not an issued workout',
    })
    return day.model_copy(update={'planned_at': at, 'recommendation': result,
        'day_role': 'medium_long_run' if medium else 'easy_run'})


def variants(days, states, hard_length, *, added_miles=2.):
    """Same added volume; at most one run per day and no hard-prefix edits."""
    ordinary = states[0].typical_easy_run_miles
    expansion = planner._ordinary_easy_expansion_reference(states[0])
    easy = [i for i in range(hard_length, len(days)) if days[i].recommendation
            and days[i].recommendation.workout_type == planner.WorkoutType.EASY]
    if not easy:
        raise ValueError('Audit needs an existing soft easy run as a prescription template')
    template = days[easy[0]].recommendation
    result = [('rest_control', (), list(days))]
    for i in range(hard_length, len(days)):
        if days[i].recommendation is None:
            at = states[i].as_of.replace(hour=12, minute=0, second=0, microsecond=0)
            changed = list(days)
            changed[i] = revised_easy(days[i], template, at, added_miles, ordinary, expansion)
            result.append(('insert_short' if added_miles < ordinary else 'insert_ordinary', (i,), changed))
    for indices in [(i,) for i in easy] + list(combinations(easy, 2)):
        changed = list(days)
        for i in indices:
            original = days[i].recommendation
            miles = planner._midpoint(original) + added_miles / len(indices)
            changed[i] = revised_easy(days[i], original, original.planned_for,
                                      miles, ordinary, expansion)
        result.append(('extend_one' if len(indices) == 1 else 'extend_two', indices, changed))
    if len(result) > 100:
        raise ValueError('Bounded audit exceeds 100 variants')
    return result


def shortfall_fees(days, states, request):
    """Exact current per-run shortfall fees; local distribution is separate."""
    parts = per_run_shortfall_components(days, states, request)
    # Preserve the original operation ordering for exact default-path parity.
    return planner.PROGRAM_FIT_UNIT * (parts['missing_equivalents'] * planner.AEROBIC_SUPPORT_FIT_FRACTION
        + (parts['missing_equivalents'] if request.health_status == planner.CurrentHealthStatus.NORMAL else 0.))


def compare_delivery(inputs, hard, hard_score, history, state_at, *, continuation_days=28,
                     added_miles_options=(1., 2., 4.)):
    start = perf_counter()
    targets = inputs['expected_target_projector'](runs(hard), len(hard) + continuation_days)
    inputs = {**inputs, '_audit_targets': targets}
    baseline = compare_candidate(inputs, hard, hard_score, state_at=state_at,
        continuation_days=continuation_days, distance_policy='finalized', timing_policy='coordinated',
        complete_role_costs=True, support_policy='bounded', completed_runs=history)
    overrides = {datetime.fromisoformat(x['at']).date(): x['requested_role'] for x in baseline['attempts']}
    combined, states, attempts = rollout(hard, inputs['daily_states'], inputs['request'], inputs['config'],
        targets, state_at=state_at, continuation_days=continuation_days,
        distance_policy='finalized', role_overrides=overrides)
    choices = []
    for amount in added_miles_options:
        batch = variants(combined, states, len(hard), added_miles=amount)
        choices.extend(batch if not choices else batch[1:])
    if len(choices) > 100:
        raise ValueError('Combined audit exceeds 100 variants')
    rows = []
    for label, edits, days in choices:
        scored = score_continuation(inputs, hard, hard_score, targets, (days, states, attempts),
            state_at=state_at, continuation_days=continuation_days, distance_policy='finalized',
            complete_role_costs=True, support_policy='bounded', completed_runs=history)
        fees = shortfall_fees(days, states, inputs['request'])
        check = check_sequence(days, inputs, state_at, edits)
        rows.append({'label': label, 'edits': edits, 'diagnostic_checks': check,
            'edits_on_original_opportunities': [days[i].date.isoformat() in
                {x['at'][:10] for x in attempts} for i in edits],
            'added_miles': sum(planner._midpoint(r) for r in runs(days)) - sum(planner._midpoint(r) for r in runs(combined)),
            'cost': scored['combined_cost'], 'cost_without_per_run_shortfall_fees': scored['combined_cost'] - fees,
            'per_run_shortfall_fees': fees, 'funding_cost': scored['combined_target_cost'],
            'recovery_cost': scored['trajectory_recovery_cost'], 'coaching_cost': scored['combined_coaching_cost'],
            'support_cost': planner.PROGRAM_FIT_UNIT * planner.AEROBIC_SUPPORT_FIT_FRACTION * scored['combined_support_violation'],
            'shape_cost': planner.PROGRAM_FIT_UNIT * scored['combined_shape_violation'],
            'rate': scored['policy_rate'], 'target': scored['soft_target'], 'sessions': scored['sessions']})
    assert abs(rows[0]['cost'] - baseline['combined_cost']) < 1e-8
    assert rows[0]['sessions'] == baseline['sessions']
    baseline_failures = set(rows[0]['diagnostic_checks']['failed_offsets'])
    for row in rows:
        row['new_guard_failures'] = sorted(set(row['diagnostic_checks']['failed_offsets']) - baseline_failures)
    return {'seconds': perf_counter() - start, 'origin': inputs['daily_states'][0].as_of.isoformat(),
            'baseline': baseline, 'rows': rows,
            'scope': 'Fixed hard prefix and unedited soft sessions. Existing guard checks are not full production acceptance.'}


def main():
    from argparse import ArgumentParser
    import json
    from pathlib import Path
    from scripts.audit_rollover_distance import investigate
    from scripts.replay_synthetic_easy_dose import synthetic_history

    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output-json', type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        parser.error('Require a new output path')
    source = json.loads(args.report.read_text())
    template, history = synthetic_history(datetime.fromisoformat(source['realized_execution']['start_at']))
    found = {}

    def observe(inputs, days, score):
        opening = inputs['daily_states'][0]
        key = opening.as_of.isoformat()
        if key in found:
            return None
        for n in (7, 14, 28):
            actual = projection._load_window(history, opening.as_of, n)
            captured = getattr(opening.recent_load, f'trailing_{n}d')
            if actual.activity_count != captured.activity_count or abs(actual.distance_miles - captured.distance_miles) > 1e-8:
                raise ValueError('Seed history is stale for the selected decision')
        known = {s.as_of: s for options in inputs['daily_state_options'] for s in options}
        known.update({s.as_of: s for s in inputs['daily_states']})
        settings = inputs['config'].get('coaching', {})
        @lru_cache(maxsize=None)
        def state_at(at):
            if at in known:
                return known[at]
            return projection._state_at(template, history, at, opening.recent_load.capacity_reference_miles,
                easy_baseline_half_life_days=float(settings.get('capacity_retention_half_life_days', 84)),
                fatigue_half_life_days=float(settings.get('continuous_fatigue_half_life_days', 7)))
        found[key] = compare_delivery(inputs, days, score, history, state_at)
        return None

    reconstruction = investigate(source, candidate_observer=observe)
    report = {'scope': __doc__, 'seconds': reconstruction['seconds'], 'decisions': list(found.values()),
              'matched_saved_decisions': reconstruction['matched_decisions'],
              'limitations': ['Synthetic observed-history fixture, not a recorded-HR daily replay.',
                  'No integration, live change, full calendar search or physiological safety claim.',
                  'One-, two- and four-mile additions are controlled probes, not new prescribed minima or goals.']}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    for decision in report['decisions']:
        print(decision['origin'], 'variants', len(decision['rows']), 'seconds', round(decision['seconds'], 3))
        for label, amount in sorted({(r['label'], r['added_miles']) for r in decision['rows']}):
            valid = [r for r in decision['rows'] if r['label'] == label and r['added_miles'] == amount
                     and r['diagnostic_checks']['checks_pass']]
            best = min(valid, key=lambda x: x['cost']) if valid else None
            print(label, amount, 'passing', len(valid), 'best', {k:best[k] for k in
                ('edits', 'cost', 'cost_without_per_run_shortfall_fees', 'rate')} if best else None)


if __name__ == '__main__':
    main()
