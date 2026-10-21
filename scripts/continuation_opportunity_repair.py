"""One best native easy insertion after coordinated soft-role selection.

Inspect unused soft dates, not only the mechanical opportunity grid. Generate
each dose with existing recommendation/allocation machinery; keep every other
workout fixed and reject any downstream guard failure. Same complete score for
selection and reporting. No new weights, dose floor changes or hard-date edits.
"""

from contextlib import closing
from math import isfinite
from time import perf_counter

from run_analysis import weekly_schedule as planner
from run_analysis.prescription_distance import UnrepresentableDistanceTarget
from scripts.continuation_checks import check_sequence
from scripts.continuation_distance import (continuation_distance_policy, continuation_distance_intents,
                                           finalize_continuation_distance)


def select_best_insertion(incumbent, scored, candidates, evaluate, validate):
    """Bounded common-baseline comparison, deterministic incumbent on ties."""
    if not isfinite(scored['combined_cost']):
        raise ValueError('Insertion selection requires a finite incumbent score')
    selected, winner, rows = incumbent, scored, []
    for index, trial, facts in candidates:
        checked = validate(trial, index)
        row = {'offset': index, 'generation': facts, 'checks': checked}
        if checked['checks_pass']:
            cost = evaluate(trial)
            row['cost'] = cost['combined_cost']
            if isfinite(row['cost']) and row['cost'] < winner['combined_cost'] - 1e-9:
                selected, winner = trial, cost
        rows.append(row)
    return selected, winner, rows


def polish(continuation, scored, *, inputs, targets, hard_length, state_at, evaluate, compare_doses=False):
    started = perf_counter()
    days, states, attempts = continuation
    if not 0 < hard_length < len(days) or len(days) != len(states) or len(targets) < len(days):
        raise ValueError('Require aligned states/targets and a nonempty hard and soft program')
    if not isfinite(scored['combined_cost']):
        raise ValueError('Insertion selection requires a finite incumbent score')
    context = {**inputs, '_audit_targets': targets}
    baseline_check = check_sequence(days, context, state_at, ())
    metrics = {'initial_cost': scored['combined_cost'], 'baseline_checks': baseline_check,
               'compare_doses': compare_doses,
               'scope': 'One insertion at most; fixed other sessions, one noon option per unused soft date.'}
    if not baseline_check['checks_pass']:
        metrics.update(final_cost=scored['combined_cost'], selected_offset=None, candidates=[],
                       rejected=[], status='baseline_guard_failure', seconds=perf_counter() - started)
        return continuation, scored, metrics
    blocked = set(inputs.get('forced_rest_offsets') or ()) | set(inputs.get('completed_run_offsets') or ())
    rejected = []

    def candidates():
        with continuation_distance_policy(states[0], [d.date for d in days], targets[:len(days)], inputs['config']) as intentions:
            for i in range(hard_length, len(days)):
                if (days[i].recommendation is not None or i in blocked
                        or days[i].completed_activities):
                    continue
                # Exact same bounded clock option as the demonstrated witness;
                # this is not time-of-day optimization or a mandatory run time.
                at = states[i].as_of.replace(hour=12, minute=0, second=0, microsecond=0)
                prior = [d.recommendation for d in days[:i] if d.recommendation
                         and d.recommendation.workout_type != planner.WorkoutType.REST]
                try:
                    state, generated = planner._select_budgeted_timed_recommendation(
                        [state_at(at)], prior, inputs['request'], inputs['config'], weekly_role='easy')
                    if generated.workout_type != planner.WorkoutType.EASY or generated.readiness.value == 'not_ready':
                        rejected.append({'offset': i, 'reason': 'recommendation_guard'})
                        continue
                except UnrepresentableDistanceTarget as error:
                    rejected.append({'offset': i, 'reason': str(error)})
                    continue
                upper_intent = intentions[days[i].date]
                # Try the previous native choice first, preserving it on ties.
                # The singleton allocator's preferred size is not necessarily
                # the best size under the complete downstream program score.
                options = (upper_intent, *(continuation_distance_intents(states[0], upper_intent,
                    inputs['config']) if compare_doses else ()))
                seen = set()
                for intent in dict.fromkeys(options):
                    try:
                        result, facts = finalize_continuation_distance(generated, state, inputs['config'],
                            targets[i], intent)
                    except UnrepresentableDistanceTarget as error:
                        rejected.append({'offset': i, 'intent_miles': intent, 'reason': str(error)})
                        continue
                    if result.workout_type != planner.WorkoutType.EASY:
                        rejected.append({'offset': i, 'reason': 'finalization_changed_role'})
                        continue
                    if result.planned_for is None or result.planned_for.date() != days[i].date:
                        rejected.append({'offset': i, 'reason': 'finalization_changed_date'})
                        continue
                    # Different intents may collapse to the same capped dose.
                    # Deduplicate the whole prescription, not just its target:
                    # different ranges have different upper-envelope loads.
                    identity = result.model_dump_json(exclude={'generated_at'})
                    if identity in seen:
                        continue
                    seen.add(identity)
                    changed = list(days)
                    changed[i] = days[i].model_copy(update={'recommendation': result,
                        'planned_at': result.planned_for,
                        'day_role': 'medium_long_run' if result.planning_role == 'medium_long' else 'easy_run',
                        'rationale': 'Offline native easy opportunity; not an issued prescription.'})
                    new_attempt = {'at': result.planned_for.isoformat(), 'requested_role': 'easy',
                        'selected_type': 'easy', 'opening_residual': state.recovery_residual_load,
                        'distance_finalization': facts, 'source': 'bounded_insertion'}
                    # Replace an explicit-rest attempt, retaining other records.
                    revised_attempts = sorted([x for x in attempts if x['at'][:10] != days[i].date.isoformat()]
                                              + [new_attempt], key=lambda x: x['at'])
                    yield i, (changed, states, revised_attempts), {
                        **facts, 'at': result.planned_for.isoformat(), 'target_miles': planner._midpoint(result),
                        'distance_range_miles': result.distance_range_miles,
                        'on_original_opportunity': any(x['at'][:10] == days[i].date.isoformat() for x in attempts)}

    with closing(candidates()) as pool:
        selected, winner, rows = select_best_insertion(continuation, scored, pool, evaluate,
            lambda trial, index: check_sequence(trial[0], context, state_at, (index,)))
    selected_offsets = [i for i, (before, after) in enumerate(zip(days, selected[0], strict=True)) if before != after]
    assert len(selected_offsets) <= 1 and all(i >= hard_length for i in selected_offsets)
    # Selection and reported score must describe identical finalized sessions.
    verified = evaluate(selected)
    assert abs(verified['combined_cost'] - winner['combined_cost']) < 1e-8
    assert verified['sessions'] == winner['sessions']
    metrics.update(final_cost=winner['combined_cost'], selected_offset=next(iter(selected_offsets), None),
                   candidates=rows, rejected=rejected, status='checked', seconds=perf_counter() - started)
    return selected, winner, metrics
