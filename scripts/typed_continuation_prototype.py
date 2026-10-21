"""Bounded offline policy rollout, not an enabled planner or a release candidate.

The base policy uses observed mean session size / projected mileage demand for
its opportunity cadence, anchored to the last actual candidate session. Existing
role selection and recovery guards choose each workout. This is a deliberately
inspectable heuristic, not an assertion that this cadence is optimal. No beam
search, fake completed evidence, database writes, or production hooks.
"""

from datetime import timedelta
from math import isfinite
from time import perf_counter
from contextlib import nullcontext

from run_analysis import weekly_schedule as planner
from run_analysis.prescription_distance import rounded_distance_target
from run_analysis.selection_objective import (
    EXPECTED_COMPLETION_STATE, VOLUME_PATH_SCORER, CONTINUATION_FUNDING_SCORER,
)


def runs(days):
    return [d.recommendation for d in days if d.recommendation
            and d.recommendation.workout_type != planner.WorkoutType.REST]


def continued_recovery(days, states, config, target, *, state_at, targets=None):
    """Score one physical sequence once, independent of its hard/soft split.

    Raw states contain completed history ONLY. Passing already projected states
    would count planned load twice. The existing residual closeout moves to the
    far end of the rollout; it is not also charged at the original hard edge.
    """
    timed = [state_at(d.recommendation.planned_for) if d.recommendation
             and d.recommendation.planned_for else s for d, s in zip(days, states, strict=True)]
    components = {}
    cost = planner._finalized_program_recovery_cost(days, states, config, target,
        session_states=timed, diagnostic_components=components,
        target_distance_trajectory=targets)
    return cost, components


def rollout(days, states, request, config, targets, *, state_at, continuation_days,
            distance_policy='stock', role_overrides=None, opportunity_mean_session=None):
    if distance_policy not in {'stock', 'finalized'}:
        raise ValueError('Unknown continuation distance policy')
    from scripts.continuation_distance import continuation_distance_policy
    if not days or len(days) != len(states):
        raise ValueError('Require nonempty aligned days and raw states')
    dates = [states[0].as_of.date()+timedelta(days=i) for i in range(len(targets))]
    context = (continuation_distance_policy(states[0], dates, targets, config)
               if distance_policy == 'finalized' else nullcontext(None))
    with context as intentions:
        return _rollout(days, states, request, config, targets, state_at=state_at,
                        continuation_days=continuation_days, intentions=intentions,
                        role_overrides=role_overrides or {}, opportunity_mean_session=opportunity_mean_session)


def _rollout(days, states, request, config, targets, *, state_at, continuation_days, intentions, role_overrides,
             opportunity_mean_session=None):
    if not isinstance(continuation_days, int) or continuation_days < 1:
        raise ValueError('Require a positive integer continuation length')
    if len(days) != len(states) or not days:
        raise ValueError('Require nonempty aligned days and raw states')
    if len(targets) < len(days) + continuation_days:
        raise ValueError('Target trajectory must cover the entire continuation')
    opening = states[0]
    observed = opening.recent_load.trailing_28d
    if not observed.activity_count or observed.distance_miles <= 0:
        raise ValueError('Prototype requires established observed session history')
    mean_session = observed.distance_miles / observed.activity_count
    # Offline matched-grid diagnostic only. This replaces the opportunity-gap
    # reference, never history, recovery/load states, doses or score evidence.
    if opportunity_mean_session is not None:
        if not isfinite(opportunity_mean_session) or opportunity_mean_session <= 0:
            raise ValueError('Opportunity mean must be finite and positive')
        mean_session = opportunity_mean_session
    origin = opening.as_of
    boundary = origin.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=len(days))
    end = boundary + timedelta(days=continuation_days)
    if any(not boundary.date() <= day < end.date() or role not in {'easy','quality','long','rest'}
           for day, role in role_overrides.items()):
        raise ValueError('Role overrides must stay within the continuation and use a known role')
    planned = runs(days)
    if any(r.planned_for is None for r in planned):
        raise ValueError('Every run must have an exact planned timestamp')
    latest = max((r.planned_for for r in planned), default=None)
    if latest is None:
        if opening.days_since_last_run is None:
            raise ValueError('Prototype needs a last-run phase anchor')
        latest = origin - timedelta(days=opening.days_since_last_run)
    combined = list(days)
    raw_states = list(states)
    tail_by_date = {}
    attempts = []
    cursor = latest
    load_cache, state_cache = {}, {}
    while True:
        # Use physical time, never the hard/soft split, to choose demand.
        # Otherwise promoting the first soft session changes its next gap.
        offset = min(len(targets)-1, max(0, (cursor.date()-origin.date()).days))
        target = sum(targets[offset]) / 2
        if not isfinite(target) or target <= 0:
            raise ValueError('Prototype requires positive finite target demand')
        conditional = EXPECTED_COMPLETION_STATE.get()
        if conditional is not None and opportunity_mean_session is None:
            # At the same physical timestamp, hypothetical completed work and
            # identical actual work must yield the same cadence reference.
            # Include all prior soft/hard work, not only opening observations.
            mean_session = conditional.mean_session_after(cursor, planned)
        gap = timedelta(days=7 / min(7., max(1., target / mean_session)))
        cursor = max(boundary, cursor + gap)
        if cursor >= end:
            break
        raw = state_at(cursor)
        next_raw = state_at(cursor + gap)
        role = planner._elapsed_workout_role([raw], planned, config,
            next_state_options=[next_raw], recommendation_load_cache=load_cache,
            projected_state_cache=state_cache)
        role = role_overrides.get(cursor.date(), role)
        if role == 'rest':
            attempts.append({'at': cursor.isoformat(), 'requested_role': role,
                             'selected_type': 'rest', 'opening_residual': None})
            continue
        projected, result = planner._select_budgeted_timed_recommendation(
            [raw], planned, request, config, weekly_role=role,
            recommendation_load_cache=load_cache, projected_state_cache=state_cache)
        attempts.append({'at': cursor.isoformat(), 'requested_role': role,
                         'selected_type': result.workout_type.value,
                         'opening_residual': projected.recovery_residual_load})
        if result.workout_type == planner.WorkoutType.REST:
            continue
        if result.distance_range_miles is None:
            raise ValueError('Continuation run is missing distance limits')
        if intentions is not None:
            from scripts.continuation_distance import finalize_continuation_distance
            result, facts = finalize_continuation_distance(result, projected, config,
                targets[(cursor.date()-origin.date()).days], intentions[cursor.date()])
            attempts[-1]['distance_finalization'] = facts
            attempts[-1]['selected_type'] = result.workout_type.value
        low, high = result.distance_range_miles
        distance = rounded_distance_target(planner._midpoint(result), minimum=low, maximum=high)
        result = result.model_copy(update={'distance_target_miles': distance})
        if cursor.date() in tail_by_date:
            raise ValueError('Prototype must not generate two sessions on one date')
        planned.append(result)
        tail_by_date[cursor.date()] = result
    for i in range(continuation_days):
        at = boundary + timedelta(days=i)
        result = tail_by_date.get(at.date())
        combined.append(planner.WeeklyScheduleDay(date=at.date(),
            planned_at=result.planned_for if result else None, recommendation=result,
            day_role=('long_run' if result and result.workout_type == planner.WorkoutType.LONG
                      else 'quality_run' if result and result.workout_type in planner.QUALITY_WORKOUT_TYPES
                      else 'medium_long_run' if result and result.planning_role == 'medium_long'
                      else 'easy_run' if result else 'rest_day'),
            rationale='Offline continuation estimate; not an issued prescription.'))
        raw_states.append(state_at(at))
    return combined, raw_states, attempts


def compare_candidate(inputs, days, score, *, state_at, continuation_days, distance_policy='stock',
                      timing_policy='greedy', complete_role_costs=False,
                      support_policy='aggregate', completed_runs=None, support_window_days=14,
                      repair_opportunities=False, compare_opportunity_doses=False,
                      history_source='observed', selected_program_observer=None,
                      opportunity_mean_session=None):
    if compare_opportunity_doses and not repair_opportunities:
        raise ValueError('Opportunity dose comparison requires opportunity repair')
    if support_policy not in {'aggregate', 'local', 'bounded'}:
        raise ValueError('Unknown support policy')
    if support_policy != 'aggregate' and (not complete_role_costs or completed_runs is None):
        raise ValueError('Local support requires complete role costs and explicit completed history')
    if repair_opportunities and (timing_policy != 'coordinated' or distance_policy != 'finalized'
                                 or support_policy != 'bounded' or not complete_role_costs):
        raise ValueError('Opportunity repair requires coordinated finalized bounded-support scoring')
    started = perf_counter()
    projector = inputs['expected_target_projector']
    required = len(days) + continuation_days
    targets = projector(runs(days), required)
    def make(overrides):
        return rollout(days, inputs['daily_states'], inputs['request'], inputs['config'],
            targets, state_at=state_at, continuation_days=continuation_days,
            distance_policy=distance_policy, role_overrides=overrides,
            opportunity_mean_session=opportunity_mean_session)
    def evaluate(continuation):
        return score_continuation(inputs, days, score, targets, continuation,
            state_at=state_at, continuation_days=continuation_days, distance_policy=distance_policy,
            complete_role_costs=complete_role_costs, support_policy=support_policy,
            completed_runs=completed_runs, support_window_days=support_window_days,
            history_source=history_source)
    if timing_policy == 'coordinated':
        if not complete_role_costs:
            raise ValueError('Role selection requires existing support/shape costs, not the partial diagnostic score')
        from scripts.coordinated_continuation import coordinate
        selected, result, metrics = (coordinate(make, evaluate, causal_opportunities=True)
            if EXPECTED_COMPLETION_STATE.get() is not None else coordinate(make, evaluate))
        # Publication must evaluate the exact sequence the coordinator chose.
        verified = evaluate(selected)
        assert abs(verified['combined_cost'] - result['combined_cost']) < 1e-8
        result['coordination'] = metrics
        if repair_opportunities:
            from scripts.continuation_opportunity_repair import polish
            selected, result, insertion = polish(selected, result, inputs=inputs, targets=targets,
                hard_length=len(days), state_at=state_at, evaluate=evaluate,
                compare_doses=compare_opportunity_doses)
            result['coordination'] = metrics
            result['opportunity_repair'] = insertion
    elif timing_policy == 'greedy':
        selected = make({})
        result = evaluate(selected)
    else:
        raise ValueError('Unknown continuation timing policy')
    result['timing_policy'] = timing_policy
    result['seconds'] = perf_counter()-started
    if selected_program_observer is not None:
        selected_program_observer(selected)
    return result


def score_continuation(inputs, days, score, targets, continuation, *, state_at,
                       continuation_days, distance_policy, complete_role_costs=False,
                       support_policy='aggregate', completed_runs=None, support_window_days=14,
                       history_source='observed'):
    """Single objective for coordination and the reported final comparison."""
    started = perf_counter()
    projector = inputs['expected_target_projector']
    combined, states, attempts = continuation
    # Hold the recovery reference fixed to isolate the continuation change.
    target = inputs['target_distance_range']
    recovery, components = continued_recovery(combined, states, inputs['config'], target,
                                             state_at=state_at)
    trajectory_recovery, trajectory_components = continued_recovery(combined, states,
        inputs['config'], target, state_at=state_at, targets=targets)
    timed_options = [[state_at(d.recommendation.planned_for)] if d.recommendation
                     and d.recommendation.planned_for else [s]
                     for d, s in zip(combined, states, strict=True)]
    coaching_components = {}
    combined_coaching, scored_sessions = planner._finalized_program_coaching_cost(combined, states,
        timed_options, inputs['request'], inputs['config'], target,
        diagnostic_components=coaching_components)
    old_coaching, _ = planner._finalized_program_coaching_cost(days, inputs['daily_states'],
        timed_options[:len(days)], inputs['request'], inputs['config'], target)
    assert abs(old_coaching-score['coaching_cost']) < 1e-8
    old_recovery, old_components = continued_recovery(days, inputs['daily_states'],
        inputs['config'], target, state_at=state_at)
    assert abs(old_recovery - score['finalized_recovery_cost']) < 1e-8
    ordinary = sum(planner.typical_easy_distance(states[0])) / 2
    soft_target = planner._average_target_range(targets)
    old_soft_target = planner._average_target_range(projector(runs(days), len(days)+28))
    old_rate = planner._soft_projected_program_rate(days, old_soft_target, ordinary,
        boundary_session=planner._recent_boundary_session(states[0])) or 0.
    sessions = [(r.planned_for - states[0].as_of).total_seconds()/86400 for r in runs(combined)]
    distances = [planner._midpoint(r) for r in runs(combined)]
    boundary_session = planner._recent_boundary_session(states[0])
    if boundary_session:
        sessions.insert(0, boundary_session[0])
        distances.insert(0, boundary_session[1])
    observed = states[0].recent_load.trailing_28d
    conditional = EXPECTED_COMPLETION_STATE.get()
    closing_mean = (conditional.mean_session_after(runs(combined)[-1].planned_for, runs(combined))
                    if conditional is not None else observed.distance_miles/observed.activity_count)
    gap_days = 7 * closing_mean / (sum(soft_target)/2)
    rate = sum(distances)*7 / (sessions[-1]-sessions[0]+gap_days)
    def violation(value, limits):
        return max(0., limits[0]-value, value-limits[1])
    # Replace the recurring-rate term, then replace recovery/coaching below.
    # The historical control retains hard-prefix funding. The explicitly
    # scoped full-path prototype below removes that quota after accounting
    # for date-local volume and the full recurring rate. Neither path is
    # cleared for production selection by a fixed-program comparison alone.
    old_violation = score['target_violation']
    new_violation = old_violation - violation(old_rate, old_soft_target) + violation(rate, soft_target)
    volume_details = None
    funding_details = None
    volume_scorer = VOLUME_PATH_SCORER.get()
    funding_scorer = CONTINUATION_FUNDING_SCORER.get()
    if funding_scorer is not None and volume_scorer is None:
        raise ValueError('Full-path funding requires additive physical volume accounting')
    if volume_scorer is not None:
        hard_volume = volume_scorer(days, inputs['daily_states'], targets[:len(days)], inputs['config'])
        full_volume = volume_scorer(combined, states, targets, inputs['config'])
        residual_squared = old_violation**2 - hard_volume['squared_violation']
        if residual_squared < -1e-8:
            raise ValueError('Hard volume was not priced consistently before continuation')
        residual = max(0., residual_squared)**.5 - violation(old_rate, old_soft_target) + violation(rate, soft_target)
        if residual < -1e-8:
            raise ValueError('Continuation replacement removed more rate violation than was scored')
        new_violation = (max(0.,residual)**2 + full_volume['squared_violation'])**.5
        volume_details = dict(hard=hard_volume,full=full_volume,residual_violation=max(0.,residual))
        if funding_scorer is not None:
            # The diagnostic prototype replaces only the inherited hard quota.
            # The shared objective is used by coordination, selection, and its
            # independent rescore; no publication-only discount is introduced.
            funding_details = funding_scorer(full_volume, rate, soft_target,
                legacy_residual=max(0., residual))
            new_violation = funding_details['target_violation']
            volume_details['residual_violation'] = funding_details['rate_violation']
    replacement = score['total_cost'] - old_recovery + recovery + planner.TARGET_FIT_UNIT * (
        (new_violation/ordinary)**2 - (old_violation/ordinary)**2)
    rate_only = replacement - recovery + old_recovery
    combined_cost = replacement - recovery + trajectory_recovery - old_coaching + combined_coaching
    support, shape = score['support_violation'], score['shape_violation']
    if complete_role_costs:
        # Reuse the detailed scorer's existing role/support terms, without
        # reallocating the committed prefix or inventing another approximation.
        # No target projector is required for these two components. Discard its
        # target/recovery results: those are already scored above with the tail.
        token = planner._FINALIZED_SCORING_TRIAL.set(None)
        try:
            role_joint = planner._joint_candidate_program_cost(
                tuple(s.offset for s in scored_sessions), states, timed_options,
                inputs['request'], inputs['config'], target,
                _allocated_override=combined, _sessions_override=scored_sessions)
        finally:
            planner._FINALIZED_SCORING_TRIAL.reset(token)
        support, shape = role_joint[1:3]
    local_support = None
    if support_policy != 'aggregate':
        if not complete_role_costs:
            raise ValueError('Local support must replace the full aggregate term')
        from scripts.aerobic_distribution import continuation_support
        local_support = continuation_support(combined, states, completed_runs,
                                             window_days=support_window_days, policy=support_policy,
                                             history_source=history_source)
        # Replace, never stack. Preserve the separate per-run shortfall and
        # fragmentation terms for this single-variable distribution experiment.
        support += local_support['violation'] - planner._rolling_aerobic_support_violation(combined, states)
    role_delta = planner.PROGRAM_FIT_UNIT * (
        (support-score['support_violation'])*planner.AEROBIC_SUPPORT_FIT_FRACTION
        + shape-score['shape_violation'])
    combined_cost += role_delta
    return {'continuation_days': continuation_days, 'seconds': perf_counter()-started,
        'combined_target_violation': new_violation,
        'combined_target_cost': planner.TARGET_FIT_UNIT * (new_violation / ordinary)**2,
        'volume_path': volume_details,
        'funding_accounting': funding_details,
        'support_policy': support_policy, 'local_support': local_support,
        'complete_role_costs': complete_role_costs, 'role_cost_delta': role_delta,
        'combined_support_violation': support, 'combined_shape_violation': shape,
        'distance_policy': distance_policy, 'rate_only_cost': rate_only,
        'trajectory_recovery_only_cost': score['total_cost']-old_recovery+trajectory_recovery,
        'combined_cost': combined_cost, 'combined_coaching_cost': combined_coaching,
        'combined_coaching_components': coaching_components,
        'trajectory_recovery_cost': trajectory_recovery,
        'trajectory_recovery_components': trajectory_components,
        'ablation_cost': replacement, 'recovery_only_cost': score['total_cost']-old_recovery+recovery,
        'recovery_cost': recovery,
        'recovery_components': components, 'replaced_recovery_components': old_components,
        'policy_rate': rate, 'old_policy_rate': old_rate, 'soft_target': soft_target,
        'attempts': attempts,
        'sessions': [(r.planned_for.isoformat(), r.workout_type.value, planner._midpoint(r))
                     for r in runs(combined[len(days):])]}


def main():
    from argparse import ArgumentParser
    from datetime import datetime
    from functools import lru_cache
    import json
    from pathlib import Path
    from scripts.audit_rollover_distance import investigate
    from scripts.replay_synthetic_easy_dose import synthetic_history
    from run_analysis import adherence_projection as projection

    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output-json', type=Path, required=True)
    parser.add_argument('--lengths', type=int, nargs='+', default=[14, 21, 28])
    parser.add_argument('--distance-policy', choices=['stock', 'finalized'], default='stock')
    parser.add_argument('--timing-policy', choices=['greedy', 'coordinated'], default='greedy')
    parser.add_argument('--complete-role-costs', action='store_true')
    parser.add_argument('--support-policy', choices=['aggregate', 'local', 'bounded'], default='aggregate')
    parser.add_argument('--support-window-days', type=int, default=14)
    parser.add_argument('--repair-opportunities', action='store_true')
    parser.add_argument('--compare-opportunity-doses', action='store_true')
    args = parser.parse_args()
    if args.output_json.exists() or any(n < 1 or n > 28 for n in args.lengths):
        parser.error('Require a new output and lengths 1–28 within the prepared target trajectory')
    if args.compare_opportunity_doses and not args.repair_opportunities:
        parser.error('Opportunity dose comparison requires --repair-opportunities')
    if args.timing_policy == 'coordinated' and not args.complete_role_costs:
        parser.error('Coordinated selection requires --complete-role-costs')
    if args.support_policy == 'local' and (not args.complete_role_costs or args.support_window_days <= 7):
        parser.error('Local support requires --complete-role-costs and a window longer than seven days')
    if args.support_policy == 'bounded' and not args.complete_role_costs:
        parser.error('Bounded support requires --complete-role-costs')
    if args.repair_opportunities and (args.timing_policy != 'coordinated' or args.distance_policy != 'finalized'
                                      or args.support_policy != 'bounded' or not args.complete_role_costs):
        parser.error('Opportunity repair requires coordinated finalized bounded-support scoring')
    report = json.loads(args.report.read_text())
    template, history = synthetic_history(datetime.fromisoformat(report['realized_execution']['start_at']))
    factories, cache = {}, {}

    def observer(inputs, days, score):
        opening = inputs['daily_states'][0]
        if args.support_policy != 'aggregate':
            # This saved-input panel covers the two decisions before the first
            # simulated upload. Refuse stale seed history if reused elsewhere.
            for n in (7, 14, 28):
                observed = projection._load_window(history, opening.as_of, n)
                captured = getattr(opening.recent_load, f'trailing_{n}d')
                if (observed.activity_count != captured.activity_count
                        or abs(observed.distance_miles - captured.distance_miles) > 1e-8):
                    raise ValueError('Seed history does not match this decision; supply its actual completed history')
        origin = opening.as_of.isoformat()
        if origin not in factories:
            settings = inputs['config'].get('coaching', {})
            capacity = opening.recent_load.capacity_reference_miles
            known = {s.as_of: s for options in inputs['daily_state_options'] for s in options}
            known.update({s.as_of: s for s in inputs['daily_states']})

            @lru_cache(maxsize=None)
            def state_at(at):
                if at in known:
                    return known[at]
                return projection._state_at(template, history, at, capacity,
                    easy_baseline_half_life_days=float(settings.get('capacity_retention_half_life_days', 84)),
                    fatigue_half_life_days=float(settings.get('continuous_fatigue_half_life_days', 7)))
            factories[origin] = state_at
        key = (origin, len(days), tuple(r.model_dump_json() for r in runs(days)))
        if key not in cache:
            cache[key] = [compare_candidate(inputs, days, score, state_at=factories[origin],
                continuation_days=n, distance_policy=args.distance_policy,
                timing_policy=args.timing_policy, complete_role_costs=args.complete_role_costs,
                support_policy=args.support_policy, completed_runs=history,
                support_window_days=args.support_window_days,
                repair_opportunities=args.repair_opportunities,
                compare_opportunity_doses=args.compare_opportunity_doses) for n in args.lengths]
        return cache[key]

    result = investigate(report, candidate_observer=observer)
    result['prototype_scope'] = __doc__
    result['limitations'] = ['Fixed hard calendars/doses; soft-role search is optional. No daily-rollout clearance.',
        ('Bounded ordinary-support credit replaces aggregate support; per-run shortfall/shape penalties remain.'
         if args.support_policy == 'bounded' else
         'Local ordinary-support windows replace aggregate support; per-run shortfall/shape penalties remain.'
         if args.support_policy == 'local' else
         'Existing support/shape included, but support is horizon-aggregate, not locally distributed.'
         if args.complete_role_costs else 'Future support/shape omitted: diagnostic comparison only.'),
        'Future funding uses an average recurring rate; target normalization remains incomplete.',
        'Finalized mode shares distance machinery, but local intent selection is a cheap base policy, not full future allocation.',
        'Legacy residual closeout remains at the far edge; length convergence must be checked.',
        'Optional opportunity repair tries one best native easy insertion on unused soft dates; not a complete calendar search.',
        'Optional opportunity dose comparison uses the existing history-led menu; no lower floor or fee change.',
        'No production changes, weights, or release claims.']
    result['unique_candidates'] = len(cache)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    for row in result['probes']:
        print(row['name'], {dose: [(p['continuation_days'], round(p['ablation_cost'], 6),
            round(p['combined_cost'], 6)) for p in data['continuation_probe']]
            for dose, data in row['fixed_tail_dose_probes'].items()})
    print('Seconds', result['seconds'], 'unique candidates', len(cache))


if __name__ == '__main__':
    main()
