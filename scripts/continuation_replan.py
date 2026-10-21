"""Offline bounded continuation selection in the real daily/upload workflow.

Beam and hard-program neighborhood scores propose a shortlist; the complete
continuation objective chooses among finalized programs. This does not run a costly rollout
at every beam node, or claim optimality beyond the disclosed shortlist. Publish
the exact selected hard prefix and independently rescore its materialized tail.
"""

from contextlib import contextmanager
from datetime import timedelta
from functools import lru_cache
from hashlib import sha256
import json
from math import isfinite
from time import perf_counter

from run_analysis import adherence_projection as projection
from run_analysis import weekly_schedule as planner
from run_analysis.aerobic_acceptance import assess_aerobic_exposure
from run_analysis.selection_objective import COMPLETED_SELECTION_HISTORY, FINALIZED_PROGRAM_SELECTOR
from scripts.audit_easy_dose_program import capture_allocations
from scripts.calendar_neighbor_prototype import NoAcceptableCalendarError
from scripts.typed_continuation_prototype import compare_candidate, runs, score_continuation
from scripts.full_score_refinement import REFINEMENT_BUDGET, refine
from scripts.continuation_checks import check_sequence
from scripts.prior_program_candidate import recover_prior_program
from scripts.finalized_distance_search import DISTANCE_REFINEMENT, refine_distances


from run_analysis.adaptive.legacy_program_identity import signature, fingerprint


def shortlist(record, *, limit):
    """Always include both search lanes, then frequency-diverse cheap finalists.

    The already materialized publication seed uses one additional budget slot.
    No legacy score is compared numerically with the continuation objective.
    """
    if not 3 <= limit <= 12:
        raise ValueError('Finalist budget must be 3–12 including the publication seed')
    rows = [r for r in record['evaluated'] if isfinite(r['total_cost'])
            and not r.get('acceptance', {}).get('rejected', False)
            and not r.get('unrepresentable_distance_target')]
    ordered = sorted(rows, key=lambda r: (r['total_cost'], len(r['offsets']), r['offsets']))
    selected, seen = [], set()
    def add(row):
        if row is None or len(selected) >= limit - 1:
            return
        key = (tuple(row['offsets']), tuple(sorted((int(k), v) for k, v in row.get('roles', {}).items())))
        if key not in seen:
            selected.append(row)
            seen.add(key)
    # Match role-specific scored rows, not just dates of a protected lane.
    for lane in ('unconstrained', 'stable'):
        target = record.get(lane)
        if target is not None:
            matches = [r for r in ordered if r['offsets'] == target['offsets']
                       and abs(r['total_cost'] - target['total_cost']) < 1e-8
                       and r['allocation_trace'] == target['allocation_trace']]
            if not matches:
                raise ValueError(f'Missing finalized {lane} lane evidence')
            add(matches[0])
    represented = {len(r['offsets']) for r in selected}
    for row in ordered:
        count = len(row['offsets'])
        if count not in represented:
            add(row)
            represented.add(count)
    for row in ordered:
        add(row)
    return selected


def stable_prefix(days, inputs):
    allowed_prior = set(inputs['prior_run_offsets']) - set(inputs['forced_rest_offsets']) - set(inputs['completed_run_offsets'])
    limit = planner.NEAR_TERM_STABILITY_DECISION_OFFSETS
    expected = {i: inputs['prior_roles_by_offset'][i] for i in allowed_prior if i < limit}
    actual = {}
    for i, d in enumerate(days[:limit]):
        r = d.recommendation
        if r and r.workout_type != planner.WorkoutType.REST:
            actual[i] = ('long' if r.workout_type == planner.WorkoutType.LONG else
                         'quality' if r.workout_type in planner.QUALITY_WORKOUT_TYPES else 'easy')
    return bool(inputs['prior_run_offsets']) and actual == expected


def combined_breakdown(hard_score, continuation, ordinary):
    """Honest combined components; hard-only subsidiary shape fields are omitted."""
    result = hard_score.model_copy(update={
        'total_cost': continuation['combined_cost'],
        'coaching_cost': continuation['combined_coaching_cost'],
        'target_violation': continuation['combined_target_violation'],
        'support_violation': continuation['combined_support_violation'],
        'shape_violation': continuation['combined_shape_violation'],
        'finalized_recovery_cost': continuation['trajectory_recovery_cost'],
        'recovery_cost_components': continuation['trajectory_recovery_components'],
        'fragmentation_violation': None, 'long_shape_violation': None,
        'medium_long_shape_violation': None,
    })
    recomputed = planner._program_selection_cost((result.target_violation, result.support_violation,
        result.shape_violation, result.finalized_recovery_cost), result.coaching_cost, ordinary,
        distance_switching_cost=result.distance_switching_cost)
    if abs(recomputed - result.total_cost) > 1e-8:
        raise ValueError('Combined score components do not reproduce the selection objective')
    return result


def hard_funding_evidence(days, inputs):
    """Mirror the publication gate on the exact finalized hard program."""

    targets = inputs['expected_target_projector'](runs(days), len(days))
    if len(targets) != len(days):
        raise ValueError('Expected target trajectory must match the hard horizon')
    integrated = planner._integrated_target_range(targets)
    completed = sum(inputs.get('completed_miles_by_offset', {}).values())
    remaining = (
        max(0.0, integrated[0] - completed),
        max(0.0, integrated[1] - completed),
    )
    boundary = planner._recent_boundary_session(inputs['daily_states'][0])
    allowance = max(
        sum(planner.typical_easy_distance(inputs['daily_states'][0])) / 2,
        (boundary or (0.0, 0.0))[1],
    )
    violation = planner._hard_horizon_funding_violation(
        days,
        remaining,
        boundary_allowance_miles=allowance,
    )
    return dict(
        violation_miles=violation,
        target=remaining,
        boundary_allowance_miles=allowance,
        planned_miles=sum(planner._midpoint(item) for item in runs(days)),
    )


def select_program(planning_days, diagnostics, inputs, record, *, limit, continuation_days):
    started = perf_counter()
    origin = inputs['daily_states'][0].as_of
    evidence = COMPLETED_SELECTION_HISTORY.get()
    if evidence is None:
        raise ValueError('Continuation selection requires an explicit completed-history ledger')
    history = evidence['runs']
    if any(r.start_time >= origin for r in history):
        raise ValueError('Completed selection history includes future work')
    if inputs['completed_miles_by_offset']:
        raise ValueError('This offline trial requires the existing next-day post-upload boundary')
    if record['as_of'] != origin.isoformat():
        raise ValueError('Calendar shortlist belongs to another decision')
    for n in (7, 14, 28):
        actual = projection._load_window(history, origin, n)
        captured = getattr(inputs['daily_states'][0].recent_load, f'trailing_{n}d')
        if actual.activity_count != captured.activity_count or abs(actual.distance_miles - captured.distance_miles) > 1e-8:
            raise ValueError('Completed history does not reproduce opening load')
    known = {s.as_of: s for choices in inputs['daily_state_options'] for s in choices}
    known.update({s.as_of: s for s in inputs['daily_states']})
    config = inputs['config'].get('coaching', {})
    @lru_cache(maxsize=None)
    def state_at(at):
        if at in known:
            return known[at]
        return projection._state_at(evidence['template'], history, at,
            inputs['daily_states'][0].recent_load.capacity_reference_miles,
            easy_baseline_half_life_days=float(config.get('capacity_retention_half_life_days', 84)),
            fatigue_half_life_days=float(config.get('continuous_fatigue_half_life_days', 7)))

    common = {k: inputs[k] for k in ('daily_states', 'daily_state_options', 'request', 'config',
        'target_distance_range', 'completed_miles_by_offset', 'expected_target_projector',
        'prior_distance_midpoints_by_date')}
    ordinary = sum(planner.typical_easy_distance(inputs['daily_states'][0])) / 2
    def hard_score(offsets, *, days=None, roles=None):
        details = {}
        sessions = None
        if days is not None:
            sessions=[]
            for i,d in enumerate(days):
                rec=d.recommendation
                if rec is None or rec.workout_type==planner.WorkoutType.REST:
                    continue
                at=rec.planned_for or d.planned_at
                native=next((state for state in inputs['daily_state_options'][i] if state.as_of==at),None)
                if native is None and inputs['daily_states'][i].as_of==at:
                    native=inputs['daily_states'][i]
                if native is None:
                    raise ValueError('Fixed hard program has no native timed state')
                sessions.append(planner._CandidateSession(i,native,rec))
        with capture_allocations() as capture:
            joint = planner._joint_candidate_program_cost(tuple(offsets), **common,
                role_overrides_by_offset=roles, reconcile_allocated_prefix=days is None,
                _allocated_override=days, _sessions_override=sessions,
                diagnostic_details=details, _collect_acceptance_facts=True)
        score = planner._score_breakdown(tuple(offsets), joint,
            float(details.get('finalized_coaching_cost', 0.)), ordinary, details)
        return days if days is not None else capture.get('days'), score, details

    seeds = [{'offsets': [i for i, d in enumerate(planning_days) if d.recommendation
                         and d.recommendation.workout_type != planner.WorkoutType.REST],
              'source': 'publication_seed', 'days': planning_days}]
    seeds += [{**r, 'source': 'surrogate_finalist'} for r in shortlist(record, limit=limit)]
    prior_seed, prior_audit = recover_prior_program(inputs, continuation_days)
    if prior_seed is not None:
        # One extra cheap fixed-sequence check, not a replacement for any of
        # the existing independent/stable/regrowth proposal slots.
        seeds.append(prior_seed)
    rows, candidates, seen, materialized_cache = [], [], {}, {}
    def evaluate(seed):
        stamp = perf_counter()
        material_key = (tuple(seed['offsets']), tuple(sorted((seed.get('roles') or {}).items())))
        if seed.get('days') is not None:
            days, hard, details = hard_score(seed['offsets'], days=seed['days'], roles=seed.get('roles'))
        else:
            if material_key not in materialized_cache:
                materialized_cache[material_key] = hard_score(seed['offsets'], roles=seed.get('roles'))
            days, hard, details = materialized_cache[material_key]
        row = {'source': seed['source'], 'offsets': seed['offsets'], 'legacy_cost': hard.total_cost,
               'roles': seed.get('roles', {})}
        rows.append(row)
        if not isfinite(hard.total_cost) or days is None:
            row['rejected'] = 'unrepresentable_hard_program'
            return None
        blocked = set(inputs['forced_rest_offsets']) | set(inputs['completed_run_offsets'])
        if (len(days) != len(inputs['daily_states'])
                or [d.date for d in days] != [s.as_of.date() for s in inputs['daily_states']]
                or any(i in blocked and d.recommendation and d.recommendation.workout_type != planner.WorkoutType.REST
                       for i, d in enumerate(days))):
            row['rejected'] = 'hard_date_constraint'
            return None
        key = fingerprint(days)
        row['hard_fingerprint'] = key
        parent = seed.get('tail_parent')
        fixed = seed.get('fixed_program')
        fixed_prefix=seed.get('fixed_prefix')
        tail_key = (fingerprint(fixed[len(days):]) if fixed is not None else
                    ('translated_prefix',fingerprint(fixed_prefix),seed.get('terminal_missing_days'))
                    if fixed_prefix is not None else
                    parent['tail_fingerprint'] if parent is not None else None)
        evaluation_key = (key, tail_key)
        if evaluation_key in seen:
            row['duplicate'] = True
            return seen[evaluation_key]
        seen[evaluation_key] = None
        assessment = assess_aerobic_exposure(**details['aerobic_acceptance_facts'],
            planning_mode=inputs['target_evidence'].planning_mode.value)
        if assessment.rejected or assessment.reason == 'missing_context':
            row['rejected'] = 'hard_acceptance:' + assessment.reason
            return None
        funding = hard_funding_evidence(days, inputs)
        row['hard_funding'] = funding
        if (
            funding['violation_miles']
            > planner.HARD_FUNDING_ROUNDING_TOLERANCE_MILES + 1e-9
        ):
            row['rejected'] = 'hard_funding_gate'
            return None
        if parent is None and fixed is None:
            captured = []
            continuation = compare_candidate(inputs, days, hard.model_dump(mode='json'), state_at=state_at,
                continuation_days=continuation_days, distance_policy='finalized', timing_policy='coordinated',
                complete_role_costs=True, support_policy='bounded', completed_runs=history,
                history_source=evidence['source'], repair_opportunities=True, compare_opportunity_doses=True,
                selected_program_observer=captured.append)
            assert len(captured) == 1
            materialized = captured[0]
            if fixed_prefix is not None:
                required=len(days)+continuation_days
                missing=int(seed['terminal_missing_days'])
                fresh=materialized[0]
                if ([d.date for d in fixed_prefix] != [d.date for d in fresh[:len(fixed_prefix)]]
                        or len(fresh)!=required or len(fixed_prefix)!=required-missing):
                    row['rejected']='translated_prefix_date_contract'
                    return None
                repairs=[]
                for suffix_days in range(missing,min(8,continuation_days)+1):
                    boundary=required-suffix_days
                    program=[*fixed_prefix[:boundary],*fresh[boundary:]]
                    checked=check_sequence(program,inputs,state_at,())
                    if not checked['checks_pass']:
                        continue
                    targets=inputs['expected_target_projector'](runs(days),required)
                    scored=score_continuation(inputs,days,hard.model_dump(mode='json'),targets,
                        (program,[state_at(origin+timedelta(days=i)) for i in range(required)],[]),
                        state_at=state_at,continuation_days=continuation_days,distance_policy='finalized',
                        complete_role_costs=True,support_policy='bounded',completed_runs=history,
                        history_source=evidence['source'])
                    scored['opportunity_repair']=dict(status='checked',baseline_checks=checked,
                        selected_offset=None,seconds=0.)
                    repairs.append((scored['combined_cost'],suffix_days,program,scored))
                if not repairs:
                    row['rejected']='terminal_repair_guard_failure'
                    return None
                _,suffix_days,program,continuation=min(repairs,key=lambda item:(item[0],item[1]))
                materialized=(program,[state_at(origin+timedelta(days=i)) for i in range(required)],[])
                row['terminal_repair']=dict(missing_days=missing,selected_suffix_days=suffix_days,
                    evaluated=len(repairs),maximum_suffix_days=min(8,continuation_days))
        else:
            # Optimize the hard calendar conditional on a concrete soft tail.
            # This is an exact full-program score, not a hard-only surrogate.
            # Reproject every following session: a borrowed tail is not assumed
            # safe just because it was feasible for its former hard prefix.
            materialized = ((fixed, [state_at(origin + timedelta(days=i)) for i in range(len(fixed))], [])
                if fixed is not None else (days + parent['program'][0][len(days):],
                            parent['program'][1], parent['program'][2]))
            checked = check_sequence(materialized[0], inputs, state_at, ())
            if not checked['checks_pass']:
                row.update(rejected='continuation_guard_failure', failures=checked['failed_offsets'])
                return None
            targets = inputs['expected_target_projector'](runs(days), len(days) + continuation_days)
            continuation = score_continuation(inputs, days, hard.model_dump(mode='json'), targets,
                materialized, state_at=state_at, continuation_days=continuation_days,
                distance_policy='finalized', complete_role_costs=True, support_policy='bounded',
                completed_runs=history, history_source=evidence['source'])
            continuation['opportunity_repair'] = dict(status='checked', baseline_checks=checked,
                selected_offset=None, seconds=0.)
            if parent is not None:
                row['fixed_tail_parent'] = parent['row']['hard_fingerprint']
            else:
                row['revalidated_prior_program'] = True
        assert signature(materialized[0][:len(days)]) == signature(days)
        checked = continuation['opportunity_repair']
        if checked['status'] != 'checked':
            row.update(rejected='continuation_guard_failure', failures=checked['baseline_checks']['failed_offsets'])
            return None
        score = combined_breakdown(hard, continuation, ordinary)
        is_stable = stable_prefix(days, inputs)
        row.update(combined_cost=score.total_cost, stable_prefix=is_stable,
            policy_rate=continuation['policy_rate'], soft_target=continuation['soft_target'],
            insertion_offset=checked['selected_offset'], insertion_seconds=checked['seconds'],
            seconds=perf_counter() - stamp)
        candidate = dict(days=days, hard=hard, continuation=continuation, program=materialized,
                         score=score, stable=is_stable, row=row,
                         tail_fingerprint=fingerprint(materialized[0][len(days):]))
        row['tail_fingerprint'] = candidate['tail_fingerprint']
        if seed['source'] == 'distance_neighbor':
            row['hard_program'] = signature(days)
        candidates.append(candidate)
        seen[evaluation_key] = candidate
        return candidate
    prior_candidate = None
    for seed in seeds:
        evaluated = evaluate(seed)
        if seed is prior_seed:
            prior_candidate = evaluated
            prior_audit.update(status='eligible' if evaluated is not None else 'rejected',
                evaluation=rows[-1].copy())
    if not candidates:
        error = NoAcceptableCalendarError(origin, 'No continuation finalist passed the existing hard and downstream guards')
        error.continuation_evidence = {'as_of': origin.isoformat(), 'failed': True,
            'finalists': rows, 'seconds': perf_counter() - started}
        raise error
    initial_candidates = list(candidates)
    distance_policy = DISTANCE_REFINEMENT.get()
    candidate_distance_refinement = (
        refine_distances(
            initial_candidates,
            evaluate,
            inputs,
            state_at,
            budget=distance_policy['candidate_budget'],
            visible_days=distance_policy['visible_days'],
            seed_candidates=initial_candidates,
            max_sweeps=1,
        )
        if distance_policy is not None
        else None
    )

    def refresh_frontier_tails(source):
        """Regenerate tails for the current independent/stable hard frontier."""

        ordered = sorted(candidates, key=lambda x: x['score'].total_cost)
        refresh = [
            ordered[0],
            next((x for x in ordered if x['stable']), ordered[0]),
        ]
        refreshed = set()
        for item in refresh:
            key = item['row']['hard_fingerprint']
            if key in refreshed:
                continue
            evaluate(
                dict(
                    offsets=list(item['score'].offsets),
                    days=item['days'],
                    source=source,
                    roles=item['row'].get('roles', {}),
                )
            )
            refreshed.add(key)
        return len(refreshed)

    refinement = None
    if REFINEMENT_BUDGET.get() is not None:
        allowed = set(range(len(planning_days))) - set(inputs['forced_rest_offsets']) - set(inputs['completed_run_offsets'])
        refinement = refine(list(candidates), evaluate, allowed, **REFINEMENT_BUDGET.get())
        if REFINEMENT_BUDGET.get().get('fixed_tail'):
            # Refresh only the best independent and stable calendars. Retain
            # their old feasible tails if the heuristic rebuild scores worse.
            refinement['tail_refresh_calendars'] = refresh_frontier_tails(
                'full_score_neighbor'
            )
    post_calendar_distance_refinement = None
    if distance_policy is not None and distance_policy.get('post_calendar_budget'):
        calendar_candidates = [
            candidate for candidate in candidates
            if candidate['row']['source'] == 'full_score_neighbor'
        ]
        if calendar_candidates:
            # A calendar can look inferior only because its initial allocation
            # puts the right total mileage on the wrong easy dates. Give the
            # single best calendar neighbor a complete coordinate lane before
            # it is pruned. This is bounded independently from the selected-
            # frontier distance pass below.
            best_calendar = min(
                calendar_candidates,
                key=lambda candidate: (
                    candidate['score'].total_cost,
                    candidate['row']['hard_fingerprint'],
                ),
            )
            post_calendar_distance_refinement = refine_distances(
                [best_calendar],
                evaluate,
                inputs,
                state_at,
                budget=distance_policy['post_calendar_budget'],
                visible_days=distance_policy['visible_days'],
                seed_candidates=[best_calendar],
            )
    distance_refinement = (
        refine_distances(
            list(candidates),
            evaluate,
            inputs,
            state_at,
            budget=distance_policy['budget'],
            visible_days=distance_policy['visible_days'],
        )
        if distance_policy is not None
        else None
    )
    final_tail_refresh_count = (
        refresh_frontier_tails('distance_tail_refresh')
        if distance_refinement is not None
        else 0
    )
    independent = min(candidates, key=lambda x: (x['score'].total_cost, len(x['score'].offsets), x['score'].offsets))
    stable = min((x for x in candidates if x['stable']), key=lambda x: x['score'].total_cost, default=None)
    tolerance = planner._near_term_stability_tolerance(independent['score'].total_cost)
    winner = stable if stable is not None and stable['score'].total_cost <= independent['score'].total_cost + tolerance else independent
    if distance_refinement is not None:
        distance_refinement['candidate_refinement'] = candidate_distance_refinement
        distance_refinement['final_tail_refresh_calendars'] = final_tail_refresh_count
        distance_refinement['selected_locally_converged'] = any(
            lane['status'] == 'locally_converged'
            and lane['hard_fingerprint'] == winner['row']['hard_fingerprint']
            and lane['tail_fingerprint'] == winner['tail_fingerprint']
            for lane in distance_refinement['lanes'])
        distance_refinement['post_calendar_candidate_refinement'] = post_calendar_distance_refinement
    if prior_candidate is not None:
        prior_audit.update(revalidated_cost=prior_candidate['score'].total_cost,
            selected=fingerprint(winner['program'][0]) == fingerprint(prior_candidate['program'][0]),
            selected_cost=winner['score'].total_cost)
        # Same-category incumbent participation must not degrade the objective.
        # Cross-category choices still obey the existing calendar stability band.
        if (winner['stable'] == prior_candidate['stable']
                and winner['score'].total_cost > prior_candidate['score'].total_cost + 1e-8):
            raise ValueError('Selected plan is worse than its eligible same-category incumbent')
    if refinement is not None:
        refinement['selected_locally_converged'] = any(
            n['hard_fingerprint'] == winner['row']['hard_fingerprint'] and n['complete']
            and (not refinement.get('fixed_tail') or n['tail_fingerprint'] == winner['tail_fingerprint'])
            and n['best_cost'] >= winner['score'].total_cost - 1e-8
            for lane in refinement['lanes'] for n in lane['neighborhoods'])
        refinement['physical_cache_hits'] = sum(bool(r.get('duplicate')) for r in rows)
        refinement['unique_hard_materializations'] = len(materialized_cache)

    # Rescore actual publication, without allocation, coordination or opportunity
    # regeneration. The selected soft insertion must not disappear on replay.
    actual_days = winner['program'][0][:len(planning_days)]
    _, actual_hard, _ = hard_score(winner['score'].offsets, days=actual_days)
    targets = inputs['expected_target_projector'](runs(actual_days), len(actual_days) + continuation_days)
    published = score_continuation(inputs, actual_days, actual_hard.model_dump(mode='json'), targets,
        winner['program'], state_at=state_at, continuation_days=continuation_days,
        distance_policy='finalized', complete_role_costs=True, support_policy='bounded',
        completed_runs=history, history_source=evidence['source'])
    publication_score = combined_breakdown(actual_hard, published, ordinary)
    if abs(publication_score.total_cost - winner['score'].total_cost) > 1e-8:
        raise ValueError('Continuation search/publication score mismatch')
    if signature(actual_days) != signature(winner['days']) or published['sessions'] != winner['continuation']['sessions']:
        raise ValueError('Continuation search/publication prescription mismatch')
    report = {'scope': __doc__, 'finalist_budget': limit, 'continuation_days': continuation_days,
        'legacy_search_proposes_only': True, 'fees': 'unchanged', 'intent_floor': 'unchanged',
        'prior_program': prior_audit,
        'history_source': evidence['source'], 'history_count': len(history),
        'simulated_completion_count': sum(r.projected for r in history),
        'latest_completed_at': max((r.start_time.isoformat() for r in history), default=None),
        'finalists': rows, 'unique_continuations': len(candidates),
        'selected_hard_fingerprint': fingerprint(winner['days']),
        'published_hard_fingerprint': fingerprint(actual_days),
        'selected_program_fingerprint': fingerprint(winner['program'][0]),
        'search_cost': winner['score'].total_cost, 'publication_cost': publication_score.total_cost,
        'volume_path': published.get('volume_path'),
        'hard_sessions': [dict(planned_for=r.planned_for.isoformat(), workout_type=r.workout_type.value,
            midpoint_miles=planner._midpoint(r), distance_low_miles=r.distance_range_miles[0],
            distance_high_miles=r.distance_range_miles[1]) for r in runs(actual_days)],
        'soft_sessions': signature(winner['program'][0][len(actual_days):]),
        'full_score_refinement': refinement,
        'distance_refinement': distance_refinement,
        'seconds': perf_counter() - started}
    return dict(planning_days=actual_days, winner=winner['score'], published=publication_score,
        independent=independent['score'], stable=stable['score'] if stable else None,
        stable_selected=winner is stable and winner is not independent,
        tolerance=tolerance, evidence=report)


@contextmanager
def continuation_replan_policy(*, enabled=False, calendar_records=None, finalist_budget=6, continuation_days=28):
    records = []
    if enabled and (calendar_records is None or not 3 <= finalist_budget <= 12 or not 1 <= continuation_days <= 28):
        raise ValueError('Require calendar evidence, 3–12 finalists and 1–28 continuation days')
    def select(*, planning_days, diagnostics, inputs):
        if not calendar_records:
            raise ValueError('Missing current calendar search evidence')
        try:
            result = select_program(planning_days, diagnostics, inputs, calendar_records[-1],
                limit=finalist_budget, continuation_days=continuation_days)
        except NoAcceptableCalendarError as error:
            records.append(getattr(error, 'continuation_evidence', {'failed': True, 'reason': str(error)}))
            raise
        records.append(result['evidence'])
        return result
    token = FINALIZED_PROGRAM_SELECTOR.set(select if enabled else None)
    try:
        yield records
    finally:
        FINALIZED_PROGRAM_SELECTOR.reset(token)
