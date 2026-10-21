"""Replay saved decisions without search; isolate a compliant upload seam offline.

No database, settings, live server, weights or gate changes. Reconstruction must
match the saved physical prescription fingerprints before probes are trusted.
"""

from argparse import ArgumentParser
from contextlib import contextmanager, nullcontext
from datetime import datetime, timedelta
from functools import lru_cache
from inspect import signature
import json
from pathlib import Path
from time import perf_counter

from run_analysis import adherence_projection as projection, weekly_schedule as planner
from run_analysis.prescription_distance import canonical_distance_policy
from run_analysis.selection_objective import (
    COMPLETED_SELECTION_HISTORY, FINALIZED_PROGRAM_SELECTOR, shared_objective_policy,
)
from run_analysis.web.schemas import PlannerScoreBreakdown, RecommendationResponse
from scripts.audit_easy_dose_program import capture_allocations
from scripts.continuation_replan import fingerprint
from scripts.continuation_checks import check_sequence
from scripts.finalized_scoring_trial import finalized_scoring_policy
from scripts.issued_plan_execution import OpeningPlanExecution
from scripts.opportunity_search_repair import opportunity_search_policy
from scripts.replay_easy_dose_program import replay_policy
from scripts.replay_synthetic_easy_dose import synthetic_history
from scripts.typed_continuation_prototype import runs, compare_candidate, score_continuation


def hard_score(inputs, offsets, *, days=None, roles=None):
    details = {}
    common = {k: inputs[k] for k in ('daily_states', 'daily_state_options', 'request', 'config',
        'target_distance_range', 'completed_miles_by_offset', 'expected_target_projector',
        'prior_distance_midpoints_by_date')}
    sessions = ([planner._CandidateSession(i, inputs['daily_states'][i].model_copy(
        update={'as_of': d.recommendation.planned_for}), d.recommendation)
        for i, d in enumerate(days) if d.recommendation
        and d.recommendation.workout_type != planner.WorkoutType.REST] if days is not None else None)
    with capture_allocations() as capture:
        joint = planner._joint_candidate_program_cost(tuple(offsets), **common,
            role_overrides_by_offset=roles, reconcile_allocated_prefix=days is None,
            _allocated_override=days, _sessions_override=sessions,
            diagnostic_details=details, _collect_acceptance_facts=True)
    score = planner._score_breakdown(tuple(offsets), joint,
        float(details.get('finalized_coaching_cost', 0.)),
        sum(planner.typical_easy_distance(inputs['daily_states'][0])) / 2, details)
    return days if days is not None else capture.get('days'), score, details


def source_row(record, selected):
    return next(row for row in record['evaluated']
        if row['offsets'] == selected['offsets']
        and abs(row['total_cost'] - selected['legacy_cost']) < 1e-8)


@contextmanager
def saved_run_time_pace(enabled=False):
    """Explicit historical replay only; restore the current daily pace policy."""
    from scripts.expected_completion import ExpectedCompletion
    original = ExpectedCompletion._pace_reference_at
    try:
        if enabled:
            ExpectedCompletion._pace_reference_at = lambda self, item: item.planned_for
        yield
    finally:
        ExpectedCompletion._pace_reference_at = original


def reconstruct(report, observer, *, historical_run_time_pace=False):
    """Reuse saved decisions, checking complete hard prescription identity."""
    from scripts.expected_completion import expected_completion_build_policy
    from scripts.volume_path_prototype import volume_path_policy
    from scripts.boundary_funding_prototype import boundary_funding_policy
    funding=report['synthetic_sandbox'].get('funding_accounting',{})
    if funding.get('enabled') and funding.get('version')!='physical_path_funding_v1':
        raise ValueError('Unsupported saved funding-accounting policy')

    class SavedDecisionsComplete(Exception):
        pass

    start = datetime.fromisoformat(report['realized_execution']['start_at'])
    template, history = synthetic_history(start)
    config = report['synthetic_sandbox']['config']
    snapshots = report['replans']
    records = report['synthetic_sandbox']['calendar_neighbor_prototype']
    original = planner.adaptive_run_day_offsets
    call_signature = signature(original)
    matched = []

    def fixed_dates(*args, **kwargs):
        bound = call_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        inputs = bound.arguments
        record = records[len(matched)]
        row = record['winner']
        matches = [r for r in record['evaluated'] if r['offsets'] == row['offsets']
                   and abs(r['total_cost'] - row['total_cost']) < 1e-8
                   and r['allocation_trace'] == row['allocation_trace']]
        roles = matches[0].get('roles', {}) if matches else {}
        inputs['_planner_diagnostics'].winning_role_overrides = {int(k): v for k, v in roles.items()}
        return list(row['offsets'])

    def selected(*, planning_days, diagnostics, inputs):
        index = len(matched)
        snapshot = snapshots[index]
        saved = snapshot['planner_diagnostics']
        continuation = saved['continuation_selection']
        winner = next(row for row in continuation['finalists']
                      if row.get('hard_fingerprint') == continuation['selected_hard_fingerprint'])
        days = planning_days
        if winner['source'] == 'distance_neighbor':
            days = unpack_soft(winner['hard_program'])
        elif winner['source'] == 'prior_program':
            from scripts.prior_program_candidate import recover_prior_program
            seed, audit = recover_prior_program(inputs, continuation['continuation_days'])
            if seed is None:
                raise ValueError(f'Cannot reconstruct saved prior-program winner: {audit}')
            days = seed['days']
        elif winner['source'] == 'full_score_neighbor':
            days, _, _ = hard_score(inputs, winner['offsets'],
                roles={int(k): v for k, v in winner.get('roles', {}).items()})
        elif winner['source'] != 'publication_seed':
            regrowth=continuation.get('suffix_regrowth',{})
            if regrowth and regrowth.get('version')!='bounded_suffix_v1':
                raise ValueError('Unsupported saved regrowth policy')
            candidate_record=dict(records[index],evaluated=records[index]['evaluated']+regrowth.get('rows',[]))
            row = source_row(candidate_record, winner)
            days, _, _ = hard_score(inputs, winner['offsets'],
                roles={int(k): v for k, v in row.get('roles', {}).items()})
        actual = fingerprint(days)
        if actual != continuation['selected_hard_fingerprint']:
            raise ValueError(f'Decision {index}: physical reconstruction mismatch {actual}')
        evidence = COMPLETED_SELECTION_HISTORY.get()
        assert evidence is not None
        observer(index, inputs, days, evidence)
        matched.append(index)
        print(f'Reconstructed {index}: {snapshot["generated_at"]}', flush=True)
        if len(matched) == len(snapshots):
            # A fail-fast report may stop midway through a simulated day. Do
            # not execute or fabricate any decisions beyond its saved ledger.
            raise SavedDecisionsComplete
        def score(name):
            return PlannerScoreBreakdown.model_validate(saved[name]) if saved.get(name) else None
        return dict(planning_days=days, winner=score('winning_candidate'),
            published=score('published_candidate'), independent=score('unconstrained_winning_candidate'),
            stable=score('near_term_stable_candidate'),
            stable_selected=saved['stability_tie_break_applied'], tolerance=saved['stability_tolerance'],
            evidence=continuation)

    execution_plan = report['synthetic_sandbox'].get('execution_plan', 'opening')
    if execution_plan not in ('opening', 'latest'):
        raise ValueError('Unsupported saved execution policy')
    contract = OpeningPlanExecution() if execution_plan == 'opening' else None
    token = FINALIZED_PROGRAM_SELECTOR.set(selected)
    try:
        planner.adaptive_run_day_offsets = fixed_dates
        with canonical_distance_policy(), shared_objective_policy(), finalized_scoring_policy(), \
                opportunity_search_policy(), replay_policy('history_trajectory'), \
                saved_run_time_pace(historical_run_time_pace), \
                expected_completion_build_policy(report['synthetic_sandbox'].get(
                    'expected_completion', {}).get('enabled', False)), \
                volume_path_policy(report['synthetic_sandbox'].get('volume_path', {}).get('enabled',False)), \
                (boundary_funding_policy() if funding.get('enabled') else nullcontext()):
            try:
                projection.simulate_adherence(template, history, config, start,
                    simulation_days=report['synthetic_sandbox'].get('requested_days', 7),
                    replan_interval_days=1, emit_post_upload_replans=True,
                    execution_schedule_selector=contract.select if contract else None)
            except SavedDecisionsComplete:
                pass
    finally:
        planner.adaptive_run_day_offsets = original
        FINALIZED_PROGRAM_SELECTOR.reset(token)
        if contract is not None:
            contract.close()
    assert len(matched) == len(snapshots)
    return matched


def raw_state_lookup(inputs, evidence):
    known = {s.as_of: s for options in inputs['daily_state_options'] for s in options}
    known.update({s.as_of: s for s in inputs['daily_states']})
    config = inputs['config'].get('coaching', {})
    @lru_cache(maxsize=None)
    def state_at(at):
        if at in known:
            return known[at]
        return projection._state_at(evidence['template'], evidence['runs'], at,
            inputs['daily_states'][0].recent_load.capacity_reference_miles,
            easy_baseline_half_life_days=float(config.get('capacity_retention_half_life_days', 84)),
            fatigue_half_life_days=float(config.get('continuous_fatigue_half_life_days', 7)))
    return state_at


def unpack_soft(rows):
    return [planner.WeeklyScheduleDay(date=row['date'], day_role=row['role'],
        recommendation=RecommendationResponse.model_validate(dict(row['prescription'],
            generated_at=row['prescription']['fitness_state_as_of'])) if row['prescription'] else None,
        rationale='Saved exact diagnostic tail.') for row in rows]


def evaluate_physical(inputs, days, evidence, *, full=None, length=28, opportunity_mean_session=None):
    state_at = raw_state_lookup(inputs, evidence)
    offsets = [i for i, d in enumerate(days) if d.recommendation
               and d.recommendation.workout_type != planner.WorkoutType.REST]
    _, hard, details = hard_score(inputs, offsets, days=days)
    common = dict(state_at=state_at, continuation_days=length, distance_policy='finalized',
        complete_role_costs=True, support_policy='bounded', completed_runs=evidence['runs'],
        history_source=evidence['source'])
    if full is None:
        scored = compare_candidate(inputs, days, hard.model_dump(mode='json'), **common,
            timing_policy='coordinated', repair_opportunities=True, compare_opportunity_doses=True,
            opportunity_mean_session=opportunity_mean_session)
    else:
        origin = inputs['daily_states'][0].as_of
        states = [state_at(origin + timedelta(days=i)) for i in range(len(full))]
        targets = inputs['expected_target_projector'](runs(days), len(full))
        scored = score_continuation(inputs, days, hard.model_dump(mode='json'), targets,
            (full, states, []), **common)
        scored['fixed_sequence_checks'] = check_sequence(full, inputs, state_at, ())
    return dict(hard=hard.model_dump(mode='json'), continuation=scored)


@contextmanager
def repeat_role_sweep(passes):
    """Diagnostic only: reopen earlier soft decisions, same grid/objective."""
    from scripts import coordinated_continuation as module
    original = module.coordinate
    def repeated(make, evaluate):
        selected, score, metrics = original(make, evaluate)
        rounds = [metrics]
        for _ in range(1, passes):
            warm = {datetime.fromisoformat(x['at']).date(): x['requested_role'] for x in selected[2]}
            selected, score, metrics = original(lambda overrides: make({**warm, **overrides}), evaluate)
            rounds.append(metrics)
            if abs(metrics['initial_cost'] - metrics['final_cost']) <= 1e-9:
                break
        return selected, score, dict(initial_cost=rounds[0]['initial_cost'],
            final_cost=score['combined_cost'], rounds=rounds)
    try:
        module.coordinate = repeated
        yield
    finally:
        module.coordinate = original


def investigate(report, captured):
    before, after = captured
    before_origin = before[0]['daily_states'][0].as_of
    hard_end = before_origin.date() + timedelta(days=21)
    full_end = hard_end + timedelta(days=28)
    physical = []
    means = [item[0]['daily_states'][0].recent_load.trailing_28d.distance_miles /
             item[0]['daily_states'][0].recent_load.trailing_28d.activity_count for item in captured]
    for (inputs, days, _), snap in zip(captured, report['replans'][-2:]):
        tail = unpack_soft(snap['planner_diagnostics']['continuation_selection']['soft_sessions'])
        physical.append(days + tail)
    # Add only the actually completed opening prescription when scoring the
    # later calendar before upload; never add it again to completed history.
    physical[1] = [before[1][0]] + physical[1]
    rows = []
    for index, (inputs, days, evidence) in enumerate(captured):
        label = ('before', 'after')[index]
        own = evaluate_physical(inputs, days, evidence, full=physical[index][index:])
        expected = report['replans'][-2 + index]['planner_diagnostics']['winning_candidate']['total_cost']
        assert abs(own['continuation']['combined_cost'] - expected) < 1e-8
        rows.append(dict(origin=label, program=label, mode='own_saved_control', **own))
        for source, sequence in zip(('before', 'after'), physical):
            origin = inputs['daily_states'][0].as_of.date()
            hard = [d for d in sequence if origin <= d.date < hard_end]
            full = [d for d in sequence if origin <= d.date < full_end]
            adjusted = dict(inputs, daily_states=inputs['daily_states'][:len(hard)],
                daily_state_options=inputs['daily_state_options'][:len(hard)])
            assert len(full) == len(hard) + 28
            fixed = evaluate_physical(adjusted, hard, evidence, full=full)
            rows.append(dict(origin=label, program=source, mode='fixed_absolute_hard_and_tail', **fixed))
            rebuilt = evaluate_physical(adjusted, hard, evidence)
            rows.append(dict(origin=label, program=source, mode='fixed_hard_regenerated_tail', **rebuilt))
            with repeat_role_sweep(3):
                repeated = evaluate_physical(adjusted, hard, evidence)
            rows.append(dict(origin=label, program=source, mode='three_role_sweeps', **repeated))
            swapped = evaluate_physical(adjusted, hard, evidence, opportunity_mean_session=means[1-index])
            rows.append(dict(origin=label, program=source, mode='other_origin_opportunity_mean_only', **swapped))
            # An available-before-upload alternative, not future actual data:
            # expected hard-prefix work plus actual history at the soft boundary.
            lookup = raw_state_lookup(adjusted, evidence)
            boundary = adjusted['daily_states'][0].as_of + timedelta(days=len(hard))
            expected = planner._project_state(lookup(boundary), runs(hard), inputs['config'])
            window = expected.recent_load.trailing_28d
            expected_mean = window.distance_miles / window.activity_count
            handoff = evaluate_physical(adjusted, hard, evidence, opportunity_mean_session=expected_mean)
            rows.append(dict(origin=label, program=source, mode='expected_handoff_mean_only',
                opportunity_mean_session=expected_mean, **handoff))
    # Was the after calendar considered before upload? Distinguish calendar
    # coverage from actual doses/roles and from six-finalist selection.
    later_dates = {d.date for d in after[1] if d.recommendation}
    later_dates.add(before_origin.date())
    earlier_offsets = sorted((d-before_origin.date()).days for d in later_dates if d < hard_end)
    record = report['synthetic_sandbox']['calendar_neighbor_prototype'][-2]
    seen = [r for r in record['evaluated'] if r['offsets'] == earlier_offsets]
    finalists = report['replans'][-2]['planner_diagnostics']['continuation_selection']['finalists']
    near = [i for i in earlier_offsets if i < planner.NEAR_TERM_STABILITY_DECISION_OFFSETS]
    return dict(probes=rows, availability=dict(before_offsets=earlier_offsets,
        evaluated_matches=seen, shortlisted_matches=[r for r in finalists if r['offsets'] == earlier_offsets],
        near_offsets=near,
        evaluated_near_match_count=sum([i for i in r['offsets'] if i < planner.NEAR_TERM_STABILITY_DECISION_OFFSETS] == near
                                      for r in record['evaluated']),
        shortlisted_near_matches=[r for r in finalists if [i for i in r['offsets']
                                 if i < planner.NEAR_TERM_STABILITY_DECISION_OFFSETS] == near]),
        opening_means=means)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output-json', type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        parser.error('Require a new output path')
    report = json.loads(args.report.read_text())
    captured = []
    started = perf_counter()
    def observe(index, inputs, days, evidence):
        if index >= len(report['replans']) - 2:
            captured.append((inputs, days, evidence))
    matched = reconstruct(report, observe)
    with canonical_distance_policy(), shared_objective_policy(), finalized_scoring_policy(), opportunity_search_policy():
        probes = investigate(report, captured)
    result = dict(scope=__doc__, matched_decisions=matched,
        seconds=perf_counter() - started,
        prescriptions=[[d.model_dump(mode='json') for d in days] for _, days, _ in captured], **probes)
    with args.output_json.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print('Seconds', result['seconds'])


if __name__ == '__main__':
    main()
