"""Reconstruct saved daily decisions with fixed calendars, not a new search.

Every published prescription and execution must match the original replay.
Counterfactual scores do not replace execution plans. No production mutations.
"""

from argparse import ArgumentParser
from datetime import datetime
from inspect import signature
import json
from pathlib import Path
from time import perf_counter

from scripts.audit_easy_dose_program import capture_allocations
from scripts.finalized_scoring_trial import finalized_scoring_policy
from scripts.opportunity_search_repair import opportunity_search_policy
from scripts.replay_easy_dose_program import replay_policy
from scripts.replay_synthetic_easy_dose import synthetic_history
from run_analysis import adherence_projection as projection
from run_analysis import weekly_schedule as planner
from run_analysis.prescription_distance import CANONICAL_DISTANCE_TARGETS, canonical_distance_policy
from run_analysis.selection_objective import shared_objective_policy


def prescriptions(days):
    return [(d.recommendation.planned_for.isoformat(), d.recommendation.workout_type.value,
             tuple(d.recommendation.distance_range_miles)) for d in days
            if d.recommendation and d.recommendation.workout_type != planner.WorkoutType.REST]


def expected_prescriptions(snapshot):
    return [(s['planned_for'], s['workout_type'],
             (s['distance_low_miles'], s['distance_high_miles']))
            for s in snapshot['planned_sessions']]


def score_calendar(inputs, offsets, roles):
    details = {}
    common = {key: inputs[key] for key in ('daily_states', 'daily_state_options',
        'request', 'config', 'target_distance_range', 'completed_miles_by_offset')}
    with capture_allocations() as captured:
        joint = planner._joint_candidate_program_cost(tuple(offsets), **common,
            expected_target_projector=inputs['expected_target_projector'],
            prior_distance_midpoints_by_date=inputs['prior_distance_midpoints_by_date'],
            role_overrides_by_offset=roles, reconcile_allocated_prefix=True,
            diagnostic_details=details)
    cost = planner._program_selection_cost(joint, details['finalized_coaching_cost'],
        sum(planner.typical_easy_distance(inputs['daily_states'][0])) / 2,
        distance_switching_cost=details.get('distance_switching_cost', 0.))
    return dict(cost=cost, joint=joint, coaching=details['finalized_coaching_cost'],
                details=details, prescriptions=prescriptions(captured['days']))


def investigate(report, *, probe_canonical_targets=False, probe_shared_objective=False):
    if (report['synthetic_sandbox']['requested_days'] != 21
            or report['mode'] not in {'perfect', 'below_range_easy'}
            or not report['realized_execution']['complete']):
        raise ValueError('This reconstruction requires a completed 21-day perfect/below-range fixture')
    start = datetime.fromisoformat(report['realized_execution']['start_at'])
    config = report['synthetic_sandbox']['config']
    state, history = synthetic_history(start)
    index = 0
    rows, executed = [], []
    begin = perf_counter()
    original_dates = planner.adaptive_run_day_offsets
    original_allocate = planner._allocate_visible_distance_ranges
    build_signature = signature(planner.build_weekly_schedule)
    date_signature = signature(original_dates)
    current = {}

    def fixed_dates(*args, **kwargs):
        bound = date_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        inputs = bound.arguments
        snapshot = report['replans'][index]
        roles = {i: role for i, role in (inputs['prior_roles_by_offset'] or {}).items()
                 if i < planner.NEAR_TERM_STABILITY_DECISION_OFFSETS}
        if not snapshot['planner_diagnostics']['stability_tie_break_applied']:
            roles = {}
        inputs['_planner_diagnostics'].winning_role_overrides = roles
        offsets = snapshot['planner_diagnostics']['winning_candidate']['offsets']
        current.update(search_inputs=inputs, roles=roles, offsets=offsets)
        if CANONICAL_DISTANCE_TARGETS.get() and probe_shared_objective:
            # Score while the history-dose adapter is active, not after the
            # build restores it. Otherwise this compares different models.
            current['canonical_search_probe'] = score_calendar(inputs, offsets, roles or None)
        if '2026-10-05' <= snapshot['generated_at'] < '2026-10-09' and not CANONICAL_DISTANCE_TARGETS.get():
            # Inside the history adapter: reproduce the search's final score,
            # then vary only the next long's date, keeping the remaining tail.
            control = score_calendar(inputs, offsets, roles or None)
            saved = snapshot['planner_diagnostics']['winning_candidate']
            assert abs(control['cost'] - saved['total_cost']) < 1e-8, (
                index, control['cost'], saved['total_cost'])
            origin = inputs['daily_states'][0].as_of.date()
            candidates = {'search_control': control}
            for day in ('2026-10-12', '2026-10-13'):
                destination = (datetime.fromisoformat(day).date() - origin).days
                old = next((i for i in offsets if
                    (origin.toordinal()+i) in {
                        datetime(2026,10,12).date().toordinal(),
                        datetime(2026,10,13).date().toordinal()}), None)
                if old is not None and (destination == old or destination not in offsets):
                    alternate = sorted(set(offsets)-{old}|{destination})
                    candidates['long_'+day] = score_calendar(inputs, alternate, roles or None)
            if snapshot['generated_at'].startswith('2026-10-07'):
                old_calendar = report['replans'][index-1]['planned_sessions']
                old_offsets = [(datetime.fromisoformat(s['planned_for']).date()-origin).days
                               for s in old_calendar]
                old_offsets = [i for i in old_offsets if 0 <= i < len(inputs['daily_states'])]
                candidates['previous_calendar_current_inputs'] = score_calendar(inputs, old_offsets, roles or None)
                # Single-date control otherwise creates a nine-day gap to
                # the subsequent long: test their coupling explicitly.
                shifts = {datetime(2026,10,13).date(): datetime(2026,10,12).date(),
                          datetime(2026,10,21).date(): datetime(2026,10,20).date()}
                paired = [(shifts.get(datetime.fromordinal(origin.toordinal()+i).date(),
                                     datetime.fromordinal(origin.toordinal()+i).date())-origin).days
                          for i in offsets]
                if len(set(paired)) == len(paired):
                    candidates['both_longs_earlier'] = score_calendar(inputs, sorted(paired), roles or None)
            current['calendar_probes'] = candidates
        return list(offsets)

    def allocate(*args, **kwargs):
        result = original_allocate(*args, **kwargs)
        if kwargs.get('allocation_selector'):
            current['selector'] = kwargs['allocation_selector']
        return result

    def probe_selector(selector, days, inputs):
        captured = {}
        original_joint = planner._joint_candidate_program_cost

        def joint(*args, **kwargs):
            details = kwargs.get('diagnostic_details')
            result = original_joint(*args, **kwargs)
            if kwargs.get('_allocated_override') is not None:
                captured.update(joint=result, details=details)
            return result

        try:
            planner._joint_candidate_program_cost = joint
            final = selector(days)
        finally:
            planner._joint_candidate_program_cost = original_joint
        details = captured['details']
        base = planner._program_selection_cost(captured['joint'],
            details['finalized_coaching_cost'],
            sum(planner.typical_easy_distance(inputs['daily_states'][0])) / 2)
        return dict(final=final, base=base, publication_switching_cost=final-base,
                    shared_distance_switching_cost=details.get('distance_switching_cost', 0.),
                    joint=captured['joint'], coaching=details['finalized_coaching_cost'],
                    recovery_components=details.get('recovery_cost_components'),
                    planned_miles=details['planned_midpoint_miles'])

    with opportunity_search_policy(), finalized_scoring_policy(), replay_policy('history_trajectory'):
        original_build = projection.build_weekly_schedule

        def build(*args, **kwargs):
            nonlocal index
            current.clear()
            bound = build_signature.bind(*args, **kwargs)
            bound.apply_defaults()
            result = original_build(*args, **kwargs)
            snapshot = report['replans'][index]
            actual = prescriptions(result.planning_days)
            expected = expected_prescriptions(snapshot)
            if actual != expected:
                raise ValueError(f"Fixed-calendar reconstruction differs at {index}: {actual} != {expected}")
            inputs = current['search_inputs']
            selector = current['selector']
            if '2026-10-05' <= snapshot['generated_at'] < '2026-10-09':
                row = dict(index=index, at=snapshot['generated_at'],
                    origin=snapshot['decision_start_date'],
                    opening_state=inputs['daily_states'][0].model_dump(mode='json'),
                    targets=inputs['expected_target_projector']([], 49),
                    published=actual, published_score=probe_selector(selector, result.planning_days, inputs),
                    saved_search=snapshot['planner_diagnostics']['winning_candidate'],
                    dose_probes=[])
                for low, high in ((3.0, 3.5), (3.2, 3.8), (3.5, 4.0), (3.7, 4.2)):
                    modified = [d.model_copy(update={'recommendation': d.recommendation.model_copy(
                        update={'distance_range_miles': (low, high)})})
                        if str(d.date) == '2026-10-09' else d for d in result.planning_days]
                    row['dose_probes'].append(dict(range=[low, high],
                        **probe_selector(selector, modified, inputs)))
                row['calendar_probes'] = current['calendar_probes']
                variants = [('canonical_target_probe', False)] if (probe_canonical_targets or probe_shared_objective) else []
                if probe_shared_objective:
                    variants.append(('shared_objective_probe', True))
                for variant, shared in variants:
                    saved_current = dict(current)
                    try:
                        stamp = perf_counter()
                        with canonical_distance_policy(), shared_objective_policy(shared,
                                prior_schedule=bound.arguments.get('prior_schedule')):
                            trial = original_build(*args, **kwargs)
                            for day in trial.planning_days:
                                if day.recommendation:
                                    planner.RecommendationResponse.model_validate(day.recommendation.model_dump())
                            row[variant] = dict(
                                seconds=perf_counter()-stamp,
                                prescriptions=prescriptions(trial.planning_days),
                                targets=[(str(d.date), d.recommendation.distance_target_miles)
                                    for d in trial.planning_days if d.recommendation],
                                score=probe_selector(current['selector'],trial.planning_days,current['search_inputs']))
                            if probe_shared_objective:
                                row[variant]['search'] = current['canonical_search_probe']
                                row[variant]['published_diagnostics'] = (
                                    trial.planner_diagnostics.published_candidate.model_dump(mode='json')
                                    if trial.planner_diagnostics.published_candidate else None)
                                if shared:
                                    scored = row[variant]['score']
                                    assert abs(scored['final'] - scored['base']
                                        - scored['shared_distance_switching_cost']) < 1e-8
                                    search = row[variant]['search']
                                    assert abs(search['cost'] - scored['final']) < 1e-8
                                    assert search['prescriptions'] == row[variant]['prescriptions']
                                    assert search['details']['allocation_trace'] == (
                                        row[variant]['published_diagnostics']['allocation_trace'])
                    finally:
                        current.clear()
                        current.update(saved_current)
                rows.append(row)
                print(f"Matched {index} {row['at']}: "
                      f"{[(p['range'], round(p['base'],4), round(p['publication_switching_cost'],4)) for p in row['dose_probes']]}", flush=True)
            index += 1
            return result

        def observe(snapshot):
            saved = report['replans'][index-1]
            for key in ('capacity_reference_miles', 'opening_continuous_distance_miles',
                        'opening_short_term_distance_miles', 'opening_recovery_residual_load',
                        'target_low_miles', 'target_high_miles'):
                actual = getattr(snapshot, key)
                expected = saved[key]
                if actual is None or expected is None:
                    assert actual == expected, (index, key, actual, expected)
                else:
                    assert abs(actual - expected) < 1e-8, (index, key, actual, expected)
            assert snapshot.generated_at.isoformat() == saved['generated_at']
            assert [list(x) for x in snapshot.target_trajectory] == saved['target_trajectory']

        try:
            planner.adaptive_run_day_offsets = fixed_dates
            planner._allocate_visible_distance_ranges = allocate
            projection.build_weekly_schedule = build
            projection.simulate_adherence(state, history, config, start,
                simulation_days=report['synthetic_sandbox']['requested_days'], weeks=3,
                replan_interval_days=1, initial_schedule=None, emit_post_upload_replans=True,
                replan_observer=observe,
                execution_observer=lambda run, _: executed.append((run.start_time.isoformat(),run.distance_miles)),
                deterministic_profile=None if report['mode']=='perfect' else
                    projection.DeterministicAdherenceProfile(
                        scenario=projection.DeterministicAdherenceScenario(report['mode'])))
        finally:
            planner.adaptive_run_day_offsets = original_dates
            planner._allocate_visible_distance_ranges = original_allocate
            projection.build_weekly_schedule = original_build
    assert index == len(report['replans'])
    assert executed == [(r['occurred_at'],r['distance_miles']) for r in report['realized_execution']['sessions']]
    return dict(scope=__doc__, matched_decisions=index, matched_executions=len(executed),
                seconds=perf_counter()-begin, cases=rows)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output-json', type=Path, required=True)
    parser.add_argument('--probe-canonical-targets', action='store_true')
    parser.add_argument('--probe-shared-objective', action='store_true')
    args = parser.parse_args()
    if args.output_json.exists():
        parser.error('Choose a new report path')
    result = investigate(json.loads(args.report.read_text()), probe_canonical_targets=args.probe_canonical_targets,
                         probe_shared_objective=args.probe_shared_objective)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open('x') as out:
        json.dump(result, out, indent=2)
        out.write('\n')
    print(f"Matched {result['matched_decisions']} decisions in {result['seconds']:.2f}s")


if __name__ == '__main__':
    main()
