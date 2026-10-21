"""Diagnose one saved rollover with fixed-calendar, matched-input controls.

No new date search, scheduler changes, database writes or execution under probes.
Reconstruct the two saved decisions, then cross-score doses and horizon lengths
inside the same active history-dose adapter used by normal planning.
"""

from argparse import ArgumentParser
from datetime import date, datetime
from inspect import signature
import json
from pathlib import Path
from time import perf_counter

from run_analysis import adherence_projection as projection
from run_analysis import weekly_schedule as planner
from run_analysis.prescription_distance import canonical_distance_policy
from run_analysis.selection_objective import shared_objective_policy
from scripts.audit_easy_dose_program import capture_allocations
from scripts.finalized_scoring_trial import finalized_scoring_policy
from scripts.opportunity_search_repair import opportunity_search_policy
from scripts.replay_easy_dose_program import replay_policy
from scripts.replay_synthetic_easy_dose import synthetic_history


def prescriptions(days):
    return [(r.planned_for.isoformat(), r.workout_type.value,
             r.distance_target_miles, list(r.distance_range_miles))
            for d in days if (r := d.recommendation) and r.workout_type != 'rest']


def investigate(report, *, candidate_observer=None):
    snapshots = report['replans'][:2]
    assert all(s['trigger'] == 'scheduled_refresh' for s in snapshots)
    start = datetime.fromisoformat(report['realized_execution']['start_at'])
    state, history = synthetic_history(start)
    config = report['synthetic_sandbox']['config']
    original_dates = planner.adaptive_run_day_offsets
    date_signature = signature(original_dates)
    captures, rows = [], []
    begin = perf_counter()
    near_dates = {date.fromisoformat(s['planned_for'][:10])
                  for s in snapshots[0]['planned_sessions'][:2]}

    def evaluate(inputs, offsets, roles, *, days=None):
        details = {}
        common = {k: inputs[k] for k in ('daily_states', 'daily_state_options',
            'request', 'config', 'target_distance_range', 'completed_miles_by_offset')}
        sessions = None
        if days is not None:
            sessions = [planner._CandidateSession(i, common['daily_states'][i].model_copy(
                update={'as_of': d.recommendation.planned_for}), d.recommendation)
                for i,d in enumerate(days) if d.recommendation]
        with capture_allocations() as captured:
            joint = planner._joint_candidate_program_cost(tuple(offsets), **common,
                expected_target_projector=inputs['expected_target_projector'],
                prior_distance_midpoints_by_date=inputs['prior_distance_midpoints_by_date'],
                role_overrides_by_offset=roles, reconcile_allocated_prefix=days is None,
                diagnostic_details=details, _allocated_override=days, _sessions_override=sessions)
        actual = days if days is not None else captured['days']
        ordinary = sum(planner.typical_easy_distance(inputs['daily_states'][0])) / 2
        result = planner._score_breakdown(tuple(offsets), joint,
            details['finalized_coaching_cost'], ordinary, details).model_dump(mode='json')
        observed = (candidate_observer(inputs, actual, result)
                    if candidate_observer is not None else None)
        record = dict(score=result, prescriptions=prescriptions(actual))
        if observed is not None:
            record['continuation_probe'] = observed
        return record, actual

    def probe(inputs, offsets, roles, label):
        chosen, days = evaluate(inputs, offsets, roles)
        exact, _ = evaluate(inputs, offsets, roles, days=days)
        assert abs(exact['score']['total_cost'] - chosen['score']['total_cost']) < 1e-8
        alternatives = {}
        for target, bounds in ((3.8, (3.5,4.0)), (4.0, (3.7,4.2))):
            changed = [d.model_copy(update={'recommendation': d.recommendation.model_copy(update={
                'distance_target_miles': target, 'distance_range_miles': bounds,
            })}) if d.date in near_dates and d.recommendation else d for d in days]
            alternatives[str(target)], _ = evaluate(inputs, offsets, roles, days=changed)
        return dict(name=label, origin=inputs['daily_states'][0].as_of.isoformat(),
            horizon_days=len(inputs['daily_states']), chosen=chosen, fixed_tail_dose_probes=alternatives)

    def fixed_dates(*args, **kwargs):
        bound = date_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        inputs = bound.arguments
        index = len(captures)
        snap = snapshots[index]
        origin = inputs['daily_states'][0].as_of.date()
        offsets = tuple(snap['planner_diagnostics']['winning_candidate']['offsets'])
        roles = {i:r for i,r in (inputs['prior_roles_by_offset'] or {}).items()
                 if i < planner.NEAR_TERM_STABILITY_DECISION_OFFSETS}
        if not snap['planner_diagnostics']['stability_tie_break_applied']:
            roles = {}
        inputs['_planner_diagnostics'].winning_role_overrides = roles
        control = probe(inputs, offsets, roles or None, f'decision_{index}_control')
        expected = snap['planner_diagnostics']['winning_candidate']
        assert abs(control['chosen']['score']['total_cost'] - expected['total_cost']) < 1e-8
        assert control['chosen']['score']['allocation_trace'] == expected['allocation_trace']
        rows.append(control)
        if index == 1:
            old_offsets = tuple((date.fromisoformat(s['planned_for'][:10])-origin).days
                                for s in snapshots[0]['planned_sessions'])
            assert min(old_offsets) >= 0 and max(old_offsets) < 20
            without_new_day = tuple(i for i in offsets if i < 20)
            cases = [('old_calendar', old_offsets, 21),
                     ('old_calendar_old_end', old_offsets, 20),
                     ('new_calendar_without_new_day_run', without_new_day, 21),
                     ('new_calendar_old_end', without_new_day, 20),
                     ('old_calendar_plus_new_day_run', (*old_offsets,20), 21)]
            for label, calendar, length in cases:
                changed = dict(inputs, daily_states=inputs['daily_states'][:length],
                               daily_state_options=inputs['daily_state_options'][:length])
                rows.append(probe(changed, calendar, roles or None, label))
        return list(offsets)

    class Enough(Exception):
        pass

    with canonical_distance_policy(), shared_objective_policy(), finalized_scoring_policy(), \
            opportunity_search_policy(), replay_policy('history_trajectory'):
        original_build = projection.build_weekly_schedule

        def build(*args, **kwargs):
            result = original_build(*args, **kwargs)
            snap = snapshots[len(captures)]
            expected = [(s['planned_for'],s['workout_type'],s['midpoint_miles'],
                         [s['distance_low_miles'],s['distance_high_miles']]) for s in snap['planned_sessions']]
            assert prescriptions(result.planning_days) == expected
            captures.append(result)
            if len(captures) == 2:
                raise Enough
            return result

        try:
            planner.adaptive_run_day_offsets = fixed_dates
            projection.build_weekly_schedule = build
            projection.simulate_adherence(state, history, config, start,
                simulation_days=2, replan_interval_days=1, emit_post_upload_replans=True)
        except Enough:
            pass
        finally:
            planner.adaptive_run_day_offsets = original_dates
            projection.build_weekly_schedule = original_build
    assert len(captures) == 2
    return dict(scope=__doc__, matched_decisions=2, seconds=perf_counter()-begin, probes=rows)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output-json', type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        parser.error('Choose a new report path')
    result = investigate(json.loads(args.report.read_text()))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    for row in result['probes']:
        print(row['name'], 'chosen',row['chosen']['score']['total_cost'],
              'near',row['chosen']['prescriptions'][:2],
              '3.8 / 4.0', {k:v['score']['total_cost'] for k,v in row['fixed_tail_dose_probes'].items()})
    print('Seconds', result['seconds'])


if __name__ == '__main__':
    main()
