"""Bounded saved-case search/funding diagnosis, without changing planner policy.

Reconstruct exact decisions, audit the dropped near-prefix representative, and
price a fixed physical program on both sides of the hard/soft accounting seam.
No live writes, weight changes, gate relaxation, or new daily simulation.
"""
from argparse import ArgumentParser
from datetime import timedelta
import json
from pathlib import Path
from time import perf_counter

from run_analysis import weekly_schedule as planner
from scripts.audit_conditional_upload import components, score, scoring
from scripts.audit_continuation_upload import (
    evaluate_physical, hard_score, raw_state_lookup, reconstruct, unpack_soft,
)
from scripts.calendar_neighbor_prototype import calendar_neighbors
from scripts.continuation_checks import check_sequence
from scripts.continuation_replan import shortlist
from scripts.volume_path_prototype import volume_path_policy


def retime(days, offset, at, options):
    """Change only timing; use an actual supplied option, never invent one."""
    if at not in {state.as_of for state in options[offset]}:
        raise ValueError('Timestamp is not an available option')
    day = days[offset]
    if day.recommendation is None or at.date() != day.date:
        raise ValueError('Require a run on the same physical date')
    updated = list(days)
    updated[offset] = day.model_copy(update={
        'planned_at': at,
        'recommendation': day.recommendation.model_copy(update={'planned_for': at}),
    })
    return updated


def cadence_trace(days, inputs):
    """Expand the existing calendar cadence term, without changing its price."""
    opening = inputs['daily_states'][0]
    result = []
    for role, kinds, age, key in (
        ('long', {planner.WorkoutType.LONG}, opening.days_since_long_run,
         'long_run_recency_reference_days'),
        ('quality', planner.QUALITY_WORKOUT_TYPES, opening.days_since_quality_run,
         'quality_recency_reference_days'),
    ):
        if age is None:
            continue
        reference = float(inputs['config'].get('coaching', {}).get(key, 7))
        previous = opening.as_of.date()
        for day in days:
            elapsed = max(0, (day.date - previous).days)
            if not planner._is_taper_or_race_date(inputs['config'], day.date):
                age += elapsed
                cost = planner._key_session_cadence_overdue_cost(age, reference)
                if cost:
                    result.append(dict(role=role, date=day.date.isoformat(), age=age, cost=cost))
            if day.recommendation and day.recommendation.workout_type in kinds:
                age = 0.
            previous = day.date
    return result


def funding_trace(inputs, hard):
    targets = inputs['expected_target_projector'](
        [d.recommendation for d in hard if d.recommendation], len(hard))
    target = tuple(sum(t[i] for t in targets) / 7 for i in (0, 1))
    ordinary = sum(planner.typical_easy_distance(inputs['daily_states'][0])) / 2
    boundary = planner._recent_boundary_session(inputs['daily_states'][0])
    allowance = max(ordinary, (boundary or (0., 0.))[1])
    raw = planner._hard_horizon_funding_violation(hard, target,
        boundary_allowance_miles=allowance)
    return dict(target=target, ordinary=ordinary, boundary=boundary,
        allowance=allowance, raw_violation=raw,
        gated_violation=planner._gate_aligned_hard_funding_violation(raw, ordinary))


def timing_stress(inputs, days, evidence):
    """Same near-term doses/dates, all supplied times for easy then quality.

    This checks model guards, not physiology or a recommendation to adhere to
    exact clock times. It neither changes the calendar nor waives a guard.
    """
    near = days[:planner.NEAR_TERM_STABILITY_DECISION_OFFSETS]
    offsets = [i for i, d in enumerate(near) if d.recommendation]
    pairs = [(a, b) for a, b in zip(offsets, offsets[1:])
             if near[a].recommendation.workout_type == planner.WorkoutType.EASY
             and near[b].recommendation.workout_type in planner.QUALITY_WORKOUT_TYPES]
    result = []
    lookup = raw_state_lookup(inputs, evidence)
    for easy, quality in pairs:
        for first in inputs['daily_state_options'][easy]:
            for second in inputs['daily_state_options'][quality]:
                shifted = retime(near, easy, first.as_of, inputs['daily_state_options'])
                shifted = retime(shifted, quality, second.as_of, inputs['daily_state_options'])
                checks = check_sequence(shifted, inputs, lookup, ())
                result.append(dict(easy_at=first.as_of.isoformat(), quality_at=second.as_of.isoformat(),
                    elapsed_hours=(second.as_of-first.as_of).total_seconds()/3600, **checks))
    return result


def investigate(report):
    if report['synthetic_sandbox'].get('volume_path', {}).get('version') != 'additive_date_path_v1':
        raise ValueError('Require additive date-path replay')
    if report['synthetic_sandbox']['expected_completion'].get('state_transition_version') != 'daily_pace_v1':
        raise ValueError('Require current expected-completion transition')
    if len(report['replans']) != 2:
        raise ValueError('This bounded panel requires the first-upload two-decision failure')
    record = report['synthetic_sandbox']['calendar_neighbor_prototype'][0]
    later = report['replans'][1]
    before_date = report['replans'][0]['decision_start_date']
    from datetime import date
    delta = (date.fromisoformat(later['decision_start_date']) - date.fromisoformat(before_date)).days
    protected = planner.NEAR_TERM_STABILITY_DECISION_OFFSETS
    near = [0] + [o + delta for o in later['planner_diagnostics']['winning_candidate']['offsets']
                  if o + delta < protected]
    matches = [r for r in record['evaluated'] if [o for o in r['offsets'] if o < protected] == near]
    if len(matches) != 1:
        raise ValueError('Expected exactly one dropped near-prefix representative')
    dropped = matches[0]
    captured, proposed = [], []
    def observe(index, inputs, days, evidence):
        captured.append((inputs, days, evidence))
        if index:
            return
        # Materialize while the replay allocator/history scopes are active.
        candidate, hard, _ = hard_score(inputs, dropped['offsets'],
            roles={int(k): v for k, v in dropped.get('roles', {}).items()})
        assert abs(hard.total_cost - dropped['total_cost']) < 1e-8
        assert hard.model_dump(mode='json')['allocation_trace'] == dropped['allocation_trace']
        proposed.append((candidate, hard))
        # One bounded second neighborhood, holding the entire near prefix.
        # This is a diagnostic, not a new search pass in the scheduler.
        allowed = (set(range(len(days))) - set(inputs['forced_rest_offsets'] or ())
                   - set(inputs['completed_run_offsets'] or ()))
        for offsets in calendar_neighbors(tuple(dropped['offsets']), allowed):
            if [o for o in offsets if o < protected] != near:
                continue
            candidate, hard, details = hard_score(inputs, offsets)
            if candidate is None or details.get('unrepresentable_distance_target'):
                continue
            proposed.append((candidate, hard))
    started = perf_counter()
    reconstruct(report, observe)
    search_seconds = perf_counter() - started
    first, after = captured
    inputs, _, evidence = first
    candidate, hard = proposed[0]
    result = dict(near_offsets=near, dropped=hard.model_dump(mode='json'),
        dropped_cadence=cadence_trace(candidate, inputs), reconstruction_and_neighbor_seconds=search_seconds,
        neighbor_count=len(proposed)-1, timing=[], timing_stress=[], continuation=[], funding=[], native_controls=[])
    with volume_path_policy():
        for i, (raw, days, history) in enumerate(captured):
            full = days + unpack_soft(report['replans'][i]['planner_diagnostics']['continuation_selection']['soft_sessions'])
            end = days[0].date + timedelta(days=21)
            native = score(raw, history, full, end, end+timedelta(days=28), historical_pace=False)
            assert abs(native['components']['total'] - report['replans'][i]['planner_diagnostics']['winning_candidate']['total_cost']) < 1e-8
            result['native_controls'].append(native['components'])
        with scoring(inputs, evidence, False):
            quality = next(i for i, d in enumerate(candidate)
                if d.recommendation and d.recommendation.workout_type in planner.QUALITY_WORKOUT_TYPES)
            for state in inputs['daily_state_options'][quality]:
                changed = retime(candidate, quality, state.as_of, inputs['daily_state_options'])
                _, priced, _ = hard_score(inputs, hard.offsets, days=changed)
                result['timing'].append(dict(at=state.as_of.isoformat(), score=priced.model_dump(mode='json')))
            # Only two complete regenerated continuations, not a rollout at
            # every neighbor. Sequence guards are reported, not assumed.
            winner = min(proposed, key=lambda item: item[1].total_cost)
            for name, (program, price) in (('dropped', proposed[0]), ('best_hard_neighbor', winner)):
                probe = evaluate_physical(inputs, program, evidence)
                result['continuation'].append(dict(name=name, components=components(probe), **probe))
                print(name, components(probe), flush=True)
        result['neighbors'] = [h.model_dump(mode='json') for _, h in proposed]
        result['shortlisted_offsets'] = [r['offsets'] for r in shortlist(record, limit=6)]
        raw, days, history = after
        with scoring(raw, history, False):
            result['timing_stress'] = timing_stress(raw, days, history)
        full = days + unpack_soft(report['replans'][1]['planner_diagnostics']['continuation_selection']['soft_sessions'])
        close = days[0].date + timedelta(days=49)
        for count in (20, 21):
            probe = score(raw, history, full, days[0].date+timedelta(days=count), close, historical_pace=False)
            with scoring(raw, history, False):
                trace = funding_trace(raw, days[:count])
            result['funding'].append(dict(hard_days=count, trace=trace, **probe))
            print('Hard boundary', count, trace, probe['components'], flush=True)
        a, b = result['funding']
        for key in ('recovery', 'coaching', 'support', 'shape', 'switching', 'policy_rate', 'soft_target'):
            assert a['components'][key] == b['components'][key], key
        for key in ('cost', 'squared_violation'):
            assert a['continuation']['volume_path']['full'][key] == b['continuation']['volume_path']['full'][key]
    return result


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output-json', type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        parser.error('Choose a new output path')
    started = perf_counter()
    result = investigate(json.loads(args.report.read_text()))
    result['seconds'] = perf_counter() - started
    with args.output_json.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print('Seconds', result['seconds'])


if __name__ == '__main__':
    main()
