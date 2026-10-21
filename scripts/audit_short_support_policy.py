"""Offline fee/eligibility factorial and equal-mileage fragmentation challenge.

Start with the selected native opportunity repair. Never change the hard plan,
weights, native dose menu, live settings or ordinary-history reference. Short
doses are explicit counterfactual probes, not newly authorized prescriptions.
All variants retain other sessions and recheck the full downstream sequence.
"""

from datetime import datetime
from functools import lru_cache
from math import isfinite
from time import perf_counter

from run_analysis import adherence_projection as projection
from run_analysis import weekly_schedule as planner
from scripts.aerobic_distribution import per_run_shortfall_components
from scripts.audit_continuation_delivery import revised_easy
from scripts.continuation_checks import check_sequence
from scripts.continuation_opportunity_repair import polish
from scripts.coordinated_continuation import coordinate
from scripts.typed_continuation_prototype import rollout, runs, score_continuation


POLICIES = ('legacy', 'distribution_plus_fragmentation', 'distribution_only')


def fee_ablations(cost, parts):
    """Cross-score the identical physical program; never refinalize a dose."""
    if not isfinite(cost):
        raise ValueError('Require finite program cost')
    return dict(zip(POLICIES, (cost, cost - parts['support_fee'], cost - parts['total_fee']), strict=True))


def placements(available, count):
    """Clustered and spread probes; bounded strides, never just packed dates."""
    if count < 1:
        raise ValueError('Require a positive piece count')
    if count > len(available):
        return
    if count == 1:
        yield from ((i,) for i in available)
        return
    maximum = (len(available) - 1) // (count - 1)
    for stride in sorted({1, 2, 4, maximum}):
        if stride > maximum:
            continue
        for start in range(len(available) - (count - 1) * stride):
            yield tuple(available[start + j * stride] for j in range(count))
    if (len(available) - 1) % (count - 1):
        # Include the full span when an integer stride cannot reach both ends.
        # Unequal index gaps make this distinct from the stride probes above.
        yield tuple(available[j * (len(available) - 1) // (count - 1)] for j in range(count))


def counterfactuals(days, states, hard_length, *, blocked=()):
    """Bounded controls, not exhaustive date search or a new minimum dose.

    Add 1/2/4 total miles as one run or equal 0.5/1/2-mile pieces, plus single
    0.5/1.5/3/3.3/3.5-mile controls. The latter two probe inside this fixture's
    existing 3.25-mile intent floor before considering a lower floor.
    Enumerate clustered and spread placements using
    bounded strides without combinatorial expansion. Split ordinary soft runs
    without adding mileage, explicitly challenging replacement fragmentation.
    """
    if len(days) != len(states) or not 0 < hard_length < len(days):
        raise ValueError('Require aligned nonempty hard and soft programs')
    blocked = set(blocked)
    available = [i for i in range(hard_length, len(days)) if i not in blocked
                 and days[i].recommendation is None and not days[i].completed_activities]
    easy = [i for i in range(hard_length, len(days)) if i not in blocked
            and not days[i].completed_activities and days[i].recommendation
            and days[i].recommendation.workout_type == planner.WorkoutType.EASY
            and days[i].day_role != 'medium_long_run']
    if not easy:
        raise ValueError('Need an ordinary soft easy session for controlled probes')
    ordinary = planner.typical_easy_distance(states[0])[0]
    expansion = planner._ordinary_easy_expansion_reference(states[0])
    template = days[easy[0]].recommendation
    yield {'family': 'rest_control', 'pieces': 0, 'added_miles': 0., 'edits': ()}, days
    seen = set()

    def build(indices, total, *, replace=None):
        # Only exactly representable equal splits; no rounding-created mileage.
        dose = total / len(indices)
        if dose <= .25 or abs(dose * 10 - round(dose * 10)) > 1e-8:
            return None
        identity = (indices, dose, replace)
        if identity in seen:
            return None
        seen.add(identity)
        changed = list(days)
        for i in indices:
            at = states[i].as_of.replace(hour=12, minute=0, second=0, microsecond=0)
            if i == replace:
                at = days[i].recommendation.planned_for
            changed[i] = revised_easy(days[i], template, at, dose, ordinary, expansion)
        return {'family': 'split_existing' if replace is not None else 'add_volume',
                'pieces': len(indices), 'dose_miles': dose, 'total_probe_miles': total,
                'added_miles': 0. if replace is not None else total,
                'replaced_offset': replace, 'edits': indices}, changed

    for total, counts in ((.5, (1,)), (1., (1, 2)), (1.5, (1,)),
                          (2., (1, 2, 4)), (3., (1,)), (3.3, (1,)), (3.5, (1,)),
                          (4., (1, 2, 4, 8))):
        for count in counts:
            for indices in placements(available, count):
                row = build(indices, total)
                if row:
                    yield row
    for source in easy:
        total = planner._midpoint(days[source].recommendation)
        for count in (2, 4, 8):
            for others in placements(available, count - 1):
                indices = tuple(sorted([source, *others]))
                row = build(indices, total, replace=source)
                if row:
                    yield row


def summarize(rows):
    """Separate modeled guards, legacy intent coverage and fee choice."""
    result = {}
    for eligibility in ('legacy_intent', 'support_probe'):
        pool = [r for r in rows if r['guards_pass']
                and (eligibility == 'support_probe' or r['legacy_intent_pass'])]
        result[eligibility] = {}
        for policy in POLICIES:
            groups = {}
            for row in pool:
                key = f"{row['family']}:added={row['added_miles']}:pieces={row['pieces']}"
                if key not in groups or row['costs'][policy] < groups[key]['costs'][policy] - 1e-9:
                    groups[key] = row
            result[eligibility][policy] = {
                'best': min(pool, key=lambda row: row['costs'][policy]) if pool else None,
                'best_by_family': groups,
            }
    return result


def compare(inputs, hard, hard_score, history, state_at, *, continuation_days=28):
    begin = perf_counter()
    targets = inputs['expected_target_projector'](runs(hard), len(hard) + continuation_days)
    context = {**inputs, '_audit_targets': targets}
    def make(overrides):
        return rollout(hard, inputs['daily_states'], inputs['request'], inputs['config'], targets,
            state_at=state_at, continuation_days=continuation_days,
            distance_policy='finalized', role_overrides=overrides)
    def evaluate(program):
        return score_continuation(inputs, hard, hard_score, targets, program, state_at=state_at,
            continuation_days=continuation_days, distance_policy='finalized',
            complete_role_costs=True, support_policy='bounded', completed_runs=history)
    selected, score, coordination = coordinate(make, evaluate)
    selected, score, repair = polish(selected, score, inputs=inputs, targets=targets,
        hard_length=len(hard), state_at=state_at, evaluate=evaluate)
    if repair['status'] != 'checked':
        raise ValueError('Baseline guard failure; do not compare policy on an invalid seed')
    days, states, attempts = selected
    initial_miles = sum(planner._midpoint(r) for r in runs(days))
    baseline_fees = per_run_shortfall_components(days, states, inputs['request'])
    baseline_costs = fee_ablations(score['combined_cost'], baseline_fees)
    rows = []
    blocked = set(inputs.get('forced_rest_offsets') or ()) | set(inputs.get('completed_run_offsets') or ())
    for facts, candidate in counterfactuals(days, states, len(hard), blocked=blocked):
        if len(rows) >= 1600:
            raise ValueError('Bounded short-support audit exceeds 1600 variants per decision')
        assert candidate[:len(hard)] == hard
        assert all(candidate[i] == days[i] for i in range(len(days)) if i not in facts['edits'])
        total = sum(planner._midpoint(r) for r in runs(candidate))
        assert abs(total - initial_miles - facts['added_miles']) < 1e-8
        checked = check_sequence(candidate, context, state_at, facts['edits'])
        # Score failed probes too for auditability, but never let them win.
        measured = evaluate((candidate, states, attempts))
        fees = per_run_shortfall_components(candidate, states, inputs['request'])
        costs = fee_ablations(measured['combined_cost'], fees)
        rows.append({**facts, 'guards_pass': checked['checks_pass'],
            'failed_offsets': checked['failed_offsets'],
            'legacy_intent_pass': all(m['within_intent_band'] for m in checked['menu']),
            'intent_checks': checked['menu'], 'costs': costs, 'fees': fees,
            'cost_deltas': {key: value - baseline_costs[key] for key, value in costs.items()},
            'funding_cost': measured['combined_target_cost'],
            'recovery_cost': measured['trajectory_recovery_cost'],
            'coaching_cost': measured['combined_coaching_cost'],
            'distribution_cost': planner.PROGRAM_FIT_UNIT * planner.AEROBIC_SUPPORT_FIT_FRACTION
                                 * measured['local_support']['violation'],
            'other_shape_cost': planner.PROGRAM_FIT_UNIT * measured['combined_shape_violation']
                                - fees['fragmentation_fee'],
            'rate': measured['policy_rate'], 'target': measured['soft_target'],
            'sessions': measured['sessions']})
    assert rows[0]['sessions'] == score['sessions']
    assert rows[0]['costs']['legacy'] == score['combined_cost']
    components = ('funding_cost', 'recovery_cost', 'coaching_cost', 'distribution_cost', 'other_shape_cost')
    for row in rows:
        explained = sum(row[key] - rows[0][key] for key in components)
        assert abs(explained - row['cost_deltas']['distribution_only']) < 1e-8
    return {'origin': states[0].as_of.isoformat(), 'seconds': perf_counter() - begin,
        'baseline': score, 'baseline_repair': repair, 'coordination': coordination,
        'rows': rows, 'summary': summarize(rows)}


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
        parser.error('Require a fresh output path')
    source = json.loads(args.report.read_text())
    template, history = synthetic_history(datetime.fromisoformat(source['realized_execution']['start_at']))
    found = {}
    def observe(inputs, hard, score):
        opening = inputs['daily_states'][0]
        key = opening.as_of.isoformat()
        if key in found:
            return None
        for n in (7, 14, 28):
            actual = projection._load_window(history, opening.as_of, n)
            captured = getattr(opening.recent_load, f'trailing_{n}d')
            if actual.activity_count != captured.activity_count or abs(actual.distance_miles - captured.distance_miles) > 1e-8:
                raise ValueError('Seed history does not match this decision')
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
        found[key] = compare(inputs, hard, score, history, state_at)
        return None
    reconstructed = investigate(source, candidate_observer=observe)
    result = {'scope': __doc__, 'seconds': reconstructed['seconds'],
        'matched_saved_decisions': reconstructed['matched_decisions'], 'decisions': list(found.values()),
        'limitations': ['Synthetic observed history, not recorded-HR or daily execution.',
            'Fixed hard prefix and unedited soft sessions; bounded probe placements, not exhaustive optimization.',
            'Fee ablations cross-score identical sequences. No alternative role coordination or native short-menu enablement.',
            'Guard checks are partial modeled constraints, not full acceptance or physiological clearance.',
            'The 0.5-mile probe edge is a falsification test, not a proposed minimum prescription.']}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    for decision in result['decisions']:
        print(decision['origin'], 'variants', len(decision['rows']), 'seconds', round(decision['seconds'], 3), flush=True)
        for eligibility, policies in decision['summary'].items():
            for policy, summary in policies.items():
                best = summary['best']
                print(eligibility, policy, {k: best[k] for k in
                    ('family', 'edits', 'pieces', 'added_miles', 'costs', 'rate')} if best else None, flush=True)


if __name__ == '__main__':
    main()
