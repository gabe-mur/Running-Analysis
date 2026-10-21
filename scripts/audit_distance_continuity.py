"""Fixed-program distance probes on saved daily/upload decisions; no new search.

Hold every other prescription and the entire soft continuation fixed. Recompute
conditional state, full objective and guards for each alternative easy-run dose.
This is a local diagnostic, not a stability replay or permission to publish it.
"""
from argparse import ArgumentParser
from datetime import date
import json
from pathlib import Path

from run_analysis import weekly_schedule as planner
from run_analysis.selection_objective import PRIOR_DISTANCE_PRESCRIPTIONS
from scripts.audit_continuation_upload import reconstruct, unpack_soft, evaluate_physical, hard_score
from scripts.audit_regrowth_gates import check_candidate
from scripts.continuation_replan import fingerprint
from scripts.expected_completion import expected_completion_policy


def replace_easy_dose(days, on_date, target, bounds):
    if not (0 < bounds[0] <= target <= bounds[1]) or abs(target * 10 - round(target * 10)) > 1e-8:
        raise ValueError('Require a positive tenth-mile target within bounds')
    matches = [i for i, d in enumerate(days) if d.date == on_date]
    if len(matches) != 1:
        raise ValueError('Require exactly one matching date')
    index = matches[0]
    rec = days[index].recommendation
    if rec is None or rec.workout_type != planner.WorkoutType.EASY:
        raise ValueError('Only easy-run distance probes are supported')
    updated = list(days)
    updated[index] = days[index].model_copy(update={'recommendation': rec.model_copy(update={
        'distance_target_miles': target, 'distance_range_miles': tuple(bounds)})})
    return updated


def components(scored):
    hard, full = scored['hard'], scored['continuation']
    result = dict(target=full['combined_target_cost'], coaching=full['combined_coaching_cost'],
        recovery=full['trajectory_recovery_cost'],
        support=planner.PROGRAM_FIT_UNIT * planner.AEROBIC_SUPPORT_FIT_FRACTION * full['combined_support_violation'],
        shape=planner.PROGRAM_FIT_UNIT * full['combined_shape_violation'],
        switching=hard['distance_switching_cost'])
    if abs(sum(result.values()) - full['combined_cost']) > 1e-8:
        raise ValueError('Score components do not sum to the full objective')
    return result


def allocation_trace(inputs, saved, days, on_date):
    """Observe existing menus and shortlist callbacks; do not add candidates."""
    original_options = planner._easy_allocation_options
    original_allocate = planner._allocate_visible_distance_ranges
    menus, shortlisted = [], []
    def options(*args, **kwargs):
        choices, ranges = original_options(*args, **kwargs)
        if kwargs['plan_date'] == on_date:
            menus.append(dict(choices=choices, explicit_ranges=ranges))
        return choices, ranges
    def allocate(*args, **kwargs):
        selector = kwargs.get('allocation_selector')
        if selector is not None:
            def observed(candidate):
                value = selector(candidate)
                rec = next(d.recommendation for d in candidate if d.date == on_date)
                shortlisted.append(dict(target=planner._midpoint(rec) if rec else None,
                    cost=value, fingerprint=fingerprint(candidate)))
                return value
            kwargs = dict(kwargs, allocation_selector=observed)
        return original_allocate(*args, **kwargs)
    winner = next(r for r in saved['finalists'] if r.get('hard_fingerprint') == saved['selected_hard_fingerprint'])
    try:
        planner._easy_allocation_options = options
        planner._allocate_visible_distance_ranges = allocate
        regenerated, score, _ = hard_score(inputs, winner['offsets'],
            roles={int(k): v for k, v in winner.get('roles', {}).items()})
    finally:
        planner._easy_allocation_options = original_options
        planner._allocate_visible_distance_ranges = original_allocate
    return dict(menus=menus, shortlisted=shortlisted,
        matches_published=fingerprint(regenerated) == fingerprint(days),
        regenerated_cost=score.total_cost)


def investigate(reports, *, on_date, indices):
    if not reports:
        raise ValueError('Require saved reports')
    programs = {}
    # Preserve each branch's exact physical prescriptions, never regenerate an
    # alternate under the receiving branch's different prior-plan inputs.
    for label, report in reports.items():
        programs[label] = {}
        def collect(index, inputs, days, evidence):
            if index in indices:
                saved = report['replans'][index]['planner_diagnostics']['continuation_selection']
                full = days + unpack_soft(saved['soft_sessions'])
                if fingerprint(full) != saved['selected_program_fingerprint']:
                    raise ValueError('Saved full-program identity mismatch')
                programs[label][index] = (days, full)
        reconstruct(report, collect)
    rows = []
    for label, report in reports.items():
        def observe(index, inputs, days, evidence):
            if index not in indices:
                return
            snapshot = report['replans'][index]
            saved = snapshot['planner_diagnostics']['continuation_selection']
            _, own_full = programs[label][index]
            prior = PRIOR_DISTANCE_PRESCRIPTIONS.get() or {}
            prior_rec = prior.get(on_date)
            variants = [('published', days, own_full)]
            # These are the actual canonical targets/ranges seen in the saved
            # Sep. 24 easy-run prescriptions, not a newly chosen search grid.
            for target, bounds in ((3.3, (3., 3.5)), (3.5, (3.2, 3.8)), (3.8, (3.5, 4.))):
                hard = replace_easy_dose(days, on_date, target, bounds)
                variants.append((f'easy_{target}', hard, hard + own_full[len(days):]))
            if prior_rec is not None:
                hard = replace_easy_dose(days, on_date, planner._midpoint(prior_rec), prior_rec.distance_range_miles)
                variants.append(('prior_dose', hard, hard + own_full[len(days):]))
            for other, other_report in reports.items():
                if other == label:
                    continue
                if (other_report['replans'][index]['generated_at'] != snapshot['generated_at']
                    or other_report['realized_execution'] != report['realized_execution']
                    or other_report['synthetic_sandbox']['config'] != report['synthetic_sandbox']['config']):
                    raise ValueError('Cross-branch probes require aligned decisions and execution')
                hard, full = programs[other][index]
                variants.append((f'other_program_{other}', hard, full))
            probes = []
            for name, hard, full in variants:
                with expected_completion_policy(inputs['config']):
                    scored = evaluate_physical(inputs, hard, evidence, full=full)
                parts = components(scored)
                if name == 'published' and abs(scored['continuation']['combined_cost'] - saved['search_cost']) > 1e-8:
                    raise ValueError('Published score did not reproduce')
                gates = check_candidate(snapshot, inputs, hard)
                run = next(d.recommendation for d in hard if d.date == on_date)
                probes.append(dict(name=name, target=planner._midpoint(run), bounds=run.distance_range_miles,
                    total=sum(parts.values()), hard_total=scored['hard']['total_cost'], components=parts,
                    recovery_components=scored['continuation']['trajectory_recovery_components'],
                    guards=scored['continuation']['fixed_sequence_checks'], gates=gates,
                    hard_fingerprint=fingerprint(hard), full_fingerprint=fingerprint(full)))
            row = dict(branch=label, index=index, generated_at=snapshot['generated_at'],
                prior_target=planner._midpoint(prior_rec) if prior_rec else None,
                prior_bounds=prior_rec.distance_range_miles if prior_rec else None,
                material_evidence=snapshot['material_evidence_reasons'],
                recovery_surprise=snapshot['recovery_surprise_units'], probes=probes,
                allocation_trace=allocation_trace(inputs, saved, days, on_date))
            rows.append(row)
            print(f'Distance probes {label}/{index}: ' + ', '.join(
                f"{p['name']}={p['total']:.6f}" for p in probes), flush=True)
        reconstruct(report, observe)
    return dict(scope=__doc__, on_date=on_date.isoformat(), rows=rows)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--reports', type=Path, nargs='+', required=True)
    parser.add_argument('--output-json', type=Path, required=True)
    parser.add_argument('--date', type=date.fromisoformat, default=date(2026, 9, 24))
    parser.add_argument('--indices', type=int, nargs='+', default=[1, 2, 3])
    args = parser.parse_args()
    if args.output_json.exists():
        parser.error('Require a new output path')
    result = investigate({p.stem: json.loads(p.read_text()) for p in args.reports},
                         on_date=args.date, indices=args.indices)
    with args.output_json.open('x') as stream:
        json.dump(result, stream, indent=2, default=str)


if __name__ == '__main__':
    main()
