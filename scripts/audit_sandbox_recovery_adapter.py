"""Compare recovery models on saved candidates; no search or live policy change."""
from argparse import ArgumentParser
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime, timedelta
from pathlib import Path
from time import perf_counter
import json

import yaml

from run_analysis.recovery_assessment import sensitivity_panel
from run_analysis.recovery import EASY_RUN_RESIDUAL_LIMIT, TAXING_RUN_RESIDUAL_LIMIT
from run_analysis.workload_accounting import Quantity
from run_analysis import weekly_schedule as planner
from scripts.audit_backward_controls import plain, digest
from scripts.audit_recovery_baseline import select_policies
from scripts.audit_recovery_generalization import recurring_requirements
from scripts.audit_recovery_placement import audit
from scripts.audit_recovery_full_score import panel
from scripts.expected_completion import ExpectedCompletion
from scripts.sandbox_recovery_adapter import (
    PhaseAssumptions, capture, history_entry, review_completion, session_key, projection_conflicts,
)
from scripts.typed_continuation_prototype import runs


def policies():
    return {name: tuple(p for p in select_policies(structured)[0]
                        if recurring_requirements(p)['supported'])
            for name, structured in (('two_channels', False), ('three_channels', True))}


def phase_assumptions():
    return PhaseAssumptions('Declared sensitivity bounds, not calibrated HR forecasts',
        Quantity(2, 3, basis='base HR phase assumption'),
        Quantity(3, 5, basis='work HR phase assumption'),
        Quantity(2, 4, basis='between-work HR phase assumption'))


def paired_panel(snapshot, hard, full, template, history, config, projector, prior, evidence,
                 *, families, distance_probe_days=7):
    started = perf_counter()
    origin = datetime.fromisoformat(snapshot['decision_start_date']).replace(tzinfo=template.as_of.tzinfo)
    model = ExpectedCompletion(template, history, config)
    assumptions = phase_assumptions()
    # This adapter compares near-seven-day assessments, not a replacement full
    # objective. Keep the untouched 49-day program in legacy scoring, explicitly
    # expose its mapping conflicts, and do not fabricate a valid tail forecast.
    def near(program):
        return [d for d in program if d.date < origin.date()+timedelta(days=7)]
    conflicts = projection_conflicts(model, full, assumptions=assumptions)
    issued = capture(model, near(full), issued_at=origin, assumptions=assumptions)
    cache, contracts = {}, {}
    assessment_seconds = 0.

    def assess(contract, key):
        nonlocal assessment_seconds
        proposed = contract.option(key)
        prefix = contract.prefix(key)
        signature = digest((prefix, proposed))
        if signature not in cache:
            t = perf_counter()
            cache[signature] = {name: dict(Counter(sensitivity_panel(prefix,
                proposed.workload, proposed.planned_for, (policy,),
                proposed_work_minutes=proposed.work_minutes)['status'] for policy in group))
                for name, group in families.items()}
            assessment_seconds += perf_counter()-t
        return cache[signature]

    def observe(program, legacy):
        contract = capture(model, near(program), issued_at=origin, assumptions=assumptions)
        contracts[legacy['hard_fingerprint']] = (contract, legacy['passes_checked_constraints'])
        comparisons, planned = [], []
        for rec in runs(program):
            if rec.planned_for.date() >= origin.date()+timedelta(days=7):
                break
            state = model.at(rec.planned_for, planned)
            limit = (TAXING_RUN_RESIDUAL_LIMIT if rec.workout_type in
                     {planner.WorkoutType.LONG, *planner.QUALITY_WORKOUT_TYPES} else EASY_RUN_RESIDUAL_LIMIT)
            key = session_key(rec.planned_for)
            comparisons.append(dict(key=key, at=rec.planned_for, kind=rec.workout_type.value,
                target=rec.planned_distance_miles,
                legacy=dict(residual=state.recovery_residual_load, limit=limit,
                            ready=state.recovery_residual_load <= limit),
                replacement_by_family=assess(contract, key)))
            planned.append(rec)
        return dict(contract_digest=contract.digest, near_seven_day_sessions=comparisons,
                    planner_influence=False)

    comparison = panel(snapshot, hard, full, template, history, config, projector, prior, evidence,
                       candidate_observer=observe, distance_probe_days=distance_probe_days,
                       include_date_probes=False)
    first, second = runs(full)[:2]
    first_key, next_key = session_key(first.planned_for), session_key(second.planned_for)
    original = issued.option(next_key)
    variants, eligibility = {}, set()
    for n, (contract, passes) in enumerate(contracts.values()):
        option = contract.option(next_key, option_key=f'existing-menu-{n}')
        if passes and option.target_miles < original.target_miles and all(
                getattr(option.workload, field).low <= getattr(original.workload, field).low+1e-8
                and getattr(option.workload, field).high <= getattr(original.workload, field).high+1e-8
                for field in ('distance_meters', 'running_minutes', 'cardio_points')):
            # Only a next-dose edit belongs in this response scope; variants
            # changing another prescription cannot donate their eligibility.
            before = json.loads(issued.payload)['prescriptions']
            after = json.loads(contract.payload)['prescriptions']
            if all(before[k] == after[k] for k in before if k != next_key):
                variants[option.key] = option
                eligibility.add(option.key)
    expected = next(e for e in issued.entries() if e.key == first_key)
    identical = replace(expected, workload=replace(expected.workload, source='observed',
        limitations=(*expected.workload.limitations, 'Identical-input accounting control, not recorded HR')))
    actual_ledger = issued.after_completion(identical)
    # Compare complete per-policy/channel results, not merely coarse statuses.
    parity = 0
    for group in families.values():
        for policy in group:
            before = sensitivity_panel(issued.prefix(next_key), original.workload, original.planned_for,
                (policy,), proposed_work_minutes=original.work_minutes)
            after = sensitivity_panel(tuple(e for e in actual_ledger if e.start < original.planned_for),
                original.workload, original.planned_for, (policy,), proposed_work_minutes=original.work_minutes)
            if before != after:
                raise AssertionError('Identical completion changed replacement recovery assessment')
            parity += 1

    native = model.ledger(second.planned_for, [first])
    native_first = next(r for r in native if r.start_time == first.planned_for)
    expected_state = model.at(second.planned_for, [first])
    completed_state = ExpectedCompletion(template, native, config).at(second.planned_for)
    if expected_state != completed_state:
        raise AssertionError('Legacy expected-completion state changed on identical upload')
    # The original simulator does not have HR. Show the actual evidence gap;
    # never certify HR compliance using copied projected zone assumptions.
    missing_hr = history_entry(native_first)
    responses = {}
    for label, completion in (('identical_accounting_control', identical), ('native_upload_missing_hr', missing_hr)):
        responses[label] = {name: review_completion(issued, completion, next_key,
            alternatives=variants.values(), eligible_keys=eligibility, policies=group,
            minimum_target_miles=min([original.target_miles, *(o.target_miles for o in variants.values())]),
            adjustment_source='Existing full-program-eligible same-session menu boundary',
            readiness='normal') for name, group in families.items()}
    comparison.update(issued_snapshot=json.loads(issued.payload),
        full_program_projection_conflicts=conflicts,
        replacement_full_program_projectable=not conflicts,
        identical_completion_policy_checks=parity, legacy_completion_parity=True,
        response_checks=responses, eligible_next_dose_reductions=len(variants),
        distinct_assessments=len(cache), assessment_seconds=assessment_seconds,
        total_panel_seconds=perf_counter()-started,
        scope='Saved fixed candidates, unchanged legacy full checks, replacement near-seven-day '
              'assessment only. No replacement whole-program ranking or daily replanning. '
              'Unknown synthetic historical HR/work remains unknown; HR phase bounds are assumptions.')
    return comparison


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--journal', type=Path, required=True)
    parser.add_argument('--indices', nargs='+', type=int, default=[0])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output path')
    families = {} if args.preflight_only else policies()
    config = yaml.safe_load(Path('config.example.yaml').read_text())
    def observe(*values):
        if args.preflight_only:
            snapshot, hard, full, template, history, config, *_ = values
            return dict(projection_conflicts=projection_conflicts(
                ExpectedCompletion(template, history, config), full, assumptions=phase_assumptions()),
                scope='Projection mapping only; no full-objective verification')
        return paired_panel(*values, families=families)
    result = audit(args.journal, config, set(args.indices), observer=observe)
    result['policies'] = {name: [asdict(p) for p in group] for name, group in families.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(plain(result), stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(f'Completed in {result["seconds"]:.2f}s', flush=True)


if __name__ == '__main__':
    main()
