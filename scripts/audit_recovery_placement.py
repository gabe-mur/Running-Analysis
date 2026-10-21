"""Inspect saved perfect-adherence recovery costs without calendar search.

Read the decision journal incrementally, verify physical prescriptions, and
reconstruct completed history from the preceding issued prescription. Reproduce
the selected full-program recovery score before any fixed-program probes.
This is a recovery-component audit, not whole-objective selection or calibration.
"""

from argparse import ArgumentParser
from dataclasses import replace
from datetime import datetime, timedelta
from functools import lru_cache
from hashlib import sha256
from math import log2
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
import json

import yaml

from run_analysis import adherence_projection as projection, weekly_schedule as planner
from run_analysis.prescription_load import prescribed_zone_minutes, prescribed_intensity_factor
from run_analysis.recovery import RECOVERY_HALF_LIFE_HOURS, EASY_RUN_RESIDUAL_LIMIT, TAXING_RUN_RESIDUAL_LIMIT
from scripts.audit_continuation_upload import unpack_soft
from scripts.continuation_replan import fingerprint
from scripts.expected_completion import ExpectedCompletion, expected_run
from scripts.replay_synthetic_easy_dose import synthetic_history
from scripts.typed_continuation_prototype import continued_recovery, runs


def selected_program(snapshot, previous):
    saved = snapshot['planner_diagnostics']['continuation_selection']
    winner = next(row for row in saved['finalists']
                  if row.get('hard_fingerprint') == saved['selected_hard_fingerprint'])
    if winner.get('hard_program'):
        hard = unpack_soft(winner['hard_program'])
    elif winner['source'] == 'prior_program' and previous:
        start = datetime.fromisoformat(snapshot['decision_start_date']).date()
        hard = [day for day in previous if start <= day.date < start + timedelta(days=21)]
        changes = {row['date']: datetime.fromisoformat(row['after'])
                   for row in saved.get('prior_program', {}).get('promoted_times', [])}
        hard = [day.model_copy(update={'recommendation': day.recommendation.model_copy(
            update={'planned_for': changes[day.date.isoformat()],
                    'fitness_state_as_of': changes[day.date.isoformat()]})})
            if day.date.isoformat() in changes else day for day in hard]
    else:
        raise ValueError('Cannot recover exact hard prescriptions')
    full = hard + unpack_soft(saved['soft_sessions'])
    if (fingerprint(hard) != saved['selected_hard_fingerprint']
            or fingerprint(full) != saved['selected_program_fingerprint']):
        raise ValueError('Saved physical prescription mismatch')
    return hard, full


def session_measurements(days, reference):
    result = []
    for rec in runs(days):
        minutes = max(planner._structured_duration_minutes(rec) or 0.,
                      rec.distance_range_miles[1] * 11.)
        moderate, hard = prescribed_zone_minutes(rec, minutes)
        load = planner._recommendation_load_units(rec, reference)
        result.append(dict(at=rec.planned_for.isoformat(), workout_type=rec.workout_type.value,
            quality_type=rec.quality_session_type.value if rec.quality_session_type else None,
            target=rec.planned_distance_miles, upper=rec.distance_range_miles[1],
            duration_denominator=minutes, moderate_minutes=moderate, hard_minutes=hard,
            intensity_factor=prescribed_intensity_factor(rec, minutes), load_units=load,
            isolated_hours_to_easy=max(0., RECOVERY_HALF_LIFE_HOURS * log2(load / EASY_RUN_RESIDUAL_LIMIT)),
            isolated_hours_to_taxing=max(0., RECOVERY_HALF_LIFE_HOURS * log2(load / TAXING_RUN_RESIDUAL_LIMIT))))
    return result


def move_session(days, source_date, destination_date):
    source = next(i for i, day in enumerate(days) if day.date == source_date)
    destination = next(i for i, day in enumerate(days) if day.date == destination_date)
    rec = days[source].recommendation
    if rec is None or days[destination].recommendation is not None:
        raise ValueError('Require a run moved to an empty date')
    at = rec.planned_for + timedelta(days=(destination_date - source_date).days)
    revised = list(days)
    revised[source] = days[source].model_copy(update={'day_role': 'rest_day', 'recommendation': None})
    revised[destination] = days[destination].model_copy(update={
        'day_role': days[source].day_role,
        'recommendation': rec.model_copy(update={'planned_for': at, 'fitness_state_as_of': at})})
    return revised


def recovery_panel(snapshot, hard, full, template, history, config, targets):
    origin = datetime.fromisoformat(snapshot['decision_start_date']).replace(tzinfo=template.as_of.tzinfo)
    model = ExpectedCompletion(template, history, config)
    opening = model.at(origin)
    for actual, saved, name in (
        (opening.recovery_residual_load, snapshot['opening_recovery_residual_load'], 'residual'),
        (opening.recent_load.continuous_distance_miles, snapshot['opening_continuous_distance_miles'], 'distance rate'),
        (opening.recent_load.continuous_short_term_distance_miles, snapshot['opening_short_term_distance_miles'], 'short rate'),
    ):
        if abs(actual - saved) > 1e-8:
            raise ValueError(f'Opening {name} mismatch: {actual} != {saved}')
    coaching = config.get('coaching', {})

    @lru_cache(None)
    def state_at(at):
        return projection._state_at(template, history, at, opening.recent_load.capacity_reference_miles,
            easy_baseline_half_life_days=float(coaching.get('capacity_retention_half_life_days', 84)),
            fatigue_half_life_days=float(coaching.get('continuous_fatigue_half_life_days', 7)))

    states = [state_at(origin + timedelta(days=i)) for i in range(len(full))]
    target = (snapshot['target_low_miles'], snapshot['target_high_miles'])

    def evaluate(program, *, use_config=config):
        cost, components = continued_recovery(program, states, use_config, target,
                                              state_at=state_at, targets=targets)
        planned, guards = [], []
        for rec in runs(program):
            conditional = model.at(rec.planned_for, planned)
            residual = conditional.recovery_residual_load
            limit = (TAXING_RUN_RESIDUAL_LIMIT if rec.workout_type in
                     {planner.WorkoutType.LONG, *planner.QUALITY_WORKOUT_TYPES} else EASY_RUN_RESIDUAL_LIMIT)
            guards.append(dict(at=rec.planned_for.isoformat(), residual=residual, limit=limit,
                               ready=residual <= limit))
            planned.append(rec)
        return dict(cost=cost, components=components, readiness_checks=guards,
                    readiness_pass=all(row['ready'] for row in guards))

    baseline = evaluate(full)
    saved = snapshot['planner_diagnostics']['winning_candidate']
    if abs(baseline['cost'] - saved['finalized_recovery_cost']) > 1e-8:
        raise ValueError(f'Recovery score mismatch: {baseline["cost"]} != {saved["finalized_recovery_cost"]}')
    for key, value in baseline['components'].items():
        if abs(value - saved['recovery_cost_components'][key]) > 1e-8:
            raise ValueError(f'Recovery component mismatch: {key}')

    probes = []
    # Probe only the hard prescription, preserving every other physical session
    # and the full saved tail. Freeze targets to isolate recovery arithmetic.
    for i, day in enumerate(hard):
        if day.recommendation is None:
            continue
        for delta in (-1, 1):
            j = i + delta
            if j < 0 or j >= len(hard) or hard[j].recommendation is not None:
                continue
            alternative = move_session(full, day.date, hard[j].date)
            probes.append(dict(source=day.date.isoformat(), destination=hard[j].date.isoformat(),
                role=day.recommendation.workout_type.value, **evaluate(alternative)))

    # Explicit diagnostic controls, not candidates: show which portion of this
    # selected recovery score depends on the old rest-day configuration.
    cadence_controls = []
    for rest_days in (0, 1, 2):
        changed = dict(config, coaching=dict(coaching, typical_rest_days_between_runs=rest_days))
        cadence_controls.append(dict(rest_days=rest_days, **evaluate(full, use_config=changed)))
    timing_controls = []
    hard_runs = runs(hard)
    for first, second in zip(hard_runs, hard_runs[1:]):
        if (first.workout_type not in planner.QUALITY_WORKOUT_TYPES
                or second.workout_type != planner.WorkoutType.LONG
                or (second.planned_for.date() - first.planned_for.date()).days != 1):
            continue
        for gap in (24., 29., 36.):
            at = first.planned_for + timedelta(hours=gap)
            if at.date() != second.planned_for.date():
                continue
            alternative = [day.model_copy(update={'recommendation': second.model_copy(
                update={'planned_for': at, 'fitness_state_as_of': at})})
                if day.recommendation is second else day for day in full]
            checked = evaluate(alternative)
            timing_controls.append(dict(quality_at=first.planned_for.isoformat(),
                long_at=at.isoformat(), gap_hours=gap, **checked))
    return dict(baseline=baseline, probes=probes, cadence_controls=cadence_controls,
        timing_controls=timing_controls,
        scope='Recovery costs only; fixed targets and tail; readiness is not full candidate feasibility. '
              'No whole-objective ranking, search coverage, physiological calibration or policy change.')


def audit(path, config, indices, *, observer=None):
    started = perf_counter()
    start = datetime.fromisoformat('2026-09-22T00:00:00+00:00')
    template, history = synthetic_history(start)
    previous = None
    prior_schedule = None
    rows = []
    source_hash = sha256()
    with path.open() as stream:
        for index, line in enumerate(stream):
            source_hash.update(line.encode())
            snapshot = json.loads(line)
            if snapshot['material_evidence_reasons'] or snapshot['recovery_surprise_units'] not in (None, 0.):
                raise ValueError('This reconstruction only supports zero-surprise perfect adherence')
            if snapshot['trigger'] == 'post_upload':
                at = datetime.fromisoformat(snapshot['source_activity_at'])
                rec = next(rec for rec in runs(previous) if rec.planned_for == at)
                history.append(replace(expected_run(rec, 11.), adherence_note='perfect'))
            hard, full = selected_program(snapshot, previous)
            origin = datetime.fromisoformat(snapshot['decision_start_date']).replace(tzinfo=start.tzinfo)
            activities = projection.planning_activities_for(history)
            _, _, evidence = planner.derive_weekly_target(activities, origin, config)
            projector = planner.make_expected_target_projector(activities, origin.date(), config,
                pace_min_mile=11., maximum_horizon_days=21,
                opening_target_range=(snapshot['target_low_miles'], snapshot['target_high_miles']),
                opening_evidence=evidence,
                ordinary_easy_midpoint_miles=snapshot['ordinary_easy_midpoint_miles'],
                prior_schedule=prior_schedule)
            targets = projector([], len(full))
            if (len(snapshot['target_trajectory']) != len(hard)
                    or any(abs(a-b)>1e-8 for x,y in zip(targets, snapshot['target_trajectory']) for a,b in zip(x,y))):
                raise ValueError(f'Target trajectory mismatch at decision {index}')
            score = snapshot['planner_diagnostics']['winning_candidate']
            row = dict(index=index, generated_at=snapshot['generated_at'],
                hard_fingerprint=fingerprint(hard), total_cost=score['total_cost'],
                recovery_cost=score['finalized_recovery_cost'],
                recovery_components=score['recovery_cost_components'],
                sessions=session_measurements(hard, snapshot['ordinary_easy_midpoint_miles']))
            if index in indices:
                if observer is None:
                    row['panel'] = recovery_panel(snapshot, hard, full, template, history, config, targets)
                    print(f'Verified recovery decision {index}; {len(row["panel"]["probes"])} fixed-target probes', flush=True)
                else:
                    row['panel'] = observer(snapshot, hard, full, template, history,
                                            config, projector, prior_schedule, evidence)
                    print(f'Completed observer audit for decision {index}', flush=True)
            rows.append(row)
            previous = full
            prior_schedule = SimpleNamespace(planning_days=hard, days=hard,
                expected_target_start_date=origin.date(),
                expected_target_trajectory=projector([], projector.prepared_horizon_days))
    if not indices <= {row['index'] for row in rows}:
        raise ValueError('Requested decision index missing from journal')
    return dict(source=str(path), source_sha256=source_hash.hexdigest(), decisions=rows, seconds=perf_counter()-started,
        assumptions='Perfect synthetic execution at 11 min/mile; fingerprints and opening state/score '
                    'must reproduce before probes. No observed HR/HRV or subjective recovery is simulated.')


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--journal', type=Path, required=True)
    parser.add_argument('--indices', type=int, nargs='+', default=[0, 9, 18, 31])
    parser.add_argument('--output-json', type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        parser.error('Choose a new output path')
    config = yaml.safe_load(Path('config.example.yaml').read_text())
    result = audit(args.journal, config, set(args.indices))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(f'{len(result["decisions"])} decisions inspected in {result["seconds"]:.2f}s')


if __name__ == '__main__':
    main()
