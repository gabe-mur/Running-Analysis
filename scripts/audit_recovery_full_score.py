"""Fixed-dose rest-placement audit; no search, policy retuning or deployment.

Reproduce saved selected scores before comparing adjacent-date redistributions.
Gate diagnostics distinguish active failures from stricter cadence advisories.
Saved finalist coverage is not exhaustive beam/search coverage.
"""

from argparse import ArgumentParser
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
import json

import yaml

from run_analysis import adherence_projection as projection, weekly_schedule as planner
from run_analysis.prescription_distance import canonical_distance_policy
from run_analysis.aerobic_acceptance import assess_aerobic_exposure
from run_analysis.selection_objective import COMPLETED_SELECTION_HISTORY, shared_objective_policy
from scripts.audit_backward_controls import plain, validate_control
from scripts.audit_continuation_upload import evaluate_physical, hard_score
from scripts.audit_distance_continuity import components
from scripts.audit_recovery_placement import audit, move_session
from scripts.audit_regrowth_gates import check_candidate
from scripts.boundary_funding_prototype import boundary_funding_policy
from scripts.continuation_replan import fingerprint
from scripts.expected_completion import ExpectedCompletion, expected_completion_policy
from scripts.finalized_scoring_trial import finalized_scoring_policy
from scripts.opportunity_search_repair import opportunity_search_policy
from scripts.volume_path_prototype import volume_path_policy


def passes_checked_constraints(sequence, gates, acceptance):
    # Strict eight-day recurrence is diagnostic, not an active rejection policy.
    return (sequence['checks_pass'] and not gates['failures']
            and not gates['whole_hard_peak_failures'] and not acceptance.rejected
            and acceptance.reason != 'missing_context')


def cadence_charges(program, opening, config):
    """Expose the existing scorer's daily overdue charges, including the soft tail."""
    charges = []
    coaching = config.get('coaching', {})
    for role, kinds, age, reference in (
        ('long', {planner.WorkoutType.LONG}, opening.days_since_long_run,
         float(coaching.get('long_run_recency_reference_days', 7))),
        ('quality', planner.QUALITY_WORKOUT_TYPES, opening.days_since_quality_run,
         float(coaching.get('quality_recency_reference_days', 7))),
    ):
        if age is None:
            continue
        occurrences = {d.date for d in program if d.recommendation and d.recommendation.workout_type in kinds}
        previous = opening.as_of.date()
        for day in program:
            if not planner._is_taper_or_race_date(config, day.date):
                age += max(0, (day.date - previous).days)
                cost = planner._key_session_cadence_overdue_cost(age, reference)
                if cost:
                    charges.append(dict(date=str(day.date), role=role, age_days=age,
                                        reference_days=reference, cost=cost))
            if day.date in occurrences:
                age = 0.
            previous = day.date
    return charges


def panel(snapshot, hard, full, template, history, config, projector, prior, target_evidence,
          *, candidate_observer=None, distance_probe_days=0, include_date_probes=True):
    origin = datetime.fromisoformat(snapshot['decision_start_date']).replace(tzinfo=template.as_of.tzinfo)
    opening = ExpectedCompletion(template, history, config).at(origin)
    coaching = config.get('coaching', {})

    def state(at):
        return projection._state_at(template, history, at, opening.recent_load.capacity_reference_miles,
            easy_baseline_half_life_days=float(coaching.get('capacity_retention_half_life_days', 84)),
            fatigue_half_life_days=float(coaching.get('continuous_fatigue_half_life_days', 7)))

    hours = sorted(config.get('weather', {}).get('automatic_run_time_hours_local', [7, 12, 19]))
    options = [[state(origin + timedelta(days=i, hours=h)) for h in hours] for i in range(21)]
    inputs = dict(daily_states=[state(origin), *[opts[0] for opts in options[1:]]],
        daily_state_options=options,
        request=projection.RecommendationRequest(health_status=projection.CurrentHealthStatus.NORMAL),
        config=config, target_distance_range=(snapshot['target_low_miles'], snapshot['target_high_miles']),
        completed_miles_by_offset={}, expected_target_projector=projector,
        prior_distance_midpoints_by_date={d.date: planner._midpoint(d.recommendation)
            for d in (prior.planning_days if prior else []) if d.recommendation},
        target_evidence=target_evidence, forced_rest_offsets=set(), completed_run_offsets=set())
    evidence = dict(template=template, runs=history, source='realized_simulation')

    def evaluate(program):
        days = program[:21]
        scored = evaluate_physical(inputs, days, evidence, full=program)
        parts = components(scored)
        gates = check_candidate(snapshot, inputs, days)
        selected = snapshot['planner_diagnostics']['continuation_selection']
        offsets = [i for i, d in enumerate(days) if d.recommendation]
        physical = fingerprint(days)
        _, _, details = hard_score(inputs, offsets, days=days)
        acceptance = assess_aerobic_exposure(**details['aerobic_acceptance_facts'],
            planning_mode=target_evidence.planning_mode.value)
        charges = cadence_charges(program, inputs['daily_states'][0], config)
        cadence_cost = sum(row['cost'] for row in charges)
        if cadence_cost > parts['coaching'] + 1e-8:
            raise ValueError('Cadence decomposition exceeds coaching cost')
        result = dict(cost=scored['continuation']['combined_cost'], components=parts,
            cadence_charges=charges, cadence_cost=cadence_cost,
            other_coaching_cost=parts['coaching'] - cadence_cost,
            aerobic_acceptance=asdict(acceptance),
            passes_checked_constraints=passes_checked_constraints(
                scored['continuation']['fixed_sequence_checks'], gates, acceptance),
            recovery_components=scored['continuation'].get('trajectory_recovery_components'),
            coaching_components=scored['continuation'].get('combined_coaching_components'),
            sequence_checks=scored['continuation']['fixed_sequence_checks'], gates=gates,
            finalist_coverage=dict(
                matching_calendar=sum(r.get('offsets') == offsets for r in selected['finalists']),
                matching_prescription=sum(r.get('hard_fingerprint') == physical
                    for r in selected['finalists'])),
            hard_fingerprint=physical)
        if candidate_observer is not None:
            # Read-only paired assessment after unchanged full-program checks.
            # Its result never enters the legacy score or candidate eligibility.
            result['replacement_diagnostic'] = candidate_observer(program, result)
        return result

    token = COMPLETED_SELECTION_HISTORY.set(evidence)
    try:
        with canonical_distance_policy(), shared_objective_policy(prior_schedule=prior), \
                finalized_scoring_policy(), opportunity_search_policy(), \
                expected_completion_policy(config), volume_path_policy(), boundary_funding_policy():
            baseline = evaluate(full)
            validate_control(snapshot, inputs, hard, full, history, baseline['cost'])
            probes = []
            for i, day in enumerate(hard if include_date_probes else []):
                if day.recommendation is None:
                    continue
                for delta in (-1, 1):
                    j = i + delta
                    if j < 0 or j >= len(hard) or hard[j].recommendation is not None:
                        continue
                    revised = move_session(full, day.date, hard[j].date)
                    result = evaluate(revised)
                    probes.append(dict(moves=[dict(source=str(day.date), destination=str(hard[j].date))],
                        role=day.recommendation.workout_type.value, **result))
            # Move all hard-horizon longs together: preserve internal long cadence,
            # while explicitly retaining its historical/soft-boundary consequences.
            longs = [d for d in hard if d.recommendation and d.recommendation.workout_type == planner.WorkoutType.LONG]
            if include_date_probes and len(longs) > 1:
                revised, moves = full, []
                for day in longs:
                    dest = day.date + timedelta(days=1)
                    if dest > hard[-1].date or next(d for d in revised if d.date == dest).recommendation:
                        break
                    revised = move_session(revised, day.date, dest)
                    moves.append(dict(source=str(day.date), destination=str(dest)))
                else:
                    probes.append(dict(moves=moves, role='paired_longs', **evaluate(revised)))
            if distance_probe_days:
                from scripts.finalized_distance_search import neighbors
                for changed in neighbors(hard, inputs, state, visible_days=distance_probe_days):
                    changes = [dict(date=str(a.date), before=a.recommendation.planned_distance_miles,
                                    after=b.recommendation.planned_distance_miles)
                               for a, b in zip(hard, changed)
                               if a.recommendation and b.recommendation and
                               a.recommendation.model_dump() != b.recommendation.model_dump()]
                    probes.append(dict(distance_changes=changes, role='existing_easy_distance_menu',
                                       **evaluate(changed+full[len(hard):])))
            for probe in probes:
                probe['delta'] = probe['cost'] - baseline['cost']
                probe['component_deltas'] = {k: v - baseline['components'][k]
                    for k, v in probe['components'].items()}
            return dict(baseline=baseline, probes=probes, config=config,
                scope='Fixed physical doses and saved soft tail. Active gates and advisory-only cadence '
                      'reported separately. Finalist absence does not prove search omission. '
                      'No candidate reoptimization, future churn test or physiological calibration.')
    finally:
        COMPLETED_SELECTION_HISTORY.reset(token)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--journal', type=Path, required=True)
    parser.add_argument('--indices', type=int, nargs='+', default=[0, 9, 18, 31])
    parser.add_argument('--output-json', type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        parser.error('Choose a new output path')
    config = yaml.safe_load(Path('config.example.yaml').read_text())
    result = audit(args.journal, config, set(args.indices), observer=panel)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(plain(result), indent=2, allow_nan=False)
    with args.output_json.open('x') as stream:
        stream.write(encoded + '\n')
    print(f'Completed in {result["seconds"]:.2f}s')


if __name__ == '__main__':
    main()
