"""Bounded read-only trial of shared workload accounting and recovery sensitivity.

Uses original files, not cached plans or prescription matches. Counterfactual
session comparisons are diagnostics, not retrospective forecasts or advice.
"""
from argparse import ArgumentParser
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from itertools import product
import json
from pathlib import Path
import sqlite3
from time import perf_counter

from run_analysis.config import load_config
from run_analysis.fit import parse_fit
from run_analysis.tcx import parse_tcx
from run_analysis.workload_profile import observed_profile, prescribed_profile
from run_analysis.workload_accounting import Quantity, observed_workload, projected_workload
from run_analysis.recovery_assessment import AssessmentPolicy, SessionEntry, sensitivity_panel
from run_analysis.web.schemas import RecommendationResponse, WorkoutStep


from run_analysis.adaptive.evidence_history import load_sessions


def candidates(at):
    # Declared test doses/rates, not recommended workouts or personal HR forecasts.
    definitions = [('easy_40', 4., (38., 42.), None, (1.4, 2.0), None),
                   ('long_90', 8., (85., 95.), None, (1.4, 2.0), None),
                   ('quality_3x5', 4.5, (41., 46.), 5., (1.4, 2.0), (3., 4.))]
    result = {}
    for label, distance, minutes, rep_minutes, base, work in definitions:
        structure = ([WorkoutStep(instruction='warmup', duration_minutes=12),
            WorkoutStep(instruction='work', phase='work', repetitions=3,
                        work_duration_minutes=rep_minutes, recovery_duration_minutes=2),
            WorkoutStep(instruction='cooldown', duration_minutes=10)] if rep_minutes else [])
        rec = RecommendationResponse(generated_at=at, fitness_state_as_of=at, planned_for=at,
            workout_type='intervals' if rep_minutes else 'easy', title=label,
            distance_target_miles=distance, distance_range_miles=(distance-.2, distance+.2),
            confidence='low', readiness='ready', structure=structure)
        result[label] = projected_workload(prescribed_profile(rec),
            duration_minutes=Quantity(*minutes, basis='declared_test_duration'),
            base_rate=Quantity(*base, basis='declared_test_hr_rate'),
            work_rate=Quantity(*work, basis='declared_test_hr_rate') if work else None)
    return result


def compact(panel):
    return dict(status=panel['status'], agreement=panel['agreement'],
                counts=dict(Counter(r['status'] for r in panel['scenarios'])),
                per_policy={name: dict(Counter(r['status'] for r in panel['scenarios']
                                               if r['policy'] == name))
                            for name in dict.fromkeys(r['policy'] for r in panel['scenarios'])})


def trial(database, config, since, reference_before, until):
    started = perf_counter()
    sessions = load_sessions(database, config, since, until)
    reference = [entry for entry, _ in sessions if entry.end < reference_before]
    if not reference:
        raise ValueError('Need a preceding reference set; cannot normalize with future doses')
    max_minutes = max(x.workload.running_minutes.high for x in reference)
    max_points = max(x.workload.cardio_points.high for x in reference)
    policies = [AssessmentPolicy(f'r{r}_c{c}_e{factor}', max_minutes*factor,
        max_points*factor, r, c) for r, c, factor in product((12., 18., 24.), (8., 12., 18.), (1., 1.25, 1.5))]
    doses = candidates(reference_before)
    cases = []
    for session, profile in sessions:
        if session.start < reference_before:
            continue
        chronological = [x for x, _ in sessions if x.end <= session.end]
        for mode, history in [('isolated_previous_session', [session]), ('recorded_accumulation', chronological)]:
            for gap in (24, 36, 48):
                for label, proposed in doses.items():
                    panel = sensitivity_panel(history, proposed, session.end+timedelta(hours=gap), policies)
                    cases.append(dict(session_key=session.key, start=session.start.isoformat(), mode=mode,
                        gap_hours=gap, next_dose=label, **compact(panel)))
    return dict(version='workload_contract_trial_v1', elapsed_seconds=perf_counter()-started,
        source='original_recorded_files', config_zones=config['zones'],
        reference_before=reference_before.isoformat(), reference_session_keys=[x.key for x in reference],
        reference_max_running_minutes=max_minutes, reference_max_cardio_points=max_points,
        policies=[asdict(x) for x in policies], candidate_doses={k: asdict(v) for k, v in doses.items()},
        sessions=[dict(key=e.key, start=e.start.isoformat(), end=e.end.isoformat(),
                       workload=asdict(e.workload), hr_coverage=p['cardiovascular']['hr_coverage'])
                  for e, p in sessions], cases=cases,
        limitations=['Sensitivity experiment, not retrospective prediction or physiological calibration',
                     'Reference maxima are scaling examples, not proof of tolerated workload',
                     'Two-demand assessment lacks fast-work, terrain and novelty protections',
                     'No calendar optimization, scheduling stability or progression claim'],
        planner_influence=False)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('config.yaml'))
    parser.add_argument('--since', required=True, help='ISO date, inclusive UTC')
    parser.add_argument('--reference-before', required=True, help='Frozen reference cutoff, ISO date UTC')
    parser.add_argument('--until', required=True, help='ISO date, exclusive UTC')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    since, cutoff, until = [datetime.fromisoformat(x).replace(tzinfo=timezone.utc)
                            for x in (args.since, args.reference_before, args.until)]
    if not since < cutoff < until:
        parser.error('Dates must be ordered since < reference-before < until')
    if args.output.exists():
        parser.error('Choose a new output path')
    result = trial(args.database, load_config(args.config), since, cutoff, until)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(seconds=result['elapsed_seconds'], sessions=len(result['sessions']),
        policies=len(result['policies']), cases=len(result['cases']),
        outcomes=dict(Counter(c['status'] for c in result['cases'])))))


if __name__ == '__main__':
    main()
