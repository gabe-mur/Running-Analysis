"""Chronological held-out phase-forecast audit, not actual-plan compliance replay.

Current config/roles held fixed. Upload availability is assumed at completion for
this experiment, not reconstructed. No cached pace/HR model results are consumed.
"""
from argparse import ArgumentParser
from dataclasses import asdict, is_dataclass, replace
from datetime import datetime, timedelta, timezone
import json
import sqlite3
from pathlib import Path
from time import perf_counter
from collections import Counter, defaultdict

from scripts.audit_workload_contract import load_sessions
from scripts.audit_workload_expectation import app_session_types
from run_analysis.config import load_config
from run_analysis.athlete_projection import (
    ForecastPhase, LoadSeed, snapshot, pace_reference, exposure_summary,
    project_load_history, issue_forecast, compare_completion,
)
from run_analysis.projection_evidence import evidence_from_activity
from run_analysis.performance_reference import phase_pace_reference
from run_analysis.workload_accounting import Quantity


def assumptions(state, day, config, *, reference_method='generalized'):
    """Predeclared sandbox intent, not fitted to the held-out trace/duration."""
    def reserve(low, high):
        rest, maximum = config['resting_hr'], config['max_hr']
        return Quantity((low-rest)/(maximum-rest), (high-rest)/(maximum-rest),
                        basis='zone_endpoints_as_trial_equilibrium_assumption_not_measured_kinetics')
    easy_hr = reserve(*config['zones']['z2'])
    work_hr = reserve(*config['zones']['z4'])
    recovery_hr = reserve(config['zones']['z1'][0], config['zones']['z2'][1])
    def reference(phase, hr, **dose):
        if reference_method == 'generalized':
            return phase_pace_reference(state, phase, equilibrium_reserve=hr, **dose)
        if reference_method != 'legacy_control':
            raise ValueError('Known reference method required')
        return pace_reference(state, phase, **dose, **({'reserve_band': hr} if phase == 'easy' else {}))
    refs = {'easy': reference('easy', easy_hr)}
    easy_speed = refs['easy']['speed_mps']
    if easy_speed is None:
        return None, refs, 'No prior easy pace; caller must supply a pace or use a time prescription'
    if day in {'2026-09-20', '2026-09-22'}:
        miles = 8.4 if day == '2026-09-20' else 4.7
        return (ForecastPhase('easy', None, miles*1609.344, easy_speed, easy_hr,
                              'sandbox_issued_distance_prior_easy_pace'),), refs, None
    target = {'seconds': 300} if day == '2026-09-18' else {'meters': 400}
    refs['work'] = reference('work', work_hr, **target)
    refs['recovery'] = reference('recovery', recovery_hr)
    if refs['work']['speed_mps'] is None or refs['recovery']['speed_mps'] is None:
        return None, refs, 'Unsupported work/recovery pace for distance-balanced structure; no easy-to-fast extrapolation'
    count, recovery, miles = (3, 120, 4.3) if day == '2026-09-18' else (6, 90, 4.6)
    phases = [ForecastPhase('warmup', 720, None, easy_speed, easy_hr, 'sandbox_12min_warmup')]
    for rep in range(count):
        phases.append(ForecastPhase(f'work_{rep+1}', target.get('seconds'), target.get('meters'),
            refs['work']['speed_mps'], work_hr, 'declared_work_target_prior_work_pace'))
        if rep < count-1:
            phases.append(ForecastPhase(f'recovery_{rep+1}', recovery, None,
                refs['recovery']['speed_mps'], recovery_hr, 'declared_recovery_duration_prior_pace'))
    phases.append(ForecastPhase('finish_easy', None, miles*1609.344, easy_speed, easy_hr,
                                'remaining_distance_not_additional_distance', True))
    return tuple(phases), refs, None


def audit(database, config, *, reference_method='generalized', since=None):
    started = perf_counter()
    roles = app_session_types(database)
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro', uri=True) as db:
        # Current metadata is held fixed in this audit, not reconstructed at upload.
        eligible = {str(key): not illness and (health in (None, '', 'normal')) and included != 0
            for key, illness, health, included in db.execute('''SELECT a.id,o.illness,o.health_tag,o.include_in_model
                FROM activities a LEFT JOIN run_overrides o ON o.activity_id=a.activity_id''')}
    history = []
    inventory = defaultdict(Counter)
    summary_only = []
    def missing_timing(activity, key, path):
        year = str(activity.start_time_utc.year)
        inventory[year]['sessions'] += 1
        inventory[year]['summary_only_missing_elapsed'] += 1
        inventory[year]['with_recorded_hr'] += int(any(p.heart_rate_bpm is not None for p in activity.trackpoints))
        summary_only.append(dict(key=key,start=activity.start_time_utc,
            lap_timer_seconds=activity.lap_recorded_time_s,meters=activity.total_distance_m,
            reason='No elapsed trace; do not infer zero load or rest'))
    def collect(activity, entry, profile):
        year = str(activity.start_time_utc.year)
        inventory[year]['sessions'] += 1
        inventory[year]['with_recorded_hr'] += int(any(p.heart_rate_bpm is not None for p in activity.trackpoints))
        inventory[year]['with_recorded_lap_hr'] += int(any(l.average_hr_bpm is not None for l in activity.laps))
        # The old workload entry can end before the last recorded point or lap
        # total. Availability-at-completion must use the entire observed extent.
        simulated_available = max(entry.end,
            activity.start_time_utc+timedelta(seconds=activity.lap_recorded_time_s or 0),
            *(p.timestamp_utc for p in activity.trackpoints if p.timestamp_utc is not None))
        history.append(evidence_from_activity(activity, config, key=entry.key, available_at=simulated_available,
            workout_type=roles[entry.key], profile=profile, capacity_eligible=eligible[entry.key]))
    load_sessions(database, config, since or datetime.min.replace(tzinfo=timezone.utc),
        datetime(2026, 9, 25, tzinfo=timezone.utc), session_types=roles, activity_observer=collect,
        missing_timing_observer=missing_timing)
    dates = {'2026-09-18', '2026-09-20', '2026-09-22', '2026-09-24'}
    cases = []
    for actual in history:
        day = actual.start.date().isoformat()
        if day not in dates:
            continue
        state = snapshot(history, actual.start-timedelta(seconds=1), actual.accounting_key)
        phases, refs, problem = assumptions(state, day, config, reference_method=reference_method)
        case = dict(day=day, activity_key=actual.key, state=exposure_summary(state),
            references=refs, excluded=state.excluded, forecast=None, comparison=None,
            unsupported_reason=problem)
        # Seed assumptions displayed, not inferred from a quiet early history.
        seed = LoadSeed(history[0].start, Quantity.point(0, 'trial_initial_zero'),
                        Quantity.point(0, 'trial_initial_zero'), 'Explicit trial zero, not established inactivity',
                        state.accounting_key)
        case['conditional_load_history'] = (dict(status='unprojectable_records_in_history',atl=None,ctl=None,ratio=None)
            if any(s['start'] < state.as_of for s in summary_only) else
            project_load_history(state, seed=seed, complete_recording_since=seed.at))
        if phases is not None:
            try:
                issued = issue_forecast(state, phases, initial_reserve=Quantity(.3, .6),
                                        initial_source='declared_pre_run_HRR_sensitivity_range')
            except ValueError as exc:
                if 'exhaust total distance' not in str(exc):
                    raise
                case['unsupported_reason'] = str(exc)
            else:
                case['forecast'] = issued
                case['comparison'] = compare_completion(issued, actual)
                # Removing all later sessions must leave the forecast unchanged.
                prior_only = snapshot(state.sessions, state.as_of, state.accounting_key)
                control = issue_forecast(prior_only, phases, initial_reserve=Quantity(.3, .6),
                                         initial_source='declared_pre_run_HRR_sensitivity_range')
                if control != issued:
                    raise AssertionError('Future evidence changed the forecast')
                case['future_removal_control_passed'] = True
                if day == '2026-09-18' and reference_method == 'generalized':
                    work_hr = next(p.equilibrium_reserve for p in phases if p.name.startswith('work_'))
                    novel_ref = phase_pace_reference(state, 'work', equilibrium_reserve=work_hr, meters=500)
                    novel_phases = tuple(replace(p, seconds=None, meters=500,
                        speed_mps=novel_ref['speed_mps']) if p.name.startswith('work_') else p for p in phases)
                    # Deliberately remove ALL prior quality pace, not just an exact format.
                    easy_only = snapshot([replace(s, pace_samples=tuple(p for p in s.pace_samples
                        if p.phase == 'easy')) for s in state.sessions], state.as_of, state.accounting_key)
                    first_phases, first_refs, problem = assumptions(easy_only, day, config)
                    if problem:
                        raise AssertionError(problem)
                    case['novel_workout_controls'] = dict(
                        three_by_500=dict(reference=novel_ref, forecast=issue_forecast(state,
                            novel_phases, initial_reserve=Quantity(.3, .6), initial_source='same_declared_sandbox_assumption')),
                        three_by_five_without_any_quality_pace=dict(references=first_refs,
                            forecast=issue_forecast(easy_only, first_phases, initial_reserve=Quantity(.3, .6),
                                initial_source='same_declared_sandbox_assumption')),
                        interpretation='Forecast-only controls; these are not additional completed workouts or accuracy tests')
        cases.append(case)
    if {c['day'] for c in cases} != dates:
        raise ValueError('Missing held-out cases')
    return dict(cases=cases, original_session_count=len(history)+len(summary_only),
        projectable_session_count=len(history),summary_only=summary_only,
        original_inventory_by_year=dict(inventory),
        since=since, history_scope='all_available_before_holdout_end' if since is None else 'explicit_cutoff',
        reference_method=reference_method,
        elapsed_seconds=perf_counter()-started, planner_influence=False,
        historical_availability_reconstructed=False,
        limitations=['Current settings and resolved roles held fixed; no claim they were available historically',
            'Pace support uses raw historical phase pace, not weather-adjusted capacity',
            'No historically verified prescription: declared sandbox intent, not compliance evaluation',
            'HR equilibrium ranges and 20/45/90-second kinetics are trial assumptions',
            'Scenario envelope is not a calibrated prediction interval or adaptation threshold'])


def json_ready(value):
    if isinstance(value, datetime):
        return value.isoformat()
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(type(value).__name__)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--database', required=True, type=Path)
    parser.add_argument('--config', type=Path, default=Path('config.yaml'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference-method', choices=('generalized', 'legacy_control'), default='generalized')
    parser.add_argument('--since', type=datetime.fromisoformat,
                        help='Optional explicit lower date bound; default includes every available year')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output path')
    since = args.since
    if since is not None:
        since = since.replace(tzinfo=timezone.utc) if since.tzinfo is None else since.astimezone(timezone.utc)
    result = audit(args.database, load_config(args.config), reference_method=args.reference_method, since=since)
    with args.output.open('x') as stream:
        json.dump(result, stream, default=json_ready, indent=2, allow_nan=False)
        stream.write('\n')
    print('sessions', result['original_session_count'], 'seconds', round(result['elapsed_seconds'], 3))
    for case in result['cases']:
        comparison = case['comparison']
        summary = None if comparison is None else {
            k: comparison[k] for k in ('cardio_status', 'central_cardio_load', 'actual_cardio',
                                      'central_elapsed_seconds', 'actual_elapsed_seconds')}
        if summary is not None:
            summary['raw_aerobic_speed_change'] = comparison['aerobic_performance']['speed_change_fraction']
        print(case['day'], case['unsupported_reason'] or summary,
              'pace sources', {k: v['source_count'] for k, v in case['references'].items()})


if __name__ == '__main__':
    main()
