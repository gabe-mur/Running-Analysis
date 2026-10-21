"""User-aligned intent replay, not a claim about old generated-plan compliance."""
from argparse import ArgumentParser
from dataclasses import asdict
from datetime import datetime, timezone
import json
import sqlite3
from pathlib import Path
from time import perf_counter

from scripts.audit_workload_contract import load_sessions
from run_analysis.config import load_config
from run_analysis.db import initialize
from run_analysis.run_feedback import list_runs
from run_analysis.workload_accounting import Quantity
from run_analysis.workload_expectation import (
    Intent, Reference, ForecastPolicy, forecast, compare_distribution, observed_features,
)


def aligned_intents(zones):
    def miles(low, high, target):
        return Quantity(low*1609.344, high*1609.344, target*1609.344,
                        'retrospective_aligned_sandbox_distance_band')
    return {
        '2026-09-18': Intent(miles(4.1, 4.5, 4.3), None,
            Quantity.point(900, '3x5_minutes_intent'), Quantity.point(300, '5_minute_effort')),
        '2026-09-20': Intent(miles(8.2, 8.6, 8.4), None, None, None, zones['z2'][1]+1),
        '2026-09-22': Intent(miles(4.5, 4.9, 4.7), None, None, None, zones['z2'][1]+1),
        '2026-09-24': Intent(miles(4.4, 4.8, 4.6), None, None, None,
            work_distance_meters=Quantity.point(2400, 'user_confirmed_6x400m'),
            longest_work_distance_meters=Quantity.point(400, 'user_confirmed_400m_efforts')),
    }


def tolerances():
    # Trial comparison margins, not injury limits or a production adherence policy.
    return dict(distance_meters=20., duration_minutes=5/60, cardio_points=.2,
        work_seconds=15., longest_work_seconds=5.,
        work_distance_meters=120., longest_work_distance_meters=20.,
        **{f'hr_{w}s': .5 for w in (30, 60, 120, 300, 600)},
        **{f'speed_{w}s': .02 for w in (30, 60, 120, 300, 600)})


from run_analysis.adaptive.evidence_history import app_session_types


def trial(database, config, since=datetime(2026, 1, 1, tzinfo=timezone.utc)):
    started = perf_counter()
    session_types = app_session_types(database)
    sessions = load_sessions(database, config,
        since, datetime(2026, 9, 25, tzinfo=timezone.utc), session_types=session_types)
    references = [Reference(e.key, e.end, p) for e, p in sessions]
    intents = aligned_intents(config['zones'])
    # Predeclared broad exploratory matching rules; retain sensitivity instead
    # of selecting whichever neighborhood makes the target appear compliant.
    policies = [ForecastPolicy(distance_ratio_limit=ratio, work_ratio_limit=work_ratio,
        minimum_sessions=3, minimum_hr_coverage=.98, hr_allowance_bpm=3., speed_allowance_mps=.1)
        for ratio, work_ratio in ((1.2, 1.5), (1.35, 2.), (1.5, 2.5))]
    cases = []
    for entry, profile in sessions:
        day = str(entry.start.date())
        if day not in intents:
            continue
        results, inferred_results = [], []
        for policy in policies:
            for include_inferred, destination in ((False, results), (True, inferred_results)):
                expected = forecast(intents[day], references, entry.start, policy,
                                    include_inferred_work=include_inferred)
                comparison = compare_distribution(expected, profile, tolerances())
                destination.append(dict(policy=asdict(policy), expectation=expected, comparison=comparison))
        cases.append(dict(start=entry.start, key=entry.key, intent=asdict(intents[day]),
                          results=results, inferred_results=inferred_results))
    if len(cases) != 4:
        raise ValueError('Expected exactly the four target sessions')
    return dict(cases=cases, elapsed_seconds=perf_counter()-started,
        reference_count=len(references), tolerances=tolerances(),
        reference_since=since,
        reference_inventory=[dict(key=e.key, start=e.start,
            session_workout_type=session_types[e.key],
            features=observed_features(p), bout_count=p['intensity_structure']['bout_count'],
            pace_inferred_work=p['intensity_structure'].get('pace_inferred_work'))
            for e, p in sessions],
        intent_provenance='User-confirmed 6x400; other sessions aligned retrospectively with chosen work. '
            'Distance bands are declared sandbox assumptions, not original issued prescriptions.',
        forecast_provenance='Only earlier completed sessions enter each expectation; target response is held out. '
            'Session eligibility uses current app-resolved classifications/overrides. '
            'Historical upload and classification availability are not reconstructed; '
            'this is not a prospective validation.',
        planner_influence=False)


def json_ready(value):
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Quantity):
        return asdict(value)
    raise TypeError(type(value).__name__)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--database', required=True, type=Path)
    parser.add_argument('--config', type=Path, default=Path('config.yaml'))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--since', default='2026-01-01', help='Reference start, inclusive UTC date')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output path')
    since = datetime.fromisoformat(args.since).replace(tzinfo=timezone.utc)
    result = trial(args.database, load_config(args.config), since)
    with args.output.open('x') as stream:
        json.dump(result, stream, default=json_ready, allow_nan=False, indent=2)
        stream.write('\n')
    print('seconds', round(result['elapsed_seconds'], 3), 'references', result['reference_count'])
    for case in result['cases']:
        print(case['start'].isoformat(), [dict(peers=len(x['expectation']['selected_reference_keys']),
            work_peers=len(x['expectation']['work_reference_keys']),
            differences=x['comparison']['differences']) for x in case['results']])
        print('with inferred candidates', [dict(peers=len(x['expectation']['selected_reference_keys']),
            work_peers=len(x['expectation']['work_reference_keys']),
            differences=x['comparison']['differences']) for x in case['inferred_results']])


if __name__ == '__main__':
    main()
