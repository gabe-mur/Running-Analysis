"""Read-only original history plus fixed hypothetical funded programs, not forecasts."""
from argparse import ArgumentParser
from collections import Counter
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
import json
import sqlite3
from time import perf_counter
from zoneinfo import ZoneInfo

from scripts.audit_program_delivery import ledger, regular, ORIGIN
from scripts.audit_workload_contract import load_sessions
from scripts.audit_workload_expectation import app_session_types
from run_analysis.config import load_config
from run_analysis.delivered_training import continuous_phase_from_activity, dose_from_profile, dose_metrics
from run_analysis.projection_evidence import evidence_from_activity
from run_analysis.recency_demand import CHANNELS, DemandEvent, DemandTerm, compare_recency
from run_analysis.workload_accounting import Quantity


def coarse_role(role):
    return ('quality' if role in {'intervals', 'tempo_threshold'} else
            'long' if role == 'long' else
            'easy' if role in {'easy', 'recovery', 'medium_long'} else 'unknown')


def scenario_rates(rows, cutoff):
    """Empirical extrema are declared transfer scenarios, not prediction intervals."""
    result = {}
    for role in ('easy', 'long', 'quality'):
        chosen = [r for r in rows if r['eligible'] and r['start'] >= cutoff and r['role'] == role]
        channels = {}
        for channel in ('running_minutes', 'cardio'):
            values = [(r['key'], r['quantities'][channel].scale(1/r['miles']))
                      for r in chosen if r['miles'] > 0 and r['quantities'][channel] is not None]
            channels[channel] = dict(keys=[key for key, _ in values],
                bounds=(Quantity(min(q.low for _, q in values), max(q.high for _, q in values),
                    basis='recent_role_per_mile_extrema_as_shared_scenario_not_forecast') if values else None))
        result[role] = channels
    return result


def future_events(motif, rates, accounting_key, *, horizon_days=28):
    unit = Quantity.point(1., 'declared_prescription_unit')
    result = []
    for p in ledger(motif):
        if p.source != 'planned' or not ORIGIN <= p.day < ORIGIN+timedelta(days=horizon_days):
            continue
        terms = [DemandTerm('miles', 'prescription_mile', p.miles, unit),
                 DemandTerm('work_minutes', 'prescription_work_minute', sum(p.work_bout_seconds or ())/60, unit)]
        terms += [DemandTerm(c, f'scenario:{p.role}:{c}', p.miles, rates[p.role][c]['bounds'])
                  for c in ('running_minutes', 'cardio')]
        result.append(DemandEvent(p.key, p.day, tuple(terms), accounting_key,
            p.role if p.role != 'quality' else p.quality_purpose, p.work_bout_seconds, 'planned'))
    return result


def at_opportunity(history, future, target):
    return [*history, *(e for e in future if e.day < target.day)]


def load_evidence(database, config, as_of):
    """Read original completed sessions without constructing a future calendar."""
    if as_of.tzinfo is None:
        raise ValueError('Aware evidence cutoff required')
    roles, rows, unresolved = app_session_types(database), [], []
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro', uri=True) as db:
        eligible = {str(key): not illness and health in (None, '', 'normal') and included != 0
            for key, illness, health, included in db.execute('''
                SELECT a.id,o.illness,o.health_tag,o.include_in_model FROM activities a
                LEFT JOIN run_overrides o ON o.activity_id=a.activity_id''')}

    def collect(activity, entry, profile):
        end = max(entry.end, activity.start_time_utc+timedelta(seconds=activity.lap_recorded_time_s or 0),
            *(p.timestamp_utc for p in activity.trackpoints if p.timestamp_utc is not None))
        if end > as_of:
            unresolved.append(dict(key=entry.key, reason='not_completed'))
            return
        evidence = evidence_from_activity(activity, config, key=entry.key, available_at=end,
            workout_type=roles[entry.key], profile=profile, capacity_eligible=eligible[entry.key])
        dose = dose_from_profile(profile, key=entry.key, start=entry.start, end=end, available_at=end,
            continuous_phase=continuous_phase_from_activity(activity, config) if roles[entry.key] == 'tempo_threshold' else None)
        measured = dose_metrics(dose)
        minutes = measured['moving_seconds'].scale(1/60) if measured['moving_seconds'] else None
        # Failure to detect structured work is not proof of zero work.
        work = (measured['work_seconds'].scale(1/60) if measured['work_seconds'] else
                Quantity(0, (end-entry.start).total_seconds()/60, basis='unknown_work_bounded_by_elapsed_time'))
        quantities = dict(miles=measured['distance_meters'].scale(1/1609.344),
            running_minutes=minutes, cardio=evidence.cardio, work_minutes=work)
        rows.append(dict(key=entry.key, start=entry.start, end=end,
            day=entry.start.astimezone(ZoneInfo(config['timezone_default'])).date(),
            role=coarse_role(roles[entry.key]), resolved_role=roles[entry.key], eligible=eligible[entry.key],
            miles=evidence.meters/1609.344, quantities=quantities, accounting_key=evidence.accounting_key,
            structure=tuple(b.work_seconds for b in dose.bouts) or None, structure_basis=dose.structure_basis))

    load_sessions(database, config, as_of-timedelta(days=84), as_of, session_types=roles,
        activity_observer=collect,
        missing_timing_observer=lambda a, key, path: unresolved.append(dict(key=key, reason='missing_elapsed_trace')))
    if not rows:
        raise ValueError('No recorded history')
    history = [DemandEvent('actual:'+r['key'], r['day'], tuple(
        DemandTerm(c, f"actual:{r['key']}:{c}", 1., r['quantities'][c]) for c in CHANNELS),
        r['accounting_key'], r['role'], r['structure'], 'observed') for r in rows]
    rates = scenario_rates(rows, as_of-timedelta(days=28))
    return rows, unresolved, history, rates


def audit(database, config, as_of):
    started = perf_counter()
    if as_of.tzinfo is None or as_of.astimezone(ZoneInfo(config['timezone_default'])).date() != ORIGIN:
        raise ValueError('This frozen scenario panel requires an aware September 25, 2026 cutoff')
    rows, unresolved, history, rates = load_evidence(database, config, as_of)
    signature = rows[0]['accounting_key']
    definitions = dict(hold=regular(), extend_easy=regular(easy=5.3),
        insert_support=regular()+[(5, .9, 'easy', .9, None, None)],
        supported_long=regular(long=8.8, easy=4.9),
        compressed=[(0, 8.8, 'long', 8.8, None, None), (1, 4.9, 'easy', 4.9, None, None),
                    (2, 4.5, 'quality', 2., 'threshold', (300., 300., 300.))])
    futures = {name: future_events(motif, rates, signature) for name, motif in definitions.items()}
    comparisons = []
    for left, right in [('insert_support', 'extend_easy'), ('compressed', 'supported_long'),
                         ('supported_long', 'extend_easy'), ('hold', 'extend_easy')]:
        for purpose in ('long', 'threshold'):
            a = [e for e in futures[left] if e.purpose == purpose]
            b = [e for e in futures[right] if e.purpose == purpose]
            for cycle, (x, y) in enumerate(zip(a, b), 1):
                r = compare_recency(at_opportunity(history, futures[left], x), x,
                                    at_opportunity(history, futures[right], y), y)
                curves = r.pop('cutoff_differences')
                r['first_definite_direction_cutoffs'] = {c: dict(
                    left_lower=next((v for v in values if v['left_minus_right_high'] is not None
                                     and v['left_minus_right_high'] < -1e-8), None),
                    left_higher=next((v for v in values if v['left_minus_right_low'] is not None
                                      and v['left_minus_right_low'] > 1e-8), None)) for c, values in curves.items()}
                comparisons.append(dict(left=left, right=right, purpose=purpose, cycle=cycle, **r))
    return dict(as_of=as_of, source='original_files_read_only_plus_declared_future_scenarios',
        sessions=rows, unresolved=unresolved, rates=rates, comparisons=comparisons,
        order_counts=dict(Counter(r['recency_demand_order'] for r in comparisons)),
        elapsed_seconds=perf_counter()-started, planner_influence=False,
        physiology_established=False, progression_accepted=False,
        limitations=['Recorded history is not independently asserted complete; missing records are not rest',
            'Current role/health metadata fixed; upload availability assumed at completion',
            'Role-specific per-mile extrema transfer is a scenario, not calibrated fatigue or load prediction',
            'Historical failed work detection remains unknown; future easy work=0 is declared intent',
            'Matched quality purpose and bouts do not establish identical physiological cost',
            'Calendar dates, not precise hours; frozen motifs are not searched or recommended schedules',
            'No muscle-damage measure, recovery threshold, optimality or daily-replanning stability claim'])


def json_ready(value):
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(type(value).__name__)


if __name__ == '__main__':
    p = ArgumentParser(description=__doc__)
    p.add_argument('--database', type=Path, required=True)
    p.add_argument('--config', type=Path, default=Path('config.yaml'))
    p.add_argument('--as-of', type=datetime.fromisoformat, required=True)
    args = p.parse_args()
    print(json.dumps(audit(args.database, load_config(args.config), args.as_of),
        default=json_ready, allow_nan=False, indent=2))
