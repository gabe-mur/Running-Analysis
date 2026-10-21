"""A3 read-only scope audit. No tuning, scheduler search, or invented intentions."""
from argparse import ArgumentParser
from collections import Counter
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from math import log
from pathlib import Path
import sqlite3
from statistics import mean, median
from time import perf_counter

from run_analysis.config import load_config
from run_analysis.continuous_training_dose import ContinuousDoseReference
from run_analysis.movement import classify_movement
from run_analysis.training_signals import HeartRateReference, observed_cardio_trace
from run_analysis.work_stimulus_evidence import observed_work_signals
from run_analysis.workload_accounting import Quantity
from scripts.audit_issued_quality_load import issue_boundary, parse_issued_structure
from scripts.audit_program_evidence_adapter import load_history
from scripts.audit_work_stimulus import encode
from scripts.audit_workload_expectation import app_session_types
from scripts.continuous_program_adapter import with_continuous_trace
from scripts.direct_quality_load import PhaseHistory, forecast_direct, phase_observations, score_direct
from scripts.program_evidence_adapter import QualityReference
from scripts.validate_quality_load import phase_layout


VERSION = 'a3_forecast_scope_v1'
AEROBIC = {'easy', 'long'}


def prior_records(records, as_of, roles, *, availability):
    """One frozen 30-day window; no future labels/measurements as numeric inputs."""
    return [r for r in records if r['role'] in roles and r['eligible']
            and as_of-timedelta(days=30) <= r['start'] < r['end'] < as_of
            and (r['end'] if availability == 'completion_assumed' else r['available_at']) <= as_of]


def baseline(records):
    rates = [r['observed_load']/r['known_seconds'] for r in records
             if r['known_seconds'] > 0 and r['observed_load'] is not None]
    paces = [r['elapsed_seconds']/r['miles'] for r in records if r['miles'] > 0]
    return dict(rate=mean(rates) if rates else None,
                seconds_per_mile=median(paces) if paces else None,
                source_keys=[r['key'] for r in records],
                rate_source_keys=[r['key'] for r in records if r['known_seconds'] > 0],
                complete_hr_sources=sum(r['complete'] for r in records),
                basis='equal-session known-HR rate; median elapsed pace; not calibrated')


def load_error(predicted, bounds, *, complete):
    """Never score partial HR as a point target or silently midpoint its bounds."""
    if predicted is None or bounds is None:
        return None
    lo, hi = bounds['low'], bounds['high']
    return dict(predicted=predicted, observed_low=lo, observed_high=hi,
                relative_bounds=[predicted/hi-1, predicted/lo-1] if lo > 0 else None,
                log_error=log(predicted/lo) if complete and predicted > 0 and lo > 0 else None)


def paired_summary(cases, names):
    paired = [c for c in cases if all(c.get(n) and c[n]['log_error'] is not None for n in names)]
    return dict(inventory=len(cases), paired_complete_hr=len(paired),
                paired_keys=[c['key'] for c in paired],
                models={n: dict(mean_absolute_log_error=mean(abs(c[n]['log_error']) for c in paired),
                                mean_signed_log_error=mean(c[n]['log_error'] for c in paired),
                                max_absolute_log_error=max(abs(c[n]['log_error']) for c in paired))
                        for n in names} if paired else {})


def support(prediction):
    """Expose extrapolation; not an accuracy gate or a confidence estimate."""
    rows = []
    for p in prediction['phases']:
        rows.append(dict(kind=p['kind'], seconds=p['end']-p['start'],
                         contributing_sessions=len(p['source_sessions']),
                         pooled_nonwork=p['fallback_pooled_nonwork']))
    return rows


def conditional_case(record, records, phase_history, accounting_key):
    # Realized duration/layout is deliberately explicit here, never pre-run accuracy.
    donors = prior_records(records, record['start'], {record['role']}, availability='completion_assumed')
    pooled = prior_records(records, record['start'], AEROBIC if record['role'] in AEROBIC else {'quality'},
                           availability='completion_assumed')
    phases = [replace(h, available_at=h.end) for h in phase_history]
    forecast = forecast_direct(record['layout'], phases, as_of=record['start'], accounting_key=accounting_key)
    own, broad = baseline(donors), baseline(pooled)
    case = dict(key=record['key'], role=record['role'], label=record['label'],
                status=forecast['status'], complete=record['complete'], coverage=record['coverage'],
                forecast_sources=forecast['source_keys'], phase_support=support(forecast),
                same_role_baseline=own, pooled_baseline=broad,
                basis='actual layout/distance; completion availability assumed; current labels and settings')
    for name, pred in [('direct', forecast['predicted_load']),
                       ('same_role_rate', None if own['rate'] is None else own['rate']*record['elapsed_seconds']),
                       ('pooled_rate', None if broad['rate'] is None else broad['rate']*record['elapsed_seconds'])]:
        case[name] = load_error(pred, record['bounds'], complete=record['complete'])
    case['known_hr_direct_diagnostic'] = score_direct(forecast, record['spans'])
    if record['role'] == 'quality':
        quality_keys = {r['key'] for r in records if r['role']=='quality'}
        quality_only = forecast_direct(record['layout'],[h for h in phases if h.key in quality_keys],
                                       as_of=record['start'],accounting_key=accounting_key)
        case['quality_only_direct'] = load_error(quality_only['predicted_load'],record['bounds'],complete=record['complete'])
        case['known_hr_quality_only_diagnostic'] = score_direct(quality_only,record['spans'])
    for name, ref in [('same_role_duration', own), ('pooled_duration', broad)]:
        seconds = ref['seconds_per_mile']*record['miles'] if ref['seconds_per_mile'] is not None else None
        case[name] = dict(predicted_seconds=seconds, actual_seconds=record['elapsed_seconds'],
                          relative_error=seconds/record['elapsed_seconds']-1 if seconds is not None else None,
                          basis='conditional on actual distance; no issued distance presumed')
    # Per-phase raw source ranges make duration/spacing extrapolation visible.
    accepted = [h for h in phases if h.key in forecast['source_keys']]
    for p, detail in zip(forecast['phases'], case['phase_support']):
        obs = [o for h in accepted for o in h.phases if o.load is not None and
               (o.kind == p['kind'] if not p['fallback_pooled_nonwork'] else o.kind != 'work')]
        durations = [o.seconds for o in obs]
        prev = [o.previous_seconds for o in obs if o.previous_seconds is not None]
        detail['outside_duration_support'] = bool(durations and not min(durations) <= detail['seconds'] <= max(durations))
        detail['outside_spacing_support'] = bool(prev and p['previous_seconds'] is not None and
                                                 not min(prev) <= p['previous_seconds'] <= max(prev))
    return case


def issued_case(match, records, phase_history, accounting_key):
    actual = next((r for r in records if r['key'] == str(match['activity_id'])), None)
    result = dict(key=str(match['activity_id']), plan_id=match['id'], assigned_role=match['workout_type'],
                  source_payload_sha256=sha256(match['recommendation_json'].encode()).hexdigest())
    if actual is None:
        return dict(result, status='missing_actual_evidence')
    payload = json.loads(match['recommendation_json'])
    try:
        issued = issue_boundary(match['schedule_generated_at'], payload, actual['start'])
    except (ValueError, KeyError) as exc:
        return dict(result, status='invalid_pre_run_provenance', reason=str(exc))
    role = 'quality' if match['workout_type'] in {'intervals', 'tempo_threshold'} else (
        'long' if match['workout_type'] == 'long' else 'easy')
    donors = prior_records(records, issued, {role}, availability='recorded')
    broad = prior_records(records, issued, AEROBIC if role in AEROBIC else {'quality'}, availability='recorded')
    own, pooled = baseline(donors), baseline(broad)
    result.update(issued_at=issued, actual_role=actual['role'], actual_label=actual['label'],
                  role_matches=role == actual['role'], same_role_baseline=own, pooled_baseline=pooled,
                  retrospective_match_not_verified_viewed=True, response_allowance=None,
                  uncertainty_calibrated=False)
    layouts = {}
    if role == 'quality':
        parsed = parse_issued_structure(payload)
        result['contract_status'] = parsed['status']
        if parsed['layout'] is not None:
            layouts['issued_timing'] = parsed['layout']
    else:
        distances = payload.get('distance_range_miles')
        durations = payload.get('duration_range_minutes')
        result.update(distance_band=distances, duration_band=durations)
        if durations and all(v is not None and v > 0 for v in durations):
            layouts = {edge: (('warmup', 0., minutes*60),) for edge, minutes in zip(('low','high'),durations)}
            result['contract_status'] = 'saved_duration_edge_scenarios_not_committed_target'
        elif distances and own['seconds_per_mile'] is not None:
            layouts = {edge: (('warmup', 0., miles*own['seconds_per_mile']),)
                       for edge, miles in zip(('low','high'),distances)}
            result['contract_status'] = 'saved_distance_edges_prior_only_median_pace_scenarios'
        else:
            result['contract_status'] = 'unresolved_extent_or_prior_duration'
    result['scenarios'] = {}
    for name, layout in layouts.items():
        prediction = forecast_direct(layout, phase_history, as_of=issued, accounting_key=accounting_key)
        seconds = layout[-1][2]
        result['scenarios'][name] = dict(status=prediction['status'], seconds=seconds,
            forecast_sources=prediction['source_keys'], phase_support=support(prediction),
            direct=load_error(prediction['predicted_load'],actual['bounds'],complete=actual['complete']),
            same_role_rate=load_error(None if own['rate'] is None else own['rate']*seconds,
                                     actual['bounds'],complete=actual['complete']),
            pooled_rate=load_error(None if pooled['rate'] is None else pooled['rate']*seconds,
                                  actual['bounds'],complete=actual['complete']))
    result['actual'] = {k:actual[k] for k in ('elapsed_seconds','miles','bounds','coverage','complete')}
    result['status'] = 'pre_run_inputs_reconstructed_not_accuracy_validated' if layouts else 'unresolved_forecast'
    return result


def audit(database, config, as_of):
    started = perf_counter()
    digest = sha256(database.read_bytes()).hexdigest()
    roles = app_session_types(database)
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        inventory = {str(r['id']):dict(r) for r in db.execute('''SELECT a.id,a.start_time_utc,
            a.created_at_utc,s.imported_at_utc,o.illness,o.health_tag,o.include_in_model
            FROM activities a LEFT JOIN activity_sources l ON l.activity_id=a.id AND l.is_primary=1
            LEFT JOIN source_files s ON s.id=l.source_file_id
            LEFT JOIN run_overrides o ON o.activity_id=a.activity_id WHERE a.sport='Running' ''')
            if r['start_time_utc'] and datetime.fromisoformat(r['start_time_utc']) < as_of}
        matches = [dict(r) for r in db.execute('''SELECT p.*,m.activity_id FROM activity_plan_matches m
            JOIN planned_workout_history p ON p.id=m.planned_workout_id ORDER BY p.schedule_generated_at,p.id''')
            if str(r['activity_id']) in inventory]
    ref = ContinuousDoseReference('A3 unchanged accounting read-only validation')
    hr = HeartRateReference(config['resting_hr'],config['max_hr'],'current settings; historical settings not reconstructed')
    records, phase_history = [], []
    def collect(run, activity):
        intervals = classify_movement(activity.trackpoints,config['moving_time']).intervals
        trace = observed_cardio_trace(activity,intervals,hr,ref.policy,include_spans=True,compute_density=False)
        meta = inventory[run.key]
        eligible = not meta['illness'] and meta['health_tag'] in (None,'','normal') and meta['include_in_model'] != 0
        end = run.start+timedelta(seconds=trace['elapsed_seconds'])
        available = max(end, *(datetime.fromisoformat(meta[k]) for k in ('created_at_utc','imported_at_utc') if meta[k]))
        if run.role == 'quality':
            continuous, _ = with_continuous_trace(run,trace['normalized_spans'],reference=ref,trace_key=run.key,
                trace_source='observed',elapsed_seconds=trace['elapsed_seconds'],
                outside_reference_seconds=trace['exclusions']['outside_reference_seconds'])
            signals = observed_work_signals(continuous,activity,config,intervals,trace['normalized_spans']) if continuous.bouts else None
            layout = phase_layout(signals.windows,trace['elapsed_seconds']) if signals else ()
        elif run.role in AEROBIC:
            layout = (('warmup',0.,trace['elapsed_seconds']),)  # Exact harness proxy, not true phase intent.
        else:
            layout = ()
        records.append(dict(key=run.key,start=run.start,end=end,available_at=available,
            eligible=eligible and trace['load_bounds_within_reference'] is not None,
            role=run.role,label=roles[run.key],miles=run.miles,elapsed_seconds=trace['elapsed_seconds'],
            known_seconds=trace['covered_seconds'],observed_load=trace['observed_load'],
            complete=trace['complete'],coverage=trace['coverage'],bounds=trace['load_bounds_within_reference'],
            spans=trace['normalized_spans'],layout=layout))
        if eligible and trace['load_bounds_within_reference'] is not None:
            phase_history.append(PhaseHistory(run.key,run.start,end,available,ref.accounting_key,
                                             phase_observations(layout,trace['normalized_spans'])))
        return run
    old = QualityReference('threshold','work_seconds',Quantity.point(900,'unused comparator'),Quantity(290,310),
                           'original-file adapter only')
    _, missing = load_history(database,config,as_of,old,
        since=min(datetime.fromisoformat(r['start_time_utc']) for r in inventory.values()),run_transform=collect)
    conditional = [conditional_case(r,records,phase_history,ref.accounting_key) for r in records if r['role'] != 'unknown']
    issued = [issued_case(m,records,phase_history,ref.accounting_key) for m in matches]
    groups = {label: paired_summary([c for c in conditional if c['label']==label],
                                   ('direct','same_role_rate','pooled_rate'))
              for label in sorted({c['label'] for c in conditional})}
    quality_ids = {k for k in inventory if roles.get(k) in {'intervals','tempo_threshold'}}
    represented = {c['key'] for c in conditional if c['role']=='quality'} | {r['key'] for r in missing}
    if quality_ids-represented:
        raise AssertionError(f'Quality inventory omitted: {quality_ids-represented}')
    if sha256(database.read_bytes()).hexdigest() != digest:
        raise AssertionError('Source DB changed during read-only audit')
    return dict(version=VERSION,as_of=as_of,database_sha256=digest,database_unchanged=True,
        settings=dict(resting_hr=config['resting_hr'],maximum_hr=config['max_hr']),
        inventory=len(inventory),loaded=len(records),missing=missing,
        inventory_without_loaded_record=sorted(set(inventory)-{r['key'] for r in records}),
        quality_inventory=len(quality_ids),conditional=conditional,conditional_groups=groups,
        saved_prescriptions=issued,issued_statuses=dict(Counter(r['status'] for r in issued)),
        elapsed_seconds=perf_counter()-started,planner_influence=False,
        limitations=['Repeatedly inspected development data, not independent prospective validation',
            'Conditional diagnostics know completed extent/layout and assume completion availability',
            'Saved-plan cohort uses recorded import/source availability; current labels and HR configuration remain retrospective',
            'No requested intensity, route, weather or readiness conditioning in the phase-rate API',
            'Whole aerobic sessions share warmup phase with quality warmups, matching the harness proxy',
            'No confidence calibration, response-allowance fit or scheduler stability claim'])


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument('--database',type=Path,default=Path('data/run_analysis.sqlite'))
    p.add_argument('--config',type=Path,default=Path('config.yaml'))
    p.add_argument('--as-of',type=datetime.fromisoformat,default=datetime(2026,9,27,tzinfo=timezone.utc))
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('Choose a new artifact path')
    result = audit(args.database,load_config(args.config),args.as_of)
    result['source_sha256'] = {str(path):sha256(path.read_bytes()).hexdigest() for path in map(Path,
        ('scripts/audit_forecast_scope.py','scripts/direct_quality_load.py','scripts/execution_forecast_contract.py',
         'scripts/new_model_daily_harness.py','scripts/validate_quality_load.py'))}
    with args.output.open('x') as stream:
        json.dump(result,stream,default=encode,indent=2,allow_nan=False)
    print(json.dumps({k:result[k] for k in ('inventory','loaded','quality_inventory','conditional_groups',
                                          'issued_statuses','elapsed_seconds')},indent=2))


if __name__ == '__main__':
    main()
