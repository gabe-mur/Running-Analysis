"""Read-only chronological conditional speed-to-HR audit on original files."""
from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timedelta, timezone
import json
from math import sqrt
from pathlib import Path
import sqlite3
from statistics import median
from time import perf_counter

from scripts.audit_athlete_projection import assumptions, json_ready
from scripts.audit_workload_contract import load_sessions
from scripts.audit_workload_expectation import app_session_types
from run_analysis.athlete_projection import ForecastPhase, forecast_trace, snapshot
from run_analysis.config import load_config
from run_analysis.hr_response import (
    ResponseEvidence, intervals_from_activity, sufficient_statistics, fit_response, predict_response,
    predict_speed_program,
)
from run_analysis.phase_calibration import observed_phases, clip_trace, mask_trace, trace_load
from run_analysis.projection_evidence import evidence_from_activity
from run_analysis.signal_structure import SignalSpan


def metrics(predicted, observed, bpm_span):
    # Both use the same valid time mask. Prediction edges may cross sample edges.
    error_square = error_sum = seconds = 0.
    index = 0
    for actual in observed:
        while index < len(predicted) and predicted[index].end <= actual.start:
            index += 1
        j = index
        while j < len(predicted) and predicted[j].start < actual.end:
            p = predicted[j]
            lo, hi = max(p.start, actual.start), min(p.end, actual.end)
            if hi > lo:
                a = clip_trace((actual,), lo, hi)[0]
                b = clip_trace((p,), lo, hi)[0]
                d0, d1 = b.first-a.first, b.last-a.last
                dt = hi-lo
                error_square += dt*(d0*d0+d0*d1+d1*d1)/3
                error_sum += dt*(d0+d1)/2
                seconds += dt
            j += 1
    if seconds <= 0:
        return None
    actual_load = trace_load(observed)
    valid = all(0 <= s.first <= 1 and 0 <= s.last <= 1 for s in predicted)
    load = trace_load(predicted) if valid else None
    return dict(seconds=seconds, hr_rmse_bpm=sqrt(error_square/seconds)*bpm_span,
        hr_bias_bpm=error_sum/seconds*bpm_span, observed_load=actual_load,
        predicted_load=load, load_fraction_error=load/actual_load-1 if load is not None and actual_load > 0 else None)


def summarize(rows):
    result = {}
    for control in ('static', 'zone_fixed_initial', 'zone_observed_initial'):
        paired = [r for r in rows if r.get(control) and r.get('dynamic')]
        load_pairs = [r for r in paired if all(r[k]['load_fraction_error'] is not None for k in (control, 'dynamic'))]
        result[control] = dict(count=len(paired), load_count=len(load_pairs),
            control_median_hr_rmse=median(r[control]['hr_rmse_bpm'] for r in paired) if paired else None,
            dynamic_median_hr_rmse=median(r['dynamic']['hr_rmse_bpm'] for r in paired) if paired else None,
            hr_improved=sum(r['dynamic']['hr_rmse_bpm'] < r[control]['hr_rmse_bpm'] for r in paired),
            control_median_absolute_load_error=median(abs(r[control]['load_fraction_error']) for r in load_pairs) if load_pairs else None,
            dynamic_median_absolute_load_error=median(abs(r['dynamic']['load_fraction_error']) for r in load_pairs) if load_pairs else None)
    return result


def audit(database, config):
    started = perf_counter()
    roles = app_session_types(database)
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro', uri=True) as db:
        eligible = {str(k): not ill and health in (None, '', 'normal') and inc != 0
            for k, ill, health, inc in db.execute('''SELECT a.id,o.illness,o.health_tag,o.include_in_model
                FROM activities a LEFT JOIN run_overrides o ON o.activity_id=a.activity_id''')}
    history, records, missing, pace_history, cases = [], [], [], [], []
    inventory = Counter()
    fit_seconds = 0.
    def collect(activity, entry, profile):
        nonlocal fit_seconds
        inventory['trace_sessions'] += 1
        available = max(entry.end, activity.start_time_utc+timedelta(seconds=activity.lap_recorded_time_s or 0),
            *(p.timestamp_utc for p in activity.trackpoints if p.timestamp_utc is not None))
        actual = evidence_from_activity(activity, config, key=entry.key, available_at=available,
            workout_type=roles[entry.key], profile=profile, capacity_eligible=eligible[entry.key])
        issued_at = actual.start-timedelta(microseconds=1)
        day = str(actual.start.date())
        if day in {'2026-09-18', '2026-09-20', '2026-09-22', '2026-09-24'}:
            state = snapshot(pace_history, issued_at, actual.accounting_key)
            prescription, _, problem = assumptions(state, day, config)
            fitted_case = fit_response(history, as_of=issued_at, accounting_key=actual.accounting_key)
            case = dict(day=day, problem=problem, fit=fitted_case, actual_load=actual.cardio,
                actual_seconds=(actual.end-actual.start).total_seconds())
            if prescription and fitted_case['dynamic']:
                trace, doses, duration = forecast_trace(prescription, initial_reserve=.45, tau_seconds=45.)
                projected, diagnostics = predict_speed_program(
                    [(p['end_seconds']-p['start_seconds'], p['speed_mps']) for p in doses],
                    fitted_case['dynamic'], initial_reserve=.45)
                case.update(baseline_load=trace_load(trace), dynamic_load=None if diagnostics['hr_outside_reference'] else trace_load(projected),
                    forecast_seconds=duration, forecast_meters=sum(p['meters'] for p in doses),
                    diagnostics=diagnostics, doses=doses)
            cases.append(case)
        pace_history.append(actual)
        rows = intervals_from_activity(activity, config)
        inventory['sessions_with_joint_hr_speed'] += bool(rows)
        if not rows or not actual.capacity_eligible:
            return
        tick = perf_counter()
        fitted = fit_response(history, as_of=actual.start-timedelta(microseconds=1), accounting_key=actual.accounting_key)
        fit_seconds += perf_counter()-tick
        observations = tuple(SignalSpan(r.start, r.end, r.first, r.last) for r in rows)
        phases = observed_phases(activity, actual, config, profile)
        traces = {}
        details = {}
        for kind in ('dynamic', 'static'):
            if fitted[kind]:
                traces[kind], details[kind] = predict_response(rows, fitted[kind])
        if phases:
            prescribed = [ForecastPhase(p['phase'], p['end']-p['start'], None, None,
                p['equilibrium_reserve'], p['origin']) for p in phases]
            for name, initial in (('zone_fixed_initial', .45), ('zone_observed_initial', rows[0].first)):
                trace, _, _ = forecast_trace(prescribed, initial_reserve=initial, tau_seconds=45.)
                traces[name] = mask_trace(trace, observations)
        scale = config['max_hr']-config['resting_hr']
        records.append(dict(key=actual.key, day=str(actual.start.date()), role=actual.role,
            fit=fitted, diagnostics=details, elapsed_seconds=(actual.end-actual.start).total_seconds(),
            covered_seconds=sum(r.end-r.start for r in rows),
            **{k: metrics(v, observations, scale) for k, v in traces.items()},
            phases=[dict(phase=p['phase'], start=p['start'], end=p['end'],
                **{k: metrics(clip_trace(v, p['start'], p['end']),
                    clip_trace(observations, p['start'], p['end']), scale) for k, v in traces.items()}) for p in phases]))
        # The held-out session enters history only after prediction/scoring.
        tick = perf_counter()
        history.append(ResponseEvidence(actual.key, actual.end, actual.available_at, actual.accounting_key,
            sufficient_statistics(rows), (min(r.speed for r in rows), max(r.speed for r in rows))))
        fit_seconds += perf_counter()-tick
    load_sessions(database, config, datetime.min.replace(tzinfo=timezone.utc),
        datetime(2026, 9, 25, tzinfo=timezone.utc), session_types=roles, activity_observer=collect,
        missing_timing_observer=lambda activity, key, path: missing.append(key))
    return dict(version='conditional_speed_hr_response_v2', records=records, inventory=dict(inventory), cases=cases,
        summary_only_keys=missing, overall=summarize(records),
        by_year={year: summarize([r for r in records if r['day'].startswith(year)]) for year in sorted({r['day'][:4] for r in records})},
        by_role={role: summarize([r for r in records if r['role'] == role]) for role in sorted({r['role'] for r in records})},
        tau_search_boundary_count=sum(bool(r['fit']['dynamic'] and r['fit']['dynamic']['tau_at_search_boundary']) for r in records),
        fit_and_statistics_seconds=fit_seconds, elapsed_seconds=perf_counter()-started,
        planner_influence=False, prospective_forecast_validation=False,
        limitations=['Conditional on actual speed and HR at each valid-segment start; not a prospective pace/load forecast',
            'Four sandbox pre-run cases are separate: same declared prescriptions/prior paces and assumed initial HR, no actual trace input',
            '41-point log response-time grid 1–600 s is a numerical experiment domain, not physiology or safety limits',
            '84-day trailing fits; fixed current HR references, resolved workout roles and health metadata',
            'Availability at completion is assumed; original upload-time history is not reconstructed',
            'Raw speed, no weather/grade/drift/sensor-lag model; no confidence intervals',
            'Zone controls use observed phase layout and do not reinitialize across missing data',
            'Current records and earlier diagnostics already inspected; this is exploratory walk-forward validation'])


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--database', required=True, type=Path)
    parser.add_argument('--config', type=Path, default=Path('config.yaml'))
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output path')
    result = audit(args.database, load_config(args.config))
    with args.output.open('x') as f:
        json.dump(result, f, default=json_ready, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'records'}, default=json_ready, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
