"""Frozen conditional quality-load experiment; no production changes or tuning.

Actual phase TIMING is a condition, not an issued prescription. Target speed and
HR never enter forecast construction. Scoring uses identical known-HR masks.
"""
from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from math import sqrt
from pathlib import Path
import sqlite3
from statistics import mean
from time import perf_counter

from run_analysis.config import load_config
from run_analysis.continuous_training_dose import ContinuousDoseReference
from run_analysis.hr_response import (ResponseEvidence, intervals_from_activity,
    sufficient_statistics, fit_response, predict_speed_program)
from run_analysis.movement import classify_movement
from run_analysis.performance_reference import geometric_mean
from run_analysis.phase_calibration import clip_trace, mask_trace
from run_analysis.signal_structure import SignalSpan
from run_analysis.training_signals import HeartRateReference, observed_cardio_trace, integrate_cardio_trace
from run_analysis.work_stimulus_evidence import observed_work_signals
from run_analysis.workload_accounting import Quantity
from scripts.audit_program_evidence_adapter import load_history
from scripts.audit_workload_expectation import app_session_types
from scripts.audit_work_stimulus import encode
from scripts.continuous_program_adapter import with_continuous_trace
from scripts.program_evidence_adapter import QualityReference
from scripts.quality_session_model import assess_quality_session, current_quality_pace, QualityHistoryPolicy


POLICY = QualityHistoryPolicy()
INITIAL_RESERVE = .45  # Existing offline forecast assumption, not target initial HR.


from run_analysis.phase_partitions import phase_layout


def load(spans, elapsed):
    return integrate_cardio_trace(spans, elapsed, compute_density=False)['observed_load']


def matched_metrics(predicted, actual, elapsed, bpm_span):
    """Unknown HR isn't zero. Invalid model HR is a failure, never clipped."""
    predicted = mask_trace(predicted, actual)
    actual = mask_trace(actual, predicted)
    seconds = sum(s.end-s.start for s in actual)
    if not seconds:
        return None
    error = square = 0.
    index = 0
    for a in actual:
        while index < len(predicted) and predicted[index].end <= a.start:
            index += 1
        j = index
        while j < len(predicted) and predicted[j].start < a.end:
            p = predicted[j]
            left, right = max(a.start, p.start), min(a.end, p.end)
            if right > left:
                aa, pp = clip_trace((a,), left, right)[0], clip_trace((p,), left, right)[0]
                d0, d1 = pp.first-aa.first, pp.last-aa.last
                error += (right-left)*(d0+d1)/2
                square += (right-left)*(d0*d0+d0*d1+d1*d1)/3
            j += 1
    observed = load(actual, elapsed)
    valid = all(0 <= s.first <= 1 and 0 <= s.last <= 1 for s in predicted)
    estimate = load(predicted, elapsed) if valid else None
    return dict(seconds=seconds, hr_rmse_bpm=sqrt(square/seconds)*bpm_span,
        hr_bias_bpm=error/seconds*bpm_span, observed_load=observed,
        predicted_load=estimate, signed_load_error=None if estimate is None else estimate-observed,
        fraction_error=estimate/observed-1 if estimate is not None and observed > 0 else None)


def forecast(layout, *, as_of, history, donors, response, accounting_key):
    """Forecast API deliberately has no held-out speed/HR input."""
    result = dict(status='unresolved_work_alignment')
    if not layout:
        return result
    refs = {b-a: current_quality_pace(history, as_of=as_of, bout_seconds=b-a,
        accounting_key=accounting_key, policy=POLICY) for kind,a,b in layout if kind == 'work'}
    result['pace_references'] = list(refs.values())
    if any(r.speed_mps is None for r in refs.values()):
        return dict(result, status='no_recent_quality_pace')
    keys = sorted({k for r in refs.values() for k,_,_ in r.session_estimates})
    selected = [donors[k] for k in keys]
    if any(d['end'] >= as_of or d['available_at'] > as_of or
           not as_of-timedelta(days=POLICY.current_days) <= d['start'] < as_of for d in selected):
        raise ValueError('Future, late or stale donor')
    nonwork = [d['nonwork_speed'] for d in selected if d['nonwork_speed'] is not None]
    rates = [d['load_rate'] for d in selected if d['load_rate'] is not None]
    result.update(source_keys=keys, rate_source_keys=[k for k in keys if donors[k]['load_rate'] is not None],
        baseline_rate=mean(rates) if rates else None,
        nonwork_source_keys=[k for k in keys if donors[k]['nonwork_speed'] is not None])
    if any(kind == 'nonwork' for kind,_,_ in layout) and not nonwork:
        return dict(result, status='no_recent_nonwork_speed')
    if response is None:
        return dict(result, status='no_supported_hr_response')
    # One vote per prior session. Include recorded standing time in nonwork pace;
    # no optimistic substitution with moving-only recovery speed.
    easy_speed = mean(nonwork) if nonwork else 0.
    flat_speed = geometric_mean([d['work_speed'] for d in selected])
    programs = {name: [(b-a, (refs[b-a].speed_mps.estimate if name == 'model' else flat_speed)
        if kind == 'work' else easy_speed) for kind,a,b in layout]
        for name in ('model', 'unconditioned_pace')}
    predictions = {name: predict_speed_program(p, response, initial_reserve=INITIAL_RESERVE)
                   for name,p in programs.items()}
    return dict(result, status='predicted', programs=programs, predictions=predictions)


def summarize(cases):
    paired = [c for c in cases if c.get('model') and c['model']['fraction_error'] is not None
              and c.get('baseline_fraction_error') is not None
              and not c['diagnostics']['model']['hr_outside_reference']]
    result = dict(total=len(cases), paired=len(paired), statuses=dict(Counter(c['status'] for c in cases)))
    for name in ('model', 'unconditioned_pace'):
        rows = [c[name] for c in paired if c[name]['fraction_error'] is not None
                and not c['diagnostics'][name]['hr_outside_reference']]
        result[name] = dict(n=len(rows), mean_absolute_fraction_error=mean(abs(r['fraction_error']) for r in rows) if rows else None,
            mean_signed_fraction_error=mean(r['fraction_error'] for r in rows) if rows else None,
            mean_hr_rmse_bpm=mean(r['hr_rmse_bpm'] for r in rows) if rows else None)
    result['baseline_mean_absolute_fraction_error'] = mean(abs(c['baseline_fraction_error']) for c in paired) if paired else None
    result['baseline_mean_signed_fraction_error'] = mean(c['baseline_fraction_error'] for c in paired) if paired else None
    result['model_wins'] = sum(abs(c['model']['fraction_error']) < abs(c['baseline_fraction_error']) for c in paired)
    result['phase_bias_load_units'] = {kind: mean(c['phases'][kind]['signed_load_error'] for c in paired
        if c['phases'].get(kind) and c['phases'][kind]['signed_load_error'] is not None)
        for kind in ('work', 'nonwork') if any(c['phases'].get(kind) for c in paired)}
    return result


def audit(database, config, as_of, *, case_observer=None, response_observer=None, activity_observer=None,
          source_observer=None):
    started = perf_counter()
    digest = sha256(database.read_bytes()).hexdigest()
    roles = app_session_types(database)
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro', uri=True) as db:
        inventory = [(str(k), datetime.fromisoformat(t)) for k,t in db.execute(
            "SELECT id,start_time_utc FROM activities WHERE sport='Running' AND start_time_utc IS NOT NULL")]
        eligible = {str(k): not ill and health in (None, '', 'normal') and inc != 0
            for k,ill,health,inc in db.execute('''SELECT a.id,o.illness,o.health_tag,o.include_in_model
                FROM activities a LEFT JOIN run_overrides o ON o.activity_id=a.activity_id''')}
        matches = {str(k): dict(generated_at=at, workout_type=role, subtype=subtype)
            for k,at,role,subtype in db.execute('''SELECT m.activity_id,p.schedule_generated_at,p.workout_type,p.quality_session_type
                FROM activity_plan_matches m JOIN planned_workout_history p ON p.id=m.planned_workout_id''')}
    expected = {k for k,t in inventory if t < as_of and roles.get(k) in {'intervals','tempo_threshold'}}
    reference = ContinuousDoseReference('unchanged quality v1 load validation')
    hr = HeartRateReference(config['resting_hr'], config['max_hr'], 'current settings; historical settings unavailable')
    assessments, response_history, donors, cases = [], [], {}, []
    def collect(row, activity):
        response_rows = intervals_from_activity(activity, config)
        if row.key in expected:
            intervals = classify_movement(activity.trackpoints, config['moving_time']).intervals
            trace = observed_cardio_trace(activity, intervals, hr, reference.policy, include_spans=True, compute_density=False)
            spans, elapsed = trace['normalized_spans'], trace['elapsed_seconds']
            run, _ = with_continuous_trace(row, spans, reference=reference, trace_key=row.key,
                trace_source='observed', elapsed_seconds=elapsed,
                outside_reference_seconds=trace['exclusions']['outside_reference_seconds'])
            signals = observed_work_signals(run, activity, config, intervals, spans) if run.bouts else None
            assessment = assess_quality_session(run, signals, reference=reference, as_of=run.available_at)
            layout = phase_layout(signals.windows, elapsed) if signals else ()
            fitted = fit_response(response_history, as_of=run.start,
                accounting_key=reference.accounting_key, lookback_days=POLICY.current_days)
            prediction = forecast(layout, as_of=run.start, history=assessments, donors=donors,
                response=fitted['dynamic'], accounting_key=reference.accounting_key)
            predictions = prediction.pop('predictions', {})
            case = dict(key=run.key, start=run.start, label=roles[run.key], layout=layout,
                actual_cardio=run.demand[1], hr_coverage=trace['coverage'], fit=fitted,
                saved_match=matches.get(run.key), **prediction)
            scale = config['max_hr']-config['resting_hr']
            case['diagnostics'] = {name: diag for name,(_,diag) in predictions.items()}
            for name,(pred,diag) in predictions.items():
                case[name] = matched_metrics(pred, spans, elapsed, scale)
            if predictions:
                pred = predictions['model'][0]
                case['phases'] = {kind: matched_metrics(
                    mask_trace(pred, tuple(SignalSpan(a,b,0,0) for k,a,b in layout if k == kind)),
                    spans, elapsed, scale) for kind in ('work','nonwork')}
                if case['diagnostics']['model']['hr_outside_reference']:
                    case['status'] = 'predicted_hr_outside_reference'
            known = sum(s.end-s.start for s in spans)
            actual_load = load(spans, elapsed)
            rate = prediction.get('baseline_rate')
            case['baseline_fraction_error'] = rate*known/actual_load-1 if rate is not None and actual_load > 0 else None
            case['known_hr_seconds'] = known
            cases.append(case)
            if case_observer is not None:
                # Diagnostic extension gets frozen forecasts and preceding donors;
                # never changes the original forecast or source history.
                case_observer(case, run, layout, spans, response_rows, predictions, donors, scale, response_history)
            # Target measurements only enter donor history AFTER forecasting/scoring.
            speed_spans = tuple(SignalSpan(r.start,r.end,r.speed,r.speed) for r in response_rows)
            nonwork_spans = mask_trace(speed_spans, tuple(SignalSpan(a,b,0,0) for k,a,b in layout if k == 'nonwork'))
            nonwork_known = sum(s.end-s.start for s in nonwork_spans)
            speeds = [b.speed_mps for b in assessment.bouts if b.speed_mps is not None and b.speed_mps > 0]
            donors[run.key] = dict(start=run.start, end=run.end, available_at=run.available_at,
                nonwork_speed=sum((s.end-s.start)*(s.first+s.last)/2 for s in nonwork_spans)/nonwork_known if nonwork_known else None,
                nonwork_known_seconds=nonwork_known, work_speed=geometric_mean(speeds) if speeds else None,
                load_rate=actual_load/known if known else None, hr_coverage=trace['coverage'])
            assessments.append(assessment)
        if source_observer is not None:
            # Diagnostic receives original source and strictly preceding response
            # history AFTER held-out scoring, before target history insertion.
            source_observer(row,activity,response_rows,reference.accounting_key,eligible[row.key],
                            tuple(response_history))
        if response_rows:
            response_history.append(ResponseEvidence(row.key, row.end, row.available_at, reference.accounting_key,
                sufficient_statistics(response_rows), (min(r.speed for r in response_rows),max(r.speed for r in response_rows)),
                eligible=eligible[row.key]))
            if response_observer is not None:
                # Diagnostic evidence is delivered AFTER the held-out callback.
                # The default fit/history/output remains unchanged.
                response_observer(row, response_rows, reference.accounting_key, eligible[row.key])
        if activity_observer is not None:
            # Original activity evidence enters extensions only AFTER held-out
            # scoring, including easy/long runs without a quality callback.
            activity_observer(row, activity, reference.accounting_key, eligible[row.key])
        return row
    old = QualityReference('threshold','work_seconds',Quantity.point(900,'discarded comparator'),
        Quantity(290,310),'physical original-file validation only')
    _, unresolved = load_history(database, config, as_of, old, since=min(t for _,t in inventory), run_transform=collect)
    if {c['key'] for c in cases} != expected:
        raise AssertionError('Missing quality cases')
    if sha256(database.read_bytes()).hexdigest() != digest:
        raise AssertionError('Source database changed')
    groups = {label: summarize([c for c in cases if c['label'] == label]) for label in sorted({c['label'] for c in cases})}
    overall = summarize(cases)
    passed = all(s['paired'] > 0 and s['model']['mean_absolute_fraction_error'] < s['baseline_mean_absolute_fraction_error']
                 for s in [overall, *groups.values()])
    return dict(version='quality_load_conditional_validation_v1', policy=POLICY, initial_reserve=INITIAL_RESERVE,
        cases=cases, donors=donors, overall=overall, groups=groups, directional_check_passed=passed,
        missing_sources=unresolved, database_unchanged=True, elapsed_seconds=perf_counter()-started, planner_influence=False,
        protocol=['Exploratory timing-conditioned hindcast, NOT issued-prescription validation',
            'Existing affine speed/HR equilibrium and first-order response, fitted only on preceding eligible 30-day history',
            'Quality v1 pace frozen; nonwork speed is equal-session mean from same prior quality contributors, including standing',
            'Initial HRR .45 fixed; actual target HR/speed never enter forecast; state carried across all phases',
            'Baseline: equal-session preceding quality load per known HR second, scaled to target known HR duration',
            'Unconditioned-pace ablation changes only work speed; same response fit, timing, nonwork and initial state',
            'Same known-time masks; all cases retained; errors equal-weighted by session, not rep',
            'Directional check: lower mean absolute relative load error overall AND each quality category; not significance',
            'Current labels/settings/health; availability assumed at completion, not original upload time',
            'Detected work (particularly old tempo autolaps) is not verified intended structure; unmodeled environment and drift',
            'No tuning, recovery/benefit validation, uncertainty calibration or scheduler integration'])


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output path')
    result = audit(Path('data/run_analysis.sqlite'),load_config(Path('config.yaml')),
        datetime(2026,9,26,4,tzinfo=timezone.utc))
    with args.output.open('x') as stream:
        json.dump(result, stream, default=encode, indent=2, allow_nan=False)
    print(json.dumps({k:result[k] for k in ('overall','groups','directional_check_passed','elapsed_seconds')},indent=2))


if __name__ == '__main__':
    main()
