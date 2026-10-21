"""All app-resolved quality sessions, original files, prior-only rolling evidence."""
from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
from time import perf_counter
from zoneinfo import ZoneInfo

from run_analysis.config import load_config
from run_analysis.movement import classify_movement
from run_analysis.signal_structure import describe_signals
from run_analysis.work_stimulus_evidence import observed_work_signals
from run_analysis.workload_accounting import Quantity
from scripts.audit_program_evidence_adapter import load_history
from scripts.audit_workload_expectation import app_session_types
from scripts.audit_work_stimulus import encode
from scripts.program_evidence_adapter import QualityReference
from scripts.speed_duration_reference import ObservedSpeedCurve, WINDOWS, freeze_frontier
from scripts.work_pace_envelope import demonstrated_envelope

DOSE_WINDOWS = (30., 60., 120., 180., 300., 420., 720., 1200.)


def audit(database, config, as_of):
    started = perf_counter()
    digest = sha256(database.read_bytes()).hexdigest()
    roles = app_session_types(database)
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro', uri=True) as db:
        inventory = [(str(k), datetime.fromisoformat(t)) for k, t in db.execute(
            "SELECT id,start_time_utc FROM activities WHERE sport='Running' AND start_time_utc IS NOT NULL")]
    expected = {k for k, t in inventory if t < as_of and roles.get(k) in {'intervals', 'tempo_threshold'}}
    since = min(t for _, t in inventory)
    curves, signals, contexts, alignment_details = [], {}, {}, {}
    def collect(row, activity):
        intervals = classify_movement(activity.trackpoints, config['moving_time']).intervals
        report = describe_signals(intervals, hr_thresholds_bpm=(), windows_seconds=WINDOWS)
        curves.append(ObservedSpeedCurve(row.key, row.start, row.end, row.available_at,
            tuple((p['window_seconds'], p['peak_mean_mps']) for p in report['speed_duration_curve']
                  if p['peak_mean_mps'] is not None and p['peak_mean_mps'] > 0)))
        signal = observed_work_signals(row, activity, config, intervals, ()) if row.bouts else None
        if signal is not None and row.role == 'quality':
            signals[row.key] = signal
        if row.role == 'quality' and signal is None:
            alignment_details[row.key] = dict(
                reason='existing_detector_did_not_resolve_work' if not row.bouts else 'work_window_alignment_failed',
                interior_laps=[dict(index=lap.lap_index, seconds=lap.total_time_s,
                    average_hr_bpm=lap.average_hr_bpm) for lap in activity.laps[1:-1]],
                current_z3_floor=config['zones']['z3'][0],
                note='Current HR settings, not reconstructed historical zones; label does not locate work')
        contexts[row.key] = {}
        for p in report['speed_duration_curve']:
            start, seconds = p['start_seconds'], p['window_seconds']
            overlap = (sum(max(0., min(b, start+seconds)-max(a, start)) for a, b in signal.windows)
                       if signal is not None and start is not None else None)
            contexts[row.key][seconds] = dict(role=row.role, supported_work_seconds=overlap,
                outside_supported_work_seconds=seconds-overlap if overlap is not None else None)
        return row
    reference = QualityReference('threshold', 'work_seconds', Quantity.point(900, 'discarded comparator'),
        Quantity(290, 310), 'physical original-file loader only; no quality score used')
    rows, unresolved = load_history(database, config, as_of, reference, run_transform=collect, since=since)
    quality = sorted((r for r in rows if r.key in expected), key=lambda r: r.start)
    accounted = {r.key for r in quality} | {u['key'] for u in unresolved if u['key'] in expected}
    if accounted != expected:
        raise AssertionError(f'Quality inventory mismatch: {expected-accounted}')
    cases = []
    for row in quality:
        panels = []
        for days in (42., 84.):
            before = freeze_frontier(curves, row.start, lookback_days=days)
            after_at = row.end+timedelta(microseconds=1)
            # Fix the original cutoff while adding this completion, avoiding rollover contamination.
            available = [c for c in curves if c.start >= row.start-timedelta(days=days)
                         and (c.end < row.start or c.key == row.key)]
            after = freeze_frontier(available, after_at,
                lookback_days=days+(after_at-row.start).total_seconds()/86400)
            def annotate(snapshot):
                snapshot['anchor_context'] = [contexts[a['source_key']][a['seconds']]
                                               for a in snapshot['fit']['anchors']]
            annotate(before); annotate(after)
            a, b = before['fit'], after['fit']
            envelope_non_decreasing = len(a['anchors']) == len(b['anchors']) == 3 and all(
                y['speed_mps'] >= x['speed_mps']-1e-10 for x, y in zip(a['anchors'], b['anchors']))
            panels.append(dict(days=days, before=before, after=after,
                slope_fell_while_envelope_non_decreased=(envelope_non_decreasing and
                    a['slope_mps'] is not None and b['slope_mps'] < a['slope_mps']-1e-8)))
        signal = signals.get(row.key)
        duration_queries = sorted({*DOSE_WINDOWS, *(b-a for a, b in signal.windows)}) if signal else list(DOSE_WINDOWS)
        prior = demonstrated_envelope(signals.values(), duration_queries, row.start)
        local = demonstrated_envelope([signal], duration_queries, row.end+timedelta(microseconds=1)) if signal else None
        expanded = demonstrated_envelope(
            [e for e in signals.values() if e.start >= row.start-timedelta(days=84)
             and (e.end < row.start or e.key == row.key)],
            duration_queries, row.end+timedelta(microseconds=1),
            lookback_days=84+(row.end-row.start).total_seconds()/86400+1e-6)
        changes = []
        for p, q in zip(prior['points'], expanded['points']):
            if p['speed_mps'] is not None and (q['speed_mps'] is None or q['speed_mps'] < p['speed_mps']-1e-9):
                raise AssertionError('Added evidence decreased demonstrated work envelope')
            if p['speed_mps'] != q['speed_mps']:
                changes.append(dict(seconds=p['seconds'], before=p['speed_mps'], after=q['speed_mps']))
        cases.append(dict(key=row.key, start=row.start, local_date=row.start.astimezone(
            ZoneInfo(config['timezone_default'])).date(), label=roles[row.key],
            bouts=len(row.bouts), work_basis=row.work_evidence_basis,
            total_work_seconds=sum(b.work_seconds for b in row.bouts) if row.bouts else None,
            aligned=signal is not None, alignment_status='aligned' if signal else 'no_supported_work_alignment',
            unresolved_alignment=alignment_details.get(row.key),
            critical_speed_panels=panels, work_pace_before=prior, own_work_pace=local,
            work_envelope_changes=changes, added_evidence_monotone=True))
    summary = dict(database_quality_count=len(expected), audited_quality_count=len(cases),
        unresolved_quality=[u for u in unresolved if u['key'] in expected],
        labels=dict(Counter(c['label'] for c in cases)), aligned=sum(c['aligned'] for c in cases),
        all_loaded_runs=len(rows), all_unresolved=unresolved,
        post_fit_single_session_anchors=sum(p['after']['fit'].get('distinct_anchor_sessions') == 1 for c in cases for p in c['critical_speed_panels']),
        post_fit_panels=sum(len(c['critical_speed_panels']) for c in cases),
        slope_opposite_to_evidence=sum(p['slope_fell_while_envelope_non_decreased'] for c in cases for p in c['critical_speed_panels']),
        affected_uploads=[c['key'] for c in cases if any(p['slope_fell_while_envelope_non_decreased']
                          for p in c['critical_speed_panels'])],
        post_fit_known_non_work_anchor_panels=sum(any((a['outside_supported_work_seconds'] or 0) > 1e-6
            for a in p['after']['anchor_context']) for c in cases for p in c['critical_speed_panels']),
        prior_work_supported_cases=sum(any(p['speed_mps'] is not None for p in c['work_pace_before']['points']) for c in cases),
        work_envelope_monotonicity_passed=all(c['added_evidence_monotone'] for c in cases))
    if sha256(database.read_bytes()).hexdigest() != digest:
        raise AssertionError('Source database changed')
    return dict(version='all_quality_reference_audit_v1', summary=summary, cases=cases,
        elapsed_seconds=perf_counter()-started, database_unchanged=True, planner_influence=False,
        limitations=['All CURRENT app quality labels, including overrides; historical label availability not reconstructed',
            'All original history loaded; individual references still use only preceding 42/84 days',
            'Dose-conditioned reference queries use actual bout durations, not a pre-run forecast validation',
            'Work-only peaks are demonstrated outputs, not maximal capacity or predictions of future pace',
            'No corrected weather/terrain, calibrated benefit, quality-mix choice or scheduler stability claim'])


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(Path('data/run_analysis.sqlite'), load_config(Path('config.yaml')),
                   datetime(2026, 9, 26, 4, tzinfo=timezone.utc))
    with args.output.open('x') as stream:
        json.dump(result, stream, default=encode, indent=2, allow_nan=False)
    print(json.dumps(dict(seconds=result['elapsed_seconds'], **result['summary']), indent=2))


if __name__ == '__main__':
    main()
