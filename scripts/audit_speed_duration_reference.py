"""Bounded original-file check of a literature-based alternative pace reference."""
from argparse import ArgumentParser
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

from run_analysis.config import load_config
from run_analysis.movement import classify_movement
from run_analysis.signal_structure import describe_signals
from run_analysis.workload_accounting import Quantity
from run_analysis.work_stimulus_evidence import observed_work_signals
from scripts.audit_program_evidence_adapter import load_history
from scripts.audit_work_stimulus import encode
from scripts.program_evidence_adapter import QualityReference
from scripts.speed_duration_reference import ObservedSpeedCurve, WINDOWS, freeze_frontier


def audit(database, config, as_of):
    started = perf_counter()
    digest = sha256(database.read_bytes()).hexdigest()
    curves, contexts = [], {}
    def collect(row, activity):
        intervals = classify_movement(activity.trackpoints, config['moving_time']).intervals
        report = describe_signals(intervals, hr_thresholds_bpm=(), windows_seconds=WINDOWS)
        signal = observed_work_signals(row, activity, config, intervals, ()) if row.bouts else None
        contexts[row.key] = {}
        for point in report['speed_duration_curve']:
            start, duration = point['start_seconds'], point['window_seconds']
            overlap = (sum(max(0., min(b, start+duration)-max(a, start)) for a, b in signal.windows)
                       if signal is not None and start is not None else None)
            contexts[row.key][duration] = dict(start_seconds=start, role=row.role,
                supported_work_seconds=overlap,
                outside_supported_work_seconds=duration-overlap if overlap is not None else None,
                maximal_effort_established=False)
        curves.append(ObservedSpeedCurve(row.key, row.start, row.end, row.available_at,
            tuple((p['window_seconds'], p['peak_mean_mps']) for p in report['speed_duration_curve']
                  if p['peak_mean_mps'] is not None and p['peak_mean_mps'] > 0)))
        return row
    reference = QualityReference('threshold', 'work_seconds', Quantity.point(900, 'discarded comparator'),
        Quantity(290, 310), 'reuse original-file physical loader, no quality score used')
    rows, unresolved = load_history(database, config, as_of, reference, run_transform=collect)
    if unresolved:
        raise ValueError(f'Unresolved originals: {unresolved}')
    # 42 days is the six-week habitual-data period in Hunter et al.; 84 is a
    # predeclared sensitivity check matching our loaded inventory, not model selection.
    snapshots = [freeze_frontier(curves, as_of, lookback_days=d) for d in (42., 84.)]
    for snapshot in snapshots:
        snapshot['anchor_context'] = [contexts[a['source_key']][a['seconds']]
                                      for a in snapshot['fit']['anchors']]
    historical = [dict(key=r.key, start=r.start, reference=freeze_frontier(curves, r.start))
                  for r in sorted(rows, key=lambda r: r.start) if r.role == 'quality']
    if sha256(database.read_bytes()).hexdigest() != digest:
        raise AssertionError('Source database changed')
    return dict(version='empirical_speed_duration_audit_v1', snapshots=snapshots,
        historical_prior_only=historical, curves=curves, database_unchanged=True,
        planner_influence=False, elapsed_seconds=perf_counter()-started,
        note='Historical early snapshots may be left-truncated by the 84-day loaded inventory; no accuracy claim')


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(Path('data/run_analysis.sqlite'), load_config(Path('config.yaml')),
                   datetime.fromisoformat('2026-09-26T04:00:00+00:00'))
    with args.output.open('x') as stream:
        json.dump(result, stream, default=encode, indent=2, allow_nan=False)
    print(json.dumps(dict(seconds=result['elapsed_seconds'], snapshots=[
        dict(days=s['lookback_days'], fit=s['fit'], deletion_sensitivity=s['deletion_sensitivity_mps'])
        for s in result['snapshots']]), indent=2))


if __name__ == '__main__':
    main()
