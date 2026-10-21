"""Closeout of v1 accounting/reference contract on every quality-labelled run.

No predictive accuracy metric, tuning, scheduler search, or production changes.
"""
from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
from time import perf_counter

from run_analysis.config import load_config
from run_analysis.continuous_training_dose import ContinuousDoseReference
from run_analysis.movement import classify_movement
from run_analysis.training_signals import HeartRateReference, observed_cardio_trace
from run_analysis.work_stimulus_evidence import observed_work_signals
from run_analysis.workload_accounting import Quantity
from scripts.audit_program_evidence_adapter import load_history
from scripts.audit_workload_expectation import app_session_types
from scripts.audit_work_stimulus import encode
from scripts.continuous_program_adapter import with_continuous_trace
from scripts.program_evidence_adapter import QualityReference
from scripts.quality_session_model import assess_quality_session, current_quality_pace, QualityHistoryPolicy, VERSION


def audit(database, config, as_of):
    started = perf_counter()
    digest = sha256(database.read_bytes()).hexdigest()
    roles = app_session_types(database)
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro', uri=True) as db:
        inventory = [(str(k), datetime.fromisoformat(t)) for k, t in db.execute(
            "SELECT id,start_time_utc FROM activities WHERE sport='Running' AND start_time_utc IS NOT NULL")]
    expected = {k for k, at in inventory if at < as_of and roles.get(k) in {'intervals','tempo_threshold'}}
    reference = ContinuousDoseReference('unchanged continuous-HRR accounting; quality component v1')
    hr = HeartRateReference(config['resting_hr'], config['max_hr'], 'current settings, historical settings unavailable')
    policy = QualityHistoryPolicy()
    assessments = []
    def collect(row, activity):
        if row.key not in expected:
            return row
        intervals = classify_movement(activity.trackpoints, config['moving_time']).intervals
        trace = observed_cardio_trace(activity, intervals, hr, reference.policy, include_spans=True, compute_density=False)
        run, _ = with_continuous_trace(row, trace['normalized_spans'], reference=reference,
            trace_key=row.key, trace_source='observed', elapsed_seconds=trace['elapsed_seconds'],
            outside_reference_seconds=trace['exclusions']['outside_reference_seconds'])
        signal = observed_work_signals(run, activity, config, intervals, trace['normalized_spans']) if run.bouts else None
        assessments.append(assess_quality_session(run, signal, reference=reference, as_of=run.available_at))
        return run
    old = QualityReference('threshold', 'work_seconds', Quantity.point(900, 'discarded comparator'),
        Quantity(290, 310), 'physical original-file validation only')
    _, unresolved = load_history(database, config, as_of, old, since=min(t for _, t in inventory), run_transform=collect)
    if {a.run.key for a in assessments} != expected:
        raise AssertionError('Not every quality-labelled run reached component closeout')
    cases = []
    for item in sorted(assessments, key=lambda a: a.run.start):
        refs = [current_quality_pace(assessments, as_of=item.run.start, bout_seconds=duration,
            accounting_key=reference.accounting_key, policy=policy)
            for duration in sorted({b.seconds for b in item.bouts})]
        cases.append(dict(key=item.run.key, start=item.run.start, label=roles[item.run.key],
            assessment=item, prior_references=refs, reference_fingerprints=[r.fingerprint for r in refs]))
    current = [current_quality_pace(assessments, as_of=as_of, bout_seconds=d,
        accounting_key=reference.accounting_key, policy=policy) for d in (90., 300., 900.)]
    if sha256(database.read_bytes()).hexdigest() != digest:
        raise AssertionError('Source database changed')
    return dict(version=VERSION, policy=policy, cases=cases, current_references=current,
        summary=dict(quality_count=len(cases), aligned=sum(a.alignment == 'aligned' for a in assessments),
            total_bouts=sum(len(a.bouts) for a in assessments),
            bouts_with_pace=sum(b.speed_mps is not None for a in assessments for b in a.bouts),
            pace_sources=dict(Counter(b.speed_basis for a in assessments for b in a.bouts)),
            cases_with_recent_reference=sum(any(r.speed_mps is not None for r in c['prior_references']) for c in cases),
            unresolved_alignment=[a.run.key for a in assessments if a.alignment != 'aligned'],
            current_source_keys=sorted({k for r in current for k, _, _ in r.session_estimates})),
        non_quality_unresolved_sources=unresolved, elapsed_seconds=perf_counter()-started,
        database_unchanged=True, planner_influence=False, predictive_validity_tested=False,
        limitations=['Current app labels/HR settings; historical upload and settings availability not reconstructed',
            'Thirty-day current pace policy is provisional and frozen before predictive evaluation',
            'Actual duration queries are conditional diagnostics, not pre-run forecasts',
            'No hypothetical HR traces inferred from target zones or held-out completion',
            'Component closure is not validated benefit/recovery or scheduler stability'])


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
