"""Descriptive full-quality inventory audit; no fitting, scoring or live writes."""
from argparse import ArgumentParser
from dataclasses import asdict
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
from time import perf_counter

from run_analysis.athlete_projection import aware
from run_analysis.config import load_config
from run_analysis.continuous_training_dose import ContinuousDoseReference
from run_analysis.movement import classify_movement
from run_analysis.quality_work_exposure import work_cardio_exposure
from run_analysis.training_signals import HeartRateReference, observed_cardio_trace
from run_analysis.work_stimulus_evidence import observed_work_signals
from run_analysis.workload_accounting import Quantity
from scripts.audit_program_evidence_adapter import load_history
from scripts.audit_work_stimulus import encode
from scripts.audit_workload_expectation import app_session_types
from scripts.continuous_program_adapter import with_continuous_trace
from scripts.program_evidence_adapter import QualityReference
from scripts.research_manifest import freeze_sources


def audit(database, config, as_of):
    aware(as_of)
    started = perf_counter()
    digest = sha256(database.read_bytes()).hexdigest()
    roles = app_session_types(database)
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro', uri=True) as db:
        inventory = [(str(k), datetime.fromisoformat(t)) for k,t in db.execute(
            "SELECT id,start_time_utc FROM activities WHERE sport='Running' AND start_time_utc IS NOT NULL")]
    expected = {k for k,t in inventory if t < as_of and roles.get(k) in {'intervals', 'tempo_threshold'}}
    reference = ContinuousDoseReference('existing continuous-dose assumptions; descriptive work-only audit')
    hr = HeartRateReference(config['resting_hr'], config['max_hr'], 'current settings; historical changes unavailable')
    old = QualityReference('threshold', 'work_seconds', Quantity.point(900, 'unused comparator'),
                           Quantity(290, 310), 'physical bridge only, not historical intention')
    cases = []
    def collect(row, activity):
        if row.key not in expected:
            return row
        intervals = classify_movement(activity.trackpoints, config['moving_time']).intervals
        trace = observed_cardio_trace(activity, intervals, hr, reference.policy,
                                     include_spans=True, compute_density=False)
        adjusted, _ = with_continuous_trace(row, trace['normalized_spans'], reference=reference,
            trace_key=row.key, trace_source='observed', elapsed_seconds=trace['elapsed_seconds'],
            outside_reference_seconds=trace['exclusions']['outside_reference_seconds'])
        signals = observed_work_signals(adjusted, activity, config, intervals, trace['normalized_spans'])
        cases.append(dict(key=row.key, start=row.start, label=roles[row.key],
            alignment_status='aligned' if signals is not None else 'unresolved_work_alignment',
            whole_cardio=adjusted.demand[1], generic_work_minutes=adjusted.demand[2],
            generic_credit=adjusted.quality, hr_coverage=trace['coverage'], hr_exclusions=trace['exclusions'],
            work=work_cardio_exposure(signals, reference=reference) if signals is not None else None))
        return row
    _, unresolved = load_history(database, config, as_of, old, since=min(t for _,t in inventory), run_transform=collect)
    seen = {c['key'] for c in cases}
    omitted = expected-seen
    attributed = {r['key'] for r in unresolved}
    if omitted-attributed or len(seen) != len(cases):
        raise AssertionError('Unattributed missing or duplicate quality inventory')
    if sha256(database.read_bytes()).hexdigest() != digest:
        raise AssertionError('Evidence database changed')
    return dict(version='quality_work_exposure_inventory_v1', as_of=as_of,
        reference=asdict(reference), expected_quality_count=len(expected),
        quality_count=len(cases), aligned_count=sum(c['work'] is not None for c in cases),
        missing_sources=unresolved, cases=cases, elapsed_seconds=perf_counter()-started,
        database_unchanged=True, planner_influence=False,
        scope='Measured descriptors only; no benefit conversion, fatigue fit or forecast validation',
        limitations=('Current resolved classifications and HR configuration; historical changes unavailable',
            'Detected work alignment is not proof of prescribed intensity',
            'Continuous-dose reference constants remain disclosed assumptions',
            'Short-effort HR lag is retained; external speed and structure are not erased',
            'No physiological equivalence, automatic quality mix or live scheduler acceptance'))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('config.yaml'))
    parser.add_argument('--as-of', type=datetime.fromisoformat, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output path')
    config = load_config(args.config)
    manifest = freeze_sources(args.manifest, config=config, entrypoint=__name__, inputs=(args.database,))
    result = audit(args.database, config, args.as_of)
    result['manifest'] = manifest
    with args.output.open('x') as stream:
        json.dump(result, stream, default=encode, indent=2, allow_nan=False)
    print(json.dumps({k: result[k] for k in ('quality_count', 'aligned_count', 'elapsed_seconds', 'database_unchanged')}))


if __name__ == '__main__':
    main()
