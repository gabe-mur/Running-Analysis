"""Original-file target exposure audit with chronological easy-speed references."""
from argparse import ArgumentParser
from dataclasses import asdict, replace
from datetime import datetime, timedelta
from hashlib import sha256
import json
from pathlib import Path
from statistics import median
from time import perf_counter

from run_analysis.config import load_config
from run_analysis.continuous_training_dose import ContinuousDoseReference
from run_analysis.movement import classify_movement
from run_analysis.performance_reference import AerobicReference, aerobic_speed
from run_analysis.projection_evidence import steady_pace_samples
from run_analysis.signal_structure import SignalSpan
from run_analysis.training_signals import HeartRateReference, observed_cardio_trace
from run_analysis.work_stimulus import StimulusTarget, assess_work_stimulus
from run_analysis.work_stimulus_evidence import observed_work_signals
from run_analysis.workload_accounting import Quantity
from scripts.audit_program_evidence_adapter import load_history, json_ready
from scripts.continuous_program_adapter import with_continuous_trace
from scripts.program_evidence_adapter import QualityReference


def audit(database, config, as_of):
    started = perf_counter()
    digest = sha256(database.read_bytes()).hexdigest()
    reference = ContinuousDoseReference('unchanged offline work-coverage reference')
    hr = HeartRateReference(config['resting_hr'], config['max_hr'], 'configured at audit; historical changes unavailable')
    targets = (
        StimulusTarget('elevated cardiovascular work', 'hrr', (.7, .8, .9, 1.), 15., 'declared test band, not threshold physiology'),
        StimulusTarget('fast running work', 'speed_over_easy', (1.05, 1.25, 1.6, 1.8), 15., 'declared test band, not VO2max or sprint pace'),
    )
    old = QualityReference('threshold', 'work_seconds', Quantity.point(900, 'discarded comparator'),
                           Quantity(290, 310), 'physical bridge validation only')
    originals = {}

    def collect(row, activity):
        intervals = classify_movement(activity.trackpoints, config['moving_time']).intervals
        trace = observed_cardio_trace(activity, intervals, hr, reference.policy,
                                     include_spans=True, compute_density=False)
        adjusted, _ = with_continuous_trace(row, trace['normalized_spans'], reference=reference,
            trace_key=row.key, trace_source='observed', elapsed_seconds=trace['elapsed_seconds'],
            outside_reference_seconds=trace['exclusions']['outside_reference_seconds'])
        originals[row.key] = activity, intervals, trace
        return adjusted

    history, unresolved = load_history(database, config, as_of, old, run_transform=collect)
    if unresolved:
        raise ValueError(f'Unresolved original sessions: {unresolved}')
    earlier_easy, cases = [], []
    for row in sorted(history, key=lambda r: r.start):
        activity, intervals, trace = originals[row.key]
        if row.role == 'quality':
            prior = [(key, knots) for key, end, knots in earlier_easy
                     if row.start-timedelta(days=84) <= end < row.start]
            frozen_at = row.start-timedelta(microseconds=1)
            baseline = AerobicReference(frozen_at, reference.accounting_key, tuple(prior), 84., .25)
            speed = aerobic_speed(baseline, .70)
            evidence = observed_work_signals(row, activity, config, intervals, trace['normalized_spans'],
                easy_speed_mps=speed['speed_mps'], reference_at=frozen_at if speed['speed_mps'] else None,
                reference_source='prior easy-only session dispersion at .70 HRR; unadjusted for terrain/weather')
            case = dict(key=row.key, start=row.start, work_minutes=row.demand[2],
                        generic_coverage=row.quality, pace_reference=speed, aligned=evidence is not None)
            if evidence is not None:
                seconds = (row.end-row.start).total_seconds()
                padded = replace(evidence, end=row.end+timedelta(minutes=20),
                    available_at=row.available_at+timedelta(minutes=20),
                    hrr=(*evidence.hrr, SignalSpan(seconds, seconds+1200, .7, .7)))
                case['targets'] = {}
                for target in targets:
                    assessed = assess_work_stimulus(evidence, target)
                    padding = assess_work_stimulus(padded, target)
                    if assessed['supported_minutes'] != padding['supported_minutes']:
                        raise AssertionError('Out-of-work HR padding changed target exposure')
                    case['targets'][target.name] = assessed
                case['padding_invariant'] = True
                case['work_windows'] = evidence.windows
            cases.append(case)
        if row.role == 'easy':
            samples = steady_pace_samples(intervals, row.start, config, hr)
            knots = {}
            for sample in samples:
                knots.setdefault(sample.mean_hr_reserve, []).append(sample.meters/sample.seconds)
            if knots:
                earlier_easy.append((row.key, row.end, tuple((r, median(v)) for r, v in sorted(knots.items()))))
    if sha256(database.read_bytes()).hexdigest() != digest:
        raise AssertionError('Source database changed during audit')
    return dict(status='offline_original_signal_audit_not_forecast_validation',
        history_count=len(history), quality_count=len(cases), aligned_count=sum(c['aligned'] for c in cases),
        cases=cases, elapsed_seconds=perf_counter()-started, database_unchanged=True, planner_influence=False,
        limitations=['Targets are explicit sensitivity assumptions, not physiological cutoffs',
            'Only earlier loaded easy runs inform pace references; historical HR configuration changes unknown',
            'Missing pace history widens uncertainty; no exact-repeat prerequisite',
            'Lap/inferred windows are existing detection, not proof of absolute physiological intensity',
            'No automatic target mix, calibrated forecast, recovery clearance or scheduler stability claim'])


def encode(value):
    if hasattr(value, '__dataclass_fields__'):
        return asdict(value)
    return json_ready(value)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(Path('data/run_analysis.sqlite'), load_config(Path('config.yaml')),
                   datetime.fromisoformat('2026-09-26T04:00:00+00:00'))
    with args.output.open('x') as handle:
        json.dump(result, handle, default=encode, indent=2, allow_nan=False)
    print(json.dumps({k: result[k] for k in ('history_count', 'quality_count', 'aligned_count',
                                           'elapsed_seconds', 'database_unchanged')}))


if __name__ == '__main__':
    main()
