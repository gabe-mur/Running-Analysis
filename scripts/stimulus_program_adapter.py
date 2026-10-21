"""Offline single-declared-target comparison using the shared program score.

Replaces generic quality coverage for this comparison; never adds a second
quality reward. Target choice is explicit, not a claim all quality is fungible.
"""
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json

from run_analysis.work_stimulus import assess_work_stimulus
from run_analysis.workload_accounting import Quantity
from scripts.continuous_program_adapter import validate_continuous_runs
from scripts.program_evidence_adapter import evaluate_accounted, reference_key


@dataclass(frozen=True)
class StimulusComparison:
    accounting: object
    target: object


def evaluate_stimulus_program(context, history, planned, signals, *, reference, target, **kwargs):
    history, planned = tuple(history), tuple(planned)
    validate_continuous_runs(history, planned, reference)
    rows = (*history, *planned)
    if set(signals)-{r.key for r in rows}:
        raise ValueError('Signal evidence has unmatched session identities')
    comparison = StimulusComparison(reference, target)
    transformed, reports = [], {}
    for row in rows:
        evidence = signals.get(row.key)
        if evidence is not None:
            if any(b.duration_range_seconds and b.duration_range_seconds[0] != b.duration_range_seconds[1]
                   for b in row.bouts):
                raise ValueError('Resolve ranged work into a declared forecast scenario before aligning signals')
            if ((evidence.key, evidence.source, evidence.start, evidence.end) !=
                    (row.key, row.source, row.start, row.end) or evidence.available_at > kwargs['as_of']):
                raise ValueError('Signal evidence identity, provenance or availability mismatch')
            if (evidence.basis != row.work_evidence_basis or len(evidence.windows) != len(row.bouts)
                    or any(abs((b-a)-bout.work_seconds) > 1e-3
                           for (a, b), bout in zip(evidence.windows, row.bouts))):
                raise ValueError('Signal windows must match supported work segments')
            report = assess_work_stimulus(evidence, target)
            credit = report['coverage']
            reports[row.key] = report
        else:
            # Missing alignment cannot silently acquire a purpose from its label.
            upper = 1. if row.work_evidence_basis == 'unknown' else min(1., row.demand[2].high/target.minutes)
            credit = Quantity(0., upper, basis='unresolved_work_signal_alignment')
            reports[row.key] = dict(status='unresolved_work_signal_alignment')
        if row.role != 'quality':
            credit = Quantity.point(0., 'not_resolved_quality')
        transformed.append(replace(row, quality=credit, quality_reference_key=reference_key(comparison)))
    result = evaluate_accounted(context, transformed[:len(history)], transformed[len(history):],
        reference=comparison, accounting_key=reference.accounting_key, **kwargs)
    result['stimulus_reports'] = reports
    result['generic_work_coverage'] = {r.key: r.quality for r in rows}
    result['stimulus_target'] = target
    # References, windows and traces are evidence even when totals happen to tie.
    result['stimulus_fingerprint'] = sha256(json.dumps(
        dict(comparison=asdict(comparison), signals={k: asdict(v) for k, v in sorted(signals.items())}),
        sort_keys=True, default=str, allow_nan=False).encode()).hexdigest()
    result['limitations'] = [
        'Single explicit target comparison, not an automatic quality mix or adaptation prediction',
        'Scenario bands are assumptions, not measured lactate/VO2max boundaries',
        'Raw pace is not terrain/weather adjusted; reference dispersion is not a confidence interval',
        'Signal uncertainty scenarios do not prove target delivery',
        'No live scheduler, recovery clearance, or progression/stability validation',
    ]
    return result
