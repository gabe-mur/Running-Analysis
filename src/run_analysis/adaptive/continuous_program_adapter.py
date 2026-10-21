"""Offline adjustment of the connected bridge; no live scheduling changes."""
from dataclasses import replace

from run_analysis.continuous_training_dose import continuous_dose, general_quality_credit, supported_work_minutes
from run_analysis.adaptive.program_evidence_adapter import evaluate_accounted, reference_key


def with_continuous_trace(run, spans, *, reference, trace_key, trace_source,
                          elapsed_seconds, outside_reference_seconds=0.):
    """Replace old zone/template accounting, retaining physical work/identity.

    Forecast spans are declared scenarios, not derived from target zones or
    future outcomes. Actual spans must come from the matched original file.
    The old bridge's credit and cardiovascular points are discarded entirely.
    """
    if trace_key != run.key or trace_source != run.source:
        raise ValueError('Trace and event must have the same identity and provenance')
    if abs((run.end-run.start).total_seconds()-elapsed_seconds) > 1e-5:
        raise ValueError('Trace and event elapsed duration must agree')
    if outside_reference_seconds != 0:
        raise ValueError('HR outside reference requires review; no clipping or bounded total')
    report = continuous_dose(spans, elapsed_seconds, reference)
    work = supported_work_minutes(run.bouts, run.demand[2], evidence_basis=run.work_evidence_basis,
                                  source=run.source, elapsed_seconds=elapsed_seconds)
    quality = general_quality_credit(work, role=run.role, reference=reference)
    report.update(quality_work_minutes=work, work_evidence_basis=run.work_evidence_basis,
                  quality_evidence_status='supported_extent' if work is not None else 'unresolved_extent')
    return replace(run, demand=(run.demand[0], report['cardio'], run.demand[2]),
                   quality=quality, accounting_key=reference.accounting_key,
                   quality_reference_key=reference_key(reference)), report


def validate_continuous_runs(history, planned, reference):
    unresolved = []
    expected_reference=reference_key(reference)
    accounting=reference.accounting_key
    for row in (*history, *planned):
        if row.accounting_key != accounting:
            raise ValueError('Do not mix cardiovascular accounting units')
        if row.quality_reference_key != expected_reference:
            raise ValueError('Recompute quality credit for the current reference')
        work = row.supported_work
        expected = general_quality_credit(work, role=row.role, reference=reference)
        if (expected.low, expected.high) != (row.quality.low, row.quality.high):
            raise ValueError('Recompute quality coverage from supported work before evaluation')
        if work is None and row.role in {'quality', 'unknown'}:
            unresolved.append(row.key)
    return unresolved


def evaluate_continuous(context, history, planned, *, reference, **kwargs):
    from run_analysis.planner_observation import model_check
    model_check()  # Observation only: never feeds the score or search budget.
    history, planned = tuple(history), tuple(planned)
    unresolved = validate_continuous_runs(history, planned, reference)
    result = evaluate_accounted(context, history, planned, reference=reference,
                               accounting_key=reference.accounting_key, **kwargs)
    result['unresolved_quality_work'] = unresolved
    result['limitations'] = [
        'Generic quality exposure coverage is not threshold/VO2max/speed equivalence',
        'Quality coverage counts supported work extent, not whole-session HR excess',
        'Missing work is unresolved, not zero; uncertainty scenarios are not evidence',
        'Continuous HR curve and quality references require sensitivity validation',
        'Forecast traces are supplied scenarios, not a validated response predictor',
        'Conditional score is not readiness; no live selector or stability claim',
        'One run per local date; overnight or second daily runs need separate support',
    ]
    return result
