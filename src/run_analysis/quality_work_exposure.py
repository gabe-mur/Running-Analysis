"""Work-only cardiovascular descriptors; not a universal quality benefit score.

Reuse the current continuous HR-load formula, with explicit work alignment.
External speed, duration and structure remain separate. This diagnostic is not
installed in the planner and adds no family coefficients or HR-lag correction.
"""
from dataclasses import asdict
from math import fsum

from .continuous_training_dose import continuous_dose
from .phase_calibration import clip_trace
from .signal_structure import SignalSpan, above_threshold
from .workload_accounting import Quantity


def work_cardio_exposure(evidence, *, reference):
    """Apply the same actual/projected mapping to supported work windows only.

    Rebase each window before integration: recovery outside a window is NOT
    missing work HR. Unknown HR *inside* work retains explicit bounds. Original
    timestamps are retained for continuity; disconnected bouts are not joined.
    The reference's existing numerical assumptions are disclosed, not refitted.
    """
    bouts, work_spans = [], []
    for left, right in evidence.windows:
        clipped = clip_trace(evidence.hrr, left, right)
        work_spans.extend(clipped)
        rebased = tuple(SignalSpan(s.start-left, s.end-left, s.first, s.last) for s in clipped)
        dose = continuous_dose(rebased, right-left, reference)
        speed = clip_trace(evidence.speed_mps, left, right)
        speed_seconds = fsum(s.end-s.start for s in speed)
        bouts.append(dict(start_seconds=left, end_seconds=right, work_seconds=right-left,
            known_hr_seconds=fsum(s.end-s.start for s in clipped),
            unknown_hr_seconds=dose['unknown_seconds'], cardio=dose['cardio'],
            high_effort_minutes=dose['high_effort_minutes'],
            known_speed_seconds=speed_seconds,
            mean_speed_mps=(fsum((s.end-s.start)*(s.first+s.last)/2 for s in speed)/speed_seconds
                            if speed_seconds else None)))
    def total(field):
        rows = [b[field] for b in bouts]
        return Quantity(fsum(q.low for q in rows), fsum(q.high for q in rows),
                        basis='work_only_'+field+'_not_adaptation')
    return dict(version='quality_work_exposure_v1', key=evidence.key,
        source=evidence.source, alignment_basis=evidence.basis,
        work_seconds=fsum(b['work_seconds'] for b in bouts),
        known_hr_seconds=fsum(b['known_hr_seconds'] for b in bouts),
        unknown_hr_seconds=fsum(b['unknown_hr_seconds'] for b in bouts),
        cardio=total('cardio'), high_effort_minutes=total('high_effort_minutes'),
        sustained_hr=above_threshold(work_spans, reference.quality_hrr),
        bouts=bouts, reference=asdict(reference), accounting_key=reference.accounting_key,
        physiological_equivalence_established=False, planner_influence=False,
        limitations=('This is internal cardiovascular exposure, not adaptation or tissue recovery',
            'Short fast efforts can retain external work with little high-HR exposure',
            'Recovery HR remains in whole-session load, not work-purpose exposure',
            'Reference parameters are existing assumptions, not athlete-fitted thresholds'))
