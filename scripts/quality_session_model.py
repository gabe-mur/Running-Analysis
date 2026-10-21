"""Frozen v1 offline quality component: measurement, recent pace, score handoff.

Not a fitted adaptation/recovery model. Historical maxima and illustrative
stimulus bands cannot supply current pace expectations. Prediction value is the
next validation question, not asserted by this engineering contract.
"""
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from hashlib import sha256
import json
from math import fsum, isfinite, log

from run_analysis.athlete_projection import aware
from run_analysis.continuous_training_dose import supported_work_minutes
from run_analysis.performance_reference import geometric_mean
from run_analysis.work_stimulus import _pieces
from run_analysis.workload_accounting import Quantity
from scripts.continuous_program_adapter import validate_continuous_runs, evaluate_continuous

VERSION = 'quality_session_component_v1'


from run_analysis.adaptive.quality_history_policy import QualityHistoryPolicy


@dataclass(frozen=True)
class BoutExposure:
    seconds: float
    recovery_after_seconds: float | None
    speed_mps: float | None
    mean_hrr: float | None
    speed_known_seconds: float
    hr_known_seconds: float
    speed_basis: str


@dataclass(frozen=True)
class QualityAssessment:
    run: object
    bouts: tuple[BoutExposure, ...]
    work_minutes: Quantity | None
    alignment: str
    version: str = VERSION


def _mean(spans, left, right):
    pieces = tuple(_pieces(spans, left, right))
    known = fsum(t for t, _, _ in pieces)
    # Partial coverage is recorded but never silently extrapolated over a bout.
    value = fsum(t*(a+b)/2 for t, a, b in pieces)/known if known > 0 and abs(known-(right-left)) <= 1e-5 else None
    return value, known


def assess_quality_session(run, signals, *, reference, as_of):
    aware(as_of)
    validate_continuous_runs((run,), (), reference)
    if run.available_at > as_of:
        raise ValueError('Session evidence not yet available')
    work = supported_work_minutes(run.bouts, run.demand[2], evidence_basis=run.work_evidence_basis,
        source=run.source, elapsed_seconds=(run.end-run.start).total_seconds())
    if signals is None:
        return QualityAssessment(run, (), work, 'unresolved_work_signals')
    if ((signals.key, signals.source, signals.start, signals.end, signals.basis) !=
        (run.key, run.source, run.start, run.end, run.work_evidence_basis)
            or signals.available_at > as_of or len(signals.windows) != len(run.bouts)):
        raise ValueError('Aligned source, identity, availability and work provenance required')
    rows = []
    for (left, right), bout in zip(signals.windows, run.bouts):
        if (bout.duration_range_seconds and bout.duration_range_seconds[0] != bout.duration_range_seconds[1]
                or abs(right-left-bout.work_seconds) > 1e-3):
            raise ValueError('Resolve work-duration scenario before aligning signals')
        speed, speed_known = _mean(signals.speed_mps, left, right)
        speed_basis = 'complete_work_trace_mean' if speed is not None else 'unresolved_work_pace'
        if bout.distance_meters is not None:
            if not isfinite(bout.distance_meters) or bout.distance_meters < 0:
                raise ValueError('Finite nonnegative supported bout distance required')
            # Recorded/inferred bout distance is independent of trace coverage.
            # Preserve it instead of demanding a perfect trackpoint at each lap edge.
            speed = bout.distance_meters/(right-left)
            speed_basis = 'supported_bout_distance_over_duration'
        hr, hr_known = _mean(signals.hrr, left, right)
        rows.append(BoutExposure(right-left, bout.recovery_after_seconds, speed, hr, speed_known, hr_known, speed_basis))
    return QualityAssessment(run, tuple(rows), work, 'aligned')


@dataclass(frozen=True)
class QualityPaceReference:
    issued_at: datetime
    requested_bout_seconds: float
    policy: QualityHistoryPolicy
    accounting_key: str
    speed_mps: Quantity | None
    # Equal session ownership; relevance reflects duration similarity, not confidence.
    session_estimates: tuple[tuple[str, float, float], ...]
    excluded: tuple[tuple[str, str], ...]
    status: str
    version: str = VERSION

    @property
    def fingerprint(self):
        payload = asdict(self)
        # Excluded/archival evidence is diagnostic, not a dependency of the
        # issued current expectation. New old/future rows must not change it.
        payload.pop('excluded')
        payload['policy'].pop('historical_days')
        return sha256(json.dumps(payload, sort_keys=True, default=str, allow_nan=False).encode()).hexdigest()


def current_quality_pace(history, *, as_of, bout_seconds, accounting_key, policy=QualityHistoryPolicy()):
    """Provisional conditional pace from actual bout means, never peak/max curves.

    Reuses phase_pace_reference's log-duration similarity and geometric averaging.
    No new fitted coefficients, exact-repeat gate, pace-at-HR extrapolation, or
    stale-history fallback. Bounds are session dispersion, not prediction limits.
    """
    aware(as_of)
    history = tuple(history)
    if not accounting_key or not isfinite(bout_seconds) or bout_seconds <= 0:
        raise ValueError('Accounting identity and positive queried bout duration required')
    if len({a.run.key for a in history}) != len(history):
        raise ValueError('Duplicate history identity')
    values, excluded = [], []
    for item in sorted(history, key=lambda a: (a.run.start, a.run.key)):
        run = item.run
        reason = ('not_observed' if run.source != 'observed' else
            'not_completed' if run.end >= as_of else 'not_available' if run.available_at > as_of else
            'accounting_mismatch' if run.accounting_key != accounting_key else
            'not_quality' if run.role != 'quality' else
            'outside_current_window' if run.start < as_of-timedelta(days=policy.current_days) else None)
        samples = [b for b in item.bouts if b.speed_mps is not None and b.speed_mps > 0]
        if reason or not samples:
            excluded.append((run.key, reason or 'no_complete_work_pace'))
            continue
        weights = [1/(1+log(b.seconds/bout_seconds)**2) for b in samples]
        values.append((run.key, geometric_mean([b.speed_mps for b in samples], weights),
                       fsum(weights)/len(weights)))
    pace = (Quantity(min(v for _, v, _ in values), max(v for _, v, _ in values),
        geometric_mean([v for _, v, _ in values], [w for _, _, w in values]),
        'recent_bout_mean_dispersion_not_prediction_interval') if values else None)
    return QualityPaceReference(as_of, bout_seconds, policy, accounting_key, pace, tuple(values), tuple(excluded),
        'provisional_recent_work_pace' if pace else 'no_recent_work_pace_explicit_forecast_needed')


def evaluate_quality_program(context, history, planned, signals, *, reference, **kwargs):
    """Single score handoff. Do not add a second load or target-match reward."""
    history, planned = tuple(history), tuple(planned)
    rows = (*history, *planned)
    if set(signals)-{r.key for r in rows}:
        raise ValueError('Unmatched signal identities')
    assessments = {r.key: assess_quality_session(r, signals.get(r.key), reference=reference,
        as_of=kwargs['as_of']) for r in rows}
    result = evaluate_continuous(context, history, planned, reference=reference, **kwargs)
    result['quality_component_version'] = VERSION
    result['quality_assessments'] = assessments
    result['quality_component_limitations'] = [
        'Supported work coverage is recurrence accounting, not physiological interchangeability',
        'Duration, bout/recovery structure, pace and HR remain separate descriptors',
        'Whole-session cardiovascular demand is retained once; target bands add no hidden benefit',
        'Forecast traces remain supplied scenarios; recent pace is a provisional estimate, not maximal output',
        'No calibrated benefit/recovery, automatic stage-intent selection or live deployment',
    ]
    return result
