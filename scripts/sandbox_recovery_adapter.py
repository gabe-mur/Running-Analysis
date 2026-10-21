"""Frozen recovery contracts for offline saved-program audits, not production.

Reuse ExpectedCompletion for projected running duration. HR phase assumptions
remain explicit; synthetic historical runs without HR do not become measured
cardio or confirmed work just because their legacy difficulty is available.
"""
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from hashlib import sha256
import json

from run_analysis.recovery_assessment import SessionEntry, replace_expected
from run_analysis.response_dose_review import DoseOption, review_next_dose
from run_analysis.recovery_response import ResponseExpectation
from run_analysis.workload_accounting import Quantity, Workload, projected_workload
from run_analysis.workload_profile import prescribed_profile
from scripts.typed_continuation_prototype import runs


@dataclass(frozen=True)
class PhaseAssumptions:
    name: str
    base: Quantity
    work: Quantity
    recovery: Quantity

    def __post_init__(self):
        if not self.name.strip() or any(q.high > 5 for q in (self.base, self.work, self.recovery)):
            raise ValueError('Named phase-rate assumptions in 0–5 points/minute required')


def session_key(at):
    # This saved sandbox has one session per timestamp. Production integration
    # must use persistent prescription/activity associations, not date matching.
    return 'sandbox:' + at.isoformat()


def history_entry(run):
    """Known synthetic distance/time, unknown HR/work; not inferred from labels."""
    duration = Quantity.point(run.moving_minutes, 'synthetic execution duration')
    workload = Workload('observed', Quantity.point(run.distance_miles*1609.344,
        'synthetic execution distance'), duration,
        Quantity(0, 5*run.moving_minutes, basis='synthetic history has no HR stream'),
        {}, ('Synthetic completed history, not athlete measurements',
             'No numeric work structure retained in ProjectionRun'))
    return SessionEntry(session_key(run.start_time), run.start_time,
        run.start_time+timedelta(minutes=run.moving_minutes), workload,
        Quantity(0, run.moving_minutes, basis='unknown work bounded by duration'))


def projected_entry(rec, run, assumptions):
    profile = prescribed_profile(rec)
    duration = Quantity.point(run.moving_minutes, 'existing expected-completion duration')
    recovery = (assumptions.recovery
                if profile['intensity_structure']['between_work_recovery_seconds'] is not None else None)
    workload = projected_workload(profile, duration_minutes=duration,
        base_rate=assumptions.base, work_rate=assumptions.work, recovery_rate=recovery)
    bouts = profile['intensity_structure']['bouts']
    if 'work_duration_missing' in profile['limitations']:
        work = Quantity(0, duration.high, basis='unresolved prescription work')
    else:
        bounds = [b['duration_range_seconds'] or (b['work_seconds'], b['work_seconds']) for b in bouts]
        work = Quantity(sum(b[0] for b in bounds)/60, sum(b[1] for b in bounds)/60,
                        basis='issued numeric work duration')
    return SessionEntry(session_key(rec.planned_for), rec.planned_for,
        rec.planned_for+timedelta(minutes=run.moving_minutes), workload, work)


def _quantity(value):
    return Quantity(**value) if value is not None else None


def _entry(value):
    w = value['workload']
    return SessionEntry(value['key'], datetime.fromisoformat(value['start']),
        datetime.fromisoformat(value['end']), Workload(w['source'],
        *(_quantity(w[n]) for n in ('distance_meters', 'running_minutes', 'cardio_points')),
        w['structure'], tuple(w['limitations'])), _quantity(value['work_minutes']))


@dataclass(frozen=True)
class IssuedRecoverySnapshot:
    # Serialized payload avoids shallow-frozen mutable structure dictionaries
    # or Pydantic recommendations changing underneath an issued expectation.
    payload: str

    @property
    def digest(self):
        return sha256(self.payload.encode()).hexdigest()

    def entries(self):
        return tuple(_entry(e) for e in json.loads(self.payload)['entries'])

    def recommendation(self, key):
        from run_analysis.web.schemas import RecommendationResponse
        return RecommendationResponse.model_validate(json.loads(self.payload)['prescriptions'][key])

    def option(self, key, *, option_key='issued'):
        rec = self.recommendation(key)
        entry = next(e for e in self.entries() if e.key == key)
        return DoseOption(option_key, key, rec.planned_for, rec.workout_type.value,
                          rec.planned_distance_miles, entry.workload, entry.work_minutes)

    def expected_response(self, key):
        value = json.loads(self.payload).get('response_expectations', {}).get(key)
        return (ResponseExpectation(value['source'], _quantity(value['effort']), _quantity(value['feeling']))
                if value is not None else ResponseExpectation('No response expectation in issued snapshot'))

    def prefix(self, next_key):
        entries = self.entries()
        at = next(e.start for e in entries if e.key == next_key)
        prior = tuple(e for e in entries if e.start < at)
        if any(e.end > at for e in prior):
            raise ValueError('Overlapping/unfinished sessions cannot form a recovery prefix')
        return prior

    def after_completion(self, completion):
        return replace_expected(self.entries(), completion)


def capture(model, program, *, issued_at, assumptions, expected_responses=None):
    if issued_at.tzinfo is None or issued_at.utcoffset() is None:
        raise ValueError('Timezone-aware issue cutoff required')
    # The native projector caches by object identity as well as physical key.
    # Own copies avoid stale identity entries if a caller edits a mutable rec
    # before taking a new snapshot; already-issued payloads stay untouched.
    prescriptions = [r.model_copy(deep=True) for r in runs(program)]
    if not prescriptions or any(r.planned_for is None or r.planned_for < issued_at for r in prescriptions):
        raise ValueError('Only unconsumed, timed prescriptions can be issued')
    native = model.ledger(max(r.planned_for for r in prescriptions)+timedelta(days=1), prescriptions)
    by_time = {r.start_time: r for r in native}
    completed = tuple(history_entry(r) for r in model.completed)
    if any(e.end > issued_at for e in completed):
        raise ValueError('Completed history is unavailable at issue cutoff')
    entries = (*completed, *(projected_entry(r, by_time[r.planned_for], assumptions) for r in prescriptions))
    return freeze_entries(prescriptions, entries, issued_at=issued_at,
                          assumptions=asdict(assumptions), expected_responses=expected_responses)


def freeze_entries(prescriptions, entries, *, issued_at, assumptions, expected_responses=None):
    """Shared snapshot boundary for native projections and HR-backed scenarios."""
    entries, prescriptions = tuple(entries), tuple(prescriptions)
    if issued_at.tzinfo is None or issued_at.utcoffset() is None:
        raise ValueError('Timezone-aware issue cutoff required')
    if len({e.key for e in entries}) != len(entries):
        raise ValueError('Duplicate session identity in frozen ledger')
    projected = {e.key: e for e in entries if e.workload.source == 'projected'}
    if set(projected) != {session_key(r.planned_for) for r in prescriptions}:
        raise ValueError('Projected ledger and issued prescriptions must match exactly')
    if any(e.end > issued_at for e in entries if e.workload.source == 'observed'):
        raise ValueError('Observed history must be available before issue')
    if any(r.planned_for < issued_at or projected[session_key(r.planned_for)].start != r.planned_for
           for r in prescriptions):
        raise ValueError('Issued forecasts must use unconsumed prescription timestamps')
    expected_responses = expected_responses or {}
    if not set(expected_responses) <= {session_key(r.planned_for) for r in prescriptions}:
        raise ValueError('Response expectation must belong to an issued session')
    payload = dict(version='sandbox_recovery_snapshot_v2', issued_at=issued_at.isoformat(),
        assumptions=assumptions, entries=[asdict(e) for e in entries],
        prescriptions={session_key(r.planned_for): r.model_dump(mode='json') for r in prescriptions},
        response_expectations={k: asdict(v) for k, v in expected_responses.items()},
        planner_influence=False)
    return IssuedRecoverySnapshot(json.dumps(payload, default=lambda x: x.isoformat(),
                                            sort_keys=True, allow_nan=False))


def projection_conflicts(model, program, *, assumptions):
    """Expose incompatible native forecasts without padding or rewriting them."""
    prescriptions = [r.model_copy(deep=True) for r in runs(program)]
    if not prescriptions:
        return []
    native = {r.start_time: r for r in model.ledger(
        max(r.planned_for for r in prescriptions)+timedelta(days=1), prescriptions)}
    conflicts = []
    for rec in prescriptions:
        try:
            projected_entry(rec, native[rec.planned_for], assumptions)
        except ValueError as error:
            profile = prescribed_profile(rec)
            conflicts.append(dict(at=rec.planned_for.isoformat(), target=rec.planned_distance_miles,
                native_minutes=native[rec.planned_for].moving_minutes,
                structured_minimum_minutes=profile['running_exposure']['minimum_duration_range_seconds'][1]/60,
                reason=str(error)))
    return conflicts


def review_completion(snapshot, completion, next_key, *, alternatives, eligible_keys,
                      policies, minimum_target_miles, adjustment_source,
                      **response_inputs):
    """Only full-program-checked variants may reach the bounded response helper.

    The caller certifies eligibility; menu membership is not sufficient. This
    adapter does not waive recovery or non-recovery gates during the experiment.
    """
    alternatives = tuple(alternatives)
    if any(o.key not in eligible_keys for o in alternatives):
        raise ValueError('Response alternatives require full-program eligibility, not just menu membership')
    expected = snapshot.prefix(next_key)
    if completion.key not in {e.key for e in expected}:
        raise ValueError('Completion must precede the issued next session')
    # For upload review, only the latest completed run may be replaced; future
    # intervening projections cannot masquerade as already completed history.
    if any(e.workload.source == 'projected' and e.key != completion.key for e in expected):
        raise ValueError('Unconsumed intervening forecasts cannot become observed history')
    actual = replace_expected(expected, completion)
    return review_next_dose(issued=snapshot.option(next_key), alternatives=alternatives,
        history=actual, expected_history=expected, policies=policies,
        report_activity_key=completion.key, expected_response=snapshot.expected_response(completion.key),
        minimum_target_miles=minimum_target_miles, adjustment_source=adjustment_source,
        **response_inputs)
