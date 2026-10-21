"""Sourced date restrictions and conditional session-distance guard, not recovery."""
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal, ROUND_FLOOR
from math import inf, isfinite
from zoneinfo import ZoneInfo

from .athlete_projection import aware


@dataclass(frozen=True)
class DateRestriction:
    key: str
    start: date
    end: date  # exclusive; supplied scope, not inferred recovery duration
    available_at: datetime
    kind: str
    source: str

    def __post_init__(self):
        aware(self.available_at)
        if (not self.key or not self.source or type(self.start) is not date or type(self.end) is not date
                or self.end <= self.start or self.kind not in {'reserved_rest', 'health_review', 'unsafe_weather'}):
            raise ValueError('Sourced, scoped restriction required')


@dataclass(frozen=True)
class ReportedDistanceReference:
    """A dated maintained-distance declaration, NOT a completed run or load.

    Permits returning to the reported distance, not 10% growth above it. The
    ordinary 30-day reference lookback replaces it with real evidence over time.
    """
    key: str
    available_at: datetime
    reference_day: date
    miles: float

    def __post_init__(self):
        aware(self.available_at)
        if not self.key or type(self.reference_day) is not date or not isfinite(self.miles) or self.miles <= 0:
            raise ValueError('Dated positive self-reported distance reference required')


def _miles(event):
    low = high = 0.
    for t in event.terms:
        if t.channel != 'miles' or t.coefficient == 0:
            continue
        low += t.coefficient*t.bounds.low if t.bounds else 0.
        high += t.coefficient*t.bounds.high if t.bounds else inf
    return low, high


def _ceiling(value):
    return float((Decimal(str(value))*Decimal('1.1')).quantize(Decimal('.1'), rounding=ROUND_FLOOR))


def assess_program_constraints(planned, history, *, as_of, timezone='UTC', restrictions=(),
                               history_complete_since=None, plan_complete=False, decision_date=None,
                               reported_distance=None):
    """Known incomplete history cannot manufacture a definite distance violation.

    Planned predecessors are conditional references, never observed tolerance.
    An explicitly blocked/failing/unresolved predecessor supplies no headroom.
    No repetition limit, recovery clock, injury probability or growth approval.
    """
    aware(as_of)
    if reported_distance is not None and reported_distance.available_at > as_of:
        raise ValueError('Future reported distance cannot supply current headroom')
    today = as_of.astimezone(ZoneInfo(timezone)).date()
    if decision_date is not None:
        if type(decision_date) is not date or not today <= decision_date <= today+timedelta(days=1):
            raise ValueError('Decision scope must start today or after today completed work')
        # Advance only the planning boundary, never evidence availability. This
        # supports a same-day upload without pretending tomorrow's facts exist.
        today = decision_date
    planned = tuple(sorted(planned, key=lambda e: (e.day, e.key)))
    history, restrictions = tuple(history), tuple(restrictions)
    if type(plan_complete) is not bool:
        raise ValueError('Explicit future coverage assertion required')
    if history_complete_since is not None and (type(history_complete_since) is not date or history_complete_since > today):
        raise ValueError('Historical completeness must begin on or before the report date')
    if len({e.key for e in (*history, *planned)}) != len(history)+len(planned):
        raise ValueError('Duplicate actual/projected identity')
    if any(e.source != 'observed' or e.day >= today for e in history):
        raise ValueError('Prior observed history required; same-day sequencing unsupported')
    if any(e.source != 'planned' or e.day < today for e in planned):
        raise ValueError('Future planned entries required')
    if len({e.day for e in planned}) != len(planned):
        raise ValueError('Same-day multiple planned sessions unsupported')
    if len({r.key for r in restrictions}) != len(restrictions):
        raise ValueError('Duplicate restriction identity')
    known = [r for r in restrictions if r.available_at <= as_of]
    admitted, excluded, rows = list(history), [], []
    for event in planned:
        low, high = _miles(event)
        if low != high or not isfinite(low) or low <= 0 or abs(low*10-round(low*10)) > 1e-8:
            raise ValueError('Committed positive one-decimal planned distance required')
        start = event.day-timedelta(days=30)
        prior = [e for e in admitted if start <= e.day < event.day]
        prior_bounds = {e.key: _miles(e) for e in prior}
        lower_max = max((lo for lo, hi in prior_bounds.values()), default=0.)
        upper_max = max((hi for lo, hi in prior_bounds.values()), default=0.)
        excluded_prefix = [e.key for e in excluded if start <= e.day < event.day]
        complete = ((event.day == today or plan_complete) and not excluded_prefix and
                    (start >= today or history_complete_since is not None and history_complete_since <= start))
        ceiling_low = _ceiling(lower_max) if lower_max > 0 else None
        ceiling_high = _ceiling(upper_max) if isfinite(upper_max) and upper_max > 0 else None
        reported = (reported_distance if reported_distance is not None
            and start <= reported_distance.reference_day <= event.day else None)
        if reported is not None:
            ceiling_low = max(ceiling_low or 0., reported.miles)
            ceiling_high = max(ceiling_high or 0., reported.miles)
        distance_state = ('within_guard' if ceiling_low is not None and low <= ceiling_low+1e-8 else
                          'exceeds_guard' if complete and ceiling_high is not None and low > ceiling_high+1e-8 else
                          'unresolved_reference')
        blockers = [dict(key=r.key, kind=r.kind, source=r.source) for r in known if r.start <= event.day < r.end]
        status = ('constraint_conflict' if blockers or distance_state == 'exceeds_guard' else
                  'constraint_evidence_incomplete' if distance_state != 'within_guard' or excluded_prefix else
                  'passes_checked_constraints_conditionally')
        reference_keys = [e.key for e in prior if lower_max > 0 and prior_bounds[e.key][0] == lower_max]
        rows.append(dict(key=event.key, day=event.day, miles=low, status=status,
            restrictions=blockers, distance_guard=distance_state, lookback_start=start,
            lower_reference_miles=lower_max if lower_max else None,
            upper_reference_miles=upper_max if complete and isfinite(upper_max) and upper_max else None,
            committed_ceiling_miles=ceiling_low, reference_keys=reference_keys,
            projected_reference_keys=[e.key for e in prior if e.source == 'planned' and e.key in reference_keys],
            excluded_prefix_keys=excluded_prefix, reference_coverage_complete=complete,
            **({'reported_distance_reference':dict(key=reported.key,miles=reported.miles,
                basis='self_report_not_physical_history')} if reported is not None else {})))
        if status == 'passes_checked_constraints_conditionally':
            admitted.append(event)
        else:
            excluded.append(event)
    status = ('constraint_conflict' if any(r['status'] == 'constraint_conflict' for r in rows) else
              'constraint_evidence_incomplete' if not rows or any(r['status'] != 'passes_checked_constraints_conditionally' for r in rows) else
              'passes_checked_constraints_conditionally')
    return dict(status=status, sessions=rows, ignored_unavailable_restriction_keys=[r.key for r in restrictions if r not in known],
        recovery_feasibility='not_established', progression_accepted=False, running_authorized=False,
        planner_influence=False,
        limitations=['Only supplied restrictions are checked; absence is not verified normal health/weather',
            'Thirty local-date lookback convention; no same-day sequencing',
            'Projected predecessors are conditional, not demonstrated tolerance',
            'Distance guard does not constrain compound progression or establish recovery'])
