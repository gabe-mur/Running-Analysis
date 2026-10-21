"""History-independent expectation contract; NOT a calibrated recovery model.

The caller supplies named response assumptions, which may come from the shared
projection API. No personal-history maximum, three-session switch, HRV prerequisite
or automatic learned coefficient. Missing assumptions remain explicit unknowns.
"""
from copy import deepcopy

from .workload_accounting import Quantity
from .workload_expectation import Intent, _utc


INTENT_KEYS = ('distance_meters', 'duration_minutes', 'work_seconds',
              'longest_work_seconds', 'work_distance_meters', 'longest_work_distance_meters')
RESPONSE_KEYS = {'duration_minutes', 'cardio_points', 'work_seconds', 'longest_work_seconds',
                 *(f'{signal}_{w}s' for signal in ('hr', 'speed') for w in (30, 60, 120, 300, 600))}


def baseline_expectation(intent: Intent, *, issued_at, assumptions, assumption_source):
    """Assemble an issued expectation without requiring past matching workouts.

    Explicit prescription quantities take precedence over response assumptions.
    A missing pace/HR assumption is unknown, not zero or an invented population
    constant. This boundary defines behavior; it does not choose numeric priors.
    """
    if not isinstance(assumption_source, str) or not assumption_source.strip():
        raise ValueError('Named baseline assumption source required')
    if set(assumptions)-RESPONSE_KEYS:
        raise ValueError('Unrecognized response assumption')
    if any(value is not None and not isinstance(value, Quantity) for value in assumptions.values()):
        raise ValueError('Response assumptions must be bounded quantities or unknown')
    expected = {key: None for key in sorted(RESPONSE_KEYS)}
    expected.update(assumptions)
    for key in INTENT_KEYS:
        value = getattr(intent, key)
        if value is not None or key not in expected:
            expected[key] = value
    duration, work = expected['duration_minutes'], expected['work_seconds']
    if duration is not None and work is not None and work.low > duration.high*60:
        raise ValueError('Session duration cannot accommodate minimum work time')
    for total, longest in (('work_seconds', 'longest_work_seconds'),
                           ('work_distance_meters', 'longest_work_distance_meters'),
                           ('distance_meters', 'work_distance_meters')):
        if expected[total] is not None and expected[longest] is not None:
            if expected[longest].low > expected[total].high:
                raise ValueError('Part of session exceeds its total')
    return dict(issued_at=_utc(issued_at), expected=expected,
        assumption_source=assumption_source, include_inferred_work=True,
        calibrated=False, personalization='none', adaptation_authorized=False,
        planner_influence=False,
        limitations=['Numeric assumptions are supplied explicitly, not validated by this contract',
                     'Unknown response is not zero load, compliance, recovery clearance or a rest order'])


def attach_historical_context(baseline, historical):
    """Keep descriptive matching separate from the authoritative baseline.

    Even many observations do not validate a personalization method by count
    alone. Conflicts are surfaced, not averaged away or used to move a run.
    A future validated updater can replace this deliberately neutral seam.
    """
    if _utc(baseline['issued_at']) != _utc(historical['issued_at']):
        raise ValueError('Baseline and history must share the same information cutoff')
    context = {}
    for key in sorted(set(baseline['expected']) | set(historical['expected'])):
        prior, empirical = baseline['expected'].get(key), historical['expected'].get(key)
        if prior is None or empirical is None:
            status = 'unknown'
        elif 'unquantified' in empirical.basis:
            status = 'inferred_context_only'
        elif empirical.low > prior.high:
            status = 'above_baseline'
        elif empirical.high < prior.low:
            status = 'below_baseline'
        else:
            status = 'overlapping'
        context[key] = dict(status=status, baseline=prior, empirical=empirical,
            reference_keys=list(historical.get('support', {}).get(key, ())))
    return dict(expectation=deepcopy(baseline), context=context,
        personalization_applied=False, adaptation_authorized=False, planner_influence=False,
        reason='Historical envelopes are diagnostic only; no validated personalization updater')
