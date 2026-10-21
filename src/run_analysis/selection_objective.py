"""Offline shared selection objective and invocation-local prior prescriptions."""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from inspect import signature


SHARED_SELECTION_OBJECTIVE = ContextVar('shared_selection_objective', default=False)
PRIOR_DISTANCE_PRESCRIPTIONS = ContextVar('prior_distance_prescriptions', default=None)
FINALIZED_PROGRAM_SELECTOR = ContextVar('finalized_program_selector', default=None)
COMPLETED_SELECTION_HISTORY = ContextVar('completed_selection_history', default=None)
EXPECTED_COMPLETION_STATE = ContextVar('expected_completion_state', default=None)
VOLUME_PATH_SCORER = ContextVar('volume_path_scorer', default=None)
CONTINUATION_FUNDING_SCORER = ContextVar('continuation_funding_scorer', default=None)
RECOVERY_BACKEND = ContextVar('offline_recovery_backend', default=None)


def build_with_selection_history(build, template, completed_runs, *args, **kwargs):
    """Explicit realized-ledger handoff for offline daily/upload selection.

    Preserve simulator ``projected`` flags: they distinguish modeled response
    from measured response, not whether a run has been executed in the replay.
    The caller supplies its realized history, never its expected future ledger.
    No hook means precisely the existing build path, without copying history.
    """
    if FINALIZED_PROGRAM_SELECTOR.get() is None:
        return build(*args, **kwargs)
    token = COMPLETED_SELECTION_HISTORY.set({
        'template': template, 'runs': tuple(completed_runs), 'source': 'realized_simulation',
    })
    try:
        return build(*args, **kwargs)
    finally:
        COMPLETED_SELECTION_HISTORY.reset(token)


def prior_distance_prescriptions(schedule):
    days = (schedule.planning_days or schedule.days) if schedule is not None else []
    return {day.date: day.recommendation for day in days
            if day.recommendation and day.recommendation.workout_type != 'rest'
            and day.recommendation.distance_range_miles is not None}


@contextmanager
def shared_objective_policy(enabled=True, *, prior_schedule=None):
    token = SHARED_SELECTION_OBJECTIVE.set(enabled)
    prior_token = PRIOR_DISTANCE_PRESCRIPTIONS.set(prior_distance_prescriptions(prior_schedule))
    try:
        yield
    finally:
        PRIOR_DISTANCE_PRESCRIPTIONS.reset(prior_token)
        SHARED_SELECTION_OBJECTIVE.reset(token)


def scope_selection_prior(build):
    """Keep all scorers in one build on the same prior; restore on failure."""
    call_signature = signature(build)

    @wraps(build)
    def scoped(*args, **kwargs):
        if not SHARED_SELECTION_OBJECTIVE.get():
            return build(*args, **kwargs)
        bound = call_signature.bind(*args, **kwargs)
        with shared_objective_policy(prior_schedule=bound.arguments.get('prior_schedule')):
            return build(*args, **kwargs)

    return scoped
