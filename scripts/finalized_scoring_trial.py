"""Opt-in offline finalized-program scoring; production defaults unchanged."""

from contextlib import contextmanager

from run_analysis import weekly_schedule as planner


@contextmanager
def finalized_scoring_policy(enabled=True):
    measurements = dict(evaluations=0, projection_seconds=0.0, coaching_seconds=0.0)
    token = planner._FINALIZED_SCORING_TRIAL.set(measurements if enabled else None)
    try:
        yield measurements
    finally:
        planner._FINALIZED_SCORING_TRIAL.reset(token)
