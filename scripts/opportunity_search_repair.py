"""Process-local bounded search repair; live production remains unchanged."""

from contextlib import contextmanager

from run_analysis import weekly_schedule as planner


@contextmanager
def opportunity_search_policy(enabled=True):
    token = planner._OPPORTUNITY_SEARCH_REPAIR.set(enabled)
    try:
        yield
    finally:
        planner._OPPORTUNITY_SEARCH_REPAIR.reset(token)
