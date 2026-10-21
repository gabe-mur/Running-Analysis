"""Explicit prospective rate revision, not delivery confirmation or a daily reset."""
from dataclasses import replace
from datetime import timedelta
from zoneinfo import ZoneInfo

from run_analysis.general_progression import RATE_VERSION, VOLUME_GROWTH_PER_7, destination
from run_analysis.stage_target_history import StageTarget
from run_analysis.staged_progression import tenth

SOURCE = 'prospective_volume_rate_amendment_v1'


def propose(stage, targets, as_of, timezone):
    """Retain today's exact dose, past path and original review; only change future slope."""
    if as_of.tzinfo is None or not stage.issued_at < as_of < stage.review_at:
        return dict(status='rate_amendment_requires_pending_phase', candidate=None)
    path = targets[-1]
    day = as_of.astimezone(ZoneInfo(timezone)).date()
    if (stage.policy.volume_path != 'gradual' or stage.policy.growth_focus != 'easy_volume'
            or stage.policy.balance is not None or not path.ramp_days or day <= path.effective_on):
        return dict(status='rate_amendment_requires_existing_daily_path', candidate=None)
    endpoint = path.effective_on + timedelta(days=path.ramp_days)
    remaining = (endpoint - day).days
    if remaining <= 0:
        return dict(status='rate_amendment_requires_pending_path', candidate=None)
    rate = (1 + VOLUME_GROWTH_PER_7) ** (stage.policy.review_days / 7) - 1
    if stage.policy.growth_fraction >= rate - 1e-10:
        return dict(status='volume_rate_already_applied', candidate=None)
    current = path.rate(day, 'volume_per_7')
    support = path.rate(day, 'supporting_easy_per_7')
    goal = min(destination(stage).volume_per_7,
               max(path.volume_goal, tenth(current * (1 + VOLUME_GROWTH_PER_7) ** (remaining / 7))))
    if goal <= path.volume_goal + 1e-8:
        return dict(status='rate_amendment_no_endpoint_increase', candidate=None)
    # No baseline rebootstrap and no claim that the pending endpoint was delivered.
    candidate = replace(stage, key=stage.key + ':rate:' + as_of.isoformat(),
        policy=replace(stage.policy, growth_fraction=rate, volume_goal_miles=goal), source=SOURCE)
    target = StageTarget(candidate.key + ':volume-path', as_of, day, current, support,
        SOURCE, remaining, goal, support + goal - current,
        review_anchor=path.review_anchor or path.available_at)
    return dict(status='conditional_rate_amendment', stage_key=stage.key,
        candidate=candidate, amended_target=target, original_targets=tuple(targets),
        original_review_at=stage.review_at, confirmed_delivery=False,
        policy=RATE_VERSION, volume_rate_per_7=VOLUME_GROWTH_PER_7)


def apply(harness, inputs, as_of, *, preview_check=None):
    """Caller explicitly authorizes policy revision; normal daily refresh never calls it."""
    from .adaptive_stage_lifecycle import accept_stage_program
    from run_analysis.recovery_response import assess_response
    race = getattr(harness, 'race_goal', None)
    if (harness.policy.general_progression != RATE_VERSION
            or race is not None and race.goal is not None):
        raise ValueError('Explicit general v2 policy required; race policy is unchanged')
    harness.sync_completions(as_of)
    if harness.current_context and harness.current_context.available_at <= as_of:
        if harness.current_context.state in {'health_concern', 'conflicting', 'little_tired'}:
            return harness, dict(status='rate_amendment_readiness_hold'), None
    for report in harness.responses:
        if harness.stage.issued_at <= report.available_at <= as_of and assess_response(
                report.expectation, effort=report.effort, feeling=report.feeling,
                feedback_conflict=report.conflict)['review_needed']:
            return harness, dict(status='rate_amendment_response_hold'), None
    review = propose(harness.stage, harness.target_history, as_of, harness.config['timezone_default'])
    if review['candidate'] is None:
        return harness, review, None
    return accept_stage_program(harness, inputs, as_of, review,
        amended_target=review['amended_target'], preview_check=preview_check)
