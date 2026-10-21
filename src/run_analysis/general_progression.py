"""Archived bounded proposals and shared general-fitness coaching constants.

The explicit continuous_general_v1 runtime uses
adaptive.continuous_general_progression, not next_general_phase.

The long/total allocation is a declared coaching development reference, not an
injury threshold or prerequisite for an independently supported single run.
Historical capacity supplies the starting stage, not the desired balance.
"""
from dataclasses import replace
from math import ceil

from .race_preparation import PreparationDestination, next_phase


LONG_SHARE = .4  # AS151 / AS119 replacement proposal; not the rejected AS118 curve.
VERSION = 'easy_first_coordinated_v1'
RATE_VERSION = 'easy_first_coordinated_v2'
VOLUME_GROWTH_PER_7 = .04  # AS152: coaching assumption, not a safety threshold.
LONG_GROWTH_PER_REVIEW = .05


def destination(stage):
    long = max(stage.long_miles, stage.goal_long_miles)
    pending = stage.policy.volume_goal_miles or stage.weekly_equivalent_miles
    volume = max(stage.weekly_equivalent_miles, pending,
                 ceil((long / LONG_SHARE - 1e-8) * 10) / 10)
    return PreparationDestination(volume, long, 'general_fitness:' + VERSION)


def next_general_phase(stage, *, as_of, longest_prior_30d_miles, version=VERSION):
    """Legacy bounded handoff; never infer delivery from a forecast.

    V1 retains the accepted rate. V2 authorizes a daily gradual volume path at
    a 4% seven-day-equivalent rate, with a separate 5% long review increment.
    The existing pending path is confirmed before this function is called.
    The caller must review actual delivery/response and atomically price/accept
    the whole programme before publishing any proposed ramp.
    """
    if version not in {VERSION, RATE_VERSION}:
        raise ValueError('Named general progression rate required')
    target = destination(stage)
    policy = stage.policy
    long_growth = None
    if version == RATE_VERSION:
        target = replace(target, source='general_fitness:' + version)
        policy = replace(policy, growth_fraction=(1 + VOLUME_GROWTH_PER_7) ** (policy.review_days / 7) - 1)
        long_growth = LONG_GROWTH_PER_REVIEW
    result = next_phase(stage, target, as_of=as_of,
                        longest_prior_30d_miles=longest_prior_30d_miles,
                        phase_policy=policy, long_growth_fraction=long_growth)
    candidate = result['candidate']
    if candidate is not None:
        candidate = replace(candidate, key=stage.key + ':general-phase:' + as_of.isoformat())
    return dict(result, candidate=candidate, destination=target,
                policy=version, long_share_reference=LONG_SHARE,
                volume_growth_fraction=policy.growth_fraction,
                long_growth_fraction=long_growth if long_growth is not None else policy.growth_fraction)
