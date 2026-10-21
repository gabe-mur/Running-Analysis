"""Offline capped linear-plus-quadratic volume contrasts, without publication.

Keep the accepted dated volume target and all demand/purpose/short-run costs.
Add a nonvanishing marginal volume value to the same rolling/handoff balances.
The coefficient is a declared policy experiment, not physiology or calibration.
"""
from math import isfinite
from types import SimpleNamespace

from .continuation_boundary_trial import continuation_boundary_cost
from .soft_horizon_trial import day_weight


def linear_volume_loss(context, evaluation):
    if context.policy.purpose_weights[1] != 0:
        raise ValueError('Volume-benefit trial requires one total-volume objective')
    if context.continuation_boundary_references:
        if 'linear_volume_loss' in evaluation.parts['continuation_boundary']:
            # Connected evaluation already computed this in its one boundary
            # pass. Never double the expensive continuation calculation.
            return evaluation.parts['continuation_boundary']['linear_volume_loss']
        # Reuse the actual dated means, handoffs and smoothing kernel. Independent
        # linear clipping per handoff matters: sqrt(mean squared losses) is wrong.
        source = SimpleNamespace(total_cost=evaluation.parts['pre_boundary_total_cost'],
            parts={k:v for k,v in evaluation.parts.items() if k!='continuation_boundary'},
            conditional_ledger=evaluation.conditional_ledger)
        result = continuation_boundary_cost(context, source,
            {role:(miles,demand) for role,miles,demand in context.continuation_boundary_references},
            include_linear_volume=True)
        assert abs(result['losses']['volume']-evaluation.parts['shortfall_by_purpose']['volume'])<1e-12
        return result['linear_volume_loss']
    if context.policy.volume_window_days is None:
        raise ValueError('Trial requires the dated rolling-volume accounting contract')
    span = (context.end-context.stage_start).days
    return sum((day_weight(context,r['day']) if context.soft_horizon_taper else 1.)
        * max(0.,1-(r['volume']['delivered']+r['volume'].get('foregone',0.))/r['volume']['target']) / span
        for r in evaluation.parts['rolling_volume']['daily'])


def volume_benefit_cost(context, evaluation, *, linear_coefficient, linear_loss=None):
    if context.volume_linear_coefficient:
        raise ValueError('Volume benefit already included by the shared evaluator; do not charge it twice')
    if not isfinite(linear_coefficient) or linear_coefficient<0:
        raise ValueError('Finite nonnegative explicit linear coefficient required')
    if context.policy.purpose_weights[1] != 0:
        raise ValueError('Volume-benefit trial requires one total-volume objective')
    linear = linear_volume_loss(context,evaluation) if linear_loss is None else linear_loss
    if not isfinite(linear) or linear<0:
        raise ValueError('Finite nonnegative linear volume loss required')
    added = linear_coefficient*linear*context.policy.purpose_weights[0]
    return dict(cost=evaluation.total_cost+added,baseline_cost=evaluation.total_cost,
        linear_coefficient=linear_coefficient,linear_volume_loss=linear,
        added_volume_cost=added,
        squared_volume_loss=evaluation.parts['shortfall_by_purpose']['volume'],
        delivery_cost=evaluation.delivery_cost+added,
        weighted_demand_cost=evaluation.weighted_demand_cost,short_run_cost=evaluation.short_run_cost,
        baseline_fingerprint=evaluation.fingerprint,
        model_version='single_volume_linear_benefit_trial_v1',
        scope='Fixed-candidate score experiment; not issued or selected by the daily scheduler')
