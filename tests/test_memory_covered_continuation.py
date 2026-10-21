from dataclasses import replace
from datetime import timedelta

import pytest

from test_continuation_boundary_trial import setup
from test_soft_horizon_trial import candidate
from run_analysis.adaptive.shared_program_evaluator import evaluate_program,MEMORY_CONTINUATION_MODEL_VERSION
from run_analysis.adaptive.continuation_boundary_trial import continuation_boundary_cost
from run_analysis.adaptive.soft_horizon_trial import day_weight


def guarded():
    ctx,refs,rows,_=setup()
    ctx=replace(ctx,continuation_guard_days=7,end=ctx.decision+timedelta(days=54),
        continuation_boundary_references=tuple((k,m,d) for k,(m,d) in refs.items()))
    # Identical explicit future shared by both near-prefix alternatives.
    tail=tuple(replace(r,key='tail:'+r.key,day=r.day+timedelta(days=21))
        for r in rows if 14<= (r.day-ctx.decision).days <33)
    return ctx,refs,(*rows,*tail)


def test_memory_boundary_and_survival_weights_cover_visible_prefix():
    ctx,refs,rows=guarded()
    assert (ctx.taper_start-ctx.decision).days==40
    result=evaluate_program(ctx,candidate(ctx,rows))
    assert result.model_version==MEMORY_CONTINUATION_MODEL_VERSION
    handoffs=[r['handoff'] for r in result.parts['continuation_boundary']['scenarios']]
    assert len(handoffs)==14
    for n in range(55):
        day=ctx.decision+timedelta(days=n)
        assert sum(day<h for h in handoffs)/14==pytest.approx(day_weight(ctx,day))
    for n in range(7):
        expires=ctx.decision+timedelta(days=n+33)
        assert expires<min(handoffs)
    assert result.total_cost==pytest.approx(result.delivery_cost+result.weighted_demand_cost+result.short_run_cost)


def test_horizon_roll_cannot_change_visible_prefix_volume_difference():
    ctx,refs,rows=guarded()
    # Fixed evidence, dates, doses and downstream; roll ONLY horizon endpoints.
    alternative=tuple(replace(r,miles=r.miles+.5) if r.key=='5' else r for r in rows)
    deltas=[]
    for shift in (0,1,3):
        c=replace(ctx,hard_end=ctx.hard_end+timedelta(days=shift),end=ctx.end+timedelta(days=shift))
        # Decision also rolls in production; moving taper scenarios farther is
        # the isolated variable here, including its changed common denominator.
        c=replace(c,continuation_guard_days=7+shift)
        evaluations=[evaluate_program(c,candidate(c,p)) for p in (rows,alternative)]
        span=(c.end-c.stage_start).days
        deltas.append({k:span*(evaluations[1].parts['shortfall_by_purpose'][k]-
            evaluations[0].parts['shortfall_by_purpose'][k]) for k in ('volume','ordinary_easy')})
    for delta in deltas[1:]:
        assert delta==pytest.approx(deltas[0],abs=1e-12)


def test_no_volume_reward_for_deleting_real_tail_work_and_future_cost_remains():
    ctx,refs,rows=guarded()
    full=evaluate_program(ctx,candidate(ctx,rows))
    sparse=evaluate_program(ctx,candidate(ctx,tuple(r for r in rows if r.day<ctx.hard_end)))
    assert sparse.parts['shortfall_by_purpose']['volume']>full.parts['shortfall_by_purpose']['volume']
    assert full.parts['continuation_boundary']['cross_demand_cost']>0
    assert any(r['day']>=ctx.hard_end for r in full.parts['session_costs'])


def test_insufficient_projection_rejected_not_silently_filled_with_mean_work():
    ctx,_,_,_=setup()
    with pytest.raises(ValueError,match='prefix memory'):
        replace(ctx,continuation_guard_days=7)
    with pytest.raises(ValueError,match='Memory-covered'):
        replace(ctx,continuation_guard_days=True)


@pytest.mark.parametrize('window',[14,28,42])
@pytest.mark.parametrize('cadence_days',[7,8])
def test_coverage_is_derived_from_accounting_not_a_fixed_forty_day_rule(window,cadence_days):
    ctx,_,_,_=setup()
    expected=max(21,7+window+cadence_days-2)
    ctx=replace(ctx,policy=replace(ctx.policy,volume_window_days=window,
        long_per_7=7/cadence_days,quality_per_7=7/cadence_days),
        continuation_guard_days=7,end=ctx.decision+timedelta(days=expected+14))
    assert (ctx.taper_start-ctx.decision).days==expected
    assert (ctx.hard_end-ctx.decision).days==21


def test_compliant_completion_preserves_volume_contrast_with_rolling_horizon():
    ctx,refs,rows=guarded()
    first=rows[0]
    later=replace(ctx,decision=ctx.decision+timedelta(days=1),
        hard_end=ctx.hard_end+timedelta(days=1),end=ctx.end+timedelta(days=1),
        history=(replace(first,source='observed'),))
    changes=[]
    for c in (ctx,later):
        plans=[]
        for extra in (0.,.5):
            p=tuple(replace(r,miles=r.miles+extra) if r.key=='5' else r
                for r in rows if r.day>=c.decision)
            plans.append(evaluate_program(c,candidate(c,p)))
        changes.append((plans[1].parts['shortfall_by_purpose']['volume']-
            plans[0].parts['shortfall_by_purpose']['volume'])*(c.end-c.stage_start).days)
    assert changes[0]==pytest.approx(changes[1],abs=1e-12)
