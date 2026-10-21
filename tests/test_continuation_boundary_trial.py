from dataclasses import replace
from datetime import timedelta
from math import exp

import pytest

from test_steady_continuation_trial import fixture
from test_soft_horizon_trial import candidate
from run_analysis.adaptive.shared_program_evaluator import evaluate_program
from run_analysis.adaptive.continuation_boundary_trial import continuation_boundary_cost
from run_analysis.adaptive.soft_horizon_trial import day_weight


def setup():
    ctx, refs, rows = fixture()
    ctx = replace(ctx, soft_horizon_taper=True, delivery_contract='sustained_v1')
    return ctx, refs, rows, evaluate_program(ctx, candidate(ctx, rows))


def test_continuation_matches_scalar_volume_and_cross_demand_reference():
    ctx, refs, rows, ev = setup()
    result = continuation_boundary_cost(ctx, ev, refs)
    for scenario in result['scenarios']:
        boundary = scenario['handoff']
        prefix = tuple(r for r in rows if r.day < boundary)
        for purpose, rate in (('volume', 18.), ('ordinary_easy', 6.)):
            expected = 0.
            for n in range((boundary - ctx.decision).days + 33):
                day = ctx.decision + timedelta(days=n)
                observations = []
                for k in range(7):
                    end = day - timedelta(days=k)
                    begin = end - timedelta(days=27)
                    value = sum(r.miles for r in prefix if begin <= r.day <= end
                                and (purpose == 'volume' or r.role == 'easy'))
                    value += max(0, (end - max(begin, boundary)).days + 1) * rate / 7
                    observations.append(value)
                expected += max(0, 1 - sum(observations) / 7 / (rate * 4)) ** 2 / 35
            assert scenario['losses'][purpose] == pytest.approx(expected, abs=1e-12)
        daily = (2 + 1 + .75 * 6 / 3) / 7
        explicit = sum(sum((r.miles / 4) * daily * exp(-((boundary - r.day).days + d) / 2)
                           for r in prefix) for d in range(200)) * 7 / 35
        assert scenario['cross_demand_by_axis'] == pytest.approx([explicit] * 3, abs=1e-12)


def test_handoff_probability_matches_existing_taper_without_randomness():
    ctx, refs, _, ev = setup()
    a = continuation_boundary_cost(ctx, ev, refs)
    assert a == continuation_boundary_cost(ctx, ev, refs)
    dates = [s['handoff'] for s in a['scenarios']]
    for offset in range(36):
        day = ctx.decision + timedelta(days=offset)
        assert sum(day < b for b in dates) / len(dates) == pytest.approx(day_weight(ctx, day))


def test_empty_plan_keeps_positive_volume_deficit_and_deletion_cannot_improve_volume():
    ctx, refs, rows, ev = setup()
    full = continuation_boundary_cost(ctx, ev, refs)
    empty = continuation_boundary_cost(ctx, evaluate_program(ctx, candidate(ctx, ())), refs)
    assert empty['volume_cost'] > full['volume_cost'] > 0
    assert empty['cross_demand_cost'] == 0
    fewer = evaluate_program(ctx, candidate(ctx, tuple(r for r in rows if r.role != 'easy')))
    assert continuation_boundary_cost(ctx, fewer, refs)['volume_cost'] > full['volume_cost']


def test_no_physical_credit_or_stage_rewrite_and_only_named_costs_replaced():
    ctx, refs, rows, ev = setup()
    result = continuation_boundary_cost(ctx, ev, refs)
    assert result['cost'] == pytest.approx(ev.total_cost - result['previous_volume_cost']
                                           + result['volume_cost'] + result['cross_demand_cost'])
    assert tuple(ev.conditional_ledger) == tuple(sorted(rows, key=lambda r: (r.day, r.key)))
    assert not result['activated_by_default']
    with pytest.raises(ValueError, match='sustained'):
        continuation_boundary_cost(replace(ctx, delivery_contract='aggregate_v1'), ev, refs)


def test_exact_completion_keeps_pairwise_ranking_with_fixed_handoff_dates():
    ctx, refs, rows, _ = setup()
    first = rows[0]
    completed = replace(first, source='observed')
    after_context = replace(ctx, decision=ctx.decision + timedelta(days=1), history=(completed,))
    scores = []
    for shift in (0, 1):
        program = tuple(replace(r, day=r.day + timedelta(days=shift))
                        if r.key == '5' else r for r in rows)
        before = evaluate_program(ctx, candidate(ctx, program))
        after = evaluate_program(after_context, candidate(after_context, program[1:]))
        scores.append((continuation_boundary_cost(ctx, before, refs)['cost'],
                       continuation_boundary_cost(after_context, after, refs)['cost']))
    assert scores[0][0] - scores[1][0] == pytest.approx(scores[0][1] - scores[1][1], abs=1e-12)


def test_realized_prefix_cannot_borrow_a_later_handoff_prescription():
    ctx, refs, rows, ev = setup()
    base = continuation_boundary_cost(ctx, ev, refs)
    boundary = base['scenarios'][0]['handoff']
    altered = tuple(replace(r, miles=r.miles + 10) if r.day >= boundary else r for r in rows)
    changed = continuation_boundary_cost(ctx, evaluate_program(ctx, candidate(ctx, altered)), refs)
    assert changed['scenarios'][0] == base['scenarios'][0]


def test_dated_targets_extend_final_rate_without_backdating_or_filling_gaps():
    from run_analysis.program_delivery import TargetSegment
    ctx, refs, rows, _ = setup()
    start = ctx.decision - timedelta(days=33)
    dated = replace(ctx, volume_targets=(TargetSegment(start, ctx.decision, 9.),
        TargetSegment(ctx.decision, ctx.end, 18.)),
        support_targets=(TargetSegment(start, ctx.end, 6.),))
    constant = replace(dated, volume_targets=(TargetSegment(start, ctx.end, 18.),))
    a = continuation_boundary_cost(dated, evaluate_program(dated, candidate(dated, rows)), refs)
    b = continuation_boundary_cost(constant, evaluate_program(constant, candidate(constant, rows)), refs)
    assert a['losses']['volume'] < b['losses']['volume']
    assert a['losses']['ordinary_easy'] == b['losses']['ordinary_easy']
    incomplete = replace(dated, volume_targets=dated.volume_targets[1:])
    with pytest.raises(ValueError, match='coverage'):
        continuation_boundary_cost(incomplete, evaluate_program(dated, candidate(dated, rows)), refs)


@pytest.mark.parametrize('exchange',[0.,.01,.5])
def test_persisted_contract_matches_prototype_without_double_charging(exchange):
    ctx,refs,rows,_=setup()
    ctx=replace(ctx,exchange=exchange)
    base=evaluate_program(ctx,candidate(ctx,rows))
    expected=continuation_boundary_cost(ctx,base,refs)
    integrated=replace(ctx,continuation_boundary_references=tuple(
        (role,miles,demand) for role,(miles,demand) in refs.items()))
    result=evaluate_program(integrated,candidate(integrated,rows))
    assert result.total_cost==pytest.approx(expected['cost'],abs=1e-12)
    assert result.total_cost==pytest.approx(result.delivery_cost+result.weighted_demand_cost+result.short_run_cost,abs=1e-12)
    assert result.conditional_ledger==base.conditional_ledger
    assert result.parts['delivery']==base.parts['delivery']
    assert result.parts['targets']==base.parts['targets']
    assert result.fingerprint!=base.fingerprint
    assert result.downstream_accounting['status']=='distributed_mean_continuation'
    with pytest.raises(ValueError,match='twice'):
        continuation_boundary_cost(integrated,result,refs)


def test_boundary_contract_rejects_incomplete_or_incompatible_references():
    ctx,refs,_,_=setup()
    values=tuple((role,miles,demand) for role,(miles,demand) in refs.items())
    with pytest.raises(ValueError,match='Three positive'):
        replace(ctx,continuation_boundary_references=values[:-1])
    with pytest.raises(ValueError,match='sustained/tapered'):
        replace(ctx,soft_horizon_taper=False,continuation_boundary_references=values)


def test_continuation_prices_dated_growth_in_both_delivery_and_future_demand():
    from run_analysis.stage_target_history import StageTarget,target_segments
    from datetime import datetime,time,timezone
    ctx,refs,rows,_=setup()
    at=datetime.combine(ctx.decision,time.min,timezone.utc)
    seed=StageTarget('seed',at,ctx.decision-timedelta(days=40),18.,6.,'seed')
    ramp=StageTarget('ramp',at,ctx.decision,18.,6.,'ramp',35,21.5,9.5)
    volume,support=target_segments((seed,ramp),as_of=at,start=seed.effective_on,end=ctx.end)
    dated=replace(ctx,volume_targets=volume,support_targets=support)
    result=continuation_boundary_cost(dated,evaluate_program(dated,candidate(dated,rows)),refs)
    flat=continuation_boundary_cost(ctx,evaluate_program(ctx,candidate(ctx,rows)),refs)
    assert result['cross_demand_cost']>flat['cross_demand_cost']
    # Last declared rate extends beyond coverage; never revert to policy defaults.
    assert result['expected_miles_per_7']==pytest.approx(volume[-1].miles_per_7_days)
    first=result['scenarios'][0]
    boundary=first['handoff']
    prefix=tuple(r for r in rows if r.day<boundary)
    from run_analysis.program_delivery import _target_total
    def rate(day):
        return _target_total(support,day,day+timedelta(days=1))*7 if day<ctx.end else support[-1].miles_per_7_days
    expected=sum(sum((r.miles/4)*(2+1+.75*rate(boundary+timedelta(days=d))/3)/7*
        exp(-((boundary-r.day).days+d)/2) for r in prefix) for d in range(200))*7/35
    assert first['cross_demand_by_axis']==pytest.approx([expected]*3,abs=1e-12)
