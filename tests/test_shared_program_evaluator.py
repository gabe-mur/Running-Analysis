from dataclasses import replace
from datetime import date, timedelta

import pytest

from run_analysis.workload_accounting import Quantity
from scripts.program_objective_trial import DemandScenario, TrialPolicy, TrialSession, components, score
from scripts.program_objective_trial import EasyDurationPolicy
from scripts.shared_program_evaluator import (
    ProgramCandidate, ProgramContext, evaluate_program,
)


START = date(2026, 9, 26)


def day(offset):
    return START + timedelta(days=offset)


def session(key, offset, miles=4.4, role='easy', source='planned'):
    return TrialSession(key, day(offset), miles, role, (
        Quantity.point(miles * 10, 'test duration'),
        Quantity.point(miles * 20, 'test cardio'),
        Quantity.point(15 if role == 'quality' else 0, 'test work'),
    ), quality_credit=1. if role == 'quality' else 0., source=source)


def context(**updates):
    values = dict(history=(session('actual', -2, miles=8.453, role='long', source='observed'),),
        stage_start=START, decision=START, hard_end=day(21), end=day(35),
        scales=(44., 88., 15.), policy=TrialPolicy(), scenario=DemandScenario(), exchange=.1)
    values.update(updates)
    return ProgramContext(**values)


def candidate():
    return ProgramCandidate(
        (session('first', 0), session('second', 3, role='quality'), session('third', 19)),
        (session('tail', 21, miles=8.4, role='long'),), 'explicit test continuation')


def test_easy_duration_curve_is_finite_local_and_does_not_reward_longer_runs():
    p = EasyDurationPolicy(10., 20., .02, 'test assumption')
    assert [p.penalty(t) for t in (0, 10, 15, 20, 40)] == pytest.approx([.02, .02, .005, 0, 0])
    assert p.penalty(19.999) < p.penalty(19.99)
    for values in ((10, 10, .01), (0, 20, .01), (10, 20, -.01), (10, float('inf'), .01)):
        with pytest.raises(ValueError):
            EasyDurationPolicy(*values, 'invalid')


def test_easy_curve_replaces_ordinary_credit_and_preserves_physical_accounting():
    ctx, plan = context(), candidate()
    plan = replace(plan, hard=(session('small', 0, miles=1.5), *plan.hard[1:]))
    old = evaluate_program(ctx, plan)
    new = evaluate_program(replace(ctx, easy_duration_policy=EasyDurationPolicy(10, 20, .02, 'test')), plan)
    assert new.parts['delivery'] == old.parts['delivery']
    assert new.parts['demand_cost'] == old.parts['demand_cost']
    assert new.parts['shortfall_by_purpose']['ordinary_easy'] == 0
    assert new.short_run_cost == pytest.approx(.005*7/35)
    assert new.total_cost == pytest.approx(new.delivery_cost+new.weighted_demand_cost+new.short_run_cost)
    assert new.fingerprint != old.fingerprint
    assert new.total_cost == score(new.parts, ctx.exchange)
    assert not new.selection_ready


def test_minimum_is_forecast_eligibility_not_deletion_of_completed_training():
    p = EasyDurationPolicy(10, 20, .02, 'test')
    actual = session('short_actual', 0, .5, source='observed')
    below = session('short_plan', 2, .5)
    uncertain = replace(session('uncertain', 3, 1.), demand=(Quantity(8, 12, basis='test'),
        Quantity.point(20, 'test'), Quantity.point(0, 'test')))
    equal = session('at_minimum', 4, 1.)
    ctx = context(history=(actual,), decision=day(1), easy_duration_policy=p)
    result = evaluate_program(ctx, ProgramCandidate((below, uncertain, equal), (), 'rest tail'))
    details = result.parts['easy_duration']
    assert details['below_minimum_keys'] == ('short_plan',)
    assert details['uncertain_minimum_keys'] == ('uncertain',)
    assert result.parts['delivery']['volume'] == pytest.approx(3.)
    assert details['cost_bounds'][0] <= details['cost'] <= details['cost_bounds'][1]


def test_easy_curve_completion_and_scope_extension_preserve_relative_comparisons():
    ctx = context(easy_duration_policy=EasyDurationPolicy(10, 20, .02, 'test'))
    first = session('first', 0, 1.5)
    a = ProgramCandidate((first, session('next', 3, 2.5)), (), 'rest tail')
    b = ProgramCandidate((first, session('next', 4, 2.5)), (), 'rest tail')
    before = [evaluate_program(ctx, p) for p in (a, b)]
    after_ctx = replace(ctx, history=(*ctx.history, replace(first, source='observed')), decision=day(1))
    after = [evaluate_program(after_ctx, replace(p, hard=p.hard[1:])) for p in (a, b)]
    assert before[0].total_cost-before[1].total_cost == pytest.approx(after[0].total_cost-after[1].total_cost)
    assert before[0].short_run_cost == after[0].short_run_cost
    extended = evaluate_program(replace(ctx, downstream_end=day(42), downstream_basis='explicit rest'), a)
    assert extended.total_cost == before[0].total_cost


def test_no_extra_easy_run_credit_above_penalty_band():
    ctx = context(easy_duration_policy=EasyDurationPolicy(10, 20, .02, 'test'))
    merged = evaluate_program(ctx, ProgramCandidate((session('a', 0, 5.),), (), 'rest tail'))
    split = evaluate_program(ctx, ProgramCandidate((session('a', 0, 2.5), session('b', 3, 2.5)), (), 'rest tail'))
    assert merged.delivery_cost == split.delivery_cost
    assert merged.short_run_cost == split.short_run_cost == 0
    # Demand may favor distribution; there is no bonus for the extra run itself.


def test_supporting_easy_volume_is_independent_of_size_count_and_dates():
    policy = EasyDurationPolicy(10, 20, .01, 'existing stage volume', supporting_easy_miles_per_7=7.)
    ctx = context(easy_duration_policy=policy)
    results = []
    for doses in ((3.5, 3.6), (2.5, 4.6), (7.1,), (2.3, 2.4, 2.4)):
        rows = tuple(session(f'e:{cycle}:{i}', cycle*7+i, miles)
                     for cycle in range(5) for i, miles in enumerate(doses))
        result = evaluate_program(ctx, ProgramCandidate(
            tuple(e for e in rows if e.day < ctx.hard_end),
            tuple(e for e in rows if e.day >= ctx.hard_end), 'fixed volume comparison'))
        funding = result.parts['easy_duration']['supporting_easy_volume']
        assert funding['target_miles'] == 35.
        assert funding['delivered_miles'] == pytest.approx(35.5)
        assert funding['shortfall'] == 0
        assert result.short_run_cost == 0
        results.append(result)
    assert all(r.delivery_cost == pytest.approx(results[0].delivery_cost) for r in results)
    # Different demand costs remain: volume credit does not imply equivalence.


def test_supporting_easy_loss_restores_existing_weight_not_an_extra_volume_charge():
    p = EasyDurationPolicy(10, 20, .01, 'test', supporting_easy_miles_per_7=7.)
    ctx = context(easy_duration_policy=p)
    result = evaluate_program(ctx, ProgramCandidate((), (), 'rest comparison'))
    assert result.parts['shortfall_by_purpose']['ordinary_easy'] == 1.
    assert result.delivery_cost == 4.  # Existing four purpose terms, not a fifth.
    half = evaluate_program(ctx, ProgramCandidate((session('easy', 0, 17.5),), (), 'arithmetic fixture only'))
    assert half.parts['shortfall_by_purpose']['ordinary_easy'] == pytest.approx(.25)
    assert half.parts['easy_duration']['supporting_easy_volume']['credited_fraction'] == .5
    # Long mileage still earns total volume, but cannot replace all supporting easy work.
    long = evaluate_program(ctx, ProgramCandidate((session('long', 0, 17.5, 'long'),), (), 'arithmetic fixture only'))
    assert long.parts['shortfall_by_purpose']['ordinary_easy'] == 1.
    assert half.parts['shortfall_by_purpose']['volume'] == long.parts['shortfall_by_purpose']['volume']


def test_supporting_easy_credit_preserves_expected_completion_and_fixed_scope():
    ctx = context(easy_duration_policy=EasyDurationPolicy(10, 20, .01, 'test', 7.))
    first = session('first', 0, 1.5)
    plans = [ProgramCandidate((first, session('next', offset, 2.5)), (), 'rest tail')
             for offset in (3, 4)]
    before = [evaluate_program(ctx, p) for p in plans]
    after_ctx = replace(ctx, history=(*ctx.history, replace(first, source='observed')), decision=day(1))
    after = [evaluate_program(after_ctx, replace(p, hard=p.hard[1:])) for p in plans]
    assert before[0].total_cost-before[1].total_cost == pytest.approx(after[0].total_cost-after[1].total_cost)
    assert before[0].parts['easy_duration'] == after[0].parts['easy_duration']
    extended = evaluate_program(replace(ctx, downstream_end=day(42), downstream_basis='test',
        downstream=(session('background', 36, 4.),)), plans[0])
    assert extended.parts['easy_duration'] == before[0].parts['easy_duration']
    assert extended.delivery_cost == before[0].delivery_cost
    with pytest.raises(ValueError):
        EasyDurationPolicy(10, 20, .01, 'invalid', 0.)


def test_search_and_publication_use_same_unchanged_prescription_and_breakdown():
    ctx, plan = context(), candidate()
    searched = evaluate_program(ctx, plan)
    published = evaluate_program(ctx, plan)
    assert searched == published
    assert searched.total_cost == searched.delivery_cost + searched.weighted_demand_cost
    assert searched.switching_cost is None
    assert not searched.selection_ready
    assert searched.conditional_ledger == (*ctx.history, *plan.hard, *plan.continuation)
    assert all(row.source == 'planned' for row in searched.conditional_ledger[len(ctx.history):])
    assert searched.conditional_ledger[0].miles == 8.453  # No rounding actual work.


@pytest.mark.parametrize('name', ['A', 'B', 'C', 'D'])
def test_existing_objective_calendars_preserve_exact_numerical_score(name):
    from scripts.audit_program_objective import calendar
    rates = {role: {axis: dict(bounds=Quantity.point(rate, 'test'))
                   for axis, rate in [('running_minutes', 10.), ('cardio', 20.)]}
             for role in ('easy', 'long', 'quality')}
    rows = calendar(name, START, 56, rates)
    ctx = context(end=day(56))
    plan = ProgramCandidate(tuple(row for row in rows if row.day < ctx.hard_end),
        tuple(row for row in rows if row.day >= ctx.hard_end), 'original repeated motif')
    expected = components(ctx.history, rows, stage_start=START, decision=START,
        end=ctx.end, scales=ctx.scales, policy=ctx.policy, scenario=ctx.scenario)
    evaluated = evaluate_program(ctx, plan)
    assert evaluated.parts == expected
    assert evaluated.total_cost == score(expected, ctx.exchange)


def test_moving_hard_tail_boundary_without_changing_work_does_not_change_score():
    ctx, plan = context(), candidate()
    original = evaluate_program(ctx, plan)
    moved = ProgramCandidate((*plan.hard, *plan.continuation), (), 'explicit rest after promoted tail')
    result = evaluate_program(replace(ctx, hard_end=day(23)), moved)
    assert result.parts == original.parts
    assert result.total_cost == original.total_cost
    assert result.fingerprint != original.fingerprint  # Different publication scope.


def test_identical_execution_only_removes_sunk_cost_and_preserves_candidate_ranking():
    ctx, plan = context(), candidate()
    alternative = replace(plan, hard=(plan.hard[0], replace(plan.hard[1], day=day(5)), plan.hard[2]))
    completed = replace(plan.hard[0], source='observed')
    later_ctx = replace(ctx, decision=day(1), history=(*ctx.history, completed))
    deltas, rankings = [], []
    for option in (plan, alternative):
        before = evaluate_program(ctx, option)
        after = evaluate_program(later_ctx, replace(option, hard=option.hard[1:]))
        assert before.parts['delivery'] == after.parts['delivery']
        assert before.parts['session_costs'][1:] == after.parts['session_costs']
        assert after.conditional_ledger.count(completed) == 1
        deltas.append(before.total_cost - after.total_cost)
        rankings.append((before.total_cost, after.total_cost))
    assert deltas[0] > 0
    assert deltas[0] == pytest.approx(deltas[1])
    assert rankings[0][0] - rankings[1][0] == pytest.approx(rankings[0][1] - rankings[1][1])


def test_empty_day_rollover_does_not_rescale_or_reset_stage():
    ctx, plan = context(), candidate()
    plan = replace(plan, hard=plan.hard[1:])
    before = evaluate_program(ctx, plan)
    after = evaluate_program(replace(ctx, decision=day(1)), plan)
    assert before.total_cost == after.total_cost
    assert before.parts['targets'] == after.parts['targets']


@pytest.mark.parametrize('factor', [.8, 1.2])
def test_nonidentical_execution_preserves_actual_dose_and_changes_future_demand(factor):
    ctx, plan = context(), candidate()
    completed = replace(plan.hard[0], source='observed')
    actual = replace(completed, miles=completed.miles * factor,
                     demand=tuple(q.scale(factor) for q in completed.demand))
    remaining = replace(plan, hard=plan.hard[1:])
    expected = evaluate_program(replace(ctx, decision=day(1), history=(*ctx.history, completed)), remaining)
    changed = evaluate_program(replace(ctx, decision=day(1), history=(*ctx.history, actual)), remaining)
    assert changed.conditional_ledger[len(ctx.history)].miles == actual.miles
    assert changed.parts['delivery']['volume'] - expected.parts['delivery']['volume'] == pytest.approx(
        actual.miles - completed.miles)
    assert (changed.weighted_demand_cost > expected.weighted_demand_cost) == (factor > 1)


def test_continuation_is_charged_and_remembers_hard_prefix():
    ctx, plan = context(), candidate()
    base = evaluate_program(ctx, plan)
    rest_tail = evaluate_program(ctx, replace(plan, continuation=(), continuation_basis='explicit rest'))
    assert base.weighted_demand_cost > rest_tail.weighted_demand_cost
    changed = replace(plan, hard=(*plan.hard[:2], replace(plan.hard[2], day=day(20))))
    changed_result = evaluate_program(ctx, changed)
    assert sum(changed_result.parts['session_costs'][-1]['axis_cost'].values()) > sum(
        base.parts['session_costs'][-1]['axis_cost'].values())


def test_downstream_work_changes_cost_not_delivery_targets_or_normalization():
    ctx, plan = context(), candidate()
    base = evaluate_program(ctx, plan)
    known = session('later_long', 36, miles=8.4, role='long')
    scoped = replace(ctx, downstream=(known,), downstream_end=day(42),
                     downstream_basis='fixed test future work, not a default continuation')
    added = evaluate_program(scoped, plan)
    assert added.parts['delivery'] == base.parts['delivery']
    assert added.parts['targets'] == base.parts['targets']
    assert added.delivery_cost == base.delivery_cost
    assert added.weighted_demand_cost > base.weighted_demand_cost
    assert added.conditional_ledger[-1] == known
    assert added.parts['delivery_end'] == ctx.end
    assert added.downstream_accounting['status'] == 'supplied_scenario'
    assert base.downstream_accounting['status'] == 'unspecified'
    assert added.fingerprint != base.fingerprint
    # Extending coverage with explicitly no further work cannot dilute the cost.
    longer = evaluate_program(replace(scoped, downstream_end=day(49)), plan)
    assert longer.total_cost == added.total_cost
    assert longer.parts['targets'] == added.parts['targets']


def test_explicit_downstream_rest_does_not_change_scores_or_invent_residual_penalty():
    ctx, plan = context(), candidate()
    base = evaluate_program(ctx, plan)
    rest = evaluate_program(replace(ctx, downstream_end=day(42),
                                   downstream_basis='explicit test rest'), plan)
    assert rest.total_cost == base.total_cost
    assert rest.parts['demand_cost'] == base.parts['demand_cost']
    assert rest.fingerprint != base.fingerprint


def test_isolated_late_run_self_cost_is_already_position_independent():
    ctx = context(history=())
    early = ProgramCandidate((), (session('easy', 25),), 'isolated')
    late = replace(early, continuation=(replace(early.continuation[0], day=day(34)),))
    assert evaluate_program(ctx, early).total_cost == evaluate_program(ctx, late).total_cost


def test_downstream_added_interaction_matches_existing_kernel_exactly():
    from math import exp
    ctx = context(history=(), scales=(44., 88., 15.))
    early = ProgramCandidate((), (session('easy', 30),), 'early')
    late = ProgramCandidate((), (session('easy', 34),), 'late')
    later = session('later', 36)
    scoped = replace(ctx, downstream=(later,), downstream_end=day(42), downstream_basis='known dose')
    a, b = evaluate_program(scoped, early), evaluate_program(scoped, late)
    # Each easy is normalized to (1, 1, 0). Their self-cost cancels. Only the
    # cross-term differs: mean of two active axes among three, normalized to 7/35.
    expected = ctx.exchange * 7/35 * 2/3 * (exp(-2/2)-exp(-6/2))
    assert b.total_cost-a.total_cost == pytest.approx(expected)
    assert a.delivery_cost == b.delivery_cost


def test_expected_completion_parity_includes_the_same_downstream_work_once():
    ctx, plan = context(), candidate()
    ctx = replace(ctx, downstream=(session('later', 36),), downstream_end=day(42),
                  downstream_basis='fixed expectation')
    other = replace(plan, hard=(plan.hard[0], replace(plan.hard[1], day=day(5)), plan.hard[2]))
    observed = replace(plan.hard[0], source='observed')
    after_ctx = replace(ctx, decision=day(1), history=(*ctx.history, observed))
    before = [evaluate_program(ctx, p) for p in (plan, other)]
    after = [evaluate_program(after_ctx, replace(p, hard=p.hard[1:])) for p in (plan, other)]
    assert before[0].total_cost-before[1].total_cost == pytest.approx(after[0].total_cost-after[1].total_cost)
    assert all(sum(row.key == 'later' for row in r.conditional_ledger) == 1 for r in (*before, *after))


@pytest.mark.parametrize('changes', [
    dict(downstream=(session('later', 36),)),
    dict(downstream_end=day(42)),
    dict(downstream_end=day(35), downstream_basis='bad end'),
    dict(downstream=(session('too_early', 34),), downstream_end=day(42), downstream_basis='bad date'),
    dict(downstream=(session('observed', 36, source='observed'),), downstream_end=day(42), downstream_basis='future observation'),
])
def test_downstream_scope_requires_explicit_valid_provenance_and_ownership(changes):
    with pytest.raises(ValueError):
        context(**changes)


def test_candidate_cannot_reuse_a_downstream_identity():
    ctx = context(downstream=(session('first', 36),), downstream_end=day(42), downstream_basis='fixed')
    with pytest.raises(ValueError, match='Unique'):
        evaluate_program(ctx, candidate())


@pytest.mark.parametrize('hard,tail', [
    ((session('wrong', 21),), ()),
    ((), (session('wrong', 20),)),
    ((), (session('wrong', 35),)),
    ((session('wrong', -1),), ()),
])
def test_interval_ownership_rejects_misplaced_work(hard, tail):
    with pytest.raises(ValueError, match='prescriptions must lie'):
        evaluate_program(context(), ProgramCandidate(hard, tail, 'test'))


@pytest.mark.parametrize('other', [session('first', 2), session('duplicate_date', 0)])
def test_duplicate_identity_or_date_is_not_double_counted(other):
    plan = candidate()
    with pytest.raises(ValueError, match='Unique'):
        evaluate_program(context(), replace(plan, hard=(*plan.hard, other)))


def test_history_candidate_identity_cannot_be_counted_twice():
    plan = candidate()
    with pytest.raises(ValueError, match='Unique'):
        evaluate_program(context(), replace(plan, hard=(replace(plan.hard[0], key='actual'),)))


@pytest.mark.parametrize('changes', [
    {'exchange': .2}, {'scales': (45., 88., 15.)},
    {'policy': TrialPolicy(weekly_miles=19.)}, {'scenario': DemandScenario(memory_days=4.)},
    {'history': (session('actual', -3, source='observed'),)},
])
def test_identity_includes_all_model_inputs(changes):
    assert evaluate_program(context(**changes), candidate()).fingerprint != evaluate_program(
        context(), candidate()).fingerprint


def test_identical_copies_and_order_have_same_identity_and_score():
    ctx, plan = context(), candidate()
    reverse = replace(plan, hard=tuple(reversed(plan.hard)))
    assert evaluate_program(ctx, plan) == evaluate_program(replace(ctx), reverse)


def test_inputs_are_snapshotted_not_backed_by_mutable_lists():
    rows, weights, axes, scales = list(candidate().hard), [1., 1., 1., 1.], [1., 1., 1.], [44., 88., 15.]
    demands = list(rows[0].demand)
    rows[0] = replace(rows[0], demand=demands)
    ctx = context(policy=TrialPolicy(purpose_weights=weights),
                  scenario=DemandScenario(axis_weights=axes), scales=scales)
    plan = ProgramCandidate(rows, candidate().continuation, 'test')
    before = evaluate_program(ctx, plan)
    weights[0] = axes[0] = scales[0] = 100.
    demands[0] = Quantity.point(1000., 'mutated input')
    rows.clear()
    assert evaluate_program(ctx, plan) == before


def test_unknown_demand_is_not_silently_zero():
    plan = candidate()
    first = plan.hard[0]
    plan = replace(plan, hard=(replace(first, demand=(*first.demand[:2], Quantity(0, 90))), *plan.hard[1:]))
    lower = evaluate_program(context(scenario=DemandScenario(bound_fraction=0)), plan)
    upper = evaluate_program(context(scenario=DemandScenario(bound_fraction=1)), plan)
    assert upper.weighted_demand_cost > lower.weighted_demand_cost


@pytest.mark.parametrize('bad', [-1., float('nan'), float('inf')])
def test_invalid_exchange_rejected(bad):
    with pytest.raises(ValueError, match='exchange'):
        context(exchange=bad)


def test_noncanonical_future_target_and_undeclared_tail_rejected():
    with pytest.raises(ValueError, match='one-decimal'):
        ProgramCandidate((session('off_grid', 0, miles=4.45),), (), 'rest')
    with pytest.raises(ValueError, match='Declare'):
        ProgramCandidate((), (), '')


def test_future_observation_is_not_allowed_into_history():
    with pytest.raises(ValueError, match='earlier observed'):
        context(history=(session('future', 0, source='observed'),))


def test_observed_work_cannot_be_passed_as_forecast():
    with pytest.raises(ValueError, match='planned work only'):
        ProgramCandidate((session('observed', 0, source='observed'),), (), 'rest')
