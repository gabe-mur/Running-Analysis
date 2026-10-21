from dataclasses import replace
from datetime import timedelta
from math import exp

import pytest

from scripts.program_objective_trial import PURPOSES, TrialPolicy
from scripts.shared_program_evaluator import ProgramCandidate, evaluate_program
from scripts.timely_purpose_trial import (
    PurposeTimingPolicy, UnknownPurposeHistory, interval_loss, timing_components,
)
from test_shared_program_evaluator import candidate, context, day, session, START


def ctx(**updates):
    return context(timing=PurposeTimingPolicy(1.), history=(
        session('prior_long', -6, 8.4, 'long', 'observed'),
        session('prior_quality', -3, 4.5, 'quality', 'observed')), **updates)


def test_exact_interval_integral_matches_fine_numerical_quadrature():
    count = 10000
    expected = sum((1-.7*exp(-(i+.5)*3/count/7))**2 for i in range(count))*3/count
    assert interval_loss(.7, 3., 7.) == pytest.approx(expected, abs=1e-9)


def test_replaces_aggregate_terms_without_double_charging_or_changing_demand():
    opening, plan = ctx(), candidate()
    old = evaluate_program(replace(opening, timing=None), plan)
    new = evaluate_program(opening, plan)
    removed = sum(old.parts['shortfall_by_purpose'][p] * opening.policy.purpose_weights[PURPOSES.index(p)]
                  for p in ('long', 'quality'))
    temporal = sum(v['weighted_loss'] for v in new.parts['timing'].values())
    assert new.delivery_cost == pytest.approx(old.delivery_cost - removed + temporal)
    assert new.weighted_demand_cost == old.weighted_demand_cost
    assert new.parts['delivery'] == old.parts['delivery']
    assert new.delivery_cost == sum(new.parts['shortfall_by_purpose'][p] * opening.policy.purpose_weights[i]
                                   for i, p in enumerate(PURPOSES))
    assert new.parts['aggregate_shortfall_by_purpose'] == old.parts['shortfall_by_purpose']
    assert not new.selection_ready


def test_partial_work_cannot_fully_reset_and_labels_without_credit_do_nothing():
    opening = ctx()
    first = session('q', 0, 4.5, 'quality')
    def evaluate(credit):
        p = ProgramCandidate((replace(first, quality_credit=credit),), (), 'explicit rest')
        return evaluate_program(opening, p).parts['timing']['quality']['daily'][0]
    none, partial, full = evaluate(0), evaluate(.8), evaluate(1.)
    assert none['after_session_coverage'] == none['opening_coverage']
    assert partial['after_session_coverage'] == .8
    assert full['after_session_coverage'] == 1.
    assert none['loss'] > partial['loss'] > full['loss']


@pytest.mark.parametrize('credit',[0.,.01,.5,.89,1.])
def test_proportional_grace_integrates_partial_clock_without_banking(credit):
    from scripts.timely_purpose_trial import envelope_day,coverage_at
    for age in range(9):
        expected=sum((1-coverage_at(credit,age+(i+.5)/10000,7.,7.,
            credit_contract='proportional_grace_v1'))**2 for i in range(10000))/10000
        actual=envelope_day(((START,credit),),START+timedelta(days=age),
            grace=7.,memory=7.,credit_contract='proportional_grace_v1')
        assert actual['loss']==pytest.approx(expected,abs=1e-9)
        if credit in (0.,1.):
            assert actual==envelope_day(((START,credit),),START+timedelta(days=age),
                grace=7.,memory=7.)
    assert coverage_at(credit,7.,7.,7.,credit_contract='proportional_grace_v1')<=1.


def test_partial_cadence_does_not_erase_delivered_work_or_change_long_and_fatigue():
    opening=replace(ctx(),timing=PurposeTimingPolicy(1.,grace_fraction=1.))
    first=replace(session('q',-1,4.5,'quality','observed'),quality_credit=.89)
    opening=replace(opening,history=(first,))
    p=ProgramCandidate((replace(session('next',6,4.5,'quality'),quality_credit=1.),),(), 'explicit rest')
    old=evaluate_program(opening,p)
    new=evaluate_program(opening,p,quality_cadence_contract='proportional_grace_v1')
    assert new.conditional_ledger==old.conditional_ledger
    assert new.conditional_ledger[0].quality_credit==.89
    assert new.weighted_demand_cost==old.weighted_demand_cost
    assert new.parts['timing']['long']==old.parts['timing']['long']
    assert all(d['loss']==0 for d in new.parts['timing']['quality']['daily'][:5])
    assert new.parts['timing']['quality']['daily'][5]['loss']>0
    assert new.fingerprint!=old.fingerprint


def test_fractional_sessions_do_not_add_up_to_a_full_cadence_bank():
    from datetime import timedelta
    from scripts.timely_purpose_trial import envelope_day
    events=((START,.5),(START+timedelta(days=1),.5))
    args=dict(grace=7.,memory=7.,credit_contract='proportional_grace_v1')
    assert envelope_day(events,START+timedelta(days=5),**args)==envelope_day(events[1:],START+timedelta(days=5),**args)
    assert envelope_day(events,START+timedelta(days=5),**args)['loss']>0


def test_clustered_full_sessions_do_not_bank_coverage_beyond_last_full_session():
    p = session('q', -1, 4.5, 'quality', 'observed')
    clustered = ctx()
    # Endpoints identical after the last full event, even with another full event the day before.
    a = evaluate_program(replace(clustered, history=(*clustered.history, p)), candidate())
    extra = session('extra', -2, 4.5, 'quality', 'observed')
    b = evaluate_program(replace(clustered, history=(*clustered.history, extra, p)), candidate())
    assert a.parts['timing'] == b.parts['timing']
    assert b.weighted_demand_cost > a.weighted_demand_cost


@pytest.mark.parametrize('kind', ['easy', 'long', 'quality'])
def test_exact_completion_removes_only_common_sunk_timing_and_demand(kind):
    opening = ctx()
    first = session('first', 0, 4.4, kind)
    a = ProgramCandidate((first, session('next', 5, 8.4, 'long')),
                         (session('tail', 24, 4.5, 'quality'),), 'same continuation')
    b = replace(a, hard=(first, replace(a.hard[1], day=day(8))))
    later = replace(opening, decision=day(1), history=(*opening.history, replace(first, source='observed')))
    deltas = []
    for plan in (a, b):
        before = evaluate_program(opening, plan)
        after = evaluate_program(later, replace(plan, hard=plan.hard[1:]))
        for role in ('long', 'quality'):
            before_days = before.parts['timing'][role]['daily'][1:]
            after_days = after.parts['timing'][role]['daily']
            for old, new in zip(before_days, after_days):
                assert old['loss'] == pytest.approx(new['loss'], abs=1e-14)
        deltas.append(before.total_cost-after.total_cost)
    assert deltas[0] == pytest.approx(deltas[1], abs=1e-14)


def test_rest_rollover_drops_only_past_area_and_never_restarts_clock():
    opening = ctx()
    a = candidate()
    a = replace(a, hard=a.hard[1:])
    b = replace(a, hard=(replace(a.hard[0], day=day(5)), *a.hard[1:]))
    first = [evaluate_program(opening, p).total_cost for p in (a, b)]
    later = [evaluate_program(replace(opening, decision=day(2)), p).total_cost for p in (a, b)]
    assert first[0]-first[1] == pytest.approx(later[0]-later[1])


def test_hard_tail_boundary_does_not_change_timing():
    opening, plan = ctx(), candidate()
    a = evaluate_program(opening, plan)
    b = evaluate_program(replace(opening, hard_end=day(23)),
        ProgramCandidate((*plan.hard, *plan.continuation), (), 'promoted identical tail'))
    assert a.parts == b.parts


def test_timing_policy_is_part_of_the_reproducibility_identity():
    opening, plan = ctx(), candidate()
    nominal = evaluate_program(opening, plan)
    changed = evaluate_program(replace(opening, timing=PurposeTimingPolicy(2.)), plan)
    assert changed.fingerprint != nominal.fingerprint
    assert changed.delivery_cost != nominal.delivery_cost


def test_no_catchup_debt_after_a_full_completion():
    # A much longer preceding drought does not demand extra work after the same full session.
    opening = ctx()
    full = session('completed', -1, 8.4, 'long', 'observed')
    a = replace(opening, history=(*opening.history, full))
    b = replace(a, history=(replace(opening.history[0], day=day(-60)), opening.history[1], full))
    assert evaluate_program(a, candidate()).parts['timing'] == evaluate_program(b, candidate()).parts['timing']


def test_missing_history_is_not_silently_overdue_or_cleared():
    with pytest.raises(UnknownPurposeHistory, match='quality'):
        evaluate_program(context(timing=PurposeTimingPolicy(1.)), candidate())
    # Scope can be deliberately restricted for a long-only diagnostic.
    assert evaluate_program(context(timing=PurposeTimingPolicy(1., ('long',))), candidate()).parts['timing'].keys() == {'long'}


@pytest.mark.parametrize('role', ['long', 'quality'])
@pytest.mark.parametrize('days', [56, 112])
def test_control_fixtures_have_equal_doses_counts_volume_and_common_continuation(role, days):
    from scripts.audit_timely_purpose import control_plans
    plans = control_plans(role, days)
    signatures, tails = [], []
    for plan in plans.values():
        rows = (*plan.hard, *plan.continuation)
        assert len({r.day for r in rows}) == len(rows)
        signatures.append(sorted((r.role, r.miles, tuple(q.low for q in r.demand), r.quality_credit) for r in rows))
        tails.append(tuple(r for r in rows if r.day >= day(28)))
    assert all(s == signatures[0] for s in signatures)
    assert all(t == tails[0] for t in tails)


def test_cost_is_bounded_and_changes_smoothly_across_eight_days():
    values = []
    for age in (7, 8, 9):
        h = (session('long', -age, 8.4, 'long', 'observed'),)
        parts = timing_components(h, (), stage_start=START, decision=START, end=day(1),
                                  policy=TrialPolicy(), timing=PurposeTimingPolicy(1., ('long',)))
        values.append(parts['long']['loss'])
    assert 0 < values[0] < values[1] < values[2] < 1
    assert abs((values[1]-values[0])-(values[2]-values[1])) < .01


@pytest.mark.parametrize('scale', [.5, 1., 2.])
def test_equal_volume_half_sessions_do_not_outscore_regular_full_purpose_on_timing(scale):
    # Isolate dose/timing value: no claim these calendars are recovery-compatible.
    timing = PurposeTimingPolicy(scale, ('long',))
    history = (session('long', -7, 8.4, 'long', 'observed'),)
    full = tuple(session(f'full{i}', i, 8.4, 'long') for i in range(0, 56, 7))
    half = tuple(session(f'half{i}', i, 4.2, 'long') for i in range(56) if i%7 in (0, 3))
    def loss(plan):
        return timing_components(history, plan, stage_start=START, decision=START, end=day(56),
                                 policy=TrialPolicy(), timing=timing)['long']['loss']
    assert sum(r.miles for r in half) == pytest.approx(sum(r.miles for r in full))
    assert loss(full) < loss(half)


@pytest.mark.parametrize('bad', [0., -1., float('nan'), float('inf')])
def test_invalid_timing_memory_rejected(bad):
    with pytest.raises(ValueError):
        PurposeTimingPolicy(bad)
