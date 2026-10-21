from datetime import date, timedelta
from dataclasses import replace
from math import ceil, exp, lcm
from types import SimpleNamespace

import pytest

from scripts.check_explicit_distribution_tail import (
    compression_switches, compression_winner, easy_concentration,
    cycle_program, explicit_context, fixture, four_dose_cycles,
    periodic_demand, periodic_recovery_audit, shift_cycle,
    decimal_compositions, stationary_batch,
    stationary_volume_batch, stationary_timing_cost,
    periodic_fatigue_lower_bound,
    mixed_key_layouts, cyclic_gaps,
)
from scripts.check_easy_distribution_profiles import Profile
from run_analysis.adaptive.program_objective_trial import (
    DemandScenario, EasyDurationPolicy, TrialPolicy, TrialSession, components,
)
from run_analysis.workload_accounting import Quantity
from run_analysis.adaptive.shared_program_evaluator import ProgramCandidate, evaluate_program
from run_analysis.program_delivery import TargetSegment
from test_memory_covered_continuation import guarded


def test_decimal_compositions_exhaust_ordered_supported_allocations():
    assert decimal_compositions(6, 2, (1, 2, 4)) == ((2, 4), (4, 2))
    assert decimal_compositions(4, 2, (1, 2, 3)) == ((1, 3), (2, 2), (3, 1))
    assert decimal_compositions(1, 2, (1, 2)) == ()
    assert decimal_compositions(3, 1, (1, 2, 3)) == ((3,),)


def test_mixed_key_layouts_cover_all_canonical_quality_anchors():
    from itertools import combinations
    keys = ('quality','long')
    expected = set()
    for second in range(1,14):
        for longs in combinations(tuple(d for d in range(1,14) if d!=second),2):
            pattern = tuple(sorted(((0,keys[0]),(second,keys[0]),*((d,keys[1]) for d in longs))))
            expected.add(min(pattern,tuple(sorted(((d-second)%14,k) for d,k in pattern))))
    actual = mixed_key_layouts(keys)
    assert len(actual)==len(set(actual))==432
    assert set(actual)==expected
    assert any(cyclic_gaps(tuple(d for d,k in row if k=='quality'),14)==(6,8) for row in actual)


def test_cyclic_gap_measurements_include_the_boundary():
    assert cyclic_gaps((0,6),14)==(6,8)
    assert cyclic_gaps((0,7),14)==(7,7)
    assert cyclic_gaps((4,),8)==(8,)


@pytest.mark.parametrize('memory,compression', [(1., 0.), (2., 0.), (3., .3)])
def test_batched_stationary_cost_matches_independent_periodic_ledger(memory, compression):
    context, demand = recovery_fixture(memory, compression)
    samples = (((0, 'quality', 3.9), (3, 'long', 8.4), (1, 'easy', 2.3), (5, 'easy', 3.3)),
               ((0, 'quality', 3.9), (3, 'long', 8.4), (1, 'easy', 1.1), (5, 'easy', 4.5)))
    vectors = [[tuple(v.estimate/s for v,s in zip(demand(r,m), context.scales)) for _,r,m in row]
               for row in samples]
    shorts = [[context.easy_duration_policy.penalty(demand(r,m)[0].estimate) if r == 'easy' else 0.
               for _,r,m in row] for row in samples]
    fatigue, short = stationary_batch(tuple(d for d,_,_ in samples[0]), vectors, shorts, context)
    for i,row in enumerate(samples):
        expected = periodic_demand(row, demand, context)
        assert fatigue[i] == pytest.approx(expected['fatigue'], abs=1e-12)
        assert short[i] == pytest.approx(expected['short'], abs=1e-12)


def test_batched_stationary_rejects_duplicate_dates():
    context, _ = recovery_fixture()
    with pytest.raises(ValueError):
        stationary_batch((0,0), [[[1.,1.,0.], [1.,1.,0.]]], [[0.,0.]], context)


@pytest.mark.parametrize('period,mixed', [(6,False),(7,False),(8,False),(9,False),(10,False),(14,False),(14,True)])
def test_variable_period_full_score_matches_existing_evaluator(period,mixed):
    ctx,_,_ = guarded()
    _,demand = recovery_fixture()
    total = 20.3*period/7
    support = round(total-ctx.policy.long_miles-3.9,1)
    # Same demand convention as the existing fixture; no database dependency.
    entries = ((0,'quality',3.9), (2,'easy',round(support/2,1)),
               (period-3,'long',ctx.policy.long_miles),
               (period-1,'easy',round(support-round(support/2,1),1)))
    if mixed:
        entries = ((0,'quality',3.9),(6,'quality',3.9),(3,'long',ctx.policy.long_miles),(11,'long',ctx.policy.long_miles),
                   (1,'easy',3.7),(5,'easy',3.8),(8,'easy',3.8),(13,'easy',3.7))
    days = lcm(period,7); days *= ceil(56/days)
    rows = tuple(TrialSession(str(n),ctx.decision+timedelta(days=n),m,r,demand(r,m),float(r=='quality'),
        'observed' if n<0 else 'planned') for n in range(-84,days) for d,r,m in entries if n%period==d)
    c = explicit_context(ctx,days)
    c = replace(c,history=tuple(r for r in rows if r.source=='observed'),stage_start=c.decision,
        volume_targets=(TargetSegment(c.decision-timedelta(days=84),c.end,20.3),),
        support_targets=(TargetSegment(c.decision-timedelta(days=84),c.end,7.5),),
        policy=replace(c.policy,purpose_weights=(1.,0.,1.,1.)))
    e = evaluate_program(c,ProgramCandidate(tuple(r for r in rows if c.decision<=r.day<c.hard_end),
        tuple(r for r in rows if c.hard_end<=r.day<c.end),'test declared variable recurrence'))
    vectors = [[tuple(v.estimate/s for v,s in zip(demand(r,m),c.scales)) for _,r,m in entries]]
    short = [[c.easy_duration_policy.penalty(demand(r,m)[0].estimate) if r=='easy' else 0. for _,r,m in entries]]
    f,s = stationary_batch(tuple(d for d,_,_ in entries),vectors,short,c,period=period)
    v = stationary_volume_batch(tuple(d for d,_,_ in entries),[[m for _,_,m in entries]],c,period=period,target=20.3)[0]
    timing = stationary_timing_cost(tuple((d,r) for d,r,_ in entries if r!='easy'),c,period=period)
    independent = periodic_demand(entries,demand,c,period=period)
    assert f[0] == pytest.approx(independent['fatigue'],abs=1e-12)
    assert f[0] == pytest.approx(e.weighted_demand_cost,abs=1e-12)
    assert s[0] == pytest.approx(e.short_run_cost,abs=1e-12)
    assert v+timing == pytest.approx(e.delivery_cost,abs=1e-12)
    assert f[0]+s[0]+v+timing == pytest.approx(e.total_cost,abs=1e-12)
    lower = periodic_fatigue_lower_bound(tuple(sum(q[a] for q in vectors[0]) for a in range(3)),c,period=period)
    assert lower <= f[0]+1e-12


@pytest.mark.parametrize('memory,compression', [(1., 0.), (2., 0.), (3., .3)])
def test_infinite_periodic_all_roles_match_explicit_ledger(memory, compression):
    origin = date(2026, 1, 1)
    entries = ((0, 'quality', 3.9), (2, 'easy', 3.9), (4, 'long', 8.4), (6, 'easy', 3.8))
    def demand(role, miles):
        return tuple(Quantity.point(v, 'test') for v in (
            miles*11, miles*(15 if role == 'quality' else 12), 12 if role == 'quality' else 0))
    c = SimpleNamespace(scales=(40, 50, 10), scenario=DemandScenario(memory_days=memory,
        timing_compression_days=compression), exchange=.01,
        easy_duration_policy=EasyDurationPolicy(10, 25, .01, 'test', 7.7))
    rows = tuple(TrialSession(str(n), origin+timedelta(days=n), m, role, demand(role, m),
        float(role == 'quality'), 'observed' if n < 0 else 'planned')
        for n in range(-224, 56) for d, role, m in entries if n%7 == d)
    old = components(tuple(r for r in rows if r.day < origin), tuple(r for r in rows if r.day >= origin),
        stage_start=origin, decision=origin, end=origin+timedelta(days=56), scales=c.scales,
        policy=TrialPolicy(), scenario=c.scenario)
    result = periodic_demand(entries, demand, c)
    assert result['fatigue'] == pytest.approx(.01*old['demand_cost'], abs=1e-12)
    assert result['demand_by_axis'] == pytest.approx(old['demand_by_axis'], abs=1e-12)
    shifted = tuple(((d+3)%7, role, m) for d, role, m in entries)
    rotation = periodic_demand(shifted, demand, c)
    assert rotation['fatigue'] == pytest.approx(result['fatigue'], abs=1e-12)
    assert rotation['short'] == result['short']


def test_explicit_control_removes_terminal_approximation_not_weights_or_history():
    ctx, _, _, = guarded()
    alternative = explicit_context(ctx, 112)
    assert alternative.policy == ctx.policy
    assert alternative.history == ctx.history
    assert alternative.scales == ctx.scales
    assert alternative.scenario == ctx.scenario
    assert alternative.exchange == ctx.exchange
    assert alternative.easy_duration_policy == ctx.easy_duration_policy
    assert alternative.hard_end == ctx.hard_end
    assert not alternative.soft_horizon_taper
    assert not alternative.continuation_boundary_references
    assert alternative.continuation_guard_days == 0
    assert ctx.continuation_guard_days == 7
    assert (alternative.end-alternative.decision).days == 112


def test_profile_curve_changes_volume_objective_not_physical_inputs():
    ctx, _, _ = guarded()
    hs = {'easy': {'cardio': {'estimate': 55}, 'running_minutes': {'estimate': 44}},
          'quality:ordinary': {'cardio': {'estimate': 70}, 'running_minutes': {'estimate': 40}}}
    case = {'state': SimpleNamespace(context=ctx), 'handoffs': hs,
            'catalog': {'easy': SimpleNamespace(role='easy', miles=4.4),
                        'quality:ordinary': SimpleNamespace(role='quality', miles=3.9)}}
    profile = Profile('moderate', 20, 11, 8.4, 3.9, 12, 3)
    old, _, _, _, _ = fixture(case, profile, ctx.scales)
    trial, _, _, _, _ = fixture(case, profile, ctx.scales, single_volume=True)
    assert trial.policy.purpose_weights == (1., 0., 1., 1.)
    assert old.policy.purpose_weights == ctx.policy.purpose_weights
    assert trial.history == old.history
    assert trial.scales == old.scales
    assert trial.scenario == old.scenario
    assert trial.exchange == old.exchange
    assert trial.easy_duration_policy == old.easy_duration_policy
    assert trial.continuation_boundary_references == old.continuation_boundary_references


@pytest.mark.parametrize('days', [21, 54])
def test_reject_nonperiodic_or_missing_continuation_window(days):
    ctx, _, _ = guarded()
    with pytest.raises(ValueError):
        explicit_context(ctx, days)


def recovery_fixture(memory=2., compression=0.):
    def demand(role, miles):
        return tuple(Quantity.point(v, 'test point demand') for v in (
            miles*11, miles*(15 if role == 'quality' else 12), 12 if role == 'quality' else 0))
    context = SimpleNamespace(scales=(40, 50, 10), scenario=DemandScenario(
        memory_days=memory, timing_compression_days=compression), exchange=.01,
        easy_duration_policy=EasyDurationPolicy(10, 25, .01, 'test', 7.7))
    return context, demand


@pytest.mark.parametrize('memory,compression', [(1., 0.), (2., 0.), (4., 0.), (3., .3)])
def test_recovery_components_and_continuous_integral_reproduce_existing_score(memory, compression):
    context, demand = recovery_fixture(memory, compression)
    entries = ((0, 'quality', 3.9), (1, 'easy', 2.3), (2, 'easy', 2.7), (4, 'long', 8.4), (6, 'easy', 2.7))
    audit = periodic_recovery_audit(entries, demand, context)
    assert sum(audit['fatigue_components'].values()) == pytest.approx(audit['score']['fatigue'], abs=1e-12)
    assert audit['integral_matches_score'] == (compression == 0)
    if compression == 0:
        assert audit['squared_residue_integral_cost'] == pytest.approx(audit['score']['fatigue'], abs=1e-12)
    assert all(v >= 0 for v in audit['variance_residue_by_axis'].values())


def test_equal_volume_changes_variance_not_mean_residue():
    context, demand = recovery_fixture()
    keys = ((0, 'quality', 3.9), (4, 'long', 8.4))
    two = periodic_recovery_audit((*keys, (2, 'easy', 3.9), (6, 'easy', 3.8)), demand, context)
    three = periodic_recovery_audit((*keys, (1, 'easy', 2.3), (2, 'easy', 2.7), (6, 'easy', 2.7)), demand, context)
    assert two['mean_residue_by_axis'] == pytest.approx(three['mean_residue_by_axis'], abs=1e-12)
    assert two['score']['fatigue'] != pytest.approx(three['score']['fatigue'], abs=1e-8)
    assert two['fatigue_components']['self'] > three['fatigue_components']['self']


def test_time_reversal_can_hide_different_fatigue_entering_quality():
    context, demand = recovery_fixture()
    after = periodic_recovery_audit(((0, 'quality', 3.9), (1, 'easy', 4.)), demand, context)
    before = periodic_recovery_audit(((0, 'quality', 3.9), (6, 'easy', 4.)), demand, context)
    assert after['score']['fatigue'] == pytest.approx(before['score']['fatigue'], abs=1e-12)
    assert after['key_sessions'][0]['preceding_by_axis']['cardio'] < before['key_sessions'][0]['preceding_by_axis']['cardio']
    assert after['key_sessions'][0]['overlap_cost'] < before['key_sessions'][0]['overlap_cost']
    assert after['key_sessions'][0]['easy_miles_next_24h'] == 4.
    assert before['key_sessions'][0]['easy_miles_next_24h'] == 0.
    assert after['fatigue_components']['easy_to_key'] == pytest.approx(before['fatigue_components']['key_to_easy'], abs=1e-12)


def test_quality_work_residue_has_no_interaction_with_easy_or_long_work_axis():
    context, demand = recovery_fixture()
    a = periodic_recovery_audit(((0, 'quality', 3.9), (1, 'easy', 3.8), (4, 'long', 8.4)), demand, context)
    b = periodic_recovery_audit(((0, 'quality', 3.9), (3, 'easy', 3.8), (6, 'long', 8.4)), demand, context)
    for audit in (a, b):
        assert audit['raw_fatigue_components_by_axis']['easy_to_key']['work_minutes'] == 0.
        assert audit['raw_fatigue_components_by_axis']['key_to_easy']['work_minutes'] == 0.
        assert next(k for k in audit['key_sessions'] if k['role'] == 'long')['preceding_by_axis']['work_minutes'] > 0.
    # Long/easy dates changed, but neither uses the remaining quality-work residue.
    assert a['score']['demand_by_axis']['work_minutes'] == b['score']['demand_by_axis']['work_minutes']


def test_fragmentation_advantage_follows_quadratic_cost_without_other_history():
    context, demand = recovery_fixture()
    origin = date(2026, 1, 1)
    def session(day, miles):
        return TrialSession(str(day), origin+timedelta(days=day), miles, 'easy', demand('easy', miles))
    def evaluate(sessions):
        return components((), sessions, stage_start=origin, decision=origin,
            end=origin+timedelta(days=7), scales=context.scales, policy=TrialPolicy(), scenario=context.scenario)
    full = evaluate((session(0, 4.),))
    split = evaluate((session(0, 2.), session(2, 2.)))
    q = tuple(v.estimate/s for v, s in zip(demand('easy', 2.), context.scales))
    for axis, value in zip(('running_minutes', 'cardio', 'work_minutes'), q):
        assert full['demand_by_axis'][axis]-split['demand_by_axis'][axis] == pytest.approx(
            value**2*(1-exp(-2/context.scenario.memory_days)), abs=1e-12)


@pytest.mark.parametrize('entries', [(), ((0, 'easy', 2.), (0, 'easy', 3.)), ((7, 'easy', 2.),)])
def test_recovery_audit_rejects_ambiguous_periodic_dates(entries):
    context, demand = recovery_fixture()
    with pytest.raises(ValueError):
        periodic_recovery_audit(entries, demand, context)


def test_compression_is_not_a_rest_day_count_and_is_phase_invariant():
    balanced = ((1, 4.), (3, 4.))
    uneven = ((1, 2.), (3, 6.))
    assert easy_concentration(balanced) == .5
    assert easy_concentration(uneven) == .625
    assert easy_concentration(tuple(((d+4)%7, m) for d, m in uneven)) == .625
    assert easy_concentration(()) == 0.


def test_concentration_blindspots_are_explicit_not_assumed_volume_protection():
    funded = ((1, 1.), (3, 3.), (6, 4.))
    assert easy_concentration(tuple((d, m/2) for d, m in funded)) == easy_concentration(funded)
    assert easy_concentration(funded[1:]) > easy_concentration(funded)
    assert easy_concentration(((3, .1),)) == pytest.approx(1.)


def test_compression_tradeoff_and_exact_switch_envelope():
    # Costs and H chosen for two known switches; fourth candidate is dominated.
    sampled = ((.05, ((0, 1.), (2, 1.), (4, 1.), (6, 1.))),
               (.051, ((2, 2.), (6, 2.))), (.054, ((2, 4.),)),
               (.06, ((1, 2.), (5, 2.))))
    assert compression_winner(sampled, 0.) == min(sampled)
    assert compression_winner(sampled, .005) == sampled[1]
    assert compression_winner(sampled, .01) == sampled[2]
    switches = compression_switches(sampled)
    assert [w for w, _ in switches] == pytest.approx((0., .004, .006))
    assert [item for _, item in switches] == list(sampled[:3])
    assert [easy_concentration(item[1]) for _, item in switches] == [.25, .5, 1.]
    # Same baseline fixture at weight zero, including short-dose penalties.
    context, demand = recovery_fixture()
    keys = ((0, 'quality', 3.9), (4, 'long', 8.4))
    patterns = (((2, 3.9), (6, 3.8)), ((1, 2.3), (2, 2.7), (6, 2.7)))
    scores = []
    for pattern in patterns:
        r = periodic_demand((*keys, *((d, 'easy', m) for d, m in pattern)), demand, context)
        scores.append((r['fatigue']+r['short'], pattern))
    assert compression_winner(scores, 0.) == min(scores)


@pytest.mark.parametrize('weight', [-1., float('nan'), float('inf')])
def test_reject_invalid_compression_weight(weight):
    with pytest.raises(ValueError):
        compression_winner(((.05, ((1, 2.),)),), weight)


@pytest.mark.parametrize('miles', [0., -1., float('nan'), float('inf')])
def test_reject_invalid_concentration_dose(miles):
    with pytest.raises(ValueError):
        easy_concentration(((1, miles),))


def test_calendar_probe_exhausts_fixed_four_dose_placements_not_just_rotations():
    keys = ('quality', 'long', 'easy_large', 'easy_small')
    cycles = four_dose_cycles(keys)
    assert len(set(cycles)) == 120
    absolute = {shift_cycle(p, phase) for p in cycles for phase in range(7)}
    assert len(absolute) == 840
    for pattern in absolute:
        assert {k for _, k in pattern} == set(keys)
        assert len({d for d, _ in pattern}) == 4
        assert all(0 <= d < 7 for d, _ in pattern)
    assert {next(d for d, k in p if k == 'long') for p in cycles} == set(range(1, 7))


def test_cycle_materialization_keeps_doses_and_hard_scope_separate():
    origin = date(2026, 9, 29)
    pattern = ((0, 'quality'), (1, 'easy_small'), (3, 'long'), (5, 'easy_large'))
    result = cycle_program(pattern, origin, 56)
    assert len(result) == 32
    assert all(sum(key == k for _, key in result) == 8 for _, k in pattern)
    assert result[0] == ('2026-09-29', 'quality')
    assert all(origin <= date.fromisoformat(d) < origin+timedelta(days=56) for d, _ in result)
    assert shift_cycle(pattern, 7) == pattern
    assert shift_cycle(shift_cycle(pattern, 2), 5) == pattern


@pytest.mark.parametrize('keys', [('q', 'l', 'e'), ('q', 'l', 'e', 'e')])
def test_calendar_probe_rejects_ambiguous_fixed_doses(keys):
    with pytest.raises(ValueError):
        four_dose_cycles(keys)
