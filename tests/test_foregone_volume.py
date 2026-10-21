from dataclasses import replace
from datetime import date,timedelta
from types import SimpleNamespace as NS

import numpy as np
import pytest

from run_analysis.adaptive.foregone_volume import ForegoneVolume,VERSION,LEGACY_VERSION,from_reviews
from run_analysis.adaptive.shared_program_evaluator import evaluate_program
from run_analysis.workload_accounting import Quantity
from test_soft_horizon_trial import candidate
from test_volume_benefit_trial import fixture


def entry(day,actual=None,issued=4.2):
    return NS(key='slot:'+day.isoformat(),contract=NS(source='original:'+day.isoformat(),
        purpose='easy_volume',targets=(NS(metric='distance_meters',
            quantity=Quantity.point(issued*1609.344,'issued')),)),
        completion=None if actual is None else NS(distance_meters=actual.scale(1609.344)))


@pytest.mark.parametrize('actual,expected',[(None,4.2),(Quantity.point(3.,'actual'),1.2),
    (Quantity.point(4.2,'actual'),0.),(Quantity.point(5.,'actual'),0.),
    (Quantity(3.,4.3,None,'uncertain'),0.)])
def test_only_unfulfilled_original_distance_can_be_foregone(actual,expected):
    day=date(2026,10,21)
    rows=from_reviews((entry(day,actual),),since=day,today=day+timedelta(days=1))
    assert sum(r.miles for r in rows)==pytest.approx(expected)
    if expected:assert rows[0].source=='original:2026-10-21'


def test_no_same_day_future_pre_adoption_or_duplicate_relief():
    day=date(2026,10,21)
    for d in (day-timedelta(days=1),day,day+timedelta(days=1)):
        assert not from_reviews((entry(d),),since=day,today=day)
    with pytest.raises(ValueError,match='One unfulfilled'):
        from_reviews((entry(day),entry(day)),since=day,today=day+timedelta(days=1))


@pytest.mark.parametrize('purpose',['threshold','speed_skill','race_specific'])
def test_all_quality_purposes_remain_quality_not_supporting_easy(purpose):
    day=date(2026,10,21);row=entry(day);row.contract.purpose=purpose
    assert from_reviews((row,),since=day,today=day+timedelta(days=1))[0].role=='quality'


def enabled(ctx):
    day=ctx.decision-timedelta(days=2)
    return replace(ctx,volume_shortfall_policy=VERSION,volume_shortfall_since=day,
        foregone_volume=(ForegoneVolume(day,'missed-original','easy',4.2,0.),))


def test_relief_changes_only_volume_score_not_physical_history_fatigue_or_targets():
    ctx,_,rows=fixture();ctx=replace(ctx,volume_linear_coefficient=.1)
    before=evaluate_program(ctx,candidate(ctx,rows));after=evaluate_program(enabled(ctx),candidate(ctx,rows))
    assert after.conditional_ledger==before.conditional_ledger
    assert after.weighted_demand_cost==before.weighted_demand_cost
    assert after.short_run_cost==before.short_run_cost
    assert after.parts['delivery']==before.parts['delivery']
    assert after.parts['targets']==before.parts['targets']
    assert after.parts['shortfall_by_purpose']['long']==before.parts['shortfall_by_purpose']['long']
    assert after.parts['shortfall_by_purpose']['quality']==before.parts['shortfall_by_purpose']['quality']
    assert after.delivery_cost<before.delivery_cost
    assert after.parts['foregone_volume']['progression_credit'] is False


def test_future_omission_or_deferral_cannot_create_relief_and_still_loses_volume_value():
    ctx,_,rows=fixture();ctx=enabled(replace(ctx,volume_linear_coefficient=.1))
    # Preserve the V1 witness. V2 removes the pre-adoption deficit, so an
    # already-surplus early dose can legitimately be more useful later.
    ctx=replace(ctx,volume_shortfall_policy=LEGACY_VERSION)
    early=next(r for r in rows if r.role=='easy' and (r.day-ctx.decision).days<7)
    free=next(ctx.decision+timedelta(days=n) for n in range(15,21)
        if all(r.day!=ctx.decision+timedelta(days=n) for r in rows))
    shifted=tuple(replace(r,day=free) if r.key==early.key else r for r in rows)
    missing=tuple(r for r in rows if r.key!=early.key)
    a=evaluate_program(ctx,candidate(ctx,rows))
    for plan in (shifted,missing):
        b=evaluate_program(ctx,candidate(ctx,plan))
        assert b.delivery_cost>a.delivery_cost
        assert b.parts['foregone_volume']==a.parts['foregone_volume']
    with pytest.raises(ValueError,match='expired'):
        replace(ctx,foregone_volume=(ForegoneVolume(ctx.decision,'future','easy',4.2,0.),))


def test_v2_funded_future_cannot_be_deleted_or_backloaded_and_clock_does_not_roll():
    from run_analysis.adaptive.program_objective_trial import TrialSession
    ctx,_,_=fixture();ctx=enabled(replace(ctx,volume_linear_coefficient=.1))
    rows=tuple(TrialSession(str(n),ctx.decision+timedelta(days=n),round(ctx.policy.weekly_miles/7,1),
        'easy',tuple(Quantity.point(1.,'test demand') for _ in range(3)),source='planned')
        for n in range((ctx.end-ctx.decision).days))
    source,target=rows[7],rows[20]
    shifted=tuple(replace(r,miles=r.miles+source.miles) if r.key==target.key else r
        for r in rows if r.key!=source.key)
    missing=tuple(r for r in rows if r.key!=source.key)
    deltas=[]
    for shift in (0,1,3):
        decision=ctx.decision+timedelta(days=shift)
        history=tuple(replace(r,source='observed') for r in rows if r.day<decision)
        c=replace(ctx,decision=decision,history=history)
        baseline=evaluate_program(c,candidate(c,tuple(r for r in rows if r.day>=decision)))
        for plan in (shifted,missing):
            other=evaluate_program(c,candidate(c,tuple(r for r in plan if r.day>=decision)))
            assert other.delivery_cost>baseline.delivery_cost
            assert other.parts['foregone_volume']==baseline.parts['foregone_volume']
        other=evaluate_program(c,candidate(c,tuple(r for r in shifted if r.day>=decision)))
        deltas.append(other.delivery_cost-baseline.delivery_cost)
        assert c.volume_shortfall_since==ctx.volume_shortfall_since
    assert deltas==pytest.approx([deltas[0]]*3,abs=1e-12)


def test_v2_opening_balance_is_signed_before_clipping_not_per_rest_day():
    from run_analysis.adaptive.program_objective_trial import TrialSession
    ctx,_,_=fixture();ctx=replace(ctx,volume_shortfall_policy=VERSION,
        volume_shortfall_since=ctx.decision,foregone_volume=())
    history=tuple(TrialSession('prior:'+str(n),ctx.decision-timedelta(days=n),
        ctx.policy.weekly_miles,'easy',tuple(Quantity.point(1.,'test demand') for _ in range(3)),
        source='observed') for n in range(1,60,7))
    ctx=replace(ctx,history=history)
    result=evaluate_program(ctx,candidate(ctx,()))
    # A fully funded but lumpy history has rest dates. Those dates are not
    # independent missing-dose allowances and must not create positive credit.
    assert result.parts['rolling_volume']['daily'][0]['volume']['foregone']==pytest.approx(0.,abs=1e-12)


def test_relief_expires_with_original_window_and_does_not_bank_surplus():
    ctx,_,rows=fixture();ctx=enabled(ctx)
    e=evaluate_program(ctx,candidate(ctx,rows))
    daily=e.parts['rolling_volume']['daily']
    assert daily[0]['volume']['foregone']>0
    expires=ctx.foregone_volume[0].day+timedelta(days=ctx.policy.volume_window_days+ctx.volume_smoothing_days-2)
    assert all(r['volume']['foregone']==0. for r in daily if r['day']>expires)
    assert all(r['volume']['delivered']>=0. for r in daily)


@pytest.mark.parametrize('distribution',['linear_v1','c1_v1'])
def test_compiled_nonzero_relief_matches_authoritative_scorer(distribution):
    from test_compiled_calendar_doses import fixture as compiled_fixture
    compiled,evaluate,*_=compiled_fixture(context_modifier=lambda c:enabled(replace(c,continuation_distribution=distribution)))
    choices=[compiled.original_levels,np.zeros(len(compiled.days),dtype=int),
        np.full(len(compiled.days),len(compiled.keys)-1,dtype=int)]
    choices.extend(np.array([(i*7+n*11)%len(compiled.keys) for i in range(len(compiled.days))]) for n in range(4))
    for x,value in zip(choices,compiled.cost(choices)):
        assert value==pytest.approx(evaluate(compiled.physical(x))['cost'],abs=1e-12)


def test_new_policy_is_explicit_and_snapshot_defaults_preserve_archives():
    from run_analysis.adaptive.new_model_daily_harness import HarnessPolicy
    from run_analysis.adaptive.decision_snapshot import _encode,_decode
    assert HarnessPolicy().volume_shortfall_policy is None
    with pytest.raises(ValueError,match='sustained'):
        HarnessPolicy(volume_shortfall_policy=VERSION)
    ctx,_,_=fixture()
    assert 'foregone_volume' not in _encode(ctx)['fields']
    assert 'volume_shortfall_policy' not in _encode(ctx)['fields']
    new=enabled(ctx)
    assert _decode(_encode(new))==new


def test_adoption_and_restart_preserve_policy_without_rebasing_delivery(tmp_path,monkeypatch):
    from test_continuous_general_progression import fixture as harness_fixture
    from run_analysis.adaptive.adaptive_stage_lifecycle import advance_stage
    from run_analysis.adaptive.new_model_daily_harness import DailyHarness
    h,inputs,config,at=harness_fixture(tmp_path,monkeypatch)
    try:
        h.policy=replace(h.policy,volume_shortfall_policy=VERSION)
        history=tuple(h.history);scales=h.scales
        new,review,event=advance_stage(h,inputs,at)
        assert new.volume_shortfall_since==at.astimezone(h.zone).date()
        assert tuple(new.history)==history and new.scales==scales
        assert review['gate_status']=='observed_delivery_supported'
        # Scenario policy export can be stale after an interrupted publication;
        # the committed snapshot, not that mutable copy, owns the policy.
        resumed=DailyHarness.resume(h.sandbox.root,config,as_of=at)
        try:
            assert resumed.policy.volume_shortfall_policy==VERSION
            assert resumed.volume_shortfall_since==new.volume_shortfall_since
            assert resumed.context(at)==new.context(at)
            assert resumed.general_progression_state==new.general_progression_state
        finally:resumed.sandbox.close()
    finally:h.sandbox.close()


def test_context_drops_expired_relief_without_resetting_adoption_clock(tmp_path,monkeypatch):
    from test_continuous_general_progression import fixture as harness_fixture
    h,_,_,at=harness_fixture(tmp_path,monkeypatch)
    try:
        h.policy=replace(h.policy,volume_shortfall_policy=VERSION)
        today=at.astimezone(h.zone).date()
        h.volume_shortfall_since=today-timedelta(days=100)
        recent=today-timedelta(days=2)
        monkeypatch.setattr('run_analysis.adaptive.adaptive_stage_lifecycle.review_entries',
            lambda *_:(entry(h.volume_shortfall_since),entry(recent)))
        c=h.context(at)
        assert c.volume_shortfall_since==h.volume_shortfall_since
        assert [r.day for r in c.foregone_volume]==[recent]
    finally:h.sandbox.close()


def test_real_supported_long_update_keeps_v2_boundary_and_actual_only_review(tmp_path,monkeypatch):
    from test_continuous_general_progression import fixture as harness_fixture,proposal
    from run_analysis.adaptive.adaptive_stage_lifecycle import advance_stage
    from run_analysis.adaptive.new_model_daily_harness import DailyHarness
    h,inputs,config,at=harness_fixture(tmp_path,monkeypatch,weekly_volume=30.)
    try:
        h.policy=replace(h.policy,volume_shortfall_policy=VERSION)
        h.volume_shortfall_since=at.astimezone(h.zone).date()-timedelta(days=14)
        initial=proposal(h,at)
        h.general_progression_state=dict(initial['continuous_state'],
            day=(at.date()-timedelta(days=1)).isoformat(),volume=31.3,support=18.,
            long_budget=8.49,long_growth_allowed=True,growth_allowed=True)
        new,review,event=advance_stage(h,inputs,at)
        assert review['credited_miles']==pytest.approx(63.)
        assert review['credited_miles']>=review['required_miles']
        assert event['stage_acceptance'] and new.stage.long_miles==8.5
        assert new.volume_shortfall_since==h.volume_shortfall_since
        resumed=DailyHarness.resume(h.sandbox.root,config,as_of=at)
        try:
            assert resumed.volume_shortfall_since==h.volume_shortfall_since
            assert resumed.general_progression_state==new.general_progression_state
            assert resumed.policy.volume_shortfall_policy==VERSION
        finally:resumed.sandbox.close()
    finally:h.sandbox.close()
