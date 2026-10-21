from dataclasses import replace
from datetime import timedelta

import pytest

from run_analysis.adaptive.volume_benefit_trial import linear_volume_loss,volume_benefit_cost
from run_analysis.adaptive.shared_program_evaluator import evaluate_program,VOLUME_BENEFIT_MODEL_VERSION
from run_analysis.adaptive.continuation_boundary_trial import continuation_boundary_cost
from test_memory_covered_continuation import guarded
from test_soft_horizon_trial import candidate


def fixture():
    ctx,refs,rows=guarded()
    return replace(ctx,policy=replace(ctx.policy,purpose_weights=(1.,0.,1.,1.))),refs,rows


def test_linear_handoff_loss_matches_independent_scalar_mean_not_sqrt_of_mean_square():
    ctx,refs,rows=fixture()
    plain=replace(ctx,continuation_boundary_references=())
    source=evaluate_program(plain,candidate(plain,rows))
    result=continuation_boundary_cost(ctx,source,refs,include_linear_volume=True)
    expected=[]
    for scenario in result['scenarios']:
        boundary=scenario['handoff']
        prefix=tuple(r for r in (*ctx.history,*rows) if r.day<boundary)
        loss=0.
        for n in range((boundary-ctx.decision).days+33):
            day=ctx.decision+timedelta(days=n)
            amounts=[]
            for k in range(7):
                stop=day-timedelta(days=k)
                start=stop-timedelta(days=27)
                amount=sum(r.miles for r in prefix if start<=r.day<=stop)
                amount+=max(0,(stop-max(start,boundary)).days+1)*ctx.policy.weekly_miles/7
                amounts.append(amount)
            loss+=max(0,1-sum(amounts)/7/(ctx.policy.weekly_miles*4))/(ctx.end-ctx.stage_start).days
        expected.append(loss)
    assert result['linear_volume_loss']==pytest.approx(sum(expected)/len(expected),abs=1e-12)


def test_only_volume_changes_default_score_and_raw_accounting_remain_exact():
    ctx,_,rows=fixture();e=evaluate_program(ctx,candidate(ctx,rows))
    assert volume_benefit_cost(ctx,e,linear_coefficient=0)['cost']==e.total_cost
    r=volume_benefit_cost(ctx,e,linear_coefficient=.1)
    assert r['weighted_demand_cost']==e.weighted_demand_cost
    assert r['short_run_cost']==e.short_run_cost
    assert r['cost']==pytest.approx(r['delivery_cost']+r['weighted_demand_cost']+r['short_run_cost'])
    assert e.parts['shortfall_by_purpose']['long']>=0
    # The candidate and its conditional physical ledger were not rewritten.
    assert tuple(r.key for r in e.conditional_ledger)==tuple(r.key for r in sorted(rows,key=lambda r:(r.day,r.key)))
    with pytest.raises(ValueError):volume_benefit_cost(ctx,e,linear_coefficient=float('nan'))
    with pytest.raises(ValueError):volume_benefit_cost(ctx,e,linear_coefficient=-1)


def test_future_surplus_cannot_cancel_earlier_linear_deficit():
    ctx,_,rows=fixture()
    e=evaluate_program(ctx,candidate(ctx,rows))
    larger=tuple(replace(r,miles=r.miles+5) if r.day>=ctx.hard_end else r for r in rows)
    other=evaluate_program(ctx,candidate(ctx,larger))
    assert 0<linear_volume_loss(ctx,other)<linear_volume_loss(ctx,e)


def test_rollover_preserves_visible_volume_contrast_with_memory_coverage():
    ctx,_,rows=fixture();first=rows[0]
    later=replace(ctx,decision=ctx.decision+timedelta(days=1),
        hard_end=ctx.hard_end+timedelta(days=1),end=ctx.end+timedelta(days=1),
        history=(replace(first,source='observed'),))
    deltas=[]
    for c in (ctx,later):
        results=[]
        for extra in (0.,.5):
            plan=tuple(replace(r,miles=r.miles+extra) if r.key=='5' else r for r in rows if r.day>=c.decision)
            e=evaluate_program(c,candidate(c,plan))
            results.append(volume_benefit_cost(c,e,linear_coefficient=.1)['added_volume_cost'])
        deltas.append((results[1]-results[0])*(c.end-c.stage_start).days)
    assert deltas[0]==pytest.approx(deltas[1],abs=1e-12)


def test_above_target_has_no_extra_linear_reward_and_scale_preserves_utility():
    from types import SimpleNamespace
    ctx,_,_=fixture()
    ctx=replace(ctx,continuation_boundary_references=())
    def result(delivered,target):
        return SimpleNamespace(parts={'rolling_volume':{'daily':[
            {'day':ctx.decision,'volume':{'delivered':delivered,'target':target}}]}})
    assert linear_volume_loss(ctx,result(100,100))==0
    assert linear_volume_loss(ctx,result(110,100))==0
    assert linear_volume_loss(ctx,result(95,100))>0
    assert linear_volume_loss(ctx,result(95,100))==linear_volume_loss(ctx,result(190,200))


@pytest.mark.parametrize('boundary',[False,True])
def test_connected_curve_matches_isolated_prototype_without_repricing_other_terms(boundary):
    ctx,_,rows=fixture()
    if not boundary:
        ctx=replace(ctx,continuation_boundary_references=())
    base=evaluate_program(ctx,candidate(ctx,rows))
    trial=volume_benefit_cost(ctx,base,linear_coefficient=.1)
    connected=replace(ctx,volume_linear_coefficient=.1)
    result=evaluate_program(connected,candidate(connected,rows))
    assert result.model_version==VOLUME_BENEFIT_MODEL_VERSION
    assert result.total_cost==pytest.approx(trial['cost'],abs=1e-12)
    assert result.delivery_cost==pytest.approx(trial['delivery_cost'],abs=1e-12)
    assert result.weighted_demand_cost==base.weighted_demand_cost
    assert result.short_run_cost==base.short_run_cost
    assert result.conditional_ledger==base.conditional_ledger
    assert result.parts['shortfall_by_purpose']==base.parts['shortfall_by_purpose']
    assert result.total_cost==pytest.approx(result.delivery_cost+result.weighted_demand_cost+result.short_run_cost)
    assert result.fingerprint!=base.fingerprint
    assert evaluate_program(replace(connected,volume_linear_coefficient=0),candidate(ctx,rows))==base
    with pytest.raises(ValueError,match='already included'):
        volume_benefit_cost(connected,result,linear_coefficient=.1)


def test_connected_boundary_computes_linear_loss_once(monkeypatch):
    import run_analysis.adaptive.continuation_boundary_trial as boundary
    ctx,_,rows=fixture();ctx=replace(ctx,volume_linear_coefficient=.1)
    original=boundary.continuation_boundary_cost
    calls=[]
    def traced(*args,**kwargs):
        calls.append(kwargs.get('include_linear_volume'))
        return original(*args,**kwargs)
    monkeypatch.setattr(boundary,'continuation_boundary_cost',traced)
    evaluate_program(ctx,candidate(ctx,rows))
    assert calls==[True]


@pytest.mark.parametrize('coefficient',[-.1,float('nan'),float('inf')])
def test_connected_curve_rejects_invalid_coefficients(coefficient):
    from run_analysis.adaptive.new_model_daily_harness import HarnessPolicy
    ctx,_,_=fixture()
    with pytest.raises(ValueError,match='finite nonnegative'):
        replace(ctx,volume_linear_coefficient=coefficient)
    with pytest.raises(ValueError,match='finite nonnegative'):
        HarnessPolicy(delivery_contract='sustained_v1',volume_linear_coefficient=coefficient)


def test_connected_curve_rejects_duplicate_volume_objective():
    ctx,_,_=fixture()
    with pytest.raises(ValueError,match='single rolling-volume'):
        replace(ctx,policy=replace(ctx.policy,purpose_weights=(1.,1.,1.,1.)),volume_linear_coefficient=.1)


@pytest.mark.parametrize('gradual',[False,True])
def test_default_curve_survives_upload_stage_acceptance_snapshot_and_restart(tmp_path,monkeypatch,gradual):
    from test_adaptive_stage_lifecycle import completed_stage
    from run_analysis.adaptive.adaptive_stage_lifecycle import advance_stage
    from run_analysis.adaptive.new_model_daily_harness import DailyHarness
    from run_analysis.adaptive.decision_snapshot import restore
    h,inputs,config,at=completed_stage(tmp_path,monkeypatch,growth_focus='easy_volume',
        continuation_boundary=True,single_volume=True,gradual=gradual)
    try:
        assert h.policy.volume_linear_coefficient==.1
        assert h.events[0]['model_version']==VOLUME_BENEFIT_MODEL_VERSION
        # The fixture supplies known seeds, but uses the real evaluator and
        # measured-HR upload accounting. This proves wiring, not search quality.
        assert h.sandbox.completed_accounted(at)
        updated,review,event=advance_stage(h,inputs,at)
        assert review['status']=='conditional_development_candidate'
        assert updated.stage.long_miles==h.stage.long_miles
        assert updated.stage.weekly_equivalent_miles>h.stage.weekly_equivalent_miles
        assert updated.policy.volume_linear_coefficient==.1
        assert event['model_version']==VOLUME_BENEFIT_MODEL_VERSION
        state=restore((h.sandbox.root/event['snapshot_file']).read_text(),expected_fingerprint=event['snapshot_fingerprint'])
        assert state.context.volume_linear_coefficient==.1
        updated.sandbox.close()
        reopened=DailyHarness.resume(tmp_path/'state',config,as_of=at)
        try:
            assert reopened.policy.volume_linear_coefficient==.1
            assert reopened.context(at).volume_linear_coefficient==.1
            assert reopened.stage==updated.stage
            event=reopened.replan(at+timedelta(days=1),reason='rollover after restart')
            assert event['model_version']==VOLUME_BENEFIT_MODEL_VERSION
        finally:
            reopened.sandbox.close()
    finally:
        h.sandbox.close()
