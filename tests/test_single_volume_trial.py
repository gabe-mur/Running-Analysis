from dataclasses import replace
from types import SimpleNamespace

import pytest

from run_analysis.adaptive.program_objective_trial import TrialPolicy
from run_analysis.adaptive.shared_program_evaluator import evaluate_program, SINGLE_VOLUME_MODEL_VERSION
from run_analysis.adaptive.adaptive_candidate_pipeline import supporting_easy_target
from test_memory_covered_continuation import guarded
from test_soft_horizon_trial import candidate


def single(ctx):
    return replace(ctx,policy=replace(ctx.policy,purpose_weights=(1.,0.,1.,1.)))


def test_only_supporting_volume_may_have_zero_weight():
    assert TrialPolicy(purpose_weights=(1,0,1,1)).purpose_weights[1]==0
    for weights in ((0,1,1,1),(1,-1,1,1),(1,0,0,1),(1,float('nan'),1,1)):
        with pytest.raises(ValueError):
            TrialPolicy(purpose_weights=weights)


def test_single_delivery_counts_volume_once_without_erasing_load_or_short_cost():
    ctx,_,rows=guarded()
    old=evaluate_program(ctx,candidate(ctx,rows))
    c=single(ctx)
    new=evaluate_program(c,candidate(c,rows))
    assert new.model_version==SINGLE_VOLUME_MODEL_VERSION
    assert new.short_run_cost==old.short_run_cost
    assert new.conditional_ledger==old.conditional_ledger
    assert new.parts['shortfall_by_purpose']['ordinary_easy']>=0  # Diagnostic retained.
    losses=new.parts['shortfall_by_purpose']
    assert new.delivery_cost==pytest.approx(losses['volume']+losses['long']+losses['quality'])
    assert new.total_cost==pytest.approx(new.delivery_cost+new.weighted_demand_cost+new.short_run_cost)


def test_diagnostic_support_target_cannot_manufacture_future_training():
    ctx,_,rows=guarded();ctx=single(ctx)
    ctx=replace(ctx,easy_duration_policy=replace(ctx.easy_duration_policy,supporting_easy_miles_per_7=100.))
    result=evaluate_program(ctx,candidate(ctx,rows))
    continuation=result.parts['continuation_boundary']
    assert continuation['expected_miles_per_7']==pytest.approx(ctx.policy.weekly_miles)


def test_remainder_has_no_ordinary_size_floor_and_accounts_for_cadence():
    stage=SimpleNamespace(weekly_equivalent_miles=14.,long_miles=8.4,ordinary_miles=4.4)
    assert supporting_easy_target(stage,3.9)==4.4  # Legacy retained for archive.
    assert supporting_easy_target(stage,3.9,single_volume=True)==1.7
    assert supporting_easy_target(stage,3.9,single_volume=True,long_per_7=.5)==5.9
    with pytest.raises(ValueError,match='revise key allocations'):
        supporting_easy_target(stage,7.,single_volume=True)


def test_no_training_remains_worse_than_training_when_support_penalty_is_removed():
    ctx,_,rows=guarded();ctx=single(ctx)
    empty=evaluate_program(ctx,candidate(ctx,()))
    full=evaluate_program(ctx,candidate(ctx,rows))
    assert empty.total_cost>full.total_cost


def test_growth_and_restart_keep_single_volume_and_hold_long_during_easy_phase(tmp_path,monkeypatch):
    from test_adaptive_stage_lifecycle import completed_stage
    from run_analysis.adaptive.adaptive_stage_lifecycle import advance_stage
    from run_analysis.adaptive.new_model_daily_harness import DailyHarness
    from run_analysis.adaptive.decision_snapshot import restore
    h,inputs,config,at=completed_stage(tmp_path,monkeypatch,growth_focus='easy_volume',
        continuation_boundary=True,single_volume=True,volume_linear_coefficient=0.)
    try:
        old_long=h.stage.long_miles
        updated,review,event=advance_stage(h,inputs,at)
        assert review['status']=='conditional_development_candidate'
        assert event['model_version']==SINGLE_VOLUME_MODEL_VERSION
        assert updated.stage.long_miles==old_long
        assert updated.stage.weekly_equivalent_miles>h.stage.weekly_equivalent_miles
        state=restore((h.sandbox.root/event['snapshot_file']).read_text(),expected_fingerprint=event['snapshot_fingerprint'])
        assert state.context.policy.purpose_weights[1]==0
        updated.sandbox.close()
        reopened=DailyHarness.resume(tmp_path/'state',config,as_of=at)
        try:
            assert reopened.training_policy.purpose_weights[1]==0
            assert reopened.stage==updated.stage
        finally:
            reopened.sandbox.close()
    finally:
        h.sandbox.close()
