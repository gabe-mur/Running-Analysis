"""Exact dose-polish wiring controls, not daily optimizer/stability acceptance."""
from datetime import timedelta

import pytest

from test_adaptive_stage_lifecycle import completed_stage
from test_continuation_boundary_lifecycle import check_replay
from run_analysis.adaptive.new_model_daily_harness import DailyHarness,HarnessPolicy
from run_analysis.adaptive.adaptive_stage_lifecycle import advance_stage


@pytest.mark.parametrize('allowance',[1,16])
def test_score_admission_budget_and_restart_survive_polishing(tmp_path,monkeypatch,allowance):
    h,inputs,config,at=completed_stage(tmp_path,monkeypatch,growth_focus='easy_volume',
        continuation_boundary=True,single_volume=True,purpose_continuation=True,
        coordinated_final_evaluations=allowance)
    try:
        for event in (h.events[0],h.replan(at,reason='post-upload wiring control')):
            check_replay(h,event)
            report=event['refinement']['coordinated_final_doses']
            assert 1<=report['additional_full_evaluations']<=allowance
            assert report['cost']<=report['original_cost']
            assert report['eligible']
            assert event['cost']==report['cost']
            assert event['evaluations']>=report['additional_full_evaluations']
            if report['changed']:
                assert not event['local_neighborhood_complete']
        root=h.sandbox.root
    finally:
        h.sandbox.close()
    reopened=DailyHarness.resume(root,config,as_of=at)
    try:
        assert reopened.policy.coordinated_final_evaluations==allowance
        check_replay(reopened,reopened.replan(at+timedelta(days=1),reason='resumed wiring control'))
    finally:
        reopened.sandbox.close()


def test_prepared_growth_is_not_refined_again_at_publication(tmp_path,monkeypatch):
    h,inputs,_,at=completed_stage(tmp_path,monkeypatch,growth_focus='easy_volume',
        continuation_boundary=True,single_volume=True,purpose_continuation=True,
        coordinated_final_evaluations=16)
    import run_analysis.adaptive.compiled_calendar_doses as module
    original=module.refine
    calls=[]
    def counted(*args,**kwargs):
        calls.append(1)
        return original(*args,**kwargs)
    monkeypatch.setattr(module,'refine',counted)
    try:
        updated,review,event=advance_stage(h,inputs,at)
        assert event['stage_acceptance']
        assert len(calls)==1  # Prepared selection must remain exact at commit.
        check_replay(updated,event)
    finally:
        h.sandbox.close()


def test_default_and_invalid_policy_cannot_silently_enable_new_search():
    assert HarnessPolicy().coordinated_final_evaluations==0
    for allowance in (-1,True,1.5):
        with pytest.raises(ValueError,match='Coordinated'):
            HarnessPolicy(coordinated_final_evaluations=allowance)
    with pytest.raises(ValueError,match='Coordinated'):
        HarnessPolicy(coordinated_final_evaluations=16)
