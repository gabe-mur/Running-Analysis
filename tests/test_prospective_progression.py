from dataclasses import replace
from datetime import datetime,timedelta,timezone
from types import SimpleNamespace

import pytest

from run_analysis.staged_progression import (
    DevelopmentStage,ProgressionPolicy,ReviewEntry,review_recent_delivery,MILE,
)
from run_analysis.stage_target_history import StageTarget
from run_analysis.delivered_training import DoseContract,DoseTarget,DeliveredDose
from run_analysis.workload_accounting import Quantity
from run_analysis.adaptive.prospective_progression import freeze_volume_review,expected_volume_review

AT=datetime(2026,9,29,tzinfo=timezone.utc)


def fixture():
    stage=DevelopmentStage('phase',AT,20.,4.,8.,17.,7.,
        ProgressionPolicy(14,.05,'easy_volume',21.,'gradual'),'fixture')
    targets=(StageTarget('path',AT,AT.date(),20.,8.,'fixture',14,21.,9.),)
    return stage,targets


def entry(key,start,miles,actual=None):
    contract=DoseContract(AT,'long_endurance',(
        DoseTarget('distance_meters',Quantity((miles-.2)*MILE,(miles+.2)*MILE,miles*MILE,'issued band')),),key)
    done=DeliveredDose(key,start,start+timedelta(hours=1),start+timedelta(hours=1),
        Quantity.point((miles if actual is None else actual)*MILE,'actual'),Quantity.point(3600,'actual'),(),'unknown','long')
    return ReviewEntry(key,start,contract,done)


def planned(key,start,miles):
    return SimpleNamespace(key=key,start=start,end=start+timedelta(hours=1),miles=miles,source='planned')


@pytest.mark.parametrize('actual',[7.8,8.,8.2,7.,10.])
def test_real_and_prospective_past_credit_share_exact_band_and_excess_rules(actual):
    stage,targets=fixture()
    done=entry('past',AT+timedelta(days=1),8.,actual)
    review=freeze_volume_review(stage,targets,(done,),AT+timedelta(days=2),review_at=stage.review_at,timezone='UTC')
    prospective=expected_volume_review(review,())
    actual_review=review_recent_delivery(stage,(done,),stage.review_at,ledger_complete=True,
        longest_prior_30d_miles=8.,required_miles=review.required_miles)
    assert prospective['expected_credited_miles']==pytest.approx(actual_review['credited_miles'])
    assert prospective['required_miles']==actual_review['required_miles']
    assert done.completion.distance_meters.low==actual*MILE


def test_missing_and_future_uploads_are_not_expected_past_completion():
    stage,targets=fixture();done=entry('past',AT+timedelta(days=1),8.)
    missing=replace(done,key='missed',completion=None)
    unavailable=replace(done,completion=replace(done.completion,available_at=AT+timedelta(days=3)))
    review=freeze_volume_review(stage,targets,(missing,unavailable),AT+timedelta(days=2),review_at=stage.review_at,timezone='UTC')
    assert review.completed_credit==()


def test_future_tail_cannot_fund_review_and_surplus_is_capped_in_real_credit():
    stage,targets=fixture()
    review=freeze_volume_review(stage,targets,(),AT,review_at=stage.review_at,timezone='UTC')
    rows=(planned('early',AT+timedelta(days=3),review.required_miles),)
    assert expected_volume_review(review,rows)['funded']
    moved=(planned('early',stage.review_at+timedelta(days=1),review.required_miles*2),)
    assert not expected_volume_review(review,moved)['funded']
    assert expected_volume_review(review,moved)['expected_credited_miles']==0


def test_small_deficit_does_not_pass_and_exact_boundary_completion_counts():
    stage,targets=fixture()
    review=freeze_volume_review(stage,targets,(),AT,review_at=stage.review_at,timezone='UTC')
    before=planned('boundary',review.end-timedelta(hours=1),review.required_miles-.1)
    assert not expected_volume_review(review,(before,))['funded']
    assert expected_volume_review(review,(replace_namespace(before,miles=review.required_miles),))['funded']
    late=planned('crosses',review.end-timedelta(minutes=30),review.required_miles)
    assert not expected_volume_review(review,(late,))['funded']


def replace_namespace(value,**changes):return SimpleNamespace(**dict(vars(value),**changes))


def test_checkpoint_requirement_survives_refresh_and_exact_partial_day_review():
    stage,targets=fixture();end=stage.review_at+timedelta(hours=4)
    a=freeze_volume_review(stage,targets,(),AT,review_at=end,timezone='UTC')
    b=freeze_volume_review(stage,targets,(),AT+timedelta(days=1),review_at=end,timezone='UTC')
    assert (a.start,a.end,a.required_miles)==(b.start,b.end,b.required_miles)
    assert a.start==AT+timedelta(hours=4)
    with pytest.raises(ValueError,match='future'):
        freeze_volume_review(stage,targets,(),end,review_at=end,timezone='UTC')


def test_expected_completion_equals_real_review_at_declared_checkpoint():
    stage,targets=fixture()
    review=freeze_volume_review(stage,targets,(),AT,review_at=stage.review_at,timezone='UTC')
    rows=(planned('a',AT+timedelta(days=1),8.),planned('b',AT+timedelta(days=5),8.),
          planned('c',AT+timedelta(days=10),review.required_miles-16))
    expected=expected_volume_review(review,rows)
    realized=review_recent_delivery(stage,tuple(entry(r.key,r.start,r.miles) for r in rows),stage.review_at,
        ledger_complete=True,longest_prior_30d_miles=8.,required_miles=review.required_miles)
    assert expected['expected_credited_miles']==pytest.approx(realized['credited_miles'])
    assert expected['funded']
    assert realized['status']=='conditional_development_candidate'


def test_snapshot_gate_is_reproducible_without_changing_score_or_physical_constraints():
    from test_decision_snapshot import freeze,evaluate,AT as SNAP_AT
    from run_analysis.adaptive.prospective_progression import ProspectiveVolumeReview
    from run_analysis.adaptive.decision_snapshot import serialize,restore
    state=freeze();base=evaluate(state)
    review=ProspectiveVolumeReview(state.stage_key,SNAP_AT,SNAP_AT,SNAP_AT+timedelta(days=14),20.,(),
        state.timezone,'explicit test gate')
    changed=replace(state,prospective_volume_review=review)
    encoded=serialize(changed)
    result=evaluate(restore(encoded,expected_fingerprint=changed.fingerprint))
    assert replace(result['evaluation'],fingerprint=base['evaluation'].fingerprint)==base['evaluation']
    assert result['evaluation'].fingerprint!=base['evaluation'].fingerprint
    assert result['constraints']==base['constraints']
    assert result['construction_eligible']==base['decision_eligible']
    assert not result['decision_eligible']
    assert 'prospective_volume_review' not in serialize(state)
    admitted=evaluate(replace(changed,prospective_volume_review=replace(review,required_miles=8.)))
    assert admitted['decision_eligible']


def test_funding_does_not_overrule_health_rejection():
    from test_decision_snapshot import freeze,evaluate,AT as SNAP_AT
    from run_analysis.session_decision import CurrentContext
    from run_analysis.adaptive.prospective_progression import ProspectiveVolumeReview
    state=freeze(contexts=(CurrentContext(SNAP_AT.date(),SNAP_AT,'health_concern','explicit report'),))
    review=ProspectiveVolumeReview(state.stage_key,SNAP_AT,SNAP_AT,SNAP_AT+timedelta(days=14),1.,(),
        state.timezone,'explicit test gate')
    result=evaluate(replace(state,prospective_volume_review=review))
    assert result['progression_delivery']['funded']
    assert not result['decision_eligible']
    assert not result['construction_eligible']


def test_harness_gate_survives_upload_stage_confirmation_and_restart(tmp_path,monkeypatch):
    from test_adaptive_stage_lifecycle import completed_stage
    from run_analysis.adaptive.adaptive_stage_lifecycle import advance_stage
    from run_analysis.adaptive.new_model_daily_harness import DailyHarness
    from run_analysis.adaptive.decision_snapshot import restore
    h,inputs,config,at=completed_stage(tmp_path,monkeypatch,growth_focus='easy_volume',
        continuation_boundary=True,single_volume=True,gradual=True,prospective_review_hour=4)
    try:
        first=restore((h.sandbox.root/h.events[0]['snapshot_file']).read_text(),
            expected_fingerprint=h.events[0]['snapshot_fingerprint'])
        assert first.prospective_volume_review
        assert h.events[0]['selected']['progression_delivery']['funded']
        updated,review,event=advance_stage(h,inputs,at)
        assert event and event['selection_contract']=='prospective_volume_review_v1'
        final=restore((updated.sandbox.root/event['snapshot_file']).read_text(),expected_fingerprint=event['snapshot_fingerprint'])
        assert (final.prospective_volume_review.start,final.prospective_volume_review.end,
            final.prospective_volume_review.required_miles)==(first.prospective_volume_review.start,
            first.prospective_volume_review.end,first.prospective_volume_review.required_miles)
        updated.sandbox.close()
        resumed=DailyHarness.resume(tmp_path/'state',config,as_of=at)
        try:
            assert resumed.policy.prospective_review_hour==4
            preview=resumed.replan(at+timedelta(seconds=1),reason='restart parity',publish=False)
            assert preview['state'].prospective_volume_review==replace(final.prospective_volume_review,available_at=at+timedelta(seconds=1))
            replay=dict(preview['search'],program=[list(p) for p in preview['program']],
                snapshot_fingerprint=preview['state'].fingerprint,manifest={'scope':'explicit fixture'})
            reused=resumed.replan(at+timedelta(seconds=1),reason='same captured search reuse',
                publish=False,replayed_selection=replay)
            assert reused['program']==preview['program']
            assert reused['search']['selected']==preview['search']['selected']
            with pytest.raises(ValueError,match='differs from the harness evidence'):
                resumed.replan(at+timedelta(seconds=1),reason='reject different evidence',publish=False,
                    replayed_selection=dict(replay,snapshot_fingerprint='wrong'))
        finally:resumed.sandbox.close()
    finally:h.sandbox.close()
