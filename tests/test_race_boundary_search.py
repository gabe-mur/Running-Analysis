"""Actual bounded C1 search on declared established training profiles.

Synthetic physical histories isolate phase wiring; this is not a recorded-HR
daily stability or physiological race-readiness claim. Explicitly run separately
from cheap controls because each case uses the real full search budget.
"""
from dataclasses import replace
from datetime import date,timedelta
import json
from pathlib import Path

import pytest

from test_adaptive_stage_lifecycle import completed_stage
from test_race_event import established_profile
from test_race_goal_integration import revision
from run_analysis.adaptive.service import AdaptivePlannerService
from run_analysis.adaptive.decision_snapshot import restore
from run_analysis.adaptive.race_accounting import allowed_dose
from scripts.research_manifest import freeze_sources


@pytest.mark.parametrize('distance',['5k','10k','half_marathon','marathon'])
def test_actual_shared_search_entering_taper(tmp_path,monkeypatch,distance):
    h,inputs,config,at=completed_stage(tmp_path,monkeypatch,growth_focus='easy_volume',
        gradual=True,single_volume=True,continuation_boundary=True,purpose_continuation=True,
        purpose_continuation_phase='known_full_phase_v1',continuation_distribution='c1_v1',
        prospective_review_hour=4,**established_profile(distance))
    # Remove only the seed-selector fixture. The real general search now runs,
    # with the same allowances as the current C1 experimental replay.
    monkeypatch.undo()
    h.policy=replace(h.policy,evaluations=10000,complete_first_search=True,
        final_winner_evaluations=1000,coordinated_final_evaluations=64,
        calendar_merge_evaluations=32)
    r=revision(h.origin,distance,days=14+{'5k':7,'10k':7,'half_marathon':10,'marathon':21}[distance])
    h.race_goal=r
    root=h.sandbox.root
    freeze_sources(root/'boundary-search.sources.zip',config=config,entrypoint=__name__,
        inputs=(Path(__file__),root/'scenario.json'))
    service=AdaptivePlannerService(h,inputs)
    try:
        result=service.refresh(at,reason='actual bounded shared taper-boundary search')
        h=service.harness;event=result['event']
        assert result['stage_review']['status']=='dated_race_growth_hold'
        assert r.phase(h.decision(at))=='taper'
        assert event['selected']['eligible']
        assert event['evaluations']>0 and h.policy.evaluations==10000
        assert event['search']!='explicit_seed_fixture_not_behavioral_acceptance'
        assert all(allowed_dose(r,date.fromisoformat(day),h.catalog[key],h.stage)
            for day,key in event['program'])
        assert not any(day==r.race_date.isoformat() for day,key in event['program'])
        state=restore((root/event['snapshot_file']).read_text(),expected_fingerprint=event['snapshot_fingerprint'])
        assert state.prospective_volume_review is None
        assert state.context.race_demand
        summary=dict(distance=distance,seconds=event['seconds'],evaluations=event['evaluations'],
            admitted=event['selected']['eligible'],cost=event['cost'],miles21=event['volume'],
            program=event['program'],scope=__doc__,live_changed=False)
        (root/'boundary-search-summary.json').write_text(json.dumps(summary,indent=2))
        print(json.dumps({k:v for k,v in summary.items() if k not in ('program','scope')}),flush=True)
    finally:service.harness.sandbox.close()


@pytest.mark.parametrize('distance',['5k','10k','half_marathon','marathon'])
@pytest.mark.parametrize('outcome',['recorded','did_not_start'])
def test_actual_shared_search_after_explicit_outcome(tmp_path,monkeypatch,distance,outcome):
    """One real search per seam; synthetic evidence, not a daily replay pass."""
    from run_analysis.adaptive.race_goal import RaceGoalApplication
    from run_analysis.adaptive.race_event import record_resolution,preparation_active
    from run_analysis.adaptive.new_model_daily_harness import synthetic_payload
    h,inputs,config,at=completed_stage(tmp_path,monkeypatch,growth_focus='easy_volume',
        gradual=True,single_volume=True,continuation_boundary=True,purpose_continuation=True,
        purpose_continuation_phase='known_full_phase_v1',continuation_distribution='c1_v1',
        prospective_review_hour=4,**established_profile(distance))
    monkeypatch.undo()
    h.policy=replace(h.policy,evaluations=10000,complete_first_search=True,
        final_winner_evaluations=1000,coordinated_final_evaluations=64,
        calendar_merge_evaluations=32)
    h.race_goal=revision(h.origin,distance,days=14 if outcome=='recorded' else 7)
    h.race_goal_history=(RaceGoalApplication(h.race_goal,h.origin,h.origin.date()),)
    root=h.sandbox.root
    freeze_sources(root/'outcome-search.sources.zip',config=config,entrypoint=__name__,
        inputs=(Path(__file__),root/'scenario.json'))
    service=AdaptivePlannerService(h,inputs);original=h.stage;actual=None
    try:
        if outcome=='recorded':
            dose=h.catalog['quality:ordinary']
            payload,_,end=synthetic_payload(dose,at+timedelta(hours=16),hr=170,
                distance_factor=h.race_goal.goal.profile.distance_miles/dose.miles)
            service.upload(payload,day=None,available_at=end,replan=False)
            actual=h.history[-1]
            stamp=end+timedelta(microseconds=1)
        else:stamp=at+timedelta(hours=4)
        record_resolution(h.sandbox.db,revision=h.race_goal,history=h.history,timezone_name='UTC',
            status=outcome,physical_key=actual.key if actual else None,
            expected_revision=h.race_goal.key,as_of=stamp)
        result=service.refresh(stamp+timedelta(microseconds=1),reason='actual bounded explicit-outcome search')
        h=service.harness;event=result['event']
        assert event['selected']['eligible'] and event['evaluations']>0
        assert h.policy.evaluations==10000
        assert event['search']!='explicit_seed_fixture_not_behavioral_acceptance'
        state=restore((root/event['snapshot_file']).read_text(),expected_fingerprint=event['snapshot_fingerprint'])
        assert state.context.race_demand==() and state.prospective_volume_review is None
        assert h.stage.weekly_equivalent_miles==original.weekly_equivalent_miles
        assert h.stage.long_miles==original.long_miles
        if actual:
            assert result['stage_review']['status']=='dated_race_growth_hold'
            assert sum(r.key==actual.key for r in state.history)==1
            assert next(r for r in state.history if r.key==actual.key).demand==actual.demand
            assert not next(r for r in state.athlete.sessions if r.key==actual.key).capacity_eligible
            assert all(allowed_dose(h.race_goal,date.fromisoformat(day),
                h.catalog[key],h.stage,resolution=h.race_event) for day,key in event['program'])
        else:
            assert not preparation_active(h,h.decision(stamp))
            assert result['stage_review']['status']=='conditional_maintenance_candidate'
            assert result['stage_review']['race_interrupted']
            assert not result['stage_review']['confirmed_delivery']
        summary=dict(distance=distance,outcome=outcome,seconds=event['seconds'],evaluations=event['evaluations'],
            admitted=True,cost=event['cost'],miles21=event['volume'],program=event['program'],
            scope='One actual full-search transition on synthetic established exposure, not daily stability',live_changed=False)
        (root/'outcome-search-summary.json').write_text(json.dumps(summary,indent=2))
        print(json.dumps({k:v for k,v in summary.items() if k not in ('program','scope')}),flush=True)
    finally:service.harness.sandbox.close()
