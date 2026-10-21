"""Shared-objective plumbing controls; not optimizer/stability acceptance."""
from datetime import date,datetime,time,timedelta
from zoneinfo import ZoneInfo

import pytest

from test_adaptive_stage_lifecycle import completed_stage
from run_analysis.adaptive.new_model_daily_harness import DailyHarness,planned_accounted
from run_analysis.adaptive.decision_snapshot import restore,evaluate_snapshot
from run_analysis.adaptive.shared_program_evaluator import CONTINUATION_MODEL_VERSION,MEMORY_CONTINUATION_MODEL_VERSION,PURPOSE_CONTINUATION_MODEL_VERSION,PHASE_CONTINUATION_MODEL_VERSION
from run_analysis.adaptive.adaptive_stage_lifecycle import advance_stage


def check_replay(h,event):
    state=restore((h.sandbox.root/event['snapshot_file']).read_text(),expected_fingerprint=event['snapshot_fingerprint'])
    assert len(state.context.continuation_boundary_references)==3
    planned=tuple(planned_accounted(f'candidate:{day}:{key}',
        datetime.combine(date.fromisoformat(day),time(18),ZoneInfo(state.timezone)),
        h.catalog[key],event['forecast_handoff_by_dose'][key],h.ref,available=state.as_of)
        for day,key in event['program'])
    replay=evaluate_snapshot(state,planned,tuple(k for _,k in event['program']),reference=h.ref)
    assert replay['evaluation'].total_cost==pytest.approx(event['cost'],abs=1e-12)
    assert replay['evaluation'].fingerprint==event['selected']['score_fingerprint']
    assert replay['decision_eligible']==event['selected']['eligible']
    assert replay['suitability']==event['selected']['suitability']
    expected=PHASE_CONTINUATION_MODEL_VERSION if h.policy.purpose_continuation_phase=='known_full_phase_v1' else PURPOSE_CONTINUATION_MODEL_VERSION if h.policy.purpose_continuation else (
        MEMORY_CONTINUATION_MODEL_VERSION if h.policy.continuation_guard_days else CONTINUATION_MODEL_VERSION)
    if h.policy.continuation_distribution=='c1_v1':
        expected+=':c1_handoff_v1'
    assert replay['evaluation'].model_version==event['model_version']==expected
    assert len(replay['evaluation'].conditional_ledger)==len(state.history)+len(planned)


@pytest.mark.parametrize('purpose_continuation,phase',[(False,None),(True,None),(True,'uniform_v1'),(True,'known_full_phase_v1')])
def test_initial_upload_rollover_restart_and_growth_keep_same_contract(tmp_path,monkeypatch,purpose_continuation,phase):
    h,inputs,config,at=completed_stage(tmp_path,monkeypatch,growth_focus='easy_volume',continuation_boundary=True,
                                    purpose_continuation=purpose_continuation,purpose_continuation_phase=phase)
    expected_phase=phase or ('known_full_phase_v1' if purpose_continuation else 'uniform_v1')
    assert h.policy.purpose_continuation_phase==expected_phase
    try:
        check_replay(h,h.events[0])
        # Fixture has uploaded actual traces since initial publication.
        check_replay(h,h.replan(at.astimezone(ZoneInfo('America/New_York')),reason='post_upload'))
        check_replay(h,h.replan(at+timedelta(days=1),reason='daily_rollover'))
        root=h.sandbox.root
    finally:
        h.sandbox.close()
    reopened=DailyHarness.resume(root,config,as_of=at+timedelta(days=1))
    try:
        assert reopened.policy.continuation_boundary
        assert reopened.policy.purpose_continuation==purpose_continuation
        assert reopened.policy.purpose_continuation_phase==expected_phase
        assert reopened.policy.continuation_guard_days==7
        ctx=reopened.context(at+timedelta(days=1))
        assert (ctx.hard_end-ctx.decision).days==21
        assert (ctx.taper_start-ctx.decision).days==40
        assert (ctx.end-ctx.decision).days==54
        check_replay(reopened,reopened.events[-1])
        check_replay(reopened,reopened.replan(at+timedelta(days=2),reason='restored_rollover'))
    finally:
        reopened.sandbox.close()


@pytest.mark.parametrize('purpose_continuation,phase',[(False,None),(True,None),(True,'uniform_v1'),(True,'known_full_phase_v1')])
def test_stage_acceptance_revalidates_integrated_score(tmp_path,monkeypatch,purpose_continuation,phase):
    h,inputs,_,at=completed_stage(tmp_path,monkeypatch,growth_focus='easy_volume',continuation_boundary=True,
                               purpose_continuation=purpose_continuation,purpose_continuation_phase=phase)
    try:
        updated,review,event=advance_stage(h,inputs,at)
        assert event['stage_acceptance']
        assert updated.policy.continuation_boundary
        assert updated.policy.purpose_continuation_phase==h.policy.purpose_continuation_phase
        check_replay(updated,event)
    finally:
        h.sandbox.close()


def test_fine_command_forecast_publication_and_restart(tmp_path,monkeypatch):
    h,inputs,config,at=completed_stage(tmp_path,monkeypatch,growth_focus='easy_volume',continuation_boundary=True)
    import run_analysis.adaptive.new_model_daily_harness as module
    try:
        easy=sorted((d for d in h.catalog.values() if d.role=='easy'),key=lambda d:d.miles)
        assert [round(d.miles*10) for d in easy]==list(range(round(easy[0].miles*10),round(easy[-1].miles*10)+1))
        fine=easy[1]
        def selector(**kwargs):
            assert fine.key in kwargs['keys']
            assert len(kwargs['exploration_keys'])<len(kwargs['keys'])
            program=tuple((day,fine.key if h.catalog[key].role=='easy' else key)
                for day,key in kwargs['previous'] if day in kwargs['dates'])
            result=kwargs['evaluate'](program)
            assert result['eligible']
            return dict(program=program,selected=result,cost=result['cost'],seconds=0.,evaluations=1)
        monkeypatch.setattr(module,'migrated_search',selector)
        event=h.replan(at,reason='fine dose wiring control')
        assert any(key==fine.key for _,key in event['program'])
        check_replay(h,event)
        expected=h.catalog
        root=h.sandbox.root
    finally:
        h.sandbox.close()
    reopened=DailyHarness.resume(root,config,as_of=at)
    try:
        assert reopened.catalog==expected
        check_replay(reopened,reopened.events[-1])
    finally:
        reopened.sandbox.close()
