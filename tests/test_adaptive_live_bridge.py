"""Integration guards, not additional optimizer/stability acceptance."""
from datetime import datetime,timedelta,timezone
from types import SimpleNamespace
import pytest

from run_analysis.adaptive.live_bridge import weather_inputs,enabled,ENGINE
from run_analysis.program_constraints import DateRestriction


AT=datetime(2026,10,3,12,tzinfo=timezone.utc)


def test_bridge_is_not_enabled_implicitly():
    assert not enabled({})
    assert not enabled({'adaptive_planner':{'engine':'unknown'}})
    assert enabled({'adaptive_planner':{'engine':ENGINE}})


def test_zone_update_changes_only_prospective_signature_and_survives_resume(tmp_path,monkeypatch):
    from copy import deepcopy
    from test_adaptive_stage_lifecycle import completed_stage
    from run_analysis.adaptive.live_bridge import input_signature
    from run_analysis.adaptive.new_model_daily_harness import DailyHarness
    h,_,config,at=completed_stage(tmp_path,monkeypatch)
    try:
        original=deepcopy(config)
        captured=SimpleNamespace(contexts=(),restrictions=(),responses=(),metadata=())
        base_signature=input_signature(h,captured,at)
        from dataclasses import replace
        from zoneinfo import ZoneInfo
        original_history=h.history
        h.history=[replace(r,available_at=r.available_at.astimezone(ZoneInfo('America/New_York')))
            for r in h.history]
        assert input_signature(h,captured,at)==base_signature
        h.history=original_history
        h.quality_runtime=dict(evidence=dict(config=original))
        assert input_signature(h,captured,at)==base_signature
        snapshot=(h.sandbox.root/h.events[-1]['snapshot_file']).read_bytes()
        history,stage,targets=tuple(h.history),h.stage,h.target_history
        issues=tuple(tuple(r) for r in h.sandbox.db.execute('SELECT * FROM sandbox_issues'))
        new=deepcopy(config);new['zones']['z2']=[140,153]
        h.config=new
        changed=input_signature(h,captured,at)
        assert changed!=base_signature and input_signature(h,captured,at)==changed
        assert (tuple(h.history),h.stage,h.target_history)==(history,stage,targets)
        assert tuple(tuple(r) for r in h.sandbox.db.execute('SELECT * FROM sandbox_issues'))==issues
        restored=DailyHarness.resume(h.sandbox.root,new,as_of=at)
        try:
            assert restored.config['zones']==new['zones']
            assert (tuple(restored.history),restored.stage,restored.target_history)==(history,stage,targets)
            assert (h.sandbox.root/h.events[-1]['snapshot_file']).read_bytes()==snapshot
        finally:restored.sandbox.close()
    finally:h.sandbox.close()


def test_provider_failure_cannot_clear_existing_dated_hazard(monkeypatch):
    day=AT.date()
    hazard=DateRestriction('app:extreme-weather:'+day.isoformat(),day,day+timedelta(days=1),
        AT-timedelta(hours=1),'unsafe_weather','known prior guard')
    h=SimpleNamespace(decision=lambda now:now.date(),zone=timezone.utc,
        policy=SimpleNamespace(hard_days=2),restrictions=(hazard,))
    import run_analysis.forecast as forecasts
    monkeypatch.setattr(forecasts,'planned_forecast_options',lambda db,cfg,root,groups:
        [[(at,None) for at in group] for group in groups])
    guards,_,_=weather_inputs(None,{},None,h,AT)
    assert guards==(hazard,)
    guards,_,_=weather_inputs(None,{},None,h,AT+timedelta(days=1))
    assert guards==()


def test_weather_suggestion_uses_ranked_remaining_options_not_accounting_clock(monkeypatch):
    from run_analysis.web.schemas import PlannedWeather
    import run_analysis.forecast as forecasts
    h=SimpleNamespace(decision=lambda now:now.date(),zone=timezone.utc,
        policy=SimpleNamespace(hard_days=2),restrictions=())
    groups_seen=[]
    def options(db,cfg,root,groups):
        groups_seen.extend(groups)
        return [[(at,PlannedWeather(forecast_time=at,temperature_f=55,apparent_temperature_f=55,
            dewpoint_f=40,wind_speed_mph=0)) for at in group] for group in groups]
    monkeypatch.setattr(forecasts,'planned_forecast_options',options)
    guards,weather,times=weather_inputs(None,{},None,h,AT)
    assert not guards
    assert times[AT.date().isoformat()].hour==18  # 07:00/12:00 have elapsed.
    tomorrow=(AT+timedelta(days=1)).date().isoformat()
    assert times[tomorrow].hour==7
    assert weather[tomorrow].forecast_time==times[tomorrow]
    assert all(at>AT+timedelta(minutes=10) for at in groups_seen[0])


def test_safe_alternate_time_is_suggested_and_all_extreme_day_still_blocked(monkeypatch):
    from run_analysis.web.schemas import PlannedWeather
    import run_analysis.forecast as forecasts
    h=SimpleNamespace(decision=lambda now:now.date(),zone=timezone.utc,
        policy=SimpleNamespace(hard_days=2),restrictions=())
    def options(db,cfg,root,groups):
        result=[]
        for index,group in enumerate(groups):
            rows=[(at,PlannedWeather(forecast_time=at,apparent_temperature_f=
                55 if index==0 and at.hour==19 else 130,temperature_f=130)) for at in group]
            result.append(sorted(rows,key=forecasts._forecast_rank))
        return result
    monkeypatch.setattr(forecasts,'planned_forecast_options',options)
    guards,weather,times=weather_inputs(None,{},None,h,AT)
    assert times[AT.date().isoformat()].hour==19
    assert weather[AT.date().isoformat()].forecast_time==times[AT.date().isoformat()]
    assert len(guards)==1 and guards[0].start==(AT+timedelta(days=1)).date()


def test_explicit_release_preserves_journal_and_survives_restart(tmp_path, monkeypatch):
    from test_adaptive_stage_lifecycle import completed_stage
    from run_analysis.adaptive.live_bridge import continuation_release_policy, persist_release_policy, runtime_version
    from run_analysis.adaptive.new_model_daily_harness import DailyHarness
    from run_analysis.adaptive.decision_snapshot import restore
    h, _, config, at = completed_stage(tmp_path, monkeypatch, growth_focus='easy_volume',
        gradual=True, continuation_boundary=True, single_volume=True,
        purpose_continuation=True, purpose_continuation_phase='known_full_phase_v1')
    try:
        root = h.sandbox.root
        prior_bytes = (root/h.events[0]['snapshot_file']).read_bytes()
        prior_completions = [tuple(r) for r in h.sandbox.completion_rows(at)]
        physical, targets, stage = tuple(h.history), h.target_history, h.stage
        assert runtime_version(h) == 201
        h.policy = continuation_release_policy(h.policy)
        with pytest.raises(ValueError, match='committed C1'):
            persist_release_policy(h)
        event = h.replan(at, reason='explicit live-contract migration fixture')
        persist_release_policy(h)
        assert runtime_version(h) == 202
        assert tuple(h.history) == physical and h.target_history == targets and h.stage == stage
        assert (root/h.events[0]['snapshot_file']).read_bytes() == prior_bytes
        assert [tuple(r) for r in h.sandbox.completion_rows(at)] == prior_completions
        state = restore((root/event['snapshot_file']).read_text(), expected_fingerprint=event['snapshot_fingerprint'])
        assert state.context.continuation_distribution == 'c1_v1'
        policy = h.policy
    finally:
        h.sandbox.close()
    reopened = DailyHarness.resume(root, config, as_of=at)
    try:
        assert reopened.policy == policy
        assert policy.evaluations == 10000 and policy.complete_first_search
        assert policy.final_winner_evaluations == 1000 and policy.coordinated_final_evaluations == 64
        assert tuple(reopened.history) == physical
    finally:
        reopened.sandbox.close()


def test_general_release_changes_only_future_policy_without_search_or_phase_reset(tmp_path, monkeypatch):
    from test_adaptive_stage_lifecycle import completed_stage
    from run_analysis.adaptive.live_bridge import general_progression_release_policy,persist_release_policy,runtime_version,continuation_release_policy
    from run_analysis.adaptive.new_model_daily_harness import DailyHarness
    h, _, config, at = completed_stage(tmp_path, monkeypatch, growth_focus='easy_volume',
        gradual=True, continuation_boundary=True, single_volume=True,
        purpose_continuation=True, continuation_distribution='c1_v1')
    try:
        root=h.sandbox.root
        before=h.stage,h.target_history,tuple(h.events),tuple(h.history),h.scales
        snapshot=(root/h.events[0]['snapshot_file']).read_bytes()
        # Match the accepted live allowances before enabling the lifecycle release.
        h.policy=continuation_release_policy(h.policy)
        from dataclasses import replace
        h.policy=replace(h.policy,general_progression='easy_first_coordinated_v1')
        assert runtime_version(h)==203  # Historical future policy remains distinguishable.
        h.policy=general_progression_release_policy(h.policy,h.stage)
        persist_release_policy(h)
        assert runtime_version(h)==204
        assert (h.stage,h.target_history,tuple(h.events),tuple(h.history),h.scales)==before
        assert (root/h.events[0]['snapshot_file']).read_bytes()==snapshot
    finally:
        h.sandbox.close()
    resumed=DailyHarness.resume(root,config,as_of=at)
    try:
        assert resumed.policy.general_progression=='easy_first_coordinated_v2'
        assert (resumed.stage,resumed.target_history,tuple(resumed.history),resumed.scales)==(
            before[0],before[1],before[3],before[4])
        assert len(resumed.events)==len(before[2])
        from run_analysis.adaptive.new_model_daily_harness import dump
        assert dump(resumed.events)==dump(before[2])
        assert (root/h.events[0]['snapshot_file']).read_bytes()==snapshot
        assert runtime_version(resumed)==204
    finally:
        resumed.sandbox.close()


def test_shared_calendar_release_is_explicit_preserves_evidence_and_restart_policy(tmp_path,monkeypatch):
    from dataclasses import replace
    from test_adaptive_stage_lifecycle import completed_stage
    from run_analysis.adaptive.live_bridge import (calendar_merge_release_policy,
        general_progression_release_policy,continuation_release_policy,persist_release_policy,
        runtime_version,input_signature)
    from run_analysis.adaptive.new_model_daily_harness import DailyHarness
    h,_,config,at=completed_stage(tmp_path,monkeypatch,growth_focus='easy_volume',
        gradual=True,continuation_boundary=True,single_volume=True,purpose_continuation=True,
        purpose_continuation_phase='known_full_phase_v1',continuation_distribution='c1_v1')
    try:
        h.policy=general_progression_release_policy(continuation_release_policy(h.policy),h.stage)
        captured=SimpleNamespace(contexts=(),restrictions=(),responses=(),metadata=())
        signature=input_signature(h,captured,at)
        before=(h.stage,h.target_history,tuple(h.history),h.scales)
        root=h.sandbox.root;old=(root/h.events[0]['snapshot_file']).read_bytes()
        old_completions=[tuple(r) for r in h.sandbox.completion_rows(at)]
        previous=h.policy
        h.policy=calendar_merge_release_policy(h.policy,h.stage)
        assert h.policy==replace(previous,calendar_merge_evaluations=32)
        assert input_signature(h,captured,at)!=signature
        assert runtime_version(h)==205
        h.replan(at,reason='explicit shared-search migration control')
        persist_release_policy(h)
        assert (h.stage,h.target_history,tuple(h.history),h.scales)==before
        assert (root/h.events[0]['snapshot_file']).read_bytes()==old
        assert [tuple(r) for r in h.sandbox.completion_rows(at)]==old_completions
        policy=h.policy
    finally:h.sandbox.close()
    reopened=DailyHarness.resume(root,config,as_of=at)
    try:
        assert reopened.policy==policy and runtime_version(reopened)==205
        assert (reopened.stage,reopened.target_history,tuple(reopened.history),reopened.scales)==before
    finally:reopened.sandbox.close()


def test_calendar_release_rejects_incompatible_contract_without_silent_defaults():
    from run_analysis.adaptive.live_bridge import calendar_merge_release_policy
    from run_analysis.adaptive.new_model_daily_harness import HarnessPolicy
    with pytest.raises(ValueError,match='accepted'):
        calendar_merge_release_policy(HarnessPolicy(),None)
