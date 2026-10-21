"""Wiring controls, NOT daily stability tests or optimizer-quality evidence."""
from dataclasses import replace
from datetime import datetime,timedelta,timezone
import pytest

from run_analysis.config import load_config
from run_analysis.continuous_training_dose import ContinuousDoseReference
from run_analysis.athlete_projection import SessionEvidence
from run_analysis.signal_structure import SignalSpan
from run_analysis.staged_progression import DevelopmentStage,ProgressionPolicy,MILE
from run_analysis.adaptive.new_model_daily_harness import accounted,point,synthetic_payload,DailyHarness
from run_analysis.adaptive.direct_quality_load import PhaseHistory,phase_observations
from run_analysis.adaptive.execution_forecast_contract import layout
from run_analysis.adaptive.adaptive_candidate_pipeline import compile_athlete,supporting_easy_target
from run_analysis.adaptive.adaptive_stage_lifecycle import advance_stage,review_entries
from run_analysis.adaptive.stage_acceptance import StageLedger
from run_analysis.adaptive.continuous_search_adapter import NoEligibleProgram
from scripts.run_adaptive_candidate_plan import build


def supplied_seed_selector(**kwargs):
    # Price known complete seeds through the REAL admission/evaluator boundary.
    # This fixture exercises lifecycle/atomic acceptance, not search coverage.
    candidates=[(kwargs['evaluate'](p),p) for p in kwargs['seed_programs']]
    admitted=[(r,p) for r,p in candidates if r['eligible']]
    if not admitted:
        raise NoEligibleProgram('fixture seeds did not fund the stage')
    result,program=min(admitted,key=lambda item:(item[0]['cost'],item[1]))
    return dict(program=program,selected=result,cost=result['cost'],seconds=0.,evaluations=len(candidates),
        local_neighborhood_complete=False,budget_exhausted=False,search='explicit_seed_fixture_not_behavioral_acceptance')


def test_decimal_supporting_share_is_not_rounded_down_by_float_noise():
    from types import SimpleNamespace
    stage=SimpleNamespace(weekly_equivalent_miles=20.,long_miles=8.4,ordinary_miles=4.4)
    assert supporting_easy_target(stage,3.9)==7.7


def test_sustained_growth_uses_ordinary_admission_and_preserves_contract_on_restart(tmp_path,monkeypatch):
    h,inputs,config,at=completed_stage(tmp_path,monkeypatch,growth_focus='easy_volume')
    import run_analysis.adaptive.new_model_daily_harness as module
    calls=[]
    def ordinary_seed_fixture(**kwargs):
        # Deliberately remove the soft projection to expose old aggregate
        # rejection. This fixture tests admission, NOT optimizer quality.
        seeds=[tuple(p for p in seed if p[0]<kwargs['hard_end']) for seed in kwargs['seed_programs']]
        result=supplied_seed_selector(**dict(kwargs,seed_programs=seeds))
        calls.append(result['selected'])
        return result
    monkeypatch.setattr(module,'migrated_search',ordinary_seed_fixture)
    try:
        assert h.policy.delivery_contract=='sustained_v1'
        with pytest.raises(ValueError,match='ordinary admission'):
            h.replan(at,reason='wrong mode',require_stage_funding=True)
        updated,review,event=advance_stage(h,inputs,at)
        import json
        assert event['stage_acceptance']
        evidence=json.loads(updated.sandbox.db.execute('SELECT evidence FROM stages WHERE key=?',
            (updated.stage.key,)).fetchone()[0])
        assert evidence['delivery_contract']=='sustained_v1'
        assert evidence['aggregate_funding_diagnostic'] is False
        assert calls[-1]['eligible']
        updated.sandbox.close()
        reopened=DailyHarness.resume(tmp_path/'state',config,as_of=at)
        try:
            assert reopened.policy.delivery_contract=='sustained_v1'
            assert reopened.context(at).volume_smoothing_days==7
        finally:
            reopened.sandbox.close()
    finally:
        h.sandbox.close()


def test_growth_rejects_mismatched_selection_proof_and_leaves_stage_unchanged(tmp_path,monkeypatch):
    h,inputs,_,at=completed_stage(tmp_path,monkeypatch,growth_focus='easy_volume')
    import run_analysis.adaptive.new_model_daily_harness as module
    def bad_proof(**kwargs):
        result=supplied_seed_selector(**kwargs)
        result['selected']=dict(result['selected'],score_fingerprint='not-this-program')
        return result
    monkeypatch.setattr(module,'migrated_search',bad_proof)
    before=h.stage
    try:
        with pytest.raises(ValueError):
            advance_stage(h,inputs,at)
        assert StageLedger.for_sandbox(h.sandbox).current==before
    finally:
        h.sandbox.close()


def completed_stage(tmp_path,monkeypatch,*,exclude_first=False,growth_focus='long_and_volume',continuation_boundary=False,gradual=False,single_volume=False,volume_linear_coefficient=None,prospective_review_hour=None,purpose_continuation=False,purpose_continuation_phase=None,coordinated_final_evaluations=0,continuation_distribution='linear_v1',weekly_volume=20.,long_miles=8.4,ordinary_miles=4.4,easy_days=(4,)):
    import run_analysis.adaptive.new_model_daily_harness as module
    monkeypatch.setattr(module,'migrated_search',supplied_seed_selector)
    config=dict(load_config(),resting_hr=50,max_hr=200,timezone_default='UTC')
    origin=datetime(2026,9,27,tzinfo=timezone.utc)
    ref=ContinuousDoseReference('sandbox upload accounting')
    work=(('warmup',600.),('work',300.),('recovery',120.),('work',300.),
          ('recovery',120.),('work',300.),('cooldown',600.))
    history,phases,sessions=[],[],[]
    for offset in range(-28,0):
        role={0:'long',2:'quality',**{d:'easy' for d in easy_days}}.get(offset%7)
        if role is None:
            continue
        miles={'long':long_miles,'quality':3.9,'easy':ordinary_miles}[role]
        steps=work if role=='quality' else (('warmup',miles*600),)
        seconds=sum(s for _,s in steps)
        start=origin+timedelta(days=offset,hours=16)
        end=start+timedelta(seconds=seconds)
        key='seed:'+str(offset)
        r=accounted(key,start,miles,steps,point(seconds/60),role,ref,source='observed',available=end)
        history.append(r)
        phases.append(PhaseHistory(key,start,end,end,ref.accounting_key,
            phase_observations(layout(steps),(SignalSpan(0,seconds,.6,.6),))))
        sessions.append(SessionEvidence(key,start,end,end,'intervals' if role=='quality' else role,
            miles*MILE,seconds,r.demand[1],ref.accounting_key))
    if exclude_first:
        sessions[0]=replace(sessions[0],capacity_eligible=False)
    inputs=dict(history=tuple(history),phase_history=tuple(phases),sessions=tuple(sessions),reference=ref)
    from run_analysis.staged_progression import tenth
    policy=ProgressionPolicy(14,.05,growth_focus,tenth(weekly_volume*1.05) if growth_focus=='easy_volume' else None,
        'gradual' if gradual else 'step')
    stage=DevelopmentStage('fixture-stage',origin,weekly_volume,ordinary_miles,long_miles,max(17.,long_miles),7.,policy,'fixture')
    candidate=compile_athlete(inputs,origin,stage=stage)
    h=build(tmp_path/'state',config,inputs,candidate,origin,evaluations=2000,
        continuation_boundary=continuation_boundary,soft_horizon_taper=continuation_boundary,single_volume=single_volume,
        purpose_continuation=purpose_continuation,purpose_continuation_phase=purpose_continuation_phase,
        volume_linear_coefficient=volume_linear_coefficient,prospective_review_hour=prospective_review_hour,
        coordinated_final_evaluations=coordinated_final_evaluations,continuation_distribution=continuation_distribution)
    event=h.replan(origin,reason='explicit lifecycle fixture seed')
    for offset in range(14):
        day=(origin+timedelta(days=offset)).date().isoformat()
        issued=h.active.get(day)
        if issued is None:
            continue
        dose=h.catalog[issued['dose']]
        payload,observed,end=synthetic_payload(dose,origin+timedelta(days=offset,hours=16),hr=140)
        h.upload(payload,day=day,available_at=end,observed_role=dose.role,observed_phases=observed)
    return h,inputs,config,origin+timedelta(days=14)


def test_capacity_exclusion_survives_snapshot_and_restart_without_erasing_load(tmp_path,monkeypatch):
    from run_analysis.adaptive.decision_snapshot import restore
    h,inputs,config,at=completed_stage(tmp_path,monkeypatch,exclude_first=True)
    key=inputs['sessions'][0].key
    try:
        event=h.events[0]
        state=restore((h.sandbox.root/event['snapshot_file']).read_text(),
            expected_fingerprint=event['snapshot_fingerprint'])
        assert not next(r for r in state.athlete.sessions if r.key==key).capacity_eligible
        assert next(r for r in state.history if r.key==key).demand[1].low>0
        root=h.sandbox.root
    finally:
        h.sandbox.close()
    from run_analysis.adaptive.factory import resume_service
    service=resume_service(root,config,as_of=at)
    resumed=service.harness
    try:
        assert resumed.capacity_eligibility[key] is False
        assert any(r.key==key for r in resumed.history)
        assert not next(r for r in service.initial_inputs['sessions'] if r.key==key).capacity_eligible
        assert any(r.role=='intervals' for r in resumed.session_evidence())
    finally:
        resumed.sandbox.close()


def test_fresh_experiment_preserves_locally_completed_issued_credit(tmp_path,monkeypatch):
    from run_analysis.adaptive.review_seed import completed_seed,validate_seed,phase_seed
    h,inputs,config,at=completed_stage(tmp_path,monkeypatch)
    try:
        assert not h.seed_review_entries
        actual=tuple(row for row in review_entries(h,at) if row.completion is not None)
        carried=completed_seed(h,at)
        assert carried==actual and carried
        assert all(row.contract.issued_at<row.completion.available_at for row in carried)
        assert validate_seed(carried,h.history,at)==carried
        seeded=phase_seed(h,at)
        assert {row.completion.key for row in carried}<={r.key for r in seeded}
        assert len(seeded)==len(h.phase_history)+len(h.sandbox.completed_history(at))
        # A second fork must retain inherited evidence without duplicating it.
        h.seed_review_entries=carried
        assert completed_seed(h,at)==carried
        fork=DailyHarness(tmp_path/'fresh',config,origin=at,catalog=tuple(h.catalog.values()),
            history=h.history,phase_history=seeded,scales=h.scales,training_policy=h.training_policy,
            search_policy=h.policy,demand_scenario=h.demand_scenario,exchange=h.exchange,timing=h.timing,
            coaching_contracts=h.coaching_contracts,easy_duration_policy=h.easy_duration_policy,
            stage=h.stage,target_history=h.target_history,seed_review_entries=carried,
            decision_source='Explicit complete-evidence fork control')
        try:
            dose=next(d for d in fork.catalog.values() if d.role=='easy')
            day=at.date().isoformat()
            intent=fork.intent(dose,at.date(),at,'slot:'+day)
            record=fork.sandbox.publish(intent,seeded)
            fork.events.append(dict(as_of=at.isoformat(),active={day:dict(
                fingerprint=record.fingerprint,dose=dose.key)}))
            payload,observed,end=synthetic_payload(dose,at+timedelta(hours=16))
            result=fork.sandbox.upload(payload,fingerprint=record.fingerprint,available_at=end,
                observed_role='easy',observed_phases=observed,phase_evidence='explicit')
            key=result['comparison']['actual_key']
            assert key not in {r.key for r in h.history}
            fork.sync_completions(end)
            assert len(fork.history)==len(h.history)+1
            assert len(completed_seed(fork,end+timedelta(seconds=1)))==len(carried)+1
        finally:
            fork.sandbox.close()
    finally:
        h.sandbox.close()


def test_stage_review_compiles_revalidates_and_publishes_atomically(tmp_path,monkeypatch):
    h,inputs,config,at=completed_stage(tmp_path,monkeypatch)
    original=h.stage
    try:
        entries=review_entries(h,at)
        assert any(e.completion and e.contract.purpose=='high_aerobic' for e in entries)
        updated,review,event=advance_stage(h,inputs,at)
        assert review['status']=='conditional_development_candidate'
        assert updated.stage.weekly_equivalent_miles==21.
        assert updated.stage.long_miles==8.8
        assert updated.catalog['quality:ordinary']==h.catalog['quality:ordinary']
        assert event['stage_acceptance']
        assert StageLedger.for_sandbox(h.sandbox).current==updated.stage
        assert h.stage==original
        assert len(updated.target_history)==2
        h.sandbox.close()
        reopened=DailyHarness.resume(tmp_path/'state',config,as_of=at)
        try:
            assert reopened.stage==updated.stage
            assert reopened.target_history==updated.target_history
            assert reopened.previous==updated.previous
        finally:
            reopened.sandbox.close()
    finally:
        h.sandbox.close()


def test_unfound_growth_candidate_holds_target_not_an_unscored_fallback(tmp_path,monkeypatch):
    h,inputs,_,at=completed_stage(tmp_path,monkeypatch)
    import run_analysis.adaptive.new_model_daily_harness as module
    def no_candidate(**kwargs):
        raise NoEligibleProgram('bounded fixture search misses all funded seeds')
    monkeypatch.setattr(module,'migrated_search',no_candidate)
    before=len(h.events)
    try:
        same,review,event=advance_stage(h,inputs,at)
        assert same is h and event is None
        assert review['status']=='growth_search_no_eligible_candidate'
        assert StageLedger.for_sandbox(h.sandbox).current==h.stage
        assert len(h.events)==before
    finally:
        h.sandbox.close()


def test_prospective_bootstrap_history_survives_publication_and_resume(tmp_path,monkeypatch):
    h,inputs,config,_=completed_stage(tmp_path,monkeypatch)
    baseline=compile_athlete(inputs,h.origin,stage=h.stage)
    new=replace(h.stage,key='prospective',weekly_equivalent_miles=21.)
    candidate=compile_athlete(inputs,h.origin,stage=new)
    h.sandbox.close()
    proposed=build(tmp_path/'prospective',config,inputs,candidate,h.origin,
                   evaluations=2000,baseline_candidate=baseline)
    try:
        proposed.replan(h.origin,reason='prospective bootstrap fixture')
        targets=proposed.target_history
        assert len(targets)==2
        assert targets[0].volume_per_7==20.
        assert targets[1].volume_per_7==21.
        assert targets[1].effective_on==h.origin.date()
        assert StageLedger.for_sandbox(proposed.sandbox).current==new
    finally:
        proposed.sandbox.close()
    resumed=DailyHarness.resume(tmp_path/'prospective',config,as_of=h.origin+timedelta(days=1))
    try:
        assert resumed.stage==new
        assert resumed.target_history==targets
    finally:
        resumed.sandbox.close()


def test_easy_volume_stage_is_funded_accepted_and_resumed_without_long_or_quality_growth(tmp_path,monkeypatch):
    h,inputs,config,at=completed_stage(tmp_path,monkeypatch,growth_focus='easy_volume')
    import run_analysis.adaptive.adaptive_stage_lifecycle as lifecycle
    compile_original=lifecycle.compile_athlete
    def compile_with_other_defaults(*args,**kwargs):
        candidate=compile_original(*args,**kwargs)
        # Prove future compiler defaults cannot reset accepted timing/weights.
        return dict(candidate,training_policy=replace(candidate['training_policy'],
            long_per_7=.875,quality_per_7=.875,purpose_weights=(2.,2.,2.,2.)))
    monkeypatch.setattr(lifecycle,'compile_athlete',compile_with_other_defaults)
    original=h.stage
    try:
        updated,review,event=advance_stage(h,inputs,at)
        assert review['status']=='conditional_development_candidate'
        assert updated.stage.weekly_equivalent_miles==21.
        assert updated.stage.long_miles==original.long_miles==8.4
        assert updated.catalog['quality:ordinary']==h.catalog['quality:ordinary']
        assert updated.training_policy==replace(h.training_policy,weekly_miles=21.)
        assert updated.easy_duration_policy.supporting_easy_miles_per_7==pytest.approx(
            h.easy_duration_policy.supporting_easy_miles_per_7+1.)
        assert event['stage_acceptance']
        assert StageLedger.for_sandbox(h.sandbox).current==updated.stage
        assert h.stage==original
        assert updated.target_history[0]==h.target_history[0]
        assert updated.target_history[-1].volume_per_7==21.
        assert updated.target_history[-1].effective_on==at.date()
        h.sandbox.close()
        resumed=DailyHarness.resume(tmp_path/'state',config,as_of=at)
        try:
            assert resumed.stage==updated.stage
            assert resumed.stage.policy.growth_focus=='easy_volume'
            assert resumed.stage.policy.volume_goal_miles==21.
            assert resumed.target_history==updated.target_history
        finally:
            resumed.sandbox.close()
    finally:
        h.sandbox.close()


def test_easy_volume_no_funded_candidate_does_not_publish_or_accept(tmp_path,monkeypatch):
    h,inputs,_,at=completed_stage(tmp_path,monkeypatch,growth_focus='easy_volume')
    import run_analysis.adaptive.new_model_daily_harness as module
    def no_candidate(**kwargs):
        raise NoEligibleProgram('bounded test failure')
    monkeypatch.setattr(module,'migrated_search',no_candidate)
    try:
        before=len(h.events)
        same,review,event=advance_stage(h,inputs,at)
        assert same is h and event is None
        assert review['status']=='growth_search_no_eligible_candidate'
        assert StageLedger.for_sandbox(h.sandbox).current==h.stage
        assert len(h.events)==before
    finally:
        h.sandbox.close()


@pytest.mark.parametrize('single_volume',[False,True])
def test_gradual_confirmation_preserves_scoring_path_and_resume(tmp_path,monkeypatch,single_volume):
    h,inputs,config,at=completed_stage(tmp_path,monkeypatch,growth_focus='easy_volume',
        gradual=True,continuation_boundary=True,single_volume=single_volume)
    try:
        before=h.replan(at,reason='pre-confirmation comparison',publish=False)
        updated,review,event=advance_stage(h,inputs,at)
        assert review['status']=='conditional_development_candidate'
        assert 40.<review['required_miles']<42.
        assert updated.stage.weekly_equivalent_miles==21.
        assert updated.stage.long_miles==8.4
        assert updated.target_history==h.target_history
        assert updated.catalog==h.catalog
        assert updated.scales==h.scales
        assert (updated.training_policy.purpose_weights[1]==0)==single_volume
        assert updated.context(at).stage_start==h.context(at).stage_start
        assert updated.context(at).volume_targets==h.context(at).volume_targets
        assert updated.context(at).support_targets==h.context(at).support_targets
        assert event['cost']==pytest.approx(before['search']['cost'],abs=1e-12)
        assert tuple(map(tuple,event['program']))==before['program']
        from run_analysis.staged_progression import review_recent_delivery
        complete=review_recent_delivery(updated.stage,(),at+timedelta(days=14),
            ledger_complete=True,longest_prior_30d_miles=8.4)
        assert complete['status']=='volume_phase_complete_reassess_next_stage'
        updated.sandbox.close()
        resumed=DailyHarness.resume(tmp_path/'state',config,as_of=at)
        try:
            assert resumed.target_history==updated.target_history
            assert resumed.context(at)==updated.context(at)
        finally:
            resumed.sandbox.close()
    finally:
        h.sandbox.close()
