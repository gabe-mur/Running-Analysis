"""Cheap driver orchestration checks; not physiological or search validation."""
from dataclasses import replace
from datetime import date,datetime,timedelta,timezone
from types import SimpleNamespace
from zoneinfo import ZoneInfo
import json

import pytest

from scripts import run_adaptive_lifecycle as driver
from test_staged_progression import stage,history,AT


def fixture_driver(monkeypatch,root,*,fail_day=None,missed_day=None,native_day=None):
    origin=datetime(2026,9,27,8,tzinfo=timezone.utc)
    calls=[]
    class Harness:
        zone=ZoneInfo('America/New_York')
        last_as_of=None
        active={}
        quality_runtime=None
        def __init__(self):
            self.stage=stage(issued_at=origin,weekly_equivalent_miles=18.6)
            self.events=[]
            self.sandbox=SimpleNamespace(close=lambda:None)
            if missed_day is not None:
                self.active={(origin+timedelta(days=missed_day)).date().isoformat():{'dose':'easy:test','fingerprint':'issued-test'}}
        def summary(self,days,uploads):
            return dict(days=days,events=list(self.events),uploads=uploads)
    h=Harness()
    if native_day is not None:
        h.quality_runtime=object()
        h.catalog={k:SimpleNamespace(role='quality',miles=3.,workout_definition=dict(
            family=k,recipe_key=k,structure=[k])) for k in ('old','new')}
    class Service:
        def __init__(self,harness,inputs):
            self.harness=harness
        def refresh(self,at,*,reason):
            offset=(at-origin).days
            calls.append(offset)
            h.last_as_of=at
            if offset>=14:
                h.stage=replace(h.stage,weekly_equivalent_miles=19.5)
            edit=dict(day='fixture',date_changed=True,role_changed=False,distance_change=None)
            event=dict(index=len(h.events),as_of=at.isoformat(),reason=reason,seconds=0.,
                local_neighborhood_complete=True,volume=50.,churn={'4':[edit] if offset==fail_day else [],'7':[]})
            if native_day is not None:
                from run_analysis.adaptive.new_model_daily_harness import churn
                program=[(native_day,'new' if offset>=2 else 'old')]
                event.update(decision=at.astimezone(h.zone).date().isoformat(),program=program)
                event['churn']=churn(h.events[-1]['program'] if h.events else (),program,
                    at.astimezone(h.zone).date(),h.catalog)
            h.events.append(event)
            return dict(event=event,stage_review=dict(status='review_due' if offset==14 else 'hold'))
    monkeypatch.setattr(driver,'load_config',lambda:dict(timezone_default='America/New_York'))
    monkeypatch.setattr(driver,'freeze_sources',lambda *a,**k:dict(fixture=True))
    monkeypatch.setattr(driver,'load_athlete',lambda *a:{})
    monkeypatch.setattr(driver,'compile_easy_volume_phase',lambda *a,**k:({'historical':True},{'new':True},{}))
    def build(*a,**k):
        assert k['baseline_candidate']=={'historical':True}
        root.mkdir()
        return h
    monkeypatch.setattr(driver,'build',build)
    monkeypatch.setattr(driver,'StageLedger',SimpleNamespace(for_sandbox=lambda *a,**k:SimpleNamespace(close=lambda:None)))
    monkeypatch.setattr(driver,'AdaptivePlannerService',Service)
    return calls


def test_checkpoint_reviews_boundary_and_continues_same_harness_to_21(tmp_path,monkeypatch):
    root=tmp_path/'run'
    calls=fixture_driver(monkeypatch,root)
    result=driver.run(root,tmp_path/'unused',days=21,checkpoint_days=14,easy_volume_phase=True)
    checkpoint=json.loads((root/'checkpoint-0014.json').read_text())
    assert checkpoint['completed_days']==14
    assert len(checkpoint['events'])==15
    assert checkpoint['stage_reviews'][-1]['status']=='review_due'
    assert checkpoint['final_stage']['weekly_equivalent_miles']==19.5
    assert result['completed_days']==21 and not result['failures']
    assert calls==list(range(21))


def test_whole_run_omission_creates_no_activity_or_upload_refresh(tmp_path,monkeypatch):
    root=tmp_path/'missed'
    calls=fixture_driver(monkeypatch,root,missed_day=1)
    monkeypatch.setattr(driver,'compatible_donor',lambda *a:pytest.fail('Missing run selected an HR donor'))
    result=driver.run(root,tmp_path/'unused',days=3,easy_volume_phase=True,missed_run_offsets=(1,))
    assert calls==[0,1,2]
    assert result['completed_days']==3 and result['uploads']==[]
    assert result['missed_runs'][0]['physical_activity_created'] is False
    assert not list(root.glob('execution-*.json'))
    assert (root/'missed-run-0001.json').exists()


def test_explicit_late_opening_clock_is_reported_without_changing_daily_refreshes(tmp_path,monkeypatch):
    root=tmp_path/'late'
    calls=fixture_driver(monkeypatch,root)
    result=driver.run(root,tmp_path/'unused',days=2,easy_volume_phase=True,opening_run_hour=19)
    assert result['opening_run_hour']==19
    assert calls==[0,1]


def test_perfect_protocol_requires_verified_primary_fulfillment_not_whole_run_labels():
    from types import SimpleNamespace as NS
    h=NS(coaching_base=object(),coaching_receipts=())
    assert driver.verify_prescribed_quality(h,'easy') is None
    receipt=NS(activity_key='q',purpose_matched=True,credit=NS(low=.89),
        assignment=NS(primary_unit='distance'))
    h.coaching_receipts=(receipt,)
    with pytest.raises(ValueError,match='not a perfect-adherence'):driver.verify_prescribed_quality(h,'q')
    receipt.credit.low=1.
    assert driver.verify_prescribed_quality(h,'q')['verified_receipt']
    receipt.purpose_matched=None
    with pytest.raises(ValueError,match='verified primary'):driver.verify_prescribed_quality(h,'q')


@pytest.mark.parametrize('hour',[-1,24,19.5])
def test_invalid_opening_hour_is_rejected_before_building(tmp_path,hour):
    with pytest.raises(ValueError,match='explicit local hour'):
        driver.run(tmp_path/'unused',tmp_path/'unused-db',opening_run_hour=hour)


def test_resume_cannot_change_opening_clock(tmp_path):
    with pytest.raises(ValueError,match='opening execution clock'):
        driver.run(tmp_path/'unused',tmp_path/'unused-db',resume=True,opening_run_hour=19)


def test_matching_generic_sandbox_keeps_strict_config_without_needing_live_bridge_records(tmp_path,monkeypatch):
    config=dict(max_hr=194,resting_hr=46,timezone_default='UTC')
    (tmp_path/'execution-sandbox.json').write_text(json.dumps(dict(measurement_context=json.dumps(dict(max_hr=194,resting_hr=46)))))
    monkeypatch.setattr('run_analysis.adaptive.hr_reference.frozen_live_config',lambda *a,**k:pytest.fail('Generic matching resume must not use adopted-reference migration'))
    assert driver.replay_config(config,tmp_path) is config


def test_changed_hr_requires_the_verified_original_accounting_reference(tmp_path,monkeypatch):
    (tmp_path/'execution-sandbox.json').write_text(json.dumps(dict(measurement_context=json.dumps(dict(max_hr=194,resting_hr=46)))))
    config=dict(max_hr=195,resting_hr=45,timezone_default='UTC')
    calls=[]
    def verified(current,*,root):
        calls.append((current,root))
        return dict(current,max_hr=194,resting_hr=46)
    monkeypatch.setattr('run_analysis.adaptive.hr_reference.frozen_live_config',verified)
    result=driver.replay_config(config,tmp_path)
    assert result['max_hr']==194 and result['resting_hr']==46
    assert config['max_hr']==195 and calls==[(config,tmp_path)]


@pytest.mark.parametrize('fail_day',[2,14])
def test_fast_failure_stops_before_checkpoint_and_continuation(tmp_path,monkeypatch,fail_day):
    root=tmp_path/'run'
    calls=fixture_driver(monkeypatch,root,fail_day=fail_day)
    result=driver.run(root,tmp_path/'unused',days=21,checkpoint_days=14,easy_volume_phase=True)
    assert result['completed_days']==fail_day
    assert result['status']=='stopped_for_attribution'
    assert calls==list(range(fail_day+1))
    assert not (root/'checkpoint-0014.json').exists()


def test_distance_fast_gate_unchanged_and_day_five_edits_remain_report_only():
    def event(value):
        edit=dict(date_changed=False,role_changed=False,distance_change=value)
        return dict(churn={'4':[edit],'7':[dict(edit,date_changed=True)]})
    assert not driver.fast_fail_edits(event(.5))
    assert driver.fast_fail_edits(event(.6))
    assert driver.fast_fail_edits(event(-.6))


@pytest.mark.parametrize('native_day,stops',[('2026-09-29',True),('2026-10-05',False)])
def test_native_driver_gates_only_previously_visible_family_swaps(tmp_path,monkeypatch,native_day,stops):
    root=tmp_path/'native'
    fixture_driver(monkeypatch,root,native_day=native_day)
    result=driver.run(root,tmp_path/'unused',days=3,easy_volume_phase=True,stability_window=7)
    assert bool(result['failures']) is stops
    assert result['completed_days']==(2 if stops else 3)


def test_shared_easy_phase_compiler_sets_bounded_focus_without_changing_baseline(monkeypatch):
    import run_analysis.adaptive.adaptive_candidate_pipeline as pipeline
    original=stage(weekly_equivalent_miles=10.)
    monkeypatch.setattr(pipeline,'compile_athlete',lambda inputs,as_of,stage=None:dict(stage=stage or original))
    baseline,candidate,proposal=pipeline.compile_easy_volume_phase(dict(sessions=history()),AT,timezone='UTC')
    assert baseline['stage']==original
    assert baseline['stage'].policy.growth_focus=='long_and_volume'
    assert candidate['stage'].policy.growth_focus=='easy_volume'
    assert candidate['stage'].policy.volume_goal_miles>candidate['stage'].weekly_equivalent_miles
    assert candidate['stage'].long_miles==original.long_miles
    assert candidate['stage']==proposal['candidate']


def test_resume_preserves_hr_provenance_and_does_not_overwrite_original_summary(tmp_path):
    import sqlite3
    from run_analysis.adaptive.new_model_daily_harness import dump
    db=sqlite3.connect(':memory:')
    db.row_factory=sqlite3.Row
    db.execute('CREATE TABLE sandbox_completions(comparison TEXT)')
    record=dict(day='2026-09-28',status='consumed',comparison=dict(actual_key='sandbox:1',distance_adherence='within'),
        provenance=dict(source_key='382',source_sha256='original-hr-source',distance_factor=1.08,duration_factor=1.08))
    (tmp_path/'execution-0001.json').write_text(dump(record))
    db.execute('INSERT INTO sandbox_completions VALUES(?)',(json.dumps(record['comparison']),))
    try:
        assert driver.restored_uploads(tmp_path,SimpleNamespace(db=db))==[record]
        assert driver.report_path(tmp_path,1)!=driver.report_path(tmp_path)
        db.execute('INSERT INTO sandbox_completions VALUES(?)',(json.dumps(dict(actual_key='missing')),))
        with pytest.raises(ValueError,match='provenance'):
            driver.restored_uploads(tmp_path,SimpleNamespace(db=db))
    finally:
        db.close()


def test_resume_skips_completed_rest_or_missed_day_without_inventing_upload():
    origin=datetime(2026,9,27,8,tzinfo=timezone.utc)
    h=SimpleNamespace(last_as_of=origin+timedelta(days=2),zone=ZoneInfo('UTC'))
    report=dict(status='bounded_screen_complete',completed_days=3)
    assert driver.resume_offset(h,origin,report)==3
    assert driver.resume_offset(h,origin,dict(status='attributed_driver_stop'))==2
    with pytest.raises(ValueError,match='boundary'):
        driver.resume_offset(h,origin,dict(report,completed_days=4))


def test_explicit_screen_clock_keeps_stage_origin_and_excludes_pre_seed_days():
    h=SimpleNamespace(origin=datetime(2026,10,3,8,tzinfo=timezone.utc),
        last_as_of=datetime(2026,10,9,22,33,tzinfo=timezone.utc),
        zone=ZoneInfo('America/New_York'),events=[dict(index=14)])
    clock,start,baseline=driver.screen_boundary(h,'2026-10-10')
    assert clock==datetime(2026,10,10,8,tzinfo=timezone.utc)
    assert start==date(2026,10,10) and baseline==14
    assert h.origin.date()==date(2026,10,3)
    h.last_as_of=clock+timedelta(days=6,hours=12)
    saved=dict(screen_start='2026-10-10',screen_baseline_event=14,
        status='bounded_screen_complete',completed_days=7)
    assert driver.resume_offset(h,clock,saved)==7
    assert driver.screen_boundary(h,None,saved)==(clock,start,baseline)
    with pytest.raises(ValueError,match='retain its explicit replay start'):
        driver.screen_boundary(h,'2026-10-11',saved)


def test_screen_start_cannot_count_earlier_seed_days_as_new_replay():
    h=SimpleNamespace(last_as_of=datetime(2026,10,9,22,tzinfo=timezone.utc),
        zone=ZoneInfo('America/New_York'),events=[dict(index=14)])
    for day in ('2026-10-03','2026-10-09'):
        with pytest.raises(ValueError,match='follow the latest seeded decision'):
            driver.screen_boundary(h,day)


def test_screen_summary_scopes_uploads_and_churn_without_rewriting_original_events():
    calls=[]
    events=[dict(index=i,churn={'4':[dict(date_changed=True)],'7':[]}) for i in range(4)]
    def summary(days,uploads,*,delivery_start):
        calls.append((days,uploads,delivery_start))
        return dict(events=events,uploads=uploads)
    h=SimpleNamespace(summary=summary,events=events)
    old=dict(day='2026-10-09');new=dict(day='2026-10-10')
    r=driver.screen_summary(h,1,[old,new],date(2026,10,10),2)
    assert calls==[(1,[new],date(2026,10,10))]
    assert [e['index'] for e in r['events']]==[2,3]
    assert r['events'][0]['churn']['4']==[]
    assert r['events'][1]['churn']['4']==events[3]['churn']['4']
    assert events[2]['churn']['4']  # Archive not mutated to manufacture stability.
    assert r['screen_baseline_event']==2 and r['initial_search_reused']


def test_inherited_completion_exemption_is_exact_and_does_not_hide_new_upload(tmp_path):
    import sqlite3
    db=sqlite3.connect(':memory:');db.row_factory=sqlite3.Row
    db.execute('CREATE TABLE sandbox_completions(comparison TEXT)')
    inherited=dict(actual_key='seed:1',source='original')
    new=dict(actual_key='new:1',source='measured')
    (tmp_path/'replay-inherited-completions.json').write_text(json.dumps([inherited]))
    for r in (inherited,new):db.execute('INSERT INTO sandbox_completions VALUES(?)',(json.dumps(r),))
    sandbox=SimpleNamespace(db=db)
    try:
        with pytest.raises(ValueError,match='one archived execution provenance'):
            driver.restored_uploads(tmp_path,sandbox)
        archived=dict(day='2026-10-10',comparison=new,provenance=dict(original_hr=True))
        (tmp_path/'execution-0001.json').write_text(json.dumps(archived))
        assert driver.restored_uploads(tmp_path,sandbox)==[archived]
        db.execute('UPDATE sandbox_completions SET comparison=? WHERE comparison=?',
            (json.dumps(dict(inherited,source='changed')),json.dumps(inherited)))
        with pytest.raises(ValueError,match='frozen seed evidence'):
            driver.restored_uploads(tmp_path,sandbox)
    finally:db.close()


def test_resume_cannot_switch_execution_scenario(tmp_path):
    (tmp_path/'lifecycle-summary.json').write_text(json.dumps(dict(scenario='prescribed',status='bounded_screen_complete')))
    with pytest.raises(ValueError,match='original execution scenario'):
        driver.run(tmp_path,tmp_path/'unused',resume=True,scenario='imperfect')


def test_seed_fork_preserves_issued_bytes_and_copies_sqlite_without_mutating_source(tmp_path,monkeypatch):
    import sqlite3
    from scripts import replay_adaptive_search
    seed=tmp_path/'seed'; seed.mkdir()
    event=dict(index=0,as_of='2026-09-29T18:25:00-04:00')
    (seed/'decision-0001.json').write_text(json.dumps(event))
    (seed/'scenario.json').write_text('{"original":true}')
    (seed/'execution-sandbox.json').write_text('{"accounting":"unchanged"}')
    (seed/'snapshot-0001.json').write_text('{"immutable":true}')
    with sqlite3.connect(seed/'sandbox.sqlite') as db:
        db.execute('CREATE TABLE original(value TEXT)')
        db.execute('INSERT INTO original VALUES("issued plan")')
    checks=[]
    monkeypatch.setattr(replay_adaptive_search,'load_decision',lambda p,i:checks.append((p,i)))
    destination=tmp_path/'fork'
    assert driver.fork_seed_plan(seed,destination)==datetime.fromisoformat(event['as_of'])
    assert checks==[(seed,1)]
    for name in ('decision-0001.json','scenario.json','execution-sandbox.json','snapshot-0001.json'):
        assert (destination/name).read_bytes()==(seed/name).read_bytes()
    with sqlite3.connect(destination/'sandbox.sqlite') as db:
        assert db.execute('SELECT value FROM original').fetchone()[0]=='issued plan'
        db.execute('UPDATE original SET value="independent replay"')
    with sqlite3.connect(seed/'sandbox.sqlite') as db:
        assert db.execute('SELECT value FROM original').fetchone()[0]=='issued plan'


@pytest.mark.parametrize('options',[dict(resume=True),dict(easy_volume_phase=True)])
def test_seed_cannot_silently_rebootstrap_or_resume(tmp_path,options):
    with pytest.raises(ValueError,match='original stage'):
        driver.run(tmp_path/'fork',tmp_path/'unused',seed_plan=tmp_path/'seed',**options)


def test_stopped_recovery_is_not_permission_to_bypass_planner_failures(tmp_path):
    with pytest.raises(ValueError,match='explicit resume'):
        driver.run(tmp_path,tmp_path/'unused',recover_stopped=True)
    (tmp_path/'stopped-0001.json').write_text(json.dumps(dict(error='No eligible program')))
    with pytest.raises(ValueError,match='Only attributed donor-boundary failure'):
        driver.run(tmp_path,tmp_path/'unused',resume=True,recover_stopped=True)


def test_invalid_purpose_phase_is_rejected_before_driver_creates_outputs(tmp_path):
    with pytest.raises(ValueError,match='Named purpose'):
        driver.run(tmp_path/'unused',tmp_path/'unused-db',purpose_continuation_phase='unknown')
    assert not (tmp_path/'unused').exists()


def test_driver_passes_explicit_phase_and_rejects_a_wrong_archived_contract(tmp_path,monkeypatch):
    root=tmp_path/'run';fixture_driver(monkeypatch,root)
    original=driver.build
    def incorrect_phase(*args,**kwargs):
        assert kwargs['purpose_continuation_phase']=='known_full_phase_v1'
        h=original(*args,**kwargs)
        h.policy=SimpleNamespace(purpose_continuation=True,purpose_continuation_phase='uniform_v1')
        return h
    monkeypatch.setattr(driver,'build',incorrect_phase)
    with pytest.raises(ValueError,match='archived purpose phase'):
        driver.run(root,tmp_path/'unused',days=2,easy_volume_phase=True,
            purpose_continuation=True,purpose_continuation_phase='known_full_phase_v1')
