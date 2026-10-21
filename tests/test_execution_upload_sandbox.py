"""Real parser/importer round trips; declared fixture margins are not policy."""
from dataclasses import replace
from datetime import timedelta
from io import BytesIO
from zipfile import ZipFile
import json

import pytest

from fit_builder import build_run
from run_analysis.config import load_config
from run_analysis.web.upload_service import UploadPayload
from scripts.execution_forecast_contract import TargetBand
from scripts.execution_upload_sandbox import ExecutionUploadSandbox
from test_execution_forecast_contract import intent
from test_direct_quality_load import AT, history


KINDS = ('warmup', 'work', 'recovery', 'work', 'cooldown')
START = AT + timedelta(hours=16)


def tcx(*, seconds=300, miles=.4, hr=140, start=START):
    points = ''.join(f'<Trackpoint><Time>{(start+timedelta(seconds=t)).isoformat()}</Time>'
        f'<DistanceMeters>{miles*1609.344*t/seconds}</DistanceMeters>'
        + (f'<HeartRateBpm><Value>{hr}</Value></HeartRateBpm>' if hr else '')
        + '</Trackpoint>' for t in range(0, seconds+1, 10))
    return UploadPayload('test.tcx', (f'<TrainingCenterDatabase><Activities><Activity Sport="Running">'
        f'<Id>{start.isoformat()}</Id><Lap StartTime="{start.isoformat()}">'
        f'<TotalTimeSeconds>{seconds}</TotalTimeSeconds><DistanceMeters>{miles*1609.344}</DistanceMeters>'
        f'<Track>{points}</Track></Lap></Activity></Activities></TrainingCenterDatabase>').encode())


@pytest.fixture
def setup(tmp_path):
    config = load_config('config.yaml')
    config.update(resting_hr=50, max_hr=200)
    sandbox = ExecutionUploadSandbox(tmp_path/'sandbox', config, create=True)
    plan = replace(intent(), accounting_key=sandbox.reference.accounting_key,
        measurement_context=sandbox.context, distance_miles=TargetBand(.4,.3,.5,'synthetic test margin'))
    prior = replace(history(), accounting_key=sandbox.reference.accounting_key)
    yield sandbox, plan, prior, config
    sandbox.close()


def upload(sandbox, record, *, seconds=60, available_at=AT+timedelta(days=1), **kwargs):
    return sandbox.upload(tcx(seconds=5*seconds, **kwargs), fingerprint=record.fingerprint,
        available_at=available_at, observed_phases=tuple((k,seconds) for k in KINDS))


def test_copied_seed_identity_cannot_collide_with_new_database_id(setup):
    sandbox,plan,prior,_=setup
    prior=replace(prior,key='sandbox:1')
    record=sandbox.publish(plan,[prior])
    result=upload(sandbox,record)
    key=result['comparison']['actual_key']
    assert key.startswith('sandbox:recording:') and key != prior.key
    retry=upload(sandbox,record,available_at=AT+timedelta(days=2))
    assert retry['status']=='duplicate' and retry['comparison']['actual_key']==key
    assert sandbox.completed_history(AT+timedelta(days=1))[0].key==key


@pytest.mark.parametrize('seconds', [50,60,70])
@pytest.mark.parametrize('miles', [.3,.4,.5])
def test_publication_restart_upload_replaces_forecast_with_actual_once(setup, seconds, miles):
    sandbox, plan, prior, config = setup
    record = sandbox.publish(plan, [prior])
    assert sandbox.state(AT)['pending'][0]['target_forecast']['predicted_load'] == 5.
    sandbox.close()
    sandbox.db = ExecutionUploadSandbox(sandbox.root, config).db
    result = upload(sandbox, record, seconds=seconds, miles=miles)
    assert result['comparison']['routine_variation_without_new_replan_signal']
    state = sandbox.state(AT+timedelta(days=1))
    assert not state['pending'] and len(state['actual']) == 1
    assert state['observed_cardio_bounds'] == pytest.approx([5*seconds/60]*2)
    assert state['actual'][0]['distance_miles'] == pytest.approx(miles)
    duplicate = upload(sandbox, record, seconds=seconds, miles=miles, available_at=AT+timedelta(days=2))
    assert duplicate['status'] == 'duplicate'
    assert sandbox.state(AT+timedelta(days=2)) == state
    assert len(sandbox.state(AT+timedelta(hours=23))['pending']) == 1
    assert not sandbox.completed_history(AT+timedelta(hours=23))
    next_plan = replace(plan, session_key='next', issued_at=AT+timedelta(days=1),
                        suggested_start=AT+timedelta(days=2))
    future = sandbox.publish(next_plan)  # Prior evidence must survive publication/restart too.
    assert future.payload['target_forecast']['source_keys'] == ['past','sandbox:1']
    assert sandbox.record(record.fingerprint) == record  # New evidence cannot alter old expectation.


@pytest.mark.parametrize('hr,status', [(170,'higher_than_declared_allowance'),(110,'lower_than_declared_allowance')])
def test_unexpected_cost_preserved_and_signalled(setup, hr, status):
    sandbox, plan, prior, _ = setup
    r = sandbox.publish(plan,[prior])
    c = upload(sandbox,r,hr=hr)['comparison']
    assert c['distance_adherence'] == 'within'
    assert c['response_status'] == status
    assert 'load_response_outside_declared_allowance' in c['review_signals']
    assert sandbox.state(AT+timedelta(days=1))['observed_cardio_bounds'][0] == c['actual_cardio']['low']
    assert sandbox.state(AT+timedelta(days=1))['comparisons'][0]['response_status'] == status


@pytest.mark.parametrize('phases,expected', [(None,'unknown'),((('work',300.),),'unresolved_alignment')])
def test_no_prescription_to_observation_leakage(setup, phases, expected):
    s,p,h,_ = setup
    r = s.publish(p,[h])
    c = s.upload(tcx(),fingerprint=r.fingerprint,available_at=AT+timedelta(days=1),
                 observed_phases=phases)['comparison']
    assert c['structure'] == expected
    assert not c['routine_variation_without_new_replan_signal']
    assert c['actual_cardio']['estimate'] == 5.


def test_changed_intent_and_readiness_distinct_from_response(setup):
    s,p,h,_ = setup
    r = s.publish(p,[h])
    c = s.upload(tcx(),fingerprint=r.fingerprint,available_at=AT+timedelta(days=1),
        observed_phases=(('work',300.),), phase_evidence='explicit',
        readiness_changes=('reported unusual fatigue',))['comparison']
    assert c['review_signals'] == ['workout_structure_changed','readiness_changed']
    assert c['calendar_action'] == 'none'


def test_missing_hr_and_undeclared_allowance_not_zero_surprise(setup):
    s,p,h,_ = setup
    r = s.publish(replace(p,load_allowance=None),[h])
    c = upload(s,r,hr=None)['comparison']
    assert c['actual_cardio']['estimate'] is None
    assert c['actual_cardio']['high'] > c['actual_cardio']['low']
    assert c['response_status'] == 'allowance_not_calibrated_or_declared'
    assert not c['routine_variation_without_new_replan_signal']


def test_revision_requires_parent_and_upload_can_explicitly_use_original(setup):
    s,p,h,_ = setup
    first = s.publish(p,[h])
    later = replace(p,issued_at=AT+timedelta(hours=17))  # Refresh after run started, before late upload.
    with pytest.raises(ValueError,match='predecessor'):
        s.publish(later,[h])
    revised = s.publish(later,[h],parent=first.fingerprint,revision_reason='explicit test revision')
    with pytest.raises(ValueError,match='pre-run'):
        upload(s,revised)
    assert not s.state(AT+timedelta(days=1))['actual']
    upload(s,first)
    assert not s.state(AT+timedelta(days=1))['pending']
    with pytest.raises(ValueError,match='Completed session'):
        s.publish(replace(p,issued_at=AT+timedelta(days=2)),[h],parent=revised.fingerprint,revision_reason='test')


def test_failed_upload_does_not_consume_and_retry_succeeds(setup):
    s,p,h,_ = setup
    r = s.publish(p,[h])
    with pytest.raises(ValueError):
        upload(s,r,available_at=START-timedelta(seconds=1))
    assert len(s.state(AT+timedelta(days=1))['pending']) == 1
    assert upload(s,r)['status'] == 'consumed'
    with pytest.raises(ValueError,match='correction'):
        upload(s,r,hr=170)  # Same physical run, changed evidence is not a new dose.
    assert len(s.state(AT+timedelta(days=1))['actual']) == 1


def test_new_measurement_context_cannot_silently_resume(setup):
    s,_,_,config = setup
    with pytest.raises(ValueError,match='configuration changed'):
        ExecutionUploadSandbox(s.root,dict(config,max_hr=205))


def test_real_fit_inside_zip_and_raw_fit_deduplicate(setup, tmp_path):
    s,p,h,_ = setup
    r = s.publish(p,[h])
    payload = build_run(tmp_path/'run.fit',start=START)
    archive = BytesIO()
    with ZipFile(archive,'w') as z:
        z.writestr('folder/activity.fit',payload)
    first = s.upload(UploadPayload('garmin.zip',archive.getvalue()),fingerprint=r.fingerprint,
                     available_at=AT+timedelta(days=1))
    second = s.upload(UploadPayload('activity.fit',payload),fingerprint=r.fingerprint,
                      available_at=AT+timedelta(days=2))
    assert first['status']=='consumed' and second['status']=='duplicate'
    assert len(s.state(AT+timedelta(days=2))['actual']) == 1
    assert s.db.execute('SELECT COUNT(*) FROM activities').fetchone()[0] == 1


def test_multiple_fit_members_rejected_without_consumption(setup, tmp_path):
    s,p,h,_ = setup
    r = s.publish(p,[h])
    payload = build_run(tmp_path/'run.fit',start=START)
    archive = BytesIO()
    with ZipFile(archive,'w') as z:
        z.writestr('a.fit',payload)
        z.writestr('b.fit',payload)
    with pytest.raises(ValueError,match='exactly one FIT'):
        s.upload(UploadPayload('two.zip',archive.getvalue()),fingerprint=r.fingerprint,
                 available_at=AT+timedelta(days=1))
    assert not s.state(AT+timedelta(days=1))['actual']


def test_seed_history_cannot_reintroduce_consumed_run_under_an_alias(setup):
    s,p,h,_ = setup
    r = s.publish(p,[h])
    upload(s,r)
    at = AT+timedelta(days=1)
    alias = replace(s.completed_history(at)[0],key='alias')
    with pytest.raises(ValueError,match='different history identities'):
        s.publish(replace(p,session_key='next',issued_at=at),[alias])
    assert s.db.execute("SELECT COUNT(*) FROM sandbox_seed_history WHERE key='alias'").fetchone()[0] == 0


def test_duplicate_cannot_reassociate_or_change_availability(setup):
    s,p,h,_ = setup
    first = s.publish(p,[h])
    second = s.publish(replace(p,session_key='other'))
    upload(s,first)
    with pytest.raises(ValueError,match='reassociation'):
        upload(s,second)
    with pytest.raises(ValueError,match='backdate'):
        upload(s,first,available_at=AT+timedelta(hours=20))
    state = s.state(AT+timedelta(days=1))
    assert len(state['actual']) == 1 and len(state['pending']) == 1


def test_distinct_run_cannot_consume_same_session_twice(setup):
    s,p,h,_ = setup
    r = s.publish(p,[h])
    upload(s,r)
    with pytest.raises(ValueError,match='another activity'):
        upload(s,r,start=START+timedelta(hours=1))
    assert len(s.state(AT+timedelta(days=1))['actual']) == 1


def test_next_day_execution_signalled_without_clock_time_penalty(setup):
    s,p,h,_ = setup
    r = s.publish(p,[h])
    c = upload(s,r,start=START+timedelta(days=1),available_at=AT+timedelta(days=2))['comparison']
    assert c['review_signals'] == ['execution_date_changed']
    assert c['calendar_action'] == 'none'


def test_publication_retry_is_idempotent_before_and_after_completion(setup):
    s,p,h,_ = setup
    original = s.publish(p,[h])
    assert s.publish(p,[h]) == original
    upload(s,original)
    assert s.publish(p,[h]) == original
    assert len(s.state(AT+timedelta(days=1))['actual']) == 1
    assert not s.state(AT+timedelta(days=1))['pending']
    with pytest.raises(ValueError,match='retry changed'):
        s.publish(replace(p,distance_miles=TargetBand(.5,.3,.6,'changed')),[h])
    assert s.db.execute('SELECT COUNT(*) FROM sandbox_issues').fetchone()[0] == 1


@pytest.mark.parametrize('seconds,miles,hr', [(50,.3,140),(60,.4,140),(70,.5,140),(60,.4,170),(60,.4,None)])
def test_persisted_physical_and_phase_completion_agree_after_restart(setup,seconds,miles,hr):
    s,p,h,config = setup
    r = s.publish(p,[h])
    phases = tuple((k,seconds) for k in KINDS)
    first = s.upload(tcx(seconds=seconds*5,miles=miles,hr=hr),fingerprint=r.fingerprint,
        available_at=AT+timedelta(days=1),observed_phases=phases,phase_evidence='explicit',observed_role='quality')
    slot,actual = s.completed_accounted(AT+timedelta(days=1))[0]
    assert slot == p.session_key and actual.role == 'quality'
    assert actual.miles == pytest.approx(miles)
    assert actual.demand[1].low == first['comparison']['actual_cardio']['low']
    assert actual.demand[1].high == first['comparison']['actual_cardio']['high']
    assert actual.bouts[0].recovery_after_seconds == seconds
    assert actual.available_at == AT+timedelta(days=1)
    s.close()
    s.db = ExecutionUploadSandbox(s.root,config).db
    assert s.completed_accounted(AT+timedelta(days=1)) == ((slot,actual),)
    assert s.completed_history(AT+timedelta(days=1))[0].key == actual.key
    assert not s.completed_accounted(AT+timedelta(hours=23))
    second = s.upload(tcx(seconds=seconds*5,miles=miles,hr=hr),fingerprint=r.fingerprint,
        available_at=AT+timedelta(days=2),observed_phases=phases,phase_evidence='explicit',observed_role='quality')
    assert second['status'] == 'duplicate'
    assert s.completed_accounted(AT+timedelta(days=2)) == ((slot,actual),)


def test_changed_source_cannot_reinterpret_consumed_physical_record(setup):
    s,p,h,_ = setup
    r = s.publish(p,[h])
    upload(s,r)
    before = s.completed_accounted(AT+timedelta(days=1))
    with pytest.raises(ValueError,match='correction'):
        upload(s,r,hr=170)
    assert s.completed_accounted(AT+timedelta(days=2)) == before
    # Reading the durable handoff cannot depend on today's primary-source choice.
    s.db.execute('UPDATE activity_sources SET is_primary=0')
    s.db.commit()
    assert s.completed_accounted(AT+timedelta(days=2)) == before


def test_role_correction_is_not_a_duplicate_and_missing_physical_record_not_reconstructed(setup):
    s,p,h,_ = setup
    r = s.publish(p,[h])
    phases = tuple((k,60) for k in KINDS)
    s.upload(tcx(),fingerprint=r.fingerprint,available_at=AT+timedelta(days=1),
        observed_phases=phases,phase_evidence='explicit',observed_role='quality')
    with pytest.raises(ValueError,match='correction'):
        s.upload(tcx(),fingerprint=r.fingerprint,available_at=AT+timedelta(days=2),
            observed_phases=phases,phase_evidence='explicit',observed_role='easy')
    s.db.execute('UPDATE sandbox_completions SET accounted=NULL')
    s.db.commit()
    with pytest.raises(ValueError,match='migration'):
        s.completed_accounted(AT+timedelta(days=2))


def test_explicit_source_correction_preserves_earlier_version_and_counts_once(setup):
    s,p,h,_=setup
    issued=s.publish(p,[h])
    phases=tuple((k,60) for k in KINDS)
    first=s.upload(tcx(),fingerprint=issued.fingerprint,available_at=AT+timedelta(days=1),
        observed_phases=phases,phase_evidence='explicit',observed_role='quality')
    before=s.completed_accounted(AT+timedelta(days=1))
    old=s.db.execute('SELECT physical_fingerprint FROM sandbox_completions').fetchone()[0]
    correction=dict(expected_physical_fingerprint=old,reason='explicit corrected original HR source')
    revised=s.upload(tcx(hr=170),fingerprint=issued.fingerprint,available_at=AT+timedelta(days=2),
        observed_phases=phases,phase_evidence='explicit',observed_role='quality',correction=correction)
    assert revised['status']=='corrected'
    assert s.completed_accounted(AT+timedelta(days=1))==before
    later=s.completed_accounted(AT+timedelta(days=2))
    assert len(later)==1 and later[0][1].key==before[0][1].key
    assert later[0][1].demand[1].low>before[0][1].demand[1].low
    assert s.superseded(before[0][1])
    assert s.state(AT+timedelta(days=1))['comparisons']==[json.loads(json.dumps(first['comparison']))]
    retry=s.upload(tcx(hr=170),fingerprint=issued.fingerprint,available_at=AT+timedelta(days=3),
        observed_phases=phases,phase_evidence='explicit',observed_role='quality',correction=correction)
    assert retry['status']=='duplicate' and retry['comparison']==json.loads(json.dumps(revised['comparison']))
    with pytest.raises(ValueError,match='current'):
        s.upload(tcx(hr=160),fingerprint=issued.fingerprint,available_at=AT+timedelta(days=3),
            observed_phases=phases,phase_evidence='explicit',observed_role='quality',correction=correction)
    assert s.db.execute('SELECT count(*) FROM sandbox_completion_versions').fetchone()[0]==1
