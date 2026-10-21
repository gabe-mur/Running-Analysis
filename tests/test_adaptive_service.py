from dataclasses import replace
from datetime import timedelta
from io import BytesIO
from types import SimpleNamespace
from zipfile import ZipFile
import pytest

from run_analysis.adaptive.new_model_daily_harness import DailyHarness
from run_analysis.adaptive.service import AdaptivePlannerService
from run_analysis.web.upload_service import UploadPayload
from test_execution_upload_sandbox import setup,AT,START,tcx


def service_fixture(setup):
    sandbox,plan,prior,config=setup
    day=START.date().isoformat()
    issue=sandbox.publish(replace(plan,session_key='slot:'+day),[prior])
    h=DailyHarness.__new__(DailyHarness)
    h.sandbox,h.config=sandbox,config
    h.active={day:dict(fingerprint=issue.fingerprint,dose='original')}
    h.events=[dict(active=dict(h.active),as_of=AT.isoformat())]
    h.executed,h.history,h.responses={},[],()
    h.evidence_as_of,h.last_as_of=AT,AT
    h.previous=((day,'original'),)
    return AdaptivePlannerService(h,{}),day


def test_service_upload_resolves_actual_role_not_assigned_quality(setup):
    service,day=service_fixture(setup)
    result=service.upload(tcx(hr=180),day=day,available_at=AT+timedelta(days=1),replan=False)
    assert result['upload']['status']=='consumed'
    assert service.harness.history[0].role=='easy'
    assert service.harness.history[0].demand[1].low>0
    assert day not in service.harness.active
    assert service.harness.responses==()
    assert service.upload(tcx(hr=180),day=day,available_at=AT+timedelta(days=2))['decision'] is None
    assert len(service.harness.history)==1


def test_fit_zip_feedback_survives_durable_completion_resync(setup,monkeypatch):
    import test_zip_feedback as fit
    monkeypatch.setattr(fit,'START',START)
    service,day=service_fixture(setup)
    buffer=BytesIO()
    with ZipFile(buffer,'w') as archive:
        archive.writestr('activity/run.fit',fit.rated_fit(45,75))
    payload=UploadPayload('activity.zip',buffer.getvalue())
    result=service.upload(payload,day=day,available_at=AT+timedelta(days=1),replan=False)
    assert result['archive']
    first=service.harness.responses[0]
    assert (first.effort,first.feeling)==(4.5,75)
    assert first.expected_at is None and first.expectation.effort is None
    # A crash before post-upload publication must not lose the report.
    service.harness.responses=()
    service.harness.sync_completions(AT+timedelta(days=2))
    assert service.harness.responses==(first,)
    assert service.upload(payload,day=day,available_at=AT+timedelta(days=2))['decision'] is None
    assert service.harness.responses==(first,)


def test_multi_activity_archive_requires_explicit_association_without_importing(setup):
    service,day=service_fixture(setup)
    buffer=BytesIO()
    with ZipFile(buffer,'w') as archive:
        archive.writestr('one.fit',b'not parsed')
        archive.writestr('two.fit',b'not parsed')
    with pytest.raises(ValueError,match='exactly one'):
        service.upload(UploadPayload('two.zip',buffer.getvalue()),day=day,available_at=AT+timedelta(days=1))
    assert not service.harness.sandbox.completed_accounted(AT+timedelta(days=1))


def test_completion_resync_preserves_source_resolved_manual_report(setup,monkeypatch):
    import test_zip_feedback as fit
    monkeypatch.setattr(fit,'START',START)
    service,day=service_fixture(setup)
    service.upload(UploadPayload('run.fit',fit.rated_fit(45,75)),
        day=day,available_at=AT+timedelta(days=1),replan=False)
    captured=replace(service.harness.responses[0],effort=8.,
        source='captured app feedback; effort=manual; feeling=fit:1')
    service.harness.responses=(captured,)
    service.harness.sync_completions(AT+timedelta(days=2))
    assert service.harness.responses==(captured,)


def test_late_upload_uses_calendar_active_at_execution_not_today(setup):
    service,day=service_fixture(setup)
    original=service.harness.active[day]['fingerprint']
    service.harness.events.append(dict(as_of=(AT+timedelta(days=1)).isoformat(),active={}))
    service.harness.active={}
    result=service.upload(tcx(),day=day,available_at=AT+timedelta(days=2),replan=False)
    assert result['upload']['comparison']['issued_fingerprint']==original
    assert len(service.harness.history)==1


def test_cancellation_before_execution_cannot_be_claimed_as_compliant(setup):
    service,day=service_fixture(setup)
    service.harness.events.append(dict(as_of=(START-timedelta(minutes=1)).isoformat(),active={}))
    service.harness.active={}
    with pytest.raises(ValueError,match='active pre-run'):
        service.upload(tcx(),day=day,available_at=AT+timedelta(days=2),replan=False)
    assert not service.harness.history


def test_explicit_unplanned_work_retains_load_without_manufacturing_compliance(setup):
    service,day=service_fixture(setup)
    before=dict(service.harness.active)
    payload=tcx()
    result=service.upload(payload,day=None,available_at=AT+timedelta(days=1),replan=False)
    comparison=result['upload']['comparison']
    assert comparison['association_kind']=='unplanned'
    assert comparison['issued_fingerprint'] is None
    assert comparison['target_load'] is None and comparison['review_signals']==['unplanned_work']
    assert not comparison['routine_variation_without_new_replan_signal']
    assert len(service.harness.history)==1 and service.harness.history[0].demand[1].low>0
    assert service.harness.active==before
    assert service.upload(payload,day=None,available_at=AT+timedelta(days=2))['upload']['status']=='duplicate'
    assert len(service.harness.history)==1
    with pytest.raises(ValueError,match='Already consumed'):
        service.upload(payload,day=day,available_at=AT+timedelta(days=2),replan=False)
