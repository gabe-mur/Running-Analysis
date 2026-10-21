"""Shared isolated recommendation/upload lifecycle; no live planner activation.

The behavioral driver and service tests call this same coordinator. Evidence
fixtures may inject observed phases through the harness, but this public upload
path always resolves original evidence independently of the assigned session.
"""
from datetime import timedelta
from threading import RLock

from run_analysis.web.upload_service import UploadPayload,validate_upload,_persist_upload
from run_analysis.web.upload_archive import fit_members
from .adaptive_stage_lifecycle import advance_stage


class AdaptivePlannerService:
    def __init__(self,harness,initial_inputs):
        self.harness=harness
        self.initial_inputs=initial_inputs
        self._lock=RLock()

    @property
    def latest(self):
        return self.harness.events[-1] if self.harness.events else None

    def amend_volume_rate(self,as_of,*,preview_check=None):
        """Explicit policy change; never pretend the current phase completed."""
        from .volume_rate_amendment import apply
        with self._lock:
            updated,review,event=apply(self.harness,self.initial_inputs,as_of,preview_check=preview_check)
            self.harness=updated
            return dict(event=event,stage_review=review)

    def refresh(self,as_of,*,reason='explicit_refresh'):
        with self._lock:
            from copy import copy
            from .race_goal import requested_revision
            working = self.harness
            from .live_bridge import enabled
            if enabled(working.config):
                from .quality_choices import install
                working=copy(working)
                install(working)
            revision = requested_revision(working, as_of)
            if revision != getattr(working, 'race_goal', None):
                working = copy(working)
                working.race_goal = revision
            from .race_event import known_resolution, resolution_history
            resolution = known_resolution(working, revision, as_of)
            if revision and revision.goal and working.decision(as_of) > revision.race_date and resolution is None:
                raise ValueError('Confirm the race outcome: select the uploaded race activity or report did not start before updating the plan')
            if resolution != getattr(working, 'race_event', None):
                working = copy(working)
                working.race_event = resolution
            outcomes=resolution_history(working,as_of)
            if outcomes != getattr(working,'race_event_history',()):
                working=copy(working)
                working.race_event_history=outcomes
            if working.stage is not None:
                working,review,event=advance_stage(working,self.initial_inputs,as_of)
            else:
                # Explicit engineering fixtures predate individualized stages.
                review,event=dict(status='no_individualized_stage_fixture'),None
            if event is None:
                event=working.replan(as_of,reason=reason)
            self.harness = working
            return dict(event=event,stage_review=review)

    def upload(self,payload,*,day,available_at,correction=None,replan=True):
        with self._lock:
            validate_upload(payload)
            archive=None
            if payload.filename.lower().endswith('.zip'):
                members=fit_members(payload.content)
                if len(members)!=1:
                    raise ValueError('An issued-slot upload requires exactly one FIT activity; split a multi-activity archive explicitly')
                archive=_persist_upload(self.harness.sandbox.root,payload)
                payload=UploadPayload(*members[0])
                validate_upload(payload)
            result=self.harness.upload(payload,day=day,available_at=available_at,correction=correction)
            decision=None
            if replan and result['status']!='duplicate':
                decision=self.refresh(max(available_at,self.harness.last_as_of)+timedelta(microseconds=1),
                    reason='post_upload')
            return dict(upload=result,decision=decision,archive=str(archive) if archive else None)
