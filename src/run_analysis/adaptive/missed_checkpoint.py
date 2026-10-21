"""Unconfirmed growth returns to accepted exposure, not compulsory catch-up.

This is a target interpretation, not a fatigue estimate or new growth credit.
The transition still needs an admitted whole programme and atomic publication.
"""
from dataclasses import replace
from datetime import datetime,time,timedelta,timezone as utc_timezone
from math import isfinite
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from run_analysis.stage_target_history import StageTarget

SOURCE='unconfirmed_checkpoint_maintenance_v1'


def propose(stage,targets,as_of,timezone,review_hour,execution_gap_miles,*,
            race_goal=None,race_event=None,race_goal_history=(),race_event_history=(),unconfirmed_opportunities=()):
    if as_of.tzinfo is None or review_hour is None:
        raise ValueError('Aware checkpoint interpretation required')
    if not isfinite(execution_gap_miles) or execution_gap_miles<0:
        raise ValueError('Nonnegative finite execution gap required')
    path=targets[-1]
    if (stage.policy.volume_path!='gradual' or not path.ramp_days
            or stage.weekly_equivalent_miles+1e-8>=path.volume_goal):
        return dict(status='no_unconfirmed_growth_checkpoint',candidate=None)
    zone=ZoneInfo(timezone)
    phase=replace(stage,issued_at=path.review_anchor or path.available_at)
    checkpoint=datetime.combine(phase.review_at.astimezone(zone).date(),time(review_hour),zone)
    if checkpoint<phase.review_at:checkpoint+=timedelta(days=1)
    opportunities=tuple(tuple(r) for r in unconfirmed_opportunities)
    if (len(set(opportunities))!=len(opportunities) or any(len(r)!=2 or not r[0]
            or not path.effective_on<=datetime.fromisoformat(r[1]).date()<as_of.astimezone(zone).date()
            for r in opportunities)):
        raise ValueError('Unique unconfirmed past-date opportunities required')
    if as_of<checkpoint and not opportunities:
        return dict(status='growth_checkpoint_not_due',candidate=None,original_checkpoint=checkpoint)
    from .race_event import no_race_recovery,resolution_for
    day=as_of.astimezone(zone).date()
    # Only a causally applied taper that actually overlaps this frozen phase
    # can explain its interruption. Clearing a goal cannot erase that evidence,
    # but cancellation before taper cannot manufacture it either.
    owner=SimpleNamespace(race_event=race_event,race_event_history=race_event_history)
    known=tuple(r for r in race_goal_history if r.applied_at<=as_of)
    interrupted=False
    for index,application in enumerate(known):
        revision=application.revision
        outcome=resolution_for(owner,revision)
        resolved=bool(revision.goal and outcome and outcome.available_at<=as_of
            and (revision.phase(day)=='complete' or no_race_recovery(revision,outcome,day)))
        if not resolved:continue
        end=min(checkpoint.astimezone(zone).date(),revision.race_date)
        if index+1<len(known):end=min(end,known[index+1].effective_on-timedelta(days=1))
        interrupted|=max(application.effective_on,revision.taper_start,
            (path.review_anchor or path.available_at).astimezone(zone).date())<=end
    if execution_gap_miles<=0 and not interrupted and not opportunities:
        return dict(status='no_uncompleted_training_evidence',candidate=None,original_checkpoint=checkpoint)
    # Accepted exposure is not re-estimated from the unsuccessful target or a
    # single missed run. Keep long/ordinary doses and all normalization intact.
    # Snapshots normalize timestamps to UTC. Identity must survive that exact
    # instant's roundtrip, even when the driver passes an athlete-local clock.
    candidate=replace(stage,key=stage.key+':maintenance:'+as_of.astimezone(utc_timezone.utc).isoformat(),issued_at=as_of,
        policy=replace(stage.policy,volume_goal_miles=stage.weekly_equivalent_miles),source=SOURCE)
    # Support at the accepted baseline, not the unconfirmed ramp endpoint.
    target=StageTarget(candidate.key+':maintenance-path',as_of,as_of.astimezone(zone).date(),
        stage.weekly_equivalent_miles,path.supporting_easy_per_7+stage.weekly_equivalent_miles-path.volume_per_7,SOURCE)
    return dict(status='conditional_maintenance_candidate',stage_key=stage.key,candidate=candidate,
        amended_target=target,original_targets=tuple(targets),original_checkpoint=checkpoint,
        confirmed_delivery=False,review_hour=review_hour,execution_gap_miles=execution_gap_miles,
        race_interrupted=interrupted,unconfirmed_opportunities=opportunities,source=SOURCE)
