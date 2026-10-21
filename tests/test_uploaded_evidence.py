from datetime import timedelta
import pytest

from run_analysis.web.upload_service import UploadPayload
from test_execution_upload_sandbox import setup,tcx,START,AT


def test_assigned_intervals_and_high_hr_do_not_label_flat_upload_as_quality(setup):
    sandbox,plan,prior,_=setup
    issued=sandbox.publish(plan,[prior])
    result=sandbox.upload(tcx(hr=180),fingerprint=issued.fingerprint,available_at=AT+timedelta(days=1),
        resolve_evidence=True,long_threshold_miles=7.)
    evidence=result['comparison']['observed_evidence']
    assert evidence['workout_type']=='easy' and not evidence['prescription_used_as_observation']
    _,physical=sandbox.completed_accounted(AT+timedelta(days=1))[0]
    assert physical.role=='easy' and physical.quality.high==0
    assert physical.demand[1].low>0


def recorded_intervals():
    laps=[]
    elapsed=distance=0.
    for seconds,speed in ((120,2),(60,4),(60,2),(60,4),(120,2)):
        start=START+timedelta(seconds=elapsed)
        track=''.join(f'<Trackpoint><Time>{(start+timedelta(seconds=t)).isoformat()}</Time>'
            f'<DistanceMeters>{distance+t*speed}</DistanceMeters><HeartRateBpm><Value>150</Value></HeartRateBpm></Trackpoint>'
            for t in range(0,seconds+1,10))
        laps.append(f'<Lap StartTime="{start.isoformat()}"><TotalTimeSeconds>{seconds}</TotalTimeSeconds>'
            f'<DistanceMeters>{seconds*speed}</DistanceMeters><TriggerMethod>Manual</TriggerMethod><Track>{track}</Track></Lap>')
        elapsed+=seconds
        distance+=seconds*speed
    return UploadPayload('recorded.tcx',(f'<TrainingCenterDatabase><Activities><Activity Sport="Running">'
        f'<Id>{START.isoformat()}</Id>'+''.join(laps)+'</Activity></Activities></TrainingCenterDatabase>').encode())


def test_recorded_work_boundaries_resolve_without_copying_prescription(setup):
    sandbox,plan,prior,_=setup
    issued=sandbox.publish(plan,[prior])
    result=sandbox.upload(recorded_intervals(),fingerprint=issued.fingerprint,available_at=AT+timedelta(days=1),
        resolve_evidence=True,long_threshold_miles=7.)
    evidence=result['comparison']['observed_evidence']
    assert evidence['workout_type']=='intervals' and evidence['aligned_work']
    assert evidence['structure_basis']=='recorded_lap_contrast'
    _,physical=sandbox.completed_accounted(AT+timedelta(days=1))[0]
    assert physical.role=='quality' and len(physical.bouts)==2
    assert physical.demand[2].low==2.
    assert [p.seconds for p in sandbox.completed_history(AT+timedelta(days=1))[0].phases]==[120,60,60,60,120]


def test_fit_feedback_reaches_evidence_without_becoming_readiness(setup,monkeypatch):
    import test_zip_feedback as fit
    monkeypatch.setattr(fit,'START',START)
    sandbox,plan,prior,_=setup
    issued=sandbox.publish(plan,[prior])
    result=sandbox.upload(UploadPayload('rated.fit',fit.rated_fit(45,75)),fingerprint=issued.fingerprint,
        available_at=AT+timedelta(days=1),resolve_evidence=True,long_threshold_miles=7.)
    evidence=result['comparison']['observed_evidence']
    assert evidence['effort']==4.5 and evidence['feeling']==75
    assert evidence['effort_source'].startswith('fit:')
    assert result['comparison']['readiness_changes']==[]
