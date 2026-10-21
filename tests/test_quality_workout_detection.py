"""Observed quality families; prescriptions/HR drift alone cannot create work."""
from dataclasses import replace
from datetime import datetime,timedelta,timezone
import sqlite3

import pytest

from run_analysis.models import Activity,Lap
from run_analysis.quality_phases import detect_continuous_quality_phase
from run_analysis.delivered_training import continuous_phase_from_activity
from run_analysis.recommendation import _quality_prescription
from run_analysis.web.schemas import QualitySessionType,WorkoutType
from run_analysis.web.schemas import RecommendationResponse,ConfidenceLevel,ReadinessFlag,SessionDifficulty,ZoneBreakdown
from run_analysis.workout_scoring import _prescription_analysis,analyze_intervals
from run_analysis.workout_detection import detect_structured_workout,recorded_interval_work_positions
from test_workout_scoring import _interval

START=datetime(2026,10,2,tzinfo=timezone.utc)
CONFIG={'zones':{'z3':[154,166]}}


def database(rows):
    db=sqlite3.connect(':memory:');db.row_factory=sqlite3.Row
    db.execute('CREATE TABLE laps (activity_id,lap_index,total_time_s,distance_m,average_hr_bpm,maximum_hr_bpm,intensity,trigger_method,start_time_utc)')
    db.execute('CREATE TABLE segments (activity_id,moving_time_s,average_hr_bpm,average_grade_percent,metrics_json,is_pathological)')
    offset=0
    for n,(seconds,speed,hr,trigger,intensity) in enumerate(rows):
        db.execute('INSERT INTO laps VALUES (1,?,?,?,?,?,?,?,?)',
            (n,seconds,seconds*speed,hr,hr+5 if hr else None,intensity,trigger,(START+timedelta(seconds=offset)).isoformat()))
        offset+=seconds
    return db


def quality_rows(kind):
    easy=(600,2.4,140,'manual',None)
    end=(300,2.4,143,'session_end',None)
    if kind==QualitySessionType.PROGRESSION:
        rows=[easy,(1200,2.9,161,'session_end',None)]
    elif kind==QualitySessionType.THRESHOLD:
        rows=[easy,(1080,3.1,168,'manual',None),end]
    else:
        work=300 if kind==QualitySessionType.LONG_INTERVALS else 45 if kind==QualitySessionType.HILL_REPEATS else 60
        # Hills are slower up than down; recorded Active/Rest distinguishes
        # these without requiring pace contrast or instant HR to reach Z4.
        rows=[easy]
        for n in range(3):
            rows.extend([(work,1.8 if kind==QualitySessionType.HILL_REPEATS else 3.5,148,'manual','Active'),
                         (90,2.4 if kind==QualitySessionType.HILL_REPEATS else 2.,150,'manual','Resting')])
        rows.append(end)
    return rows


@pytest.mark.parametrize('kind',list(QualitySessionType))
def test_every_quality_template_has_observed_detection_and_analysis(kind):
    rows=quality_rows(kind)
    with database(rows) as db:
        result=detect_structured_workout(db,1,[],z3_floor=154)
        assert result is not None
        assert result.workout_type==_quality_prescription(kind)['workout_type']
        if result.workout_type==WorkoutType.TEMPO_THRESHOLD:
            phase=detect_continuous_quality_phase(db,CONFIG,1)
            assert phase is not None and phase.duration_seconds==rows[1][0]
        else:
            assert len(recorded_interval_work_positions(db.execute('SELECT * FROM laps').fetchall()))==3
            intervals=[_interval(n,n,speed,seconds,hr) for n,(seconds,speed,hr,_,_) in enumerate(rows)]
            analysis=analyze_intervals(db,1,intervals)
            assert analysis.available and analysis.source=='recorded_laps'
            assert analysis.work_repetition_count==3


@pytest.mark.parametrize('rows',[
    [(600,2.4,140,'distance',None),(1200,2.9,161,'session_end',None)],
    [(600,2.4,140,'manual',None),(1200,2.4,161,'session_end',None)], # HR drift, no pace change
    [(600,2.4,140,'manual',None),(1200,2.9,145,'session_end',None)], # faster but aerobic
    [(600,2.4,140,None,None),(1200,2.9,161,None,None)], # no terminal work boundary
    [(600,2.4,140,'manual',None),(120,2.9,161,'session_end',None)], # tiny finish, not sustained
])
def test_easy_negative_controls_do_not_become_quality(rows):
    with database(rows) as db:
        assert detect_structured_workout(db,1,[],z3_floor=154) is None
        assert detect_continuous_quality_phase(db,CONFIG,1) is None


def test_progression_uses_actual_boundary_not_the_prescribed_halfway_point():
    rows=[(591,1613.32/591,136,'manual',None),(1594,4748.89/1594,160,'session_end',None)]
    with database(rows) as db:
        detection=detect_structured_workout(db,1,[],z3_floor=154)
        phase=detect_continuous_quality_phase(db,CONFIG,1)
        assert detection.source=='recorded_progression_laps'
        assert phase.structure=='progression_finish'
        assert phase.start_offset_seconds==591
        assert phase.distance_miles==pytest.approx(4748.89/1609.344)


def test_original_activity_retains_manual_boundary_metadata():
    activity=Activity('test','Running',None,None,[],[
        Lap(0,START,591,1613.32,None,136,152,None,None,'manual'),
        Lap(1,START+timedelta(seconds=643),1594,4748.89,None,160,171,None,None,'session_end')],[])
    phase=continuous_phase_from_activity(activity,CONFIG)
    assert phase is not None and phase.start_offset_seconds==643 # recorded pause, not timer sum
    activity.laps[0].trigger_method='distance'
    assert continuous_phase_from_activity(activity,CONFIG) is None


def test_manual_short_hills_use_elevation_even_when_uphill_is_slower():
    rows=[(600,2.4,140,'manual',None),
        (45,1.8,148,'manual',None),(90,2.4,150,'manual',None),
        (45,1.8,148,'manual',None),(90,2.4,150,'manual',None),
        (300,2.4,143,'session_end',None)]
    intervals=[]
    for n,(seconds,speed,hr,_,_) in enumerate(rows):
        raw=_interval(n,n,speed,seconds,hr)
        rise=5 if n in {1,3} else -5 if n in {2,4} else 0
        intervals.append(replace(raw,start=replace(raw.start,altitude_m=100),end=replace(raw.end,altitude_m=100+rise)))
    with database(rows) as db:
        detection=detect_structured_workout(db,1,intervals,z3_floor=154)
        assert detection.workout_type==WorkoutType.INTERVALS
        assert recorded_interval_work_positions(db.execute('SELECT * FROM laps').fetchall(),intervals)=={1,3}


def test_all_active_auto_laps_are_not_recorded_work_recovery():
    rows=[(600,speed,160,'distance','Active') for speed in (2.3,3.5,2.3,3.5,2.3)]
    with database(rows) as db:
        assert detect_structured_workout(db,1,[],z3_floor=154) is None


@pytest.mark.parametrize('early,expected',[(True,False),(False,True)])
def test_detected_progression_is_not_automatically_compliant(early,expected):
    first=1613.32 if early else 2.25*1609.344
    second=4748.89 if early else 2.25*1609.344
    rows=[(first/2.7,2.7,136,'manual',None),(second/3.,3.,160,'session_end',None)]
    template=_quality_prescription(QualitySessionType.PROGRESSION)
    prescription=RecommendationResponse(generated_at=START-timedelta(hours=1),
        fitness_state_as_of=START-timedelta(hours=1),planned_for=START,
        confidence=ConfidenceLevel.MODERATE,readiness=ReadinessFlag.READY,
        quality_session_type=QualitySessionType.PROGRESSION,
        distance_range_miles=(4.2,4.8),target_zones=template['zones'],
        **{k:v for k,v in template.items() if k not in {'zones','distance'}})
    difficulty=SessionDifficulty(distance_miles=(first+second)/1609.344,
        moving_minutes=40,elapsed_minutes=40,stopped_minutes=0,
        zone_breakdown=ZoneBreakdown(easy_minutes=10,moderate_minutes=25,hard_minutes=0),
        is_quality_session=True)
    with database(rows) as db:
        analysis=_prescription_analysis(db,CONFIG,1,difficulty,prescription,
            timing_delta_hours=0,distance_delta_miles=.2467 if early else 0,match_confidence='high')
        assert analysis.detected_work_minutes==pytest.approx(rows[1][0]/60)
        assert analysis.detection_source=='recorded_progression_laps'
        assert analysis.prescribed_quality_completed is expected
        if early:
            assert analysis.execution_status=='Progression detected; execution differed'
            assert 'after 1.0 miles' in analysis.summary


@pytest.mark.parametrize('kind',list(QualitySessionType))
def test_every_quality_template_round_trips_original_tcx_without_plan_labels(tmp_path,kind):
    from run_analysis.adaptive.execution_upload_sandbox import ExecutionUploadSandbox
    from run_analysis.config import load_config
    from run_analysis.web.upload_service import UploadPayload
    config=load_config()
    config['zones']['z3'][0]=154
    elapsed=distance=0.
    laps=[]
    for seconds,speed,hr,trigger,intensity in quality_rows(kind):
        start=START+timedelta(seconds=elapsed)
        points=''.join(f'<Trackpoint><Time>{(start+timedelta(seconds=t)).isoformat()}</Time>'
            f'<DistanceMeters>{distance+t*speed}</DistanceMeters><HeartRateBpm><Value>{hr}</Value></HeartRateBpm>'
            '</Trackpoint>' for t in [*range(0,int(seconds),5),seconds])
        laps.append(f'<Lap StartTime="{start.isoformat()}"><TotalTimeSeconds>{seconds}</TotalTimeSeconds>'
            f'<DistanceMeters>{speed*seconds}</DistanceMeters><AverageHeartRateBpm><Value>{hr}</Value></AverageHeartRateBpm>'
            f'<TriggerMethod>{"Manual" if trigger=="manual" else "Time"}</TriggerMethod>'
            +(f'<Intensity>{intensity}</Intensity>' if intensity else '')+f'<Track>{points}</Track></Lap>')
        elapsed+=seconds;distance+=speed*seconds
    payload=UploadPayload('observed.tcx',('<TrainingCenterDatabase><Activities><Activity Sport="Running">'
        f'<Id>{START.isoformat()}</Id>'+''.join(laps)+'</Activity></Activities></TrainingCenterDatabase>').encode())
    sandbox=ExecutionUploadSandbox(tmp_path/'sandbox',config,create=True)
    try:
        result=sandbox.upload(payload,fingerprint=None,available_at=START+timedelta(days=1),
            resolve_evidence=True,long_threshold_miles=7.)
        evidence=result['comparison']['observed_evidence']
        assert evidence['workout_type']==_quality_prescription(kind)['workout_type'].value
        assert evidence['aligned_work'] and not evidence['prescription_used_as_observation']
        _,physical=sandbox.completed_accounted(START+timedelta(days=1))[0]
        assert physical.role=='quality'
        assert len(physical.bouts)==(1 if kind in {QualitySessionType.PROGRESSION,QualitySessionType.THRESHOLD} else 3)
    finally:
        sandbox.close()
