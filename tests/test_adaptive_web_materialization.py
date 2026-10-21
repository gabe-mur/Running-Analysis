"""API fidelity controls, not new search or physiological acceptance."""
from datetime import date, datetime, timedelta
import pytest

from test_adaptive_stage_lifecycle import completed_stage
from run_analysis.adaptive.web_materialization import render
from run_analysis.web.schemas import WeeklyScheduleResponse,WorkoutType
from run_analysis.web.schemas import ReadinessFlag
from run_analysis.session_decision import CurrentContext
from zoneinfo import ZoneInfo


@pytest.mark.parametrize('kind,basis',[
    ('short_intervals','distance'),('long_intervals','time'),('threshold','time'),
    ('fartlek','time'),('hill_repeats','time'),('progression','portion')])
def test_workout_definitions_have_distinct_presentation_structures(kind,basis):
    from run_analysis.adaptive.web_materialization import prescription_basis
    from run_analysis.web.schemas import QualitySessionType
    assert prescription_basis('quality',QualitySessionType(kind))==basis
    assert prescription_basis('easy',None)=='distance'
    assert prescription_basis('long',None)=='distance'


def test_distance_display_uses_frozen_original_effort_pace_not_a_later_upload():
    from types import SimpleNamespace
    from run_analysis.adaptive.web_materialization import phase_distances
    from run_analysis.workload_profile import WorkBout
    from run_analysis.staged_progression import MILE
    at=datetime(2026,10,3,tzinfo=ZoneInfo('UTC'))
    source=SimpleNamespace(key='source', end=at-timedelta(days=1), available_at=at-timedelta(days=1),
        bouts=(WorkBout(100,90,400),WorkBout(100,None,400)))
    newer=SimpleNamespace(key='newer', end=at+timedelta(days=1), available_at=at+timedelta(days=1),
        bouts=(WorkBout(100,90,800),WorkBout(100,None,800)))
    h=SimpleNamespace(history=[source],coaching_contracts={'q':SimpleNamespace(
        source='frozen;quality_structure:source',issued_at=at)})
    dose=SimpleNamespace(key='q')
    intent={'distance_miles':{'target':3.8}, 'phases':[
        {'kind':k,'seconds':{'target':s}} for k,s in (
            ('warmup',600),('work',100),('recovery',90),('work',100),('cooldown',600))]}
    import copy
    before=copy.deepcopy(intent)
    distances=phase_distances(h,dose,intent)
    assert sum(distances)==pytest.approx(3.8)
    assert distances[1]*MILE==pytest.approx(400)
    assert distances[3]*MILE==pytest.approx(400)
    h.history.append(newer)
    assert phase_distances(h,dose,intent)==distances
    assert intent==before
    source.available_at=at+timedelta(seconds=1)
    fallback=phase_distances(h,dose,intent)
    assert fallback!=distances and sum(fallback)==pytest.approx(3.8)


def test_strength_rotation_is_render_only_and_uploaded_run_day_is_not_empty(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from run_analysis.web.schemas import StrengthSessionType
    h,_,config,_=completed_stage(tmp_path,monkeypatch)
    try:
        config={**config,'coaching':{**config.get('coaching',{}),
            'strength_training':{'enabled':True,'leg_frequency_days':8,'push_pull_frequency_days':8}}}
        event=h.events[0]
        now=datetime.fromisoformat(event['as_of'])
        today=now.astimezone(ZoneInfo(config['timezone_default'])).date()
        snapshot=(h.sandbox.root/event['snapshot_file']).read_bytes()
        records={d:h.sandbox.record(row['fingerprint']).serialized for d,row in event['active'].items()}
        before=render(h,event,config)
        prior={d.date:d.strength_suggestion.session_type for d in before.planning_days if d.strength_suggestion}
        after=render(h,event,config,prior_strength_plan=prior)
        assert [d.strength_suggestion for d in after.planning_days]==[d.strength_suggestion for d in before.planning_days]
        assert [d.recommendation for d in after.planning_days]==[d.recommendation for d in before.planning_days]
        # A real upload removes today's issued run. Test that its physical role
        # still blocks treating that newly empty card as a fresh lifting slot.
        h.history=(*h.history,SimpleNamespace(start=now,available_at=now,role='quality',miles=0))
        uploaded={**event,'active':{d:row for d,row in event['active'].items() if d!=today.isoformat()}}
        result=render(h,uploaded,config,prior_strength_plan={today:StrengthSessionType.LEGS})
        current=next(d for d in result.planning_days if d.date==today)
        assert current.recommendation is None and current.strength_suggestion is None
        assert (h.sandbox.root/event['snapshot_file']).read_bytes()==snapshot
        assert records=={d:h.sandbox.record(row['fingerprint']).serialized for d,row in event['active'].items()}
        assert h.events[0]==event and len(h.events)==1
    finally:
        h.sandbox.close()


@pytest.mark.parametrize('phase,wording',[
    ('did_not_start','Race not started'),('complete','Race preparation complete')])
def test_finished_race_summary_does_not_claim_active_preparation_gap(tmp_path,monkeypatch,phase,wording):
    h,_,config,_=completed_stage(tmp_path,monkeypatch)
    try:
        event=dict(h.events[0],race_preparation=dict(distance='10k',race_date='2026-09-01',
            phase=phase,preparation_gap=True))
        rendered=render(h,event,config)
        assert rendered.summary.startswith(wording)
        assert 'below the selected' not in rendered.summary
        assert 'general-fitness' in rendered.summary.lower()
        assert 'cleared' not in rendered.summary
    finally:h.sandbox.close()


def test_render_uses_exact_durable_prescription_and_hides_soft_tail(tmp_path,monkeypatch):
    h,_,config,_=completed_stage(tmp_path,monkeypatch,growth_focus='easy_volume',
        gradual=True,continuation_boundary=True,single_volume=True,purpose_continuation=True)
    try:
        event=h.events[0]
        result=render(h,event,config)
        assert result.planner_version==201
        assert len(result.days)==7 and len(result.planning_days)==21
        for day in result.planning_days:
            issued=event['active'].get(day.date.isoformat())
            if not issued:
                assert day.recommendation is None
                continue
            intent=h.sandbox.record(issued['fingerprint']).payload['intent']
            recommendation=day.recommendation
            assert recommendation.distance_target_miles==intent['distance_miles']['target']
            assert sum(s.distance_miles for s in recommendation.structure)==pytest.approx(intent['distance_miles']['target'])
            assert recommendation.planned_distance_miles==h.catalog[issued['dose']].miles
            assert sum(s.duration_minutes for s in recommendation.structure)==pytest.approx(
                sum(p['seconds']['target'] for p in intent['phases'])/60)
            if recommendation.workout_type==WorkoutType.INTERVALS:
                assert [s.duration_minutes for s in recommendation.structure if s.phase=='work']==[
                    p['seconds']['target']/60 for p in intent['phases'] if p['kind']=='work']
                assert recommendation.target_zones==[]  # No invented instantaneous HR prescription.
                # The frozen workout definition, not the availability of a
                # distance estimate, determines the unit of its timed efforts.
                from run_analysis.web.schemas import QualitySessionType
                expected='time' if recommendation.quality_session_type==QualitySessionType.LONG_INTERVALS else 'distance'
                assert all(s.prescription_basis==expected for s in recommendation.structure)
        encoded=result.model_dump_json()
        restored=WeeklyScheduleResponse.model_validate_json(encoded)
        assert len(restored.days)==7 and restored.planning_days==[]
        assert restored.visible_7d_scheduled_miles==result.visible_7d_scheduled_miles
        assert h.events[0]==event
    finally:
        h.sandbox.close()


def test_rest_label_does_not_claim_exact_recovery_hours_or_reallocate(tmp_path,monkeypatch):
    h,_,config,_=completed_stage(tmp_path,monkeypatch)
    try:
        result=render(h,h.events[0],config)
        for day in result.days:
            if day.recommendation is None:
                assert 'full projected' in day.rationale
                assert 'hours' not in day.rationale
        assert result.target_evidence.peak_7d_miles>=result.target_evidence.recent_7d_miles
        assert len(result.trailing_days)==7
        assert result.trailing_days[-1].date==result.start_date-timedelta(days=1)
    finally:
        h.sandbox.close()


def test_review_display_advances_date_without_issuing_fake_rest_program(tmp_path,monkeypatch):
    h,_,config,_=completed_stage(tmp_path,monkeypatch)
    try:
        event=h.events[0]
        now=datetime.fromisoformat(event['as_of'])+timedelta(days=3)
        rendered=render(h,event,config,review_reason='Explicit evidence review',review_as_of=now)
        assert rendered.start_date==h.decision(now)
        assert rendered.generated_at==now
        assert rendered.planner_status=='review_required'
        assert all(d.recommendation is None and d.strength_suggestion is None for d in rendered.planning_days)
        assert rendered.run_count==0
        assert len(h.events)==1
        assert event['decision']!=rendered.start_date.isoformat()
    finally:
        h.sandbox.close()


def test_missing_readiness_is_explicit_not_a_workout_role_or_plan_change(tmp_path,monkeypatch):
    h,_,config,_=completed_stage(tmp_path,monkeypatch)
    try:
        event=h.events[0]
        as_of=datetime.fromisoformat(event['as_of'])
        today=as_of.astimezone(ZoneInfo(config['timezone_default'])).date()
        h.current_context=None
        unknown=render(h,event,config)
        h.current_context=CurrentContext(today,as_of,'usual','test check-in')
        ready=render(h,event,config)
        for a,b in zip(unknown.planning_days,ready.planning_days):
            assert a.date==b.date
            if a.recommendation is None:
                assert b.recommendation is None
                continue
            assert a.recommendation.readiness==ReadinessFlag.UNKNOWN
            if a.date==today:
                assert 'No check-in recorded for today' in a.recommendation.readiness_reason
                assert b.recommendation.readiness==ReadinessFlag.READY
            else:
                assert a.recommendation.readiness_label=='Check on the day'
                assert b.recommendation.readiness==ReadinessFlag.UNKNOWN
                assert b.recommendation.readiness_label=='Check on the day'
            assert a.recommendation.workout_type==b.recommendation.workout_type
            assert a.recommendation.planned_distance_miles==b.recommendation.planned_distance_miles
        assert len(h.events)==1
    finally:
        h.sandbox.close()


@pytest.mark.parametrize('state',['usual','little_tired'])
def test_advisory_weather_times_and_dated_checkin_do_not_rewrite_contracts(tmp_path,monkeypatch,state):
    h,_,config,_=completed_stage(tmp_path,monkeypatch)
    try:
        event=h.events[0]
        now=datetime.fromisoformat(event['as_of'])
        zone=ZoneInfo(config['timezone_default'])
        today=now.astimezone(zone).date()
        h.current_context=CurrentContext(today,now,state,'test check-in')
        baseline=render(h,event,config)
        times={day.date.isoformat():datetime.combine(day.date,datetime.min.time(),zone)+timedelta(hours=7)
            for day in baseline.planning_days if day.recommendation}
        records={d: h.sandbox.record(row['fingerprint']).serialized for d,row in event['active'].items()}
        suggested=render(h,event,config,suggested_times=times)
        for before,after in zip(baseline.planning_days,suggested.planning_days):
            assert before.date==after.date
            if not after.recommendation:continue
            assert after.planned_at==times[after.date.isoformat()]
            assert after.recommendation.time_recommendation_basis=='weather'
            assert before.recommendation.distance_target_miles==after.recommendation.distance_target_miles
            assert before.recommendation.structure==after.recommendation.structure
            if after.date>today:
                assert after.recommendation.readiness==ReadinessFlag.UNKNOWN
                assert after.recommendation.readiness_label=='Check on the day'
        assert records=={d: h.sandbox.record(row['fingerprint']).serialized for d,row in event['active'].items()}
        tomorrow=render(h,event,config,display_as_of=now+timedelta(days=1))
        assert all(d.recommendation.readiness==ReadinessFlag.UNKNOWN
            for d in tomorrow.planning_days if d.recommendation)
        first=next(d for d in baseline.planning_days if d.recommendation)
        wrong=datetime.combine(first.date+timedelta(days=1),datetime.min.time(),zone)
        with pytest.raises(ValueError,match='cannot change'):
            render(h,event,config,suggested_times={first.date.isoformat():wrong})
        assert len(h.events)==1
    finally:
        h.sandbox.close()
