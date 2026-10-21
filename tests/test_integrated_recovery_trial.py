from dataclasses import replace
from datetime import timedelta
import pytest

from run_analysis.recovery_assessment import AssessmentPolicy,assess_next_session
from scripts.integrated_recovery_trial import (Run,recovery_path,volume_and_cadence,neighbors,optimize,
    signature,churn,backward_compare)
from scripts.hr_backed_sandbox import transformed_entry
from test_hr_backed_sandbox import donor
from test_expected_completion import setup


def sample():
    _,_,rec=setup('easy',3.8)
    rec=rec.model_copy(update={'planned_for':rec.planned_for.replace(hour=12)})
    run=Run('easy',transformed_entry(donor(),rec,source='projected'))
    origin=rec.planned_for.replace(hour=0)
    return origin,run,AssessmentPolicy('test',160,320,36,36,24,36)


@pytest.mark.parametrize('phase',[0,-1,1])
def test_fast_recovery_matches_reference_with_end_time_decay_and_fixed_timing_phase(phase):
    origin,run,policy=sample()
    history=[run.on(run.day-timedelta(days=2),key='past')]
    program=[run,run.on(run.day+timedelta(days=2),key='next')]
    rows=recovery_path(history,program,policy,phase)['sessions']
    ledger=[r.entry for r in history]
    for r,row in zip(program,rows):
        offset=timedelta(hours=phase*(6 if r.day.toordinal()%2 else -6))
        at=r.entry.start+offset
        direct=assess_next_session(ledger,r.entry.workload,at,policy,proposed_work_minutes=r.entry.work_minutes)
        assert row['demands']==pytest.approx({k:v['total_high'] for k,v in direct['channels'].items()})
        ledger.append(replace(r.entry,start=at,end=r.entry.end+offset))


def test_completed_run_is_counted_once_in_full_volume_path():
    origin,run,policy=sample()
    next_run=run.on(run.day+timedelta(days=2),key='next')
    end=origin+timedelta(days=28); targets=[(16,18)]*28
    expected=volume_and_cadence([], [run,next_run],origin,end,targets,3.75)
    after=volume_and_cadence([run],[next_run],origin,end,targets,3.75)
    assert expected==after
    with pytest.raises(ValueError,match='Duplicate'):
        volume_and_cadence([run],[run,next_run],origin,end,targets,3.75)


def test_search_can_change_frequency_and_key_dates_but_not_immutable_tail():
    origin,run,_=sample()
    long=replace(run.on(run.day+timedelta(days=3),key='long'),role='long')
    tail=run.on(run.day+timedelta(days=21),key='tail')
    program=(run,long,tail)
    options=list(neighbors(program,origin.date(),(run,)))
    assert any(len(p)>len(program) for p in options)
    assert any(len(p)<len(program) for p in options)
    assert any(next(r for r in p if r.role=='long').day!=long.day for p in options)
    assert all(tail in p for p in options)
    assert all(len({r.day for r in p})==len(p) for p in options)


def test_reserved_date_never_receives_a_new_run():
    origin,run,_=sample()
    blocked=frozenset({run.day+timedelta(days=1)})
    assert all(not any(r.day in blocked for r in p) for p in neighbors((run,),run.day,(run,),blocked=blocked))


def test_exhausted_search_is_explicit_not_certified_converged():
    origin,run,policy=sample()
    _,info=optimize([], (run,),origin.date(),(run,),origin,origin+timedelta(days=28),
                    [(16,18)]*28,3.75,policy,budget=1)
    assert not info['neighborhood_complete']


def test_zero_surprise_input_source_does_not_change_new_scorer():
    origin,run,policy=sample()
    later=run.on(run.day+timedelta(days=2),key='later')
    actual=replace(run,entry=replace(run.entry,workload=replace(run.entry.workload,source='observed')))
    assert recovery_path([run],[later],policy)==recovery_path([actual],[later],policy)
    assert churn((later,),(later,),origin.date())==[]


def test_recovery_feasibility_actually_influences_calendar_search():
    origin,run,_=sample()
    restrictive=AssessmentPolicy('restrictive test only',55,110,36,36)
    plan=tuple(run.on(run.day+timedelta(days=i),key=f'run{i}') for i in range(4))
    common=([],plan,origin.date(),(run,),origin,origin+timedelta(days=28),[(8,10)]*28,3.75,restrictive)
    chosen,info=optimize(*common,budget=1000)
    unguarded,other=optimize(*common,budget=1000,recovery_enabled=False)
    assert info['neighborhood_complete'] and info['compatible']
    assert signature(chosen)!=signature(unguarded)
    assert max(recovery_path([],unguarded,restrictive,p)['total_excess'] for p in (-1,0,1))>0


def test_moved_added_session_cannot_collide_with_another_added_identity():
    origin,run,_=sample()
    key=f'added-easy/{run.day}/0'
    existing=run.on(run.day+timedelta(days=3),key=key)
    for candidate in neighbors((existing,),run.day,(run,),reserved_keys={f'added-easy/{run.day}/1'}):
        keys=[r.entry.key for r in candidate]
        assert len(set(keys))==len(keys)
        assert f'added-easy/{run.day}/1' not in keys


def test_backward_audit_does_not_inject_later_overexecution_into_prior_inputs():
    from scripts.audit_hr_recovery_programs import scaled_entry
    origin,run,policy=sample()
    later=run.on(run.day+timedelta(days=2),key='next')
    actual=replace(run,entry=scaled_entry(run.entry,1.25))
    result=backward_compare(((),(run,later),run.day),[actual],(later,),origin,
        origin+timedelta(days=28),[(16,18)]*28,3.75,policy)
    assert result['prior_incumbent']==result['later_program_under_prior_inputs']
    assert result['prior_incumbent']!=result['later_under_current_inputs']


def test_volume_integral_matches_existing_full_path_implementation():
    from math import log
    from scripts.volume_path_prototype import volume_path
    from run_analysis.web.schemas import WeeklyScheduleDay
    from run_analysis import weekly_schedule as planner
    origin,run,_=sample()
    base,_,rec=setup('easy',3.8)
    history=[run.on(run.day-timedelta(days=2),key='past')]
    program=[run]
    opening=history[0].miles*log(2)*2**(-((origin-history[0].entry.start).total_seconds()/86400)/7)
    base=base.model_copy(update={'as_of':origin,'recent_load':base.recent_load.model_copy(
        update={'continuous_distance_miles':opening})})
    ordinary=sum(planner.typical_easy_distance(base))/2
    days=[]; states=[]
    for i in range(7):
        state=base.model_copy(update={'as_of':origin+timedelta(days=i)}); states.append(state)
        prescription=rec.model_copy(update={'planned_for':run.entry.start}) if i==0 else None
        days.append(WeeklyScheduleDay(date=state.as_of.date(),recommendation=prescription,
            day_role='easy_run' if i==0 else 'rest_day',rationale='test'))
    reference=volume_path(days,states,[(16,18)]*7,{})
    result=volume_and_cadence(history,program,origin,origin+timedelta(days=7),[(16,18)]*7,ordinary)
    assert result['volume']==pytest.approx(reference['cost'])
