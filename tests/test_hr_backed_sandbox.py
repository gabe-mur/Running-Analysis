from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
import pytest

from run_analysis.recovery_assessment import AssessmentPolicy
from run_analysis.workload_accounting import Quantity, compare_execution
from scripts.hr_backed_sandbox import HRDonor, transformed_entry, issued_envelope, choose_donor
from scripts.audit_hr_backed_recovery import replay
from test_projected_duration import prescription
from test_expected_completion import setup
from test_typed_continuation import fixture


def donor(role='easy', key='recorded-1', rate=2):
    q=lambda x:Quantity.point(x,'recorded fixture HR integral')
    return HRDonor(key,role,datetime(2026,1,1,tzinfo=timezone.utc),4,44,q(rate),
        q(4) if role=='intervals' else None,q(2) if role=='intervals' else None,
        15 if role=='intervals' else None,3 if role=='intervals' else 0,1,'recorded test phases')


def bank():
    return tuple(donor(role,f'{role}-{i}',rate=2+i*.3)
                 for role in ('easy','long','intervals') for i in range(3))


def test_resizing_preserves_measured_hr_rate_not_measured_run_identity():
    original=donor(); saved=asdict(original)
    _,_,rec=setup('easy',4)
    a=transformed_entry(original,rec)
    b=transformed_entry(original,rec,miles=8,at=rec.planned_for+timedelta(days=3))
    assert b.workload.running_minutes.low == a.workload.running_minutes.low*2
    assert b.workload.cardio_points.low == a.workload.cardio_points.low*2
    assert b.workload.structure['distance_scale']==2
    assert 'synthetic' in b.workload.distance_meters.basis
    assert asdict(original)==saved


def test_quality_resizes_recorded_work_and_recovery_hr_separately():
    _,_,rec=prescription()
    result=transformed_entry(donor('intervals'),rec)
    assert result.workload.running_minutes.low==41
    assert result.work_minutes.low==15
    assert result.workload.cardio_points.low==15*4+26*2
    assert result.workload.structure['bout_count']==3
    assert result.workload.structure['donor_key']=='recorded-1'
    assert result.workload.structure['work_time_scale']==1


def test_issue_envelope_is_frozen_before_distance_adherent_execution():
    _,_,rec=setup('easy',4)
    original=donor()
    expected=issued_envelope(original,rec)
    for miles in (3.8,4,4.2):
        actual=transformed_entry(original,rec,miles=miles)
        assert set(compare_execution(expected.workload,actual.workload).values())=={'within'}
    harder=transformed_entry(replace(original,total_rate=Quantity.point(4,'different recorded HR')),rec)
    assert compare_execution(expected.workload,harder.workload)['cardio_points']=='above'


def test_donor_selection_is_role_matched_and_not_conditioned_on_recovery_score():
    _,_,rec=setup('easy',4)
    selected=choose_donor(bank(),rec)
    assert selected.role=='easy'
    alternative=choose_donor(bank(),rec,exclude=(selected.key,))
    assert alternative.key!=selected.key
    changed=tuple(replace(d,total_rate=Quantity.point(5,'changed response')) for d in bank())
    assert choose_donor(changed,rec).key==selected.key
    _,_,quality=prescription()
    assert choose_donor(bank(),quality).role=='intervals'
    with pytest.raises(ValueError,match='donor'):
        choose_donor((donor('easy'),),quality)


@pytest.mark.parametrize('scenario',['exact','in_range','time_shift','different_donor','above_range'])
def test_daily_upload_hr_replay_preserves_bank_and_reports_adherence(scenario):
    _,make=fixture()
    days,states=make(12,{i:('easy',4) for i in (0,2,4,6,8,10)})
    original=bank(); before=asdict(original[0])
    result=replay(original,days,states[0].as_of,
        {'fixture':(AssessmentPolicy('arithmetic',130,320,24,24),)},scenario=scenario,days=7)
    assert len(result['daily'])==7
    assert len(result['uploads'])==4
    assert result['distance_compliant']==(3 if scenario=='above_range' else 4)
    assert result['numeric_work_compliant']==4
    assert asdict(original[0])==before
    assert not result['planner_influence']
    if scenario=='exact':
        for row in result['uploads']:
            assert row['before']==row['after']
    if scenario=='different_donor':
        assert all(row['expected_donor']!=row['actual_donor'] for row in result['uploads'])
