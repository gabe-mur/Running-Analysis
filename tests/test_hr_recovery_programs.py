from dataclasses import replace
from datetime import timedelta
import pytest

from run_analysis.recovery_assessment import AssessmentPolicy, assess_next_session
from run_analysis.workload_accounting import Quantity
from scripts.audit_hr_recovery_programs import (trace, score, tradeoff, demand_delta, scaled_entry,
    numeric_fixture, ownership_split, RECOVERY_COMPONENTS, COACHING_COMPONENTS, recovery_controls)
from scripts.hr_backed_sandbox import transformed_entry
from test_hr_backed_sandbox import donor
from test_expected_completion import setup


def fixture():
    _,_,rec=setup('easy',4)
    entry=transformed_entry(donor(),rec)
    previous=replace(entry,key='prior',start=entry.start-timedelta(days=1),end=entry.end-timedelta(days=1))
    return previous,entry,AssessmentPolicy('test',100,200,24,24)


def test_cached_demand_matches_assessor_across_budgets():
    previous,event,p=fixture()
    cached=trace([previous],[(0,'easy',event)],p)
    changed=replace(p,running_minutes_envelope=50,cardio_points_envelope=90)
    result=score(cached,changed)
    direct=assess_next_session([previous],event.workload,event.start,changed,proposed_work_minutes=event.work_minutes)
    assert result['full']==pytest.approx((direct['excess_low'],direct['excess_high']))
    assert trace([previous],[(0,'easy',event)],changed)==cached


def test_soft_tail_keeps_physical_load_from_hard_plan():
    _,event,p=fixture()
    last=replace(event,key='last',start=event.start+timedelta(days=20),end=event.end+timedelta(days=20))
    tail=replace(event,key='tail',start=event.start+timedelta(days=21),end=event.end+timedelta(days=21))
    p=replace(p,running_minutes_envelope=50,cardio_points_envelope=100)
    result=score(trace([],[(20,'easy',last),(21,'easy',tail)],p),p)
    reset=score(trace([],[(21,'easy',tail)],p),p)
    assert result['hard']==(0,0)
    assert result['tail'][0]>reset['tail'][0]
    assert result['full']==result['tail']


def test_stage_scaling_changes_end_time_and_preserves_rates():
    _,event,_=fixture()
    bigger=scaled_entry(event,2)
    assert bigger.start==event.start
    assert bigger.end-event.start==2*(event.end-event.start)
    assert bigger.workload.cardio_points.low==event.workload.cardio_points.low*2
    assert event.workload.running_minutes.low==44


def test_timing_changes_follow_whole_path_not_independent_gap_minima():
    previous,event,p=fixture()
    next_event=replace(event,key='second',start=event.start+timedelta(days=1),end=event.end+timedelta(days=1))
    rows=trace([previous],[(0,'easy',event),(1,'easy',next_event)],p,timing='alternating_early')
    first=replace(event,start=event.start-timedelta(hours=6),end=event.end-timedelta(hours=6))
    expected=assess_next_session([previous,first],next_event.workload,next_event.start+timedelta(hours=6),p,
                                 proposed_work_minutes=next_event.work_minutes)
    assert rows[1]['totals']['running_minutes']==(expected['channels']['running_minutes']['total_low'],
                                                expected['channels']['running_minutes']['total_high'])


@pytest.mark.parametrize('other,recovery,expected',[
    (-1,(-2,-1),'candidate_dominates'),(1,(1,2),'baseline_dominates'),
    (1,(-2,-1),'weight_or_evidence_dependent'),(-1,(1,2),'weight_or_evidence_dependent'),
    (0,(-1,1),'tie_or_uncertain'),(0,(0,0),'tie_or_uncertain')])
def test_tradeoffs_do_not_choose_an_arbitrary_exchange_weight(other,recovery,expected):
    assert tradeoff(other,recovery)==expected


def test_interval_comparison_does_not_cancel_uncertainty():
    assert demand_delta((1,3),(2,4))==(-3,1)


def test_unbounded_hr_is_not_ranked_as_zero():
    _,event,p=fixture()
    rows=trace([],[(0,'easy',event)],p)
    rows[0]['totals']['cardio_points']=(0,None)
    with pytest.raises(ValueError,match='Unbounded'):
        score(rows,p)


def test_numeric_easy_prescription_unchanged():
    _,_,rec=setup('easy',4)
    assert numeric_fixture(rec) is rec


def legacy_components():
    r={k:float(i+1) for i,k in enumerate(RECOVERY_COMPONENTS)}
    c={k:0. for k in COACHING_COMPONENTS}
    c.update(load_ratio=12.,weather=3.,key_session_cadence=4.)
    parts=dict(recovery=sum(r.values()),coaching=sum(c.values()),target=7.,support=2.,shape=1.,switching=.5)
    return dict(cost=sum(parts.values()),components=parts,recovery_components=r,coaching_components=c)


def test_single_owner_removes_all_recovery_and_ratio_but_preserves_other_objectives():
    legacy=legacy_components()
    result=ownership_split(legacy)
    assert result['single_owner_retained']==7+2+1+.5+3+4
    assert result['eligible_for_score_comparison']
    assert 'not_assessed' in result['production_eligibility']
    assert result['without_short_density']>result['single_owner_retained']
    assert legacy['coaching_components']['load_ratio']==12


def test_unresolved_readiness_is_not_silently_erased_or_treated_as_clearance():
    legacy=legacy_components()
    legacy['coaching_components']['readiness']=100
    legacy['components']['coaching']+=100; legacy['cost']+=100
    result=ownership_split(legacy)
    assert not result['eligible_for_score_comparison']
    assert result['unresolved_readiness']=={'readiness':100}
    assert result['single_owner_retained']>=100


def test_unknown_or_unreconciled_legacy_term_fails_boundary():
    legacy=legacy_components(); legacy['recovery_components']['new_penalty']=1
    with pytest.raises(ValueError,match='Unmapped'): ownership_split(legacy)
    legacy=legacy_components(); legacy['cost']+=1; legacy['components']['coaching']+=1
    with pytest.raises(ValueError,match='reproduce'): ownership_split(legacy)


def test_legacy_eligibility_does_not_veto_replacement_comparison():
    legacy=legacy_components()
    legacy['passes_checked_constraints']=True
    expected=ownership_split(legacy)
    legacy['passes_checked_constraints']=False
    legacy['sequence_checks']={'checks_pass':False,'failed_offsets':[2]}
    assert ownership_split(legacy)==expected


def test_provisional_volume_cannot_be_stacked_on_finalized_volume_owner():
    legacy=legacy_components()
    legacy['coaching_components']['mileage_path']=1
    legacy['components']['coaching']+=1; legacy['cost']+=1
    with pytest.raises(ValueError,match='Provisional'): ownership_split(legacy)


def test_compression_controls_preserve_work_and_do_not_have_calendar_failure_rules():
    from test_projected_duration import prescription
    _,_,easy=setup('easy',4)
    _,_,long=setup('long',8)
    _,_,quality=prescription()
    origin=easy.planned_for
    events=[]
    for i,rec in enumerate((easy,long,quality)):
        rec=rec.model_copy(update={'planned_for':origin+timedelta(days=i*2)})
        events.append((i*2,rec.workout_type.value,transformed_entry(donor(
            'intervals' if i==2 else 'long' if i==1 else 'easy'),rec)))
    loose=AssessmentPolicy('deliberately permissive control',10000,10000,24,24)
    result=recovery_controls([],events,{'test':(loose,)})
    assert result['same_work_compressed']['hard_miles']==sum(r.planned_distance_miles for r in (easy,long,quality))
    for row in result.values():
        assert row['families']['test']['status_counts']=={'model_supported':1}
