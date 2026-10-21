from copy import deepcopy
from datetime import datetime, timedelta, timezone

import pytest

from scripts.audit_delivery_contract import audit


START=datetime(2026,9,22,tzinfo=timezone.utc)


def fixture(days=28, *, miles=4., every=2, targets=lambda i:17., defer=False):
    actual=[] if defer else [dict(occurred_at=(START+timedelta(days=i,hours=19)).isoformat(),
        workout_type='easy',distance_miles=miles) for i in range(days) if i%every==0]
    snapshots=[]
    for i in range(days):
        now=START+timedelta(days=i)
        planned=[dict(planned_for=(now+timedelta(days=j,hours=7)).isoformat(),workout_type='easy',
                      midpoint_miles=4.) for j in range(21)
                 if (j%2==1 if defer else (i+j)%every==0)]
        snapshots.append(dict(generated_at=now.isoformat(),decision_start_date=now.date().isoformat(),
            planned_sessions=planned,target_trajectory=[(targets(i),targets(i)+1)]*21,
            capacity_reference_miles=17.,trigger='scheduled_refresh',material_evidence_reasons=[]))
    return dict(status='passed',mode='perfect',failures=[],replans=snapshots,
        realized_execution=dict(start_at=START.isoformat(),observed_until=(START+timedelta(days=days)).isoformat(),
            complete=True,historical_sessions=[],sessions=actual))


def test_lowering_targets_does_not_erase_original_under_delivery():
    report=fixture(targets=lambda i:17. if i==0 else 12.)
    original=deepcopy(report)
    result=audit(report)
    first=result['windows'][0]
    assert first['actual_miles']==28.
    assert first['first_issued_integrated_low']==34.
    assert first['latest_pre_day_integrated_low']==pytest.approx(173/7)
    assert first['latest_target_masks_original_shortfall']
    assert first['opening_capacity_weekly_miles']==17.
    assert report==original


def test_promising_tomorrow_forever_is_detected_independently_of_score():
    report=fixture(defer=True)
    for snap in report['replans']:
        snap['planner_diagnostics']={'winning_candidate':{'total_cost':0.}}
    result=audit(report)
    assert result['total_completed_miles']==0
    assert len(result['deferrals'])==27
    assert result['deferrals'][-1]['consecutive_observed_deferrals']==27
    assert any(f['code']=='no_work_despite_positive_original_target' for f in result['findings'])
    assert result['original_gate_status']=='passed'  # Independent oracle catches the blind spot.


def test_observation_and_reference_coverage_are_not_assumed():
    short=audit(fixture(days=3))
    assert short['status']=='insufficient_observation'
    assert short['matured_14_day_windows']==0
    data=fixture()
    data['replans']=data['replans'][:1]
    result=audit(data)
    assert not result['daily_decision_coverage_complete']
    assert len(result['missing_daily_decision_dates'])==27
    assert not result['windows'][-1]['reference_complete']
    assert result['windows'][-1]['first_issued_integrated_low'] is None


def test_partial_day_is_never_matured_or_filled_with_history():
    data=fixture(days=14)
    data['realized_execution']['observed_until']=(START+timedelta(days=13,hours=20)).isoformat()
    data['realized_execution']['complete']=False
    data['realized_execution']['historical_sessions']=[dict(occurred_at=(START-timedelta(days=1)).isoformat(),distance_miles=100.)]
    result=audit(data)
    assert result['complete_observation_days']==13
    assert not result['windows']
    assert result['total_completed_miles']==28.


def test_full_windows_allow_uneven_days_without_day_level_shortfall_failure():
    result=audit(fixture(miles=5.))
    assert result['matured_14_day_windows']==15
    assert result['matured_28_day_windows']==1
    assert all(not r['latest_target_masks_original_shortfall'] for r in result['windows'])
    assert not any(f['code']=='no_work_despite_positive_original_target' for f in result['findings'])
    assert not result['deferrals']


def test_falling_capacity_is_reported_without_rebasing_the_opening_benchmark():
    report=fixture(targets=lambda i:17.-i*.1)
    for i,snap in enumerate(report['replans']):
        snap['capacity_reference_miles']=17.-i*.1
        snap['target_low_miles']=17.-i*.1
        snap['target_high_miles']=18.-i*.1
    result=audit(report)
    assert result['reference_series'][-1]['capacity_reference_miles']==pytest.approx(14.3)
    assert all(w['opening_capacity_weekly_miles']==17. for w in result['windows'])


def test_same_day_clock_shift_is_not_a_date_deferral_and_upload_is_not_daily_coverage():
    data=fixture(days=2,defer=True)
    first=data['replans'][0]
    after=deepcopy(first)
    after['generated_at']=(START+timedelta(hours=9)).isoformat()
    after['trigger']='post_upload'
    after['planned_sessions'][0]['planned_for']=(START+timedelta(days=1,hours=19)).isoformat()
    data['replans']=[first,after]
    result=audit(data)
    assert not result['deferrals']
    assert result['post_upload_decisions']==1
    assert result['missing_daily_decision_dates']==[(START+timedelta(days=1)).date().isoformat()]


def test_material_evidence_is_retained_not_declared_unjustified():
    data=fixture(targets=lambda i:17. if i==0 else 12.)
    for snap in data['replans'][1:]:
        snap['material_evidence_reasons']=['illness_fixture']
    result=audit(data)
    assert result['windows'][0]['material_evidence']
    assert result['findings'][0]['material_evidence']
    assert 'Not release clearance' in result['interpretation']


def test_retroactive_lower_target_cannot_rewrite_the_completed_day():
    data=fixture(days=14,targets=lambda i:17.)
    last=deepcopy(data['replans'][-1])
    last['generated_at']=(START+timedelta(days=13,hours=22)).isoformat()
    last['target_trajectory']=[(1.,2.)]*21
    data['replans'].append(last)
    result=audit(data)
    assert result['daily'][-1]['reference']['latest']==(17.,18.)


def test_ledger_after_observation_is_rejected_not_counted():
    data=fixture(days=3)
    data['realized_execution']['sessions'][0]['occurred_at']=(START+timedelta(days=4)).isoformat()
    with pytest.raises(ValueError,match='out-of-observation'):
        audit(data)


@pytest.mark.parametrize('defer',[False,True])
def test_real_daily_simulator_exposes_deferral_and_keeps_upload_replans(monkeypatch,defer):
    """Adversarial policy drives the real loop; this is not optimizer clearance."""
    from run_analysis import adherence_projection as projection
    from run_analysis.projection_gate import _json_ready
    from scripts.replay_synthetic_easy_dose import synthetic_history
    from test_issued_plan_execution import schedule
    from test_recommendation import CONFIG
    builds=[]
    def build(states,*args,**kwargs):
        origin=states[0].as_of
        result=schedule(origin,planned_day=0)
        prototype=result.days[0].recommendation
        days=[]
        for offset in range(21):
            when=origin.replace(hour=7,minute=0,second=0,microsecond=0)+timedelta(days=offset)
            absolute=(when.date()-START.date()).days
            run=(offset%2==1) if defer else (absolute%2==0)
            days.append(result.days[0].model_copy(update=dict(date=when.date(),planned_at=when if run else None,
                day_role='easy_run' if run else 'rest_day',recommendation=prototype.model_copy(
                    update=dict(planned_for=when,fitness_state_as_of=when)) if run else None)))
        result.days=days[:7]
        result.planning_days=days
        builds.append(origin)
        return result
    monkeypatch.setattr(projection,'build_weekly_schedule',build)
    template,history=synthetic_history(START)
    snapshots,actual=[],[]
    projection.simulate_adherence(template,history,CONFIG,START,simulation_days=14,
        replan_interval_days=1,emit_post_upload_replans=True,replan_observer=snapshots.append,
        execution_observer=lambda run,latest:actual.append(run))
    report=dict(mode='perfect',status='adversarial_fixture',replans=_json_ready(snapshots),
        realized_execution=dict(start_at=START.isoformat(),observed_until=(START+timedelta(days=14)).isoformat(),
            complete=True,historical_sessions=[dict(occurred_at=r.start_time.isoformat(),distance_miles=r.distance_miles) for r in history],
            sessions=[dict(occurred_at=r.start_time.isoformat(),distance_miles=r.distance_miles,workout_type=r.workout_type.value) for r in actual]))
    result=audit(report)
    assert result['daily_decision_coverage_complete']
    assert result['matured_14_day_windows']==1
    if defer:
        assert len(builds)==14 and not actual
        assert len(result['deferrals'])==13
        assert result['windows'][0]['no_work_despite_positive_original_target']
    else:
        assert len(actual)==7
        assert result['post_upload_decisions']==7
        assert len(builds)==21  # All daily decisions plus every completion.
        assert not result['deferrals']
