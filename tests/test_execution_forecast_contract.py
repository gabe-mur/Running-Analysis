from dataclasses import replace
from datetime import timedelta
from itertools import product

import pytest

from run_analysis.workload_accounting import Quantity
from scripts.direct_quality_load import forecast_direct
from scripts.execution_forecast_contract import (TargetBand,IntendedPhase,LoadAllowance,IntendedExecution,
    CompletedExecution,FrozenExecution,issue_execution,compare_execution,save_execution,read_execution,layout)
from test_direct_quality_load import AT,history


def intent(allowance=True):
    return IntendedExecution('session',AT,AT+timedelta(hours=18),
        tuple(IntendedPhase(k,TargetBand(60.,50.,70.,'synthetic test bounds, not defaults'))
              for k in ('warmup','work','recovery','work','cooldown')),
        TargetBand(4.,3.8,4.2,'synthetic distance range'),'units','synthetic HR normalization',
        LoadAllowance(.5,.5,'synthetic allowance, not validated physiology') if allowance else None)


def completed(record,seconds=60.,distance=4.,load_delta=0.,kinds=None,load_bounds=None):
    phases=tuple((k,seconds) for k in (kinds or ('warmup','work','recovery','work','cooldown')))
    expected=forecast_direct(layout(phases),[history()],as_of=AT,accounting_key='units')['predicted_load']
    start=AT+timedelta(hours=16)  # Before suggested time, still after issuance.
    end=start+timedelta(seconds=sum(s for _,s in phases))
    return CompletedExecution('observed','session',start,end,end,phases,distance,
        Quantity(*load_bounds,basis='synthetic missing-HR bounds') if load_bounds else
        Quantity.point(expected+load_delta,'synthetic completion, not an observed run'),
        'units','synthetic HR normalization')


def compare(record,actual,**kwargs):
    return compare_execution(record,actual,as_of=actual.available_at,**kwargs)


def test_target_not_max_and_adherence_does_not_rewrite_actual_load():
    record=issue_execution(intent(),[history()])
    p=record.payload
    assert p['target_forecast']['predicted_load']==pytest.approx(5.)
    assert p['execution_scenarios']['high'] > p['target_forecast']['predicted_load']
    actual=completed(record,seconds=70.,distance=4.2)
    result=compare(record,actual)
    assert result['routine_variation_without_new_replan_signal']
    assert result['target_load']==5.
    assert result['actual_cardio']['estimate']==pytest.approx(350/60)
    assert result['actual_distance_miles']==4.2
    assert result['review_signals']==[] and result['calendar_action']=='none'


@pytest.mark.parametrize('distance',[3.8,4.,4.2])
@pytest.mark.parametrize('seconds',[50.,60.,70.])
def test_all_declared_edges_count_as_adherent_with_matching_response(distance,seconds):
    record=issue_execution(intent(),[history()])
    result=compare(record,completed(record,seconds=seconds,distance=distance))
    assert result['routine_variation_without_new_replan_signal']
    assert result['distance_adherence']=='within'


@pytest.mark.parametrize('delta,expected',[(1.,'higher_than_declared_allowance'),(-1.,'lower_than_declared_allowance')])
def test_in_range_execution_does_not_hide_higher_or_lower_cost(delta,expected):
    record=issue_execution(intent(),[history()])
    result=compare(record,completed(record,load_delta=delta))
    assert result['distance_adherence']=='within'
    assert result['response_status']==expected
    assert 'load_response_outside_declared_allowance' in result['review_signals']
    assert not result['routine_variation_without_new_replan_signal']


def test_readiness_change_remains_visible_and_does_not_command_calendar_edits():
    record=issue_execution(intent(),[history()])
    result=compare(record,completed(record),readiness_changes=('new fatigue report',))
    assert result['review_signals']==['readiness_changed']
    assert result['calendar_action']=='none'


def test_outside_range_is_review_signal_not_a_schedule_lock_or_automatic_edit():
    record=issue_execution(intent(),[history()])
    result=compare(record,completed(record,seconds=80.,distance=4.5))
    assert result['review_signals']==['execution_outside_declared_range']
    assert result['calendar_action']=='none'


def test_changed_workout_and_unknown_phases_are_not_false_adherence():
    record=issue_execution(intent(),[history()])
    actual=completed(record,kinds=('warmup','work','cooldown'))
    changed=compare(record,actual)
    assert changed['structure']=='changed'
    assert changed['response_status']=='changed_intent_not_adherent_response_test'
    unknown=compare(record,replace(actual,phases=None))
    assert unknown['structure']=='unknown'
    assert unknown['execution_conditioned_load'] is None
    assert not unknown['routine_variation_without_new_replan_signal']


def test_inferred_phase_mismatch_is_not_proof_of_changed_intent():
    record=issue_execution(intent(),[history()])
    actual=replace(completed(record,kinds=('warmup','work','cooldown')),phase_evidence='inferred')
    result=compare(record,actual)
    assert result['structure']=='unresolved_alignment'
    assert result['response_status']=='alignment_unknown'
    assert 'workout_structure_changed' not in result['review_signals']


def test_no_prediction_allowance_is_inferred_from_adherence_or_source_dispersion():
    record=issue_execution(intent(allowance=False),[history()])
    result=compare(record,completed(record))
    assert result['response_status']=='allowance_not_calibrated_or_declared'
    assert not result['routine_variation_without_new_replan_signal']
    assert record.payload['conditional_model_load_bounds'] is not None


def test_missing_hr_bounds_can_make_response_indeterminate():
    record=issue_execution(intent(),[history()])
    result=compare(record,completed(record,load_bounds=(4.9,8.)))
    assert result['response_status']=='uncertain_due_to_load_bounds'
    assert result['actual_cardio']['estimate'] is None


def test_frozen_issue_excludes_old_future_late_data_and_survives_round_trip(tmp_path):
    original=issue_execution(intent(),[history()])
    extras=[history('old',day=-31),history('future',day=1),replace(history('late'),available_at=AT+timedelta(days=1))]
    assert issue_execution(intent(),[history(),*extras])==original
    copy=original.payload
    copy['target_forecast']['predicted_load']=999.
    assert original.payload['target_forecast']['predicted_load']==5.
    path=tmp_path/'issued.json'
    save_execution(original,path)
    restored=read_execution(path)
    assert restored==original
    assert compare(restored,completed(restored))==compare(original,completed(original))
    with pytest.raises(FileExistsError):
        save_execution(original,path)
    with pytest.raises(ValueError):
        FrozenExecution(original.serialized+' ',original.fingerprint)


def test_revision_keeps_original_and_must_precede_actual_start():
    original=issue_execution(intent(),[history()])
    new=replace(intent(),issued_at=AT+timedelta(hours=1),phases=intent().phases[:2]+intent().phases[-1:])
    revised=issue_execution(new,[history()],parent=original,revision_reason='Athlete changed repetitions before running')
    assert revised.payload['parent_fingerprint']==original.fingerprint
    assert len(original.payload['intent']['phases'])==5
    actual=completed(original,kinds=('warmup','work','cooldown'))
    assert compare(original,actual)['structure']=='changed'
    assert compare(revised,actual)['structure']=='same'
    late=issue_execution(replace(new,issued_at=actual.end+timedelta(seconds=1)),[history()],parent=revised,revision_reason='Too late')
    with pytest.raises(ValueError):
        compare(late,actual)
    with pytest.raises(ValueError):
        issue_execution(new,[history()],parent=original)


def test_distance_precision_unknown_margins_and_accounting_mismatch():
    with pytest.raises(ValueError):
        replace(intent(),distance_miles=TargetBand(3.75,3.5,4.,'new target must be displayed precisely'))
    unknown=replace(intent(),phases=tuple(replace(p,seconds=TargetBand(60,None,None,'target only')) for p in intent().phases))
    record=issue_execution(unknown,[history()])
    assert record.payload['conditional_model_load_bounds'] is None
    assert compare(record,completed(record))['phase_adherence']==['unknown']*5
    with pytest.raises(ValueError):
        compare(record,replace(completed(record),measurement_context='different max/resting HR'))


def test_model_box_bounds_cover_all_mixed_duration_corners_without_calling_high_extreme():
    varied=replace(history(),phases=tuple(replace(p,load=p.load*(1.2 if i%2 else .8)) for i,p in enumerate(history().phases)))
    record=issue_execution(intent(),[varied])
    low,high=record.payload['conditional_model_load_bounds']
    for values in product((50.,60.,70.),repeat=5):
        forecast=forecast_direct(layout(zip(('warmup','work','recovery','work','cooldown'),values)),[varied],as_of=AT,accounting_key='units')
        assert low <= forecast['predicted_load'] <= high


def test_cold_start_keeps_forecast_unknown_not_zero():
    record=issue_execution(intent(),[])
    assert record.payload['target_forecast']['predicted_load'] is None
    assert record.payload['conditional_model_load_bounds'] is None
    result=compare(record,completed(record))
    assert result['response_status']=='unresolved_forecast'


def test_time_of_day_is_advisory_but_a_different_execution_date_stays_visible():
    record=issue_execution(intent(),[history()])
    actual=completed(record)
    assert compare(record,actual)['execution_date_status']=='same_day'
    late=replace(actual,start=actual.start+timedelta(days=1),end=actual.end+timedelta(days=1),
        available_at=actual.available_at+timedelta(days=1))
    result=compare(record,late)
    assert 'execution_date_changed' in result['review_signals']
    assert not result['routine_variation_without_new_replan_signal']
    assert result['calendar_action']=='none'


def test_unavailable_completion_and_wrong_session_are_rejected():
    record=issue_execution(intent(),[history()])
    actual=completed(record)
    with pytest.raises(ValueError):
        compare_execution(record,actual,as_of=actual.start)
    with pytest.raises(ValueError):
        compare(record,replace(actual,session_key='other session'))
