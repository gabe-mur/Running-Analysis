from copy import deepcopy
from datetime import datetime,timedelta,timezone

import pytest

from scripts.audit_issued_quality_load import parse_issued_structure, issue_boundary, whole_load_error
from scripts.direct_quality_load import checked_layout, forecast_direct
from test_direct_quality_load import history, AT


def prescription(reps=3,seconds=300.,recovery=120.,placement='between efforts'):
    return dict(generated_at=AT.isoformat(),structure=[dict(duration_minutes=12.),
        dict(phase='work',repetitions=reps,work_duration_minutes=seconds/60,
             recovery_duration_minutes=recovery/60,instruction=placement),dict(duration_minutes=10.)])


def test_parse_between_efforts_preserves_exact_issued_duration():
    result = parse_issued_structure(prescription())
    assert result['total_seconds'] == 41*60
    assert result['recovery_count'] == 2
    assert [k for k,_,_ in result['layout']] == ['warmup','work','recovery','work','recovery','work','cooldown']


def test_after_each_recovery_not_silently_removed_or_merged_into_cooldown():
    result = parse_issued_structure(prescription(8,60,90,'after each'))
    assert result['total_seconds'] == 42*60
    assert result['recovery_count'] == 8
    assert [k for k,_,_ in result['layout']][-2:] == ['recovery','cooldown']
    assert checked_layout(result['layout']) == result['layout']
    forecast = forecast_direct(result['layout'],[history()],as_of=AT,accounting_key='units')
    assert [p['kind'] for p in forecast['phases']][-2:] == ['recovery','cooldown']


def test_variable_fartlek_or_missing_phase_extent_is_unknown_not_midpoint():
    payload = prescription()
    payload['structure'][1].update(work_duration_minutes=None,work_duration_range_minutes=[.75,1.5])
    payload['structure'][0]['duration_minutes'] = None
    result = parse_issued_structure(payload)
    assert result['layout'] is None
    assert result['status'] == 'unresolved_prescribed_durations'


def test_ambiguous_recovery_position_never_guessed():
    assert parse_issued_structure(prescription(placement='jog to recover'))['status']=='unresolved_recovery_placement'
    assert parse_issued_structure(prescription(placement='after each and between efforts'))['layout'] is None


def test_issue_uses_later_persistence_not_projected_fitness_date():
    payload=prescription()
    payload['fitness_state_as_of']=(AT+timedelta(hours=5)).isoformat()
    assert issue_boundary((AT+timedelta(seconds=1)).isoformat(),payload,AT+timedelta(hours=1)) == AT+timedelta(seconds=1)
    with pytest.raises(ValueError):
        issue_boundary((AT+timedelta(hours=2)).isoformat(),payload,AT+timedelta(hours=1))


def test_full_plan_comparison_keeps_unexecuted_tail_and_observed_bounds():
    result=whole_load_error(60.,40.,45.)
    assert result['predicted_load']==60.
    assert result['relative_error_bounds']==pytest.approx((60/45-1,.5))
    assert whole_load_error(None,40.,45.) is None


def test_issued_layout_is_independent_of_completed_timing_and_keeps_input_immutable():
    payload=prescription()
    before=deepcopy(payload)
    result=parse_issued_structure(payload)
    assert payload==before
    assert result['layout'][0]==('warmup',0.,720.)
    assert result['layout'][-1][2]==2460.


def test_mixed_implicit_explicit_layout_rejected():
    with pytest.raises(ValueError):
        checked_layout((('warmup',0.,60.),('work',60.,120.),('nonwork',120.,180.)))


def test_continuous_threshold_has_no_invented_repetitions_or_recovery():
    payload=dict(structure=[dict(duration_minutes=12),
        dict(phase='work',duration_minutes=18,instruction='Run at threshold effort'),dict(duration_minutes=10)])
    result=parse_issued_structure(payload)
    assert result['layout']==(('warmup',0.,720.),('work',720.,1800.),('cooldown',1800.,2400.))
    assert result['work_seconds_total']==1080 and result['recovery_count']==0


def test_explicit_progression_keeps_halves_without_inventing_hr_response():
    payload=dict(structure=[dict(phase='warmup',duration_minutes=20,prescription_basis='portion',target_zones=['Z1','Z2']),
        dict(phase='work',duration_minutes=18,prescription_basis='portion',target_zones=['Z2','Z3'])])
    result=parse_issued_structure(payload)
    assert result['work_seconds_total']==1080 and result['repetitions']==1
    assert result['primary_units']==['portion','portion']
    assert result['intended_effort_cues']==[['Z1','Z2'],['Z2','Z3']]
    # Historic textual halves alone do NOT define a numeric time prescription.
    for step in payload['structure']:step.pop('duration_minutes')
    assert parse_issued_structure(payload)['layout'] is None


def test_nonuniform_numeric_phases_keep_all_work_and_explicit_final_recovery():
    payload=dict(structure=[dict(phase=kind,duration_minutes=minutes,prescription_basis=unit)
        for kind,minutes,unit in [('warmup',12,'time'),('work',2,'distance'),('recovery',1.5,'time'),
            ('work',2.5,'distance'),('recovery',1.5,'time'),('cooldown',10,'time')]])
    result=parse_issued_structure(payload)
    assert result['work_seconds'] is None and result['work_seconds_total']==270
    assert result['recovery_count']==2
    assert result['primary_units'][1]=='distance'


def test_assignment_inventory_keeps_threshold_progression_and_different_session_cases():
    import sqlite3
    from scripts.audit_issued_quality_load import matched_assignments
    db=sqlite3.connect(':memory:')
    db.executescript('''CREATE TABLE activities(id,start_time_utc);
        CREATE TABLE activity_plan_matches(activity_id,planned_workout_id,match_confidence);
        CREATE TABLE planned_workout_history(id,schedule_generated_at,recommendation_json,
            plan_date,planned_for,workout_type);''')
    for i,kind in enumerate(('intervals','tempo_threshold','easy'),1):
        db.execute('INSERT INTO activities VALUES(?,?)',(i,AT.isoformat()))
        db.execute('INSERT INTO activity_plan_matches VALUES(?,?,?)',(i,i,1))
        db.execute('INSERT INTO planned_workout_history VALUES(?,?,?,?,?,?)',
            (i,(AT-timedelta(hours=1)).isoformat(),'{}','test','test',kind))
    assert [r[0] for r in matched_assignments(db,AT+timedelta(seconds=1))]==[1,2]
    assert matched_assignments(db,AT)==[]
    db.close()
