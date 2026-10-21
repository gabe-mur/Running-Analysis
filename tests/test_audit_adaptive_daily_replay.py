"""Report arithmetic only; no claim of physiological or search validation."""
from datetime import date
from types import SimpleNamespace

from scripts.audit_adaptive_daily_replay import edit_counts,target_integral,cardio_comparison


def test_churn_keeps_small_distance_edits_and_does_not_call_them_zero():
    rows=[dict(day='2026-10-02',date_changed=False,role_changed=False,distance_change=.1),
          dict(day='2026-10-04',date_changed=True,role_changed=False,distance_change=None)]
    event=dict(index=1,as_of='2026-09-30T04:00:00-04:00',reason='daily_rollover',
        churn={'4':rows[:1],'7':rows})
    assert edit_counts([event],4)['dose_edits']==1
    assert edit_counts([event],4)['largest_absolute_dose_edit']==.1
    seven=edit_counts([event],7)
    assert seven['edited_slots']==2 and seven['date_edits']==1 and seven['type_edits']==0


def test_target_integral_uses_dated_rates_not_a_calendar_week_quota():
    segments=[SimpleNamespace(start=date(2026,9,28),end=date(2026,10,1),miles_per_7_days=17.5),
        SimpleNamespace(start=date(2026,10,1),end=date(2026,10,5),miles_per_7_days=21.)]
    assert target_integral(segments,date(2026,9,30),date(2026,10,3))==8.5


def test_partial_hr_coverage_retains_bounds_and_does_not_invent_a_point_estimate():
    r=cardio_comparison(dict(target_load=100.,actual_cardio=dict(low=90.,high=110.,estimate=None)))
    assert r['actual_cardio'] is None and r['cardio_difference_percent'] is None
    assert r['actual_cardio_bounds']==[90.,110.]
    assert r['cardio_difference_percent_bounds'][0]<0<r['cardio_difference_percent_bounds'][1]
def test_quality_to_quality_swap_cannot_hide_behind_unchanged_role():
    from types import SimpleNamespace
    from scripts.audit_adaptive_daily_replay import quality_edit_counts
    def dose(family,recipe,steps):
        return SimpleNamespace(role='quality',workout_definition=dict(family=family,recipe_key=recipe,structure=steps))
    catalog={'old':dose('progression','progression',['half easy','half Z3']),
        'new':dose('threshold','threshold_continuous',['warmup','threshold','cooldown'])}
    edit=dict(day='2026-10-10',before='old',after='new',role_changed=False,date_changed=False,distance_change=0.)
    events=[dict(index=1,as_of='2026-10-09',churn={'7':[edit]})]
    result=quality_edit_counts(events,{1:catalog},7)
    assert result['family_edits']==result['recipe_edits']==result['structure_edits']==1


def test_explicit_checkpoint_report_keeps_partial_status(tmp_path):
    import json
    import pytest
    from scripts.audit_adaptive_daily_replay import replay_report
    checkpoint=dict(status='checkpoint_no_fast_fail',completed_days=7,continuation_days=21)
    (tmp_path/'checkpoint-0007.json').write_text(json.dumps(checkpoint))
    (tmp_path/'lifecycle-summary.json').write_text(json.dumps(dict(
        status='bounded_screen_complete',completed_days=21,requested_days=21)))
    path,report=replay_report(tmp_path,'checkpoint-0007.json')
    assert path.name=='checkpoint-0007.json'
    assert report['completed_days']==7 and report['requested_days']==21
    assert report['status']=='checkpoint_no_fast_fail'
    assert replay_report(tmp_path)[1]['completed_days']==21
    with pytest.raises(ValueError,match='inside the replay root'):
        replay_report(tmp_path,'../checkpoint-0007.json')
    assert json.loads(path.read_text())==checkpoint


def test_native_quality_gate_distinguishes_recipe_swaps_from_same_recipe_doses():
    from scripts.run_adaptive_lifecycle import fast_fail_quality_edits
    def dose(family,recipe,structure):
        return SimpleNamespace(role='quality',workout_definition=dict(
            family=family,recipe_key=recipe,structure=structure))
    catalog={'old':dose('long_intervals','aerobic_timed',['2x3 min']),
        'new':dose('short_intervals','pickups_distance',['3x400 m']),
        'dose':dose('long_intervals','aerobic_timed',['3x3 min'])}
    edit=dict(day='2026-10-10',before='old',after='new',role_changed=False,
        date_changed=False,distance_change=.2)
    event=dict(index=1,as_of='2026-10-09',churn={'7':[edit]})
    assert len(fast_fail_quality_edits(event,catalog,7))==1
    edit['after']='dose'
    assert fast_fail_quality_edits(event,catalog,7)==[]


def test_first_visible_tail_revision_is_not_a_visible_promise_change():
    from scripts.audit_adaptive_daily_replay import visible_edit_counts,visible_dose_history
    from run_analysis.adaptive.new_model_daily_harness import churn
    def dose(family,miles):
        return SimpleNamespace(role='quality',miles=miles,workout_definition=dict(
            family=family,recipe_key=family,structure=[family]))
    catalog={'old':dose('long_intervals',3.1),'new':dose('short_intervals',3.3)}
    def event(index,start,key):
        program=[('2026-10-16',key)]
        return dict(index=index,decision=start,as_of=start,reason='post_upload',program=program,
            churn=churn(events[-1]['program'] if index else (),program,date.fromisoformat(start),catalog))
    events=[]
    events.append(event(0,'2026-10-09','old'))  # UI ends October15.
    events.append(event(1,'2026-10-10','new'))  # UI first shows October16.
    catalogs={i:catalog for i in range(3)}
    result=visible_edit_counts(events,catalogs,7)
    assert result['prescription_edits']['edited_slots']==0
    assert result['quality_edits']['family_edits']==0
    assert result['entering_window_edits'][0]['day']=='2026-10-16'
    assert visible_dose_history(events,catalogs,7)['changed_dates']==[]
    events.append(event(2,'2026-10-11','old'))
    result=visible_edit_counts(events,catalogs,7)
    assert result['quality_edits']['family_edits']==1
    assert visible_dose_history(events,catalogs,7)['family_edits']==1
    from scripts.run_adaptive_lifecycle import fast_fail_quality_edits
    from scripts.audit_adaptive_daily_replay import visible_churn_event
    assert fast_fail_quality_edits(visible_churn_event(events,catalogs,7),catalog,7)


def test_visible_rest_promises_and_cumulative_small_doses_are_kept():
    from scripts.audit_adaptive_daily_replay import visible_edit_counts,visible_dose_history
    from run_analysis.adaptive.new_model_daily_harness import churn
    catalog={str(m):SimpleNamespace(role='easy',miles=m,workout_definition=None) for m in (3.,3.3,3.6)}
    events=[]
    for i,key in enumerate((None,'3.0','3.3','3.6')):
        program=[('2026-10-12',key)] if key else []
        events.append(dict(index=i,decision='2026-10-09',as_of='2026-10-09',reason='post_upload',
            program=program,churn=churn(events[-1]['program'] if i else (),program,date(2026,10,9),catalog)))
    catalogs={i:catalog for i in range(4)}
    result=visible_edit_counts(events,catalogs,7)
    assert result['prescription_edits']['date_edits']==1
    assert result['prescription_edits']['dose_edits']==2
    history=visible_dose_history(events,catalogs,7)
    assert round(history['largest_dose_span'],5)==.6
    assert round(history['total_dose_movement'],5)==.6
