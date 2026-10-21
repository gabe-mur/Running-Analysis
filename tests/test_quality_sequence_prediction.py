from datetime import timedelta
from dataclasses import replace

import pytest

from run_analysis.adaptive.direct_quality_load import PhaseDose, forecast_direct
from test_direct_quality_load import AT, history


def predict(records,layout,**kwargs):
    return forecast_direct(layout,records,as_of=AT,accounting_key='units',**kwargs)


def sequence():
    return (PhaseDose('warmup',60,None,60,1),PhaseDose('work',60,60,60,1),
        PhaseDose('recovery',60,60,60,1),PhaseDose('work',60,60,60,3),
        PhaseDose('cooldown',60,60,60,1))


def test_first_work_and_later_work_separate_without_intensity_or_family_input():
    prior=[history(phases=sequence())]
    layout=tuple((p.kind,i*60.,(i+1)*60.) for i,p in enumerate(sequence()))
    base=predict(prior,layout)
    candidate=predict(prior,layout,sequence_context=True)
    assert base['phases'][1]['rate']==base['phases'][3]['rate']==pytest.approx(2/60)
    assert candidate['phases'][1]['rate']==pytest.approx(1/60)
    assert candidate['phases'][3]['rate']==pytest.approx(3/60)
    assert candidate['predicted_load']==pytest.approx(base['predicted_load'])
    assert all(p['sequence_context_supported'] for p in candidate['phases'])


def test_novel_adjacency_keeps_supported_baseline_not_new_exact_workout_gate():
    prior=[history(phases=sequence())]
    layout=(('work',0.,90.),('cooldown',90.,150.))
    base=predict(prior,layout)
    candidate=predict(prior,layout,sequence_context=True)
    assert candidate['status']=='predicted'
    assert candidate['phases'][0]['sequence_context_fallback']
    assert candidate['phases'][0]['rate']==base['phases'][0]['rate']


def test_availability_missing_work_and_original_default_unchanged():
    good=history(phases=sequence())
    later=replace(good,key='late',available_at=AT+timedelta(seconds=1))
    layout=(('warmup',0.,60.),('work',60.,120.))
    assert predict([good],layout)==predict([good],layout,sequence_context=False)
    assert predict([good],layout,sequence_context=True)==predict([good,later],layout,sequence_context=True)
    assert predict([],layout,sequence_context=True)['predicted_load'] is None


def test_repeated_representation_does_not_multiply_session_vote():
    a=sequence();b=tuple(replace(p,load=p.load*2) for p in a)
    # Repeat complete observed blocks. Each has the same work-after-recovery
    # context; additional repetitions must not create additional source votes.
    long=(a[0],*a[1:4]*10,a[-1])
    layout=(('warmup',0.,60.),('work',60.,120.),('recovery',120.,180.),('work',180.,240.))
    one=predict([history(phases=a),history('other',phases=b)],layout,sequence_context=True)
    many=predict([history(phases=long),history('other',phases=b)],layout,sequence_context=True)
    assert one['phases'][3]['rate']==pytest.approx(many['phases'][3]['rate'])


def test_report_unknown_observations_and_issued_deviations_remain_explicit():
    from scripts.audit_quality_sequence_prediction import stats, issued_summary
    assert stats([None,dict(fraction_error=None)])['mae'] is None
    exact=dict(predicted_load=12.,observed_low=10.,observed_high=10.,relative_error_bounds=(.2,.2))
    better=dict(exact,predicted_load=11.,relative_error_bounds=(.1,.1))
    cases=[dict(key='known',current_actual_role='intervals',contract=dict(status='resolved'),
        whole_plan_vs_execution=exact,sequence_trial=dict(issued=dict(whole_plan_vs_execution=better)),
        execution_differences=dict(full_adherence_established=False)),
        dict(key='unknown',current_actual_role='tempo_threshold',contract=dict(status='unresolved')),
        dict(key='different',current_actual_role='easy',contract=dict(status='resolved'),
        whole_plan_vs_execution=exact,sequence_trial=dict(issued=dict(whole_plan_vs_execution=better)))]
    report=issued_summary(cases)
    assert len(report['cases'])==3 and report['paired_quality']==1
    assert report['baseline_mae_bounds']==pytest.approx((.2,.2))
    assert report['candidate_mae_bounds']==pytest.approx((.1,.1))
    assert not report['prospective_acceptance']


def test_issued_error_bounds_keep_partial_hr_and_crossing_zero():
    from scripts.audit_quality_sequence_prediction import issued_summary
    load=dict(relative_error_bounds=(-.1,.2))
    row=dict(key='partial',current_actual_role='intervals',contract=dict(status='resolved'),
        whole_plan_vs_execution=load,sequence_trial=dict(issued=dict(whole_plan_vs_execution=load)))
    result=issued_summary([row])
    assert result['paired_quality']==1
    assert result['baseline_mae_bounds']==pytest.approx((0.,.2))


def test_console_summary_serializes_unresolved_dates():
    import json
    from scripts.audit_quality_sequence_prediction import summarize
    from scripts.audit_work_stimulus import encode
    cases=[dict(key='missing',start=AT,direct_load=dict(metrics=None,
        forecast=dict(status='no_recent_phase_evidence')),sequence_trial=dict(metrics=None))]
    summary=summarize(cases)
    assert summary['baseline_supported']==0 and summary['baseline']['mae'] is None
    assert json.loads(json.dumps(summary,default=encode))['unsupported'][0]['start']==AT.isoformat()
