from datetime import timedelta

import pytest

from run_analysis.signal_structure import SignalSpan
from scripts.validate_quality_load import phase_layout, matched_metrics, forecast, summarize
from test_quality_session_model import AT, REF, assessed


MODEL = dict(intercept=.2, slope=.15, tau_seconds=45., speed_domain=(0., 8.))


def donor(a):
    return dict(start=a.run.start, end=a.run.end, available_at=a.run.available_at,
        nonwork_speed=2., work_speed=3., load_rate=.02)


def test_layout_covers_elapsed_without_adding_or_losing_time():
    assert phase_layout(((100.,220.),(300.,420.)),500.) == (
        ('nonwork',0.,100.),('work',100.,220.),('nonwork',220.,300.),
        ('work',300.,420.),('nonwork',420.,500.))
    assert phase_layout((),500.) == ()
    with pytest.raises(ValueError):
        phase_layout(((100.,220.),(200.,420.)),500.)


def test_load_comparison_masks_missing_hr_instead_of_treating_it_as_zero():
    prediction = (SignalSpan(0.,120.,.6,.6),)
    actual = (SignalSpan(0.,30.,.6,.6),SignalSpan(90.,120.,.6,.6))
    result = matched_metrics(prediction,actual,120.,130.)
    assert result['seconds'] == 60.
    assert result['observed_load'] == pytest.approx(1.)
    assert result['predicted_load'] == pytest.approx(1.)
    assert result['fraction_error'] == pytest.approx(0.)
    assert result['hr_rmse_bpm'] == pytest.approx(0.)


def test_linear_hr_errors_are_integrated_exactly_and_invalid_hr_is_not_clipped():
    actual = (SignalSpan(0.,60.,.5,.5),)
    result = matched_metrics((SignalSpan(0.,60.,.4,.6),),actual,60.,100.)
    assert result['hr_rmse_bpm'] == pytest.approx(10./3**.5)
    assert result['hr_bias_bpm'] == pytest.approx(0.)
    invalid = matched_metrics((SignalSpan(0.,60.,1.1,1.1),),actual,60.,100.)
    assert invalid['predicted_load'] is None
    assert invalid['fraction_error'] is None


def test_forecast_uses_only_prior_pace_and_carries_hr_state_across_phases():
    a = assessed()
    layout = phase_layout(((100.,220.),(300.,420.)),500.)
    result = forecast(layout, as_of=AT, history=[a], donors={a.run.key:donor(a)},
        response=MODEL, accounting_key=REF.accounting_key)
    assert result['status'] == 'predicted'
    assert result['source_keys'] == ['one']
    trace, diag = result['predictions']['model']
    assert trace[0].first == .45
    assert diag['segment_initializations'] == 1
    assert all(x.last == y.first for x,y in zip(trace,trace[1:]))
    assert trace[-1].end == 500.
    assert result['programs']['model'] == [(100.,2.),(120.,3.),(80.,2.),(120.,3.),(80.,2.)]
    # Future or old measurements cannot change this prediction; no target input
    # slot exists for observed speed/HR in the forecasting API.
    extra = [assessed(key='future',day=1,speed=8.),assessed(key='old',day=-60,speed=8.)]
    again = forecast(layout, as_of=AT, history=[a,*extra], donors={a.run.key:donor(a)},
        response=MODEL, accounting_key=REF.accounting_key)
    assert again['programs'] == result['programs']
    assert again['predictions'] == result['predictions']


def test_delayed_or_stale_donor_metadata_is_rejected():
    a = assessed()
    for changes in ({'available_at':AT+timedelta(seconds=1)}, {'start':AT-timedelta(days=31)}):
        with pytest.raises(ValueError):
            forecast(phase_layout(((100.,220.),),500.),as_of=AT,history=[a],
                donors={a.run.key:dict(donor(a),**changes)},response=MODEL,accounting_key=REF.accounting_key)


def test_missing_evidence_is_not_a_zero_prediction_or_a_pass():
    result = forecast(phase_layout(((100.,220.),),500.),as_of=AT,history=[],donors={},
        response=MODEL,accounting_key=REF.accounting_key)
    assert result['status'] == 'no_recent_quality_pace'
    assert 'predictions' not in result
    summary = summarize([result, {'status':'unresolved_work_alignment'}])
    assert summary['total'] == 2 and summary['paired'] == 0
    assert summary['model']['mean_absolute_fraction_error'] is None
    assert summary['baseline_mean_absolute_fraction_error'] is None
