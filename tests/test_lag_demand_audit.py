from copy import deepcopy

from scripts.audit_lag_demand_reference import CONTROLS,signal_gate,summarize


def groups():
    controls={name:dict(supported_bouts=8,mean_absolute_log_error=.1 if name!='demand_regression' else .08)
              for name in CONTROLS}
    return {name:dict(common_cases=4,common_bouts=8,controls=deepcopy(controls))
            for name in ('all','intervals','tempo_threshold')}


def test_gate_requires_added_signal_over_mean_hr_not_just_duration():
    data=groups()
    assert signal_gate(data)['diagnostic_signal_gate_passed']
    data['all']['controls']['mean_hrr_regression']['mean_absolute_log_error']=.07
    assert not signal_gate(data)['diagnostic_signal_gate_passed']


def test_better_subset_cannot_hide_lost_support_or_a_failing_category():
    data=groups()
    data['all']['controls']['demand_regression']['supported_bouts']=7
    assert not signal_gate(data)['diagnostic_signal_gate_passed']
    data=groups()
    data['intervals']['controls']['demand_regression']['mean_absolute_log_error']=.11
    assert not signal_gate(data)['diagnostic_signal_gate_passed']


def test_unknown_target_demand_is_visible_not_a_zero_error_or_prediction_pass():
    case=dict(lag_reference_diagnostic=dict(models={name:dict(status='unidentified')
        for name in CONTROLS if name!='duration_kernel'},records=[dict(seconds=60,
            demand_measurement=dict(status='incomplete_hr',demand=None),predictions={})]))
    result=summarize([case])
    assert result['total_cases']==1 and result['common_cases']==0
    assert result['demand_statuses']=={'incomplete_hr':1}
    assert result['controls']['demand_regression']['pace_mae'] is None
    assert not signal_gate({name:result for name in ('all','intervals','tempo_threshold')})['diagnostic_signal_gate_passed']
