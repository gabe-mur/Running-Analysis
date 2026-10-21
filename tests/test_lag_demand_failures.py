from math import exp, log

import pytest

from scripts.audit_lag_demand_failures import design_diagnostic


def row(key, seconds, demand, speed, source=None):
    return dict(key=key, source_key=source or key, seconds=seconds, demand=demand,
                mean_hrr=demand, speed_mps=speed, kind='diagnostic')


def test_constant_measurement_window_does_not_identify_duration_but_does_identify_intensity():
    rows = [row(str(i), 120, demand, exp(.3+.8*demand)) for i, demand in enumerate((.5, .6, .7, .8))]
    result = design_diagnostic(rows, 'demand')
    assert result['rank'] == 2 and result['reduced_intensity_rank'] == 2
    assert result['constant_duration'] and result['coefficients'] is None


def test_covariance_decomposition_and_fit_are_exact_not_a_new_coefficient_tune():
    rows = [row(str(i), seconds, demand, exp(.3+.8*demand-.04*log(seconds)))
            for i, (seconds, demand) in enumerate(((60, .5), (60, .8), (180, .6), (300, .9), (600, .7)))]
    result = design_diagnostic(rows, 'demand')
    assert result['coefficients'] == pytest.approx((.3, .8, -.04))
    assert result['conditional_covariance'] == pytest.approx(
        result['between_source_covariance']+result['within_source_covariance'])
    assert result['within_source_covariance'] == pytest.approx(0)


def test_sole_nonwindow_duration_source_has_unit_design_leverage():
    rows = [row(str(i), 120, demand, exp(.3+.8*demand)) for i, demand in enumerate((.5, .6, .7, .8))]
    rows.append(row('longer', 1000, .85, 3.2))
    result = design_diagnostic(rows, 'demand')
    source = next(s for s in result['source_diagnostics'] if s['key'] == 'longer')
    assert source['design_leverage'] == pytest.approx(1)
