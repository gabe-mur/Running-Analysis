from copy import deepcopy
import math

import numpy as np
import pytest

from run_analysis.pace_comparability import fit_comparison_calibration
from run_analysis.physiology import relative_humidity_from_dewpoint
from scripts.humidity_hr_trial import window_features, fit_joint, forward_joint
from tests.test_pace_comparability import config, row
from tests.test_thermal_hr_trial import synthetic


def data(humidity_scale=.7):
    rows = synthetic(); rng = np.random.default_rng(200)
    for r in rows:
        b = rng.normal(0, .025)
        r.update(humidity_burden=b, humidity_percent=65+200*b,
            pace_no_weather=r['pace_no_weather']*math.exp(humidity_scale*b))
    return rows


def test_humidity_at_reference_is_zero_at_all_temperatures(config):
    cal = fit_comparison_calibration([], config); cal['heat']['mean'] = 0
    rh = relative_humidity_from_dewpoint(config['reference_conditions']['temperature_f'], config['reference_conditions']['dewpoint_f'])
    for temperature in (46., 55., 80.):
        original = row(config, temperature_f=temperature, relative_humidity_percent=rh)
        before = deepcopy(original)
        result = window_features([original], [1.], config, cal)
        assert result['humidity_burden'] == pytest.approx(0.)
        assert original == before


def test_humidity_sign_and_dewpoint_fallback(config):
    cal = fit_comparison_calibration([], config); cal['heat']['mean'] = 0
    dry = window_features([row(config, temperature_f=80., relative_humidity_percent=30.)], [1.], config, cal)
    wet = window_features([row(config, temperature_f=80., relative_humidity_percent=95.)], [1.], config, cal)
    assert wet['humidity_burden'] > 0 > dry['humidity_burden']
    assert window_features([row(config, relative_humidity_percent=None)], [1.], config, cal)['humidity_available']


def test_bad_or_missing_humidity_is_not_known_neutral(config):
    cal = fit_comparison_calibration([], config)
    for rh in (None, 500., float('nan')):
        r = row(config, relative_humidity_percent=rh, dewpoint_f=None)
        result = window_features([r], [1.], config, cal)
        assert not result['humidity_available'] and result['humidity_burden'] is None


def test_joint_model_recovers_independent_and_null_humidity():
    fit = fit_joint(data())
    assert fit['scales'] == pytest.approx([1., .7], abs=.03)
    assert abs(fit_joint(data(0.))['scales'][1]) < .01
    assert fit_joint(data(-.7))['scales'][1] < 0


def test_no_humidity_information_does_not_create_effect():
    rows = data()
    for r in rows:
        r['humidity_burden'] = 0.
    assert fit_joint(rows)['scales'][1] == 0


def test_seasonal_humidity_is_not_mistaken_for_a_correction():
    rows = data(0.)
    for r in rows:
        # A calendar-block constant supplies no within-block weather evidence.
        from scripts.humidity_hr_trial import day
        r['humidity_burden'] = .01*math.sin(math.floor(day(r)/56))
    fit = fit_joint(rows)
    assert fit['scales'][1] == pytest.approx(0., abs=1e-9)


def test_missing_temperature_retains_an_unavailable_diagnostic(config):
    cal = fit_comparison_calibration([], config)
    result = window_features([row(config, temperature_f=None)], [1.], config, cal)
    assert not result['humidity_available'] and result['humidity_burden'] is None


def test_temperature_control_matches_previous_air_fit():
    from scripts.thermal_hr_trial import fit_weather
    rows = data()
    assert fit_joint(rows, humidity=False)['scales'][0] == pytest.approx(fit_weather(rows, model='air')['scale'])


def test_forward_does_not_see_target_or_future_performance():
    rows = data(); old = deepcopy(rows); first = forward_joint(rows)
    target = first['cases'][4]['activity_id']
    changed = forward_joint([dict(r, pace_no_weather=r['pace_no_weather']*.8) if r['activity_id'] >= target else r for r in rows])
    assert changed['cases'][4]['prediction'] == pytest.approx(first['cases'][4]['prediction'])
    assert changed['cases'][4]['scales'] == first['cases'][4]['scales']
    assert old == rows


def test_uniform_gains_survive_and_duplicate_runs_rejected():
    rows = data(); original = fit_joint(rows)
    better = fit_joint([dict(r, pace_no_weather=r['pace_no_weather']/1.02) for r in rows])
    assert better['scales'] == pytest.approx(original['scales'], abs=1e-10)
    with pytest.raises(ValueError, match='Unique'):
        fit_joint(rows+[rows[0]])
