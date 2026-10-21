from copy import deepcopy
from datetime import datetime, timedelta, timezone
import math

import numpy as np
import pytest

from scripts.thermal_hr_trial import study_shape, hr_shift, thermal_windows, fit_weather, forward
from run_analysis.pace_comparability import fit_comparison_calibration, compare_run_windows
from tests.test_pace_comparability import config, row


def synthetic(*, scale=1., seasonal_only=False, null=False):
    rng = np.random.default_rng(20261015)
    start = datetime(1970, 1, 1, tzinfo=timezone.utc)+timedelta(days=56*350)
    result = []
    for block in range(10):
        for i in range(10):
            b = .02*math.sin(block/2)+(0. if seasonal_only else rng.normal(0, .025))
            level = math.log(10)-.012*block-.002*i
            result.append(dict(activity_id=len(result), start_time_utc=(start+timedelta(days=56*block+3*i)).isoformat(),
                pace_no_weather=math.exp(level+(0 if null else scale*b)), current_burden=0., thermal_burden=b,
                air_burden=b, temperature_f=55+200*b, sampled_hr_bpm=148., sampled_minutes=20., duration_minutes=35.,
                device='same', area='same', hour_sin=0., hour_cos=0., stop_fraction=0., previous_7d_miles=20., previous_28d_miles=80.))
    return result


def test_shape_uses_published_means_not_athlete_outcomes():
    shape = study_shape()
    assert shape['measurement_minutes'] == [15., 30., 45., 60.]
    assert shape['observed_hr_difference_bpm'] == [7., 11., 14., 15.]
    fitted = shape['amplitude_bpm']*(-np.expm1(-np.array(shape['measurement_minutes'])/shape['tau_minutes']))
    assert np.max(abs(fitted-shape['observed_hr_difference_bpm'])) < .5


def test_cooler_and_warmer_conditions_change_hr_mapping_in_opposite_directions(config):
    ref = row(config, average_hr_bpm=148.)
    cold = row(config, temperature_f=46., average_hr_bpm=148.)
    warm = row(config, temperature_f=72., average_hr_bpm=148.)
    assert hr_shift(ref, config) == pytest.approx(0.)
    assert hr_shift(cold, config) < 0 < hr_shift(warm, config)
    values, details = thermal_windows([cold, warm], config)
    assert values[0]['average_hr_bpm'] > 148 > values[1]['average_hr_bpm']
    assert all(d['applied'] for d in details)


def test_legacy_heat_is_disabled_in_candidate_not_applied_twice(config):
    cal = fit_comparison_calibration([], config); neutral = deepcopy(cal); neutral['heat']['mean'] = 0.
    warm = row(config, temperature_f=80., dewpoint_f=65.)
    shifted, _ = thermal_windows([warm], config)
    correct = compare_run_windows(shifted, [1.], config, neutral)
    double = compare_run_windows(shifted, [1.], config, cal)
    assert correct['contributions_seconds_per_mile']['weather'] == pytest.approx(0.)
    assert double['pace_min_mile'] < correct['pace_min_mile']


def test_missing_weather_preserves_original_section_and_reports_unknown(config):
    source = row(config, temperature_f=None); original = deepcopy(source)
    shifted, diagnostics = thermal_windows([source], config)
    assert len(shifted) == 1 and shifted[0] == original and source == original
    assert diagnostics[0]['reason'] == 'missing_thermal_context'


def test_known_personal_scale_can_be_learned_without_seasonal_fitness_becoming_weather():
    rows = synthetic(scale=1.6)
    fitted = fit_weather(rows, model='personal')
    assert fitted['scale'] == pytest.approx(1.6, abs=.03)
    seasonal = fit_weather(synthetic(scale=0, seasonal_only=True), model='personal')
    assert seasonal['scale'] == pytest.approx(1., abs=1e-9)
    assert seasonal['personal_information_fraction'] == pytest.approx(0., abs=1e-8)


def test_actual_null_can_override_literature_prior_without_forcing_a_positive_effect():
    fitted = fit_weather(synthetic(null=True), model='personal')
    assert abs(fitted['scale']) < .05
    negative = fit_weather(synthetic(scale=-.5), model='personal')
    assert negative['scale'] < 0


def test_uniform_real_gain_is_not_absorbed_by_temperature_fit():
    rows = synthetic(); before = deepcopy(rows)
    fitted = fit_weather(rows, model='personal')
    changed = fit_weather([dict(r, pace_no_weather=r['pace_no_weather']/1.03) for r in rows], model='personal')
    assert changed['scale'] == pytest.approx(fitted['scale'], abs=1e-10)
    assert rows == before


def test_forward_fits_cannot_observe_target_or_later_outcomes():
    rows = synthetic()
    original = forward(rows, model='personal')
    target = original['cases'][5]['activity_id']
    changed = forward([dict(r, pace_no_weather=r['pace_no_weather']*.8) if r['activity_id'] >= target else r for r in rows], model='personal')
    case = next(r for r in changed['cases'] if r['activity_id'] == target)
    assert case['prediction'] == pytest.approx(original['cases'][5]['prediction'])
    assert case['scale'] == pytest.approx(original['cases'][5]['scale'])


def test_duplicate_run_does_not_multiply_evidence():
    rows = synthetic()
    with pytest.raises(ValueError, match='Unique'):
        fit_weather(rows+[rows[0]], model='personal')


def test_no_weather_control_cannot_accidentally_apply_existing_heat():
    rows = synthetic()
    for r in rows:
        r['current_burden'] = .04
    fitted = fit_weather(rows, model='none')
    assert fitted['scale'] == 0.
    cases = forward(rows, model='none')
    assert cases['cases']
    with pytest.raises(ValueError, match='Known'):
        fit_weather(rows, model='typo')
