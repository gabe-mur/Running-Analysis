from copy import deepcopy

import numpy as np
import pytest

from scripts.aerobic_coherent_history import CoherentHistory
from scripts.aerobic_shared_state import fit_shared_state


def data():
    rng = np.random.default_rng(91)
    times = np.arange(-140., 1., 4.)
    gradients = np.column_stack([np.linspace(.1, .4, len(times)), np.sin(times/40)])
    return dict(y=2.3+.02*np.sin(times/35)+rng.normal(0., .01, len(times)),
        measurement=np.full(len(times), .012**2), gradients=gradients,
        covariance=np.diag([.02**2, .01**2]), times=times)


def test_curve_and_both_comparisons_share_exactly_the_same_points():
    curve = CoherentHistory(**data())
    history = {v['anchor']:v for v in curve.series([0., -28., -90., -40.])}
    for days in (28, 90):
        change = curve.compare(0., days)
        assert change['current_pace_min_mile'] == history[0.]['pace_min_mile']
        assert change['prior_pace_min_mile'] == history[-days]['pace_min_mile']
        assert change['estimate'] == history[-days]['log_pace']-history[0.]['log_pace']


def test_local_fit_is_existing_full_ml_with_one_period_and_full_shared_covariance():
    source = data(); curve = CoherentHistory(**source); point = curve.level(0.)
    mask = source['times'] > -28
    fitted = fit_shared_state(source['y'][mask], source['measurement'][mask], source['gradients'][mask],
        source['covariance'], np.zeros(sum(mask), int), times=source['times'][mask])
    assert point['log_pace'] == fitted['period_log_pace'][0]
    assert point['conditional_variance'] == fitted['conditional_mean_covariance'][0][0]


def test_future_runs_outside_period_and_viewport_do_not_change_a_historical_level():
    source = data(); curve = CoherentHistory(**source); expected = curve.level(-28.)
    altered = deepcopy(source)
    altered['y'][(source['times'] <= -56) | (source['times'] > -28)] += .1
    assert CoherentHistory(**altered).level(-28.) == expected
    curve.series([-100., -90., -60., -28., 0.])
    assert curve.level(-28.) == expected


def test_uniform_real_gain_is_retained_without_refitting_other_periods():
    source = data(); original = CoherentHistory(**source); altered = deepcopy(source)
    altered['y'][source['times'] > -28] -= np.log1p(.02)
    changed = CoherentHistory(**altered)
    assert original.level(0.)['pace_min_mile']/changed.level(0.)['pace_min_mile'] == pytest.approx(1.02, rel=1e-10)
    for days in (28, 90):
        assert changed.level(-days) == original.level(-days)
        assert changed.compare(0., days)['estimate']-original.compare(0., days)['estimate'] == pytest.approx(np.log1p(.02), abs=1e-10)


def test_sparse_history_preserves_one_observation_but_invents_no_noise_estimate():
    source = dict(y=[2.3], measurement=[.01**2], gradients=[[1.]], covariance=[[.02**2]], times=[0.])
    curve = CoherentHistory(**source)
    assert curve.level(0.)['evidence'] == 'single_observation'
    assert curve.level(0.)['conditional_variance'] is None
    assert not curve.compare(0., 28)['available']
    assert not curve.level(28.)['available']  # Exact left boundary excluded.


def test_inputs_and_cached_points_cannot_be_mutated_by_callers():
    source = data(); before = deepcopy(source); curve = CoherentHistory(**source)
    expected = curve.level(0.); changed = curve.level(0.)
    changed['weights'][0] = 999.
    source['y'][:] = -10
    assert curve.level(0.) == expected
    assert not np.any(curve.y == -10)
    for key in before:
        if key != 'y':
            np.testing.assert_equal(source[key], before[key])
    with pytest.raises(ValueError):
        curve.compare(0., 7)
