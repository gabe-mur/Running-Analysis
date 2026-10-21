import numpy as np
import pytest

from run_analysis.aerobic_sampling import robust_speed_samples, sampling_precision
from run_analysis.objective_modeling import _robust_weighted_speed, speed_to_pace_min_mile


@pytest.mark.parametrize('n', [1, 2, 9, 70, 200])
def test_batch_is_same_estimator_for_ragged_repeated_and_outlying_observations(n):
    rng=np.random.default_rng(1234+n)
    values=rng.lognormal(1.,.2,n);values[-1]*=3
    weights=rng.lognormal(-1.,2.,n)
    samples=[rng.integers(n,size=i) for i in range(1,101)]
    before=values.copy(),weights.copy()
    expected=[_robust_weighted_speed(values[ix].tolist(),weights[ix].tolist()) for ix in samples]
    np.testing.assert_allclose(robust_speed_samples(values,weights,samples),expected,rtol=2e-14,atol=2e-14)
    np.testing.assert_array_equal(values,before[0]);np.testing.assert_array_equal(weights,before[1])


def test_batch_keeps_identical_block_draws_floor_and_sampling_uncertainty():
    rng=np.random.default_rng(42)
    rows=[dict(moving_minutes_into_run=i) for i in range(75)]
    weights=rng.lognormal(0,2,75);speeds=rng.lognormal(0,.3,75)
    speeds[:10]=.01
    legacy=lambda ix:speed_to_pace_min_mile(_robust_weighted_speed(speeds[ix].tolist(),weights[ix].tolist()))
    original=sampling_precision(rows,weights,speeds,256,123,estimator=legacy)
    batch=sampling_precision(rows,weights,speeds,256,123)
    assert batch[0]==original[0]
    for a,b in zip(batch[1],original[1],strict=True):
        assert a['blocks']==b['blocks'] and a['block_minutes']==b['block_minutes']
        assert a['sampling_sd_min_mile']==pytest.approx(b['sampling_sd_min_mile'],rel=2e-13,abs=1e-13)
