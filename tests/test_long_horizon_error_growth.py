from copy import deepcopy
from datetime import datetime, timedelta, timezone
import math

import pytest

from scripts.audit_long_horizon_error_growth import chronological, coverage_summary, descriptive

AT = datetime(2024,1,1,tzinfo=timezone.utc)


def examples(power=.21,n=25):
    result = []
    for i in range(n):
        origin = AT+timedelta(days=i*14)
        for j,h in enumerate((14,28,56,84,112)):
            error = (.009+i*.0001)*h**power
            result.append(dict(origin=origin.isoformat(),origin_id=i,target_id=f'{i}_{h}',
                outcome_ids=[i*10+j],outcome_end=(origin+timedelta(days=h)).isoformat(),
                elapsed=h/28,horizon=h,predicted=10.,actual=10*math.exp(-error),
                log_error=error,regime='health_affected'))
    return result


def test_known_slow_error_growth_persists_to_held_out_long_horizons():
    report = descriptive(examples())
    for h,r in report.items():
        assert r['short_fitted_power']==pytest.approx(.21)
        assert r['observed_28_to_long_power']==pytest.approx(.21)
        assert r['horizon_predictions']['quoted_slow_hypothesis']['predicted_to_actual_rms']==pytest.approx(1.)
        assert r['horizon_predictions']['sqrt_time']['predicted_to_actual_rms']>1.


def test_real_square_root_growth_is_not_forced_to_look_slow():
    report = descriptive(examples(.5))
    for h,r in report.items():
        assert r['short_fitted_power']==pytest.approx(.5)
        assert r['horizon_predictions']['quoted_slow_hypothesis']['predicted_to_actual_rms']<1.
        assert r['horizon_predictions']['sqrt_time']['predicted_to_actual_rms']==pytest.approx(1.)


def test_long_horizon_labels_never_fit_the_short_error_power():
    rows = examples()
    changed = deepcopy(rows)
    for r in changed:
        if r['horizon']>=56:
            r['log_error']*=3.
    a,b = descriptive(rows),descriptive(changed)
    for h in a:
        assert a[h]['short_fitted_power']==b[h]['short_fitted_power']
        assert a[h]['observed_28_to_long_power']!=b[h]['observed_28_to_long_power']


def test_chronological_calibration_has_no_future_error_or_target_run_leakage():
    rows = examples()
    report = chronological(rows,4)
    assert report['predictions']
    for r in report['predictions']:
        assert r['latest_calibration']<r['origin']
        assert set(r['calibration_outcome_ids']).isdisjoint(r['outcome_ids'])
    changed = deepcopy(rows)
    cutoff = (AT+timedelta(days=220)).isoformat()
    for r in changed:
        if r['outcome_end']>=cutoff:
            r['log_error']*=4.
    a,b = report['predictions'],chronological(changed,4)['predictions']
    key = lambda r:(r['origin'],r['target_id'],r['method'])
    before = lambda r:r['origin']<=cutoff
    assert {key(r):(r['exponent'],r['radius']) for r in a if before(r)} == {
        key(r):(r['exponent'],r['radius']) for r in b if before(r)}


def test_powers_have_identical_test_cases_and_calibration_support():
    report = chronological(examples(),8)
    methods = {r['method'] for r in report['predictions']}
    keys = [{(r['origin'],r['target_id']) for r in report['predictions'] if r['method']==m} for m in methods]
    assert all(k==keys[0] for k in keys)
    assert all(r['calibration_origins']>=8 for r in report['predictions'])
    assert not chronological(examples(n=4),8)['predictions']


def test_horizon_matching_does_not_compare_different_origin_compositions():
    rows = examples(n=10)
    rows = [r for r in rows if r['horizon']!=112 or r['origin_id']<4]
    report = descriptive(rows)
    assert report['112']['origins']==4
    assert report['56']['origins']==10
    assert all(v['origins']==4 for v in report['112']['stats'].values())


def test_interval_scoring_penalizes_needlessly_wide_bands():
    a = {**examples()[0],'radius':.1,'covered':True,'width_seconds':100.}
    b = {**a,'radius':.5,'width_seconds':500.}
    narrow,wide = coverage_summary([a]),coverage_summary([b])
    assert narrow['centered_coverage']==wide['centered_coverage']==1.
    assert narrow['mean_interval_score_log']<wide['mean_interval_score_log']
