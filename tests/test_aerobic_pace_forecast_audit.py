import math

import numpy as np
import pytest

from scripts.audit_aerobic_pace_forecast import (
    damped_days, fatigue_audit, fit_trend, forecast, score, trend_holdouts,
)


def test_damping_is_continuous_bounded_and_has_the_correct_half_life():
    tau = 56/math.log(2)
    assert damped_days(0, 56) == 0
    assert damped_days(56, 56) == pytest.approx(tau/2)
    assert damped_days(10000, 56) == pytest.approx(tau)
    assert damped_days(211, 56) < damped_days(211, 84) < 211
    assert damped_days(211, None) == 211
    with pytest.raises(ValueError):
        damped_days(-1, 56)
    with pytest.raises(ValueError):
        damped_days(10, 0)


def test_fit_recovers_log_trend_and_does_not_mutate_inputs():
    t = np.arange(0, 120, 4.)
    y = math.log(10)-.001*t
    original = y.copy()
    f = fit_trend(y, t, np.ones(len(t)), history_days=84)
    assert f['slope'] == pytest.approx(-.001)
    assert f['log_level'] == pytest.approx(y[-1])
    assert forecast(f, 211) > forecast(f, 211, half_life=None)
    np.testing.assert_array_equal(y, original)
    assert fit_trend(y[:4], t[:4], np.ones(4)) is None


def test_feature_centering_is_training_only_and_requires_known_feature():
    t = np.arange(0, 120, 4.)
    feature = np.sin(t*.8)+.01*t
    y = math.log(10)-.001*t+.03*feature
    f = fit_trend(y, t, np.ones(len(t)), history_days=366, feature=feature)
    assert f['slope'] == pytest.approx(-.001)
    assert f['feature_coefficient']/f['feature_scale'] == pytest.approx(.03)
    assert math.log(forecast(f, 0, feature_value=float(feature[-1]))) == pytest.approx(y[-1])
    with pytest.raises(ValueError):
        forecast(f, 10)


def samples():
    t = np.arange(0, 210, 4.)
    rows = [dict(activity_id=i, start_time=f'2026-01-{i+1:02d}',
        pace_min_mile=10*math.exp(-.001*v)) for i,v in enumerate(t)]
    return rows, t, np.ones(len(t))*.001


def test_chronological_holdouts_have_no_future_outcome_leakage():
    rows,t,variance=samples()
    a=trend_holdouts(rows,t,variance)
    changed=[dict(r) for r in rows]
    for r in changed[30:]:
        r['pace_min_mile']*=1.5
    b=trend_holdouts(changed,t,variance)
    assert len(a)==len(b)
    for left,right in zip(a,b):
        assert left['origin_id']<left['target_id']
        if left['origin_id']<30:
            assert left['predicted']==pytest.approx(right['predicted'])
    assert score([p for p in a if p['model']=='linear'])['mae_seconds_per_mile']<1e-8


def test_scoring_does_not_give_repeated_targets_extra_total_weight():
    rows=[dict(target_id=1,predicted=10,actual=9),dict(target_id=2,predicted=10,actual=10)]
    assert score(rows)['mae_seconds_per_mile']==30
    assert score(rows+[rows[0]])['mae_seconds_per_mile']==30
    assert score([])['pairs']==0


def test_fatigue_can_explain_a_real_effect_without_forcing_its_sign():
    rows,t,variance=samples()
    fatigue=np.sin(t*.8)+.01*t
    for r,v,f in zip(rows,t,fatigue):
        r['pace_min_mile']=10*math.exp(-.001*v+.04*f)
    result=fatigue_audit(rows,t,variance,fatigue)
    assert result['pace_effect_high_vs_low_seconds_per_mile']>0
    assert result['summaries']['time_plus_fatigue']['mae_seconds_per_mile'] < result['summaries']['time_only']['mae_seconds_per_mile']
    inverse=[{**r,'pace_min_mile':10*math.exp(-.001*v-.04*f)} for r,v,f in zip(rows,t,fatigue)]
    assert fatigue_audit(inverse,t,variance,fatigue)['pace_effect_high_vs_low_seconds_per_mile']<0


def test_missing_fatigue_is_not_a_zero_load_observation():
    rows,t,variance=samples()
    assert not fatigue_audit(rows,t,variance,[np.nan]*len(rows))['available']


def test_multi_run_targets_are_held_out_and_require_several_observations():
    rows,t,variance=samples()
    predictions=trend_holdouts(rows,t,variance,target_mode='block')
    assert predictions
    assert all(p['outcome_runs']>=2 for p in predictions)
    assert all(all(int(i)>p['origin_id'] for i in p['target_id'].split(',')) for p in predictions)
    assert score([p for p in predictions if p['model']=='linear'])['mae_seconds_per_mile']<1e-8
    with pytest.raises(ValueError):
        trend_holdouts(rows,t,variance,target_mode='invalid')
