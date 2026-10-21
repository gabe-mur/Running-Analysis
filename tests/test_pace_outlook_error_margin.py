from datetime import datetime, timedelta, timezone
import math

import numpy as np
import pytest

from scripts.audit_pace_outlook_error_margin import (
    audit, interval, matched_growth, scaled_errors, transfer_test, weighted_quantile, weights,
)

AT = datetime(2025, 1, 1, tzinfo=timezone.utc)


def row(i, horizon=28, error=.02):
    origin = AT+timedelta(days=i*40)
    return dict(origin=origin.isoformat(), outcome_end=(origin+timedelta(days=horizon)).isoformat(),
        origin_id=i, outcome_ids=[1000+2*i+(horizon==28)], target_id=f'{i}_{horizon}',
        horizon=horizon, elapsed=horizon/28, predicted=10., actual=10*math.exp(-error),
        log_error=error, regime='consistent')


def test_margin_uses_forecast_misses_not_model_spread_or_current_pace_floor():
    rows = [row(i, error=e) for i,e in enumerate((.01,.02,.03,.04))]
    band = interval(rows, days=28, pace=9, exponent=0)
    assert band['low_pace'] < band['high_pace'] < 9
    assert band['centered_low_pace'] < 9 < band['centered_high_pace']
    assert band['absolute_error_80_log'] == pytest.approx(.037)
    assert band['high_pace'] == pytest.approx(9*math.exp(-.01))


def test_error_scaling_uses_actual_elapsed_time_not_the_horizon_label():
    a = row(1, horizon=28, error=.03)
    a['elapsed'] = .5
    assert scaled_errors([a], 56, .5)[0] == pytest.approx(-.06)
    assert scaled_errors([a], 56, 1)[0] == pytest.approx(-.12)
    assert scaled_errors([a], 56, 0)[0] == pytest.approx(-.03)


def test_same_origin_pairs_recover_synthetic_error_growth():
    rows = [row(i, horizon=h, error=(.01+i*.001)*math.sqrt(h)) for i in range(10) for h in (14,28)]
    assert matched_growth(rows)['exponent'] == pytest.approx(.5)
    assert not matched_growth([row(0)])['available']


def test_repeated_origins_share_one_vote_when_outcomes_do_not_overlap():
    a = row(1)
    repeat = {**a, 'outcome_ids': [9999], 'target_id': 'repeat'}
    b = row(2)
    assert weights([a,repeat,b]) == pytest.approx([.25,.25,.5])


def test_transfer_calibration_excludes_future_and_needs_distinct_outcomes():
    errors = [row(i, 14, error=.01*(i+1)) for i in range(5)]
    target = row(6, 28, error=.02)
    result = transfer_test([*errors,target], 14,28,.5)
    assert result['all']['pairs'] == 1
    p = result['predictions'][0]
    assert p['band']['latest_error_outcome'] < target['origin']
    future = row(8,14,error=.9)
    assert transfer_test([*errors,target,future],14,28,.5)['predictions'][0]['band'] == p['band']
    duplicates = [{**r, 'outcome_ids': [10]} for r in errors]
    assert transfer_test([*duplicates,target],14,28,.5)['all']['pairs'] == 0


def test_health_contexts_remain_separate_and_do_not_change_center():
    rows = [row(i, horizon=h, error=(.01+i*.001)*math.sqrt(h)) for i in range(6) for h in (14,28)]
    mixed = [{**r, 'regime':'health_affected', 'origin':f'2026-01-{i+1:02d}',
        'outcome_ids':[2000+i], 'log_error':-.3} for i,r in enumerate(rows[:4])]
    report = audit([*rows,*mixed],211,9.)
    a = report['continued_consistency']['combined_short_error_projection']['independent_increments']
    b = report['all_recorded_contexts']['combined_short_error_projection']['independent_increments']
    assert a['pairs'] == len(rows) and b['pairs'] == len(rows)+len(mixed)
    assert math.sqrt(a['centered_low_pace']*a['centered_high_pace']) == pytest.approx(9.)
    assert b['absolute_error_80_log'] > a['absolute_error_80_log']


def test_invalid_inputs_are_not_silently_clipped():
    with pytest.raises(ValueError):
        scaled_errors([row(0)],0,.5)
    with pytest.raises(ValueError):
        weighted_quantile([float('nan')],[1.],[.8])
    with pytest.raises(ValueError):
        interval([row(0)],days=211,pace=-1.,exponent=.5)
