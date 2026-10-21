from copy import deepcopy
from datetime import datetime,timedelta,timezone

import pytest

from scripts.audit_progress_responsiveness import inject,inject_windows,speed_factor,estimate,replay_case,response,summarize
from run_analysis.aerobic_progress import build_aerobic_progress

ORIGIN=datetime(2026,1,1,tzinfo=timezone.utc)


def rows():
    return [dict(activity_id=i,start_time=(ORIGIN+timedelta(days=i*3)).isoformat(),
        pace_min_mile=10.,uncertainty_95_min_mile=.5,trend_weight=1.,included_in_trend=True,
        shared_log_speed_gradient=[0.],distance_miles=3.) for i in range(40)]


@pytest.mark.parametrize('gain',[.03,.05,.10,-.05])
def test_known_sustained_speed_change_survives_exact_production_smoothing(gain):
    original=rows();before=deepcopy(original);onset=ORIGIN+timedelta(days=60)
    changed=inject(original,onset,gain)
    end=ORIGIN+timedelta(days=100)
    expected=build_aerobic_progress(changed,as_of=end,covariance=[[0]],history_days=1)
    actual=estimate(changed,datetime.fromisoformat(expected['last_observation']))
    assert actual['pace_min_mile']==pytest.approx(expected['current']['pace_min_mile'])
    assert response(estimate(original,end),estimate(changed,end))==pytest.approx(gain*100)
    assert original==before


def test_paired_response_does_not_count_the_original_trend_as_injected_signal():
    original=rows()
    for i,r in enumerate(original):r['pace_min_mile']-=i*.025
    case=replay_case(original,ORIGIN+timedelta(days=60),.05)
    assert case['settled_response_percent']==pytest.approx(5,abs=.05)
    assert 0<case['half_response']['days']<=28
    assert len(case['points'])>10


def test_gradual_injection_has_no_future_effect_and_window_loading_is_copied():
    onset=ORIGIN+timedelta(days=50)
    assert speed_factor(onset-timedelta(seconds=1),onset,.1,28)==1
    assert speed_factor(onset,onset,.1,28)==1
    assert speed_factor(onset+timedelta(days=28),onset,.1,28)==pytest.approx(1.1)
    window=dict(start_time_utc=(onset+timedelta(days=14)).isoformat(),moving_pace_min_mile=10.,average_hr_bpm=148)
    before=deepcopy(window);result=inject_windows([window],onset,.1,28)[0]
    assert result['moving_pace_min_mile']==pytest.approx(10/1.1**.5)
    assert result['average_hr_bpm']==148 and window==before


def test_excluded_workouts_cannot_create_response_and_diagnostic_variants_are_explicit():
    original=rows();original[-1].update(included_in_trend=False,trend_weight=0)
    changed=deepcopy(original);changed[-1]['pace_min_mile']=2.
    end=ORIGIN+timedelta(days=118)
    for variant in ('deployed','no_huber','equal_uncertainty','equal_context','equal_weights_no_huber'):
        assert estimate(original,end,variant=variant)==estimate(changed,end,variant=variant)
    assert summarize([])['half_response_days']['median'] is None
