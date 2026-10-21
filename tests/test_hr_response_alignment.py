from copy import deepcopy
from datetime import datetime,timedelta,timezone
from types import SimpleNamespace

import numpy as np
import pytest

from run_analysis.hr_response_alignment import TAUS,response_step,context_for_windows,training_losses,held_out_tau,corrected_windows
from run_analysis.pace_comparability import fit_comparison_calibration
from tests.test_pace_comparability import config,row


def synthetic(config,*,gain=1.,tau=60.,steady=False,steady_speed=2.7):
    start=datetime(2026,1,1,tzinfo=timezone.utc);intervals=[];windows=[]
    state=2.7
    for second in range(0,1800,10):
        speed=steady_speed if steady else (3.9 if (second//90)%2==0 else 2.2)
        before=state;average,state=response_step(state,speed,10,tau)
        a=SimpleNamespace(timestamp_utc=start+timedelta(seconds=second),heart_rate_bpm=config['resting_hr']+34*before)
        b=SimpleNamespace(timestamp_utc=start+timedelta(seconds=second+10),heart_rate_bpm=config['resting_hr']+34*state)
        intervals.append(SimpleNamespace(start=a,end=b,elapsed_s=10.,moving_time_s=10.,distance_m=speed*gain*10,
                                         elevation_delta_m=0.))
    for second in range(300,1681,60):
        chosen=intervals[second//10:(second+120)//10]
        distance=sum(i.distance_m for i in chosen)
        windows.append(row(config,window_start_time_utc=(start+timedelta(seconds=second)).isoformat(),
            window_end_time_utc=(start+timedelta(seconds=second+120)).isoformat(),
            elapsed_time_s=120.,stopped_time_s=0.,moving_minutes_into_run=(second+60)/60,
            moving_pace_min_mile=120/distance*1609.344/60,
            average_hr_bpm=np.mean([(i.start.heart_rate_bpm+i.end.heart_rate_bpm)/2 for i in chosen])))
    context_for_windows(intervals,windows,config)
    return intervals,windows


def test_exact_response_integrates_constant_effort_and_stop():
    mean,end=response_step(3.,0.,60.,60.)
    assert end==pytest.approx(3/np.e)
    assert mean==pytest.approx(3*(1-1/np.e))
    assert response_step(3.,3.,60.,60.)==(3.,3.)
    assert response_step(3.,1.,60.,0.)==(1.,1.)


def test_true_delay_is_identified_without_run_level_fitness_target(config):
    _,rows=synthetic(config)
    cal=fit_comparison_calibration([],config);cal['parameters']=[1.,0.,0.]
    loss=training_losses(rows[0]['_hr_response_training'],config,cal)
    assert TAUS[int(np.argmin(loss))]==60.
    assert min(loss)<loss[0]/20


def test_held_out_run_cannot_pick_its_own_lag_and_null_can_win():
    losses={1:list(range(len(TAUS))),2:list(range(len(TAUS))),3:list(range(len(TAUS))),4:[100.]*(len(TAUS)-1)+[0.]}
    assert held_out_tau(losses,4)==0
    losses[4]=[0.]*(len(TAUS)-1)+[10000.]
    assert held_out_tau(losses,4)==0
    assert held_out_tau({1:[1]*len(TAUS)},99)==0


def test_uniform_real_improvement_survives_alignment(config):
    _,old=synthetic(config);_,new=synthetic(config,gain=1.03)
    for tau in range(len(TAUS)):
        before=corrected_windows(old,tau);after=corrected_windows(new,tau)
        np.testing.assert_allclose([a['moving_pace_min_mile'] for a in after],
            [b['moving_pace_min_mile']/1.03 for b in before],rtol=1e-12)


def test_steady_fast_running_is_not_penalized_and_inputs_not_rewritten(config):
    _,rows=synthetic(config,steady=True);before=deepcopy(rows)
    for tau in range(len(TAUS)):
        changed=corrected_windows(rows,tau)
        np.testing.assert_allclose([r['moving_pace_min_mile'] for r in changed],[r['moving_pace_min_mile'] for r in rows])
    assert rows==before


def test_higher_delayed_hr_is_available_before_fitness_hr_ceiling(config):
    _,rows=synthetic(config,steady=True,steady_speed=3.8)
    # Samples above the scoring ceiling still teach the delayed cost. Do not
    # infer their absence from the already-selected submaximal scoring windows.
    samples=rows[0]['_hr_response_training']
    assert any(s['hr']>float(config['model']['maximum_reliable_hr_bpm']) for s in samples)


def test_unknown_response_window_is_missing_not_neutral_precision(config):
    _,rows=synthetic(config)
    rows[0]['hr_response_context']['speed_ratios'][2]=None
    assert len(corrected_windows(rows,2))==len(rows)-1
    assert corrected_windows([{}],2)==[]


@pytest.mark.parametrize('seconds,tau',[(0,1),(1,-1),(float('nan'),1)])
def test_invalid_response_intervals_rejected(seconds,tau):
    with pytest.raises(ValueError):response_step(1,1,seconds,tau)
