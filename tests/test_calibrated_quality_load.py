from dataclasses import replace
from datetime import datetime,timedelta,timezone

import pytest

from run_analysis.adaptive.calibrated_quality_load import (
    PhaseCalibrationHistory,PhaseLoadRatio,calibration_phases,forecast_calibrated)
from run_analysis.signal_structure import SignalSpan

AT=datetime(2026,10,8,tzinfo=timezone.utc)
LAYOUT=(('warmup',0,600),('work',600,900),('recovery',900,1020),('work',1020,1320),('cooldown',1320,1920))
MODEL=dict(intercept=.2,slope=.2,tau_seconds=60.,speed_domain=(0.,5.))


def forecast(speed=3.,history=()):
    return forecast_calibrated(LAYOUT,tuple((b-a,speed if k=='work' else 2.) for k,a,b in LAYOUT),
        MODEL,history,as_of=AT,accounting_key='units',initial_reserve=.45)


def donor(ratio=1.):
    return PhaseCalibrationHistory('earlier',AT-timedelta(days=1),AT-timedelta(hours=23),
        AT-timedelta(hours=22),'units',(PhaseLoadRatio('work',300,600,300,ratio),))


def test_intensity_and_recovery_state_change_with_requested_pace():
    slow,fast=forecast(2.5),forecast(3.5)
    assert slow['status']==fast['status']=='predicted'
    assert fast['predicted_load']>slow['predicted_load']
    assert fast['phases'][2]['load']>slow['phases'][2]['load']
    assert slow['phases'][0]['load']==fast['phases'][0]['load']
    assert all(p['identity_correction'] for p in slow['phases'])


def test_calibration_changes_work_without_fabricating_nonwork_or_zero_load():
    a,b=forecast(),forecast(history=(donor(2.),))
    assert b['phases'][1]['load']==pytest.approx(a['phases'][1]['load']*2)
    assert b['phases'][0]['load']==a['phases'][0]['load']
    assert not b['phases'][1]['identity_correction']


def test_future_unavailable_stale_and_incompatible_donors_cannot_affect_query():
    base=forecast()
    for d in (replace(donor(),available_at=AT+timedelta(seconds=1)),
              replace(donor(),accounting_key='other'),
              replace(donor(),start=AT-timedelta(days=40),end=AT-timedelta(days=39))):
        assert forecast(history=(d,))['predicted_load']==base['predicted_load']
    with pytest.raises(ValueError,match='Duplicate'):
        forecast(history=(donor(),donor()))


def test_source_ratios_use_identical_hr_masks_not_missing_time_as_zero():
    p=(SignalSpan(0,600,.6,.6),)
    actual=(SignalSpan(0,300,.6,.6),)
    rows=calibration_phases((('work',0,600),),p,actual)
    assert rows[0].known_seconds==300 and rows[0].load_ratio==pytest.approx(1.)
    assert calibration_phases((('work',0,600),),p,())[0].load_ratio is None


def test_invalid_hr_not_clipped_and_layout_mismatch_rejected():
    p=tuple((b-a,6.) for k,a,b in LAYOUT)
    r=forecast_calibrated(LAYOUT,p,MODEL,(),as_of=AT,accounting_key='units',initial_reserve=.45)
    assert r['status']=='predicted_hr_outside_reference' and r['predicted_load'] is None
    with pytest.raises(ValueError,match='exact phase layout'):
        forecast_calibrated(LAYOUT,p[:-1],MODEL,(),as_of=AT,accounting_key='units',initial_reserve=.45)


def test_ratio_history_cannot_accept_observed_cardio_load_units():
    from run_analysis.adaptive.direct_quality_load import PhaseDose
    with pytest.raises(ValueError,match='Immutable phase calibration'):
        replace(donor(),phases=(PhaseDose('work',300,600,300,10.),))
