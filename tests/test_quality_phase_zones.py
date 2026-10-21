from dataclasses import replace

import pytest

from scripts.audit_quality_phase_zones import band_fraction, zone_profile, exposure_forecast
from run_analysis.adaptive.direct_quality_load import PhaseDose
from run_analysis.signal_structure import SignalSpan
from run_analysis.training_signals import HeartRateReference
from test_direct_quality_load import AT, history


def test_exact_zone_accounting_has_no_integer_gap_and_uses_observed_wall_time():
    config=dict(zones={f'z{i}': [50+(i-1)*10,59+(i-1)*10] for i in range(1,6)})
    reference=HeartRateReference(40,100,'test')
    spans=(SignalSpan(0,60,0,1),)
    profile=zone_profile(spans,0,60,config,reference)
    assert sum(profile['zone_seconds'].values())==pytest.approx(60)
    assert profile['zone_seconds']['below_z1']==pytest.approx(10)
    assert profile['zone_seconds']['z1']==pytest.approx(10)
    assert profile['mean_bpm']==pytest.approx(70)


def test_band_overlap_keeps_startup_and_gap_explicit():
    spans=(SignalSpan(0,60,.4,.8),)
    assert band_fraction(spans,0,60,(.6,.9),(.6,.9))==pytest.approx(.5)
    assert band_fraction((SignalSpan(30,60,.7,.8),),0,60,(.6,.9),(.6,.9))==1
    assert band_fraction((),0,60,(.6,.9),(.6,.9)) is None
    # Same ramp matches throughout; a flat finishing band only matches half.
    assert band_fraction(spans,0,60,(.39,.41),(.79,.81))==pytest.approx(1)
    assert band_fraction(spans,0,60,(.6,.9),(.6,.9))==pytest.approx(.5)


def command(low,high):
    return [dict(kind='work',start=dict(low=low,high=high),finish=dict(low=low,high=high))]


def test_overlap_weights_whole_phase_cost_not_only_high_hr_samples():
    past=history(phases=(PhaseDose('work',60,None,60,2),))
    profiles={past.key:[dict(start=0,end=60,spans=(SignalSpan(0,60,.4,.8),))]}
    f=exposure_forecast((('work',0,60),),[past],profiles,command(.6,.9),as_of=AT,accounting_key='units')
    assert f['predicted_load']==pytest.approx(2)
    assert f['phases'][0]['exposure_matching_sources'][0]['match_fractions']==pytest.approx([.5])


def test_higher_observed_exposure_selects_higher_rate_without_family_constants():
    low=history('low',phases=(PhaseDose('work',60,None,60,1),))
    high=history('high',phases=(PhaseDose('work',60,None,60,3),))
    profiles={r.key:[dict(start=0,end=60,spans=(SignalSpan(0,60,value,value),))]
              for r,value in ((low,.55),(high,.85))}
    def predict(band):
        return exposure_forecast((('work',0,60),),[low,high],profiles,command(*band),as_of=AT,accounting_key='units')
    assert predict((.5,.6))['predicted_load']==pytest.approx(1)
    assert predict((.8,.9))['predicted_load']==pytest.approx(3)
    unsupported=predict((.95,1))
    assert unsupported['predicted_load']==pytest.approx(2)
    assert unsupported['phases'][0]['exposure_match_fallback']


def test_future_source_cannot_supply_exposure_match():
    past=history('past',phases=(PhaseDose('work',60,None,60,1),))
    future=replace(past,key='future',available_at=AT.replace(year=2027))
    profiles={'past':[dict(start=0,end=60,spans=(SignalSpan(0,60,.5,.5),))]}
    f=exposure_forecast((('work',0,60),),[past,future],profiles,command(.5,.6),as_of=AT,accounting_key='units')
    assert f['predicted_load']==pytest.approx(1)
    assert f['source_keys']==['past']
