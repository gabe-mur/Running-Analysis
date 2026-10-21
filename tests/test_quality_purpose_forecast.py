from dataclasses import replace
from math import fsum

import pytest

from run_analysis.adaptive.quality_catalogue_forecast import forecast_prescription_point
from run_analysis.adaptive.quality_dose_policy import purpose_contract
from run_analysis.adaptive.quality_purpose_forecast import forecast_purposes
from run_analysis.adaptive.work_exposure_matching import PhaseTrace
from run_analysis.signal_structure import SignalSpan
from test_quality_catalogue_forecast import CONFIG, REF, evidence
from test_quality_block_contract import block, prescription, pace
from test_quality_dose_policy import stage, AT


def inputs(p):
    h,t=evidence()
    contract=purpose_contract(stage(),config=CONFIG,issued_at=AT,easy_pace=pace(.4),faster_pace=pace(.28))
    f=forecast_prescription_point(p,h,t,CONFIG,reference=REF)
    return h,t,contract,f


def test_purpose_uses_same_sources_no_hr_inverse_or_equal_family_credit():
    p=prescription(block('work',300,cues=('Z4',),speed_pace=.28))
    h,t,c,f=inputs(p)
    result=forecast_purposes(p,f,h,t,CONFIG,contract=c)
    assert result['credits']['sustained'].low>0 and result['credits']['faster'].low>0
    assert result['opportunity'].low==max(q.low for q in result['credits'].values())
    assert set(x['key'] for x in result['reports']['sustained']['contributors'])<=set(f['source_keys'])
    assert not result['physiology_equivalence_established']


def test_short_efforts_cannot_be_summed_into_one_sustained_bout():
    p=prescription(block('work',60,cues=('Z4',),speed_pace=.28),block('recovery',90),
        block('work',60,cues=('Z4',),speed_pace=.28))
    h,t,c,f=inputs(p)
    result=forecast_purposes(p,f,h,t,CONFIG,contract=c)
    assert result['credits']['sustained'].high==0
    assert result['credits']['faster'].low>0


def test_missing_hr_support_remains_unknown_not_fabricated_sustained_credit():
    p=prescription(block('work',300,cues=('Z5',),speed_pace=.28))
    h,t,c,f=inputs(p)
    result=forecast_purposes(p,f,h,t,CONFIG,contract=c)
    assert (result['credits']['sustained'].low,result['credits']['sustained'].high)==(0,1)
    assert result['credits']['faster'].low>0  # Explicit command-speed scenario, not achieved HR.
    altered=dict(f,phases=[])
    with pytest.raises(ValueError,match='exact complete clock'):
        forecast_purposes(p,altered,h,t,CONFIG,contract=c)
