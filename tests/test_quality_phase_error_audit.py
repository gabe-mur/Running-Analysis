import pytest

from scripts.audit_direct_quality_load import phase_errors
from run_analysis.adaptive.direct_quality_load import PhaseDose


def test_known_masks_and_errors_do_not_hide_cancelling_recovery_cooldown():
    prediction=dict(status='predicted',phases=[dict(kind=k,start=i*60,end=(i+1)*60,rate=1.)
        for i,k in enumerate(('warmup','work','recovery','cooldown'))])
    actual=(PhaseDose('warmup',60,None,60,60),PhaseDose('work',60,60,60,60),
        PhaseDose('recovery',60,60,30,15),PhaseDose('cooldown',60,60,60,75))
    result=phase_errors(prediction,actual)
    assert result['recovery']['fraction_error']==1.
    assert result['recovery']['signed_load_error']==15.
    assert result['cooldown']['signed_load_error']==-15.
    assert result['recovery']['known_seconds']==30


def test_no_hr_and_unsupported_prediction_are_not_zero_error():
    prediction=dict(status='predicted',phases=[dict(kind='work',start=0,end=60,rate=1.)])
    result=phase_errors(prediction,(PhaseDose('work',60,None,0,None),))
    assert result['work']['observed_load'] is None and result['work']['fraction_error'] is None
    assert phase_errors(dict(status='no_recent_phase_evidence'),()) is None


def test_phase_layout_mismatch_is_rejected():
    prediction=dict(status='predicted',phases=[dict(kind='work',start=0,end=60,rate=1.)])
    with pytest.raises(ValueError,match='exact forecast layout'):
        phase_errors(prediction,(PhaseDose('work',50,None,50,50),))
