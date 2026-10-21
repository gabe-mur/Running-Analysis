import pytest
from datetime import datetime, timedelta, timezone

from run_analysis.signal_structure import SignalSpan
from scripts.audit_quality_response_repair import (observed_speed_control,
    detailed_layout, measured_phase_speeds, phase_pace_program)


def test_actual_speed_control_preserves_time_and_only_imputes_unknown_speed():
    rows,missing = observed_speed_control([(10.,2.),(10.,4.)],
        (SignalSpan(2.,8.,3.,3.),SignalSpan(12.,20.,5.,5.)))
    assert rows == [(2.,2.),(6.,3.),(2.,2.),(2.,4.),(8.,5.)]
    assert missing == 6.
    assert sum(t for t,_ in rows) == 20.


def test_actual_speed_control_does_not_introduce_hr_or_reset_boundaries():
    from run_analysis.hr_response import predict_speed_program
    program,missing = observed_speed_control([(10.,2.),(10.,4.)],
        (SignalSpan(0.,20.,3.,3.),))
    assert missing == 0
    model = dict(intercept=.2,slope=.1,tau_seconds=45.,speed_domain=(0,6))
    prediction,diagnostic = predict_speed_program(program,model,initial_reserve=.45)
    assert prediction[0].first == .45
    assert diagnostic['segment_initializations'] == 1
    assert all(a.last == b.first for a,b in zip(prediction,prediction[1:]))


def test_control_rejects_overlapping_or_outside_speed():
    for spans in ((SignalSpan(0.,21.,3.,3.),),
                  (SignalSpan(0.,10.,3.,3.),SignalSpan(9.,20.,3.,3.))):
        with pytest.raises(ValueError):
            observed_speed_control([(20.,2.)],spans)


LAYOUT = (('nonwork',0.,10.),('work',10.,20.),('nonwork',20.,30.),
          ('work',30.,40.),('nonwork',40.,50.))
AT = datetime(2026,9,26,tzinfo=timezone.utc)


def phase_donor(speed=2.):
    return dict(start=AT-timedelta(days=1),end=AT-timedelta(hours=23),available_at=AT-timedelta(hours=23),
        phases={k:dict(speed=v,known_seconds=10.) for k,v in
            [('warmup',speed),('recovery',0.),('cooldown',speed+1)]})


def test_phase_pace_repair_changes_no_work_or_timing_and_preserves_standing():
    assert [k for k,_,_ in detailed_layout(LAYOUT)] == ['warmup','work','recovery','work','cooldown']
    original = [(10.,2.5),(10.,4.),(10.,2.5),(10.,3.9),(10.,2.5)]
    program,sources = phase_pace_program(LAYOUT,original,['old'],{'old':phase_donor()},AT)
    assert program == [(10.,2.),(10.,4.),(10.,0.),(10.,3.9),(10.,3.)]
    assert all(not r['fallback_pooled_nonwork'] for r in sources.values())


def test_phase_speed_is_one_session_vote_and_explicit_fallback_not_training_ban():
    original = [(10.,2.5)]*5
    a,b = phase_donor(2.),phase_donor(4.)
    for d in (a,b):
        d['phases']['recovery']['speed'] = None
    a['phases']['warmup']['known_seconds'] = 10000.
    program,sources = phase_pace_program(LAYOUT,original,['a','b'],{'a':a,'b':b},AT)
    assert program[0][1] == 3.  # Equal sessions, not trackpoint/time votes.
    assert program[2][1] == 2.5
    assert sources['recovery']['fallback_pooled_nonwork']


@pytest.mark.parametrize('change',[
    {'available_at':AT+timedelta(seconds=1)}, {'end':AT}, {'start':AT-timedelta(days=31)}])
def test_phase_pace_repair_rejects_unavailable_future_and_old_donors(change):
    with pytest.raises(ValueError):
        phase_pace_program(LAYOUT,[(10.,2.)]*5,['a'],{'a':dict(phase_donor(),**change)},AT)


def test_measurement_tracks_unknown_speed_without_inventing_zero():
    data = measured_phase_speeds(LAYOUT,(SignalSpan(0.,5.,2.,2.),SignalSpan(20.,30.,0.,0.)))
    assert data['warmup'] == dict(known_seconds=5.,speed=2.)
    assert data['recovery']['speed'] == 0.
    assert data['cooldown']['speed'] is None
