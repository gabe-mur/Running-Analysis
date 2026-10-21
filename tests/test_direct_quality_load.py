from dataclasses import replace
from datetime import datetime,timedelta,timezone

import pytest

from run_analysis.signal_structure import SignalSpan
from scripts.direct_quality_load import PhaseDose, PhaseHistory, phase_observations, forecast_direct, score_direct


AT = datetime(2026,9,26,tzinfo=timezone.utc)
LAYOUT = (('nonwork',0.,60.),('work',60.,120.),('nonwork',120.,180.),('work',180.,240.),('nonwork',240.,300.))


def history(key='past',phases=None,day=-1):
    phases = phase_observations(LAYOUT,(SignalSpan(0.,300.,.6,.6),)) if phases is None else phases
    return PhaseHistory(key,AT+timedelta(days=day),AT+timedelta(days=day,minutes=5),
        AT+timedelta(days=day,minutes=5),'units',phases)


def predict(records,layout=LAYOUT,**kwargs):
    return forecast_direct(layout,records,as_of=AT,accounting_key='units',**kwargs)


def test_direct_load_is_additive_and_not_mean_hr_shortcut():
    spans = (SignalSpan(0.,150.,.3,.3),SignalSpan(150.,300.,.9,.9))
    observations = phase_observations(LAYOUT,spans)
    from run_analysis.training_signals import integrate_cardio_trace
    assert sum(p.load for p in observations) == pytest.approx(integrate_cardio_trace(spans,300.,compute_density=False)['observed_load'])
    assert sum(p.load for p in observations) > 5.  # g(.6) * 5 min is not g(HR(t)) integrated.
    result = predict([history()])
    assert result['predicted_load'] == pytest.approx(5.)
    assert score_direct(result,(SignalSpan(0.,300.,.6,.6),))['fraction_error'] == pytest.approx(0.)


def test_missing_hr_is_unknown_and_scoring_matches_known_time():
    observations = phase_observations(LAYOUT,(SignalSpan(0.,30.,.6,.6),))
    assert observations[0].known_seconds == 30.
    assert observations[0].load == pytest.approx(.5)
    assert observations[1].load is None
    assert predict([history(phases=observations)])['status'] != 'predicted'
    result = score_direct(predict([history()]),(SignalSpan(0.,30.,.6,.6),))
    assert result['seconds'] == 30.
    assert result['predicted_load'] == pytest.approx(.5)
    assert result['observed_load'] == pytest.approx(.5)


def test_novel_reps_do_not_require_same_workout_and_missing_recovery_falls_back():
    prior = (PhaseDose('warmup',300.,None,300.,5.),PhaseDose('work',900.,300.,900.,30.),PhaseDose('cooldown',300.,900.,300.,5.))
    result = predict([history(phases=prior)])
    assert result['status'] == 'predicted'
    assert result['phases'][2]['fallback_pooled_nonwork']
    assert result['phases'][1]['rate'] == pytest.approx(2/60)
    assert not result['phases'][1]['fallback_pooled_nonwork']


def test_no_future_old_late_or_other_accounting_can_change_forecast():
    good = history()
    rejected = [history('future',day=1),history('old',day=-31),
        replace(good,key='late',available_at=AT+timedelta(seconds=1)),replace(good,key='other',accounting_key='other')]
    assert predict([good]) == predict([good,*rejected])
    assert predict([])['predicted_load'] is None
    assert predict([good],layout=())['status'] == 'unresolved_work_alignment'
    with pytest.raises(ValueError):
        predict([good,good])


def test_one_session_vote_not_one_vote_per_rep_or_trackpoint():
    a = PhaseDose('work',60.,60.,60.,2.)
    b = PhaseDose('work',60.,60.,60.,4.)
    layout = (('work',0.,60.),)
    once = predict([history(phases=(a,)),history('other',phases=(b,))],layout)
    many = predict([history(phases=(a,)*20),history('other',phases=(b,))],layout)
    assert once['predicted_load'] == pytest.approx(many['predicted_load'])
    assert once['predicted_load'] == pytest.approx(3.)


def test_spacing_conditions_work_without_new_coefficients():
    fast = PhaseDose('work',60.,60.,60.,3.)
    slow = PhaseDose('work',60.,300.,60.,1.)
    prior = [history(phases=(fast,slow,PhaseDose('warmup',60.,None,60.,1.)))]
    short = predict(prior,(('nonwork',0.,60.),('work',60.,120.)))
    long = predict(prior,(('nonwork',0.,300.),('work',300.,360.)))
    assert short['phases'][1]['rate'] > long['phases'][1]['rate']
    assert predict(prior,(('nonwork',0.,60.),('work',60.,120.)),conditioned=False)['phases'][1]['rate'] == pytest.approx(2/60)


def test_appending_cooldown_does_not_change_earlier_predicted_work():
    shorter = LAYOUT[:-1]
    a,b = predict([history()],shorter),predict([history()])
    assert a['phases'] == b['phases'][:-1]
    assert b['predicted_load']-a['predicted_load'] == pytest.approx(1.)


@pytest.mark.parametrize('changes',[{'seconds':-1.},{'known_seconds':61.},{'known_seconds':0.,'load':0.}, {'load':None},{'load':float('nan')}])
def test_invalid_evidence_rejected(changes):
    with pytest.raises(ValueError):
        replace(PhaseDose('work',60.,None,60.,1.),**changes)


def test_invalid_layout_and_unavailable_source_rejected():
    with pytest.raises(ValueError):
        predict([history()],(('work',10.,60.),))
    with pytest.raises(ValueError):
        replace(history(),available_at=AT-timedelta(days=2))
