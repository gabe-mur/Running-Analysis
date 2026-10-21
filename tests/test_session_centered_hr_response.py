from dataclasses import replace
from datetime import timedelta
from math import exp

import pytest

from run_analysis.hr_response import ResponseInterval, fit_response
from scripts.session_centered_hr_response import fit_session_centered
from test_hr_response import NOW, GRID, evidence


def fixture(key,intercept,offset):
    rows,hr,at = [],.45,0.
    for speed in (1.,2.,1.5,2.5):
        speed += offset
        for _ in range(36):
            equilibrium = intercept+.1*speed
            last = equilibrium+(hr-equilibrium)*exp(-5/60.)
            rows.append(ResponseInterval(at,at+5,speed,hr,last))
            hr,at = last,at+5
    return evidence(tuple(rows),key=key)


def fit(history):
    return fit_session_centered(history,as_of=NOW,accounting_key='units',taus=GRID)


def test_within_session_response_recovers_known_slope_despite_baseline_confounding():
    history = [fixture('slow-high-hr',.4,0.),fixture('fast-low-hr',.1,2.)]
    result = fit(history)['dynamic']
    assert result['slope'] == pytest.approx(.1,abs=1e-8)
    assert result['intercept'] == pytest.approx(.25,abs=1e-8)
    assert result['tau_seconds'] == 60.
    assert result['session_intercepts'] == pytest.approx({'slow-high-hr':.4,'fast-low-hr':.1})
    pooled = fit_response(history,as_of=NOW,accounting_key='units',taus=GRID)['dynamic']
    assert pooled is None or abs(pooled['slope']-.1) > .01


def test_one_session_reduces_to_original_fit_and_has_no_extra_forecast_information():
    history = [fixture('one',.25,0.)]
    a = fit(history)['dynamic']
    b = fit_response(history,as_of=NOW,accounting_key='units',taus=GRID)['dynamic']
    for k in ('slope','intercept','tau_seconds'):
        assert a[k] == pytest.approx(b[k])


def test_future_old_late_ineligible_and_wrong_units_do_not_change_fit():
    good = fixture('good',.25,0.)
    rejected = [replace(good,key='future',end=NOW+timedelta(days=1),available_at=NOW+timedelta(days=1)),
        replace(good,key='old',end=NOW-timedelta(days=31)),
        replace(good,key='late',available_at=NOW+timedelta(seconds=1)),
        replace(good,key='ill',eligible=False),replace(good,key='wrong',accounting_key='other')]
    assert fit([good]) == fit([good,*rejected])


def test_insufficient_within_session_variation_cannot_identify_a_slope():
    rows = tuple(ResponseInterval(i,i+1,2.,.5,.5) for i in range(20))
    assert fit([evidence(rows)])['dynamic'] is None
    assert fit([])['dynamic'] is None
    with pytest.raises(ValueError):
        fit([fixture('duplicate',.2,0.),fixture('duplicate',.3,0.)])
