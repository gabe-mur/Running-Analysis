from dataclasses import replace
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from run_analysis.hr_response import ResponseInterval, predict_response, sufficient_statistics
from scripts.elapsed_hr_response import (
    VERSION, ElapsedEvidence, elapsed_statistics, fit_elapsed_response, predict_elapsed_response,
)

NOW = datetime(2026, 10, 8, tzinfo=timezone.utc)
GRID = (10., 30., 60., 120.)
MODEL = dict(version=VERSION, intercept=.15, slope=.12, tau_seconds=60.,
    elapsed_coefficient_per_minute=.003, speed_domain=(0., 8.), elapsed_domain_minutes=(0., 60.))


def synthetic(step=5., model=MODEL):
    rows, state, t = [], .3, 0.
    for v in (2., 4., 1., 3., 2.5):
        for _ in range(round(180/step)):
            row = ResponseInterval(t, t+step, v, state, state)
            pred, _ = predict_elapsed_response((row,), model)
            last = pred[0].last
            rows.append(replace(row, last=last))
            state, t = last, t+step
    return tuple(rows)


def evidence(rows=None, key='past', **changes):
    rows = synthetic() if rows is None else rows
    return replace(ElapsedEvidence(key, NOW-timedelta(days=1), NOW-timedelta(days=1), 'units',
        elapsed_statistics(rows, GRID), (min(r.speed for r in rows),max(r.speed for r in rows)),
        (min(r.start for r in rows)/60, max(r.end for r in rows)/60), taus=GRID), **changes)


def fit(history):
    return fit_elapsed_response(history, as_of=NOW, accounting_key='units', taus=GRID)


def test_recovers_known_parameters_with_no_family_or_phase_inputs():
    m = fit([evidence()])['dynamic']
    assert m['tau_seconds']==60
    for k in ('intercept','slope','elapsed_coefficient_per_minute'):
        assert m[k]==pytest.approx(MODEL[k], abs=1e-8)
    assert m['training_rmse_reserve']<1e-7


def test_zero_elapsed_term_recovers_existing_model_and_statistic_submatrix():
    rows = synthetic(model=dict(MODEL, elapsed_coefficient_per_minute=0))
    a = elapsed_statistics(rows, GRID)
    b = sufficient_statistics(rows, GRID)[1:]
    assert a[:, (0, 1, 3)][:, :, (0, 1, 3)]==pytest.approx(b, abs=1e-12)
    old, _ = predict_response(rows, MODEL)
    new, _ = predict_elapsed_response(rows, dict(MODEL, elapsed_coefficient_per_minute=0))
    assert np.array([(s.first,s.last) for s in new])==pytest.approx(
        np.array([(s.first,s.last) for s in old]), abs=1e-12)


def test_future_late_stale_excluded_and_other_units_cannot_change_fit():
    past = evidence()
    extras = [evidence(key='future', end=NOW+timedelta(days=1), available_at=NOW+timedelta(days=1)),
        evidence(key='late', available_at=NOW+timedelta(days=1)),
        evidence(key='old', end=NOW-timedelta(days=90)), evidence(key='ill', eligible=False),
        evidence(key='other', accounting_key='other')]
    assert fit([past])==fit([past,*extras])


def test_sampling_density_and_equal_session_ownership():
    a, b = fit([evidence(synthetic(step=1))])['dynamic'], fit([evidence(synthetic(step=10))])['dynamic']
    for k in ('tau_seconds','slope','intercept','elapsed_coefficient_per_minute'):
        assert a[k]==pytest.approx(b[k], abs=1e-8)
    assert fit([evidence(),evidence(key='second')])['dynamic']==fit([evidence()])['dynamic']


def test_no_measurement_resets_at_phase_boundaries_or_teacher_forcing():
    rows = synthetic()
    changed = tuple(replace(r, first=r.first if i==0 else .1, last=.1) for i,r in enumerate(rows))
    assert predict_elapsed_response(rows,MODEL)==predict_elapsed_response(changed,MODEL)


def test_gaps_reset_state_but_not_session_elapsed_time_and_are_not_filled():
    rows = (ResponseInterval(0, 5, 2, .3, .3),ResponseInterval(600, 605, 2, .3, .3))
    pred, d = predict_elapsed_response(rows, MODEL)
    assert d['segment_initializations']==2
    assert sum(s.end-s.start for s in pred)==10
    assert pred[1].first==.3
    assert pred[1].last>pred[0].last  # Same speed/state, later elapsed time.


def test_domain_extrapolation_invalid_hr_and_missing_fit_are_explicit():
    _, d = predict_elapsed_response((ResponseInterval(3590, 3610, 9, .9, .9),),
        dict(MODEL, elapsed_coefficient_per_minute=.1))
    assert d['elapsed_extrapolation_seconds']==10
    assert d['speed_extrapolation_seconds']==20
    assert d['hr_outside_reference']
    assert fit([])['dynamic'] is None
    constant = (ResponseInterval(0, 60, 2, .3, .5),)
    assert fit([evidence(constant)])['dynamic'] is None


def test_causal_evidence_immutable_incompatible_and_duplicate_rejected():
    e = evidence()
    with pytest.raises(ValueError):
        e.statistics[:] = 0
    with pytest.raises(ValueError,match='Duplicate'):
        fit([e,e])
    with pytest.raises(ValueError,match='grid'):
        fit_elapsed_response([e],as_of=NOW,accounting_key='units')
    with pytest.raises(ValueError,match='Causal'):
        evidence(available_at=NOW-timedelta(days=2))
    with pytest.raises(ValueError,match='experimental'):
        predict_elapsed_response(synthetic(),dict(MODEL,version='old'))
