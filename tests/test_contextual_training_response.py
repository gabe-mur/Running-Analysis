from copy import deepcopy
from datetime import datetime, timedelta, timezone
import math

import pytest

from run_analysis.training_response import fit_response
from scripts.audit_contextual_training_response import (
    attach_current_midpoint, choose, context, correction, evaluate, fit, predict, sequential_blocks,
)


AT = datetime(2025, 1, 1, tzinfo=timezone.utc)


def examples():
    rows = []
    for i in range(35):
        for h in (14, 28):
            at = AT + timedelta(days=10*i)
            prior = 10 + i/3 + 2*math.cos(i)
            delivered = 12 + i/4 + 3*math.sin(i)
            row = dict(origin=at.isoformat(), origin_id=i, horizon=h, elapsed=h/28,
                outcome_end=(at+timedelta(days=h)).isoformat(), target_id=f'{i}_{h}',
                outcome_ids=[1000+2*i+(h==28)], response_variance=.001,
                start_pace=10., prior_volume=prior, prior_quality=.5,
                delivered_quality=.5+.2*math.sin(i/2), miles=delivered*h/7/100,
                log_miles=math.log1p(delivered*h/7/(50+i*10)),
                consistency=dict(regime='consistent', health_events=[]))
            row = context(row)
            row['change'] = -.01*row['elapsed']-.05*row['miles']+.03*row['usual_miles']
            row['actual'] = row['start_pace']*math.exp(row['change'])
            rows.append(row)
    return rows


def test_simple_fits_reproduce_existing_response_controls():
    rows = examples()
    for key in ('elapsed', 'miles'):
        model = 'calendar' if key == 'elapsed' else key
        assert fit(rows, (key,))['coefficients'] == pytest.approx(fit_response(rows, model)['coefficients'])


def test_context_can_learn_actual_training_dependence_without_starting_pace_reversion():
    rows = examples()
    fitted = fit(rows[:40], ('elapsed', 'miles', 'usual_miles'))
    assert fitted['coefficients'] == pytest.approx(dict(elapsed=-.01, miles=-.05, usual_miles=.03))
    assert predict(fitted, rows[-1]) == pytest.approx(rows[-1]['change'])
    changed = {**rows[-1], 'start_pace': 40., 'start_level': math.log(40)}
    assert predict(fitted, changed) == predict(fitted, rows[-1])
    assert predict(fitted, dict(elapsed=0., miles=0., usual_miles=0.)) == 0.


def test_completed_out_of_sample_errors_learn_signed_correction():
    rows = [{**r, 'log_error': -.02*r['elapsed']} for r in examples()]
    assert correction(rows)['coefficient'] == pytest.approx(.02)
    assert correction(rows[:8]) is None  # only four independent origins


def test_future_labels_cannot_change_earlier_fit_update_or_selection():
    rows = examples()
    changed = deepcopy(rows)
    cutoff = (AT+timedelta(days=240)).isoformat()
    for row in changed:
        if row['outcome_end'] >= cutoff:
            row['change'] += .3
            row['actual'] *= math.exp(.3)
    a, b = evaluate(rows), evaluate(changed)
    key = lambda r: (r['origin'], r['target_id'], r['method'])
    assert {key(r): r['predicted'] for r in a if r['origin'] <= cutoff} == {
        key(r): r['predicted'] for r in b if r['origin'] <= cutoff}
    assert any(r.get('latest_update') for r in a)
    assert any(r['method'] == 'selected' for r in a)
    for r in a:
        assert r['latest_fit'] < r['origin']
        assert not r['latest_update'] or r['latest_update'] < r['origin']
        assert not r.get('latest_selection') or r['latest_selection'] < r['origin']


def test_exact_v3_midpoint_requires_every_policy_and_form_on_same_case():
    results = {}
    for i in range(4):
        predictions = []
        for method, pace in [('pooled/calendar', 10.), ('current/calendar', 8.+i/10),
                ('current/log_miles', 9.+i/10)]:
            predictions.append(dict(origin='2026-01-01', target_id='a', method=method,
                actual=10., predicted=pace, latest_fit='2025-12-31', coefficients={}))
        results[str(i)] = dict(predictions=predictions)
    missing = deepcopy(results)
    missing['3']['predictions'].pop()
    attach_current_midpoint(missing)
    assert all(not any(r['method']=='current/v3_midpoint' for r in d['predictions']) for d in missing.values())
    attach_current_midpoint(results)
    for data in results.values():
        row = data['predictions'][-1]
        assert row['method'] == 'current/v3_midpoint'
        assert row['scenario_count'] == 8
        assert row['predicted'] == pytest.approx((8.+9.3)/2)


def test_selection_common_cases_and_sequential_intervals_are_error_blind():
    prior = []
    for i in range(10):
        for method, error in [('a', .02), ('b', .01)]:
            prior.append(dict(origin=str(i), target_id=str(i), outcome_ids=[i], horizon=14,
                outcome_end=str(i), method=method, log_error=error, predicted=10*math.exp(error), actual=10))
    assert choose(prior, ['a', 'b'])['method'] == 'b'
    first = sequential_blocks(examples())
    altered = [{**r, 'actual': 50., 'change': 20.} for r in examples()]
    assert [r['target_id'] for r in sequential_blocks(altered)] == [r['target_id'] for r in first]
    assert all(a['outcome_end'] <= b['origin'] for a, b in zip(first, first[1:]))
