from copy import deepcopy
from datetime import datetime, timedelta, timezone
import math

import numpy as np
import pytest

from run_analysis.training_response import POLICIES
from scripts.audit_pace_smoothing_forecast import (
    common_targets, comparisons, evaluate, fit_at, forecast, prefix_levels, rebuild,
)

AT = datetime(2025, 1, 1, tzinfo=timezone.utc)


def snapshot(n=60, step=False):
    rows = [dict(activity_id=i, start_time=(AT+timedelta(days=i*3)).isoformat(),
        pace_min_mile=10*math.exp(-.03 if step and i>=n-8 else (0. if step else -.001*i)))
        for i in range(n)]
    return dict(rows=rows, measurement=[.001]*n, gradients=[[.1] for _ in rows], covariance=[[.01]])


def examples(n=40):
    rows = []
    for i in range(n):
        at = AT+timedelta(days=i*7)
        change = -.01
        rows.append(dict(origin=at.isoformat(), origin_id=i, target_id=f'{i+1},{i+2}',
            outcome_ids=[i+1,i+2], outcome_end=(at+timedelta(days=14)).isoformat(),
            horizon=14, elapsed=.5, log_miles=.1, response_variance=.001,
            start_level=math.log(10), start_pace=10., change=change, actual=10*math.exp(change),
            training_by_outcome=[dict(weight=.5),dict(weight=.5)],
            consistency=dict(regime='consistent', health_events=[])))
    return rows


def test_prefixes_do_not_use_future_observations_for_any_window():
    source = snapshot()
    altered = deepcopy(source)
    for r in altered['rows'][30:]:
        r['pace_min_mile'] *= 1.5
    for days in (28,42,56):
        a, b = prefix_levels(source,days), prefix_levels(altered,days)
        assert {i:a[i] for i in range(30)} == {i:b[i] for i in range(30)}


def test_linear_signal_is_preserved_and_wider_window_can_lag_a_real_change():
    for days in (28,42,56):
        levels = prefix_levels(snapshot(),days)
        assert levels[59] == pytest.approx(math.log(10)-.001*59)
    changed = snapshot(step=True)
    a,b = prefix_levels(changed,28), prefix_levels(changed,56)
    assert b[59] > a[59]  # slower current estimate after a genuine improvement


def test_rebuild_changes_response_not_training_or_the_common_target():
    source = snapshot()
    rows = examples()
    levels = prefix_levels(source,56)
    a = deepcopy(rows)
    result = rebuild(rows,levels)
    assert rows == a
    for old,new in zip(rows,result):
        assert new['log_miles'] == old['log_miles']
        assert new['consistency'] == old['consistency']
        assert new['actual'] == pytest.approx(math.exp(.5*(levels[old['outcome_ids'][0]]+levels[old['outcome_ids'][1]])))
    target = common_targets(rows,source)
    assert target[rows[0]['origin'],rows[0]['target_id']]['benchmark'] == rows[0]['actual']


def test_every_policy_form_required_and_future_response_embargoed():
    rows = examples()
    policies = {p:deepcopy(rows) for p in POLICIES}
    origin = rows[20]['origin']
    a = fit_at(policies,origin)
    assert a is not None
    changed = deepcopy(policies)
    for rows_ in changed.values():
        for r in rows_:
            if r['outcome_end']>=origin:
                r['change']+=1.
    assert fit_at(changed,origin) == a
    changed['gap7_fraction0.667'] = rows[:2]
    assert fit_at(changed,origin) is None
    assert all(f['latest_outcome']<origin for fits in a.values() for f in fits.values())


def test_default_window_ablations_coincide_and_score_a_fixed_target():
    rows = examples()
    policies = {p:deepcopy(rows) for p in POLICIES}
    targets = {(r['origin'],r['target_id']):dict(benchmark=r['actual'],raw=r['actual'],start=r['start_pace']) for r in rows}
    results = evaluate(policies,policies,targets,28)
    assert results
    for origin in {r['origin'] for r in results}:
        chosen = [r for r in results if r['origin']==origin and r['mode']!='unchanged_anchor']
        assert len({r['predicted'] for r in chosen})==1
        assert max(abs(r['predicted']-r['fixed_benchmark']) for r in chosen)<1e-10
    assert all(r['latest_training_outcome']<r['origin'] for r in results)


def test_future_response_labels_do_not_change_earlier_forecasts():
    rows = examples()
    policies = {p:deepcopy(rows) for p in POLICIES}
    changed = deepcopy(policies)
    cutoff = rows[25]['origin']
    for cohort in changed.values():
        for r in cohort:
            if r['outcome_end']>=cutoff:
                r['change']+=.3
                r['actual']*=math.exp(.3)
    targets = {(r['origin'],r['target_id']):dict(benchmark=r['actual'],raw=r['actual'],start=10.) for r in rows}
    a,b = evaluate(policies,policies,targets,56), evaluate(changed,policies,targets,56)
    key = lambda r:(r['origin'],r['target_id'],r['mode'])
    assert {key(r):r['predicted'] for r in a if r['origin']<=cutoff} == {
        key(r):r['predicted'] for r in b if r['origin']<=cutoff}


def test_own_target_can_look_perfect_without_improving_fixed_or_run_targets():
    base = dict(origin='2026-09-01', origin_id=1, target_id='2', outcome_ids=[2], outcome_end='2026-09-15',
        horizon=14, elapsed=.5, regime='consistent', persistence=10., fixed_benchmark=10.,raw_target=10.)
    a = {**base,'bandwidth':28,'mode':'joint','predicted':10.1,'own_benchmark':10.}
    b = {**base,'bandwidth':56,'mode':'joint','predicted':11.,'own_benchmark':11.}
    report = comparisons([a,b])['consistent_all']['56']['joint']['all']
    assert report['own_smoother']['candidate']['mae_seconds']==0.
    assert report['fixed_benchmark']['candidate']['mae_seconds']>report['fixed_benchmark']['baseline']['mae_seconds']
    assert report['raw_runs']['candidate']['mae_seconds']>report['raw_runs']['baseline']['mae_seconds']
