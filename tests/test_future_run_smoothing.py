from copy import deepcopy
from datetime import datetime, timedelta, timezone
import math

import pytest

from run_analysis.training_response import POLICIES
from scripts.audit_future_run_smoothing import actual_target, individual_runs, margins, summarize
from scripts.audit_pace_smoothing_forecast import fit_at, forecast

AT = datetime(2025,1,1,tzinfo=timezone.utc)


def prediction(i,horizon=14,days=28,mode='joint',error=.02):
    origin = AT+timedelta(days=i*40)
    return dict(origin=origin.isoformat(),origin_id=i,target_id=f'{i}_{horizon}',
        outcome_ids=[i*2+(horizon==28)],outcome_end=(origin+timedelta(days=horizon)).isoformat(),
        elapsed=horizon/28,horizon=horizon,bandwidth=days,mode=mode,regime='consistent',
        predicted=10.,fixed_benchmark=20.,raw_target=10*math.exp(-error),persistence=10.)


def test_primary_target_is_actual_corrected_runs_not_the_old_benchmark():
    r = prediction(0)
    a,b = actual_target(r),actual_target({**r,'fixed_benchmark':100.})
    assert a['actual']==b['actual']==r['raw_target']
    assert a['log_error']==b['log_error']==pytest.approx(.02)
    assert a['benchmark_actual']!=b['benchmark_actual']


def test_primary_scores_do_not_change_if_old_benchmark_changes():
    rows = [prediction(i,days=d,error=.02 if d==28 else .01) for i in range(5) for d in (28,42,56)]
    a = summarize([actual_target(r) for r in rows])
    b = summarize([actual_target({**r,'fixed_benchmark':r['fixed_benchmark']*2}) for r in rows])
    assert a==b
    result = a['consistent_all']['42']['joint']['all']
    assert result['candidate']['mae_seconds']<result['baseline']['mae_seconds']


def test_margin_calibration_uses_actual_run_misses_only():
    rows = [prediction(i,horizon=h,days=d,mode=m,error=.01*math.sqrt(h))
        for i in range(5) for h in (14,28) for d in (28,42,56) for m in ('joint','rate_only','anchor_only')]
    final = {str(d):dict(may_joint=9.,may_rate_only=9.,current_pace=10.) for d in (28,42,56)}
    a = margins([actual_target(r) for r in rows],final,211)
    b = margins([actual_target({**r,'fixed_benchmark':100.}) for r in rows],final,211)
    assert a['projections']==b['projections']
    for pair,methods in a['transfer_checks'].items():
        for method,report in methods.items():
            assert report['all']==b['transfer_checks'][pair][method]['all']
            assert report['sequential']==b['transfer_checks'][pair][method]['sequential']
    assert a['observed_actual_future_error_growth']['exponent']==pytest.approx(.5)
    exponents = a['fixed_exponents']
    assert 'previous_benchmark_power' not in exponents
    raw_changed = [actual_target({**r,'raw_target':r['raw_target']*.98}) for r in rows]
    assert margins(raw_changed,final,211)['projections']!=a['projections']


def test_a_different_may_point_cannot_masquerade_as_smaller_forecast_error():
    rows = [actual_target(prediction(i,horizon=h,days=d,mode=m,error=.01*math.sqrt(h)))
        for i in range(5) for h in (14,28) for d in (28,42,56) for m in ('joint','rate_only','anchor_only')]
    final = {str(d):dict(may_joint=9.,may_rate_only=9.,current_pace=10.) for d in (28,42,56)}
    altered = deepcopy(final)
    altered['42']['may_joint'] = 5.
    a = margins(rows,final,211)['projections']['continued_consistency']['42']['joint']['sqrt_time']
    b = margins(rows,altered,211)['projections']['continued_consistency']['42']['joint']['sqrt_time']
    assert a['same_center']==b['same_center']
    assert a['candidate_center']['absolute_error_80_log']==b['candidate_center']['absolute_error_80_log']
    assert a['candidate_center']['centered_low_pace']!=b['candidate_center']['centered_low_pace']


def fixture():
    snapshot = dict(rows=[dict(activity_id=i,pace_min_mile=10*math.exp(-.001*i)) for i in range(45)])
    rows = []
    for i in range(35):
        origin = AT+timedelta(days=i*7)
        for horizon in (14,28):
            exposures = [dict(days=7*j,total_miles=10*j,
                at=(origin+timedelta(days=7*j)).isoformat(),weight=.5) for j in (1,2)]
            rows.append(dict(origin=origin.isoformat(),origin_id=i,start_pace=10.,start_level=math.log(10),
                horizon=horizon,target_id=f'{i+1},{i+2}',outcome_ids=[i+1,i+2],
                outcome_end=(origin+timedelta(days=14)).isoformat(),training_by_outcome=exposures,
                elapsed=.375,log_miles=.1,origin_cumulative_miles=100+i*10,
                response_variance=.001,change=-.01,actual=10*math.exp(-.01),
                consistency=dict(regime='consistent')))
    return {d:{p:deepcopy(rows) for p in POLICIES} for d in (28,42,56)},snapshot


def test_individual_forecasts_are_unique_origin_run_pairs_not_horizon_repeats():
    policies,snapshot = fixture()
    rows = individual_runs(policies,snapshot)
    assert rows
    keys = [(r['origin'],r['target_id'],r['bandwidth'],r['mode']) for r in rows]
    assert len(keys)==len(set(keys))
    assert all(len(r['outcome_ids'])==1 for r in rows)
    raw = {r['activity_id']:r['pace_min_mile'] for r in snapshot['rows']}
    assert all(r['actual']==raw[r['outcome_ids'][0]] for r in rows)


def test_individual_forecast_uses_exact_future_date_and_preceding_dose():
    policies,snapshot = fixture()
    row = next(r for r in individual_runs(policies,snapshot) if r['mode']=='joint' and r['bandwidth']==42)
    parent = next(r for r in policies[28]['gap7_fraction0.500'] if r['origin']==row['origin'])
    run_id = row['outcome_ids'][0]
    ix = parent['outcome_ids'].index(run_id)
    exposure = parent['training_by_outcome'][ix]
    features = dict(elapsed=exposure['days']/28,
        log_miles=math.log1p(exposure['total_miles']/parent['origin_cumulative_miles']))
    assert row['predicted']==pytest.approx(forecast(fit_at(policies[42],row['origin']),features,10.))
    assert row['latest_training_outcome']<row['origin']<row['outcome_end']


def test_future_labels_cannot_change_earlier_individual_forecasts():
    policies,snapshot = fixture()
    changed = deepcopy(policies)
    cutoff = (AT+timedelta(days=180)).isoformat()
    for variants in changed.values():
        for rows in variants.values():
            for r in rows:
                if r['outcome_end']>=cutoff:
                    r['change']+=.5
    a,b = individual_runs(policies,snapshot),individual_runs(changed,snapshot)
    key = lambda r:(r['origin'],r['target_id'],r['bandwidth'],r['mode'])
    assert {key(r):r['predicted'] for r in a if r['origin']<=cutoff} == {
        key(r):r['predicted'] for r in b if r['origin']<=cutoff}
