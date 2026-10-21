from copy import deepcopy
from datetime import datetime,timedelta,timezone
import math

import pytest

from scripts.audit_training_rate_persistence import (
    descriptive_rate,evaluate,matched_comparison,recent_rate,select_update,stretches,
)

AT=datetime(2026,1,1,tzinfo=timezone.utc)


def clean_context():
    return dict(flags=[],health_events=[],maximum_gap_days=2,minimum_volume_fraction=1.,minimum_frequency_fraction=1.)


def fixture(changing=False):
    def level(day):
        return math.log(10)-.001*min(day,140)+(.0005 if changing else -.001)*max(0,day-140)
    observations=[dict(id=i,time=AT+timedelta(days=day),available_at=AT+timedelta(days=day,hours=1),
        log_pace=level(day),benchmark_log_pace=level(day),variance=.001,past_context=clean_context())
        for i,day in enumerate(range(0,430,3))]
    rows=[]
    for phase in range(3):
        for j in range(12):
            i=phase*12+j;day=28+phase*112+j*7;at=AT+timedelta(days=day,hours=2)
            rows.append(dict(origin=at.isoformat(),origin_id=i,outcome_end=(at+timedelta(days=14)).isoformat(),
                horizon=14,elapsed=.5,target_id=str(i),outcome_ids=[100+i],response_variance=.001,
                start_pace=math.exp(level(day)),actual=math.exp(level(day+14)),change=level(day+14)-level(day)))
    return rows,observations


def test_persistent_rate_predicts_later_stretches_without_fabricated_taper():
    rows,observations=fixture();predictions,episodes=evaluate(rows,observations)
    assert len(episodes)==3
    fixed=[r for r in predictions if r['method']=='fixed_previous_stretches']
    assert fixed and max(abs(r['predicted']-r['actual']) for r in fixed)<1e-10
    assert all(r['rate_per_day']==pytest.approx(-.001) for r in fixed)
    assert descriptive_rate(rows)==pytest.approx(-.001)


def test_recent_observations_can_learn_a_genuine_slowdown_without_sign_constraints():
    rows,observations=fixture(changing=True)
    last=rows[-1]
    rate=recent_rate(observations,last['origin'],42,'run')
    assert rate['rate_per_day']==pytest.approx(.0005)
    predictions,_=evaluate(rows,observations)
    comparison=matched_comparison(predictions,'run_42','fixed_previous_stretches',episode=3)
    assert comparison['candidate']['mae_seconds']<1e-9
    assert comparison['baseline']['mae_seconds']>1


def test_future_labels_and_observations_do_not_change_earlier_fit_or_selection():
    rows,observations=fixture(changing=True);changed=deepcopy(rows);changed_obs=deepcopy(observations)
    cutoff=AT+timedelta(days=260)
    for r in changed:
        if datetime.fromisoformat(r['outcome_end'])>=cutoff:
            r['change']+=.4;r['actual']*=math.exp(.4)
    for r in changed_obs:
        if r['available_at']>cutoff:r['log_pace']+=.4;r['benchmark_log_pace']+=.4
    a,episodes=evaluate(rows,observations);b,_=evaluate(changed,changed_obs)
    key=lambda r:(r['origin'],r['target_id'],r['method'])
    before=lambda r:datetime.fromisoformat(r['origin'])<=cutoff
    assert {key(r):r['predicted'] for r in a if before(r)}=={key(r):r['predicted'] for r in b if before(r)}
    first={e['episode']:e['first_origin'] for e in episodes}
    for r in a:
        if r['latest_input']:
            assert r['latest_input']<=r['origin']
            if r['method']=='fixed_previous_stretches':assert r['latest_input']<first[r['episode']]
        if r['latest_selection_outcome']:assert r['latest_selection_outcome']<r['origin']


def test_clean_update_does_not_use_observations_with_known_disrupted_context():
    rows,observations=fixture()
    for r in observations:r['past_context']['health_events']=[dict(status='illness')]
    origin=rows[-1]['origin']
    assert recent_rate(observations,origin,84,'run') is not None
    assert recent_rate(observations,origin,84,'clean_run') is None


def test_stretch_grouping_is_training_cohort_timing_only_not_improvement_sign():
    rows,_=fixture();a=stretches(rows)
    changed=deepcopy(rows)
    for r in changed:r['actual']*=2;r['change']+=math.log(2)
    assert stretches(changed)==a
    assert all(stretches(rows,gap)==a for gap in (12,14,21))


def test_update_selection_uses_common_cases_not_a_favorable_subset():
    prior=[]
    for i in range(10):
        for method,error in [('run_42',.02),('run_56',.01)]:
            prior.append(dict(method=method,origin=str(i),target_id=str(i),horizon=14,
                outcome_end=str(i),outcome_ids=[i],actual=10,predicted=10*math.exp(error),log_error=error))
    assert select_update(prior,['run_42','run_56'])['method']=='run_56'
    prior.extend(dict(method='run_42',origin=str(i),target_id=str(i),horizon=14,outcome_end=str(i),
        outcome_ids=[i],actual=10,predicted=10,log_error=0) for i in range(10,100))
    assert select_update(prior,['run_42','run_56'])['method']=='run_56'
