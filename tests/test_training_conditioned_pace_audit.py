from copy import deepcopy
from datetime import datetime,timedelta,timezone
import math
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from scripts.audit_training_conditioned_pace import (
    disjoint_summary,evaluate,make_examples,metrics,outcome_training,paired_summary,perfect_adherence_scenario,
    predict_change,ridge_fit,training_between,
)


AT=datetime(2026,1,1,tzinfo=timezone.utc)


def test_training_excludes_origin_outcome_and_incomplete_activity():
    def run(i,start,end=None):
        return dict(id=i,time=AT+timedelta(days=start),
            available_at=AT+timedelta(days=end if end is not None else start),
            miles=5,minutes=50,quality=False,long=False)
    runs=[run(0,0),run(1,1),run(2,2),run(3,3),run(4,1,4)]
    value=training_between(runs,AT,AT+timedelta(days=3))
    assert value['run_ids']==[1,2]
    assert value['total_miles']==10
    assert value['volume']==pytest.approx(10*7/3)
    with pytest.raises(ValueError):
        training_between(runs,AT,AT)


def test_examples_use_preceding_training_for_each_outcome_not_one_block_cutoff():
    dates=[AT+timedelta(days=i*3) for i in range(70)]
    rows=[dict(activity_id=i,start_time=d.isoformat(),available_at=(d+timedelta(hours=1)).isoformat(),
        pace_min_mile=10*math.exp(-i*.001)) for i,d in enumerate(dates)]
    training=[dict(id=i,time=d,available_at=d+timedelta(hours=1),miles=5,minutes=50,
        quality=i%5==0,long=i%5==1) for i,d in enumerate(dates)]
    examples=make_examples(rows,np.ones(len(rows)),training,{},ZoneInfo('UTC'))
    assert examples
    for r in examples:
        assert r['origin_id'] not in r['delivered_run_ids']
        for outcome,exposure in zip(r['outcome_ids'],r['training_by_outcome']):
            assert outcome not in exposure['run_ids']
            assert all(i<outcome for i in exposure['run_ids'])
        # An earlier outcome is legitimate completed training by a later run.
        assert r['outcome_ids'][0] in r['training_by_outcome'][-1]['run_ids']
        assert r['outcome_runs']>=2
        assert datetime.fromisoformat(r['origin'])<datetime.fromisoformat(r['outcome_start'])
        assert datetime.fromisoformat(r['outcome_end'])>datetime.fromisoformat(r['outcome_start'])
        assert r['end_fatigue'] is None


def test_october_quality_and_long_are_not_discarded_between_outcomes():
    def at(day):
        return datetime(2026,10,day,12,tzinfo=timezone.utc)
    runs=[dict(id=i,time=at(day),available_at=at(day)+timedelta(hours=1),
        miles=miles,minutes=60,quality=day==16,long=day in {12,19})
        for i,day,miles in [(390,12,8.934),(391,14,4.924),(392,16,3.604),(393,19,8.916)]]
    aggregate=outcome_training(runs,at(10),[at(14),at(19),at(22)],[1,1,1],{},ZoneInfo('UTC'))
    a,b,c=aggregate['by_outcome']
    assert a['run_ids']==[390]
    assert b['run_ids']==[390,391,392]
    assert c['run_ids']==[390,391,392,393]
    assert b['total_miles']==pytest.approx(17.462)
    assert c['total_miles']==pytest.approx(26.378)
    assert c['long_miles']>b['long_miles']
    current=training_between(runs,at(1),at(20))
    assert 393 in current['run_ids'] and current['total_miles']==pytest.approx(26.378)


def test_aggregate_uses_same_outcome_weights_for_training_as_for_log_pace():
    def at(day):return AT+timedelta(days=day)
    runs=[dict(id=1,time=at(1),miles=5,minutes=50,quality=False,long=False),
        dict(id=2,time=at(3),miles=8,minutes=80,quality=False,long=True)]
    value=outcome_training(runs,at(0),[at(2),at(4)],[.8,.2],{},ZoneInfo('UTC'))
    assert value['total_miles']==pytest.approx(.8*5+.2*13)
    assert value['union_miles']==13
    assert value['volume']==pytest.approx(.8*5*7/2+.2*13*7/4)
    with pytest.raises(ValueError):
        outcome_training(runs,at(0),[at(2)],[-1],{},ZoneInfo('UTC'))


def synthetic(effect=-.004):
    rng=np.random.default_rng(44)
    examples=[]
    for i in range(90):
        origin=AT+timedelta(days=i*4)
        delivered=float(rng.uniform(8,35));prior=float(rng.uniform(8,30))
        level=float(math.log(10)+rng.normal(0,.04));trend=float(rng.normal(0,.02))
        change=.01+.1*trend+effect*(delivered-20)
        examples.append(dict(origin=origin.isoformat(),outcome_start=(origin+timedelta(days=21)).isoformat(),
            outcome_end=(origin+timedelta(days=35)).isoformat(),horizon=28,
            target_id=str(i),outcome_ids=[i],origin_id=i,start_pace=math.exp(level),
            actual=math.exp(level+change),change=change,elapsed=1.,start_level=level,
            start_trend=trend,linear_trend=trend,prior_volume=prior,prior_quality=.5,
            delivered_volume=delivered,volume_change=delivered-prior,delivered_quality=.5,
            delivered_frequency=4.,delivered_minutes=delivered*10.,end_fatigue=None))
    return examples


def test_training_reduces_error_when_it_contains_real_predictive_information():
    predictions=evaluate(synthetic(),penalty=1.)
    result=paired_summary(predictions,'state_volume','state')['28']
    assert result['mae_improvement_seconds_per_mile']>10
    assert result['candidate']['pairs']==result['baseline']['pairs']
    assert result['coefficient_negative_fraction']==1.


def test_effect_direction_is_not_forced_to_reward_more_volume():
    predictions=evaluate(synthetic(effect=.004),penalty=1.)
    assert all(p['coefficients']['delivered_volume']>0 for p in predictions if p['model']=='state_volume')


def test_complete_outcome_embargo_prevents_future_label_leakage():
    examples=synthetic();changed=deepcopy(examples)
    cutoff=AT+timedelta(days=160)
    for r in changed:
        if datetime.fromisoformat(r['outcome_end'])>=cutoff:
            r['change']+=.5
    a=evaluate(examples);b=evaluate(changed)
    assert len(a)==len(b)
    for p,q in zip(a,b):
        assert datetime.fromisoformat(p['latest_training_outcome'])<datetime.fromisoformat(p['origin'])
        if datetime.fromisoformat(p['origin'])<=cutoff:
            assert p['predicted']==pytest.approx(q['predicted'])


def test_future_training_is_a_conditional_input_not_inferred_adherence():
    examples=synthetic();fit=ridge_fit(examples[:40],('elapsed','start_level','start_trend','delivered_volume'))
    original=examples[-1];more={**original,'delivered_volume':original['delivered_volume']+5}
    assert predict_change(fit,more)<predict_change(fit,original)
    assert fit['mean'][3]==pytest.approx(np.mean([r['delivered_volume'] for r in examples[:40]]))
    assert ridge_fit(examples[:7],('elapsed',)) is None


def test_missing_fatigue_does_not_become_zero_or_disable_primary_comparison():
    predictions=evaluate(synthetic())
    assert any(p['model']=='state_volume' for p in predictions)
    assert not any(p['model'] in {'state_prior_fatigue','plus_volume_fatigue'} for p in predictions)


def test_metrics_report_actual_outcome_count_and_empty_horizons():
    predictions=evaluate(synthetic())
    result=paired_summary(predictions,'state_volume','state')
    assert result['84']['candidate']['pairs']==0
    assert result['28']['candidate']['unique_outcome_runs']>0
    assert metrics([])['pairs']==0


def test_disjoint_outcome_check_keeps_baselines_paired():
    predictions=evaluate(synthetic())
    result=disjoint_summary(predictions,'state_volume','state')['28']
    assert result['candidate']['pairs']==result['baseline']['pairs']
    assert result['mae_improvement_seconds_per_mile']>0


def test_adherence_scenario_includes_oct19_in_history_and_labels_soft_tail():
    at=datetime(2026,10,20,12,tzinfo=timezone.utc)
    training=[dict(id=393,time=at-timedelta(days=1),available_at=at-timedelta(hours=22),
        miles=8.916,minutes=96,quality=False,long=True)]
    state=dict(current=dict(pace_min_mile=10.15),progress_trend=dict(log_pace_change_per_day=-.0006))
    program=[dict(day=datetime(2026,10,21).date(),miles=5),
        dict(day=datetime(2026,10,26).date(),miles=9),
        dict(day=datetime(2026,11,12).date(),miles=4),
        dict(day=datetime(2026,11,17).date(),miles=10)]
    result=perfect_adherence_scenario(synthetic(),state,training,program,as_of=at,zone=ZoneInfo('UTC'),
        hard_end=datetime(2026,11,10).date(),plan_end=datetime(2026,12,1).date())
    assert result['available'] and result['prior_28d_run_ids']==[393]
    assert result['prior_28d_volume_per_7']==pytest.approx(8.916/4)
    scenario=result['scenarios'][0]
    assert scenario['horizon_days']==28 and scenario['date']=='2026-11-17'
    # Nov 12's own 4 miles do not predict its own pace, but count for Nov 17.
    assert scenario['training_miles']==16 and scenario['issued_miles']==14
    assert scenario['tentative_continuation_miles']==2
    assert scenario['outcome_dates']==['2026-11-12','2026-11-17']
    assert len(scenario['fitting_sensitivity'])==3
    assert scenario['primary_pace_min_mile']>0


def test_adherence_scenario_does_not_invent_unknown_long_term_schedule():
    at=datetime(2026,10,20,12,tzinfo=timezone.utc)
    state=dict(current=dict(pace_min_mile=10.15),progress_trend=dict(log_pace_change_per_day=-.0006))
    result=perfect_adherence_scenario(synthetic(),state,[],[],as_of=at,zone=ZoneInfo('UTC'),
        hard_end=at.date(),plan_end=at.date())
    assert not result['available'] and not result['scenarios']
