from copy import deepcopy
from datetime import datetime,timedelta,timezone
import math

import numpy as np
import pytest

from scripts.audit_consistent_training_response import annotate,comparisons,evaluate,regime
from run_analysis.training_response import POLICIES,classify_consistency

AT=datetime(2026,1,1,tzinfo=timezone.utc)


def run(day,miles=3):
    return dict(time=AT+timedelta(days=day),available_at=AT+timedelta(days=day,hours=1),miles=miles)


def case(origin=42,end=70):
    return dict(origin=(AT+timedelta(days=origin,hours=2)).isoformat(),
        training_by_outcome=[dict(at=(AT+timedelta(days=end)).isoformat())])


def test_consistency_is_relative_to_person_not_an_absolute_mileage_floor():
    regular=[run(d,miles=.25) for d in range(0,100,2)]
    value=regime(case(),regular,[],era_start=AT)
    assert value['regime']=='consistent' and value['baseline_miles_per_7']<1
    assert value['health_events']==[]


def test_excluded_illness_run_still_labels_the_surrounding_response():
    regular=[run(d) for d in range(0,100,2)]
    illness=[dict(at=AT+timedelta(days=55),status='illness_recovery',activity_id=99)]
    value=regime(case(),regular,illness,era_start=AT)
    assert value['regime']=='health_affected'
    assert value['health_events'][0]['activity_id']==99
    # Also catches health context BEFORE the origin without inventing recovery.
    illness[0]['at']=AT+timedelta(days=30)
    assert regime(case(),regular,illness,era_start=AT)['regime']=='health_affected'
    illness[0]['status']='little_tired'
    assert regime(case(),regular,illness,era_start=AT)['regime']=='consistent'


def test_gap_and_relative_volume_reduction_are_distinct_and_thresholds_explicit():
    regular=[run(d) for d in range(0,100,2)]
    gapped=[r for r in regular if not AT+timedelta(days=51)<=r['time']<AT+timedelta(days=59)]
    assert regime(case(),gapped,[],era_start=AT)['regime']=='gap_affected'
    permissive=regime(case(),gapped,[],era_start=AT,max_gap_days=10)
    assert 'gap_affected' not in permissive['flags']
    low=[run(d,miles=.3 if d>=44 else 3) for d in range(0,100,2)]
    value=regime(case(),low,[],era_start=AT)
    assert value['regime']=='reduced_training' and value['maximum_gap_days']<7


def test_gap_crossing_context_start_cannot_be_hidden_by_clipping_boundary():
    training=[run(d) for d in [0,2,4,6,8,20,*range(22,100,2)]]
    # Context starts day 14, in the middle of the actual day 8–20 gap.
    value=regime(case(),training,[],era_start=AT)
    assert value['regime']=='gap_affected' and value['maximum_gap_days']==12


def test_reusing_context_metrics_is_identical_for_all_policy_cutoffs():
    training=[run(d,miles=.4 if 50<=d<60 else 3) for d in range(0,160,2)]
    context=regime(case(),training,[],era_start=AT)
    for policy in POLICIES.values():
        assert classify_consistency(context,**policy)==regime(case(),training,[],era_start=AT,**policy)


def test_window_prefilter_retains_exact_full_history_volume_and_frequency_checks():
    training=[run(d,miles=1+(d%7)/10) for d in range(-300,160,2)]
    e=case();origin=datetime.fromisoformat(e['origin']);start=origin-timedelta(days=28)
    end=datetime.fromisoformat(e['training_by_outcome'][0]['at'])
    baseline=[r for r in training if start<r['available_at']<=origin]
    volume=sum(r['miles'] for r in baseline)
    old=[]
    for offset in range(14,math.floor((end-start).total_seconds()/86400)+1):
        at=start+timedelta(days=offset)
        runs=[r for r in training if at-timedelta(days=14)<r['available_at']<=at]
        old.append((sum(r['miles'] for r in runs)/(volume/2),len(runs)/(len(baseline)/2)))
    metrics=regime(e,training,[],era_start=AT)
    assert metrics['minimum_volume_fraction']==min(v for v,f in old)
    assert metrics['minimum_frequency_fraction']==min(f for v,f in old)


def test_labels_ignore_pace_and_do_not_use_training_or_health_after_outcome():
    training=[run(d) for d in range(0,72,2)];e=case()
    first=regime(e,training,[],era_start=AT)
    changed={**e,'actual':1,'change':-10,'start_pace':100}
    assert regime(changed,[*training,run(80,100)],
        [dict(at=AT+timedelta(days=80),status='injury_affected')],era_start=AT)==first


def test_starting_a_new_history_cannot_be_declared_established_consistency():
    value=regime(case(origin=10,end=38),[run(d) for d in range(0,100,2)],[],era_start=AT)
    assert value['regime']=='insufficient_baseline'


def synthetic():
    rng=np.random.default_rng(6);rows=[]
    for i in range(65):
        at=AT+timedelta(days=7*i);dose=float(rng.uniform(.3,1.4));elapsed=float(rng.uniform(.7,1.2))
        clean=i%2==0;change=(-.04 if clean else .02)*dose
        rows.append(dict(origin=at.isoformat(),origin_id=i,outcome_end=(at+timedelta(days=28)).isoformat(),
            horizon=28,target_id=str(i),outcome_ids=[i],response_variance=.001,elapsed=elapsed,
            miles=dose,log_miles=math.log1p(dose),change=change,start_pace=10,actual=10*math.exp(change),
            consistency=dict(regime='consistent' if clean else 'health_affected')))
    return rows


def test_stratification_can_recover_known_clean_training_response_on_matched_cases():
    predictions=evaluate(synthetic());result=comparisons(predictions)['28']['miles']
    assert result['consistent_only']['mae_seconds']<1e-9
    assert result['pooled']['mae_seconds']>1
    assert result['consistent_only']['pairs']==result['pooled']['pairs']==result['persistence']['pairs']


def test_chronological_stratified_fit_does_not_see_future_outcomes_or_regimes():
    rows=synthetic();changed=deepcopy(rows);cut=AT+timedelta(days=300)
    for r in changed:
        if datetime.fromisoformat(r['outcome_end'])>=cut:
            r['change']+=.4;r['actual']*=math.exp(.4);r['consistency']['regime']='consistent'
    a=evaluate(rows);b=evaluate(changed)
    key=lambda r:(r['origin'],r['target_id'],r['fit_group'],r['model'])
    before=lambda r:datetime.fromisoformat(r['origin'])<=cut
    assert {key(r):r['predicted'] for r in a if before(r)}=={key(r):r['predicted'] for r in b if before(r)}
    assert all(r['latest_training_outcome']<r['origin'] for r in a)
