"""A tiny planner rounding gap is not missed execution or extra physical credit."""
from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace

import pytest

from run_analysis.adaptive.continuous_general_progression import propose
from run_analysis.stage_target_history import StageTarget
from run_analysis.staged_progression import (
    DELIVERY_ROUNDING_MILES, MILE, ResponseReport, review_recent_delivery,
)
from run_analysis.workload_accounting import Quantity
from test_stage_acceptance import ledger_entries
from test_staged_progression import stage


def fixture():
    s=stage(weekly_equivalent_miles=20.)
    rows=ledger_entries(s)
    # Preserve both long contracts; shorten one easy so issued/actual total is 39.
    e=rows[1]
    target=replace(e.contract.targets[0],quantity=Quantity.point(2.2*MILE,'issued'))
    rows[1]=replace(e,contract=replace(e.contract,targets=(target,)),
        completion=replace(e.completion,distance_meters=Quantity.point(2.2*MILE,'actual')))
    return s,rows


def review(s,rows,**kwargs):
    options=dict(ledger_complete=True,longest_prior_30d_miles=s.long_miles,
        distance_credit='actual_distance_v2',observation_anchor=s.issued_at,
        required_miles=39.05193340962192,rounding_tolerance_miles=DELIVERY_ROUNDING_MILES)
    options.update(kwargs)
    return review_recent_delivery(s,rows,s.review_at,**options)


@pytest.mark.parametrize('shortfall',[.000001,.05193340962192,.1])
def test_one_total_increment_accepts_fully_delivered_plan_without_inventing_miles(shortfall):
    s,rows=fixture()
    r=review(s,rows,required_miles=39.+shortfall)
    assert r['status']=='observed_delivery_supported'
    a=r['volume_accounting']
    assert a['rounding_tolerance_applied'] and a['changes_progression_gate']
    assert a['rounding_tolerance_miles']==.1
    assert a['credited_miles']==pytest.approx(39.)
    assert a['prescribed_miles']==pytest.approx(39.)
    assert a['required_miles']==pytest.approx(39.+shortfall)
    assert a['attribution']=='planner_underprescription'


@pytest.mark.parametrize('shortfall',[.100001,.2,1.,3.])
def test_meaningful_planner_gap_is_not_tolerated_or_multiplied_per_run(shortfall):
    s,rows=fixture()
    r=review(s,rows,required_miles=39.+shortfall)
    assert r['status']=='recent_delivery_hold_targets'
    assert not r['volume_accounting']['rounding_tolerance_applied']


@pytest.mark.parametrize('cause',['undershoot','missing','late','uncertain'])
def test_rounding_is_not_an_execution_shortfall_allowance(cause):
    s,rows=fixture()
    e=rows[1]
    if cause=='missing':
        rows[1]=replace(e,completion=None)
    else:
        done=e.completion
        if cause=='late':done=replace(done,available_at=s.review_at+timedelta(seconds=1))
        elif cause=='uncertain':done=replace(done,distance_meters=Quantity(
            2.19*MILE,2.21*MILE,2.2*MILE,'measurement bounds'))
        else:done=replace(done,distance_meters=Quantity.point(2.19*MILE,'actual'))
        rows[1]=replace(e,completion=done)
    r=review(s,rows)
    assert r['status']=='recent_delivery_hold_targets'
    assert not r['volume_accounting']['rounding_tolerance_applied']
    assert r['credited_miles']<39.


def test_old_default_remains_strict_and_fully_funded_review_needs_no_allowance():
    s,rows=fixture()
    old=review(s,rows,rounding_tolerance_miles=0.)
    assert old['status']=='recent_delivery_hold_targets'
    assert 'rounding_tolerance_miles' not in old['volume_accounting']
    funded=review(s,rows,required_miles=39.)
    assert funded['status']=='observed_delivery_supported'
    assert not funded['volume_accounting']['rounding_tolerance_applied']


@pytest.mark.parametrize('cause',['no_endurance','health','adverse','incomplete_ledger','too_early','session_guard'])
def test_rounding_does_not_bypass_other_growth_guards(cause):
    s,rows=fixture()
    opts={}
    if cause=='no_endurance':rows=[replace(e,contract=replace(e.contract,purpose='easy_volume')) for e in rows]
    elif cause in ('health','adverse'):
        opts['responses']=(ResponseReport(s.review_at,'health_concern' if cause=='health' else 'adverse','test'),)
    elif cause=='incomplete_ledger':opts['ledger_complete']=False
    elif cause=='too_early':opts['observation_anchor']=s.issued_at+timedelta(days=1)
    else:opts['longest_prior_30d_miles']=1.
    r=review(s,rows,**opts)
    assert r['status'] in {'endurance_reference_not_represented','health_review_required',
        'dose_specific_review_required','incomplete_review_ledger','insufficient_observation_history',
        'held_long_requires_session_guard_review'}


@pytest.mark.parametrize('opts',[
    {'rounding_tolerance_miles':-.1}, {'rounding_tolerance_miles':.2},
    {'rounding_tolerance_miles':float('nan')}, {'rounding_tolerance_miles':float('inf')},
    {'observation_anchor':None}, {'distance_credit':'issued_nominal_v1'},
])
def test_allowance_is_bounded_and_explicitly_continuous(opts):
    s,rows=fixture()
    with pytest.raises(ValueError,match='Rounding tolerance'):
        review(s,rows,**opts)


def test_continuous_caller_uses_tolerance_without_repricing_target_or_actuals():
    s,rows=fixture()
    required=39.05193340962192
    target=StageTarget('test',s.issued_at,s.issued_at.date(),required/2,10.,'test')
    h=SimpleNamespace(stage=s,planner_preferences=None,general_progression_state=None,
        target_history=(target,),decision=lambda at:at.date(),zone=s.issued_at.tzinfo,
        policy=SimpleNamespace(progression_credit='actual_distance_v2'))
    r=propose(h,s.review_at,rows,longest=s.long_miles)
    assert r['gate_status']=='observed_delivery_supported'
    assert r['status']=='continuous_growth'
    assert r['credited_miles']==pytest.approx(39.)
    assert r['required_miles']==pytest.approx(required)
    assert r['target'].volume_per_7==pytest.approx(required/2)
    assert r['candidate'] is None and not r['long_progression']
    assert h.general_progression_state is None and h.target_history==(target,)
