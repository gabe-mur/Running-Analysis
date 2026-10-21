"""Numerical/causal controls, not a newly searched planner acceptance."""
from dataclasses import replace
from datetime import timedelta

import pytest

from run_analysis.adaptive.realized_growth import ReferenceDose, estimate, shadow_proposal, review_reference
from run_analysis.workload_accounting import Quantity
from run_analysis.staged_progression import MILE
from test_prescription_progression import opportunity, AT, ZONE


def course(rate=.02, *, scale=20., days=112, long_gap=7, frequency=4, level=1.,
           nominal_rate=.04, shift=0):
    rows=[]; refs=[]
    # Exact fixed-level reference for these synthetic calendars. Its shape can
    # vary (held key doses, cadence, frequency); it never comes from actuals or
    # division by a changing nominal target.
    for day in range(days):
        when=AT+timedelta(days=day+shift)
        if day % long_gap==0: base=scale*.45
        elif day%7 in ({3,5,6} if frequency==4 else {3,5}): base=scale*.55/(frequency-1)
        else: continue
        key=str(day)
        rows.append(opportunity(base*level*(1+rate)**(day/7),nominal=base*(1+nominal_rate)**(day/7),when=when,key=key))
        refs.append(ReferenceDose(key,when,AT-timedelta(days=1),base,'fixed','synthetic fixed-level calendar'))
    return rows,refs


def fit(rows,refs,at=None,**kw):
    return estimate(rows,refs,as_of=at or max(e.completion.end for e in rows)+timedelta(hours=1),zone=ZONE,**kw)


@pytest.mark.parametrize('scale',[5.,20.,60.])
@pytest.mark.parametrize('rate',[0.,.01,.02,.04,.06,-.01])
def test_actual_growth_not_prescription_growth_owns_rate(scale,rate):
    rows,refs=course(rate,scale=scale)
    result=fit(rows,refs)
    assert result['growth_per_7']==pytest.approx(min(.04,max(0.,rate)),abs=1e-10)
    assert result['physical_credit_added']==0 and result['planner_influence'] is False


@pytest.mark.parametrize('gap',[6,7,8,10])
@pytest.mark.parametrize('frequency',[3,4])
def test_long_spacing_and_low_frequency_do_not_cause_window_cliffs(gap,frequency):
    rows,refs=course(.02,long_gap=gap,frequency=frequency)
    for day in range(20,112):
        result=fit(rows,refs,AT+timedelta(days=day,hours=8))
        assert result['growth_per_7']==pytest.approx(.02,abs=1e-10)


def test_refreshing_or_passing_rest_days_cannot_change_signal_or_bank_credit():
    rows,refs=course(.02)
    first=fit(rows,refs)
    for days in (0,1,2,8,35):
        again=fit(rows,refs,max(e.completion.end for e in rows)+timedelta(days=days,hours=1))
        for key in ('growth_per_7','level_ratio','last_opportunity_at','included'):
            assert again[key]==first[key]


def test_below_prescribed_level_is_not_the_same_as_zero_actual_growth():
    rows,refs=course(.02,level=.95)
    result=fit(rows,refs)
    assert result['growth_per_7']==pytest.approx(.02)
    last=max((e.scheduled_at-AT).days for e in rows)
    assert result['level_ratio']==pytest.approx(.95*1.02**(last/7))


def test_flat_actuals_stay_flat_even_when_prescriptions_keep_rising():
    rows,refs=course(0.,nominal_rate=.10)
    assert fit(rows,refs)['growth_per_7']==0.


def test_future_upload_same_day_pending_then_omission_then_late_upload():
    rows,refs=course(.02,days=21)
    last=rows[-1]; at=last.scheduled_at+timedelta(hours=1)
    late=replace(last,completion=replace(last.completion,available_at=at+timedelta(days=2)))
    changed=rows[:-1]+[late]
    assert len(fit(changed,refs,at)['included'])==len(rows)-1
    missed=fit(changed,refs,at+timedelta(days=1))
    assert missed['included'][-1]['actual_miles']==0.
    assert missed['growth_per_7']<.02
    assert fit(changed,refs,at+timedelta(days=3))['growth_per_7']==pytest.approx(.02)


def test_excess_and_shortfall_at_same_clock_net_without_positive_only_tokens():
    rows,refs=course(.02)
    last=rows[-1]
    split=replace(last,key='split',completion=replace(last.completion,key='split-physical',
        distance_meters=last.completion.distance_meters.scale(.5)))
    half=replace(last,completion=replace(last.completion,distance_meters=last.completion.distance_meters.scale(.5)))
    reference=replace(refs[-1],opportunity='split',miles=refs[-1].miles*.5)
    refs=refs[:-1]+[replace(refs[-1],miles=refs[-1].miles*.5),reference]
    assert fit(rows[:-1]+[half,split],refs)['growth_per_7']==pytest.approx(.02)


def test_supplemental_miles_are_explicitly_linked_and_counted_once():
    rows,refs=course(.02)
    last=rows[-1]
    extra=replace(last.completion,key='extra',distance_meters=last.completion.distance_meters.scale(.1))
    smaller=replace(last,completion=replace(last.completion,distance_meters=last.completion.distance_meters.scale(.9)))
    result=fit(rows[:-1]+[smaller],refs,supplemental=((last.key,extra),))
    assert result['growth_per_7']==pytest.approx(.02)
    with pytest.raises(ValueError,match='counted twice'):
        fit(rows,refs,supplemental=((last.key,last.completion),))


def test_unknown_or_posthoc_reference_cannot_be_assumed_to_be_no_growth():
    rows,refs=course()
    assert fit(rows,refs[:-1])['status']=='evidence_unresolved'
    changed=refs[:-1]+[replace(refs[-1],available_at=rows[-1].scheduled_at)]
    with pytest.raises(ValueError,match='precede original issuance'): fit(rows,changed)
    with pytest.raises(ValueError,match='mix or reset'):
        fit(rows,refs[:-1]+[replace(refs[-1],baseline_key='new')])


def test_uncertain_distance_is_not_misclassified_as_zero_growth():
    rows,refs=course()
    last=rows[-1]
    rows[-1]=replace(last,completion=replace(last.completion,distance_meters=Quantity(3*MILE,5*MILE,4*MILE,'uncertain')))
    assert fit(rows,refs)['status']=='evidence_unresolved'


def test_no_artificial_minimum_number_of_runs_or_new_fourteen_day_delay():
    rows,refs=course(.02,days=5)
    assert len(rows)==2
    assert fit(rows,refs)['growth_per_7']==pytest.approx(.02)


@pytest.mark.parametrize('rate',[0.,.01,.02,.04])
def test_held_long_and_easy_first_growth_do_not_become_a_false_four_percent(rate):
    rows,refs=course(rate)
    for i,(row,ref) in enumerate(zip(rows,refs)):
        day=(row.scheduled_at-AT).days
        is_long=day%7==0
        factor=1. if is_long else 1+((1+rate)**(day/7)-1)/.55
        rows[i]=replace(row,completion=replace(row.completion,
            distance_meters=Quantity.point(ref.miles*factor*MILE,'held-long actual')))
        refs[i]=replace(ref,component='long' if is_long else 'easy',component_share=.45 if is_long else .55)
    assert fit(rows,refs)['growth_per_7']==pytest.approx(rate,abs=.002)


def test_future_scheduled_opportunity_cannot_pull_the_fit_forward():
    rows,refs=course()
    at=rows[-1].scheduled_at-timedelta(hours=1)
    last=replace(rows[-1],completion=replace(rows[-1].completion,
        start=at-timedelta(hours=2),end=at-timedelta(hours=1),available_at=at-timedelta(hours=1)))
    assert fit(rows[:-1]+[last],refs,at)==fit(rows[:-1],refs,at)


@pytest.mark.parametrize('shift',range(7))
def test_day_of_week_and_view_origin_do_not_define_the_growth_measurement(shift):
    rows,refs=course(.02,shift=shift)
    assert fit(rows,refs)['growth_per_7']==pytest.approx(.02,abs=1e-10)


@pytest.mark.parametrize('round_commands',[False,True])
def test_daily_closed_loop_does_not_multiply_the_growth_rate_by_completion_again(round_commands):
    from run_analysis.adaptive.smoothed_progression import advance
    rows,refs=course(.04,days=28)
    volume=20*1.04**4; rates=[]
    at=AT+timedelta(days=27,hours=8)
    state=review_reference(None,volume=volume,
        physical=dict(actual_per_7=volume,delivery_ratio=1.,mean_age_days=14),
        signal=fit(rows,refs,at),funding=dict(funded_level=volume),baseline_volume=20.,
        as_of=at,zone=ZONE,destination=100.)['state']
    for day in range(28,182):
        at=AT+timedelta(days=day,hours=8)
        signal=fit(rows,refs,at)
        result=review_reference(state,volume=volume,
            physical=dict(actual_per_7=volume,delivery_ratio=1.,mean_age_days=14),
            signal=signal,funding=dict(funded_level=volume),baseline_volume=20.,
            as_of=at,zone=ZONE,destination=100.)
        p=result['proposal']
        current=result['state']['volume']
        unchanged=advance(current,p,elapsed_days=0,destination=100.)
        # Numerical advance expects the proposal's current (recognized) base.
        # Recognition itself is owned by the stateful review, not by advance.
        assert p['current_volume']==result['state']['volume']
        assert unchanged==current
        volume=result['state']['volume']; state=result['state']
        rates.append(p['growth_per_7'])
        if day%7 in (0,3,5,6):
            base=9 if day%7==0 else 11/3
            miles=volume*base/20
            if round_commands: miles=round(miles,1)
            when=AT+timedelta(days=day)
            rows.append(opportunity(miles,nominal=miles,when=when,key=str(day)))
            refs.append(ReferenceDose(str(day),when,AT-timedelta(days=1),base,'fixed','closed-loop synthetic reference'))
    expected=20*1.04**(182/7)
    # Quantized 0.1-mile commands cause a measurable small fit difference,
    # not the former 4% -> approximately 2% repeated rate discount.
    assert volume==pytest.approx(expected,rel=.003 if round_commands else 1e-10)
    assert min(rates)>.039 if round_commands else min(rates)==pytest.approx(.04)


def test_response_guard_remains_but_old_funding_cannot_veto_demonstrated_volume():
    rows,refs=course(.02)
    signal=fit(rows,refs)
    physical=dict(actual_per_7=20.,delivery_ratio=1.,mean_age_days=14)
    p=shadow_proposal(20.,physical,signal,dict(funded_level=20.),baseline_volume=20.,permitted=False)
    assert p['growth_per_7']==0 and p['rate_status']=='response_veto'
    p=shadow_proposal(20.,physical,signal,dict(funded_level=10.),baseline_volume=20.)
    assert p['growth_per_7']==pytest.approx(.02)
    assert p['observed_above_funded_level'] is True
    assert p['recognized_planning_floor']==pytest.approx(20*signal['level_ratio'])


def test_frozen_reference_roundtrip_and_input_order_are_deterministic():
    from dataclasses import asdict
    from datetime import datetime
    import json
    rows,refs=course(.02)
    encoded=json.loads(json.dumps([asdict(r) for r in refs],default=lambda d:d.isoformat()))
    decoded=[ReferenceDose(**dict(r,scheduled_at=datetime.fromisoformat(r['scheduled_at']),
        available_at=datetime.fromisoformat(r['available_at']))) for r in encoded]
    first=fit(rows,refs); resumed=fit(rows,decoded)
    assert first==resumed
    assert fit(tuple(reversed(rows)),tuple(reversed(decoded)))['growth_per_7']==pytest.approx(first['growth_per_7'])


def test_approximate_reference_is_visible_and_cannot_authorize_progression():
    rows,refs=course(.02)
    refs=[replace(r,basis='diagnostic_easy_ramp_removal_v1') for r in refs]
    with pytest.raises(ValueError,match='diagnostic-only'): fit(rows,refs)
    signal=fit(rows,refs,allow_diagnostic_reference=True)
    assert signal['reference_quality']=='diagnostic_approximation'
    assert signal['growth_per_7']==pytest.approx(.02)
    p=shadow_proposal(20.,dict(actual_per_7=20.,delivery_ratio=1.,mean_age_days=14),
        signal,dict(funded_level=20.),baseline_volume=20.)
    assert p['growth_per_7']==0 and p['rate_status']=='unverified_no_growth_reference'
