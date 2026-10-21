"""Mileage-conservation controls before another costly planner replay."""
from dataclasses import replace
from datetime import timedelta
from math import exp, log1p

import pytest

from run_analysis.adaptive.cadence_mass_growth import CalendarDose, estimate
from run_analysis.adaptive.realized_growth import review_reference
from run_analysis.workload_accounting import Quantity
from test_prescription_progression import opportunity, AT, ZONE


def course(*,level=20.,rate=.04,lengths=(7,)*16,redistribute=False,held_keys=False):
    rows=[]; refs=[]; clock=AT; elapsed=0.
    def integral(start,days,growth):
        slope=log1p(growth)/7
        return level*days/7 if slope==0 else level/7/slope*(exp(slope*(start+days))-exp(slope*start))
    for n,days in enumerate(lengths):
        end=clock+timedelta(days=days)
        actual=integral(elapsed,days,rate)
        intended=integral(elapsed,days,.04)
        fractions=(.45,.10,.17,.28) if not redistribute or n%2==0 else (.45,.26,.17,.12)
        doses=([level*.45*days/7,(actual-level*.62*days/7)*.3,
                level*.17*days/7,(actual-level*.62*days/7)*.7] if held_keys else [actual*f for f in fractions])
        slots=tuple(f'{n}:{j}' for j in range(4))
        for j,(fraction,miles) in enumerate(zip(fractions,doses)):
            when=clock+timedelta(days=days*j/4)
            e=opportunity(miles,nominal=intended*fraction,when=when,key=slots[j])
            rows.append(e)
            refs.append(CalendarDose(e.key,when,AT-timedelta(days=1),intended*fraction,
                'fixed-calendar','Synthetic original calendar',clock,end,slots))
        clock=end; elapsed+=days
    return rows,refs,clock+timedelta(hours=1),level*exp(log1p(rate)*elapsed/7)


def fit(rows,refs,at,baseline=20.):
    return estimate(rows,refs,as_of=at,zone=ZONE,baseline_volume=baseline)


@pytest.mark.parametrize('level',[5.,20.,60.])
@pytest.mark.parametrize('rate',[0.,.01,.02,.04,.06,-.01])
@pytest.mark.parametrize('lengths',[(7,)*12,(8,)*12,(6,8,10,7)*3])
def test_absolute_actual_mass_owns_growth_not_nominal_or_distribution(level,rate,lengths):
    rows,refs,at,current=course(level=level,rate=rate,lengths=lengths,redistribute=True)
    result=fit(rows,refs,at,baseline=level)
    assert result['growth_per_7']==pytest.approx(max(0.,min(.04,rate)),abs=1e-9)
    assert result['measured_volume_per_7']==pytest.approx(current,rel=1e-9)
    assert result['physical_credit_added']==0 and not result['calendar_cutoff']


@pytest.mark.parametrize('rate',[0.,.02,.04])
def test_held_long_and_quality_with_easy_first_growth_keep_total_mass(rate):
    rows,refs,at,current=course(rate=rate,held_keys=True)
    result=fit(rows,refs,at)
    assert result['growth_per_7']==pytest.approx(rate,abs=1e-9)
    assert result['measured_volume_per_7']==pytest.approx(current,rel=1e-9)


def test_dose_shape_and_original_prescription_values_are_not_a_ruler():
    rows,refs,at,_=course(rate=0.)
    first=fit(rows,refs,at)
    # Reproduce the failed 2.6/4.0/2.3 denominator pattern, plus arbitrarily
    # severe dose differences. It can no longer change actual measured mass.
    changed=[replace(r,miles=(2.6,4.,2.3,.1)[i%4]) for i,r in enumerate(refs)]
    after=fit(rows,changed,at)
    for key in ('level_ratio','growth_per_7','closed_spans','measured_volume_per_7'):
        assert after[key]==first[key]


def test_redistributing_constant_volume_cannot_create_growth():
    a,refs,at,_=course(rate=0.)
    b,_,_,_=course(rate=0.,redistribute=True)
    first,second=fit(a,refs,at),fit(b,refs,at)
    assert first['measured_volume_per_7']==second['measured_volume_per_7']==pytest.approx(20.)
    assert first['growth_per_7']==second['growth_per_7']==0.
    assert [r['actual_miles'] for r in first['closed_spans']]==[r['actual_miles'] for r in second['closed_spans']]


def test_flat_higher_actual_volume_is_recognized_without_extra_growth_or_reroll_credit():
    rows,refs,at,_=course(level=22.,rate=0.)
    signal=fit(rows,refs,at,baseline=20.)
    kwargs=dict(physical=dict(actual_per_7=22.,delivery_ratio=1.,mean_age_days=14),signal=signal,
        funding=dict(funded_level=20.),baseline_volume=20.,as_of=at,zone=ZONE,destination=100.)
    first=review_reference(None,volume=20.,**kwargs)
    assert first['state']['volume']==pytest.approx(22.)
    assert first['state']['growth_per_7']==0.
    second=review_reference(first['state'],volume=first['state']['volume'],**kwargs)
    assert second['newly_recognized_increase']==0. and second['state']==first['state']


def test_open_span_is_not_zero_or_growth_and_old_observations_do_not_expire():
    rows,refs,at,_=course(rate=.02)
    clock=AT+timedelta(days=35,hours=4)
    first=fit(rows,refs,clock)
    assert len(first['closed_spans'])==5
    # Only the first open-span run is due. Do not extrapolate its dose ratio
    # across the other runs, and do not count the pending ones as missed.
    later=fit(rows,refs,clock+timedelta(hours=2))
    assert later['level_ratio']==first['level_ratio']
    assert later['growth_per_7']==first['growth_per_7']
    all_done=fit(rows,refs,at)
    rested=fit(rows,refs,at+timedelta(days=60))
    assert rested['closed_spans']==all_done['closed_spans']
    assert rested['level_ratio']==all_done['level_ratio']


def test_known_open_span_miss_holds_stale_growth_but_late_actual_reconciles():
    rows,refs,_,_=course()
    clock=AT+timedelta(days=36,hours=4)
    key='5:0'
    changed=[replace(e,completion=None) if e.key==key else e for e in rows]
    result=fit(changed,refs,clock)
    assert result['pending_missed_opportunity'] and result['growth_per_7']==0.
    assert fit(rows,refs,clock)['growth_per_7']==pytest.approx(.04)


def test_old_miss_in_leading_partial_coverage_does_not_hold_recovered_growth_forever():
    rows,refs,at,_=course()
    rows[0]=replace(rows[0],completion=None)
    # Adoption can start between long dates. That leading fragment cannot
    # measure a full exposure interval, but its old miss is not perpetually
    # pending after later, fully observed training demonstrates recovery.
    refs=[replace(r,span_start=None,span_end=None,span_slots=()) if i<4 else r
        for i,r in enumerate(refs)]
    result=fit(rows,refs,at)
    assert not result['pending_missed_opportunity']
    assert result['growth_per_7']==pytest.approx(.04)
    assert len(result['closed_spans'])==15


def test_new_miss_in_trailing_partial_coverage_still_holds_immediately():
    rows,refs,at,_=course()
    rows[-1]=replace(rows[-1],completion=None)
    refs=[replace(r,span_start=None,span_end=None,span_slots=()) if i>=len(refs)-4 else r
        for i,r in enumerate(refs)]
    result=fit(rows,refs,at)
    assert result['pending_missed_opportunity']
    assert result['growth_per_7']==0.


def test_unknown_distance_and_membership_changes_are_not_fitted_as_success():
    rows,refs,at,_=course()
    last=rows[-1]
    changed=rows[:-1]+[replace(last,completion=replace(last.completion,
        distance_meters=Quantity(0.,1.,None,'unknown')))]
    assert fit(changed,refs,at)['status']=='evidence_unresolved'
    bad=refs[:-1]+[replace(refs[-1],span_slots=refs[-1].span_slots[:-1]+('other',refs[-1].opportunity))]
    assert fit(rows,bad,at)['status']=='calendar_membership_reconciliation_required'


def test_duplicate_physical_or_calendar_identity_is_rejected():
    rows,refs,at,_=course()
    with pytest.raises(ValueError,match='Duplicate'):fit(rows+[rows[0]],refs,at)
    with pytest.raises(ValueError,match='Duplicate'):fit(rows,refs+[refs[0]],at)


def test_same_actual_mass_has_no_day_of_week_preference():
    rows,refs,at,_=course(rate=.02,lengths=(8,)*12)
    first=fit(rows,refs,at)
    for days in range(1,7):
        delta=timedelta(days=days)
        shifted=[replace(e,scheduled_at=e.scheduled_at+delta,
            contract=replace(e.contract,issued_at=e.contract.issued_at+delta),
            completion=replace(e.completion,start=e.completion.start+delta,end=e.completion.end+delta,
                available_at=e.completion.available_at+delta)) for e in rows]
        frozen=[replace(r,scheduled_at=r.scheduled_at+delta,available_at=r.available_at+delta,
            span_start=r.span_start+delta,span_end=r.span_end+delta) for r in refs]
        after=fit(shifted,frozen,at+delta)
        assert after['level_ratio']==pytest.approx(first['level_ratio'])
        assert after['growth_per_7']==pytest.approx(first['growth_per_7'])


@pytest.mark.parametrize('rate',[.02,.04])
def test_daily_controller_does_not_discount_or_double_count_the_actual_growth_rate(rate):
    rows,refs,_,_=course(rate=rate,lengths=(7,)*20,held_keys=True)
    volume=20*(1+rate)**2; state=None
    for day in range(14,113):
        at=AT+timedelta(days=day,hours=1)
        signal=fit(rows,refs,at)
        result=review_reference(state,volume=volume,
            physical=dict(actual_per_7=volume,delivery_ratio=1.,mean_age_days=14),
            signal=signal,funding=dict(funded_level=20.),baseline_volume=20.,
            as_of=at,zone=ZONE,destination=100.)
        volume=result['state']['volume'];state=result['state']
        assert state['growth_per_7']==pytest.approx(rate,abs=1e-9)
        assert volume==pytest.approx(20*(1+rate)**(day/7),rel=1e-9)
        # An extra same-day refresh does not re-credit the measured level.
        later=at+timedelta(hours=1)
        again=review_reference(state,volume=volume,
            physical=dict(actual_per_7=volume,delivery_ratio=1.,mean_age_days=14),
            signal=fit(rows,refs,later),funding=dict(funded_level=20.),baseline_volume=20.,
            as_of=later,zone=ZONE,destination=100.)
        assert again['state']['volume']==volume
        state=again['state']
