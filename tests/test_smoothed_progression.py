"""Numerical shadow controls, NOT calendar stability or physiological validation."""
from dataclasses import replace
from datetime import datetime,timedelta,timezone

import pytest

from run_analysis.adaptive.smoothed_progression import support,growth,advance,kernel,mass
from run_analysis.stage_target_history import StageTarget
from run_analysis.staged_progression import MILE
from run_analysis.workload_accounting import Quantity
from test_staged_progression import stage,entry

AT=datetime(2026,10,19,tzinfo=timezone.utc)
ANCHOR=AT-timedelta(days=140)


def targets(volume=20.):
    return (StageTarget('reference',ANCHOR,ANCHOR.date(),volume,8.,'fixed test reference'),)


def running(miles=20., *, shift=0, factor=1., weeks=20):
    s=stage(issued_at=ANCHOR)
    rows=[]
    for i in range(weeks):
        for j,(offset,share) in enumerate(((0,.45),(2,.15),(3,.15),(5,.25))):
            when=ANCHOR+timedelta(days=i*7+offset+(shift if j==0 else 0),hours=12)
            e=entry(s,miles*share*factor,key=f'{i}:{j}')
            target=replace(e.contract.targets[0],quantity=Quantity.point(miles*share*MILE,'issued'))
            done=replace(e.completion,start=when,end=when+timedelta(hours=1),
                available_at=when+timedelta(hours=1),moving_seconds=Quantity.point(3600,'actual'),
                distance_meters=Quantity.point(miles*share*factor*MILE,'actual'))
            rows.append(replace(e,scheduled_at=when,contract=replace(e.contract,
                purpose='long_endurance' if j==0 else 'easy_volume',targets=(target,)),completion=done))
    return rows


def evidence(rows=None, at=AT, ts=None):
    return support(running() if rows is None else rows,targets() if ts is None else ts,
        as_of=at,anchor=ANCHOR,zone=timezone.utc)


def test_kernel_has_no_expiry_cliff_and_older_long_fades_continuously():
    for age in (7.,14.,28.,90.):
        assert abs(kernel(age-.00001,7)-kernel(age+.00001,7))<1e-6
    assert kernel(0,7)==0 and 0<mass(140,7)<=1
    for age in (0.000001,.1,1.,7.,14.):
        eps=min(age/2,1e-5)
        assert (mass(age+eps,7)-mass(age-eps,7))/(2*eps)==pytest.approx(kernel(age,7),rel=1e-5)


@pytest.mark.parametrize('miles',[5.,20.,60.])
def test_matching_volume_across_scales_has_same_support_not_a_distance_breakpoint(miles):
    e=evidence(running(miles),ts=targets(miles))
    baseline=evidence()
    assert e['delivery_ratio']==pytest.approx(baseline['delivery_ratio'])
    assert growth(miles,e)['support_fraction']==pytest.approx(growth(20.,baseline)['support_fraction'])


def test_shifting_long_one_day_changes_support_smoothly_not_window_membership():
    base,shifted=running(),running(shift=1)
    for hour in range(0,24*14,6):
        # Both courses are fully recorded here; do not compare a future week
        # with no supplied runs to an otherwise ongoing periodic schedule.
        at=AT-timedelta(days=14)+timedelta(hours=hour)
        a,b=evidence(base,at),evidence(shifted,at)
        assert abs(a['actual_per_7']-b['actual_per_7'])<.5
        assert abs(growth(20.,a)['growth_per_7']-growth(20.,b)['growth_per_7'])<.02


def test_rounding_sized_gap_reduces_rate_slightly_instead_of_zeroing_it():
    ratio=39/39.05193340962192
    p=growth(19.6,dict(delivery_ratio=ratio,mean_age_days=14,actual_per_7=19.6*ratio))
    assert .035<p['growth_per_7']<.04
    assert p['physical_credit_added']==0


@pytest.mark.parametrize('factor',[.94,1.,1.08])
def test_over_and_undershoot_use_actuals_and_never_exceed_maximum(factor):
    e=evidence(running(factor=factor))
    base=evidence()
    assert e['actual_per_7']==pytest.approx(base['actual_per_7']*factor)
    assert e['issued_per_7']==pytest.approx(base['issued_per_7'])
    p=growth(20.,e)
    assert 0<=p['growth_per_7']<=.04+1e-12
    if factor==.94:assert p['growth_per_7']<growth(20.,base)['growth_per_7']
    if factor==1.08:assert p['growth_per_7']==pytest.approx(.04)


def test_omission_lowers_support_without_miles_credit_or_catchup_debt():
    rows=running();missed=[replace(e,completion=None) if e.key=='18:0' else e for e in rows]
    full,short=evidence(rows),evidence(missed)
    assert short['actual_per_7']<full['actual_per_7']
    assert short['issued_per_7']==pytest.approx(full['issued_per_7'])
    assert '18:0' not in short['completed_keys'] and not short['future_credit']
    assert growth(20.,short)['growth_per_7']<growth(20.,full)['growth_per_7']


def test_no_future_work_or_future_target_can_buy_support():
    rows=running();old=evidence(rows)
    future=rows[-1]
    future=replace(future,key='future',scheduled_at=AT+timedelta(days=1),
        contract=replace(future.contract,issued_at=AT+timedelta(seconds=1)),
        completion=replace(future.completion,key='future',start=AT+timedelta(days=1),
            end=AT+timedelta(days=1,hours=1),available_at=AT+timedelta(days=1,hours=1),
            distance_meters=Quantity.point(1000*MILE,'future scenario')))
    ts=(*targets(),StageTarget('future',AT+timedelta(seconds=1),AT.date(),1000.,8.,'future'))
    assert evidence([*rows,future],ts=ts)==old


def test_uncertainty_lower_bound_and_late_upload_are_not_nominal_completion():
    rows=running();e=rows[-1]
    bounds=replace(e,completion=replace(e.completion,distance_meters=Quantity(0.,10*MILE,5*MILE,'uncertain')))
    late=replace(e,completion=replace(e.completion,available_at=AT+timedelta(days=1)))
    zero=replace(e,completion=replace(e.completion,distance_meters=Quantity.point(0.,'actual')))
    assert evidence([*rows[:-1],bounds])['actual_per_7']==evidence([*rows[:-1],zero])['actual_per_7']
    assert evidence([*rows[:-1],late])['actual_per_7']==evidence(rows[:-1])['actual_per_7']


def test_duplicate_observations_cannot_buy_growth():
    rows=running()
    with pytest.raises(ValueError,match='Duplicate opportunity'):evidence([*rows,rows[0]])
    with pytest.raises(ValueError,match='Completion counted twice'):
        evidence([*rows,replace(rows[0],key='different')])


def test_intraday_refresh_and_infinite_gap_cannot_compound_or_bank_growth():
    p=growth(20.,dict(delivery_ratio=1.,mean_age_days=14,actual_per_7=20.))
    for _ in range(100):assert advance(20.,p,elapsed_days=0,destination=60.)==20.
    assert advance(20.,p,elapsed_days=1,destination=60.)==pytest.approx(20.*1.04**(1/7))
    assert advance(20.,p,elapsed_days=100000,destination=60.)==pytest.approx(20.*1.04**2)
    hold=growth(20.,dict(delivery_ratio=.5,mean_age_days=14,actual_per_7=10.))
    assert advance(20.,hold,elapsed_days=100000,destination=60.)==20.


def test_prohibiting_growth_is_a_hard_override_not_smoothing_health_concerns():
    p=growth(20.,dict(delivery_ratio=10.,mean_age_days=14,actual_per_7=200.),permitted=False)
    assert p['growth_per_7']==0 and p['support_fraction']==0
    assert advance(20.,p,elapsed_days=7,destination=60.)==20.


def test_target_and_observations_are_not_mutated_or_dependent_on_candidate_schedule():
    rows=running();ts=targets();before=tuple(rows)
    a=evidence(rows,ts=ts)
    for _ in range(3):assert evidence(rows,ts=ts)==a
    assert tuple(rows)==before and ts==targets()


def test_startup_is_normalized_without_inventing_pre_anchor_training():
    start=AT-timedelta(days=14)
    ts=(StageTarget('new',start,start.date(),5.,2.,'new user'),)
    e=support([],ts,as_of=AT,anchor=start,zone=timezone.utc)
    assert e['actual_per_7']==0 and e['target_per_7']==pytest.approx(5.)
    assert growth(5.,e)['growth_per_7']==0


def test_flat_actual_training_cannot_fund_indefinitely_rising_targets():
    rows=running(weeks=60);ts=targets();volume=20.;rates=[]
    for n in range(181):
        at=AT+timedelta(days=n)
        e=evidence(rows,at=at,ts=ts)
        p=growth(volume,e)
        previous_volume=volume
        if n:volume=advance(volume,p,elapsed_days=1,destination=60.)
        # A falling ceiling holds the accepted target, not a downward rebase;
        # any new increase must still fit the currently observed headroom.
        assert volume<=max(previous_volume,e['actual_per_7']*(1.04**2))+1e-8
        ts=(*ts,StageTarget(f'day{n}',at,at.date(),volume,8.+volume-20.,'test',
            growth_per_7=p['growth_per_7'],growth_limit=60.))
        rates.append(p['growth_per_7'])
    assert max(rates[-30:])<.001
    assert volume<20.*1.04**3


@pytest.mark.parametrize('scale',[5.,20.,60.])
def test_following_rising_prescriptions_advances_without_rate_or_scale_artifact(scale):
    # Fixed-date engineering course, NOT a recommended run frequency or a
    # searched calendar. Exercise the controller feedback rather than only
    # showing that it can hold flat training.
    rows=running(scale);ts=targets(scale);volume=scale;values=[]
    for n in range(43):
        at=AT+timedelta(days=n,hours=4)
        e=evidence(rows,at=at,ts=ts);p=growth(volume,e)
        previous=volume
        if n:volume=advance(volume,p,elapsed_days=1,destination=scale*3)
        assert previous<=volume<=previous*1.04**(1/7)+1e-8
        ts=(*ts,StageTarget(f'day{n}',at,at.date(),volume,scale*.4+volume-scale,'test',
            growth_per_7=p['growth_per_7'],growth_limit=scale*3))
        if n%7 in (0,2,3,5):
            share={0:.45,2:.15,3:.15,5:.25}[n%7];template=rows[-1]
            start=at.replace(hour=18);end=start+timedelta(hours=1)
            q=Quantity.point(volume*share*MILE,'observed prescribed distance')
            target=replace(template.contract.targets[0],quantity=q)
            contract=replace(template.contract,issued_at=at,targets=(target,))
            done=replace(template.completion,key=f'new{n}',start=start,end=end,
                available_at=end,distance_meters=q)
            rows.append(replace(template,key=f'new{n}',scheduled_at=start,contract=contract,completion=done))
        values.append(volume)
    assert .09<values[21]/scale-1<1.04**3-1
    assert values[42]>values[21]
