from datetime import datetime,date,timedelta,timezone
from dataclasses import replace
import pytest

from run_analysis.stage_target_history import StageTarget,target_segments
from run_analysis.program_delivery import _target_total


AT=datetime(2026,9,27,8,tzinfo=timezone.utc)
DAY=AT.date()


def test_growth_does_not_retroactively_raise_previous_targets():
    old=StageTarget('old',AT-timedelta(days=14),DAY-timedelta(days=40),20.,8.,'seed')
    new=StageTarget('new',AT,DAY,21.,8.4,'accepted_growth')
    volume,support=target_segments((new,old),as_of=AT,start=DAY-timedelta(days=27),end=DAY+timedelta(days=21))
    assert _target_total(volume,DAY-timedelta(days=27),DAY+timedelta(days=1))==pytest.approx((27*20+21)/7)
    assert _target_total(support,DAY,DAY+timedelta(days=7))==8.4
    earlier,_=target_segments((new,old),as_of=AT-timedelta(seconds=1),start=DAY-timedelta(days=27),end=DAY+timedelta(days=21))
    assert _target_total(earlier,DAY,DAY+timedelta(days=7))==20.


def test_missing_or_conflicting_history_fails_not_new_target_backfill():
    row=StageTarget('stage',AT,DAY,20.,8.,'fixture')
    for records,start in (((row,),DAY-timedelta(days=1)),((row,replace(row,stage_key='other')),DAY)):
        with pytest.raises(ValueError,match='lead-in'):
            target_segments(records,as_of=AT,start=start,end=DAY+timedelta(days=21))


def test_gradual_path_is_anchored_capped_and_not_backdated_or_rounded():
    seed=StageTarget('seed',AT,DAY-timedelta(days=40),18.6,6.3,'seed')
    ramp=StageTarget('ramp',AT,DAY,18.6,6.3,'bounded growth',14,19.5,7.2)
    def resolve(at,start,end):
        return target_segments((seed,ramp),as_of=at,start=start,end=end)
    volume,support=resolve(AT,DAY-timedelta(days=1),DAY+timedelta(days=100))
    for offset in (-1,0,1,7,14,15,99):
        day=DAY+timedelta(days=offset)
        rate=18.6+.9*min(1,max(0,offset/14))
        assert _target_total(volume,day,day+timedelta(days=1))*7==pytest.approx(rate)
        assert _target_total(support,day,day+timedelta(days=1))*7==pytest.approx(rate-12.3)
    later,_=resolve(AT+timedelta(days=7),DAY+timedelta(days=7),DAY+timedelta(days=45))
    for offset in range(7,45):
        day=DAY+timedelta(days=offset)
        assert _target_total(later,day,day+timedelta(days=1))==_target_total(volume,day,day+timedelta(days=1))


def test_dated_review_uses_the_ramp_not_flat_endpoint():
    from run_analysis.stage_target_history import target_miles_between
    ramp=StageTarget('ramp',AT,DAY,18.6,6.3,'bounded growth',14,19.5,7.2)
    volume,_=target_segments((ramp,),as_of=AT,start=DAY,end=DAY+timedelta(days=15))
    end=AT+timedelta(days=14)
    expected=sum((18.6+.9*n/14)/7 for n in range(14))+.9/7/3
    assert target_miles_between(volume,AT,end,timezone.utc)==pytest.approx(expected)
    assert expected>18.6*2
    assert expected<19.5*2


@pytest.mark.parametrize('changes',[dict(ramp_days=-1),dict(ramp_days=14),
    dict(volume_goal=20.),dict(ramp_days=14,volume_goal=17.,support_goal=8.)])
def test_invalid_ramp_rejected(changes):
    with pytest.raises(ValueError):
        StageTarget('bad',AT,DAY,18.6,6.3,'fixture',**changes)
