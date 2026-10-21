from datetime import timedelta

import pytest

from run_analysis import weekly_schedule as planner
from scripts.audit_zero_surprise_search import copy_doses, copy_times, translate_saved_program
from scripts.continuation_replan import signature
from test_typed_continuation import fixture


def shifted(day, delta):
    rec=day.recommendation
    if rec is not None:
        rec=rec.model_copy(update={'planned_for':rec.planned_for+delta,
            'fitness_state_as_of':rec.fitness_state_as_of+delta})
    return day.model_copy(update={'date':day.date+delta,'planned_at':
        day.planned_at+delta if day.planned_at else None,'recommendation':rec})


def test_boundary_translation_consumes_prefix_and_uses_explicit_extension():
    _,make=fixture()
    previous,_=make(4,{0:('easy',3.),2:('easy',4.)})
    terminal=shifted(previous[-1],timedelta(days=1))
    current=[*previous[1:],terminal]
    before=signature(previous),signature(current)
    hard,full,extension=translate_saved_program(previous,current,origin=current[0].date,
        hard_end=current[0].date+timedelta(days=2),full_end=current[0].date+timedelta(days=4))
    assert hard==previous[1:3]
    assert full==previous[1:]+[terminal]
    assert extension==[terminal]
    assert (signature(previous),signature(current))==before


def test_boundary_translation_rejects_nonadvanced_origin():
    _,make=fixture(); days,_=make(3,{0:('easy',3.)})
    with pytest.raises(ValueError,match='advanced boundary'):
        translate_saved_program(days,days,origin=days[0].date,
            hard_end=days[1].date,full_end=days[-1].date+timedelta(days=1))


def test_dose_copy_is_role_bounded_and_preserves_time():
    _,make=fixture()
    base,_=make(4,{0:('long',8.),1:('easy',3.),3:('easy',3.5)})
    donor,_=make(4,{0:('long',8.5),1:('easy',4.),3:('easy',4.)})
    original=signature(base)
    copied,changed=copy_doses(base,donor,{planner.WorkoutType.LONG})
    assert changed==[base[0].date.isoformat()]
    assert copied[0].recommendation.planned_distance_miles==8.5
    assert copied[1].recommendation.planned_distance_miles==3.
    assert copied[0].recommendation.planned_for==base[0].recommendation.planned_for
    assert signature(base)==original


def test_dose_copy_rejects_type_change():
    _,make=fixture()
    base,_=make(2,{0:('easy',3.)}); donor,_=make(2,{0:('long',8.)})
    with pytest.raises(ValueError,match='workout type'):
        copy_doses(base,donor,{planner.WorkoutType.EASY})


def test_time_promotion_copies_only_native_time_fields():
    _,make=fixture()
    base,_=make(2,{0:('easy',4.)}); donor,_=make(2,{0:('easy',3.)})
    donor[0].recommendation.planned_for+=timedelta(hours=2)
    donor[0].recommendation.fitness_state_as_of+=timedelta(hours=2)
    donor[0].planned_at+=timedelta(hours=2)
    original=base[0].recommendation.planned_distance_miles
    result=copy_times(base,donor,{base[0].date})
    assert result[0].recommendation.planned_for==donor[0].recommendation.planned_for
    assert result[0].planned_at==donor[0].planned_at
    assert result[0].recommendation.planned_distance_miles==original
    with pytest.raises(ValueError,match='Missing time-promotion date'):
        copy_times(base,donor,{base[-1].date+timedelta(days=1)})
