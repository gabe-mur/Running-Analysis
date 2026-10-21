from datetime import timedelta

import pytest

from scripts.audit_visible_week_edits import investigate, replace_prescribed_dose, splice_tail
from scripts.continuation_replan import signature
from test_typed_continuation import fixture


def test_dose_swap_preserves_time_and_other_days_and_copies_semantics():
    _,make=fixture()
    days,_=make(3,{0:('easy',3.3),2:('easy',3.5)})
    donor,_=make(3,{0:('easy',4.),2:('easy',4.)})
    donor[0].recommendation.planning_role='ordinary_easy'
    donor[0].recommendation.planned_for+=timedelta(hours=2)
    original=signature(days)
    result=replace_prescribed_dose(days,donor,{days[0].date})
    assert result[0].recommendation.planned_distance_miles==4.
    assert result[0].recommendation.planning_role=='ordinary_easy'
    assert result[0].recommendation.planned_for==days[0].recommendation.planned_for
    assert result[1:]==days[1:]
    assert signature(days)==original


def test_missing_dose_date_and_different_workout_are_rejected():
    _,make=fixture()
    days,_=make(3,{0:('easy',3.3)})
    donor,_=make(3,{0:('long',8.)})
    with pytest.raises(ValueError,match='easy prescriptions'):
        replace_prescribed_dose(days,donor,{days[0].date})
    with pytest.raises(ValueError,match='Missing dose date'):
        replace_prescribed_dose(days,days,{days[-1].date+timedelta(days=1)})


def test_tail_splice_preserves_exact_prefix_and_donor_without_mutation():
    _,make=fixture()
    days,_=make(4,{0:('easy',3.3),2:('easy',3.5)})
    donor,_=make(4,{0:('easy',4.),3:('long',8.)})
    before=signature(days),signature(donor)
    result=splice_tail(days,donor,days[2].date)
    assert result==days[:2]+donor[2:]
    assert (signature(days),signature(donor))==before
    with pytest.raises(ValueError,match='identical complete date grids'):
        splice_tail(days,donor[:-1],days[2].date)
    with pytest.raises(ValueError,match='inside the program'):
        splice_tail(days,donor,days[0].date)


@pytest.mark.parametrize('indices',[(5,5,7),(7,6,5),(5,6)])
def test_cross_frame_indices_must_be_distinct_and_chronological(indices):
    with pytest.raises(ValueError,match='three distinct chronological'):
        investigate({}, {'decisions':[]}, indices=indices)


def test_missing_verified_frames_and_duplicate_dose_dates_are_rejected():
    with pytest.raises(ValueError,match='Missing verified'):
        investigate({}, {'decisions':[]}, indices=(5,6,7))
    from datetime import date
    with pytest.raises(ValueError,match='distinct target and partner'):
        investigate({}, {'decisions':[]}, target=date(2026,10,2),partner=date(2026,10,2))
