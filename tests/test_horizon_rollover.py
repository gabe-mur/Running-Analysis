from datetime import timedelta

import pytest

from scripts.audit_horizon_rollover import absolute_slice, align_rest_rollover, exact_timed_inputs
from scripts.continuation_replan import signature
from test_typed_continuation import fixture


def test_absolute_slice_preserves_physical_prescriptions_and_exact_endpoints():
    _,make=fixture()
    days,_=make(49,{2:('easy',3.5),48:('long',8.)})
    result=absolute_slice(days,days[1].date,days[-1].date+timedelta(days=1))
    assert signature(result)==signature(days[1:])
    assert len(result)==48


def test_unknown_day_is_not_padded_as_rest():
    _,make=fixture()
    days,_=make(49,{2:('easy',3.5)})
    with pytest.raises(ValueError,match='Missing'):
        absolute_slice(days,days[0].date,days[-1].date+timedelta(days=2))


def test_duplicate_dates_are_rejected():
    _,make=fixture()
    days,_=make(3,{2:('easy',3.5)})
    with pytest.raises(ValueError,match='duplicate'):
        absolute_slice(days+[days[-1]],days[0].date,days[-1].date+timedelta(days=1))


def test_documented_elapsed_rest_can_be_restored_before_later_plan():
    _,make=fixture()
    days,_=make(50,{2:('easy',3.5)})
    before,after=align_rest_rollover(days[:49],days[1:])
    assert signature(after)==signature(days)
    assert before==days[:49]


def test_elapsed_run_cannot_be_mistaken_for_rest_or_double_counted():
    _,make=fixture()
    days,_=make(50,{0:('easy',3.3),2:('easy',3.5)})
    with pytest.raises(ValueError,match='expected-completion'):
        align_rest_rollover(days[:49],days[1:])


def test_missing_soft_session_time_gets_raw_state_without_retiming_or_mutation():
    state_at,make=fixture()
    days,states=make(3,{2:('easy',3.5)})
    inputs=dict(daily_states=states,daily_state_options=[[s] for s in states])
    original=signature(days)
    adjusted,added=exact_timed_inputs(inputs,days,state_at)
    at=days[2].recommendation.planned_for
    assert added==[at.isoformat()]
    assert adjusted['daily_state_options'][2][-1]==state_at(at)
    assert len(inputs['daily_state_options'][2])==1
    assert signature(days)==original
    _,again=exact_timed_inputs(adjusted,days,state_at)
    assert again==[]


def test_bad_raw_state_time_is_not_relabelled_as_prescription_time():
    _,make=fixture()
    days,states=make(3,{2:('easy',3.5)})
    inputs=dict(daily_states=states,daily_state_options=[[s] for s in states])
    with pytest.raises(ValueError,match='timed state mismatch'):
        exact_timed_inputs(inputs,days,lambda at:states[2])
