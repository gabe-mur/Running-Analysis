from math import log

import pytest

from scripts.volume_path_prototype import volume_path, volume_path_policy, replace_volume_cost
from run_analysis.selection_objective import VOLUME_PATH_SCORER
from run_analysis import weekly_schedule as planner
from test_typed_continuation import fixture, CONFIG


def priced(sessions, length=35, target=(17.,18.), trace=True):
    _, program=fixture()
    days,states=program(length,sessions)
    return days,states,volume_path(days,states,[target]*length,CONFIG,trace=trace)


def test_completion_preserves_every_shared_date_price_and_prefix_cost_is_additive():
    days,states,before=priced({0:('long',8.),4:('easy',4.),8:('easy',4.),13:('long',8.)})
    after_states=states[1:]
    after_states[0]=after_states[0].model_copy(update={'recent_load':after_states[0].recent_load.model_copy(
        update={'continuous_distance_miles':before['days'][0]['rate']})})
    after=volume_path(days[1:],after_states,[(17.,18.)]*34,CONFIG,trace=True)
    assert before['days'][1:]==after['days']
    consumed=before['days'][0]['squared_violation']
    assert before['squared_violation']==pytest.approx(consumed+after['squared_violation'])


def test_shortfall_beyond_seven_days_is_not_invisible_or_repaid_by_later_running():
    prefix={0:('easy',4.),2:('long',8.),5:('easy',4.)}
    _,_,empty=priced(prefix,length=21)
    _,_,supplied=priced({**prefix,8:('easy',4.),11:('long',8.),14:('easy',4.),17:('easy',4.)},length=21)
    assert empty['days'][:7]==supplied['days'][:7]
    assert empty['underload_squared_violation']>supplied['underload_squared_violation']
    _,_,catchup=priced({**prefix,20:('long',20.)},length=21)
    assert catchup['days'][:20]==empty['days'][:20]
    assert catchup['squared_violation']>=sum(r['squared_violation'] for r in empty['days'][:20])


def test_more_dates_do_not_rescale_or_hide_existing_cost_and_tail_is_scored():
    sessions={0:('long',8.),4:('easy',4.),8:('easy',4.)}
    _,_,short=priced(sessions,length=21)
    _,_,long=priced(sessions,length=49)
    assert short['days']==long['days'][:21]
    assert long['squared_violation']>short['squared_violation']


def test_shortfall_is_gradual_and_one_session_allowance_is_retained():
    _,program=fixture()
    days,states=program(1,{})
    half_life=CONFIG.get('coaching',{}).get('continuous_fatigue_half_life_days',7)
    ordinary=sum(planner.typical_easy_distance(states[0]))/2
    tolerance=ordinary*log(2)*7/half_life
    ending=states[0].recent_load.continuous_distance_miles*.5**(1/half_life)
    zero=volume_path(days,states,[(ending+tolerance-.01,ending+tolerance+1)],CONFIG)
    tiny=volume_path(days,states,[(ending+tolerance+.01,ending+tolerance+1)],CONFIG)
    twice=volume_path(days,states,[(ending+tolerance+.02,ending+tolerance+1)],CONFIG)
    assert zero['cost']==0
    assert 0<tiny['cost']<.001
    assert twice['cost']==pytest.approx(4*tiny['cost'])


def test_overshoot_is_priced_and_inputs_are_not_changed():
    days,states,normal=priced({0:('easy',4.)},length=7)
    before=[s.model_dump() for s in states]
    bigger=[days[0].model_copy(update={'recommendation':days[0].recommendation.model_copy(
        update={'distance_target_miles':30.,'distance_range_miles':(29.8,30.2)})}),*days[1:]]
    high=volume_path(bigger,states,[(17.,18.)]*7,CONFIG)
    assert high['overload_squared_violation']>normal['overload_squared_violation']
    assert [s.model_dump() for s in states]==before


@pytest.mark.parametrize('bad', [(0.,1.),(18.,17.),(float('nan'),18.),(17.,float('inf'))])
def test_invalid_target_rejected(bad):
    _,program=fixture()
    days,states=program(1,{})
    with pytest.raises(ValueError):
        volume_path(days,states,[bad],CONFIG)


def test_missing_continuous_load_or_misaligned_dates_fail_explicitly():
    _,program=fixture()
    days,states=program(2,{})
    with pytest.raises(ValueError):
        volume_path(days,states,[(17.,18.)],CONFIG)
    with pytest.raises(ValueError,match='dates'):
        volume_path(days[::-1],states,[(17.,18.)]*2,CONFIG)
    states[0]=states[0].model_copy(update={'recent_load':states[0].recent_load.model_copy(
        update={'continuous_distance_miles':None})})
    with pytest.raises(ValueError,match='observed'):
        volume_path(days,states,[(17.,18.)]*2,CONFIG)


def test_context_restores_on_error_and_records_small_scoring_work():
    _,program=fixture()
    days,states=program(2,{})
    with pytest.raises(RuntimeError):
        with volume_path_policy() as metrics:
            VOLUME_PATH_SCORER.get()(days,states,[(17.,18.)]*2,CONFIG)
            assert metrics['evaluations']==1
            raise RuntimeError('fixture')
    assert VOLUME_PATH_SCORER.get() is None


def test_replacement_removes_old_path_once_and_keeps_other_costs():
    original=dict(total=12.,target=4.,target_violation=1.)
    result=replace_volume_cost(original,1.,dict(cost=2.),3.5)
    assert result['total']==10.
    with pytest.raises(ValueError):
        replace_volume_cost(original,2.,dict(cost=2.),3.5)


def test_hard_and_tail_volume_replace_once_in_real_combined_score():
    from scripts.audit_conditional_upload import scoring
    from scripts.audit_continuation_upload import evaluate_physical
    from scripts.replay_synthetic_easy_dose import synthetic_history

    state_at,program=fixture()
    full,states=program(28,{0:('easy',4.),3:('long',8.),7:('easy',4.),11:('long',8.),
                          15:('easy',4.),19:('easy',4.),23:('long',8.),26:('easy',4.)})
    template,history=synthetic_history(states[0].as_of)
    evidence=dict(template=template,runs=history,source='realized_simulation')
    inputs=dict(daily_states=states[:21],daily_state_options=[
        [state_at(d.recommendation.planned_for)] if d.recommendation else [s]
        for d,s in zip(full[:21],states[:21])],
        request=planner.RecommendationRequest(health_status='normal'),config=CONFIG,
        target_distance_range=(17.,18.),completed_miles_by_offset={},prior_distance_midpoints_by_date={},
        expected_target_projector=lambda recommendations,n:tuple((17.,18.) for _ in range(n)))
    with scoring(inputs,evidence,False), volume_path_policy():
        result=evaluate_physical(inputs,full[:21],evidence,full=full,length=7)
    details=result['continuation']['volume_path']
    direct=volume_path(full,states,[(17.,18.)]*28,CONFIG)
    assert details['full']['squared_violation']==direct['squared_violation']
    expected=planner.TARGET_FIT_UNIT*(details['residual_violation']/direct['ordinary_miles'])**2+direct['cost']
    assert result['continuation']['combined_target_cost']==pytest.approx(expected)
    assert details['full']['squared_violation']>=details['hard']['squared_violation']
