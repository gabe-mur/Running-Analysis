from math import inf, nan

import pytest

from run_analysis import weekly_schedule as planner
from run_analysis.selection_objective import CONTINUATION_FUNDING_SCORER
from scripts.boundary_funding_prototype import full_path_funding, boundary_funding_policy
from scripts.volume_path_prototype import volume_path, volume_path_policy
from scripts.audit_conditional_upload import scoring
from scripts.audit_continuation_upload import evaluate_physical
from scripts.replay_synthetic_easy_dose import synthetic_history
from test_typed_continuation import fixture, CONFIG


def price(sessions, rate, length=49):
    _, program = fixture()
    days, states = program(length, sessions)
    path = volume_path(days, states, [(17., 18.)]*length, CONFIG, trace=True)
    result = full_path_funding(path, rate, (17., 18.), legacy_residual=max(0., 17.-rate, rate-18.))
    return path, result


def test_exact_boundary_witness_removes_quota_not_actual_volume_cost():
    path = dict(ordinary_miles=3.5, squared_violation=.12009442653374562)
    before = full_path_funding(path, 17.88365, (17.63469,18.72041), legacy_residual=4.557142857142857)
    after = full_path_funding(path, 17.88365, (17.63469,18.72041), legacy_residual=0.)
    assert before['target_cost'] == after['target_cost']
    assert before['removed_cost'] == pytest.approx(59.335860058309024)
    assert before['volume_cost'] > 0
    assert before['rate_cost'] == 0


def test_removes_cross_term_but_preserves_real_sustained_rate_shortfall():
    result = full_path_funding(dict(ordinary_miles=3.5, squared_violation=2.),
        15., (17.,18.), legacy_residual=5.)
    assert result['rate_violation'] == 2.
    assert result['removed_hard_quota_violation'] == 3.
    weight = planner.TARGET_FIT_UNIT/3.5**2
    assert result['target_cost'] == pytest.approx(weight*(2.+2.**2))
    assert result['removed_cost'] == pytest.approx(weight*(5.**2-2.**2))


def test_persistent_forty_percent_undertraining_still_costs_more():
    steady = {i: ('easy', miles) for start in range(0,49,7)
              for i,miles in ((start,4.),(start+2,4.),(start+4,9.))}
    _, normal = price(steady, 17.)
    _, short = price({i:(kind,miles*.6) for i,(kind,miles) in steady.items()}, 10.2)
    assert short['volume_cost'] > normal['volume_cost']
    assert short['rate_cost'] > normal['rate_cost']
    assert short['target_cost'] > normal['target_cost']


def test_late_catchup_cannot_erase_missed_earlier_training_even_at_equal_rate():
    steady = {i: ('easy', 5.) for i in range(0,49,2)}
    late = {i: ('easy', 5.) for i in range(21,49,2)}
    empty, _ = price({}, 0.)
    delayed, result = price(late, 17.)
    _, normal = price(steady, 17.)
    assert delayed['days'][:21] == empty['days'][:21]
    assert result['volume_cost'] >= planner.TARGET_FIT_UNIT / delayed['ordinary_miles']**2 * sum(
        day['squared_violation'] for day in empty['days'][:21])
    assert result['target_cost'] > normal['target_cost']


def test_excess_and_terminal_work_remain_in_physical_path():
    steady = {i: ('easy', 5.) for i in range(0,49,2)}
    normal, baseline = price(steady, 17.)
    excess, high = price({i:('easy',m*2) for i,(_,m) in steady.items()}, 34.)
    assert excess['overload_squared_violation'] > normal['overload_squared_violation']
    assert high['rate_cost'] > baseline['rate_cost']
    terminal, _ = price({**steady,20:('long',20.)}, 17.)
    assert terminal['days'][:20] == normal['days'][:20]
    assert terminal['days'][21]['rate'] > normal['days'][21]['rate']


def test_repeated_future_promises_do_not_reset_declining_actual_opening_load():
    # Algebraic state control, not a replacement for actual daily replay:
    # same promised schedule after successive periods with no executed work.
    _, make = fixture()
    days, states = make(35, {i:('easy',5.) for i in range(4,35,2)})
    opening = states[0].recent_load.continuous_distance_miles
    half_life = CONFIG.get('coaching',{}).get('continuous_fatigue_half_life_days',7)
    prices = []
    for missed_days in (0,7,14):
        changed = [states[0].model_copy(update={'recent_load':states[0].recent_load.model_copy(
            update={'continuous_distance_miles':opening*.5**(missed_days/half_life)})}),*states[1:]]
        path = volume_path(days,changed,[(17.,18.)]*len(days),CONFIG)
        prices.append(full_path_funding(path,17.,(17.,18.),legacy_residual=0.)['target_cost'])
    assert prices[0] < prices[1] < prices[2]


@pytest.mark.parametrize('rate',[16.9,16.,15.])
def test_even_modest_sustained_underload_remains_priced(rate):
    result = full_path_funding(dict(ordinary_miles=3.5,squared_violation=0.),
        rate,(17.,18.),legacy_residual=17.-rate)
    assert result['target_cost'] > 0
    assert result['rate_violation'] == pytest.approx(17.-rate)


@pytest.mark.parametrize('rate,target,legacy', [(nan,(17.,18.),0.),(inf,(17.,18.),0.),
    (-1.,(17.,18.),0.),(17.,(18.,17.),0.),(17.,(0.,18.),0.),(15.,(17.,18.),1.)])
def test_invalid_or_inconsistent_accounting_rejected(rate,target,legacy):
    with pytest.raises(ValueError):
        full_path_funding(dict(ordinary_miles=3.5,squared_violation=0.),rate,target,legacy_residual=legacy)


def test_scope_restores_on_failure_and_nested_scope():
    assert CONTINUATION_FUNDING_SCORER.get() is None
    with pytest.raises(RuntimeError):
        with boundary_funding_policy() as metrics:
            outer = CONTINUATION_FUNDING_SCORER.get()
            with boundary_funding_policy():
                assert CONTINUATION_FUNDING_SCORER.get() is not outer
            assert CONTINUATION_FUNDING_SCORER.get() is outer
            outer(dict(ordinary_miles=3.5,squared_violation=0.),17.,(17.,18.),legacy_residual=0.)
            assert metrics['evaluations'] == 1
            raise RuntimeError('fixture')
    assert CONTINUATION_FUNDING_SCORER.get() is None


def integrated():
    state_at, make = fixture()
    full, states = make(35, {0:('easy',4.),3:('long',8.),7:('easy',4.),11:('long',8.),
        15:('easy',4.),19:('easy',4.),20:('long',8.),24:('easy',4.),28:('long',8.),32:('easy',4.)})
    template, history = synthetic_history(states[0].as_of)
    evidence = dict(template=template,runs=history,source='realized_simulation')
    def evaluate(cut):
        inputs = dict(daily_states=states[:cut], daily_state_options=[
            [state_at(d.recommendation.planned_for)] if d.recommendation else [s]
            for d,s in zip(full[:cut],states[:cut])],
            request=planner.RecommendationRequest(health_status='normal'),config=CONFIG,
            target_distance_range=(17.,18.),completed_miles_by_offset={},prior_distance_midpoints_by_date={},
            expected_target_projector=lambda recommendations,n:tuple((17.,18.) for _ in range(n)))
        with scoring(inputs,evidence,False):
            return evaluate_physical(inputs,full[:cut],evidence,full=full,length=len(full)-cut)
    return evaluate


def test_actual_combined_objective_has_no_hard_cut_dependence_or_changed_recovery():
    evaluate = integrated()
    with volume_path_policy():
        old = evaluate(20)['continuation']
        with boundary_funding_policy():
            results = [evaluate(cut)['continuation'] for cut in (19,20,21,22)]
    for row in results:
        assert row['combined_cost'] == pytest.approx(results[0]['combined_cost'])
        assert row['trajectory_recovery_cost'] == old['trajectory_recovery_cost']
        assert row['combined_coaching_cost'] == old['combined_coaching_cost']
        assert row['combined_target_cost'] == pytest.approx(row['funding_accounting']['target_cost'])
    assert old['combined_cost']-results[1]['combined_cost'] == pytest.approx(
        results[1]['funding_accounting']['removed_cost'])


def test_funding_scope_cannot_run_without_volume_path():
    evaluate = integrated()
    with boundary_funding_policy(), pytest.raises(ValueError,match='requires additive'):
        evaluate(20)
