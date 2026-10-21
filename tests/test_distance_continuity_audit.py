from copy import deepcopy

import pytest

from scripts.audit_distance_continuity import replace_easy_dose, components, allocation_trace
from scripts.continuation_replan import signature
from run_analysis import weekly_schedule as planner
from test_typed_continuation import fixture


def test_distance_probe_changes_only_selected_easy_target_and_bounds():
    _, make = fixture()
    days, _ = make(5, {0: ('easy', 3.3), 2: ('long', 8.), 4: ('easy', 4.)})
    before = deepcopy(signature(days))
    changed = replace_easy_dose(days, days[0].date, 3.5, (3.2, 3.8))
    expected = deepcopy(before)
    expected[0]['prescription'].update(distance_target_miles=3.5, distance_range_miles=[3.2, 3.8])
    assert signature(changed) == expected
    assert signature(days) == before


@pytest.mark.parametrize('target,bounds', [(3.35, (3.,4.)), (4.1, (3.,4.)), (0., (0.,1.))])
def test_probe_rejects_invalid_targets(target, bounds):
    _, make = fixture()
    days, _ = make(1, {0: ('easy', 3.3)})
    with pytest.raises(ValueError, match='tenth-mile'):
        replace_easy_dose(days, days[0].date, target, bounds)


def test_probe_does_not_modify_quality_or_long_doses():
    _, make = fixture()
    days, _ = make(1, {0: ('long', 8.)})
    with pytest.raises(ValueError, match='Only easy'):
        replace_easy_dose(days, days[0].date, 3.5, (3.2,3.8))


def test_components_require_exact_full_score_accounting():
    full = dict(combined_target_cost=1., combined_coaching_cost=2., trajectory_recovery_cost=3.,
        combined_support_violation=.2, combined_shape_violation=.3)
    full['combined_cost'] = 6. + .2 * planner.PROGRAM_FIT_UNIT * planner.AEROBIC_SUPPORT_FIT_FRACTION + .3 * planner.PROGRAM_FIT_UNIT + .192
    scored = dict(hard={'distance_switching_cost': .192}, continuation=full)
    result = components(scored)
    assert sum(result.values()) == pytest.approx(full['combined_cost'])
    full['combined_cost'] += 1.
    with pytest.raises(ValueError, match='do not sum'):
        components(scored)


def test_small_round_trip_is_free_but_different_published_targets_anchor_later_scores():
    _, make = fixture()
    days, _ = make(1, {0: ('easy', 3.3)})
    moderate = replace_easy_dose(days, days[0].date, 3.5, (3.2, 3.8))
    longer = replace_easy_dose(days, days[0].date, 3.8, (3.5, 4.))
    def prior(program):
        return {d.date: d.recommendation for d in program}
    assert planner._distance_stability_cost(moderate, prior(days)) == 0
    assert planner._distance_stability_cost(days, prior(moderate)) == 0
    assert planner._distance_stability_cost(days, prior(longer)) == pytest.approx(6.)
    assert planner._distance_stability_cost(longer, prior(days)) == pytest.approx(6.)


def test_allocation_instrumentation_restores_hooks_when_scoring_fails(monkeypatch):
    from scripts import audit_distance_continuity as audit
    original_options = planner._easy_allocation_options
    original_allocate = planner._allocate_visible_distance_ranges
    def fail(*args, **kwargs):
        assert planner._easy_allocation_options is not original_options
        assert planner._allocate_visible_distance_ranges is not original_allocate
        raise RuntimeError('probe failure')
    monkeypatch.setattr(audit, 'hard_score', fail)
    saved = dict(selected_hard_fingerprint='test', finalists=[dict(hard_fingerprint='test', offsets=[])])
    with pytest.raises(RuntimeError, match='probe failure'):
        allocation_trace({}, saved, [], None)
    assert planner._easy_allocation_options is original_options
    assert planner._allocate_visible_distance_ranges is original_allocate
