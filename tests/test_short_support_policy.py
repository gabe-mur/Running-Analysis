from datetime import timedelta

import pytest

from run_analysis import weekly_schedule as planner
from scripts.aerobic_distribution import bounded_support, per_run_shortfall_components
from scripts.audit_short_support_policy import counterfactuals, fee_ablations, placements, summarize
from scripts.easy_dose_prototype import Observation, build_profile
from scripts.typed_continuation_prototype import runs
from test_typed_continuation import fixture


@pytest.mark.parametrize('health,fragmentation', [('normal', 20 / 3), ('little_tired', 0.)])
def test_separate_fee_terms_match_legacy_formula(health, fragmentation):
    _, program = fixture()
    days, states = program(1, {0: ('easy', 2.)})
    assert planner.typical_easy_distance(states[0])[0] == 3.
    parts = per_run_shortfall_components(days, states, planner.RecommendationRequest(health_status=health))
    assert parts['support_fee'] == pytest.approx(4.)
    assert parts['fragmentation_fee'] == pytest.approx(fragmentation)
    assert parts['total_fee'] == pytest.approx(4. + fragmentation)


def test_medium_long_and_quality_do_not_acquire_shortfall_fees():
    _, program = fixture()
    days, states = program(3, {0: ('easy', 2.), 1: ('long', 8.), 2: ('intervals', 2.)})
    days[0] = days[0].model_copy(update={'day_role': 'medium_long_run'})
    request = planner.RecommendationRequest(health_status='normal')
    parts = per_run_shortfall_components(days, states, request)
    assert parts['total_fee'] == 0.
    with pytest.raises(ValueError, match='aligned'):
        per_run_shortfall_components(days, states[:1], request)


def test_ablations_remove_only_named_components():
    parts = {'support_fee': 4., 'fragmentation_fee': 20 / 3, 'total_fee': 32 / 3}
    before = dict(parts)
    costs = fee_ablations(50., parts)
    assert costs['legacy'] == 50.
    assert costs['distribution_plus_fragmentation'] == 46.
    assert costs['distribution_only'] == pytest.approx(50. - 32 / 3)
    assert parts == before
    with pytest.raises(ValueError, match='finite'):
        fee_ablations(float('nan'), parts)


def test_probes_hold_hard_prefix_and_total_volume_and_respect_blocks():
    _, program = fixture()
    days, states = program(49, {18: ('long', 8.), 23: ('easy', 4.), 30: ('easy', 4.), 40: ('easy', 4.)})
    original = [d.model_dump_json() for d in days]
    original_miles = sum(planner._midpoint(r) for r in runs(days))
    rows = list(counterfactuals(days, states, 21, blocked=(21, 22, 24)))
    assert len(rows) < 1600
    assert {r['pieces'] for r, _ in rows} == {0, 1, 2, 4, 8}
    assert any(r['family'] == 'split_existing' for r, _ in rows)
    for facts, candidate in rows:
        assert candidate[:21] == days[:21]
        assert not set(facts['edits']) & {21, 22, 24}
        assert all(candidate[i] == days[i] for i in range(49) if i not in facts['edits'])
        actual_miles = sum(planner._midpoint(r) for r in runs(candidate))
        assert actual_miles - original_miles == pytest.approx(facts['added_miles'])
        for i in facts['edits']:
            r = candidate[i].recommendation
            assert r.distance_target_miles * 10 == pytest.approx(round(r.distance_target_miles * 10))
            if r.distance_target_miles < 3.:
                assert r.planning_role == 'support_easy'
    assert [d.model_dump_json() for d in days] == original


def test_no_rounding_mileage_is_created_in_replacement_splits():
    _, program = fixture()
    days, states = program(35, {23: ('easy', 4.1)})
    rows = list(counterfactuals(days, states, 21))
    assert all(r['family'] != 'split_existing' for r, _ in rows)


def test_fragmentation_challenge_includes_spread_not_only_clustered_dates():
    pool = list(range(21, 49))
    for count in (2, 4, 8):
        choices = list(placements(pool, count))
        assert tuple(pool[:count]) in choices
        assert max(x[-1] - x[0] for x in choices) >= 24
        assert len(choices) == len(set(choices))
        assert all(len(x) == count and len(set(x)) == count for x in choices)


def test_probes_exclude_completed_empty_dates():
    from types import SimpleNamespace
    _, program = fixture()
    days, states = program(35, {23: ('easy', 4.)})
    days[24] = days[24].model_copy(update={'completed_activities': [SimpleNamespace(distance_miles=2.)]})
    assert all(24 not in r['edits'] for r, _ in counterfactuals(days, states, 21))


def test_selection_separates_eligibility_from_guard_checks_and_fees():
    def row(name, cost, guard, intent):
        return {'family': name, 'added_miles': 2., 'pieces': 1, 'guards_pass': guard,
                'legacy_intent_pass': intent, 'costs': {p: cost for p in
                ('legacy', 'distribution_plus_fragmentation', 'distribution_only')}}
    rows = [row('rest', 10., True, True), row('short', 5., True, False),
            row('unsafe', -100., False, False)]
    found = summarize(rows)
    assert found['legacy_intent']['distribution_only']['best']['family'] == 'rest'
    assert found['support_probe']['distribution_only']['best']['family'] == 'short'


def test_repeated_support_uploads_do_not_shrink_ordinary_history():
    _, program = fixture()
    _, states = program(1, {})
    now = states[0].as_of
    ordinary = Observation('ordinary', now - timedelta(days=30), 4., planning_role='ordinary_easy')
    support = [Observation(f'short-{i}', now - timedelta(days=i + 1), .5,
                           planning_role='support_easy', prescribed_range=(.3, .7)) for i in range(25)]
    profile = build_profile([ordinary, *support], now, half_life_days=84)
    assert profile.ordinary_planning_miles == 4.
    assert profile.ordinary_count == 1 and profile.support_count == 25
    assert profile.support_planning_miles == .5


def test_distribution_credit_is_not_created_by_splitting_same_day_files():
    _, program = fixture()
    days, _ = program(21, {})
    dates = [d.date for d in days]
    one = bounded_support(dates, [(dates[0], 4.)], ordinary_miles=3.)
    split = bounded_support(dates, [(dates[0], .5)] * 8, ordinary_miles=3.)
    assert one == split


def test_distributing_equal_small_volume_can_avoid_credit_spillage():
    # Important limitation: bounded credit prevents banking but does NOT by
    # itself penalize replacing an ordinary run with tiny runs on many dates.
    _, program = fixture()
    days, _ = program(14, {})
    dates = [d.date for d in days]
    one = bounded_support(dates, [(dates[0], 4.)], ordinary_miles=3.)
    split = bounded_support(dates, [(dates[i], .5) for i in range(8)], ordinary_miles=3.)
    assert split['violation'] < one['violation']
