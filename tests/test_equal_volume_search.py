from datetime import date, timedelta
from itertools import islice

import pytest

from run_analysis.adaptive.calendar_operators import equal_volume_neighbors
from run_analysis.adaptive.continuous_search_adapter import search
from run_analysis.adaptive import full_score_refinement, finalized_distance_search


MILES = {'a': 2.1, 'b': 3.1, 'c': 3.2, 'd': 4.2, 'e': 6.3, 'long': 8.4}
ROLES = {k: 'long' if k == 'long' else 'easy' for k in MILES}


def test_exact_volume_merge_split_redistribution_and_physical_guards():
    prior = {0: 'long', 1: 'a', 2: 'a', 3: 'a', 6: 'long'}
    target = ((0, 'long'), (2, 'c'), (3, 'b'), (6, 'long'))
    rows = list(equal_volume_neighbors(prior, range(8), ROLES, MILES))
    assert target in rows
    assert tuple(sorted(prior.items())) in list(equal_volume_neighbors(dict(target), range(8), ROLES, MILES))
    assert ((0, 'long'), (2, 'e'), (6, 'long')) in rows
    assert len(rows) == len(set(rows))
    assert tuple(sorted(prior.items())) not in rows
    assert rows == list(equal_volume_neighbors(dict(reversed(list(prior.items()))), reversed(range(8)), ROLES, MILES))
    for row in rows:
        assert len(row) == len(dict(row))
        assert dict(row)[0] == dict(row)[6] == 'long'
        assert sum(round(MILES[k] * 10) for _, k in row) == 231
        assert set(dict(row)) <= set(range(8))
    pair = {1: 'b', 2: 'c'}
    assert ((1, 'c'), (2, 'b')) in list(equal_volume_neighbors(pair, range(4), ROLES, MILES))


def test_groups_are_interleaved_not_fixed_week_buckets():
    prior = {1: 'a', 2: 'a', 3: 'a', 8: 'a', 9: 'a', 10: 'a'}
    rows = list(islice(equal_volume_neighbors(prior, range(12), ROLES, MILES), 30))
    assert any(all(dict(row).get(d) == 'a' for d in (1, 2, 3))
               and any(dict(row).get(d) != 'a' for d in (8, 9, 10)) for row in rows)
    assert not list(equal_volume_neighbors({1: 'a', 9: 'a'}, (1, 9), ROLES, MILES))


@pytest.mark.parametrize('bad', [dict(MILES, a=2.11), dict(MILES, a=0), dict(MILES, a=float('nan')), {'a': 2.1}])
def test_metadata_never_silently_rounds_or_infers_miles(bad):
    with pytest.raises(ValueError):
        list(equal_volume_neighbors({}, (), ROLES, bad))


@pytest.mark.parametrize('eligible', [True, False])
def test_real_selector_combines_edits_without_rewarding_intermediates(monkeypatch, eligible):
    dates = tuple((date(2026, 10, 1) + timedelta(days=i)).isoformat() for i in range(7))
    prior = tuple((dates[i], 'a') for i in (1, 2, 3))
    target = ((dates[2], 'c'), (dates[3], 'b'))
    monkeypatch.setattr(full_score_refinement, 'refine', lambda *a, **k: dict(lanes=[]))
    monkeypatch.setattr(finalized_distance_search, 'refine_distances', lambda *a, **k: dict(lanes=[]))
    calls = []
    def evaluate(p):
        calls.append(p)
        return dict(cost=0. if p == target else 5. if p == prior else 10., eligible=eligible or p != target)
    result = search(dates=dates, keys=tuple(MILES), roles=ROLES, dose_miles=MILES,
        previous=prior, hard_end=dates[-1], seed_keys=('a',), budget=2000, evaluate=evaluate)
    assert target in calls
    assert result['program'] == (target if eligible else prior)
    assert result['continuity']['tolerance'] is None
    assert len(calls) == len(set(calls)) == result['evaluations'] <= 2000
    assert sum(result['refinement']['global_new_by_source'].values()) == len(calls)
    # A final certificate must include redistribution, not just the older edits.
    if result['local_neighborhood_complete']:
        assert result['refinement']['final_joint'][-1]['complete']


@pytest.mark.parametrize('budget', [2, 10, 100, 1000])
def test_shared_cap_and_cache_accounting_with_compound_moves(budget):
    dates = tuple((date(2026, 10, 1) + timedelta(days=i)).isoformat() for i in range(10))
    calls = []
    def evaluate(p):
        calls.append(p)
        return dict(cost=-sum(MILES[k] for _, k in p), eligible=True)
    result = search(dates=dates, keys=tuple(MILES), roles=ROLES, dose_miles=MILES,
        previous=((dates[0], 'a'),), hard_end=dates[-1], budget=budget, evaluate=evaluate)
    assert len(calls) == len(set(calls)) == result['evaluations'] <= budget
    assert sum(result['refinement']['global_new_by_source'].values()) == len(calls)
    if budget == 1000:
        assert result['refinement']['calendar'] is not None
        assert result['refinement']['protected_redistribution']['reserved'] > 0
        assert result['refinement']['protected_redistribution']['passes']
        assert result['refinement']['final_check_reserved'] > 0
        from scripts.audit_equal_volume_neighbors import budget_summary
        summary = budget_summary(result)
        assert summary['passes']['calendar']['evaluations'] is not None


def test_coordinate_sweep_compares_later_large_gain_before_accepting_tiny_gain(monkeypatch):
    dates = tuple((date(2026, 10, 1) + timedelta(days=i)).isoformat() for i in range(5))
    # The coarse beam cannot build the long-containing alternatives. Both are
    # reached by the same final coordinate sweep, with the tiny gain first.
    prior = ((dates[1], 'a'), (dates[4], 'long'))
    tiny = ((dates[0], 'a'), (dates[4], 'long'))
    target = ((dates[1], 'e'), (dates[4], 'long'))
    monkeypatch.setattr(full_score_refinement, 'refine', lambda *a, **k: dict(lanes=[]))
    monkeypatch.setattr(finalized_distance_search, 'refine_distances', lambda *a, **k: dict(lanes=[]))
    result = search(dates=dates, keys=tuple(MILES), roles=ROLES, previous=prior,
        hard_end=dates[-1], seed_keys=('a',), budget=1000,
        evaluate=lambda p: dict(cost=1. if p == target else 4.999 if p == tiny else 5. if p == prior else 10., eligible=True))
    assert result['refinement']['final_joint'][0]['best_cost'] == 1.
    assert result['program'] == target


def test_final_check_spends_reserved_calls_and_certifies_only_final_winner():
    dates = tuple((date(2026, 10, 1) + timedelta(days=i)).isoformat() for i in range(15))
    miles = {str(i): i / 10 for i in range(10, 81, 5)}
    kwargs = dict(dates=dates, keys=tuple(miles), roles={k: 'easy' for k in miles},
        dose_miles=miles, previous=((dates[2], '20'), (dates[9], '30')),
        hard_end=dates[-1], budget=1000,
        evaluate=lambda p: dict(cost=(sum(miles[k] for _, k in p)-20)**2+len(p)/100, eligible=True))
    result = search(**kwargs)
    assert sum(v for k, v in result['refinement']['global_new_by_source'].items() if k.startswith('reserved_')) > 0
    assert result['evaluations'] <= 1000
    assert result['local_neighborhood_complete']
    check = result['refinement']['final_joint'][-1]
    from run_analysis.adaptive.continuous_search_adapter import fingerprint
    assert check['phase'] == 'final_check' and check['complete']
    assert check['fingerprint'] == fingerprint(result['program'])
    repeated = search(**kwargs)
    assert repeated['program'] == result['program']
    assert repeated['evaluations'] == result['evaluations']


def test_frozen_witness_groups_are_actual_neighbors():
    # No saved score oracle or hindsight seed: the operator itself contains
    # both three-to-two fixes found in the separate equal-volume audit.
    prior = {d: 'a' for d in (0, 1, 2, 7, 8, 9, 14, 15, 16)}
    prior.update({3: 'long', 10: 'long', 17: 'long'})
    rows = set(equal_volume_neighbors(prior, range(21), ROLES, MILES))
    for start in (7, 14):
        target = {d: k for d, k in prior.items() if d not in range(start, start + 3)}
        target.update({start + 1: 'c', start + 2: 'b'})
        assert tuple(sorted(target.items())) in rows
