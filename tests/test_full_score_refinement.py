from types import SimpleNamespace

import pytest

from scripts.calendar_neighbor_prototype import calendar_neighbors
from scripts.full_score_refinement import refine, full_score_refinement_policy, REFINEMENT_BUDGET


def candidate(offsets, cost, stable=False, source='surrogate_finalist', tag=''):
    return dict(score=SimpleNamespace(offsets=offsets, total_cost=cost), stable=stable,
                row=dict(hard_fingerprint=str(offsets)+tag, source=source))


def test_complete_descent_reaches_known_minimum_and_caches_across_starts():
    calls = []
    def evaluate(seed):
        offsets = tuple(seed['offsets'])
        calls.append(offsets)
        return candidate(offsets, len(set(offsets) ^ {0, 2}))
    result = refine([candidate((0,), 1), candidate((2,), 1, stable=True)],
                    evaluate, {0, 1, 2}, budget=100, starts=3)
    assert len(calls) == len(set(calls))
    assert result['calendar_cache_hits'] > 0
    assert all(lane['status'] == 'locally_converged' for lane in result['lanes'])
    assert all(lane['final_offsets'] == [0, 2] for lane in result['lanes'])
    assert all(lane['neighborhoods'][-1]['complete'] for lane in result['lanes'])


def test_budget_exhaustion_is_not_local_convergence():
    result = refine([candidate((1,), 0)], lambda seed: None, {0, 1, 2}, budget=1, starts=1)
    assert result['evaluated_calendars'] == 1
    assert result['lanes'][0]['status'] == 'budget_exhausted'
    assert not result['lanes'][0]['neighborhoods'][-1]['complete']


def test_exact_budget_can_still_prove_convergence_and_rejections_are_cached():
    allowed = {0, 2, 3}
    seed = (0, 2)
    neighbors = calendar_neighbors(seed, allowed)
    seen = []
    def reject(item):
        seen.append(tuple(item['offsets']))
        return None
    result = refine([candidate(seed, 0), candidate(seed, 1, stable=True, tag='role')],
                    reject, allowed, budget=len(neighbors), starts=2)
    assert set(seen) == set(neighbors)
    assert all(set(x) <= allowed for x in seen)
    assert all(l['status'] == 'locally_converged' for l in result['lanes'])


def test_retains_stable_and_publication_starts_not_only_cheapest():
    items = [candidate((0,), 0), candidate((1,), 1), candidate((2,), 10, stable=True),
             candidate((3,), 20, source='publication_seed')]
    result = refine(items, lambda seed: None, {0, 1, 2, 3}, budget=100, starts=3)
    assert [lane['seed_fingerprint'] for lane in result['lanes']] == [items[i]['row']['hard_fingerprint'] for i in (0, 2, 3)]


def test_context_restores_on_exception_and_disabled_by_default():
    assert REFINEMENT_BUDGET.get() is None
    with pytest.raises(RuntimeError):
        with full_score_refinement_policy(budget=9):
            assert REFINEMENT_BUDGET.get()['budget'] == 9
            raise RuntimeError()
    assert REFINEMENT_BUDGET.get() is None
    with pytest.raises(ValueError):
        with full_score_refinement_policy(budget=0):
            pass


def test_fixed_tail_cache_includes_suffix_and_passes_parent():
    left = candidate((0, 2), 0.)
    right = candidate((0, 2), 1., stable=True, tag='other_tail')
    left['tail_fingerprint'] = 'left'
    right['tail_fingerprint'] = 'right'
    seen = []
    def reject(seed):
        seen.append((tuple(seed['offsets']), seed['tail_parent']['tail_fingerprint']))
        return None
    result = refine([left, right], reject, {0, 1, 2}, budget=100, starts=2, fixed_tail=True)
    assert len(seen) == len(set(seen))
    assert {key for key, tail in seen if tail == 'left'} == {key for key, tail in seen if tail == 'right'}
    assert result['version'] == 'fixed_tail_calendar_descent_v1'
    assert all(l['status'] == 'locally_converged' for l in result['lanes'])


def test_fixed_tail_policy_restores_after_exception():
    with pytest.raises(RuntimeError):
        with full_score_refinement_policy(budget=80, starts=2, fixed_tail=True):
            assert REFINEMENT_BUDGET.get()['fixed_tail']
            raise RuntimeError('fixture')
    assert REFINEMENT_BUDGET.get() is None


def test_calendar_budget_is_shared_between_large_neighborhoods():
    items=[candidate((i,),float(i),stable=i==1,source='publication_seed' if i==2 else 'seed')
           for i in range(3)]
    calls=[]
    def proposals(current,allowed):
        origin=current['score'].offsets[0]
        for j in range(10):
            yield (origin,j),dict(origin=origin)
    result=refine(items,lambda seed:calls.append(seed['origin']),set(range(3)),
                  budget=8,starts=3,proposal_adapter=proposals)
    assert calls==[0,1,2,0,1,2,0,1]
    assert all(l['status']=='budget_exhausted' for l in result['lanes'])
    assert [l['neighborhoods'][0]['evaluations'] for l in result['lanes']]==[3,3,2]
