from itertools import combinations
from copy import deepcopy

import pytest

from scripts.calendar_neighbor_prototype import calendar_neighbors
from scripts.search_landscape import edit_distance, panel, extend_panel, rank, correlation, summarize


def test_edit_distance_matches_complete_one_edit_graph():
    allowed=set(range(5))
    candidates=[c for n in range(1,6) for c in combinations(sorted(allowed),n)]
    for a in candidates:
        neighbors=set(calendar_neighbors(a,allowed))
        assert edit_distance(a,a)==0
        for b in candidates:
            assert edit_distance(a,b)==edit_distance(b,a)
            assert (edit_distance(a,b)==1)==(b in neighbors)
    assert edit_distance((0,),(4,))==2  # remove+add cheaper than four day moves


def test_panel_exhausts_both_neighborhoods_and_is_reproducible():
    allowed={0,1,3,4,5,6}
    a,b=(0,3,5),(1,4)
    selected,populations=panel(a,b,allowed,per_count=3)
    assert (selected,populations)==panel(a,b,allowed,per_count=3)
    rows=dict(selected)
    for label,seed in (('full_winner',a),('cheap_winner',b)):
        assert label in rows[seed]
        for candidate in calendar_neighbors(seed,allowed):
            assert label+'_neighbor' in rows[candidate]
    assert all(set(offsets)<=allowed for offsets,_ in selected)
    assert all(len([x for x,labels in selected if len(x)==n and 'stratified_control' in labels])==min(3,total)
               for n,total in populations.items())


def test_tied_rank_and_correlation():
    assert rank([4,2,2,8])==[2.,.5,.5,3.]
    assert correlation(rank([1,2,3]),rank([3,2,1]))==pytest.approx(-1.)
    assert correlation([1,1],[2,3]) is None
    assert correlation([],[]) is None


def test_map_summary_keeps_infeasible_low_scores_out_of_best_and_exposes_reversal():
    rows=[dict(offsets=(0,2),hard_cost=1.,full_cost=9.,eligible=True,status='eligible',
               sources=['full_winner'],prescriptions=[]),
          dict(offsets=(0,3),hard_cost=2.,full_cost=1.,eligible=True,status='eligible',
               sources=['full_winner_neighbor'],prescriptions=[]),
          dict(offsets=(0,4),hard_cost=3.,full_cost=2.,eligible=True,status='eligible',
               sources=['stratified_control'],prescriptions=[]),
          dict(offsets=(1,),hard_cost=0.,full_cost=0.,eligible=False,
               status='sequence_or_gate_rejection',sources=['stratified_control'],prescriptions=[])]
    result=summarize(rows)
    assert result['best']['offsets']==(0,3)
    assert result['eligible']==3 and result['full_scored']==4
    assert result['full_winner']['improving_neighbors']==1
    assert result['local_smoothness']['strict_order_reversals']==1
    assert result['low_score_clusters'][0]['component_sizes']==[2]
    assert result['strata']['stratified_control']['sampled']==2
    assert result['strata']['stratified_control']['eligible']==1
    assert summarize(rows)==result


def test_empty_eligible_map_is_not_a_pass():
    result=summarize([dict(status='hard_acceptance_rejection',eligible=False)])
    assert result['eligible']==0 and 'best' not in result


def test_invalid_control_count_rejected():
    with pytest.raises(ValueError):
        panel((0,),(0,),{0,1},per_count=-1)


def test_refinement_exhausts_neighbors_and_reuses_without_mutating_original():
    original=dict(rows=[dict(offsets=[0,2],sources=['full_winner'],full_cost=10.),
                        dict(offsets=[0,3],sources=['full_winner_neighbor'],full_cost=5.)],
                  summary=dict(best=dict(offsets=[0,3])))
    saved=deepcopy(original)
    candidates,cache,seed=extend_panel(original,{0,1,2,3,4})
    assert seed==(0,3)
    assert len(cache)==2
    mapping=dict(candidates)
    assert 'refined_winner' in mapping[seed]
    assert all('refined_winner_neighbor' in mapping[x] for x in calendar_neighbors(seed,{0,1,2,3,4}))
    cache[(0,3)]['sources'].append('changed')
    assert original==saved


def test_unobserved_refinement_seed_rejected():
    with pytest.raises(ValueError,match='evaluated'):
        extend_panel(dict(rows=[],summary=dict(best=dict(offsets=[0]))),{0,1})
