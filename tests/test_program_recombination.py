import pytest

from run_analysis.adaptive.program_recombination import recombinations
from run_analysis.adaptive.continuous_search_adapter import search


def test_half_open_ownership_both_directions_and_no_parent_mutation():
    left=(('2026-10-01','a'),('2026-10-03','b'))
    right=(('2026-10-02','c'),('2026-10-03','d'))
    result=list(recombinations((left,right),('2026-10-03',)))
    assert {p for p,_ in result}=={
        (('2026-10-01','a'),('2026-10-03','d')),
        (('2026-10-02','c'),('2026-10-03','b'))}
    assert left==(('2026-10-01','a'),('2026-10-03','b'))


def test_duplicate_parents_cuts_and_original_programs_not_repriced():
    left=(('2026-10-01','a'),); right=(('2026-10-03','b'),)
    result=list(recombinations((left,right,left),('2026-10-02','2026-10-02')))
    assert {p for p,_ in result}=={(),(*left,*right)}
    with pytest.raises(ValueError):
        list(recombinations(((('2026-10-01','a'),('2026-10-01','b')),),('2026-10-02',)))


@pytest.mark.parametrize('prefer_prior_prefix,eligible',[(True,True),(False,True),(True,False)])
def test_selector_prices_both_directions_and_rejects_invalid_cheap_splices(prefer_prior_prefix,eligible):
    dates=tuple(f'2026-10-{i:02d}' for i in range(1,5))
    left=tuple(zip(dates,('easy','easy','long','long')))
    right=tuple(zip(dates,('long','long','easy','easy')))
    target=tuple((d,'easy' if prefer_prior_prefix else 'long') for d in dates)
    calls=[]
    def evaluate(p):
        calls.append(p)
        return dict(cost=0. if p==target else 5. if p==left else 6. if p==right else 10.,
                    eligible=p!=target or eligible)
    result=search(dates=dates,keys=('easy','long'),roles={'easy':'easy','long':'long'},
        previous=left,seed_programs=(right,),seed_keys=('long' if prefer_prior_prefix else 'easy',),hard_end=dates[-1],
        budget=500,evaluate=evaluate,recombination_dates=(dates[2],))
    assert result['program']==(target if eligible else left)
    assert result['refinement']['global_new_by_source'].get('prefix_suffix_recombination',0)>0
    assert result['continuity']['tolerance'] is None
    assert len(calls)==len(set(calls))==result['evaluations']<=500
    if eligible:
        # A newly recombined winner inherits no certificate from its parents.
        assert not result['local_neighborhood_complete']
