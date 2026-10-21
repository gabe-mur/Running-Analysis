import json
from pathlib import Path

import pytest

from run_analysis.adaptive.continuous_search_adapter import rebase_edit, search
from run_analysis.adaptive import full_score_refinement, finalized_distance_search


def test_rebase_preserves_unrelated_winner_changes_and_applies_whole_swap():
    source=(('a','easy:3.2'),('b','quality'),('c','easy'))
    edit=(('a','easy:3.2'),('b','easy'),('c','quality'))
    winner=(('a','easy:3.1'),('b','quality'),('c','easy'))
    assert rebase_edit(source,edit,winner)==(('a','easy:3.1'),('b','easy'),('c','quality'))


@pytest.mark.parametrize('winner', [
    (('a','long'),('b','quality')),  # changed dose at a touched date
    (('b','quality'),),            # source run deleted
    (('a','easy'),('b','quality'),('c','long')),  # insertion destination occupied
])
def test_rebase_rejects_conflicting_preconditions(winner):
    assert rebase_edit((('a','easy'),('b','quality')),
                       (('b','quality'),('c','easy')),winner) is None


def test_rebase_can_delete_and_insert_without_overwriting_another_date():
    assert rebase_edit((('a','easy'),('b','quality')), (('a','easy'),),
        (('a','long'),('b','quality')))==(('a','long'),)
    assert rebase_edit((('a','easy'),), (('a','easy'),('c','quality')),
        (('a','long'),))==(('a','long'),('c','quality'))
    assert rebase_edit((('a','easy'),),(('a','easy'),),(('a','long'),)) is None


@pytest.mark.parametrize('combined_cost,eligible,expected', [(1.,True,1.),(9.,True,3.),(1.,False,3.)])
def test_search_reprices_combination_and_never_trusts_old_gain(monkeypatch,combined_cost,eligible,expected):
    dates=('2026-10-01','2026-10-02','2026-10-03','2026-10-04','2026-10-05')
    source=((dates[1],'easy'),(dates[3],'long'))
    distance=((dates[1],'short'),(dates[3],'long'))
    moved=((dates[1],'easy'),(dates[4],'long'))
    combined=((dates[1],'short'),(dates[4],'long'))
    monkeypatch.setattr(full_score_refinement,'refine',lambda *a,**k:dict(lanes=[]))
    monkeypatch.setattr(finalized_distance_search,'refine_distances',lambda *a,**k:dict(lanes=[]))
    calls=[]
    def evaluate(p):
        calls.append(p)
        return dict(cost={source:5.,distance:3.,moved:4.,combined:combined_cost}.get(p,20.),
            eligible=p!=combined or eligible)
    result=search(dates=dates,keys=('easy','short','long'),previous=source,
        roles={'easy':'easy','short':'easy','long':'long'},seed_keys=('easy',),
        budget=500,hard_end=dates[-1],evaluate=evaluate)
    assert combined in calls
    assert result['cost']==expected
    trace=result['refinement']['promising_edit_reuse']['passes'][0]
    assert trace['before_cost']==3.
    assert trace['after_cost']==expected
    assert trace['accepted']==(1 if expected==1. else 0)
    assert trace['evaluations']<=32
    assert len(calls)==len(set(calls))==result['evaluations']<=500
    assert result['refinement']['global_new_by_source']['rebased_promising_edit']>0


def test_real_last_step_rebases_to_the_independently_repriced_program():
    root=Path(__file__).resolve().parents[1]/'output'
    trace_path=root/'search-fair-nearby-10k-20260928/report.json'
    audit_path=root/'search-witness-final-check-20260928/report.json'
    if not trace_path.exists() or not audit_path.exists():
        pytest.skip('Optional archived real-run diagnostic not installed')
    trace=json.loads(trace_path.read_text())['search']
    audit=json.loads(audit_path.read_text())['traced_checks']
    origin=next(s['best_program'] for s in trace['refinement']['final_joint'] if s['phase']=='final_check')
    actual=rebase_edit(origin,audit[0]['best']['program'],trace['program'])
    assert actual==tuple(map(tuple,audit[1]['best']['program']))
    assert audit[1]['best']['eligible']
    assert audit[1]['best']['cost']<trace['cost']
