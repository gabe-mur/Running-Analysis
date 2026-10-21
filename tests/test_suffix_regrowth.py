from copy import deepcopy
from types import SimpleNamespace

import pytest

from scripts import suffix_regrowth_prototype as module
from test_typed_continuation import fixture


def row(offsets,cost,**extras):
    return dict(offsets=offsets,total_cost=cost,roles={},allocation_trace=[],**extras)


def test_seed_diversity_not_just_cheapest_remote_suffixes():
    rows=[row([0,2,4],1),row([0,2,6],2),row([0,3,5],3),row([1,4,6],4)]
    record=dict(evaluated=rows)
    original=deepcopy(record)
    assert module.diverse_seeds(record,limit=3)==[rows[0],rows[2],rows[3]]
    assert record==original


def test_seed_filter_keeps_role_lanes_and_excludes_invalid_programs():
    a=row([0,2,4],1)
    b=dict(a,roles={'0':'long'},total_cost=2)
    invalid=[row([1,4],0,acceptance={'rejected':True}),
             row([2,5],0,unrepresentable_distance_target='no dose'),row([3,5],float('inf'))]
    assert module.diverse_seeds(dict(evaluated=invalid+[a,b]))==[a,b]


@pytest.mark.parametrize('cuts',[(),(0,),(21,),(4,4)])
def test_invalid_cuts_fail_before_generation(cuts):
    with pytest.raises(ValueError):
        module.propose(dict(daily_states=[None]*21),{},None,cuts=cuts)


def setup(monkeypatch):
    lookup,make=fixture()
    days,states=make(10,{0:('easy',3.3),2:('easy',3.3),6:('easy',3.5)})
    inputs=dict(daily_states=states,forced_rest_offsets={5},completed_run_offsets={8},
                config={},request=object(),expected_target_projector=lambda p,n:[(17.,18.)]*n,
                target_evidence=SimpleNamespace(planning_mode=SimpleNamespace(value='established')))
    seed=row([0,2,6],1.)
    hard=SimpleNamespace(total_cost=1.,model_dump=lambda **kw:seed)
    calls=[]
    def hard_score(inputs,offsets,**kwargs):
        calls.append((tuple(offsets),kwargs))
        return days,hard,{'aerobic_acceptance_facts':dict(sessions=[],target_trajectory=[],
            ordinary_miles=3.5)}
    monkeypatch.setattr(module,'hard_score',hard_score)
    monkeypatch.setattr(module,'raw_state_lookup',lambda *a:lookup)
    return inputs,seed,days,states,calls


def test_regrowth_blocks_unavailable_dates_and_reallocates_current_prefix(monkeypatch):
    inputs,seed,days,states,calls=setup(monkeypatch)
    def rollout(prefix,raw,request,config,targets,**kw):
        assert prefix==days[:4] and raw==states[:4]
        assert kw['role_overrides']=={states[5].as_of.date():'rest',states[8].as_of.date():'rest'}
        assert kw['continuation_days']==6
        return days,states,[]
    monkeypatch.setattr(module,'rollout',rollout)
    rows,programs,metrics=module.propose(inputs,dict(evaluated=[seed]),{},cuts=(4,))
    assert len(rows)==1 and metrics['materialized']==1
    assert calls[-1][1]['roles']=={0:'easy',2:'easy'}
    assert len(programs)==1


def test_bad_rollout_cannot_smuggle_run_onto_blocked_day(monkeypatch):
    inputs,seed,days,states,_=setup(monkeypatch)
    _,make=fixture()
    bad,raw=make(10,{0:('easy',3.3),2:('easy',3.3),5:('easy',3.5)})
    monkeypatch.setattr(module,'rollout',lambda *a,**kw:(bad,raw,[]))
    with pytest.raises(ValueError,match='blocked'):
        module.propose(inputs,dict(evaluated=[seed]),{},cuts=(4,))


def test_race_request_fails_explicitly_not_silently_dropped(monkeypatch):
    monkeypatch.setattr(module.planner,'configured_race_goal',lambda config:('race','date'))
    with pytest.raises(ValueError,match='Race-date'):
        module.propose(dict(daily_states=[None]*21,config={}),{},None)


def test_seed_diversity_retains_same_dates_with_different_materialized_roles():
    easy=dict(row([0,2,3,4],1.),allocation_trace=['0:easy_run:easy:3.3','3:easy_run:easy:3.3'])
    quality=dict(row([0,2,3,6],2.),allocation_trace=['0:easy_run:easy:3.3','3:quality_run:intervals:4.5'])
    assert module.diverse_seeds(dict(evaluated=[easy,quality]))==[easy,quality]


def test_reserved_slots_preserve_lanes_and_restore_on_error():
    from scripts import continuation_replan
    original=continuation_replan.shortlist
    independent=row([0,2,4],1.)
    stable=dict(row([0,3,5],2.),roles={0:'easy'})
    regrown=[row([1,4,6],4.,proposal_source='suffix_regrowth'),
             row([2,4,6],5.,proposal_source='suffix_regrowth')]
    record=dict(evaluated=[independent,stable,row([0,2,5],3.),*regrown],
                unconstrained=independent,stable=stable)
    with pytest.raises(RuntimeError):
        with module.regrowth_finalist_policy():
            selected=continuation_replan.shortlist(record,limit=6)
            assert selected[:2]==[independent,stable]
            assert all(r in selected for r in regrown)
            assert len(selected)<=5
            raise RuntimeError('restore')
    assert continuation_replan.shortlist is original


def test_reservation_is_noop_without_eligible_regrowth():
    from scripts import continuation_replan
    record=dict(evaluated=[row([0,2],1.),row([0,3],2.)])
    expected=continuation_replan.shortlist(record,limit=6)
    with module.regrowth_finalist_policy():
        assert continuation_replan.shortlist(record,limit=6)==expected


def test_daily_hook_uses_current_history_preserves_input_and_restores(monkeypatch):
    from scripts import continuation_replan
    from run_analysis.selection_objective import COMPLETED_SELECTION_HISTORY
    evidence={'source':'test','runs':('current',)}
    token=COMPLETED_SELECTION_HISTORY.set(evidence)
    record=dict(evaluated=[row([0,2],1.)])
    original_record=deepcopy(record)
    new=row([0,3],2.,proposal_source='suffix_regrowth')
    def propose(inputs,supplied_record,supplied_history):
        assert supplied_record is record
        assert supplied_history is evidence
        return [new],{},dict(seconds=.1)
    def select(days,diagnostics,inputs,augmented,**kwargs):
        assert augmented['evaluated']==record['evaluated']+[new]
        assert kwargs==dict(limit=6,continuation_days=28)
        return {'evidence':{}}
    monkeypatch.setattr(module,'propose',propose)
    monkeypatch.setattr(continuation_replan,'select_program',select)
    original_shortlist=continuation_replan.shortlist
    try:
        with module.regrowth_replan_policy() as records:
            result=continuation_replan.select_program([],None,{},record,limit=6,continuation_days=28)
            assert result['evidence']['suffix_regrowth'] is records[0]
            assert records[0]['rows']==[new]
        assert continuation_replan.select_program is select
        assert continuation_replan.shortlist is original_shortlist
        assert record==original_record
    finally:
        COMPLETED_SELECTION_HISTORY.reset(token)


def test_daily_hook_missing_history_restores_without_search(monkeypatch):
    from scripts import continuation_replan
    original=continuation_replan.select_program
    with pytest.raises(ValueError,match='completed-history'):
        with module.regrowth_replan_policy():
            continuation_replan.select_program([],None,{}, {},limit=6,continuation_days=28)
    assert continuation_replan.select_program is original
