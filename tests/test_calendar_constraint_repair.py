"""Changed calendar constraints repair candidates, never change their scores."""
from datetime import date,datetime,timedelta,timezone
from types import SimpleNamespace as NS
from collections import Counter

import pytest

from run_analysis.adaptive.calendar_constraint_repair import (
    constraint_repair_neighbors,role_date_domains,
)
from run_analysis.adaptive.continuous_search_adapter import search,NoEligibleProgram,fingerprint
from run_analysis.program_constraints import DateRestriction

DATES=tuple((date(2026,10,18)+timedelta(days=i)).isoformat() for i in range(10))
ROLES={'e':'easy','q':'quality','l':'long'}
PRIOR=((DATES[0],'e'),(DATES[2],'l'),(DATES[4],'e'),(DATES[5],'q'),(DATES[6],'e'))


def allowed(blocked=(DATES[0],)):
    return {r:frozenset(set(DATES)-set(blocked)) for r in ROLES.values()}


def test_deletion_and_nearby_moves_preserve_every_legal_slot_and_dose():
    rows=constraint_repair_neighbors(PRIOR,DATES,ROLES,allowed())
    assert rows[0]==PRIOR[1:]
    assert tuple(sorted(((DATES[1],'e'),*PRIOR[1:]))) in rows
    assert tuple(sorted(((DATES[3],'e'),*PRIOR[1:]))) in rows
    for p in rows:
        assert dict(PRIOR[1:]).items()<=dict(p).items()
        assert DATES[0] not in dict(p)
        assert Counter(k for _,k in p) in (Counter(k for _,k in PRIOR),Counter(k for _,k in PRIOR[1:]))
    assert rows==constraint_repair_neighbors(reversed(PRIOR),reversed(DATES),ROLES,allowed())


def test_multiple_conflicts_are_repaired_together_and_deletion_cannot_be_pruned():
    domains=allowed((DATES[0],DATES[2]))
    rows=constraint_repair_neighbors(PRIOR,DATES,ROLES,domains,limit=8)
    assert rows[0]==PRIOR[2:]
    assert any(len(p)==len(PRIOR) for p in rows)
    assert len(rows)<=8
    assert all(all(d in domains[ROLES[k]] for d,k in p) for p in rows)
    assert all(len(dict(p))==len(p) for p in rows)
    for p in rows:
        assert dict(PRIOR[2:]).items()<=dict(p).items()


def test_occupied_preferred_weekday_can_swap_without_losing_other_runs():
    prior=((DATES[0],'l'),(DATES[1],'e'),(DATES[5],'q'))
    domains={'long':frozenset((DATES[1],))}
    rows=constraint_repair_neighbors(prior,DATES,ROLES,domains)
    assert ((DATES[0],'e'),(DATES[1],'l'),(DATES[5],'q')) in rows
    assert all(Counter(k for _,k in p)<=Counter(k for _,k in prior) for p in rows)
    assert all(len(dict(p))==len(p) for p in rows)


def test_no_conflict_no_repair_and_no_far_or_forbidden_move():
    assert constraint_repair_neighbors(PRIOR,DATES,ROLES,{})==()
    rows=constraint_repair_neighbors(PRIOR,DATES,ROLES,allowed(),radius_days=1)
    assert rows==(PRIOR[1:],tuple(sorted(((DATES[1],'e'),*PRIOR[1:]))))
    assert constraint_repair_neighbors(PRIOR,DATES,ROLES,allowed(),limit=1)==(PRIOR[1:],)
    with pytest.raises(ValueError):
        constraint_repair_neighbors(PRIOR,DATES,ROLES,allowed(),limit=0)


def test_domains_use_only_known_restrictions_and_intersect_long_weekday():
    at=datetime(2026,10,18,tzinfo=timezone.utc)
    restrictions=tuple(DateRestriction(str(i),date.fromisoformat(DATES[i]),date.fromisoformat(DATES[i+1]),
        at+timedelta(seconds=1 if i==2 else 0),kind,'test')
        for i,kind in enumerate(('reserved_rest','health_review','unsafe_weather')))
    domains=role_date_domains(DATES,ROLES,restrictions=restrictions,as_of=at,long_dates=set(DATES[:4]))
    assert domains['easy']==frozenset(DATES[2:])
    assert domains['long']==frozenset(DATES[2:4])
    assert role_date_domains(DATES,ROLES,restrictions=(),as_of=at)=={}


@pytest.mark.parametrize('winner',['repair','independent'])
def test_actual_search_scores_repairs_without_forcing_them_or_exceeding_budget(winner):
    clean=PRIOR[1:];target=tuple(sorted(((DATES[1],'e'),*clean)))
    independent=((DATES[1],'q'),(DATES[8],'l'))
    calls=[]
    def evaluate(p):
        assert DATES[0] not in dict(p)
        calls.append(p)
        cost=(0. if p==(target if winner=='repair' else independent) else
            1. if p in (target,independent) else 2.)
        return dict(cost=cost,eligible=bool(p),construction_eligible=True)
    result=search(dates=DATES,keys=tuple(ROLES),previous=PRIOR,evaluate=evaluate,budget=80,
        hard_end=DATES[7],roles=ROLES,allowed_role_dates=allowed(),
        seed_programs=(independent,),complete_first=True)
    assert result['program']==(target if winner=='repair' else independent)
    assert clean in calls and target in calls and independent in calls
    report=result['prior_constraint_repair']
    assert report['admitted_candidates']>0
    assert fingerprint(target) in report['fingerprints']
    assert report['evaluations']<=report['maximum_new_evaluations']
    assert len(calls)==len(set(calls))==result['evaluations']<=80
    assert result['selected']['cost']==0.
    assert result['continuity']['tolerance'] is None


def test_repaired_calendar_cannot_bypass_full_evaluator_rejection():
    calls=[]
    def evaluate(p):
        calls.append(p)
        return dict(cost=0.,eligible=False,construction_eligible=False)
    with pytest.raises(NoEligibleProgram):
        search(dates=DATES,keys=tuple(ROLES),previous=PRIOR,evaluate=evaluate,budget=40,
            hard_end=DATES[7],roles=ROLES,allowed_role_dates=allowed(),complete_first=True)
    assert PRIOR[1:] in calls


def test_valid_repair_does_not_disable_independent_beam_fallback():
    result=search(dates=DATES,keys=tuple(ROLES),previous=PRIOR,
        evaluate=lambda p:dict(cost=abs(len(p)-4),eligible=bool(p),construction_eligible=True),
        budget=100,hard_end=DATES[7],roles=ROLES,allowed_role_dates=allowed(),
        complete_first=True)
    assert result['prior_constraint_repair']['admitted_candidates']>0
    assert result['calendar_beam']['skipped'] is False
    assert result['evaluations']<=100


def test_coached_dose_funding_skips_known_date_conflict_instead_of_growing_easy():
    from run_analysis.adaptive.coached_pipeline import fund_bound_prior
    at=datetime(2026,10,18,tzinfo=timezone.utc)
    r=DateRestriction('rest',date.fromisoformat(DATES[0]),date.fromisoformat(DATES[1]),at,'reserved_rest','user')
    calls=[]
    def evaluate(p):
        calls.append(p)
        return dict(cost=0.,eligible=False,constraints='constraint_conflict')
    p,report=fund_bound_prior(PRIOR,dates=DATES,catalog={},physical={},
        state=NS(restrictions=(r,),as_of=at),evaluate=evaluate)
    assert p==PRIOR and calls==[PRIOR]
    assert report['evaluations']==1
    assert report['reason']=='date_constraint_needs_calendar_repair'


def test_actual_harness_passes_all_role_domains_without_publishing(tmp_path,monkeypatch):
    from test_adaptive_stage_lifecycle import completed_stage
    import run_analysis.adaptive.new_model_daily_harness as module
    h,_,_,at=completed_stage(tmp_path,monkeypatch)
    blocked=h.decision(at)+timedelta(days=1)
    h.restrictions=(DateRestriction('new-rest',blocked,blocked+timedelta(days=1),at,'reserved_rest','user'),)
    seen=[]
    class SearchBoundaryReached(Exception):
        pass
    def capture(**kwargs):
        seen.append(kwargs['allowed_role_dates'])
        raise SearchBoundaryReached
    monkeypatch.setattr(module,'migrated_search',capture)
    before=tuple(h.events),tuple(h.history),h.stage
    try:
        with pytest.raises(SearchBoundaryReached):
            h.replan(at,reason='constraint domain wiring',publish=False)
        assert seen and set(seen[0])=={'easy','long','quality'}
        assert all(blocked.isoformat() not in domain for domain in seen[0].values())
        assert (tuple(h.events),tuple(h.history),h.stage)==before
    finally:
        h.sandbox.close()
