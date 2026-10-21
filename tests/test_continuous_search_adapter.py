from itertools import product

import pytest

from scripts import full_score_refinement, finalized_distance_search
from scripts.continuous_search_adapter import search, fingerprint
from scripts.continuous_search_adapter import NoEligibleProgram


DATES = ('2026-09-27','2026-09-28','2026-09-29')
ROLES = {'short':'easy','easy':'easy','long':'long'}


def test_coaching_transform_is_scored_before_cache_and_physical_selection():
    roles = {'easy':'easy', 'qcheap':'quality', 'qhard':'quality'}
    calls = []
    def bind(p):
        return tuple((d, 'qhard' if roles[k] == 'quality' else k) for d, k in p)
    def evaluate(p):
        calls.append(p)
        assert all(k != 'qcheap' for _, k in p)
        return dict(cost=0. if p == ((DATES[1], 'qhard'),) else 10., eligible=True)
    result = search(dates=DATES, keys=tuple(roles), roles=roles, previous=((DATES[1], 'qcheap'),),
        evaluate=evaluate, budget=100, hard_end=DATES[-1], program_transform=bind)
    assert result['program'] == ((DATES[1], 'qhard'),)
    assert result['cost'] == 0.
    assert calls[0] == result['program']
    assert len(calls) == len(set(calls)) == result['evaluations']
    assert result['coaching_program_transform']
    assert not result['local_neighborhood_complete']


@pytest.mark.parametrize('transform', [lambda p: tuple((DATES[2], k) for _, k in p),
    lambda p: tuple((d, 'easy') for d, k in p)])
def test_coaching_transform_cannot_silently_move_dates_or_change_roles(transform):
    with pytest.raises(ValueError, match='preserve dates and roles'):
        search(dates=DATES, keys=('easy','q'), roles={'easy':'easy','q':'quality'},
            previous=((DATES[0],'q'),), evaluate=lambda p:dict(cost=0.,eligible=True),
            budget=20, hard_end=DATES[-1], program_transform=transform)


def test_coaching_transform_must_be_idempotent():
    def flip(p):
        return tuple((d,'q2' if k=='q1' else 'q1') for d,k in p)
    with pytest.raises(ValueError, match='idempotent'):
        search(dates=DATES, keys=('q1','q2'), roles={'q1':'quality','q2':'quality'},
            previous=((DATES[0],'q1'),), evaluate=lambda p:dict(cost=0.,eligible=True),
            budget=20, hard_end=DATES[-1], program_transform=flip)


def test_coached_terminal_dates_reprice_the_final_dose_without_extra_allowance(monkeypatch):
    import run_analysis.adaptive.continuous_search_adapter as module
    dates=(*DATES,'2026-09-30')
    roles=dict(q='quality',e1='easy',e2='easy')
    seed=((dates[0],'q'),(dates[3],'e1'))
    refined=((dates[1],'q'),(dates[2],'e2'))
    moved=((dates[0],'q'),(dates[2],'e2'))
    # The late dose is first discovered by final recombination, AFTER the main
    # calendar checks. Only the terminal date sweep can find this improvement.
    monkeypatch.setattr(module,'recombinations',lambda *a:iter(((refined,'fixture'),)))
    for operator in ('calendar_neighbors','opening_opportunity_neighbors','paired_move_neighbors',
                     'equal_volume_neighbors','decimal_merge_neighbors','decimal_pair_neighbors',
                     'quality_easy_neighbors'):
        monkeypatch.setattr(module,operator,lambda *a,**k:iter(()))
    monkeypatch.setattr(full_score_refinement,'refine',lambda *a,**k:dict(lanes=[]))
    monkeypatch.setattr(finalized_distance_search,'refine_distances',lambda *a,**k:dict(lanes=[]))
    calls=[]
    def evaluate(p):
        calls.append(p)
        return dict(cost=0. if p==moved else 1. if p==refined else 2. if p==seed else 10.,
            eligible=p in (seed,refined,moved))
    result=search(dates=dates,keys=tuple(roles),roles=roles,previous=seed,
        seed_programs=(seed,),evaluate=evaluate,budget=200,hard_end=dates[-1],
        recombination_dates=(dates[2],),dose_miles=dict(q=3.,e1=1.,e2=2.),
        complete_first=True,final_winner_budget=10,program_transform=lambda p:p)
    assert result['program']==moved and result['cost']==0.
    assert len(calls)==len(set(calls))==result['evaluations']<=210
    assert result['refinement']['global_new_by_source']['final_winner_calendar']>0
    assert not result['local_neighborhood_complete']


@pytest.mark.parametrize('supplied',[False,True])
def test_funded_search_constructs_prefixes_but_only_selects_full_programs(supplied):
    seed=((DATES[0],'easy'),(DATES[2],'easy'))
    calls=[]
    def evaluate(p):
        assert len(dict(p))==len(p)
        calls.append(p)
        return dict(cost=float(len(p)),eligible=len(p)>=2,construction_eligible=True)
    result=search(dates=DATES,keys=('easy',),roles={'easy':'easy'},hard_end=DATES[-1],
        previous=(),budget=100,evaluate=evaluate,seed_programs=(seed,) if supplied else ())
    assert len(result['program'])==2
    assert result['cost']==2.  # Exhaustive optimum: two of three available dates.
    assert result['selected']['eligible']
    assert len(calls)==len(set(calls))==result['evaluations']


def test_complete_seeds_never_enter_prefix_beam_even_without_construction_contract():
    seed=((DATES[0],'easy'),(DATES[2],'easy'))
    result=search(dates=DATES,keys=('easy',),roles={'easy':'easy'},hard_end=DATES[-1],
        previous=(),budget=100,seed_programs=(seed,),
        evaluate=lambda p:dict(cost=float(len(p)),eligible=len(p)>=2))
    assert len(result['program'])==2


def test_complete_first_reaches_neighbors_before_exhausting_supplied_library():
    from datetime import date,timedelta
    dates=tuple((date(2026,10,1)+timedelta(days=i)).isoformat() for i in range(14))
    seeds=tuple(((a,'easy'),(b,'easy')) for i,a in enumerate(dates) for b in dates[i+1:])
    target=((dates[0],'easy'),(dates[2],'easy'))
    calls=[]
    def evaluate(p):
        calls.append(p)
        return dict(cost=sum(dict(p).get(d)!=dict(target).get(d) for d in dates),
            eligible=len(p)>=2,construction_eligible=True)
    result=search(dates=dates,keys=('easy',),roles={'easy':'easy'},hard_end=dates[-1],
        previous=(),budget=500,evaluate=evaluate,seed_programs=seeds,complete_first=True)
    opening=result['refinement']['opening']
    assert opening['examined_seeds']==opening['seed_limit']==25<len(seeds)
    assert opening['evaluations_before_refinement']<=28
    assert result['calendar_beam']['skipped']
    assert not result['calendar_beam']['complete']  # No false coverage claim.
    assert result['program']==target and result['cost']==0.
    assert len(calls)==len(set(calls))==result['evaluations']<=500


def test_complete_first_keeps_beam_when_supplied_starts_are_rejected():
    invalid=((DATES[0],'long'),)
    result=search(**options(),seed_programs=(invalid,),complete_first=True,
        evaluate=lambda p:dict(cost=-len(p),eligible=all(k!='long' for _,k in p)))
    assert not result['calendar_beam']['skipped']
    assert result['calendar_beam']['completed_dates']>0
    assert all(k!='long' for _,k in result['program'])


def test_complete_first_seed_membership_does_not_depend_on_incumbent_cache_hit():
    from datetime import date,timedelta
    dates=tuple((date(2026,10,1)+timedelta(days=i)).isoformat() for i in range(14))
    seeds=tuple(((a,'easy'),(b,'easy')) for i,a in enumerate(dates) for b in dates[i+1:])
    result=search(dates=dates,keys=('easy',),roles={'easy':'easy'},hard_end=dates[-1],
        previous=seeds[0],budget=500,seed_programs=seeds,complete_first=True,
        evaluate=lambda p:dict(cost=float(len(p)),eligible=len(p)>=2,construction_eligible=True))
    assert result['refinement']['opening']['examined_seeds']==25
    assert result['refinement']['global_new_by_source']['supplied_independent_seed']==24


def test_final_native_check_rechecks_quality_easy_pairs_without_new_budget(monkeypatch):
    import run_analysis.adaptive.continuous_search_adapter as module
    roles={'q3':'quality','q32':'quality','e4':'easy','e38':'easy'}
    miles=dict(q3=3.,q32=3.2,e4=4.,e38=3.8)
    prior=((DATES[0],'q3'),(DATES[2],'e4'))
    wanted=((DATES[0],'q32'),(DATES[2],'e38'))
    # Isolate the terminal seam: earlier operators do not offer this witness.
    monkeypatch.setattr(module,'quality_easy_neighbors',lambda *a,**kw:
        iter((wanted,)) if kw.get('priority_before') is not None else iter(()))
    def evaluate(p):
        valid=p in (prior,wanted)
        return dict(cost=0. if p==wanted else 1.,eligible=valid,construction_eligible=valid)
    result=search(dates=DATES,keys=tuple(roles),roles=roles,dose_miles=miles,
        previous=prior,seed_programs=(prior,),evaluate=evaluate,budget=20,
        hard_end=DATES[-1],complete_first=True,final_winner_budget=5,
        fair_coordinates=True,stable_days=2)
    assert result['program']==wanted
    assert result['evaluations']<=25
    assert result['refinement']['global_new_by_source']['final_winner_quality_easy']==1


@pytest.mark.parametrize('allowance',[1,100])
def test_final_recombination_winner_gets_coupled_dose_recheck(monkeypatch,allowance):
    import run_analysis.adaptive.continuous_search_adapter as module
    dates=(*DATES,'2026-09-30')
    roles={k:'easy' for k in ('e1','e2','e3')}
    seed=((dates[0],'e3'),(dates[3],'e1'))
    splice=((dates[1],'e3'),(dates[2],'e1'))
    settled=((dates[1],'e2'),(dates[2],'e2'))
    monkeypatch.setattr(module,'recombinations',lambda *a:iter(((splice,'fixture'),)))
    for operator in ('calendar_neighbors','opening_opportunity_neighbors','paired_move_neighbors'):
        monkeypatch.setattr(module,operator,lambda *a,**k:iter(()))
    monkeypatch.setattr(module,'equal_volume_neighbors',lambda *a,**k:iter(()))
    monkeypatch.setattr(module,'decimal_merge_neighbors',lambda *a,**k:iter(()))
    monkeypatch.setattr(full_score_refinement,'refine',lambda *a,**k:dict(lanes=[]))
    monkeypatch.setattr(finalized_distance_search,'refine_distances',lambda *a,**k:dict(lanes=[]))
    def evaluate(p):
        return dict(cost=0. if p==settled else 1. if p==splice else 2. if p==seed else 10.,
            eligible=True)
    baseline=search(dates=dates,keys=tuple(roles),roles=roles,previous=seed,seed_programs=(seed,),
        evaluate=evaluate,budget=500,hard_end=dates[-1],recombination_dates=(dates[2],),
        dose_miles=dict(e1=1.,e2=2.,e3=3.),complete_first=True)
    result=search(dates=dates,keys=tuple(roles),roles=roles,previous=seed,seed_programs=(seed,),
        evaluate=evaluate,budget=500,hard_end=dates[-1],recombination_dates=(dates[2],),
        dose_miles=dict(e1=1.,e2=2.,e3=3.),complete_first=True,final_winner_budget=allowance)
    assert result['program']==settled and result['cost']==0.
    assert result['evaluations']<=500+allowance
    trace=result['refinement']['final_winner_doses']
    assert trace['additional_allowance']==allowance
    assert baseline['program']==splice
    assert trace['passes'][0]['before_cost']==baseline['cost']
    assert trace['passes'][0]['before_cost']==1.
    assert trace['passes'][0]['after_cost']==0.
    if trace['certified_fingerprint'] is not None:
        assert trace['certified_fingerprint']==fingerprint(settled)
        assert trace['passes'][-1]['scan_complete']
    assert not result['local_neighborhood_complete']  # Pair-only certificate.


def test_complete_first_fallback_can_find_later_admitted_complete_seed(monkeypatch):
    # Unfinished prefixes cannot establish admission; only a later whole-plan
    # seed does. The capped library is NOT an eligibility rule.
    from datetime import date,timedelta
    dates=tuple((date(2026,10,1)+timedelta(days=i)).isoformat() for i in range(14))
    seeds=tuple(((a,'easy'),(b,'easy')) for i,a in enumerate(dates) for b in dates[i+1:])
    target=seeds[-1]
    result=search(dates=dates,keys=('easy',),roles={'easy':'easy'},hard_end=dates[-1],
        previous=(),budget=500,seed_programs=seeds,complete_first=True,
        evaluate=lambda p:dict(cost=0.,eligible=p==target,construction_eligible=False))
    assert result['program']==target
    assert result['calendar_beam']['complete_seed_fallback']
    assert result['evaluations']<=500


def test_construction_respects_physical_rejection_and_never_publishes_underfunded():
    calls=[]
    def evaluate(p):
        calls.append(p)
        allowed=all(d==DATES[1] for d,k in p)
        return dict(cost=float(len(p)),eligible=allowed and len(p)>=2,
            construction_eligible=allowed)
    with pytest.raises(NoEligibleProgram):
        search(dates=DATES,keys=('easy',),roles={'easy':'easy'},hard_end=DATES[-1],
            previous=(),budget=100,evaluate=evaluate)
    # A forbidden prefix must never be expanded into a later larger candidate.
    assert not any(len(p)>1 and DATES[0] in dict(p) for p in calls)
    assert not any(len(p)>2 for p in calls)


@pytest.mark.parametrize('budget',[2,3,10])
def test_funded_construction_obeys_one_shared_evaluation_budget(budget):
    calls=[]
    def evaluate(p):
        calls.append(p)
        return dict(cost=float(len(p)),eligible=len(p)>=2,construction_eligible=True)
    try:
        result=search(dates=DATES,keys=('easy',),roles={'easy':'easy'},hard_end=DATES[-1],
            previous=(),budget=budget,evaluate=evaluate)
        assert result['selected']['eligible']
    except NoEligibleProgram:
        pass
    assert len(calls)==len(set(calls))<=budget


def options(**kwargs):
    return dict(dates=DATES,keys=tuple(ROLES),roles=ROLES,hard_end=DATES[-1],
                previous=(),budget=500,**kwargs)


def test_calls_existing_refiners_in_existing_pre_calendar_post_order(monkeypatch):
    calls = []
    original_calendar = full_score_refinement.refine
    original_distance = finalized_distance_search.refine_distances
    def calendar(*a,**k):
        calls.append(('calendar',k.get('proposal_adapter') is not None))
        return original_calendar(*a,**k)
    def distance(*a,**k):
        calls.append(('distance',k['seed_candidates'] is not None))
        return original_distance(*a,**k)
    monkeypatch.setattr(full_score_refinement,'refine',calendar)
    monkeypatch.setattr(finalized_distance_search,'refine_distances',distance)
    result = search(**options(),frontier_repair=True,
        evaluate=lambda p:dict(cost=1/(1+len(p)),eligible=True))
    assert calls==[('distance',True),('calendar',True),('distance',True),('distance',False)]
    assert result['search']=='reused_full_score_and_finalized_distance_refiners_v8_rebased_frontier'
    assert result['refinement']['pre_calendar_distance']['seed_mode']=='all_retained'
    assert result['local_neighborhood_complete']
    assert result['refinement']['final_joint'][-1]['complete']


def test_tiny_exhaustive_control_and_deterministic_results():
    target = {DATES[0]:'easy',DATES[2]:'long'}
    def cost(p):
        values = dict(p)
        return sum(values.get(d)!=target.get(d) for d in DATES)
    def evaluate(p):
        return dict(cost=cost(p),eligible=True)
    a,b = search(**options(),evaluate=evaluate),search(**options(),evaluate=evaluate)
    exhaustive = [tuple((d,k) for d,k in zip(DATES,ks) if k)
                  for ks in product((None,*ROLES),repeat=3)]
    assert a['program']==b['program']==tuple(target.items())
    assert a['cost']==b['cost']==min(map(cost,exhaustive))
    assert a['evaluations']==b['evaluations']


def test_prior_full_physical_program_and_tail_revalidated_before_exploration():
    prior = ((DATES[0],'short'),(DATES[2],'long'))
    calls = []
    def evaluate(p):
        calls.append(p)
        return dict(cost=0. if p==prior else 10.,eligible=True)
    result = search(**dict(options(),previous=prior),evaluate=evaluate)
    assert calls[0]==prior and result['program']==prior
    assert result['prior_cost']==0.
    assert result['prior_revalidation']['retained_as_candidate']
    assert result['prior_revalidation']['physical_fingerprint']==fingerprint(prior)
    assert len(calls)==len(set(calls))==result['evaluations']


def test_continuity_mechanism_uses_explicit_new_units_not_old_tolerance():
    prior = ((DATES[0],'easy'),)
    better = ((DATES[1],'easy'),)
    def evaluate(p):
        return dict(cost=.3 if p==prior else 0. if p==better else 10.,eligible=True)
    independent = search(**dict(options(),previous=prior),evaluate=evaluate)
    retained = search(**dict(options(),previous=prior),evaluate=evaluate,stability_tolerance=.5)
    assert independent['program']==better
    assert independent['continuity']['status']=='not_calibrated'
    assert retained['program']==prior
    assert retained['continuity']['status']=='explicit_scenario'


def test_different_allocations_and_tails_never_share_scores():
    left = ((DATES[0],'short'),(DATES[2],'long'))
    right = ((DATES[0],'short'),(DATES[2],'easy'))
    calls = []
    def evaluate(p):
        calls.append(p)
        return dict(cost=0. if p==right else 1. if p==left else 10.,eligible=True)
    result = search(**dict(options(),previous=left),evaluate=evaluate,seed_programs=(right,),stable_days=1)
    assert result['program']==right
    assert calls.count(left)==calls.count(right)==1
    assert len(calls)==len(set(calls))
    identities = {lane['seed_fingerprint'] for lane in result['refinement']['calendar']['lanes']}
    assert fingerprint(left) in identities and fingerprint(right) in identities


def test_new_constraints_reject_prior_instead_of_blind_continuity():
    prior = ((DATES[0],'long'),)
    result = search(**dict(options(),previous=prior),
        evaluate=lambda p:dict(cost=-len(p),eligible=all(d!=DATES[0] for d,k in p)))
    assert not result['prior_revalidation']['eligible']
    assert all(d!=DATES[0] for d,_ in result['program'])


@pytest.mark.parametrize('budget',[2,10,30])
def test_global_budget_counts_real_evaluator_calls_and_reports_exhaustion(budget):
    calls=[]
    def evaluate(p):
        calls.append(p)
        return dict(cost=-len(p),eligible=True)
    result=search(**dict(options(),budget=budget),evaluate=evaluate)
    assert result['evaluations']==len(calls)<=budget
    assert len(calls)==len(set(calls))
    assert result['budget_exhausted']
    final=result['refinement']['final_joint'][-1]
    assert result['local_neighborhood_complete']==(final['complete'] and final['cost']==final['best_cost'])


def test_new_origin_retains_unconsumed_tail_and_filters_only_elapsed_dates():
    old = (('2026-09-26','long'),(DATES[0],'easy'),(DATES[2],'short'))
    remaining=old[1:]
    result=search(**dict(options(),previous=old),
        evaluate=lambda p:dict(cost=0. if p==remaining else 10.,eligible=True))
    assert result['program']==remaining
    assert result['prior_revalidation']['physical_fingerprint']==fingerprint(remaining)


def test_unknown_prior_dose_is_not_silently_rebuilt():
    with pytest.raises(ValueError,match='migration'):
        search(**dict(options(),previous=((DATES[0],'removed-dose'),)),
               evaluate=lambda p:dict(cost=0.,eligible=True))


def test_prior_extension_is_priced_before_independent_seeds_without_score_bonus():
    prior=((DATES[0],'easy'),)
    independent=((DATES[1],'long'),)
    extended=prior+((DATES[2],'short'),)
    calls=[]
    def evaluate(p):
        calls.append(p)
        return dict(cost=0. if p==extended else 1. if p==independent else 2.,eligible=True)
    result=search(**dict(options(),previous=prior),evaluate=evaluate,
                  previous_end=DATES[2],seed_programs=(independent,))
    assert result['program']==extended
    assert calls.index(extended)<calls.index(independent)
    assert result['prior_revalidation']['extension_evaluations']==len(ROLES)
    assert result['prior_revalidation']['extension_dates']==(DATES[2],)
    # No current-domain extension is inferred from the last scheduled activity.
    calls.clear()
    result=search(**dict(options(),previous=prior),evaluate=evaluate,
                  previous_end='2026-09-30',seed_programs=(independent,))
    assert result['prior_revalidation']['extension_evaluations']==0


def test_continuations_compare_the_contender_found_by_final_joint_search(monkeypatch):
    import run_analysis.adaptive.continuous_search_adapter as module
    from datetime import date,timedelta
    dates=tuple((date(2026,9,27)+timedelta(days=i)).isoformat() for i in range(9))
    prior=((dates[0],'easy'),(dates[-1],'long'))
    changed=((dates[1],'easy'),(dates[-1],'long'))
    # Isolate the final joint discovery. Coarse beam cannot propose the long;
    # prior is retained as the only good full seed. No legacy score is involved.
    monkeypatch.setattr(full_score_refinement,'refine',lambda *a,**k:dict(lanes=[]))
    monkeypatch.setattr(finalized_distance_search,'refine_distances',lambda *a,**k:dict(lanes=[]))
    seen=[]
    def rebuild(frontiers,*args,**kwargs):
        seen.append([c['physical'] for c in frontiers])
        return dict(lanes=[])
    monkeypatch.setattr(module,'rebuild_frontiers',rebuild)
    result=module.search(dates=dates,keys=('easy','long'),roles={'easy':'easy','long':'long'},
        hard_end=dates[-2],previous=prior,seed_keys=('easy',),stable_days=2,budget=1500,
        frontier_repair=True,
        evaluate=lambda p:dict(eligible=True,cost=8. if p==changed else 10. if p==prior else 100.))
    assert seen==[[changed,prior]]
    assert result['program']==changed  # Conditional prefix search is not a lock.
    assert result['local_neighborhood_complete']
    assert result['refinement']['post_joint_calendar_distance'] is not None


@pytest.mark.parametrize('budget',[300,500,1000])
def test_continuation_reservation_does_not_interrupt_baseline_joint_search(monkeypatch,budget):
    import run_analysis.adaptive.continuous_search_adapter as module
    from datetime import date,timedelta
    dates=tuple((date(2026,9,27)+timedelta(days=i)).isoformat() for i in range(9))
    prior=((dates[0],'easy'),(dates[-1],'long'))
    changed=((dates[1],'easy'),(dates[-1],'long'))
    monkeypatch.setattr(full_score_refinement,'refine',lambda *a,**k:dict(lanes=[]))
    monkeypatch.setattr(finalized_distance_search,'refine_distances',lambda *a,**k:dict(lanes=[]))
    calls=[]
    def evaluate(p):
        calls.append(p)
        return dict(eligible=True,cost=8. if p==changed else 10. if p==prior else 100.)
    args=dict(dates=dates,keys=('easy','long'),roles={'easy':'easy','long':'long'},
        hard_end=dates[-2],previous=prior,seed_keys=('easy',),stable_days=2,
        budget=budget,evaluate=evaluate)
    baseline=module.search(**args,frontier_repair=False)
    expected=tuple(calls)
    calls.clear()
    at_refresh=[]
    def rebuild(*args,**kwargs):
        at_refresh.append(tuple(calls))
        return dict(lanes=[])
    monkeypatch.setattr(module,'rebuild_frontiers',rebuild)
    repaired=module.search(**args,frontier_repair=True)
    assert at_refresh==[expected]
    assert repaired['program']==baseline['program']
    assert repaired['local_neighborhood_complete']==baseline['local_neighborhood_complete']


def test_unproven_continuation_experiment_does_not_change_default_search():
    result=search(**options(),stable_days=1,
        evaluate=lambda p:dict(cost=-len(p),eligible=True))
    assert result['search']=='reused_full_score_and_finalized_distance_refiners_v8_rebased'
    assert result['refinement']['frontier_continuation'] is None
    assert result['refinement']['post_calendar_candidate_distance'] is None
    assert result['refinement']['post_joint_calendar_distance'] is None
