from datetime import timedelta
from types import SimpleNamespace

from scripts.prior_program_candidate import recover_prior_program
from scripts.continuation_replan import fingerprint, signature
from test_typed_continuation import fixture


def case():
    _, make=fixture()
    full,states=make(49,{2:('easy',3.5),25:('easy',4.)})
    hard=full[:21]
    saved=dict(selected_hard_fingerprint=fingerprint(hard), selected_program_fingerprint=fingerprint(full),
               soft_sessions=signature(full[21:]))
    prior=SimpleNamespace(planning_days=hard,days=hard[:7],
        planner_diagnostics=SimpleNamespace(continuation_selection=saved))
    return dict(prior_schedule=prior,daily_states=states[:21]),full,saved


def test_exact_prior_is_candidate_pending_checks_not_forced_selection():
    inputs,full,_=case()
    seed,audit=recover_prior_program(inputs,28)
    assert audit['status']=='pending_revalidation'
    assert fingerprint(seed['fixed_program'])==fingerprint(full)
    assert seed['source']=='prior_program'
    assert seed['offsets']==[2]


def test_rest_boundary_is_translated_without_padding_and_times_are_promoted():
    inputs,full,_=case()
    inputs['daily_states']=[s.model_copy(update={'as_of':s.as_of+timedelta(days=1)}) for s in inputs['daily_states']]
    seed,audit=recover_prior_program(inputs,28)
    assert audit['status']=='pending_revalidation'
    assert audit['translation']=='advanced_boundary'
    assert audit['consumed_dates']==[full[0].date.isoformat()]
    assert seed['terminal_missing_days']==1
    assert len(seed['fixed_prefix'])==48
    assert [d.date for d in seed['days']]==[s.as_of.date() for s in inputs['daily_states']]
    assert audit['promoted_times']


def test_unresolved_elapsed_run_is_not_silently_consumed():
    inputs,_,_=case()
    inputs['daily_states']=[s.model_copy(update={'as_of':s.as_of+timedelta(days=3)}) for s in inputs['daily_states']]
    seed,audit=recover_prior_program(inputs,28)
    assert seed is None
    assert audit['reason']=='elapsed_prescription_unresolved'


def test_missing_tail_is_not_rest():
    inputs,_,saved=case()
    saved['soft_sessions'].pop()
    seed,audit=recover_prior_program(inputs,28)
    assert seed is None
    assert audit['reason']=='soft_boundary_changed_or_missing'


def test_corrupted_prior_identity_is_rejected():
    inputs,_,saved=case()
    saved['selected_program_fingerprint']='wrong'
    seed,audit=recover_prior_program(inputs,28)
    assert seed is None
    assert audit['reason']=='saved_program_identity_mismatch'


def test_invalid_saved_tail_is_rejected_without_disabling_fresh_search():
    inputs,_,saved=case()
    saved['soft_sessions']=[{'date':'not-a-date'}]
    seed,audit=recover_prior_program(inputs,28)
    assert seed is None
    assert audit['reason']=='invalid_saved_continuation'


def test_no_saved_prior_does_not_change_opening_search():
    seed,audit=recover_prior_program({'prior_schedule':None},28)
    assert seed is None
    assert audit['reason']=='no_saved_full_program'


def test_current_health_revalidation_can_reject_a_previously_issued_program():
    from scripts.continuation_checks import check_sequence
    from run_analysis import weekly_schedule as planner
    from test_recommendation import CONFIG
    state_at,make=fixture()
    full,states=make(49,{2:('easy',3.5),25:('easy',4.)})
    inputs=dict(daily_states=states[:21], request=planner.RecommendationRequest(
        health_status='pain_or_injury_concern'), config=CONFIG)
    checked=check_sequence(full,inputs,state_at,())
    assert not checked['checks_pass']


def test_search_can_improve_a_prior_program_instead_of_locking_it():
    from scripts.full_score_refinement import refine
    def candidate(offsets,cost,source):
        return dict(score=SimpleNamespace(offsets=offsets,total_cost=cost),stable=True,
            row=dict(hard_fingerprint=str(offsets),source=source),tail_fingerprint='same')
    prior=candidate((0,),10.,'prior_program')
    def evaluate(seed):
        return candidate(tuple(seed['offsets']),5.,'full_score_neighbor')
    result=refine([prior],evaluate,{0,1},budget=10,starts=1,fixed_tail=True)
    assert result['lanes'][0]['path']==[10.,5.]
    assert result['lanes'][0]['final_fingerprint']!=prior['row']['hard_fingerprint']
