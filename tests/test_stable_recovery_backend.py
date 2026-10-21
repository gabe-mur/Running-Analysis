from dataclasses import replace
from datetime import timedelta
import pytest

from scripts.stable_recovery_backend import StableRecoveryBackend, recovery_backend_policy
from scripts.expected_completion import ExpectedCompletion, expected_run
from run_analysis.selection_objective import EXPECTED_COMPLETION_STATE, RECOVERY_BACKEND
from run_analysis.recovery_assessment import assess_next_session, AssessmentPolicy
from run_analysis import weekly_schedule as planner
from run_analysis.recommendation import recommend_next_run
from run_analysis.web.schemas import RecommendationRequest, WorkoutType
from test_expected_completion import setup
from test_hr_backed_sandbox import donor
from test_recommendation import CONFIG


@pytest.fixture
def case():
    template, history, rec = setup('easy', 3.8)
    backend = StableRecoveryBackend(dict(easy=donor(), long=donor('long'),
        quality=donor('intervals')), history)
    model = ExpectedCompletion(template, history, CONFIG)
    token = EXPECTED_COMPLETION_STATE.set(model)
    try:
        yield template, history, rec, backend, model
    finally:
        EXPECTED_COMPLETION_STATE.reset(token)


def test_new_assessment_matches_reference_nominal_ledger(case):
    _,_,rec,backend,model = case
    later = rec.model_copy(update={'planned_for':rec.planned_for+timedelta(days=2)})
    first = backend.entry(rec, [])
    target = backend.entry(later, [rec])
    direct = assess_next_session([*backend.completed.values(), first], target.workload,
        later.planned_for, backend.policy, proposed_work_minutes=target.work_minutes)
    actual = next(p for p in backend.check(later,[rec])['paths'] if p['phase']==0)
    assert actual['demands'] == pytest.approx({k:v['total_high'] for k,v in direct['channels'].items()})


def test_exact_completion_preserves_demand_without_double_count(case):
    template,history,rec,backend,model = case
    later = rec.model_copy(update={'planned_for':rec.planned_for+timedelta(days=2)})
    before = backend.check(later,[rec])
    backend.issued[rec.planned_for] = backend.entry(rec, [])
    execution = expected_run(rec,11)
    backend.complete(execution,rec)
    token = EXPECTED_COMPLETION_STATE.set(ExpectedCompletion(template,[*history,execution],CONFIG))
    try:
        backend.reset_build()
        after = backend.check(later,[])
        # Nominal is identical. Timing alternatives intentionally become known
        # execution times on completion; never mistake that for duplicate load.
        nominal = lambda x: next(p for p in x['paths'] if p['phase']==0)['demands']
        assert nominal(before) == pytest.approx(nominal(after))
        with pytest.raises(ValueError,match='duplicates|Duplicate'):
            backend.check(later,[rec])
    finally:
        EXPECTED_COMPLETION_STATE.reset(token)


def test_legacy_default_and_context_restoration(case):
    template,_,rec,backend,_ = case
    request = RecommendationRequest(health_status='normal')
    original = recommend_next_run(template,request,CONFIG)
    with pytest.raises(RuntimeError):
        with recovery_backend_policy(backend):
            assert RECOVERY_BACKEND.get() is backend
            raise RuntimeError('exit')
    assert RECOVERY_BACKEND.get() is None
    restored = recommend_next_run(template,request,CONFIG)
    assert restored.model_dump(exclude={'generated_at'}) == original.model_dump(exclude={'generated_at'})


def test_health_veto_is_not_cleared(case):
    template,_,_,backend,_ = case
    with recovery_backend_policy(backend):
        rec = recommend_next_run(template, RecommendationRequest(health_status='pain_or_injury_concern'), CONFIG)
        assert rec.workout_type == WorkoutType.REST
        assert backend.guard(template,rec) == rec


def test_new_policy_rejects_too_large_demand(case):
    template,_,rec,backend,_ = case
    backend.policy = AssessmentPolicy('restrictive test',10,20,36,36,5,36)
    with recovery_backend_policy(backend):
        assert not backend.check(rec)['compatible']
        assert backend.guard(template,rec).workout_type == WorkoutType.REST


def test_context_registers_prefix_and_removes_old_components(case):
    template,_,rec,backend,_ = case
    future = template.model_copy(update={'as_of':rec.planned_for+timedelta(days=2)})
    later = rec.model_copy(update={'planned_for':future.as_of})
    with recovery_backend_policy(backend):
        state = planner._project_state(future,[rec],CONFIG)
        assert backend.prefix(state) == (rec,)
        assert backend.check_state(state,later) == backend.check(later,[rec])
        day = planner.WeeklyScheduleDay(date=later.planned_for.date(), recommendation=later,
            day_role='easy_run', rationale='test')
        diagnostics = {}
        score = planner._finalized_program_recovery_cost([day],[state],CONFIG,
            diagnostic_components=diagnostics)
        assert set(diagnostics) == {'new_demand_excess'}
        assert score == sum(diagnostics.values())
