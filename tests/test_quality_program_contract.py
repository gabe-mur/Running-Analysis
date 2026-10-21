from dataclasses import replace

import pytest

from run_analysis.adaptive.quality_program_contract import PurposeDelivery, QualityProgramme
from run_analysis.adaptive.shared_program_evaluator import ProgramCandidate, evaluate_program
from run_analysis.adaptive.timely_purpose_trial import PurposeTimingPolicy
from run_analysis.workload_accounting import Quantity
from test_quality_purpose_contract import contract
from test_shared_program_evaluator import context, session


def delivery(key,sustained=0.,faster=1.):
    return PurposeDelivery(key,Quantity.point(sustained,'fixture'),Quantity.point(faster,'fixture'),
        Quantity.point(max(sustained,faster),'fixture opportunity'))


def programme(rows):
    return QualityProgramme(contract(),(),tuple(rows))


def plan(keys):
    rows=tuple(replace(session('candidate:'+session('x',i).day.isoformat()+':'+key,i,
        role='quality'),quality_credit=1.) for i,key in keys)
    return ProgramCandidate(tuple(r for r in rows if r.day<session('x',21).day),
        tuple(r for r in rows if r.day>=session('x',21).day),'fixture tail')


def test_one_budget_and_noncompensating_purpose_deficit():
    qp=programme((delivery('fast'),delivery('sustained',1.,0.)))
    ctx=context(timing=PurposeTimingPolicy(1.,grace_fraction=.5),quality_programme=qp)
    fast=evaluate_program(ctx,plan([(0,'fast'),(7,'fast'),(14,'fast'),(21,'fast'),(28,'fast')]))
    varied=evaluate_program(ctx,plan([(0,'fast'),(7,'sustained'),(14,'fast'),(21,'sustained'),(28,'fast')]))
    for result in (fast,varied):
        p=result.parts['quality_programme']
        assert p['loss']==max(p['shared_opportunity_loss'],p['descriptor_loss'])
        assert sum(r['share'] for r in p['purposes'].values())==1.
        assert result.total_cost==pytest.approx(result.delivery_cost+result.weighted_demand_cost)
    assert fast.parts['quality_programme']['purposes']['sustained']['loss']>varied.parts['quality_programme']['purposes']['sustained']['loss']
    assert fast.parts['demand_cost']==varied.parts['demand_cost']


def test_missing_credit_is_not_free_and_cadence_cannot_change_silently():
    ctx=context(timing=PurposeTimingPolicy(1.,grace_fraction=.5),quality_programme=programme((delivery('fast'),)))
    with pytest.raises(ValueError,match='Missing or ambiguous'):
        evaluate_program(ctx,plan([(0,'unknown')]))
    with pytest.raises(ValueError,match='accepted cadence'):
        evaluate_program(replace(ctx,quality_programme=replace(ctx.quality_programme,
            contract=replace(contract(),cadence_days=8.))),plan([(0,'fast')]))
    with pytest.raises(ValueError,match='cannot silently bypass'):
        evaluate_program(ctx,plan([(0,'fast')]),timing_history=())


def test_completion_keeps_credit_and_relative_cost_when_decision_advances():
    alternatives=(plan([(0,'fast'),(7,'sustained')]),plan([(0,'fast'),(8,'sustained')]))
    qp=programme((delivery('fast'),delivery('sustained',1.,0.)))
    ctx=context(timing=PurposeTimingPolicy(1.,grace_fraction=.5),quality_programme=qp)
    before=[evaluate_program(ctx,p) for p in alternatives]
    first=replace(alternatives[0].hard[0],key='uploaded',source='observed')
    after_ctx=replace(ctx,decision=session('x',1).day,history=(*ctx.history,first),
        quality_programme=replace(qp,observed=(delivery('uploaded'),)))
    after=[evaluate_program(after_ctx,replace(p,hard=p.hard[1:])) for p in alternatives]
    assert before[0].total_cost-before[1].total_cost==pytest.approx(after[0].total_cost-after[1].total_cost)


def test_typed_restore_and_continuation_use_the_existing_handoff_dates():
    from run_analysis.adaptive.decision_snapshot import _encode,_decode
    from test_known_purpose_phase import conditioned
    from test_soft_horizon_trial import candidate
    ctx,rows=conditioned()
    quality=[r for r in rows if r.role=='quality']
    renamed=tuple(replace(r,key='candidate:'+r.day.isoformat()+':fast') if r.role=='quality' else r for r in rows)
    qp=programme((delivery('fast'),))
    ctx=replace(ctx,quality_programme=qp,continuation_distribution='c1_v1')
    assert _decode(_encode(ctx))==ctx
    result=evaluate_program(ctx,candidate(ctx,renamed))
    expected=[r['handoff'] for r in result.parts['continuation_boundary']['scenarios']]
    for purpose in result.parts['quality_programme']['purposes'].values():
        assert [r['handoff'] for r in purpose['continuation']['phase_by_handoff']]==expected
