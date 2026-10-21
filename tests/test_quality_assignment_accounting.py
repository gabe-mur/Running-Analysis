from dataclasses import replace
from datetime import timedelta

import pytest

from run_analysis.adaptive.quality_assignment_accounting import (
    IssuedQualityAssignment, reconcile_assignment, advance_sequence,completion_json,read_completion)
from run_analysis.adaptive.continuous_program_adapter import evaluate_continuous
from run_analysis.workload_accounting import Quantity
from test_continuous_program_adapter import run, context, REF, AT
from test_quality_coaching_policy import sequence


def assignment(unit='time', required=720., rank=0):
    q=sequence()
    return IssuedQualityAssignment(q.stage_key,rank,q.assignment(rank).key,
        AT-timedelta(days=1),q.assignment(rank).purpose,unit,Quantity.point(required,'issued primary'))


def test_primary_units_ranges_partial_and_extra_are_not_fatigue_or_role_credit():
    actual=run(source='observed',work=12.)
    a=assignment()
    assert reconcile_assignment(a,actual,purpose_matched=True).credit.low==1.
    shorter=run(source='observed',work=6.)
    assert reconcile_assignment(a,shorter,purpose_matched=True).credit.low==.5
    assert reconcile_assignment(a,run(source='observed',work=18.),purpose_matched=True).credit.high==1.
    assert reconcile_assignment(a,actual,purpose_matched=False).credit.high==0.
    assert reconcile_assignment(a,run(source='observed',role='easy',work=0),purpose_matched=True).credit.high==0.
    ranged=replace(a,required=Quantity(360,720,540,'offered work range'))
    assert reconcile_assignment(ranged,shorter,purpose_matched=True).credit.low==1.
    distance=replace(actual,bouts=tuple(replace(b,distance_meters=400.) for b in actual.bouts))
    assert reconcile_assignment(assignment('distance',2400),distance,purpose_matched=True).credit.low==1.
    # Running faster must not turn the same 400m commands into poor adherence.
    faster=run(source='observed',work=6.)
    faster=replace(faster,bouts=tuple(replace(b,distance_meters=400.) for b in faster.bouts))
    assert reconcile_assignment(assignment('distance',2400),faster,purpose_matched=True).credit.low==1.
    assert reconcile_assignment(assignment('distance',2400),actual,purpose_matched=True).credit.low==0.


def test_receipt_is_causal_and_binds_original_evidence_not_mutable_role():
    actual=run(source='observed')
    receipt=reconcile_assignment(assignment(),actual,purpose_matched=True)
    assert receipt.for_run(actual).low==1.
    assert read_completion(completion_json(receipt))==receipt
    assert read_completion(completion_json(receipt)).for_run(actual).low==1.
    with pytest.raises(ValueError,match='primary evidence'):
        replace(receipt,credit=Quantity.point(.5,'tampered')).for_run(actual)
    with pytest.raises(ValueError,match='physical evidence'):
        receipt.for_run(replace(actual,miles=4.1))
    with pytest.raises(ValueError,match='physical evidence'):
        replace(receipt,available_at=receipt.available_at+timedelta(seconds=1)).for_run(actual)
    with pytest.raises(ValueError,match='pre-issued'):
        reconcile_assignment(replace(assignment(),issued_at=actual.end),actual,purpose_matched=True)
    with pytest.raises(ValueError,match='Observed'):
        reconcile_assignment(assignment(),run(),purpose_matched=True)


def test_completion_and_miss_have_explicit_persistent_sequence_semantics():
    q=sequence(); actual=run(source='observed',work=6.)
    a=replace(assignment(),issued_at=AT.fromisoformat(q.issued_at))
    actual=replace(actual,start=a.issued_at+timedelta(days=1),end=a.issued_at+timedelta(days=1,minutes=40),
        available_at=a.issued_at+timedelta(days=1,minutes=40))
    partial=reconcile_assignment(a,actual,purpose_matched=True)
    assert advance_sequence(q,partial).consumed==1
    assert advance_sequence(q,reconcile_assignment(a,actual,purpose_matched=False))==q
    unknown=replace(actual,work_evidence_basis='unknown',bouts=(),demand=(*actual.demand[:2],Quantity(0,12,basis='unknown')))
    assert advance_sequence(q,reconcile_assignment(a,unknown,purpose_matched=True))==q
    with pytest.raises(ValueError,match='persisted'):
        advance_sequence(advance_sequence(q,partial),partial)


def test_shared_provider_owns_history_opening_and_diagnostics_without_load_mutation():
    planned=run(work=6.); actual=run(-1,source='observed',work=6.)
    credit=lambda r: Quantity.point(1.,'same frozen fulfillment')
    result=evaluate_continuous(context(),(actual,),(planned,),reference=REF,
        as_of=AT,timezone='UTC',purpose_fraction=.5,quality_credit_provider=credit)
    assert all(x.quality_credit==1. for x in result['evaluation'].conditional_ledger)
    assert all(v.low==1. for v in result['quality_credit'].values())
    assert result['opening_coverage_bounds']['quality'].low>0.
    assert [x.demand for x in result['evaluation'].conditional_ledger]==[actual.demand,planned.demand]
    with pytest.raises(ValueError,match='nonquality'):
        evaluate_continuous(context(),(),(run(role='easy',work=0),),reference=REF,
            as_of=AT,timezone='UTC',purpose_fraction=.5,quality_credit_provider=credit)


def test_expected_upload_does_not_change_relative_future_scores():
    first=run(work=6.)
    def evaluate(history,future,at):
        return evaluate_continuous(context(),history,future,reference=REF,as_of=at,
            timezone='UTC',purpose_fraction=.5,
            quality_credit_provider=lambda r:Quantity.point(1. if r.role=='quality' else 0.,'issued fulfillment'))['evaluation']
    alternatives=[(first,run(d,work=12.)) for d in (3,4)]
    before=[evaluate((),p,AT) for p in alternatives]
    actual=replace(first,key='uploaded',source='observed',available_at=first.end,work_evidence_basis='explicit_segments')
    receipt=reconcile_assignment(assignment(required=360),actual,purpose_matched=True)
    assert receipt.credit.low==1.
    after=[evaluate((actual,),p[1:],first.end) for p in alternatives]
    assert before[0].total_cost-before[1].total_cost==pytest.approx(after[0].total_cost-after[1].total_cost,abs=1e-12)
    assert all(x.conditional_ledger[0].quality_credit==1. for x in after)
