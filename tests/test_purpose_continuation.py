from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace

import pytest

from test_continuation_boundary_trial import setup
from test_soft_horizon_trial import candidate
from run_analysis.adaptive.continuation_boundary_trial import purpose_handoff_losses
from run_analysis.adaptive.decision_snapshot import _encode, _decode
from run_analysis.adaptive.shared_program_evaluator import (
    evaluate_program, PURPOSE_CONTINUATION_MODEL_VERSION,
)
from run_analysis.adaptive.timely_purpose_trial import envelope_day


def enabled():
    ctx,refs,rows,_=setup()
    return replace(ctx,continuation_boundary_references=tuple(
        (k,m,d) for k,(m,d) in refs.items()),purpose_continuation=True),rows


def test_purpose_tail_matches_all_nominal_phases_and_charges_once():
    ctx,rows=enabled()
    result=evaluate_program(ctx,candidate(ctx,rows))
    old=evaluate_program(replace(ctx,purpose_continuation=False),candidate(ctx,rows))
    boundary=result.parts['continuation_boundary']
    total=0.
    span=(ctx.end-ctx.stage_start).days
    for role,index in (('long',2),('quality',3)):
        values=[]
        for scenario in boundary['scenarios']:
            handoff=scenario['handoff']
            events=[(r.day,1.) for r in rows if r.role==role and r.day<handoff]
            expected=sum(sum(envelope_day(events,handoff+timedelta(days=d),
                grace=7,memory=7)['loss'] for d in range(delay)) for delay in range(7))/7/span
            assert scenario['purpose_tail_losses'][role]==pytest.approx(expected,abs=1e-12)
            values.append(expected)
        expected=sum(values)/len(values)
        assert result.parts['shortfall_by_purpose'][role]==pytest.approx(
            old.parts['shortfall_by_purpose'][role]+expected,abs=1e-12)
        total+=expected*ctx.policy.purpose_weights[index]
    assert result.total_cost==pytest.approx(old.total_cost+total,abs=1e-12)
    assert result.total_cost==pytest.approx(result.delivery_cost+result.weighted_demand_cost+
                                         result.short_run_cost,abs=1e-12)
    assert result.weighted_demand_cost==old.weighted_demand_cost
    assert result.conditional_ledger==old.conditional_ledger
    assert result.parts['delivery']==old.parts['delivery']
    assert result.model_version==PURPOSE_CONTINUATION_MODEL_VERSION
    assert result.fingerprint!=old.fingerprint


def test_later_prescription_cannot_repair_earlier_handoff_and_deletion_is_costly():
    ctx,rows=enabled()
    handoffs=(ctx.hard_end+timedelta(days=1),ctx.end)
    base=purpose_handoff_losses(ctx,rows,handoffs)
    later=tuple(replace(r,quality_credit=0.) if r.role=='quality' and r.day>=handoffs[0]
                else r for r in rows)
    altered=purpose_handoff_losses(ctx,later,handoffs)
    assert altered['quality']['by_handoff'][0]==base['quality']['by_handoff'][0]
    missing=purpose_handoff_losses(ctx,tuple(r for r in rows if r.role!='long'),handoffs)
    assert missing['long']['loss']>base['long']['loss']


@pytest.mark.parametrize('cadence',[7.,8.,8.5])
def test_partial_quality_credit_does_not_receive_a_full_reset(cadence):
    ctx,rows=enabled()
    ctx=replace(ctx,policy=replace(ctx.policy,quality_per_7=7/cadence))
    half=tuple(replace(r,quality_credit=.5) if r.role=='quality' else r for r in rows)
    absent=tuple(r for r in rows if r.role!='quality')
    f=lambda p:purpose_handoff_losses(ctx,p,(ctx.end,))['quality']['loss']
    assert f(rows)<f(half)<f(absent)


def test_timing_override_is_respected_without_changing_physical_ledger():
    ctx,rows=enabled()
    history=(replace(rows[0],key='timing-only',day=ctx.end-timedelta(days=1),source='observed'),)
    # Direct helper control; production evaluator validates override dates.
    result=purpose_handoff_losses(ctx,rows,(ctx.end,),timing_history=history)
    assert result['long']['loss']==0.
    assert all(r.key!='timing-only' for r in rows)
    empty=purpose_handoff_losses(ctx,(),(ctx.end,))
    initial=purpose_handoff_losses(ctx,(),(ctx.end,),timing_opening_coverage=(('long',1.),))
    assert initial['long']['loss']<empty['long']['loss']


def test_archived_context_serialization_omits_new_default_and_restores_new_contract():
    ctx,rows=enabled()
    old=replace(ctx,purpose_continuation=False)
    encoded=_encode(old)
    assert 'purpose_continuation' not in encoded['fields']
    assert _decode(encoded)==old
    assert _decode(_encode(ctx))==ctx
    with pytest.raises(ValueError,match='Purpose continuation'):
        replace(ctx,purpose_continuation=1)
    with pytest.raises(ValueError,match='Purpose continuation'):
        replace(ctx,continuation_boundary_references=())


def test_unordered_handoffs_cannot_borrow_future_prefix_evidence():
    ctx,rows=enabled()
    with pytest.raises(ValueError,match='Ordered unique'):
        purpose_handoff_losses(ctx,rows,(ctx.end,ctx.hard_end))


def test_integrated_timing_override_does_not_create_physical_work():
    ctx,rows=enabled()
    markers=tuple(replace(r,key='marker:'+r.role,day=ctx.decision-timedelta(days=1),
                          source='observed') for r in rows[:2])
    base=evaluate_program(replace(ctx,purpose_continuation=False),candidate(ctx,()),
                          timing_history=markers)
    result=evaluate_program(ctx,candidate(ctx,()),timing_history=markers)
    expected=purpose_handoff_losses(ctx,(),tuple(s['handoff']
        for s in result.parts['continuation_boundary']['scenarios']),timing_history=markers)
    for role in ('long','quality'):
        assert result.parts['shortfall_by_purpose'][role]==pytest.approx(
            base.parts['shortfall_by_purpose'][role]+expected[role]['loss'])
    assert result.conditional_ledger==base.conditional_ledger==()
