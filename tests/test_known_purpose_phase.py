from dataclasses import replace
from datetime import timedelta

import pytest

from test_purpose_continuation import enabled
from test_soft_horizon_trial import candidate
from run_analysis.adaptive.continuation_boundary_trial import purpose_handoff_losses
from run_analysis.adaptive.decision_snapshot import _encode, _decode
from run_analysis.adaptive.new_model_daily_harness import HarnessPolicy
from run_analysis.adaptive.shared_program_evaluator import (
    evaluate_program, PHASE_CONTINUATION_MODEL_VERSION, PURPOSE_CONTINUATION_MODEL_VERSION,
)


def conditioned():
    ctx,rows=enabled()
    return replace(ctx,purpose_continuation_phase='known_full_phase_v1'),rows


@pytest.mark.parametrize('cadence',[3,7,8,12])
def test_full_known_phase_cannot_create_a_second_unrelated_wait(cadence):
    ctx,rows=conditioned();handoff=ctx.end
    ctx=replace(ctx,policy=replace(ctx.policy,long_per_7=7/cadence))
    full=next(r for r in rows if r.role=='long')
    for age in range(1,cadence+1):
        row=replace(full,day=handoff-timedelta(days=age))
        result=purpose_handoff_losses(ctx,(row,),(handoff,))['long']
        assert result['loss']==0.
        phase=result['phase_by_handoff'][0]
        assert phase['age_days']==age and phase['arrival_delay_days']==cadence-age
        assert phase['last_full_completion']==row.day
    late=replace(full,day=handoff-timedelta(days=cadence+1))
    assert purpose_handoff_losses(ctx,(late,),(handoff,))['long']['loss']>0.


@pytest.mark.parametrize('kind',['missing','partial','overdue','same_day','future'])
def test_unestablished_phase_retains_legacy_uncertainty_not_a_free_catchup(kind):
    ctx,rows=conditioned();handoff=ctx.end
    row=replace(next(r for r in rows if r.role=='quality'),day=handoff-timedelta(days=2))
    changes={'partial':dict(quality_credit=.6),'overdue':dict(day=handoff-timedelta(days=8)),
        'same_day':dict(day=handoff),'future':dict(day=handoff+timedelta(days=1))}
    program=() if kind=='missing' else (replace(row,**changes[kind]),)
    result=purpose_handoff_losses(ctx,program,(handoff,))['quality']
    old=purpose_handoff_losses(replace(ctx,purpose_continuation_phase='uniform_v1'),program,(handoff,))['quality']
    assert result['loss']==old['loss']>0.
    assert result['phase_by_handoff'][0]['basis']=='uniform_missing_or_overdue_full_phase'


def test_partial_session_improves_coverage_without_postponing_known_full_phase():
    ctx,rows=conditioned();handoff=ctx.end
    full=replace(next(r for r in rows if r.role=='quality'),day=handoff-timedelta(days=6))
    partial=replace(full,key='partial',day=handoff-timedelta(days=1),quality_credit=.8)
    a=purpose_handoff_losses(ctx,(full,),(handoff,))['quality']
    b=purpose_handoff_losses(ctx,(full,partial),(handoff,))['quality']
    assert a==b and b['phase_by_handoff'][0]['arrival_delay_days']==1
    late=replace(full,day=handoff-timedelta(days=9))
    assert purpose_handoff_losses(ctx,(late,partial),(handoff,))['quality']['loss']<purpose_handoff_losses(ctx,(late,),(handoff,))['quality']['loss']


def test_proportional_quality_clock_is_identical_in_selectable_and_handoff_domains():
    ctx,rows=conditioned();handoff=ctx.end
    row=replace(next(r for r in rows if r.role=='quality'),day=handoff-timedelta(days=1),quality_credit=.5)
    result=purpose_handoff_losses(ctx,(row,),(handoff,),quality_cadence_contract='proportional_grace_v1')
    phase=result['quality']['phase_by_handoff'][0]
    assert phase['arrival_delay_days']==3  # ceil(.5*7-1), not a fresh seven-day reset.
    assert phase['basis']=='known_proportional_renewal_phase'
    assert phase['last_full_completion'] is None
    assert result['quality']['loss']>0  # Fractional last day before nominal arrival.
    assert result['long']==purpose_handoff_losses(ctx,(row,),(handoff,))['long']
    full=replace(row,quality_credit=1.)
    assert purpose_handoff_losses(ctx,(full,),(handoff,),quality_cadence_contract='proportional_grace_v1')['quality']['loss']==purpose_handoff_losses(ctx,(full,),(handoff,))['quality']['loss']


def test_noninteger_cadence_uses_explicit_next_daily_grid_arrival():
    ctx,rows=conditioned();handoff=ctx.end
    ctx=replace(ctx,policy=replace(ctx.policy,long_per_7=7/8.5))
    row=replace(next(r for r in rows if r.role=='long'),day=handoff-timedelta(days=8))
    result=purpose_handoff_losses(ctx,(row,),(handoff,))['long']
    from run_analysis.adaptive.timely_purpose_trial import envelope_day
    expected=envelope_day(((row.day,1.),),handoff,grace=8.5,memory=8.5)['loss']/(ctx.end-ctx.stage_start).days
    assert result['phase_by_handoff'][0]['arrival_delay_days']==1
    assert result['loss']==pytest.approx(expected,abs=1e-12)


def test_only_purpose_tail_changes_other_costs_and_physical_ledger_are_identical():
    ctx,rows=conditioned();program=candidate(ctx,rows)
    current=evaluate_program(ctx,program)
    old=evaluate_program(replace(ctx,purpose_continuation_phase='uniform_v1'),program)
    without=evaluate_program(replace(ctx,purpose_continuation=False,purpose_continuation_phase='uniform_v1'),program)
    assert current.model_version==PHASE_CONTINUATION_MODEL_VERSION
    assert old.model_version==PURPOSE_CONTINUATION_MODEL_VERSION
    assert current.fingerprint!=old.fingerprint
    assert current.conditional_ledger==old.conditional_ledger==without.conditional_ledger
    assert current.weighted_demand_cost==old.weighted_demand_cost==without.weighted_demand_cost
    assert current.short_run_cost==old.short_run_cost
    for name in ('volume_cost','raw_cross_demand_cost','expected_miles_per_7','expected_support_per_7'):
        assert current.parts['continuation_boundary'][name]==old.parts['continuation_boundary'][name]
    assert current.total_cost==pytest.approx(without.total_cost+current.parts['continuation_boundary']['purpose_tail_cost'],abs=1e-12)
    assert current.total_cost==pytest.approx(current.delivery_cost+current.weighted_demand_cost+current.short_run_cost,abs=1e-12)


def test_tail_deletion_remains_costly_and_later_sessions_cannot_repair_earlier_handoff():
    ctx,rows=conditioned();handoffs=(ctx.hard_end+timedelta(days=1),ctx.end)
    base=purpose_handoff_losses(ctx,rows,handoffs)
    changed=purpose_handoff_losses(ctx,tuple(replace(r,quality_credit=0.) if r.day>=handoffs[0]
        and r.role=='quality' else r for r in rows),handoffs)
    assert base['quality']['by_handoff'][0]==changed['quality']['by_handoff'][0]
    assert changed['quality']['by_handoff'][1]>base['quality']['by_handoff'][1]
    assert purpose_handoff_losses(ctx,(),(ctx.end,))['long']['loss']>base['long']['loss']


def test_timing_override_is_respected_and_does_not_create_physical_work():
    ctx,rows=conditioned()
    markers=tuple(replace(r,key='marker:'+r.role,day=ctx.decision-timedelta(days=1),source='observed') for r in rows[:2])
    result=evaluate_program(ctx,candidate(ctx,()),timing_history=markers)
    assert result.conditional_ledger==()
    expected=purpose_handoff_losses(ctx,(),tuple(r['handoff'] for r in result.parts['continuation_boundary']['scenarios']),timing_history=markers)
    assert result.parts['continuation_boundary']['purpose_continuation']==expected


def test_exact_prescription_completion_does_not_change_phase_cost_or_pairwise_ranking():
    ctx,rows=conditioned();first=rows[0]
    after=replace(ctx,decision=ctx.decision+timedelta(days=1),history=(replace(first,source='observed'),))
    scores=[]
    for shift in (0,1):
        program=tuple(replace(r,day=r.day+timedelta(days=shift)) if r.key=='5' else r for r in rows)
        before=evaluate_program(ctx,candidate(ctx,program))
        completed=evaluate_program(after,candidate(after,program[1:]))
        assert before.parts['continuation_boundary']['purpose_continuation']==completed.parts['continuation_boundary']['purpose_continuation']
        scores.append((before.total_cost,completed.total_cost))
    assert scores[0][0]-scores[1][0]==pytest.approx(scores[0][1]-scores[1][1],abs=1e-12)


def test_versioned_phase_roundtrip_and_validation_preserve_archived_contract():
    ctx,_=conditioned();old=replace(ctx,purpose_continuation_phase='uniform_v1')
    assert 'purpose_continuation_phase' not in _encode(old)['fields']
    assert _decode(_encode(old))==old
    assert _decode(_encode(ctx))==ctx
    for invalid in ('unknown',True):
        with pytest.raises(ValueError,match='phase'):
            replace(ctx,purpose_continuation_phase=invalid)
    with pytest.raises(ValueError,match='phase'):
        replace(ctx,purpose_continuation=False)
    with pytest.raises(ValueError,match='phase'):
        HarnessPolicy(purpose_continuation_phase='known_full_phase_v1')
