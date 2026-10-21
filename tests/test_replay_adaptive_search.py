from hashlib import sha256
from datetime import timedelta
import json
from zipfile import ZipFile

import pytest

from scripts.replay_adaptive_search import (
    replayed_prior,tapered_case, neutral_cadence_evaluator, recurrence_summary,
    aperiodic_starting_calendars,
)


def test_causal_handoff_accepts_only_the_declared_prior_origin_and_unchanged_inputs(tmp_path):
    earlier=tmp_path/'decision-0002.json'
    current=tmp_path/'decision-0003.json'
    earlier.write_text('{"origin":2}')
    current.write_text('{"origin":3}')
    archive=tmp_path/'sources.zip'
    with ZipFile(archive,'x') as source:
        source.writestr('manifest.json',json.dumps(dict(input_sha256={
            str(p):sha256(p.read_bytes()).hexdigest() for p in (earlier,current)})))
    result=tmp_path/'result.json'
    program=[['2026-09-30','easy:2.1']]
    result.write_text(json.dumps(dict(program=program,manifest=dict(path=str(archive),
        sha256=sha256(archive.read_bytes()).hexdigest()))))
    assert replayed_prior(result,current)==program
    with pytest.raises(ValueError,match='earlier decision'):
        replayed_prior(result,earlier)
    current.write_text('{"origin":3,"changed":true}')
    with pytest.raises(ValueError,match='earlier decision'):
        replayed_prior(result,current)


def test_causal_handoff_rejects_changed_provenance_archive(tmp_path):
    archive=tmp_path/'sources.zip'
    archive.write_bytes(b'changed')
    result=tmp_path/'result.json'
    result.write_text(json.dumps(dict(manifest=dict(path=str(archive),sha256='wrong'))))
    with pytest.raises(ValueError,match='archive changed'):
        replayed_prior(result,tmp_path/'decision-0003.json')


def test_taper_ablation_preserves_verified_source_state_and_archive_score():
    from dataclasses import dataclass
    @dataclass(frozen=True)
    class Context:
        soft_horizon_taper: bool=False
    @dataclass(frozen=True)
    class State:
        context: Context
    original=State(Context())
    calls=[]
    def evaluate(program,*,state_override=None):
        calls.append((program,state_override))
        return dict(cost=1. if state_override.context.soft_horizon_taper else 2.,eligible=True)
    case=dict(state=original,evaluate=evaluate,archived_cost=2.)
    changed=tapered_case(case)
    assert not original.context.soft_horizon_taper
    assert changed['state'].context.soft_horizon_taper
    assert changed['archived_cost']==case['archived_cost']==2.
    assert changed['evaluate'](())['cost']==1.
    assert calls==[((),changed['state'])]


def test_neutral_cadence_changes_only_timing_not_actual_load_or_volume():
    from dataclasses import replace
    from scripts.shared_program_evaluator import evaluate_program
    from scripts.timely_purpose_trial import PurposeTimingPolicy
    from test_memory_covered_continuation import guarded
    from test_soft_horizon_trial import candidate
    ctx,_,rows=guarded()
    ctx=replace(ctx,timing=PurposeTimingPolicy(1.,grace_fraction=1.))
    plan=candidate(ctx,rows)
    actual=evaluate_program(ctx,plan)
    neutral=neutral_cadence_evaluator(ctx,plan)
    assert neutral.conditional_ledger==actual.conditional_ledger
    assert neutral.weighted_demand_cost==actual.weighted_demand_cost
    assert neutral.short_run_cost==actual.short_run_cost
    assert neutral.parts['rolling_volume']==actual.parts['rolling_volume']
    assert neutral.parts['session_costs']==actual.parts['session_costs']
    assert neutral.fingerprint!=actual.fingerprint
    assert neutral.parts['timing']['long']['opening_coverage']==1.
    assert neutral.parts['timing']['quality']['opening_coverage']==1.


def test_neutral_cadence_is_independent_of_historical_key_phase_only():
    from dataclasses import replace
    from scripts.timely_purpose_trial import PurposeTimingPolicy
    from test_shared_program_evaluator import context, candidate, session
    ctx=context(timing=PurposeTimingPolicy(1.,grace_fraction=1.),history=(
        session('long',-2,8.4,'long','observed'),session('quality',-5,4.5,'quality','observed')))
    shifted=replace(ctx,history=(replace(ctx.history[0],day=ctx.history[0].day-timedelta(days=2)),
        ctx.history[1]))
    a,b=(neutral_cadence_evaluator(c,candidate()) for c in (ctx,shifted))
    assert a.parts['timing']==b.parts['timing']
    assert a.weighted_demand_cost!=b.weighted_demand_cost


def test_connected_neutral_override_preserves_eligibility_and_physical_ledger():
    from test_program_evidence_adapter import fixtures,evaluate
    history,plan=fixtures()
    actual=evaluate(history,plan)
    neutral=evaluate(history,plan,program_evaluator=neutral_cadence_evaluator)
    assert neutral['constraints']==actual['constraints']
    assert neutral['suitability']==actual['suitability']
    assert neutral['decision_eligible']==actual['decision_eligible']
    assert neutral['quality_credit']==actual['quality_credit']
    assert neutral['evaluation'].conditional_ledger==actual['evaluation'].conditional_ledger
    assert neutral['evaluation'].weighted_demand_cost==actual['evaluation'].weighted_demand_cost
    assert neutral['evaluation'].parts['timing']!=actual['evaluation'].parts['timing']


def test_timing_override_default_preserves_identity_and_rejects_future_markers():
    from dataclasses import replace
    from scripts.shared_program_evaluator import evaluate_program
    from scripts.timely_purpose_trial import PurposeTimingPolicy
    from test_shared_program_evaluator import context,candidate
    ctx=context(timing=PurposeTimingPolicy(1.,grace_fraction=1.))
    a=evaluate_program(ctx,candidate())
    b=evaluate_program(ctx,candidate(),timing_history=None,timing_opening_coverage=None)
    assert a==b
    with pytest.raises(ValueError,match='precede'):
        evaluate_program(ctx,candidate(),timing_history=(replace(ctx.history[0],day=ctx.decision),))
    with pytest.raises(ValueError,match='requires a timing policy'):
        evaluate_program(replace(ctx,timing=None),candidate(),timing_history=())


def test_recurrence_is_measured_after_selection_and_scoped_to_report_dates():
    from datetime import date,timedelta
    from types import SimpleNamespace
    catalog={'e':SimpleNamespace(role='easy',miles=2.3),'l':SimpleNamespace(role='long',miles=8.9)}
    dates=tuple((date(2026,9,29)+timedelta(days=i)).isoformat() for i in range(21))
    periodic=tuple((d,'l' if i%7==0 else 'e') for i,d in enumerate(dates) if i%7 in (0,3))
    result=recurrence_summary((*periodic,('2026-10-30','l')),catalog,dates)
    assert 7 in result['exact_dose_periods']
    assert result['key_gaps']['long']==[7,7]
    nonperiodic=tuple(p for p in periodic if p[0]!=dates[10])
    assert 7 not in recurrence_summary(nonperiodic,catalog,dates)['exact_role_periods']


def test_aperiodic_starts_preserve_doses_and_domain_without_repeat_constraint():
    from datetime import date
    from types import SimpleNamespace
    from collections import Counter
    dates=tuple((date(2026,9,29)+timedelta(days=i)).isoformat() for i in range(54))
    catalog={'q':SimpleNamespace(role='quality',miles=3.9),
        'l':SimpleNamespace(role='long',miles=8.9),'e':SimpleNamespace(role='easy',miles=3.3)}
    seed=tuple((day,{0:'q',2:'e',4:'l',6:'e'}[i%7]) for i,day in enumerate(dates) if i%7 in (0,2,4,6))
    starts=aperiodic_starting_calendars(dates,(seed,)*9)
    assert len(starts)==9 and starts==aperiodic_starting_calendars(dates,(seed,)*9)
    assert len(set(starts))==len(starts)
    for program in starts:
        assert Counter(k for _,k in program)==Counter(k for _,k in seed)
        assert set(d for d,_ in program)<=set(dates)
        assert len(dict(program))==len(program)
        assert 7 not in recurrence_summary(program,catalog,dates)['exact_role_periods']
    # At least one genuinely irregular key cadence, not just a rotated template.
    assert any(len(set(recurrence_summary(p,catalog,dates)['key_gaps']['long']))>1 for p in starts)
