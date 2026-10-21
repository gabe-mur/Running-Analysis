"""Bounded upload/new-score integration diagnostic, NOT daily calendar search.

The physical metadata below is explicit synthetic evidence. These fixtures do
not establish tolerance policy or replace the missing production/search adapter.
The candidate universe and stage end stay fixed to isolate the upload seam.
"""
from dataclasses import replace
from datetime import datetime, timedelta
from itertools import product
import json

import pytest

from run_analysis.continuous_training_dose import general_quality_credit
from run_analysis.workload_accounting import Quantity
from run_analysis.workload_profile import WorkBout
from scripts.continuous_program_adapter import evaluate_continuous
from scripts.execution_forecast_contract import TargetBand
from scripts.program_evidence_adapter import AccountedRun, reconcile_completion, reference_key
from scripts.program_objective_trial import TrialPolicy, DemandScenario
from scripts.shared_program_evaluator import ProgramContext
from scripts.timely_purpose_trial import PurposeTimingPolicy
from test_execution_upload_sandbox import setup, tcx, AT, KINDS, START


def point(value):
    return Quantity.point(value,'explicit integration fixture, not fitted physiology')


def physical(key,start,miles,minutes,cardio,role,ref,*,source='planned',available=None,
             work_seconds=()):
    """Fixture-only physical bridge; role/movement/work must not come from HR."""
    work = point(sum(work_seconds)/60)
    return AccountedRun(key,start,start+timedelta(minutes=minutes),
        available or AT,miles,role,(point(minutes),cardio,work),
        general_quality_credit(work,role=role,reference=ref),source,
        tuple(WorkBout(seconds,None) for seconds in work_seconds),ref.accounting_key,
        reference_key(ref),'prescribed' if source=='planned' else 'explicit_segments')


def candidates(ref):
    """36 explicit alternatives, not a claimed exhaustive or optimal search."""
    for easy_day,miles,long_day,quality_day in product((4,5,6),(3.8,4.,4.2),(8,9),(12,13)):
        rows = []
        for day,role,distance,minutes,cardio,work in (
            (easy_day,'easy',miles,miles*10,miles*10,()),
            (long_day,'long',8.4,84.,84.,()),
            (quality_day,'quality',4.,40.,60.,(300.,300.,300.)),
            (16,'easy',4.4,44.,44.,()),(19,'long',8.4,84.,84.,()),
            (23,'quality',4.,40.,60.,(300.,300.,300.)),(27,'easy',4.4,44.,44.,()),
            (30,'long',8.4,84.,84.,())):
            rows.append(physical(f'future:{day}',AT+timedelta(days=day,hours=7),distance,
                minutes,point(cardio),role,ref,work_seconds=work))
        yield f'e{easy_day}:{miles}:l{long_day}:q{quality_day}',tuple(rows)


@pytest.mark.parametrize('seconds,miles,hr', [
    (600,4.,140),       # Exactly expected response and execution.
    (500,3.8,140),      # Lower adherence edges, expected cost for actual duration.
    (700,4.2,140),      # Upper adherence edges, expected cost for actual duration.
    (600,4.,170),       # Same prescription, materially higher measured cost.
    (600,4.,110),       # Same prescription, materially lower measured cost.
    (600,4.,None),      # Missing HR must not become zero or certain cost.
])
def test_uploaded_load_reaches_new_evaluator_and_next_day_accounting(setup,seconds,miles,hr,tmp_path):
    sandbox,plan,prior,config = setup
    plan = replace(plan,distance_miles=TargetBand(4.,3.8,4.2,'synthetic adherence bounds'),
        phases=tuple(replace(p,seconds=TargetBand(600.,500.,700.,'synthetic duration bounds')) for p in plan.phases))
    record = sandbox.publish(plan,[prior])
    ref = sandbox.reference
    target_load = record.payload['target_forecast']['predicted_load']
    expected = physical(plan.session_key,plan.suggested_start,4.,50.,point(target_load),
                        'quality',ref,work_seconds=(600.,600.))
    past = (
        physical('prior-long',AT-timedelta(days=6)+timedelta(hours=7),8.4,84.,point(84.),'long',
                 ref,source='observed',available=AT-timedelta(days=5)),
        physical('prior-quality',AT-timedelta(days=3)+timedelta(hours=7),4.,40.,point(60.),'quality',
                 ref,source='observed',available=AT-timedelta(days=2),work_seconds=(300.,)*3),
    )
    context = ProgramContext((),AT.date(),AT.date(),(AT+timedelta(days=21)).date(),
        (AT+timedelta(days=35)).date(),(40.,40.,15.),TrialPolicy(),DemandScenario(),
        .01,PurposeTimingPolicy(1.),demand_accounting_key=ref.accounting_key)
    end = START+timedelta(seconds=5*seconds)
    uploaded = sandbox.upload(tcx(seconds=5*seconds,miles=miles,hr=hr),fingerprint=record.fingerprint,
        available_at=end,observed_phases=tuple((k,seconds) for k in KINDS),phase_evidence='explicit')
    sandbox.close()
    # Restart, do not use the in-memory upload result as the evaluator's load.
    from scripts.execution_upload_sandbox import ExecutionUploadSandbox
    sandbox.db = ExecutionUploadSandbox(sandbox.root,config).db
    state = sandbox.state(end)
    assert not state['pending'] and len(state['actual'])==1
    saved = state['actual'][0]
    measured_cardio = Quantity(**saved['cardio'])
    actual = physical(saved['key'],datetime.fromisoformat(saved['start']),saved['distance_miles'],
        5*seconds/60,measured_cardio,'quality',ref,source='observed',available=end,
        work_seconds=tuple(s for k,s in saved['phases'] if k=='work'))
    rows = []
    # Independently exercise low/middle/high missing-HR scenarios, not just midpoint.
    for fraction in (0.,.5,1.):
        ctx = replace(context,scenario=replace(context.scenario,bound_fraction=fraction))
        def evaluate(h,p,at):
            return evaluate_continuous(ctx,h,p,reference=ref,as_of=at,timezone='UTC',
                purpose_fraction=.5,history_complete_since=(AT-timedelta(days=90)).date())
        before,after = {},{}
        rolling = {day:{} for day in (1,2,3)}
        for key,remaining in candidates(ref):
            before[key] = evaluate(past,(expected,*remaining),AT)['evaluation'].total_cost
            h,p = reconcile_completion(past,(expected,*remaining),actual,
                                       replaces_key=expected.key,as_of=end)
            evaluated = evaluate(h,p,end)
            after[key] = evaluated['evaluation'].total_cost
            consumed = [r for r in evaluated['evaluation'].conditional_ledger if r.key==actual.key]
            assert len(consumed)==1 and consumed[0].demand[1]==measured_cardio
            assert all(r.key!=expected.key for r in evaluated['evaluation'].conditional_ledger)
            assert evaluated['constraint_eligible']
            for day in (1,2,3):
                # No missing run has occurred yet: earliest future candidate is day 4.
                refreshed = evaluate(h,p,AT+timedelta(days=day))['evaluation'].total_cost
                rolling[day][key] = refreshed
        base = next(iter(before))
        relative_change = max(abs((after[k]-after[base])-(before[k]-before[base])) for k in before)
        old = min(before,key=lambda k:(before[k],k))
        new = min(after,key=lambda k:(after[k],k))
        absolute_refresh = max(abs(costs[k]-after[k]) for costs in rolling.values() for k in after)
        relative_refresh = max(abs((costs[k]-costs[base])-(after[k]-after[base]))
                               for costs in rolling.values() for k in after)
        refresh_winners = [min(costs,key=lambda k:(costs[k],k)) for costs in rolling.values()]
        rows.append(dict(demand_fraction=fraction,candidates=len(before),winner_before=old,winner_after=new,
            winner_changed=old!=new,max_relative_score_change=relative_change,
            max_absolute_refresh_score_change=absolute_refresh,
            max_relative_refresh_score_change=relative_refresh,refresh_winners=refresh_winners,
            before_score=before[old],after_score=after[new]))
        # The first rollover has the identical next-day decision boundary.
        assert max(abs(rolling[1][k]-after[k]) for k in after) < 1e-10
        # Later elapsed rest days remove common sunk timing loss. Absolute score
        # invariance was an invalid initial assertion; candidate gaps/ranks are
        # the invariants. This does not alter the evaluator or excuse churn.
        assert relative_refresh < 1e-10 and all(k==new for k in refresh_winners)
        if (seconds,miles,hr)==(600,4.,140):
            assert relative_change < 1e-10 and old==new
        if hr in (110,170):
            assert relative_change > 1e-8  # Integration must not mask genuine changed cost.
    report = dict(seconds_per_phase=seconds,miles=miles,hr=hr,
        comparison=state['comparisons'][0],target_load=target_load,actual_load=saved['cardio'],
        cases=rows,scope='fixed-candidate new-evaluator seam; not daily calendar search or physiology validation')
    (tmp_path/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(seconds=seconds,miles=miles,hr=hr,response=uploaded['comparison']['response_status'],
        routine=uploaded['comparison']['routine_variation_without_new_replan_signal'],
        actual_load=saved['cardio'],cases=rows)))
