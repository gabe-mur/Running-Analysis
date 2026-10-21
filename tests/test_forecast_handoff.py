from dataclasses import asdict, replace
from datetime import timedelta

import pytest

from run_analysis.workload_accounting import Quantity
from scripts.forecast_handoff import ForecastScope, PhaseRateScenario
from scripts.execution_forecast_contract import issue_execution, compare_execution, save_execution, read_execution
from scripts.program_objective_trial import demand_value
from test_execution_forecast_contract import intent, completed
from test_direct_quality_load import history, AT


def scoped(**changes):
    return replace(intent(),forecast_scope=ForecastScope('controlled_threshold','test time band',
                                                       'all running fixture',**changes))


def rate(kind='work',**changes):
    return PhaseRateScenario(kind,Quantity(1.,4.,2.,'declared test scenario'),AT-timedelta(days=1),
                             'units','controlled_threshold','explicit test prior',**changes)


def test_declared_bounds_reach_handoff_without_becoming_confidence_or_response_allowance():
    record = issue_execution(scoped(),[history()])
    h = record.payload['planning_handoff']
    assert h['usable'] and h['status']=='provisional'
    assert h['running_minutes']['estimate']==5
    assert h['running_minutes']['low']==pytest.approx(250/60)
    assert h['running_minutes']['high']==pytest.approx(350/60)
    assert h['cardio']['estimate']==pytest.approx(5)
    assert not h['uncertainty_calibrated'] and not h['response_allowance_derived']
    assert h['confidence_interval'] is None
    assert record.payload['intent']['load_allowance']==asdict(intent().load_allowance)


def test_unsupported_work_never_borrows_easy_rate_or_becomes_free():
    easy = replace(history(),phases=tuple(p for p in history().phases if p.kind!='work'))
    record = issue_execution(scoped(),[easy])
    assert record.payload['target_forecast']['predicted_load'] is None
    assert not record.payload['planning_handoff']['usable']
    assert record.payload['planning_handoff']['unresolved_phases']==['work','work']


def test_dated_work_rate_fallback_survives_issue_reload_and_upload(tmp_path):
    easy = replace(history(),phases=tuple(p for p in history().phases if p.kind!='work'))
    record = issue_execution(scoped(rate_scenarios=(rate(),)),[easy])
    handoff = record.payload['planning_handoff']
    assert handoff['usable']
    assert handoff['cardio']['estimate']==pytest.approx(7.)  # 3 easy minutes + 2 work at 2/min.
    assert handoff['cardio']['low'] < 7 < handoff['cardio']['high']
    path = tmp_path/'issue.json'
    save_execution(record,path)
    restored = read_execution(path)
    actual = replace(completed(record),cardio=Quantity.point(7.,'fixture actual'))
    result = compare_execution(restored,actual,as_of=actual.available_at)
    assert result['execution_conditioned_load']==pytest.approx(7.)
    assert result['routine_variation_without_new_replan_signal']


@pytest.mark.parametrize('change',[
    dict(available_at=AT+timedelta(seconds=1)),dict(accounting_key='wrong'),dict(intensity_key='different_effort')])
def test_future_or_mismatched_fallback_cannot_be_used(change):
    with pytest.raises(ValueError,match='available and match'):
        issue_execution(scoped(rate_scenarios=(replace(rate(),**change),)),[history()])


def test_different_declared_effort_rates_override_history_only_explicitly():
    scenarios = tuple(rate(kind) for kind in ('warmup','work','recovery','cooldown'))
    forced = issue_execution(scoped(mode='declared_phase_rates',rate_scenarios=scenarios),[history()])
    assert forced.payload['target_forecast']['predicted_load']==pytest.approx(10.)
    fallback = issue_execution(scoped(rate_scenarios=scenarios),[history()])
    assert fallback.payload['target_forecast']['predicted_load']==pytest.approx(5.)
    assert not issue_execution(scoped(mode='declared_phase_rates',rate_scenarios=(rate(),)),[history()]).payload['planning_handoff']['usable']


def test_no_scope_or_duration_box_does_not_claim_supported_handoff():
    assert not issue_execution(intent(),[history()]).payload['planning_handoff']['usable']
    p = scoped()
    p = replace(p,phases=tuple(replace(x,seconds=replace(x.seconds,low=None,high=None)) for x in p.phases))
    assert issue_execution(p,[history()]).payload['planning_handoff']['reason']=='duration_scenarios_not_declared'


def test_sensitivity_uses_nominal_not_midpoint_or_worst_case():
    q = Quantity(1.,9.,3.,'asymmetric scenario')
    assert [demand_value(q,f) for f in (0,.25,.5,.75,1)]==[1.,2.,3.,6.,9.]
    assert demand_value(Quantity(1.,9.),.5)==5.
    assert demand_value(Quantity(0.,99.,3.,'wider only'),.5)==3.


def test_full_score_nominal_survives_wider_bounds_and_expected_completion():
    from test_r3_decision_controls import programme, evaluate, cost, AT as NOW
    plans = [programme(easy=((2,3.5),(6,3.6))),programme(easy=((1,3.5),(5,3.6)))]
    bounded = [tuple(replace(r,demand=(r.demand[0],Quantity(r.demand[1].low*.7,r.demand[1].high*2,
        r.demand[1].low,'asymmetric nominal control'),r.demand[2])) for r in rows) for rows in plans]
    before = [cost(evaluate(rows)) for rows in bounded]
    assert before==pytest.approx([cost(evaluate(rows)) for rows in plans])
    history = programme(cycles=range(-2,0),source='observed')
    actual = replace(plans[0][0],source='observed',available_at=plans[0][0].end,work_evidence_basis='explicit_segments')
    after = [cost(evaluate(rows[1:],history=(*history,actual),as_of=NOW+timedelta(days=1))) for rows in bounded]
    assert before[0]-before[1]==pytest.approx(after[0]-after[1])


def test_duration_command_is_not_execution_bounds_but_workload_sensitivity_remains():
    from test_r3_decision_controls import row, evaluate, contracts, cost, context, q
    from scripts.program_objective_trial import DemandScenario
    run = row(1,4.)
    limits = contracts()
    limits['easy'] = replace(limits['easy'],ordinary=replace(limits['easy'].ordinary,running_minutes=q(40.)))
    bounded = replace(run,demand=(Quantity(36.,52.,40.,'declared duration'),*run.demand[1:]),command_running_minutes=40.)
    result = evaluate((bounded,),supplied=limits)
    assert result['decision_eligible']
    assert cost(result)==pytest.approx(cost(evaluate((run,),supplied=limits)))
    high = evaluate((bounded,),supplied=limits,ctx=replace(context(),scenario=DemandScenario(bound_fraction=1)))
    assert cost(high)>cost(result)
    limits['easy'] = replace(limits['easy'],ordinary=replace(limits['easy'].ordinary,running_minutes=q(39.)))
    assert not evaluate((bounded,),supplied=limits)['decision_eligible']
    with pytest.raises(ValueError,match='Planned duration command'):
        replace(bounded,command_running_minutes=41.)
    with pytest.raises(ValueError,match='Planned duration command'):
        replace(bounded,source='observed',available_at=bounded.end)
