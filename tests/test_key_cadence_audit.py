from datetime import date
import pytest

from scripts.check_rollover_search_repair import cadence_case
from scripts.audit_key_cadence import mean_key_entry
from run_analysis.adaptive.shared_program_evaluator import ProgramContext
from run_analysis.adaptive.program_objective_trial import TrialPolicy,DemandScenario,EasyDurationPolicy
from run_analysis.adaptive.timely_purpose_trial import PurposeTimingPolicy
from run_analysis.program_delivery import TargetSegment
from run_analysis.workload_accounting import Quantity


def fixture():
    q=tuple(Quantity.point(1.,'test') for _ in range(3))
    context=ProgramContext(history=(),stage_start=date(2026,9,1),decision=date(2026,10,5),
        hard_end=date(2026,10,26),end=date(2026,11,28),scales=(1.,1.,1.),
        policy=TrialPolicy(weekly_miles=19.3,long_miles=8.9,purpose_weights=(1.,0.,1.,1.),volume_window_days=28),
        scenario=DemandScenario(),exchange=.01,timing=PurposeTimingPolicy(1.,grace_fraction=1.),
        easy_duration_policy=EasyDurationPolicy(10.,25.,.01,'test',6.5),
        volume_targets=(TargetSegment(date(2026,9,1),date(2026,11,28),19.3),),
        support_targets=(TargetSegment(date(2026,9,1),date(2026,11,28),6.5),),
        soft_horizon_taper=True,delivery_contract='sustained_v1',continuation_guard_days=7,
        continuation_boundary_references=tuple((role,miles,q) for role,miles in [('easy',4.4),('long',8.9),('quality',3.9)]),
        purpose_continuation=True)
    from dataclasses import dataclass
    @dataclass(frozen=True)
    class State:
        context:ProgramContext
    return dict(state=State(context),event={'stage_runtime':{'target_history':[{'support_goal':7.5}]}},
        history='unchanged',handoffs='unchanged')


def test_eight_day_change_preserves_volume_evidence_and_past_support():
    original=fixture(); new=cadence_case(original,8); a=original['state'].context;b=new['state'].context
    assert b.policy.long_per_7==b.policy.quality_per_7==.875
    assert b.policy.weekly_miles==a.policy.weekly_miles
    assert b.volume_targets==a.volume_targets and b.history==a.history and b.scenario==a.scenario
    assert b.end==a.end and b.hard_end==a.hard_end and b.scales==a.scales and b.exchange==a.exchange
    assert b.support_targets[0].miles_per_7_days==6.5
    assert b.support_targets[0].end==b.decision==b.support_targets[1].start
    assert b.support_targets[1].miles_per_7_days==pytest.approx(8.1)
    assert b.easy_duration_policy.supporting_easy_miles_per_7==pytest.approx(8.1)
    assert new['event']['stage_runtime']['target_history'][-1]['support_goal']==pytest.approx(9.1)
    assert original['event']['stage_runtime']['target_history'][-1]['support_goal']==7.5
    assert b.volume_smoothing_days==8 and a.volume_smoothing_days==7
    assert b.taper_start>a.taper_start
    assert new['handoffs']==original['handoffs']


def test_seven_day_identity_and_bad_inputs():
    original=fixture();same=cadence_case(original,7)
    assert same is original
    assert same['state'].context.policy==original['state'].context.policy
    assert same['state'].context.easy_duration_policy==original['state'].context.easy_duration_policy
    for bad in (0,1,8.,True):
        with pytest.raises(ValueError):cadence_case(original,bad)


def test_carryover_averages_use_actual_accounting_axis_names():
    rows=[dict(role='long',preceding_by_axis={'running_minutes':a,'cardio':2*a,'work_minutes':3*a}) for a in (1.,3.)]
    assert mean_key_entry(rows)=={'long':{'running_minutes':2.,'cardio':4.,'work_minutes':6.}}
    assert mean_key_entry([])=={}
