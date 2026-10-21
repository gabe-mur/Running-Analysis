from dataclasses import asdict
from datetime import timedelta

from scripts.audit_seven_day_stability import assess
from scripts.audit_backward_controls import plain
from run_analysis.projection_gate import ProjectionGateConfig
from test_projection_gate import _replan, _session, OPENING


def report(snapshots,complete=True):
    return dict(replans=plain(snapshots),gate_policy=asdict(ProjectionGateConfig(enforce_stability=True)),
        stopped_early=not complete,realized_execution=dict(complete=complete),failures=[])


def test_day_six_distance_edit_is_found_outside_old_four_day_gate():
    before=_replan(_session(6,midpoint_miles=4.),target=(0.,0.),planning_mode='recovery')
    after=_replan(_session(6,midpoint_miles=3.3),target=(0.,0.),planning_mode='recovery',generated_at=OPENING+timedelta(hours=1))
    result=assess(report([before,after]))
    assert any(f.code=='no_evidence_distance_churn' for f in result['seven_day_failures'])
    assert result['churn']['4']['distance_changes']==0
    assert result['churn']['7']['distance_changes']==1
    assert not result['passed_seven_day_gates']


def test_short_passing_replay_is_not_seven_day_clearance():
    result=assess(report([_replan(target=(0.,0.),planning_mode='recovery')]))
    assert result['seven_day_failures']==[]
    assert not result['complete_seven_day_screen']
    assert not result['zero_seven_day_edits']


def test_small_edits_remain_visible_below_existing_failure_threshold():
    before=_replan(_session(6,midpoint_miles=4.),target=(0.,0.),planning_mode='recovery')
    after=_replan(_session(6,midpoint_miles=3.8),target=(0.,0.),planning_mode='recovery',generated_at=OPENING+timedelta(hours=1))
    result=assess(report([before,after]))
    assert result['seven_day_failures']==[]
    assert result['churn']['7']['distance_changes']==1


def test_seven_refresh_dates_with_a_missing_day_are_not_complete_coverage():
    snapshots=[_replan(target=(0.,0.),planning_mode='recovery',generated_at=OPENING+timedelta(days=d))
        for d in (0,1,2,3,4,5,7)]
    result=assess(report(snapshots))
    assert not result['consecutive_daily_refreshes']
    assert not result['complete_seven_day_screen']
