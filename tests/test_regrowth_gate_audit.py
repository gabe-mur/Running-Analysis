from dataclasses import asdict

from run_analysis.adherence_projection import ProjectionReplan
from scripts.audit_regrowth_gates import check_candidate
from test_typed_continuation import fixture


def test_gate_audit_does_not_copy_incumbent_mileage_readiness_or_peak():
    _,make=fixture()
    days,states=make(21,{0:('easy',1.)})
    saved=ProjectionReplan(generated_at=states[0].as_of,opening_load_ratio=None,
        target_low_miles=17.,target_high_miles=18.,planned_sessions=(),committed_sessions=(),
        planning_mode='established',ordinary_easy_midpoint_miles=3.5,
        capacity_reference_miles=18.,peak_projected_continuous_mileage_rate=999.)
    inputs=dict(daily_states=states,config={},expected_target_projector=lambda p,n:[(17.,18.)]*n)
    result=check_candidate(asdict(saved),inputs,days)
    assert result['miles']==1.
    assert result['peak_visible']!=999.
    assert 'hard_horizon_underfunded' in {x['code'] for x in result['failures']}
    assert result['integrated_minimum']==51.
    assert 'future daily/upload churn' in result['unchecked']
    not_ready=[days[0].model_copy(update={'recommendation':days[0].recommendation.model_copy(
        update={'readiness':type(days[0].recommendation.readiness)('not_ready')})}),*days[1:]]
    changed=check_candidate(asdict(saved),inputs,not_ready)
    assert 'selected_not_ready_run' in {x['code'] for x in changed['failures']}
