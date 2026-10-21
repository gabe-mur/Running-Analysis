from copy import deepcopy
import numpy as np
import pytest

from scripts.audit_aerobic_three_layers import inputs,measurement_inputs,simulations
from scripts.audit_progress_responsiveness import at
from tests.test_small_aerobic_changes import rows


def fixture():
    source=rows()
    for row in source:row['shared_log_speed_gradient']=[.1,.2,.3,1.]
    report=dict(runs=source,calibration=dict(covariance=(np.eye(3)*.00001).tolist(),heat=dict(sd=.001)))
    inventory=dict(runs=[dict(activity_id=r['activity_id'],uncertainty_95_min_mile=r['uncertainty_95_min_mile'],
        sampling_precision=[dict(sampling_sd_min_mile=.05)],log_variance_parts=dict(parameter=.0001,
            whole_run_dispersion=.01,missing_conditions=.0001,terrain_transfer=0.,wind_transfer=0.)) for r in source])
    return report,inventory,at(source[-1])


def test_input_layers_preserve_floor_and_fixed_periods_without_mutating_evidence():
    report,inventory,end=fixture();before=deepcopy(report)
    selected,m,g,groups,latest,used,eligible=inputs(report,inventory,end)
    assert set(groups)=={0,1,2,3}
    assert all(at(r)<=end for r in selected)
    cov=np.zeros((4,4));cov[:3,:3]=report['calibration']['covariance'];cov[3,3]=.001**2
    shared=np.einsum('ij,jk,ik->i',g,cov,g)
    pace=np.array([r['pace_min_mile'] for r in selected])
    assert np.all((m+shared)*pace**2>=(10/60)**2-1e-12)
    assert report==before


def test_stale_precision_inventory_is_rejected():
    report,inventory,end=fixture()
    for v in inventory['runs']:v['uncertainty_95_min_mile']=100
    with pytest.raises(ValueError,match='stale'):inputs(report,inventory,end)


def test_measurement_inputs_are_independent_of_period_grouping_and_evidence_frame():
    report,inventory,end=fixture()
    selected,m,g,groups,latest,used,eligible=inputs(report,inventory,end)
    all_m,all_g,all_used=measurement_inputs(report,inventory,eligible)
    index={r['activity_id']:i for i,r in enumerate(eligible)}
    mask=[index[r['activity_id']] for r in selected]
    np.testing.assert_array_equal(all_m[mask],m)
    np.testing.assert_array_equal(all_g[mask],g)
    assert [all_used[i] for i in mask]==used


def test_all_control_scenarios_retained_and_simulation_is_reproducible():
    report,inventory,end=fixture()
    selected,m,g,groups,latest,used,eligible=inputs(report,inventory,end)
    cov=np.zeros((4,4))
    a=simulations(selected,m,g,cov,groups,latest,eligible,2,123)
    b=simulations(selected,m,g,cov,groups,latest,eligible,2,123)
    assert a==b and len(a['cells'])==40
    assert len({v['scenario'] for v in a['cells']})==10
    assert {v['gain_percent'] for v in a['cells']}=={0,.5,1,2}
