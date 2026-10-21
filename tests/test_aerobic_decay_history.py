from copy import deepcopy
import math

import numpy as np
import pytest

from scripts.aerobic_decay_history import PrecisionDecayHistory
from scripts.audit_aerobic_decay import METHODS, model, summaries, paired_blocks, forward_check


def data():
    return dict(y=np.log([10.,9.,11.,10.]),measurement=np.full(4,.01**2),gradients=np.zeros((4,1)),
        covariance=np.array([[.02**2]]),times=np.array([-60.,-28.,-7.,0.]))


def test_exact_exponential_positive_weights_and_no_28_day_drop():
    source=data();curve=PrecisionDecayHistory(**source)
    weights=np.exp2(source['times']/14);weights/=weights.sum()
    point=curve.level(0.)
    np.testing.assert_allclose(point['weights'],weights)
    assert all(w>0 for w in point['weights'])
    assert point['pace_min_mile']==pytest.approx(math.exp(weights@source['y']))
    assert point['old_evidence_weight']>0


def test_without_new_runs_normalized_level_does_not_drift_but_staleness_grows():
    curve=PrecisionDecayHistory(**data())
    assert curve.level(1000.)['pace_min_mile']==curve.level(0.)['pace_min_mile']
    assert curve.level(1000.)['days_since_observation']==1000
    assert curve.level(-100.)['available'] is False


def test_future_observations_cannot_change_past_level_and_input_remains_immutable():
    source=data();before=deepcopy(source);curve=PrecisionDecayHistory(**source)
    expected=curve.level(-7.)
    source['y'][-1]=100.
    assert PrecisionDecayHistory(**source).level(-7.)==expected
    for key in before:
        if key!='y':np.testing.assert_array_equal(source[key],before[key])
    source['y'][:]=-100.
    assert curve.level(-7.)==expected
    changed=curve.level(-7.);changed['weights'][0]=99
    assert curve.level(-7.)==expected


def test_shared_parameter_uncertainty_does_not_average_away():
    source=data();source['gradients'][:]=1.
    result=PrecisionDecayHistory(**source).level(0.)
    w=np.array(result['weights'])
    assert result['conditional_variance']==pytest.approx((w*w)@source['measurement']+.02**2)
    assert '95' not in result['uncertainty_kind']


def test_sparse_initial_estimate_preserves_the_observation_not_zero_variance():
    source=dict(y=[math.log(10)],measurement=[.01**2],gradients=[[1.]],covariance=[[.02**2]],times=[0.])
    result=PrecisionDecayHistory(**source).level(0.)
    assert result['pace_min_mile']==pytest.approx(10.)
    assert result['effective_run_count']==1
    assert result['conditional_variance']==pytest.approx(.01**2+.02**2)


def test_uniform_change_is_preserved_and_partial_recent_change_is_not_instantaneous():
    source=data();curve=PrecisionDecayHistory(**source)
    changed=deepcopy(source);changed['y']-=math.log1p(.02)
    assert curve.level(0.)['pace_min_mile']/PrecisionDecayHistory(**changed).level(0.)['pace_min_mile']==pytest.approx(1.02)
    partial=deepcopy(source);partial['y'][-1]-=math.log1p(.02)
    response=curve.level(0.)['pace_min_mile']/PrecisionDecayHistory(**partial).level(0.)['pace_min_mile']-1
    assert 0<response<.02


def test_boxcar_ablation_has_the_same_marginal_weights_but_exact_original_cutoff():
    point=PrecisionDecayHistory(**data(),boxcar_days=28).level(0.)
    assert point['indices']==[2,3]
    assert point['weights']==[.5,.5]
    assert point['old_evidence_weight']==0


@pytest.mark.parametrize('changes',[dict(half_life_days=0),dict(half_life_days=float('nan')),
    dict(boxcar_days=-1),dict(times=[0.,-1.,-2.,-3.]),dict(measurement=[0.,1.,1.,1.]),dict(covariance=[[-1.]])])
def test_invalid_data_fails_explicitly(changes):
    with pytest.raises(ValueError):PrecisionDecayHistory(**{**data(),**changes})


def test_registered_methods_no_silent_hyperparameter_search_and_paired_denominators():
    assert model(data(),'decay_14d').half_life==14
    with pytest.raises(ValueError):model(data(),'decay_13d')
    cases=[dict(as_of=f'2026-{month:02d}-01T00:00:00+00:00',
        errors_seconds_per_mile={name:(10 if name=='current_28d' else 9) for name in METHODS}) for month in range(1,5)]
    summary=summaries(cases);paired=paired_blocks(cases,draws=50)
    assert all(value['n']==4 for value in summary.values())
    assert paired['decay_14d']['mae_difference_seconds_per_mile']==-1
    assert paired['decay_14d']['descriptive_95_block_range']==[-1.,-1.]


def test_forward_adjustments_exclude_the_target_and_all_later_runs(monkeypatch):
    from scripts import audit_aerobic_decay as audit
    from tests.test_pace_comparability import row
    from run_analysis.config import load_config
    from pathlib import Path
    config=load_config(Path(__file__).parents[1]/'config.example.yaml')
    primary=[];runs=[];selected={};seen_calibration=[];seen_models=[]
    for i in range(7):
        when=f'2026-10-{i+1:02d}T00:00:00+00:00'
        window=row(config,activity_id=i,start_time_utc=when)
        primary.append(window);selected[i]=(i,[window])
        runs.append(dict(activity_id=i,start_time=when,trend_weight=1.,included_in_trend=True,
            terrain_coverage=1.,weather_coverage=1.,sampled_hr_bpm=148.))
    report=dict(runs=runs);before=deepcopy((report,primary,selected,config))
    def calibrate(rows,config):
        identities=[r['activity_id'] for r in rows];seen_calibration.append(identities)
        return dict(covariance=np.zeros((3,3)).tolist(),heat=dict(sd=.01),contributing_runs=len(identities))
    def normalize(rows,weights,config,calibration,**kwargs):
        return dict(pace_min_mile=10.,uncertainty_95_min_mile=.2,shared_log_speed_gradient=[0.]*4,
            comparison_precision=dict(sampling_precision=[],log_variance_parts={}))
    class FakeModel:
        def level(self,anchor):return dict(pace_min_mile=10.)
    def fitted(data,name):
        seen_models.append(len(data['y']));return FakeModel()
    monkeypatch.setattr(audit.pc,'fit_comparison_calibration',calibrate)
    monkeypatch.setattr(audit.pc,'compare_run_windows',normalize)
    monkeypatch.setattr(audit,'model',fitted)
    result=forward_check(report,primary,selected,config)
    assert seen_calibration==[list(range(4)),list(range(5)),list(range(6))]
    assert seen_models==[4]*len(METHODS)+[5]*len(METHODS)+[6]*len(METHODS)
    assert [case['activity_id'] for case in result['cases']]==[4,5,6]
    assert (report,primary,selected,config)==before


def test_svg_comparison_needs_no_plotting_dependency_and_excludes_old_axis_outliers(tmp_path):
    from scripts.audit_aerobic_decay import plot, history_diagnostics
    from xml.etree import ElementTree
    points=[dict(as_of='2020-01-01T00:00:00+00:00',pace_min_mile=100.),
        dict(as_of='2026-10-10T00:00:00+00:00',pace_min_mile=10.5),
        dict(as_of='2026-10-14T00:00:00+00:00',pace_min_mile=10.)]
    histories={name:deepcopy(points) for name in METHODS}
    path=tmp_path/'comparison.svg';plot(dict(histories=histories),path)
    root=ElementTree.fromstring(path.read_text())
    assert root.attrib['viewBox']=='0 0 1200 580'
    assert '10:00' in path.read_text() and '100:00' not in path.read_text()
    with pytest.raises(ValueError):plot(dict(histories=histories),tmp_path/'bad.png')
    sparse=history_diagnostics({name:rows[-1:] for name,rows in histories.items()})
    assert all(value['maximum_step'] is None for value in sparse.values())
