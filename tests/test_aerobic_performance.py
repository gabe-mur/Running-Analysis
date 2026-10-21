from copy import deepcopy
from datetime import timedelta
import sqlite3

import numpy as np
import pytest

from run_analysis import aerobic_performance as service
from run_analysis.read_cache import invariant_snapshot
from scripts.aerobic_local_trend import LocalPaceTrend as ResearchCurve
from tests.test_aerobic_candidate import report
from tests.test_pace_comparability import config


def test_native_trend_matches_audited_direct_regression_exactly():
    t=np.array([-70.,-60.,-40.,-35.,-20.,-5.,0.])
    values=dict(y=np.log([10.,9.8,10.1,9.7,9.9,9.8,9.6]), measurement=np.full(7,.01**2),
        gradients=np.array([[.01*i,.02] for i in range(7)]), covariance=np.diag([.01,.02]), times=t)
    native=service.LocalPaceTrend(**values);research=ResearchCurve(**values)
    for anchor in np.arange(-70.,1.,.5):
        assert native.level(anchor)==research.level(anchor)


def test_current_and_historical_values_are_independent_of_viewport_and_clock(config,tmp_path):
    evidence,end=report(config);before=deepcopy(evidence)
    path=tmp_path/'cache.sqlite'
    with sqlite3.connect(path) as db:
        db.execute('create table marker(value integer)')
    with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True) as db:
        with invariant_snapshot(db):
            a=service.progress_from_comparability(db,evidence,as_of=end,history_days=14)
            b=service.progress_from_comparability(db,evidence,as_of=end+timedelta(days=1),history_days=365)
            assert a['current']==b['current']
            assert a['progress_trend']==b['progress_trend']
            assert a['unadjusted']['progress_trend']==b['unadjusted']['progress_trend']
            assert a['smoothing_comparison']['current']==b['smoothing_comparison']['current']
            assert a['unadjusted']['current']==b['unadjusted']['current']
            shared={r['as_of']:r for r in b['history']}
            assert all(r==shared[r['as_of']] for r in a['history'])
            comparison={r['as_of']:r for r in b['smoothing_comparison']['history']}
            assert all(r==comparison[r['as_of']] for r in a['smoothing_comparison']['history'])
            raw={r['as_of']:r for r in b['unadjusted']['history']}
            assert all(r==raw[r['as_of']] for r in a['unadjusted']['history'])
            assert a['observations']==b['observations'][-len(a['observations']):]
            assert b['days_since_observation']==1.
            assert db.total_changes==0
    assert evidence==before


def test_unadjusted_curve_uses_recorded_sections_with_identical_smoothing_weights(config):
    from datetime import datetime
    from run_analysis.aerobic_candidate import measurement_inputs
    from run_analysis.pace_temperature import comparison_covariance
    evidence,end=report(config);before=deepcopy(evidence);result=service._fit(evidence)
    rows=sorted(evidence['runs'],key=lambda r:r['start_time'])
    times=np.array([(datetime.fromisoformat(r['start_time'])-end).total_seconds()/86400 for r in rows])
    covariance=comparison_covariance(evidence['calibration'])
    measurement,gradients=measurement_inputs(rows,covariance)
    expected=service.LocalPaceTrend(np.log([r['observed_pace_min_mile'] for r in rows]),
        measurement,gradients,covariance,times,bandwidth_days=28.,gap_days=28.)
    assert result['unadjusted']['current']['pace_min_mile']==pytest.approx(expected.level(0.)['pace_min_mile'])
    for point in result['unadjusted']['history']:
        anchor=(datetime.fromisoformat(point['as_of'])-end).total_seconds()/86400
        assert point['available']==expected.level(anchor)['available']
        if point['available']:
            assert point['pace_min_mile']==pytest.approx(expected.level(anchor)['pace_min_mile'])
        assert 'conditional_variance' not in point
    for row,t in zip(result['observations'],times,strict=True):
        assert row['unadjusted_trend_pace_min_mile']==pytest.approx(expected.level(t)['pace_min_mile'])
    assert evidence==before


def test_missing_recorded_pace_is_not_replaced_by_adjusted_pace(config):
    evidence,end=report(config);base=service._fit(evidence)
    for row in evidence['runs']:
        row['observed_pace_min_mile']=None
    result=service._fit(evidence)
    assert not result['unadjusted']['available']
    assert all(not r['available'] for r in result['unadjusted']['history'])
    assert all(r['unadjusted_trend_pace_min_mile'] is None for r in result['observations'])
    assert result['current']==base['current']
    assert result['history']==base['history']


@pytest.mark.parametrize('slope,direction',[(-.002,'improving'),(0.,'maintaining'),(.002,'declining')])
def test_headline_direction_describes_the_broad_run_trajectory_without_changing_the_curve(slope,direction):
    t=np.arange(-63.,1.,7.)
    curve=service.LocalPaceTrend(np.log(10.)+slope*t,np.full(len(t),.005**2),
        np.zeros((len(t),1)),np.zeros((1,1)),t)
    trend=curve.progress_trend()
    assert trend['direction']==direction
    assert trend['log_pace_change_per_day']==pytest.approx(slope,abs=1e-13)
    assert trend['span_days']==63
    assert trend['pace_change_seconds_per_mile']==pytest.approx(600*(1-np.exp(-slope*63)),abs=1e-10)
    assert not trend['early_estimate']
    assert curve.level(0.)['pace_min_mile']==pytest.approx(10.)


def test_headlines_require_current_segment_support_and_keep_noisy_direction_inconclusive():
    def curve(t,y):
        return service.LocalPaceTrend(y,np.full(len(t),.02**2),np.zeros((len(t),1)),np.zeros((1,1)),t)
    assert curve([0.],[2.3]).progress_trend()['direction']=='insufficient_data'
    assert curve([-2.,-1.,0.],[2.3,2.29,2.28]).progress_trend()['direction']=='insufficient_data'
    assert curve([-100.,-90.,-80.,0.],[2.3,2.2,2.1,2.]).progress_trend()['direction']=='insufficient_data'
    t=np.arange(-63.,1.,7.)
    noisy=curve(t,np.log(10.)+np.array([-.08,.08,-.08,.08,-.08,.08,-.08,.08,-.08,.08]))
    assert noisy.progress_trend()['direction']=='maintaining'


def test_progress_assessment_uses_available_history_and_marks_shorter_than_eight_weeks_early():
    for span,early in ((28.,True),(49.,True),(56.,False),(84.,False)):
        t=np.linspace(-span,0,10)
        curve=service.LocalPaceTrend(np.log(10.)-.002*t,np.full(10,.005**2),
            np.zeros((10,1)),np.zeros((1,1)),t)
        trend=curve.progress_trend()
        assert trend['direction']=='improving'
        assert trend['early_estimate'] is early
        assert trend['span_days']==span


def test_broad_headline_does_not_revert_to_latest_tangent_or_require_a_previous_block():
    t=np.arange(-84.,1.,7.)
    # Strong progress through the block, with a modest recent slowdown.
    y=np.log(11.)-.002*(t+84)+.004*np.maximum(0,t+21)
    curve=service.LocalPaceTrend(y,np.full(len(t),.005**2),np.zeros((len(t),1)),np.zeros((1,1)),t)
    trend=curve.progress_trend()
    assert trend['direction']=='improving'
    assert trend['run_count']==13
    assert curve.level(0.)['log_pace']>curve.level(-1.)['log_pace']


def test_older_observations_cannot_change_the_twelve_week_assessment():
    t=np.arange(-168.,1.,7.)
    y=np.log(10.)-.001*t
    def assess(values):
        return service.LocalPaceTrend(values,np.full(len(t),.005**2),
            np.zeros((len(t),1)),np.zeros((1,1)),t).progress_trend()
    base=assess(y)
    altered=y.copy();altered[t<-84]+=1
    assert assess(altered)==base
    assert base['span_days']==84
    assert base['run_count']==13


def test_one_run_has_one_point_no_invented_pre_run_trend(config):
    evidence,end=report(config);evidence['runs']=evidence['runs'][-1:]
    fit=service._fit(evidence)
    assert fit['current']['run_count']==1
    assert len(fit['history'])==len(fit['observations'])==1
    assert len(fit['smoothing_comparison']['history'])==1
    assert fit['smoothing_comparison']['current']['pace_min_mile']==pytest.approx(evidence['runs'][0]['pace_min_mile'])
    assert fit['current']['pace_min_mile']==pytest.approx(evidence['runs'][0]['pace_min_mile'])
    assert 'estimated_range_95_min_mile' not in fit['current']


def test_gap_is_missing_not_flat_or_invented_fitness_loss(config):
    evidence,end=report(config);evidence['runs']=[evidence['runs'][0],evidence['runs'][-1]]
    fit=service._fit(evidence)
    assert any(not r['available'] and 'pace_min_mile' not in r for r in fit['history'])
    assert any(not r['available'] and 'pace_min_mile' not in r for r in fit['smoothing_comparison']['history'])
    assert any(not r['available'] and 'pace_min_mile' not in r for r in fit['unadjusted']['history'])
    assert fit['current']['run_count']==1


def test_clean_outlier_is_retained_without_new_robustness_gate(config):
    evidence,end=report(config)
    evidence['runs'][10]['pace_min_mile']*=.85
    fit=service._fit(evidence)
    assert len(fit['observations'])==len(evidence['runs'])
    observed=fit['observations'][10]
    assert observed['pace_min_mile']==evidence['runs'][10]['pace_min_mile']
    assert observed['trend_pace_min_mile']!=observed['pace_min_mile']


def test_sampling_and_pause_descriptions_do_not_change_score_or_trend(config):
    evidence,end=report(config);base=service._fit(evidence)
    for r in evidence['runs']:
        r['sampling_support']={'broader_sampling_difference_seconds':17.,'retained_minutes':20.}
        r['continuity']={'long_stop_count':1,'restart_moving_minutes':[10.]}
    changed=service._fit(evidence)
    assert changed['current']==base['current']
    assert changed['history']==base['history']
    assert changed['observations'][0]['continuity']['long_stop_count']==1


def test_section_screen_remains_visible_and_exclusions_do_not_become_bad_fitness_scores(config):
    evidence,end=report(config)
    evidence['section_screen']={'version':'settled-effort-context-v1','excluded_runs':[
        {'activity_id':999,'start_time':end.isoformat(),'reason':'insufficient_settled_sections'}]}
    result=service._fit(evidence)
    assert result['section_screen']==evidence['section_screen']
    assert not any(r['activity_id']==999 for r in result['observations'])


def test_new_reader_does_not_require_months_or_a_prior_comparison(config):
    evidence,end=report(config);evidence['runs']=evidence['runs'][-2:]
    fit=service._fit(evidence)
    assert fit['available']
    assert 'change' not in fit and 'change_90d' not in fit
    assert 'fitness_level' not in fit
    assert service._fit({'runs':[]})['available'] is False


def test_input_validation_rejects_duplicate_physical_runs_and_nonfinite_precision(config):
    evidence,end=report(config);evidence['runs'].append(evidence['runs'][0])
    with pytest.raises(ValueError,match='Duplicate physical'):
        service._fit(evidence)
    with pytest.raises(ValueError,match='positive precision'):
        service.LocalPaceTrend([2.],[float('nan')],[[0.]],[[0.]],[0.])


def test_smoothing_comparison_reuses_identical_scores_precision_and_default_fit(config):
    from run_analysis.aerobic_candidate import measurement_inputs
    from run_analysis.pace_temperature import comparison_covariance
    from datetime import datetime
    evidence,end=report(config);before=deepcopy(evidence);result=service._fit(evidence)
    rows=sorted(evidence['runs'],key=lambda r:r['start_time'])
    times=np.array([(datetime.fromisoformat(r['start_time'])-end).total_seconds()/86400 for r in rows])
    covariance=comparison_covariance(evidence['calibration'])
    measurement,gradients=measurement_inputs(rows,covariance)
    values=np.log([r['pace_min_mile'] for r in rows])
    for bandwidth,key in ((28.,None),(14.,'smoothing_comparison')):
        expected=service.LocalPaceTrend(values,measurement,gradients,covariance,times,bandwidth_days=bandwidth)
        actual=result if key is None else result[key]
        assert actual['current']==expected.level(0.)
        for point in actual['history']:
            anchor=(datetime.fromisoformat(point['as_of'])-end).total_seconds()/86400
            assert point['pace_min_mile']==pytest.approx(expected.level(anchor)['pace_min_mile'])
    assert [r['pace_min_mile'] for r in result['observations']]==[r['pace_min_mile'] for r in rows]
    assert evidence==before
