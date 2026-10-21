from copy import deepcopy
from datetime import datetime, timedelta, timezone
import sqlite3

import numpy as np
import pytest

from run_analysis import aerobic_candidate as service
from run_analysis.aerobic_coherent_history import CoherentHistory
from run_analysis.pace_comparability import compare_run_windows, fit_comparison_calibration
from run_analysis.read_cache import invariant_snapshot
from scripts.aerobic_coherent_history import CoherentHistory as ResearchCurve
from scripts.audit_aerobic_three_layers import measurement_inputs as research_measurement
from tests.test_aerobic_coherent_history import data
from tests.test_pace_comparability import row, config


def report(config):
    end=datetime(2026,10,14,tzinfo=timezone.utc)
    calibration=fit_comparison_calibration([],config);runs=[]
    for identity,days in enumerate(range(116,-1,-4)):
        windows=[row(config,moving_minutes_into_run=float(minute),moving_pace_min_mile=(10. if days>=28 else 9.8)+.02*np.sin(minute)) for minute in range(5,41)]
        value=compare_run_windows(windows,[1.]*len(windows),config,calibration,precision_seed=20261014+identity)
        runs.append(dict(value,activity_id=identity,start_time=(end-timedelta(days=days)).isoformat(),included_in_trend=True,trend_weight=1.))
    return dict(calibration=calibration,runs=runs),end


@pytest.fixture
def fast_test(monkeypatch):
    def assessed(self,*args,**kwargs):
        return dict(estimate=.02,covered_95=True,observed_unconverged_kernels=[],optimizer_warning_draws=0,pvalue_bounds=[.1,1.])
    monkeypatch.setattr(service.GaussianChangeProfile,'bootstrap_test',assessed)


def test_promoted_curve_is_numerically_identical_to_audited_research():
    source=data();native=CoherentHistory(**source);old=ResearchCurve(**source)
    assert native.series([0,-28,-90])==old.series([0,-28,-90])
    assert native.compare(0,90)==old.compare(0,90)


def test_live_sampling_and_precision_match_the_frozen_audit_arithmetic(config):
    evidence,end=report(config);rows=evidence['runs']
    cov=np.zeros((4,4));cov[:3,:3]=evidence['calibration']['covariance'];cov[3,3]=evidence['calibration']['heat']['sd']**2
    inventory=dict(runs=[dict(activity_id=r['activity_id'],uncertainty_95_min_mile=r['uncertainty_95_min_mile'],**r['comparison_precision']) for r in rows])
    m,g=service.measurement_inputs(rows,cov)
    expected_m,expected_g,_=research_measurement(evidence,inventory,rows)
    np.testing.assert_array_equal(m,expected_m)
    np.testing.assert_array_equal(g,expected_g)


def test_connected_candidate_matches_curve_without_invented_probabilities_or_ranges(config,fast_test):
    evidence,end=report(config);before=deepcopy(evidence);result=service._fit(evidence)
    assert result['version']==service.VERSION and result['available']
    assert result['change']['speed_change_percent']==pytest.approx(100*(10/9.8-1),abs=.02)
    assert result['change']['zero_change_included'] is True
    assert 'conditional_probability_faster' not in result['change']
    assert 'estimated_change_range_95_percent' not in result['change']
    assert result['change']['direction']=='stable_or_uncertain'
    assert evidence==before


def test_new_user_has_initial_point_and_no_fabricated_comparison(config,fast_test):
    evidence,end=report(config);evidence['runs']=evidence['runs'][-1:]
    result=service._fit(evidence)
    assert result['current']['evidence']=='provisional'
    assert result['prior'] is None and result['change'] is None


def test_cache_reuses_fit_across_viewports_and_clock_changes_without_database_writes(tmp_path,config,fast_test,monkeypatch):
    evidence,end=report(config);path=tmp_path/'cache.sqlite'
    with sqlite3.connect(path) as setup:
        setup.execute('create table marker(value integer)')
    original=service._fit;calls=[]
    def counted(*args):
        calls.append(1);return original(*args)
    monkeypatch.setattr(service,'_fit',counted)
    with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True) as db:
        with invariant_snapshot(db):
            short=service.progress_from_comparability(db,evidence,as_of=end,history_days=14,config=config)
            longer=service.progress_from_comparability(db,evidence,as_of=end+timedelta(days=1),history_days=365,config=config)
            assert len(calls)==1 and short['current']==longer['current'] and short['change']==longer['change']
            assert short['fitness_level']['value']==longer['fitness_level']['value']
            assert longer['days_since_observation']==1.
            assert db.total_changes==0
