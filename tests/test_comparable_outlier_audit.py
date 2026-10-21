from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import pytest

from run_analysis.config import load_config
from run_analysis.pace_comparability import compare_run_windows, fit_comparison_calibration
from scripts.audit_comparable_outliers import weighted, intervals_summary, window_audit
from tests.test_pace_comparability import row


def test_missing_values_are_not_treated_as_zero_or_an_independent_gps_match():
    assert weighted([None,10.,20.],[100.,1.,1.])==15.
    assert weighted([None],[1.]) is None
    result=intervals_summary([])
    assert result['pace_min_mile'] is None and result['hr_bpm'] is None
    assert result['gps_device_distance_ratio'] is None


def windows(config,minutes):
    return [row(config,moving_minutes_into_run=float(minute),heart_rate_change_bpm=0.,speed_change_fraction=0.)
        for minute in minutes]


def canonical(rows,config,calibration):
    from run_analysis.objective_modeling import _primary_window_weight
    return compare_run_windows(rows,[_primary_window_weight(w,config) for w in rows],config,calibration)


def test_short_run_reference_support_is_diagnostic_not_an_automatic_source_rejection():
    config=load_config(Path(__file__).parents[1]/'config.example.yaml')
    calibration=fit_comparison_calibration([],config);rows=windows(config,[5.,6.,7.])
    expected=canonical(rows,config,calibration);before=deepcopy((rows,config,calibration,expected))
    audit=window_audit(rows,expected,config,calibration)
    assert audit['reference_support']['basis']=='unsupported_extrapolation'
    assert not audit['reference_support']['supported']
    assert audit['reference_support']['gap_minutes']==13.
    assert audit['scenarios']['near_twenty'] is None
    assert audit['scenarios']['current']['pace_min_mile']>0
    assert (rows,config,calibration,expected)==before


def test_near_reference_and_uniform_diagnostics_use_all_supplied_windows_not_best_pace():
    config=load_config(Path(__file__).parents[1]/'config.example.yaml')
    calibration=fit_comparison_calibration([],config);rows=windows(config,[5.,10.,17.,20.,23.,30.])
    result=window_audit(rows,canonical(rows,config,calibration),config,calibration)
    assert result['reference_support']['basis']=='interpolation'
    assert result['usable_windows']==6
    assert len(result['window_profile'])==6
    assert sum(r['relative_weight'] for r in result['window_profile'])==pytest.approx(1.)
    assert result['scenarios']['near_twenty']['pace_min_mile']==pytest.approx(10.)
    assert result['scenarios']['uniform_windows']['pace_min_mile']==pytest.approx(10.)


def test_stale_normalized_input_cannot_silently_supply_a_different_comparison():
    config=load_config(Path(__file__).parents[1]/'config.example.yaml')
    calibration=fit_comparison_calibration([],config);rows=windows(config,[10.,20.,30.])
    with pytest.raises(ValueError,match='disagree'):
        window_audit(rows,dict(pace_min_mile=9.),config,calibration)
