from copy import deepcopy
from pathlib import Path

import pytest

from run_analysis.config import load_config
from run_analysis.pace_comparability import fit_comparison_calibration
from scripts.audit_outlier_hr_curves import matched_sections, section, summarize_matches, hr_bands, after_warmup
from tests.test_pace_comparability import row


def sample(minute=10, hr=148, raw=10, environmental=10, half='early'):
    return dict(minute=minute, hr=hr, raw=raw, environmental=environmental, seconds=120,
                stability=1, half=half, heat_c=15, grade_ratio=1, wind_log=0)


def test_no_hr_extrapolation_or_single_peer_support():
    target = [sample()]
    assert matched_sections(target, {1: [sample(hr=140)], 2: [sample(hr=140)]}) == []
    assert matched_sections(target, {1: [sample()]*10}) == []


def test_entire_diagnostic_window_must_follow_warmup_not_just_its_center():
    assert not after_warmup(dict(moving_minutes_into_run=5, elapsed_time_s=120, stopped_time_s=0))
    assert after_warmup(dict(moving_minutes_into_run=6, elapsed_time_s=120, stopped_time_s=0))


def test_nearest_context_not_best_pace_selects_peer_section():
    target = [sample()]
    peers = {1: [sample(raw=100), sample(minute=11, raw=10)], 2: [sample(raw=12)]}
    a = matched_sections(target, peers)
    assert a[0]['peer']['raw'] == 56
    peers[1][0]['raw'] = 8
    b = matched_sections(target, peers)
    assert b[0]['peer']['raw'] == 10
    assert a[0]['peer']['minute'] == b[0]['peer']['minute'] == 10


def test_uniform_shift_is_detected_in_both_halves_and_hr_bands():
    target = [sample(minute=10, hr=140, raw=9, environmental=9),
              sample(minute=20, hr=152, raw=9, environmental=9, half='late')]
    peers = {i: [sample(minute=10, hr=140), sample(minute=20, hr=152)] for i in (1, 2, 3)}
    matches = matched_sections(target, peers)
    summary = summarize_matches(matches)
    assert summary['early']['raw_difference_seconds'] == -60
    assert summary['late']['raw_difference_seconds'] == -60
    assert summary['all']['peer_runs'] == 3
    assert summary['all']['matched_minutes'] == 4
    assert len(hr_bands(matches)) == 2


def test_raw_shift_and_environmental_adjustment_are_kept_distinct():
    target = [sample(raw=9, environmental=10)]
    matches = matched_sections(target, {1: [sample()], 2: [sample()]})
    assert matches[0]['raw_difference_seconds'] == -60
    assert matches[0]['environmental_difference_seconds'] == 0


def test_summary_weights_minutes_not_reference_or_stability_preferences():
    target = [sample(minute=10, raw=9, half='early'), sample(minute=20, raw=10, half='late')]
    target[0]['stability'] = .01
    peers = {i: [sample(minute=10), sample(minute=20)] for i in (1, 2)}
    summary = summarize_matches(matched_sections(target, peers))
    assert summary['all']['raw_difference_seconds'] == -30
    assert summary['early']['raw_difference_seconds'] == -60
    assert summary['late']['raw_difference_seconds'] == 0


def test_closer_conditions_require_actual_weather_not_unknown_as_cool():
    target = [sample()]
    peer = sample(); peer['heat_c'] = None
    assert matched_sections(target, {1: [peer], 2: [peer]}, closer_conditions=True) == []


def test_empty_summary_is_missing_not_zero_difference():
    value = summarize_matches([])
    assert value['all']['windows'] == 0
    assert value['all']['raw_difference_seconds'] is None
    assert value['all']['hr_range'] is None


def test_environmental_section_never_applies_hr_or_time_normalization():
    config = load_config(Path(__file__).parents[1]/'config.example.yaml')
    calibration = fit_comparison_calibration([], config)
    w = row(config, average_hr_bpm=160., moving_minutes_into_run=40., elapsed_time_s=120., stopped_time_s=0.)
    a = section(w, config, calibration, 20)
    changed = deepcopy(calibration)
    changed['parameters'][0] = 100
    changed['parameters'][1] = 100
    b = section(w, config, changed, 20)
    assert a['raw'] == b['raw']
    assert a['environmental'] == b['environmental']
    assert a['half'] == 'late'
