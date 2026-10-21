from dataclasses import replace

import numpy as np
import pytest

from run_analysis.aerobic_response_evidence import BalanceBlock, TAUS
from run_analysis.aerobic_hr_timing import statistics, fit, cross_half_loss, aligned_windows, response_corrections
from run_analysis.hr_response_alignment import response_step
from run_analysis.pace_comparability import fit_comparison_calibration
from tests.test_pace_comparability import config, row


def observed(*, gain=1., restarted=False):
    rows = []; state = .6
    for t in range(0, 1800, 10):
        demand = 1.3 if t % 180 < 90 else .75
        if restarted and t == 900:
            state = .45
        mean, end = response_step(state, demand, 10, 60)
        change = end-state
        rows.append(BalanceBlock(t+(300 if restarted and t >= 900 else 0), 10, (t+5)/60,
            demand*2.7*gain*10, mean*10, max(0., change), min(0., change), 10, 0, state, end))
        state = end
    return rows


def test_joint_initial_state_and_response_recovers_level_including_restart():
    for restart in (False, True):
        fitted = fit(statistics(observed(restarted=restart)))
        i = TAUS.index(60.)
        assert fitted['inverse_reference_speed'][i] == pytest.approx(1/2.7, abs=1e-12)
        assert fitted['losses'][i] < 1e-25
        assert int(np.argmin(fitted['losses'])) == i


def test_later_measured_hr_does_not_reset_model_inside_a_bout():
    original = observed()
    changed = [r if i == 0 else replace(r, first_hr_effort=.2, last_hr_effort=.2) for i, r in enumerate(original)]
    a, b = statistics(original), statistics(changed)
    assert a['x'] == pytest.approx(b['x'])
    assert a['offset'] == pytest.approx(b['offset'])


def test_real_improvement_preserved_without_changing_response_fit_or_data():
    a, b = [fit(statistics(observed(gain=g))) for g in (1., 1.03)]
    assert a['inverse_reference_speed']/b['inverse_reference_speed'] == pytest.approx(np.full(len(TAUS), 1.03))
    assert a['losses'] == pytest.approx(b['losses'], abs=1e-15)


def test_half_prediction_uses_one_level_for_opposite_half():
    stats = statistics(observed())
    assert cross_half_loss(stats, TAUS.index(60.)) < 1e-25
    assert cross_half_loss(stats, 0) > .001


def test_original_prediction_api_and_alignment_experiment_are_separate():
    from run_analysis.hr_response import ResponseInterval, fit_response, predict_speed_program
    from run_analysis.hr_response_alignment import context_for_windows
    assert ResponseInterval and fit_response and predict_speed_program and context_for_windows


def test_only_timing_is_adjusted_and_observed_residuals_remain(config):
    from datetime import datetime, timezone
    origin = datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp()
    rows = [row(config, average_hr_bpm=140., window_start_time_utc='2026-01-01T00:05:00+00:00',
        window_end_time_utc='2026-01-01T00:07:00+00:00')]
    corrections = [dict(start=300., end=420., effort_adjustment=.05)]
    cal = fit_comparison_calibration([], config)
    result = aligned_windows(rows, corrections, origin_epoch=origin, config=config, calibration=cal)
    assert result[0]['average_hr_bpm'] > 140.
    assert result[0]['moving_pace_min_mile'] == rows[0]['moving_pace_min_mile']
    assert rows[0]['average_hr_bpm'] == 140.
    # More actual measured HR still means more aligned HR, not a replacement
    # by the same fitted curve or arbitrary return to the run-level average.
    higher = [dict(rows[0], average_hr_bpm=150.)]
    assert aligned_windows(higher, corrections, origin_epoch=origin, config=config, calibration=cal)[0]['average_hr_bpm'] > result[0]['average_hr_bpm']


def test_missing_context_preserves_original_window_and_explains_fallback(config):
    from datetime import datetime, timezone
    origin = datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp()
    rows = [row(config, window_start_time_utc='2026-01-01T00:05:00+00:00', window_end_time_utc='2026-01-01T00:07:00+00:00')]
    result = aligned_windows(rows, [], origin_epoch=origin, config=config, calibration=fit_comparison_calibration([], config))
    assert len(result) == len(rows)
    assert result[0]['average_hr_bpm'] == rows[0]['average_hr_bpm']
    assert not result[0]['hr_timing']['available']


def test_response_difference_conserves_null_and_does_not_force_new_initial_hr():
    rows = observed(restarted=True)
    zero = response_corrections(rows, 1/2.7, 0.)
    assert all(abs(c['effort_adjustment']) < 1e-12 for c in zero)
    actual = response_corrections(rows, 1/2.7, 60.)
    assert actual[0]['effort_adjustment'] > 0  # Startup HR below current effort.
    assert actual[90]['effort_adjustment'] > 0  # Recorded restart HR, not sleeping HR.


def test_connected_reader_retains_runs_and_windows_without_mutating_canonical_inputs(config, monkeypatch):
    from copy import deepcopy
    from datetime import datetime, timedelta, timezone
    from run_analysis.aerobic_hr_timing import build_hr_aligned_comparability
    from run_analysis.pace_comparability import compare_run_windows
    from run_analysis.objective_modeling import _primary_window_weight
    cal = fit_comparison_calibration([], config)
    cal['parameters'] = [1., 0., 0.]
    origin = datetime(2026, 1, 1, tzinfo=timezone.utc)
    blocks = observed()
    rows = []; runs = []
    for identity in range(4):
        windows = [row(config, activity_id=identity,
            window_start_time_utc=(origin+timedelta(seconds=t)).isoformat(),
            window_end_time_utc=(origin+timedelta(seconds=t+120)).isoformat()) for t in range(300, 1500, 120)]
        value = compare_run_windows(windows, [_primary_window_weight(w, config) for w in windows], config, cal)
        runs.append(dict(activity_id=identity, included_in_trend=True, trend_weight=1., **value))
        for w in windows:
            w['_aerobic_response'] = dict(blocks=blocks, origin_epoch=origin.timestamp(), coverage={})
        rows.extend(windows)
    report = dict(runs=runs, calibration=cal)
    before, source = deepcopy(report), deepcopy(rows)
    monkeypatch.setattr('run_analysis.model_windows.load_overlapping_model_windows', lambda *args, **kwargs: (rows, {}))
    changed = build_hr_aligned_comparability(None, report, config)
    assert len(changed['runs']) == len(report['runs'])
    assert changed['hr_response']['excluded_runs'] == 0
    for r in changed['runs']:
        assert r['hr_response']['original_windows'] == r['hr_response']['retained_windows'] == 10
        assert r['observed_pace_min_mile'] == pytest.approx(10.)
        assert r['sampled_hr_bpm'] == pytest.approx(config['target_hr'])
        assert r['comparison_precision'] is not None
    assert report == before and rows == source
