from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from run_analysis.signal_structure import SignalSpan, above_threshold, duration_curve, speed_duration_curve, describe_signals
from run_analysis.workload_profile import observed_profile
from run_analysis.tcx import parse_tcx
from test_tcx import make_tcx
from test_workload_profile import CONFIG


def test_equal_exposure_does_not_mean_equal_continuity():
    sustained = [SignalSpan(0, 300, 170, 170), SignalSpan(300, 600, 140, 140)]
    intermittent = [SignalSpan(i*60, (i+1)*60, 170 if i % 2 == 0 else 140,
                               170 if i % 2 == 0 else 140) for i in range(10)]
    a, b = above_threshold(sustained, 160), above_threshold(intermittent, 160)
    assert a['above_seconds'] == b['above_seconds'] == 300
    assert a['longest_continuous_seconds'] == 300
    assert b['longest_continuous_seconds'] == 60
    assert a['episode_count'] == 1
    assert b['episode_count'] == 5


def test_linear_threshold_crossings_are_exact_and_sampling_independent():
    whole = above_threshold([SignalSpan(0, 60, 140, 180)], 160)
    split = above_threshold([SignalSpan(0, 20, 140, 140+40/3),
                            SignalSpan(20, 60, 140+40/3, 180)], 160)
    assert whole['above_seconds'] == pytest.approx(30)
    assert split['above_seconds'] == pytest.approx(whole['above_seconds'])
    assert split['episode_count'] == whole['episode_count'] == 1
    assert split['longest_continuous_seconds'] == pytest.approx(30)


def test_gap_is_not_recovery_or_sustained_work():
    spans = [SignalSpan(0, 60, 170, 170), SignalSpan(90, 150, 170, 170)]
    result = above_threshold(spans, 160)
    assert result['episode_count'] == 2
    assert result['longest_continuous_seconds'] == 60
    assert result['gaps'] == [dict(elapsed_seconds=30, known_below_threshold_seconds=0,
                                  unknown_seconds=30)]
    missing = above_threshold([], 160)
    assert missing['episode_count'] is None
    assert missing['fraction_of_known'] is None


def test_continuous_signal_needs_no_reps_or_lap_boundaries():
    result = above_threshold([SignalSpan(0, 1200, 165, 165)], 160)
    assert result['above_seconds'] == result['longest_continuous_seconds'] == 1200


def test_peak_speed_curve_is_exact_without_bridging_gaps():
    spans = [SignalSpan(0, 60, 2, 2), SignalSpan(60, 120, 4, 4), SignalSpan(120, 180, 2, 2)]
    curve = speed_duration_curve(spans, (30, 60, 120, 180, 240))
    assert [x['peak_mean_mps'] for x in curve] == pytest.approx([4, 4, 3, 8/3, None])
    gapped = [spans[0], spans[2]]
    assert speed_duration_curve(gapped, (120,))[0]['peak_mean_mps'] is None


def test_speed_curve_is_invariant_to_sample_density():
    coarse = [SignalSpan(0, 30, 2, 2), SignalSpan(30, 150, 4, 4), SignalSpan(150, 180, 2, 2)]
    fine = [SignalSpan(t, t+5, 4 if 30 <= t < 150 else 2, 4 if 30 <= t < 150 else 2)
            for t in range(0, 180, 5)]
    assert speed_duration_curve(coarse, (30, 60, 120, 180)) == speed_duration_curve(fine, (30, 60, 120, 180))


def test_curve_matches_independent_dense_window_integration():
    # Piecewise constant integer-second signal allows an independent sliding sum.
    values = [2, 4, 3, 5, 1, 3, 4, 2]*8
    spans = [SignalSpan(i, i+1, value, value) for i, value in enumerate(values)]
    for window in (3, 7, 13, 30):
        expected = max(sum(values[i:i+window])/window for i in range(len(values)-window+1))
        assert speed_duration_curve(spans, (window,))[0]['peak_mean_mps'] == pytest.approx(expected)


def test_hr_curve_finds_interior_peak_without_any_zone_thresholds():
    spans = [SignalSpan(0, 60, 140, 180), SignalSpan(60, 120, 180, 140)]
    result = duration_curve(spans, (60,))[0]
    assert result['start_seconds'] == pytest.approx(30)
    assert result['peak_mean_value'] == pytest.approx(170)
    fine = [SignalSpan(i, i+1, 180-abs(60-i)*2/3, 180-abs(60-i-1)*2/3)
            for i in range(120)]
    other = duration_curve(fine, (60,))[0]
    assert other['peak_mean_value'] == pytest.approx(result['peak_mean_value'])
    assert other['start_seconds'] == pytest.approx(30)


def test_lap_labels_or_missing_laps_do_not_change_recorded_signals(tmp_path):
    activity = parse_tcx(make_tcx(tmp_path/'run.tcx')).activities[0]
    initial = observed_profile(activity, CONFIG)['intensity_structure']['recorded_signals']
    activity.notes = 'maximal intervals or easy recovery: not an input'
    for lap in activity.laps:
        lap.intensity = 'Active'
        lap.trigger_method = 'Manual'
    assert observed_profile(activity, CONFIG)['intensity_structure']['recorded_signals'] == initial
    activity.laps.clear()
    assert observed_profile(activity, CONFIG)['intensity_structure']['recorded_signals'] == initial


def interval(start, seconds=10, speed=3., hr=160, flags=(), moving=None):
    origin = datetime(2026, 9, 24, tzinfo=timezone.utc)
    return SimpleNamespace(start=SimpleNamespace(timestamp_utc=origin+timedelta(seconds=start), heart_rate_bpm=hr),
        end=SimpleNamespace(timestamp_utc=origin+timedelta(seconds=start+seconds), heart_rate_bpm=hr),
        moving_time_s=seconds if moving is None else moving, elapsed_s=seconds,
        distance_m=seconds*speed, distance_source='device', flags=list(flags))


def test_missing_hr_preserves_speed_and_invalid_speed_preserves_hr():
    rows = [interval(0, hr=None), interval(10, flags=('implausible_distance_speed',))]
    result = describe_signals(rows, hr_thresholds_bpm=[150], windows_seconds=[10])
    assert result['coverage']['hr_seconds'] == 10
    assert result['coverage']['speed_seconds'] == 10
    assert result['speed_duration_curve'][0]['peak_mean_mps'] == 3


def test_pause_and_recording_gap_break_both_signals():
    rows = [interval(0), interval(10, moving=0), interval(20, flags=('large_recording_gap',)), interval(30)]
    result = describe_signals(rows, hr_thresholds_bpm=[150], windows_seconds=[15])
    assert result['hr_exposure'][0]['episode_count'] == 2
    assert result['coverage']['hr_seconds'] == result['coverage']['speed_seconds'] == 20
    assert result['speed_duration_curve'][0]['peak_mean_mps'] is None


def test_unknown_signal_does_not_manufacture_a_best_pace():
    result = describe_signals([], hr_thresholds_bpm=[150], windows_seconds=[30])
    assert result['speed_duration_curve'][0]['peak_mean_mps'] is None
    assert not result['planner_influence']


def test_invalid_or_overlapping_signal_spans_rejected():
    with pytest.raises(ValueError):
        SignalSpan(10, 0, 1, 1)
    with pytest.raises(ValueError):
        above_threshold([SignalSpan(0, 20, 1, 1), SignalSpan(10, 30, 1, 1)], 2)
    with pytest.raises(ValueError):
        speed_duration_curve([SignalSpan(0, 10, 1, 3)], [5])
