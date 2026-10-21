from copy import deepcopy
from pathlib import Path

import pytest

from run_analysis.config import load_config
from run_analysis.pace_comparability import fit_comparison_calibration
from scripts.audit_standardization_sampling import (
    weight, select, bounds, support_minutes, estimate, half_check, gain_check,
)
from tests.test_pace_comparability import row


@pytest.fixture
def config():
    return load_config(Path(__file__).parents[1]/'config.example.yaml')


def window(config, minute=20, **changes):
    return row(config, moving_minutes_into_run=minute, elapsed_time_s=120., stopped_time_s=0., **changes)


def test_broader_weights_do_not_select_by_pace_or_calendar_outcome(config):
    a = window(config, moving_pace_min_mile=8.)
    b = window(config, moving_pace_min_mile=13.)
    for mode in ('current', 'no_time', 'stable', 'minutes'):
        assert weight(a, config, mode) == weight(b, config, mode)


def test_no_time_factor_removes_time_only_and_broad_also_removes_hr(config):
    a, b = window(config, minute=10), window(config, minute=40)
    assert weight(a, config, 'current') != weight(b, config, 'current')
    assert weight(a, config, 'no_time') == weight(b, config, 'no_time')
    high = window(config, average_hr_bpm=config['target_hr']+10)
    assert weight(high, config, 'no_time') < weight(a, config, 'no_time')
    assert weight(high, config, 'stable') == weight(a, config, 'stable')


def test_stability_is_retained_separately_from_minute_sensitivity(config):
    a = window(config)
    b = window(config, heart_rate_change_bpm=15.)
    assert weight(a, config, 'stable') > weight(b, config, 'stable')
    assert weight(a, config, 'minutes') == weight(b, config, 'minutes')


def test_overlap_is_not_extra_support_and_input_order_does_not_matter(config):
    rows = [window(config, minute=m) for m in (6, 7, 8)]
    assert bounds(rows[0]) == (5., 7.)
    assert support_minutes(rows) == 4.
    assert support_minutes(rows[::-1]+rows) == 4.


def test_pause_exclusion_checks_whole_window_not_center(config):
    rows = [window(config, minute=m) for m in (8, 9, 10, 14, 15, 16, 19)]
    assert [r['moving_minutes_into_run'] for r in select(rows, [10.], 'pause_5')] == [8, 9, 16, 19]
    assert [r['moving_minutes_into_run'] for r in select(rows, [10.], 'pause_8')] == [8, 9, 19]
    assert [r['moving_minutes_into_run'] for r in select(rows, [10.], 'before_pause')] == [8, 9]
    assert select(rows, [], 'before_pause') == rows
    assert select(rows, [], 'pause_5') == rows


def test_full_warmup_is_explicit_and_missing_is_not_zero(config):
    rows = [window(config, minute=5), window(config, minute=6)]
    assert select(rows, [], 'existing') == rows
    assert select(rows, [], 'full_warmup') == rows[1:]
    assert estimate([], config, fit_comparison_calibration([], config), 'current') is None


def test_heldout_halves_require_real_nonoverlap_support(config):
    calibration = fit_comparison_calibration([], config)
    short = [window(config, minute=m) for m in (6, 7, 8)]
    assert half_check(short, 22, config, calibration, 'stable') is None
    rows = [window(config, minute=m) for m in range(6, 30)]
    assert half_check(rows, 30, config, calibration, 'stable')['mean_absolute_error_seconds'] == pytest.approx(0)


def test_uniform_changes_and_input_immutability(config):
    rows = [window(config, minute=m) for m in range(6, 30)]
    before = deepcopy(rows)
    calibration = fit_comparison_calibration([], config)
    for mode in ('current', 'no_time', 'stable', 'minutes'):
        for gain in gain_check(rows, config, calibration, mode):
            assert gain['recovered_speed_percent'] == pytest.approx(gain['planted_speed_percent'], abs=.001)
    assert rows == before


def test_localized_late_improvement_is_not_removed_as_bad_outlier(config):
    calibration = fit_comparison_calibration([], config)
    ordinary = [window(config, minute=m) for m in range(6, 40)]
    faster = [dict(r, moving_pace_min_mile=9.5 if r['moving_minutes_into_run'] > 23 else 10.) for r in ordinary]
    # Both halves are long enough that a genuine sustained faster phase is not
    # simply clipped to the unchanged half by the robust aggregator.
    baseline = estimate(ordinary, config, calibration, 'stable')['pace_min_mile']
    changed = estimate(faster, config, calibration, 'stable')['pace_min_mile']
    assert changed < baseline-.1


def test_before_pause_can_remove_all_support_without_fabricating_score(config):
    rows = [window(config, minute=m) for m in (14, 15, 16)]
    assert select(rows, [8.], 'before_pause') == []


def test_heldout_reference_is_not_changed_to_favor_exclusions(config):
    calibration = fit_comparison_calibration([], config)
    reference = [window(config, minute=m, moving_pace_min_mile=10. if m < 20 else 11.)
                 for m in range(6, 40)]
    selected = [r for r in reference if r['moving_minutes_into_run'] < 20
                or r['moving_minutes_into_run'] > 30]
    result = half_check(selected, 40, config, calibration, 'stable', target_rows=reference)
    midpoint = (5+40)/2
    early = [r for r in selected if bounds(r)[1] <= midpoint-1]
    late_target = [r for r in reference if bounds(r)[0] >= midpoint+1]
    expected = 60*(estimate(early, config, calibration, 'stable')['pace_min_mile']
                   -estimate(late_target, config, calibration, 'minutes')['pace_min_mile'])
    assert result['early_to_late_error_seconds'] == pytest.approx(expected)
