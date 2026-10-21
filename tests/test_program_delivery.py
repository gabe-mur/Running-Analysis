from dataclasses import replace
from datetime import date, timedelta

import pytest

from run_analysis.program_delivery import ProgramExposure, TargetSegment, audit_program, compare_allocation, screen_development
from scripts.audit_program_delivery import ORIGIN, ledger, regular, panel


def audit(rows=None, target=17.3, **changes):
    kwargs = dict(coverage_start=ORIGIN-timedelta(days=56), coverage_end=ORIGIN+timedelta(days=112),
        evaluation_starts=[ORIGIN], spans=(28,), coverage_complete=True)
    kwargs.update(changes)
    return audit_program(ledger(regular()) if rows is None else rows,
        [TargetSegment(kwargs['coverage_start'], kwargs['coverage_end'], target)], **kwargs)


@pytest.fixture(scope='module')
def comparisons():
    return panel()


def test_sparse_perfectly_executable_program_is_underfunded(comparisons):
    r = comparisons['summaries']['sparse']
    assert r['own_target_pattern'] == 'all_checked_intervals_below_target'
    assert r['opening28']['miles'] == pytest.approx(45.6)
    assert r['opening28']['target_miles'] == pytest.approx(69.2)


def test_holding_and_growth_are_not_compared_to_secret_different_targets(comparisons):
    r = comparisons['summaries']['hold']
    assert r['own_target_pattern'] == 'all_checked_intervals_at_or_above_target'
    assert r['common_growth_target_pattern'] == 'all_checked_intervals_below_target'


def test_equal_funded_mileage_does_not_hide_compression(comparisons):
    a = comparisons['summaries']['supported_long']
    b = comparisons['summaries']['compressed']
    assert a['opening28']['miles'] == pytest.approx(b['opening28']['miles'])
    assert a['opening28_max_gap'] == 3
    assert b['opening28_max_gap'] == 5
    assert a['opening28']['maximum_running_streak'] == 1
    assert b['opening28']['maximum_running_streak'] == 3
    assert not b['opening28_vs_hold']['quality_structure_or_purpose_changed']
    assert b['opening28_vs_hold']['quality_dates_changed']
    assert comparisons['reports']['compressed']['recovery_feasibility'] == 'not_assessed'


def test_frontloading_cannot_hide_behind_a_28_day_total(comparisons):
    r = comparisons['summaries']['frontloaded']
    assert r['opening28']['miles'] == pytest.approx(72.8)
    assert r['opening28_max_gap'] == 17
    assert r['opening28']['maximum_running_streak'] == 12
    assert r['own_target_pattern'] == 'mixed_or_boundary_sensitive'


def test_short_support_addition_keeps_ordinary_runs_and_quality_unchanged(comparisons):
    r = comparisons['summaries']['insert_support']['opening28_vs_hold']
    assert r['miles_change'] == pytest.approx(3.6)
    assert r['nonlong_miles_change'] == pytest.approx(3.6)
    assert r['running_days_change'] == 4
    assert r['quality_work_seconds_change'] == 0
    assert not r['quality_structure_or_purpose_changed']


def test_long_growth_cannot_hide_lost_support(comparisons):
    r = comparisons['summaries']['long_with_less_support']['opening28_vs_hold']
    assert r['miles_change'] > 0
    assert r['nonlong_miles_change'] == pytest.approx(-.8)
    assert comparisons['summaries']['long_with_less_support']['own_target_pattern'] == 'all_checked_intervals_below_target'


def test_quality_funded_growth_is_not_mistaken_for_unchanged_quality(comparisons):
    r = comparisons['summaries']['quality_funded']['opening28_vs_hold']
    assert r['quality_work_seconds_change'] == 1200
    assert r['quality_structure_or_purpose_changed']
    assert r['easy_miles_change_range'] == pytest.approx((0., 0.))


def test_cutoff_control_is_mixed_not_global_underfunding(comparisons):
    r = comparisons['summaries']['period_eight']
    assert r['own_target_pattern'] == 'mixed_or_boundary_sensitive'
    low, high = r['target_ratio_range']
    assert low < 1 < high


def test_missing_easy_or_role_attribution_is_not_auto_easy_support():
    item = ProgramExposure('x', ORIGIN, 8., 'unknown', 'observed')
    r = audit([item])['intervals'][0]
    assert r['easy_miles_lower'] == 0 and r['easy_miles_upper'] == 8
    assert r['nonlong_miles'] is None
    assert (r['nonlong_miles_lower'], r['nonlong_miles_upper']) == (0, 8)


def test_missing_quality_work_is_unknown_not_zero():
    item = ProgramExposure('x', ORIGIN, 4., 'quality', 'observed')
    r = audit([item])['intervals'][0]
    assert r['known_quality_work_seconds'] == 0
    assert r['total_quality_work_seconds'] is None


def test_same_day_file_splitting_cannot_buy_frequency_or_lower_concentration():
    items = ledger(regular())
    whole = next(e for e in items if e.day == ORIGIN)
    split = [e for e in items if e != whole]+[
        replace(whole, key='part1', miles=4.2, easy_miles=4.2),
        replace(whole, key='part2', miles=4.2, easy_miles=4.2)]
    a, b = audit(items)['intervals'][0], audit(split)['intervals'][0]
    for field in ('miles', 'running_days', 'largest_day_fraction', 'gaps', 'running_streaks', 'role_running_days'):
        assert a[field] == b[field]


def test_no_history_or_tail_means_censored_edges_not_verified_rest():
    r = audit([ProgramExposure('one', ORIGIN+timedelta(days=10), 4., 'easy', 'observed', 4.)])['intervals'][0]
    assert r['gaps']['all']['left_censored'] and r['gaps']['all']['right_censored']
    assert r['gaps']['all']['maximum_bracketed_days'] is None


def test_gaps_use_both_history_and_tail():
    items = [ProgramExposure(str(d), ORIGIN+timedelta(days=d), 4., 'easy', 'observed', 4.)
             for d in (-3, 2, 10, 26, 30)]
    r = audit(items)['intervals'][0]
    gaps = r['gaps']['all']['bracketed']
    assert gaps[0]['days'] == 5 and gaps[-1]['days'] == 4
    assert not r['gaps']['all']['left_censored'] and not r['gaps']['all']['right_censored']


def test_incomplete_upload_coverage_cannot_establish_underfunding():
    r = audit([], coverage_complete=False)
    assert r['numerical_funding_pattern'] == 'all_checked_intervals_below_target'
    assert r['funding_evidence_status'] == 'unverified_ledger_coverage'


def test_expected_completion_changes_source_not_accounting():
    rows = ledger(regular())
    assert audit(rows) == audit([replace(e, source='observed') for e in rows])


def test_horizon_extension_preserves_shared_fully_bracketed_intervals():
    rows = ledger(regular())
    end = ORIGIN+timedelta(days=40)
    short = audit([e for e in rows if e.day < end], coverage_end=end)
    assert short == audit(rows)


def test_target_integration_has_no_week_reset():
    start, end = ORIGIN, ORIGIN+timedelta(days=28)
    split = ORIGIN+timedelta(days=11)
    report = audit_program(ledger(regular()), [TargetSegment(start, split, 17.3), TargetSegment(split, end, 18.2)],
        coverage_start=start-timedelta(days=56), coverage_end=start+timedelta(days=112),
        evaluation_starts=[start], spans=(28,), coverage_complete=True)
    assert report['intervals'][0]['target_miles'] == pytest.approx(11*17.3/7+17*18.2/7)


def test_quality_purpose_change_is_visible_even_at_equal_work_total():
    rows = ledger(regular())
    different = [replace(e, quality_purpose='speed_skill') if e.role == 'quality' else e for e in rows]
    r = compare_allocation(audit(rows), audit(different))[0]
    assert r['quality_work_seconds_change'] == 0
    assert r['quality_structure_or_purpose_changed']


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError, match='one decimal'):
        ProgramExposure('x', ORIGIN, 4.444, 'easy', 'planned')
    with pytest.raises(ValueError, match='Duplicate'):
        x = ProgramExposure('x', ORIGIN, 4., 'easy', 'planned')
        audit([x, x])
    with pytest.raises(ValueError, match='cover requested'):
        audit([], coverage_end=ORIGIN+timedelta(days=20))
    with pytest.raises(ValueError, match='Target coverage'):
        audit_program([], [], coverage_start=ORIGIN, coverage_end=ORIGIN+timedelta(days=28), evaluation_starts=[ORIGIN], spans=(28,))


def test_all_candidates_share_history_and_none_authorize_growth(comparisons):
    a = [e for e in ledger(regular()) if e.day < ORIGIN]
    b = [e for e in ledger(regular(long=8.8, easy=4.9)) if e.day < ORIGIN]
    assert a == b
    assert all(not r['progression_accepted'] for r in comparisons['reports'].values())


def test_development_screen_cannot_promote_sparse_or_quality_funded_growth(comparisons):
    sparse = comparisons['summaries']['sparse']['development_screen']
    assert 'underfunded_across_checked_intervals' in sparse['review_reasons']
    quality = comparisons['summaries']['quality_funded']['development_screen']
    assert 'quality_dose_or_purpose_changed' in quality['review_reasons']
    long = comparisons['summaries']['long_with_less_support']['development_screen']
    assert 'supporting_distance_reduced_across_checked_intervals' in long['review_reasons']


def test_funded_alternatives_still_require_recovery_and_distribution(comparisons):
    for name in ('hold', 'extend_easy', 'insert_support', 'supported_long', 'compressed'):
        result = comparisons['summaries'][name]['development_screen']
        assert result['next_check'] == 'dose_specific_recovery_and_distribution'
        assert not result['progression_accepted']
    # Compression is not a numerical volume failure or an automatic safety pass.
    assert comparisons['summaries']['compressed']['opening28']['maximum_running_streak'] == 3


def test_mixed_cutoff_result_requires_review_not_global_failure(comparisons):
    result = comparisons['summaries']['period_eight']['development_screen']
    assert 'cutoff_or_delivery_pattern_requires_review' in result['review_reasons']
    assert 'underfunded_across_checked_intervals' not in result['review_reasons']
