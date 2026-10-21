from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from run_analysis.athlete_projection import SessionEvidence, PaceSample
from run_analysis.demonstrated_training import (
    demonstrated_baseline, SessionIntent, UntimedSession, RecordingCoverage,
)
from run_analysis.workload_accounting import Quantity

NOW = datetime(2026, 9, 25, 12, tzinfo=timezone.utc)


def run(key='a', days=1, meters=6000, minutes=40, **kwargs):
    end = NOW-timedelta(days=days)
    defaults = dict(role='easy', cardio=Quantity.point(40, 'test'), accounting_key='a')
    defaults.update(kwargs)
    return SessionEvidence(key, end-timedelta(minutes=minutes), end, end,
        meters=meters, moving_seconds=minutes*60, **defaults)


def report(history, **kwargs):
    return demonstrated_baseline(history, NOW, **kwargs)['windows']['28']


def intent(key, purpose):
    return SessionIntent(key, purpose, NOW-timedelta(days=10), 'fixture_issued_plan')


def test_support_addition_does_not_shrink_ordinary_or_replace_actual_distance():
    original = run(meters=6200)
    intents = [intent('a', 'ordinary'), intent('b', 'support')]
    before = report([original], intents=intents)
    after = report([original, run('b', days=2, meters=1500)], intents=intents)
    assert before['ordinary'] == after['ordinary']
    assert after['ordinary']['meters']['median'] == 6200
    assert after['total_meters'] == 7700
    assert after['support']['meters']['median'] == 1500


def test_short_easy_is_not_automatically_support_or_ordinary():
    result = report([run(meters=1500, ordinary_reference_eligible=True)])
    assert result['easy_intent_unknown_keys'] == ['a']
    assert result['ordinary']['meters']['count'] == 0
    assert result['eligible_easy']['meters']['median'] == 1500


def test_late_intent_and_late_future_upload_do_not_teach_baseline():
    a = run()
    late = replace(run('late', days=2), available_at=NOW+timedelta(days=1))
    future = run('future', days=-1)
    tag = SessionIntent('a', 'ordinary', NOW+timedelta(seconds=1), 'late_report')
    result = report([a, late, future], intents=[tag])
    assert result['all_sessions'] == report([a])['all_sessions']
    assert result['ordinary']['meters']['count'] == 0
    assert result['unavailable_completion_keys'] == ['late']


def test_health_excluded_still_counts_actual_exposure():
    result = report([run(capacity_eligible=False)], intents=[intent('a', 'ordinary')])
    assert result['total_meters'] == 6000
    assert result['reference_eligible']['meters']['count'] == 0
    assert result['ordinary']['meters']['count'] == 0
    assert result['reference_excluded_keys'] == ['a']


def test_equal_volume_does_not_hide_concentration():
    spread = report([run(str(i), days=i+1) for i in range(4)])
    concentrated = report([run(str(i), days=i*7+1, meters=12000) for i in range(2)])
    assert spread['total_meters'] == concentrated['total_meters']
    assert spread['run_count'] == 4 and concentrated['run_count'] == 2
    assert spread['largest_session_distance_fraction'] == .25
    assert concentrated['largest_session_distance_fraction'] == .5
    assert concentrated['start_gaps_days'] == [7]


def test_hr_missing_or_different_units_never_erase_external_dose():
    result = report([run(cardio=None), run('b', days=2, accounting_key='other')])
    assert result['total_meters'] == 12000
    assert result['cardio_by_accounting_signature']['a']['total_upper'] is None
    assert result['cardio_by_accounting_signature']['other']['total_upper'] == 40


def test_empty_uploads_are_not_verified_inactivity():
    assert report([])['history_status'] == 'recorded_only'
    coverage = RecordingCoverage(NOW-timedelta(days=90), NOW, 'explicit_athlete_confirmation')
    assert report([], coverage=coverage)['history_status'] == 'asserted_complete'
    assert report([], coverage=replace(coverage, available_at=NOW+timedelta(days=1)))['history_status'] == 'recorded_only'


def test_unresolved_or_late_records_prevent_complete_claim():
    coverage = RecordingCoverage(NOW-timedelta(days=90), NOW, 'confirmed')
    missing = UntimedSession('summary', NOW-timedelta(days=2), NOW, 'original_summary')
    result = report([], coverage=coverage, untimed=[missing])
    assert result['unresolved_timing_keys'] == ['summary']
    assert result['history_status'] == 'recorded_only'
    late = replace(run(), available_at=NOW+timedelta(days=1))
    assert report([late], coverage=coverage)['history_status'] == 'recorded_only'


def test_boundary_whole_completion_and_stale_context_are_explicit():
    boundary = run(days=28)
    assert report([boundary])['run_count'] == 0
    assert demonstrated_baseline([boundary], NOW)['windows']['84']['run_count'] == 1
    stale = demonstrated_baseline([run(days=100)], NOW)
    assert stale['windows']['84']['run_count'] == 0
    assert stale['days_since_last_completion'] == 100
    assert stale['capacity_established'] is False


def test_two_sessions_same_local_day_count_as_one_running_date():
    a = run()
    b = replace(a, key='b', start=a.start+timedelta(hours=3),
                end=a.end+timedelta(hours=3), available_at=a.end+timedelta(hours=3))
    result = report([a, b], timezone='America/New_York')
    assert result['run_count'] == 2
    assert result['local_running_dates'] == ['2026-09-24']


def test_long_maximum_has_context_not_capacity_and_quality_has_structure():
    work = PaceSample('work', 300, 1100, 'recorded_work_lap')
    result = report([run(role='long', meters=18000), run('b', days=3, role='long', meters=12000),
                     run('c', days=5, role='intervals', pace_samples=(work, work, work))])
    assert result['long']['meters']['median'] == 15000
    assert result['long']['meters']['maximum'] == 18000
    assert result['quality_work'][0]['total_sampled_work_seconds'] == 900
    assert result['quality_work'][0]['longest_sampled_bout_seconds'] == 300


def test_quality_label_without_work_samples_is_unknown_not_zero_work():
    result = report([run(role='intervals')])
    assert result['quality']['meters']['count'] == 1
    assert result['quality_work'][0]['total_sampled_work_seconds'] is None
    assert result['quality_work'][0]['structure_status'] == 'unknown'


def test_known_late_untimed_record_cannot_establish_complete_history():
    coverage = RecordingCoverage(NOW-timedelta(days=90), NOW, 'confirmed')
    missing = UntimedSession('summary', NOW-timedelta(days=2), NOW+timedelta(days=1), 'original_summary')
    result = report([], coverage=coverage, untimed=[missing])
    assert result['history_status'] == 'recorded_only'
    assert result['unavailable_untimed_keys'] == ['summary']


def test_reporting_rollover_does_not_change_retained_actual_doses():
    history = [run('a', days=1), run('b', days=3)]
    before = demonstrated_baseline(history, NOW)
    after = demonstrated_baseline(history, NOW+timedelta(days=1))
    for window in before['windows']:
        assert before['windows'][window]['all_sessions'] == after['windows'][window]['all_sessions']


def test_local_date_is_not_silently_utc_date():
    a = run()
    # UTC Sep 24 01:00 is Sep 23 in New York.
    start = NOW.replace(day=24, hour=1)
    a = replace(a, start=start, end=start+timedelta(minutes=40), available_at=start+timedelta(minutes=40))
    assert report([a], timezone='America/New_York')['local_running_dates'] == ['2026-09-23']


def test_athlete_scaling_keeps_distribution_and_frequency_not_personal_mileage():
    history = [run(str(i), days=i+1, meters=(i+1)*2000) for i in range(4)]
    before = report(history)
    after = report([replace(s, meters=s.meters*2) for s in history])
    assert after['total_meters'] == before['total_meters']*2
    assert after['eligible_easy']['meters']['median'] == before['eligible_easy']['meters']['median']*2
    assert after['run_count'] == before['run_count']
    assert after['largest_session_distance_fraction'] == before['largest_session_distance_fraction']


@pytest.mark.parametrize('windows', [(), (0,), (-1,), (float('nan'),), (7, 7)])
def test_invalid_windows_rejected(windows):
    with pytest.raises(ValueError):
        report([], windows=windows)


def test_duplicate_and_overlap_sources_rejected():
    with pytest.raises(ValueError, match='Duplicate'):
        report([run(), run()])
    with pytest.raises(ValueError, match='Overlapping'):
        report([run(), run('b')])
