from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from run_analysis.config import load_config
from scripts.audit_pace_context import chronological_cases, summarize, MODELS


def rows():
    at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return [dict(activity_id=day, start_time_utc=(at+timedelta(days=day)).isoformat(),
        activity_end_time_utc=(at+timedelta(days=day, hours=1)).isoformat(),
        moving_pace_min_mile=10+.01*day+.02*window, average_hr_bpm=145+window,
        moving_minutes_into_run=10+window*10, grade_energy_ratio=1.01,
        average_grade_percent=.2, temperature_f=65+day, dewpoint_f=50,
        relative_humidity_percent=60.) for day in range(10) for window in range(2)]


def test_only_completed_prior_runs_train_and_future_response_cannot_change_prior_predictions():
    cfg = load_config(Path('config.yaml'))
    data = rows()
    original, _ = chronological_cases(data, cfg)
    altered, _ = chronological_cases([dict(r, moving_pace_min_mile=25.) if r['activity_id']==9 else r
                                     for r in data], cfg)
    for a, b in zip(original, altered):
        assert all(k < a['key'] for k in a['prior_keys'])
        for name in MODELS:
            assert a['results'][name]['mean_predicted_pace_min_mile'] == b['results'][name]['mean_predicted_pace_min_mile']


def test_many_duplicate_windows_do_not_give_one_run_extra_training_weight():
    cfg = load_config(Path('config.yaml'))
    data = rows()
    original, _ = chronological_cases(data, cfg)
    duplicated, _ = chronological_cases(data+[r for r in data if r['activity_id']==0]*20, cfg)
    for a, b in zip(original, duplicated):
        for name in MODELS:
            assert a['results'][name]['mean_predicted_pace_min_mile'] == pytest.approx(
                b['results'][name]['mean_predicted_pace_min_mile'], abs=1e-9)


def test_unfinished_previous_run_not_used_and_minimum_is_reported():
    cfg = load_config(Path('config.yaml'))
    data = rows()
    data = [dict(r, activity_end_time_utc='2027-01-01T00:00:00+00:00') if r['activity_id']==0 else r for r in data]
    cases, skipped = chronological_cases(data, cfg)
    assert 6 in skipped and all(0 not in c['prior_keys'] for c in cases)


def test_context_gain_must_survive_both_chronological_halves():
    cases = [dict(results={name: dict(mae_seconds_per_mile=(10 if name=='B' else error)) for name in MODELS})
             for error in (11, 11, 2, 2)]
    result = summarize(cases, 2)
    assert not result['models']['D']['consistent_incremental_gain']
