from dataclasses import replace
from datetime import datetime, timedelta, timezone
from math import exp

import numpy as np
import pytest

from run_analysis.hr_response import (
    ResponseInterval, ResponseEvidence, sufficient_statistics, fit_response, predict_response, predict_speed_program,
)
from scripts.audit_hr_response import metrics, summarize
from run_analysis.signal_structure import SignalSpan


NOW = datetime(2026, 9, 25, tzinfo=timezone.utc)
GRID = (10., 30., 60., 120.)


def synthetic(tau=60., step=5., intercept=.15, slope=.15):
    result, hr, time = [], .3, 0.
    for speed in (2., 4., 1., 3., 2.5):
        for _ in range(round(180/step)):
            last = intercept+slope*speed+(hr-intercept-slope*speed)*exp(-step/tau)
            result.append(ResponseInterval(time, time+step, speed, hr, last))
            hr, time = last, time+step
    return tuple(result)


def evidence(rows=None, key='past', **changes):
    rows = synthetic() if rows is None else rows
    return replace(ResponseEvidence(key, NOW-timedelta(days=1), NOW-timedelta(days=1), 'units',
        sufficient_statistics(rows, GRID), (min(r.speed for r in rows), max(r.speed for r in rows)), taus=GRID), **changes)


def fit(history):
    return fit_response(history, as_of=NOW, accounting_key='units', taus=GRID)


def test_recovers_identifiable_model_without_phase_or_workout_labels():
    result = fit([evidence()])
    model = result['dynamic']
    assert model['tau_seconds'] == 60
    assert model['intercept'] == pytest.approx(.15, abs=1e-8)
    assert model['slope'] == pytest.approx(.15, abs=1e-8)
    assert model['training_rmse_reserve'] < 1e-7
    assert model['training_rmse_reserve'] < result['static']['training_rmse_reserve']
    assert not model['tau_at_search_boundary']


def test_future_old_late_uploaded_and_other_unit_data_cannot_affect_fit():
    past = evidence()
    rejected = [evidence(key='future', end=NOW+timedelta(days=1), available_at=NOW+timedelta(days=1)),
        evidence(key='late', available_at=NOW+timedelta(days=1)),
        evidence(key='old_upload', end=NOW-timedelta(days=90)),
        evidence(key='other', accounting_key='other'), evidence(key='ill', eligible=False)]
    assert fit([past]) == fit([past, *rejected])


def test_no_data_not_zero_and_duplicate_sessions_rejected():
    assert fit([])['dynamic'] is None
    with pytest.raises(ValueError, match='Duplicate'):
        fit([evidence(), evidence()])


def test_sampling_density_does_not_change_identical_continuous_model():
    a = fit([evidence(synthetic(step=1))])['dynamic']
    b = fit([evidence(synthetic(step=10))])['dynamic']
    assert a['tau_seconds'] == b['tau_seconds']
    assert a['slope'] == pytest.approx(b['slope'], abs=1e-8)
    assert a['intercept'] == pytest.approx(b['intercept'], abs=1e-8)


def test_one_session_one_vote_regardless_of_recording_repetitions():
    rows = synthetic()
    repeated = rows+tuple(replace(r, start=r.start+1000, end=r.end+1000) for r in rows)
    assert sufficient_statistics(repeated, GRID) == pytest.approx(sufficient_statistics(rows, GRID))


def test_predictions_do_not_read_later_heart_rates():
    rows = synthetic()
    model = fit([evidence()])['dynamic']
    changed = tuple(replace(r, first=r.first if i == 0 else .1, last=.1) for i, r in enumerate(rows))
    assert predict_response(rows, model) == predict_response(changed, model)


def test_missing_gap_resets_only_next_segment_and_does_not_fill_unknown_time():
    rows = synthetic()
    model = fit([evidence()])['dynamic']
    trace, detail = predict_response((rows[0], rows[10]), model)
    assert detail['segment_initializations'] == 2
    assert trace[1].start == rows[10].start
    assert trace[1].first == rows[10].first
    assert sum(s.end-s.start for s in trace) == 10


def test_unsupported_physiology_is_flagged_not_clipped():
    model = fit([evidence()])['dynamic']
    rows = (ResponseInterval(0, 600, 15, .5, .9),)
    trace, detail = predict_response(rows, model)
    assert detail['speed_extrapolation_seconds'] == 600
    assert detail['hr_outside_reference']
    assert trace[-1].last > 1
    result = metrics(trace, (SignalSpan(0, 600, .5, .9),), 150)
    assert result['predicted_load'] is None


def test_degenerate_static_relation_is_not_reported_as_identified():
    rows = tuple(ResponseInterval(i, i+1, 2, .5, .5) for i in range(10))
    assert fit([evidence(rows)])['static'] is None


def test_stats_are_copied_and_immutable():
    stats = sufficient_statistics(synthetic(), GRID)
    e = replace(evidence(), statistics=stats)
    stats[:] = 0
    assert np.any(e.statistics != 0)
    with pytest.raises(ValueError):
        e.statistics[:] = 0


@pytest.mark.parametrize('change', [{'end': -1}, {'speed': -1}, {'last': 2}, {'first': float('nan')}])
def test_invalid_intervals_rejected(change):
    with pytest.raises(ValueError):
        replace(synthetic()[0], **change)


def test_overlapping_time_and_incompatible_grid_rejected():
    with pytest.raises(ValueError, match='nonoverlapping'):
        sufficient_statistics((synthetic()[0], synthetic()[0]), GRID)
    with pytest.raises(ValueError, match='grids'):
        fit_response([evidence()], as_of=NOW, accounting_key='units')


def test_metrics_and_paired_summary_accounting():
    actual = (SignalSpan(0, 30, .5, .5), SignalSpan(60, 90, .5, .5))
    prediction = (SignalSpan(0, 30, .6, .6), SignalSpan(60, 90, .6, .6))
    result = metrics(prediction, actual, 150)
    assert result['seconds'] == 60
    assert result['hr_bias_bpm'] == pytest.approx(15)
    assert result['hr_rmse_bpm'] == pytest.approx(15)
    assert result['predicted_load'] == pytest.approx(1)
    summary = summarize([dict(static=result, dynamic=result), dict(static=result)])
    assert summary['static']['count'] == 1
    assert summary['static']['hr_improved'] == 0


def test_raw_adapter_preserves_missing_hr_and_does_not_bridge_gaps(tmp_path):
    from run_analysis.hr_response import intervals_from_activity
    from run_analysis.tcx import parse_tcx
    from test_tcx import make_tcx
    from test_workload_profile import CONFIG
    activity = parse_tcx(make_tcx(tmp_path/'valid.tcx')).activities[0]
    rows = intervals_from_activity(activity, CONFIG)
    assert len(rows) == 1
    assert rows[0].speed == pytest.approx(2)
    activity.trackpoints[-1].heart_rate_bpm = None
    assert intervals_from_activity(activity, CONFIG) == ()
    activity = parse_tcx(make_tcx(tmp_path/'gap.tcx', end='2024-07-01T12:20:00Z')).activities[0]
    assert intervals_from_activity(activity, CONFIG) == ()


def test_evidence_cannot_predate_completion():
    with pytest.raises(ValueError, match='completion'):
        evidence(end=NOW, available_at=NOW-timedelta(seconds=1))


def test_prescribed_program_carries_hr_between_phases_without_actual_inputs():
    model = fit([evidence()])['dynamic']
    trace, details = predict_speed_program(((180, 4), (180, 1)), model, initial_reserve=.3)
    assert details['segment_initializations'] == 1
    assert trace[-1].end == 360
    assert trace[360].first == trace[359].last
    assert trace[360].first > .7
    assert trace[-1].last < .35
    coarse, _ = predict_speed_program(((180, 4), (180, 1)), model, initial_reserve=.3, step_seconds=2)
    assert trace[-1].last == pytest.approx(coarse[-1].last)


@pytest.mark.parametrize('program', [(), ((-1, 2),), ((20, -1),)])
def test_invalid_programs_rejected(program):
    with pytest.raises(ValueError):
        predict_speed_program(program, fit([evidence()])['dynamic'], initial_reserve=.3)


def test_cli_serializes_timestamped_cases_in_both_artifact_and_console(tmp_path, monkeypatch, capsys):
    import json
    import sys
    from scripts import audit_hr_response
    path = tmp_path/'result.json'
    monkeypatch.setattr(sys, 'argv', ['audit', '--database', 'unused', '--output', str(path)])
    monkeypatch.setattr(audit_hr_response, 'load_config', lambda path: {})
    monkeypatch.setattr(audit_hr_response, 'audit', lambda database, config: {'cases': [{'issued_at': NOW}]})
    audit_hr_response.main()
    assert json.loads(path.read_text()) == json.loads(capsys.readouterr().out)
