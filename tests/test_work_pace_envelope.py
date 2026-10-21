from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from run_analysis.signal_structure import SignalSpan
from run_analysis.work_stimulus import WorkSignals
from scripts.work_pace_envelope import demonstrated_envelope, work_spans

AT = datetime(2026, 9, 26, tzinfo=timezone.utc)


def evidence(key='a', speed=3., days=1, windows=((100., 220.), (300., 420.))):
    end = AT-timedelta(days=days)
    return WorkSignals(key, 'observed', end-timedelta(seconds=600), end, end,
        windows, (), (SignalSpan(0., 600., speed, speed),), 'explicit_segments')


def test_never_concatenates_short_reps_to_support_longer_effort():
    result = demonstrated_envelope([evidence()], (60., 120., 180.), AT)
    assert [p['speed_mps'] for p in result['points']] == [3., 3., None]
    assert result['points'][-1]['status'] == 'no_prior_continuous_work_support'
    assert not result['selection_ready']


def test_fast_recovery_and_warmup_cannot_buy_work_performance():
    ordinary = evidence()
    padded = replace(ordinary, speed_mps=(SignalSpan(0., 100., 20., 20.),
        SignalSpan(100., 220., 3., 3.), SignalSpan(220., 300., 20., 20.),
        SignalSpan(300., 420., 3., 3.), SignalSpan(420., 600., 20., 20.)))
    assert demonstrated_envelope([ordinary], (120.,), AT) == demonstrated_envelope([padded], (120.,), AT)


def test_trace_gap_breaks_continuity_even_inside_work_window():
    e = replace(evidence(windows=((100., 420.),)), speed_mps=(
        SignalSpan(100., 200., 3., 3.), SignalSpan(220., 420., 3., 3.)))
    assert demonstrated_envelope([e], (240.,), AT)['points'][0]['speed_mps'] is None


def test_arbitrary_novel_duration_supported_without_matching_template():
    e = evidence(windows=((100., 400.),))
    assert demonstrated_envelope([e], (137.,), AT)['points'][0]['speed_mps'] == 3.


def test_removing_better_evidence_cannot_raise_any_reference_component():
    slow, fast = evidence('slow', 3.), evidence('fast', 4.)
    both = demonstrated_envelope([slow, fast], (30., 99., 120., 180.), AT)
    deleted = demonstrated_envelope([slow], (30., 99., 120., 180.), AT)
    for a, b in zip(both['points'], deleted['points']):
        assert b['speed_mps'] is None or b['speed_mps'] <= a['speed_mps']


def test_new_short_fast_work_cannot_lower_longer_duration_reference():
    long = evidence('long', 3., windows=((0., 600.),))
    short = evidence('short', 4., windows=((100., 220.),))
    old = demonstrated_envelope([long], (60., 300.), AT)
    new = demonstrated_envelope([long, short], (60., 300.), AT)
    assert [p['speed_mps'] for p in old['points']] == [3., 3.]
    assert [p['speed_mps'] for p in new['points']] == [4., 3.]


def test_rolling_expiration_is_not_confused_with_addition_monotonicity():
    old = evidence('old', 4., days=41)
    recent = evidence('recent', 3.)
    before = demonstrated_envelope([old, recent], (60.,), AT, lookback_days=42)
    later = demonstrated_envelope([old, recent], (60.,), AT+timedelta(days=2), lookback_days=42)
    assert before['points'][0]['speed_mps'] == 4.
    assert later['points'][0]['speed_mps'] == 3.


def test_original_loader_default_window_and_explicit_full_history(monkeypatch):
    from scripts import audit_program_evidence_adapter as module
    calls = []
    monkeypatch.setattr(module, 'app_session_types', lambda db: {})
    monkeypatch.setattr(module, 'load_sessions', lambda db, cfg, since, until, **kwargs:
                        calls.append((since, until)))
    module.load_history(None, {}, AT, None)
    start = AT-timedelta(days=2000)
    module.load_history(None, {}, AT, None, since=start)
    assert calls == [(AT-timedelta(days=84), AT), (start, AT)]


def test_no_future_self_old_or_unavailable_evidence_and_order_invariance():
    valid = evidence()
    late = replace(evidence('late', 20.), available_at=AT+timedelta(days=1))
    future = evidence('future', 20., days=-1)
    old = evidence('old', 20., days=90)
    rows = [valid, late, future, old]
    result = demonstrated_envelope(rows, (120.,), AT)
    assert result == demonstrated_envelope([valid], (120.,), AT)
    assert result == demonstrated_envelope(reversed(rows), (120.,), AT)
    assert demonstrated_envelope([valid], (120.,), valid.start)['points'][0]['speed_mps'] is None


def test_clipping_linear_spans_preserves_endpoints():
    e = replace(evidence(windows=((100., 200.),)), speed_mps=(SignalSpan(0., 600., 0., 6.),))
    assert work_spans(e) == (SignalSpan(100., 200., 1., 2.),)
    assert demonstrated_envelope([e], (100.,), AT)['points'][0]['speed_mps'] == pytest.approx(1.5)


@pytest.mark.parametrize('durations', [(), (0.,), (-1.,), (float('nan'),), (60., 60.)])
def test_invalid_durations(durations):
    with pytest.raises(ValueError):
        demonstrated_envelope([], durations, AT)


def test_observed_identity_required_and_missing_reference_not_zero():
    with pytest.raises(ValueError):
        demonstrated_envelope([evidence(), evidence()], (60.,), AT)
    planned = replace(evidence(), source='planned', available_at=AT-timedelta(days=2))
    with pytest.raises(ValueError):
        demonstrated_envelope([planned], (60.,), AT)
    empty = demonstrated_envelope([], (60.,), AT)
    assert empty['points'][0]['speed_mps'] is None
