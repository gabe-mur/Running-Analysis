from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from run_analysis.continuous_training_dose import ContinuousDoseReference, continuous_dose
from run_analysis.quality_work_exposure import work_cardio_exposure
from run_analysis.signal_structure import SignalSpan
from run_analysis.work_stimulus import WorkSignals

REF = ContinuousDoseReference('unchanged test accounting, not fitted physiology')
AT = datetime(2026, 10, 8, tzinfo=timezone.utc)


def evidence(hrr=.85, windows=((600., 1200.),)):
    return WorkSignals('test', 'observed', AT, AT+timedelta(seconds=2400),
        AT+timedelta(seconds=2400), windows, (SignalSpan(0, 2400, hrr, hrr),),
        (SignalSpan(0, 2400, 3., 3.),), 'explicit_segments')


def test_same_minutes_different_intensity_not_same_exposure():
    moderate, harder = [work_cardio_exposure(evidence(v), reference=REF) for v in (.7, .85)]
    assert moderate['work_seconds'] == harder['work_seconds'] == 600
    assert moderate['high_effort_minutes'].high < harder['high_effort_minutes'].low
    assert harder['high_effort_minutes'].low == pytest.approx(10)
    assert not harder['physiological_equivalence_established']


def test_recovery_padding_cannot_buy_quality_or_become_unknown_work():
    e = evidence(.7)
    raised = replace(e, hrr=(SignalSpan(0, 600, .95, .95),
        SignalSpan(600, 1200, .7, .7), SignalSpan(1200, 2400, .95, .95)))
    a, b = [work_cardio_exposure(v, reference=REF) for v in (e, raised)]
    assert a == b
    assert a['unknown_hr_seconds'] == 0
    assert continuous_dose(raised.hrr, 2400, REF)['cardio'].low > continuous_dose(e.hrr, 2400, REF)['cardio'].low


def test_unknown_work_hr_is_bounded_not_zero_and_speed_survives():
    e = replace(evidence(), hrr=(SignalSpan(600, 900, .85, .85),))
    r = work_cardio_exposure(e, reference=REF)
    assert r['unknown_hr_seconds'] == 300
    assert r['high_effort_minutes'].low == pytest.approx(5)
    assert r['high_effort_minutes'].high > r['high_effort_minutes'].low
    assert r['bouts'][0]['known_speed_seconds'] == 600
    assert r['bouts'][0]['mean_speed_mps'] == 3


def test_short_fast_work_not_discarded_or_fabricated_as_high_hr():
    e = evidence(.6, ((600, 645), (735, 780)))
    r = work_cardio_exposure(e, reference=REF)
    assert r['work_seconds'] == 90
    assert r['high_effort_minutes'].high == 0
    assert len(r['bouts']) == 2 and all(b['mean_speed_mps'] == 3 for b in r['bouts'])


def test_repetition_gaps_and_missing_hr_break_continuity():
    e = evidence(windows=((600, 900), (990, 1290)))
    r = work_cardio_exposure(e, reference=REF)
    assert r['sustained_hr']['longest_continuous_seconds'] == 300
    assert r['sustained_hr']['above_seconds'] == 600
    missing = replace(evidence(), hrr=(SignalSpan(600, 800, .85, .85), SignalSpan(1000, 1200, .85, .85)))
    assert work_cardio_exposure(missing, reference=REF)['sustained_hr']['longest_continuous_seconds'] == 200


def test_linear_trace_and_adjacent_window_splitting_preserve_accounting():
    e = replace(evidence(), hrr=(SignalSpan(600, 1200, .6, .95),))
    split = replace(e, windows=((600, 900), (900, 1200)),
        hrr=(SignalSpan(600, 900, .6, .775), SignalSpan(900, 1200, .775, .95)))
    a, b = [work_cardio_exposure(v, reference=REF) for v in (e, split)]
    for field in ('cardio', 'high_effort_minutes'):
        assert a[field].low == pytest.approx(b[field].low)
        assert a[field].high == pytest.approx(b[field].high)
    assert a['sustained_hr']['longest_continuous_seconds'] == pytest.approx(b['sustained_hr']['longest_continuous_seconds'])


def test_changing_reference_is_disclosed_not_hidden_family_weight():
    e = evidence()
    r = work_cardio_exposure(e, reference=replace(REF, quality_hrr=.9, source='declared sensitivity only'))
    assert r['reference']['quality_hrr'] == .9
    assert r['high_effort_minutes'].high < 10
    assert not r['planner_influence']
