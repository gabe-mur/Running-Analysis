import pytest

from run_analysis.aerobic_entry import describe_entry, moving_hr_spans
from run_analysis.signal_structure import SignalSpan as S


def describe(spans, elapsed=900, **kw):
    args = dict(elapsed_seconds=elapsed, low_bpm=141, high_bpm_exclusive=154,
                confirmation_seconds=300, required_fraction=.8)
    args.update(kw)
    return describe_entry(spans, **args)


def test_ramp_crossing_and_duration_accounted_without_warmup_double_subtraction():
    r = describe([S(0, 120, 121, 141), S(120, 900, 141, 150)])
    assert r['first_entry_seconds'] == r['sustained_entry_seconds'] == 120
    assert r['zone_seconds'] == r['zone_seconds_from_sustained_entry'] == 780
    assert r['confirmation_available_seconds'] == 420


def test_short_spike_is_not_sustained_entry():
    r = describe([S(0, 10, 145, 145), S(10, 180, 130, 130), S(180, 900, 145, 145)])
    assert r['first_entry_seconds'] == 0 and r['sustained_entry_seconds'] == 180


def test_brief_known_dip_does_not_restart_confirmation():
    r = describe([S(0, 100, 145, 145), S(100, 120, 139, 139), S(120, 900, 145, 145)])
    assert r['sustained_entry_seconds'] == 0 and r['zone_seconds'] == 880


def test_unknown_or_stopped_gap_cannot_be_bridged():
    r = describe([S(0, 100, 145, 145), S(101, 900, 145, 145)])
    assert r['sustained_entry_seconds'] == 101
    assert r['known_moving_hr_seconds'] == 899


def test_above_zone_does_not_count_as_easy():
    r = describe([S(0, 900, 160, 160)])
    assert r['zone_seconds'] == 0 and r['sustained_entry_seconds'] is None


def test_missing_or_short_data_does_not_mean_zero_benefit():
    for spans, elapsed in [([], 900), ([S(0, 120, 145, 145)], 120)]:
        r = describe(spans, elapsed)
        assert r['sustained_entry_seconds'] is None
        assert not r['physiological_warmup_established']


def test_recording_already_in_band_is_visible():
    r = describe([S(50, 900, 145, 145)])
    assert r['sustained_entry_seconds'] == 50
    assert r['already_at_or_above_band_on_first_observation']
    assert r['first_observed_hr_offset_seconds'] == 50


def test_high_edge_matches_continuous_zone_accounting():
    assert describe([S(0, 900, 154, 154)])['zone_seconds'] == 0
    assert describe([S(0, 900, 153, 153)])['zone_seconds'] == 900


def test_interpolated_band_integral_is_sample_rate_independent():
    a = describe([S(0, 900, 120, 180)])
    b = describe([S(i, i+30, 120+i/15, 120+(i+30)/15) for i in range(0, 900, 30)])
    assert a == b
    assert a['zone_seconds'] == pytest.approx(195)


@pytest.mark.parametrize('spans', [[S(0, 600, 140, 150), S(500, 900, 140, 150)], [S(-1, 900, 140, 150)]])
def test_bad_chronology_rejected(spans):
    with pytest.raises(ValueError):
        describe(spans)


def test_adapter_keeps_stops_gaps_and_missing_hr_out_of_continuity():
    from datetime import datetime, timedelta, timezone
    from types import SimpleNamespace as NS
    origin = datetime(2026, 9, 25, tzinfo=timezone.utc)
    def interval(start, *, moving=10, flags=(), hr=145):
        return NS(start=NS(timestamp_utc=origin+timedelta(seconds=start), heart_rate_bpm=hr),
            end=NS(timestamp_utc=origin+timedelta(seconds=start+10), heart_rate_bpm=145),
            moving_time_s=moving, elapsed_s=10, flags=flags)
    spans = moving_hr_spans(NS(start_time_utc=origin), [interval(0), interval(10, moving=0),
        interval(20, flags=['large_recording_gap']), interval(30, hr=None), interval(40)])
    assert [(s.start, s.end) for s in spans] == [(0, 10), (40, 50)]
