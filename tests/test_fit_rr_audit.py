import numpy as np
from datetime import datetime, timedelta, timezone

from scripts.audit_fit_rr import screen_rr, rr_context_windows, inspect


def test_stable_series_is_not_flagged_and_is_not_modified():
    rr = np.full(100, .4)
    before = rr.copy()
    assert not screen_rr(rr).any()
    np.testing.assert_array_equal(rr, before)


def test_isolated_short_and_long_intervals_are_screened():
    rr = np.full(100, .4)
    rr[20], rr[60] = .034, 5.026
    assert np.flatnonzero(screen_rr(rr)).tolist() == [20, 60]


def test_sensitivity_is_explicit_not_a_fixed_artifact_truth():
    rr = np.full(100, .4)
    rr[50] = .49
    assert screen_rr(rr, .2)[50]
    assert not screen_rr(rr, .3)[50]


def test_pause_partial_end_and_unanchored_tail_are_not_clean_exercise_windows():
    start=datetime(2026,9,25,tzinfo=timezone.utc)
    at=lambda seconds:start+timedelta(seconds=seconds)
    batches=[(at(5),[.4]*10),(at(130),[.4]*10),(at(245),[.4]*10),(at(250),[.4,5.1])]
    following=[at(10),at(135),at(250),None]
    events=[dict(timestamp=at(t),event_type=kind) for t,kind in
            [(0,'start'),(180,'stop_all'),(190,'start'),(250,'stop_all')]]
    r=rr_context_windows(batches,following,['running']*3+['stopped'],events,start,250)
    assert len(r['windows'])==3
    assert r['windows'][0]['continuous_context_candidate']
    assert not r['windows'][1]['continuous_context_candidate']
    assert r['windows'][1]['pause_overlap_seconds']==10
    assert not r['windows'][2]['full_wall_time_window']
    assert r['missing_following_record_intervals']==2
    assert r['after_final_stop_source_order_intervals']==2
    assert r['source_order_stopped_rr_seconds']==5.5
    assert not any(w['validated_for_dfa'] for w in r['windows'])


def test_window_crossing_record_brackets_are_not_forced_into_one_window():
    start=datetime(2026,9,25,tzinfo=timezone.utc)
    r=rr_context_windows([(start+timedelta(seconds=118),[.4]*10)],
        [start+timedelta(seconds=123)],['running'],[],start,240)
    assert all(w['source_bracketed_running_intervals']==0 for w in r['windows'])


def test_pause_does_not_truncate_wall_clock_rr_groups():
    from tests.fit_builder import FitWriter,UINT8,UINT16,UINT32,ENUM,fit_timestamp
    start=datetime(2026,9,25,tzinfo=timezone.utc)
    writer=FitWriter()
    writer.message(21,{253:(UINT32,fit_timestamp(start)),0:(ENUM,0),1:(ENUM,0)})
    for offset in (0,250):
        writer.message(20,{253:(UINT32,fit_timestamp(start+timedelta(seconds=offset))),3:(UINT8,144)})
        writer.message(78,{0:(UINT16,400)})
    writer.message(18,{2:(UINT32,fit_timestamp(start)),7:(UINT32,250000),8:(UINT32,10000),
        9:(UINT32,100000),16:(UINT8,144),17:(UINT8,145),193:(UINT8,20),192:(UINT8,75)})
    r=inspect(writer.build())
    assert [w['approx_start_s'] for w in r['rr']['approximate_two_minute_windows']]==[0,240]
    assert len(r['rr_context']['windows'])==3
    assert r['decoded_feedback']['perceived_exertion']==2
    assert r['decoded_feedback']['workout_feel']==75
