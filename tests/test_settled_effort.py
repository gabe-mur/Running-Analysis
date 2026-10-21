from copy import deepcopy
from datetime import datetime,timedelta,timezone
from types import SimpleNamespace

import pytest

from run_analysis.settled_effort import annotate_settled_context,independent_sections
from tests.test_pace_comparability import config


def trace(*,fast=False,slowdown=False,rising=False,late_hr=False):
    origin=datetime(2026,1,1,tzinfo=timezone.utc);intervals=[]
    for sec in range(0,1200,5):
        speed=3.4 if fast else 2.7
        if slowdown and 420<=sec<450:speed*=.4
        hr=148.
        if rising and 450<=sec<570:hr=140+(sec-450)/6
        if late_hr and sec>=570:hr=175.
        a=SimpleNamespace(timestamp_utc=origin+timedelta(seconds=sec),heart_rate_bpm=hr)
        b=SimpleNamespace(timestamp_utc=origin+timedelta(seconds=sec+5),heart_rate_bpm=hr)
        intervals.append(SimpleNamespace(start=a,end=b,elapsed_s=5.,moving_time_s=5.,stopped_time_s=0.,
                                         distance_m=5*speed,elevation_delta_m=0.))
    rows=[dict(window_start_time_utc=(origin+timedelta(seconds=s)).isoformat(),
               window_end_time_utc=(origin+timedelta(seconds=s+120)).isoformat()) for s in range(300,1021,60)]
    return intervals,rows


def test_steady_fast_and_slow_face_identical_checks(config):
    a,rows=trace();b,fast=trace(fast=True)
    annotate_settled_context(a,rows,config);annotate_settled_context(b,fast,config)
    assert all(r['settled_context']['usable'] for r in rows)
    assert [r['settled_context'] for r in rows]==[r['settled_context'] for r in fast]


def test_prior_slowdown_invalidates_later_fast_low_hr_section(config):
    intervals,rows=trace(slowdown=True)
    annotate_settled_context(intervals,rows,config)
    # Window starts after slowdown ended: its own HR/speed are steady, but
    # preceding effort is not. Existing same-window checks miss this context.
    r=rows[3] # Starts at 8 minutes; slowdown was at 7 to 7.5.
    assert not r['settled_context']['usable']
    assert 'effort_transition' in r['settled_context']['reasons']
    assert rows[-1]['settled_context']['usable'] # Recovers; not a run penalty.


def test_following_hr_spike_above_scoring_ceiling_is_not_hidden(config):
    intervals,rows=trace(late_hr=True)
    annotate_settled_context(intervals,rows,config)
    assert not rows[2]['settled_context']['usable']
    assert 'hr_context_range' in rows[2]['settled_context']['reasons']


def test_hr_still_settling_is_not_accepted_just_for_steady_pace(config):
    intervals,rows=trace(rising=True)
    annotate_settled_context(intervals,rows,config)
    assert any('hr_still_changing' in r['settled_context']['reasons'] for r in rows)


def test_overlap_does_not_satisfy_two_independent_sections(config):
    _,rows=trace()
    assert independent_sections(rows[:2])==1
    assert independent_sections(rows[:3])==2
    assert independent_sections([])==0


def test_uniform_real_gain_preserves_selection_and_source_data(config):
    intervals,rows=trace(slowdown=True);changed=deepcopy(intervals);other=deepcopy(rows)
    for i in changed:i.distance_m*=1.03
    before=deepcopy(intervals)
    annotate_settled_context(intervals,rows,config);annotate_settled_context(changed,other,config)
    assert [r['settled_context']['usable'] for r in rows]==[r['settled_context']['usable'] for r in other]
    assert intervals==before


def test_long_moving_gaps_are_unknown_not_invented_steady_context(config):
    intervals,rows=trace()
    intervals[84].end.timestamp_utc+=timedelta(seconds=40)
    intervals[84].elapsed_s+=40
    annotate_settled_context(intervals,rows,config)
    assert any('unknown_moving_gap' in r['settled_context']['reasons'] for r in rows)
