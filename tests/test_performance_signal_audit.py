from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from scripts.audit_performance_signals import terrain_speed_curve
from scripts.audit_exercise_hrv import matched_checks
from run_analysis.physiology import grade_energy_ratio


def intervals(grade=0):
    origin=datetime(2026,1,1,tzinfo=timezone.utc)
    return [SimpleNamespace(elevation_delta_m=grade*150/100,distance_m=150.,distance_source='device',
        moving_time_s=60.,elapsed_s=60.,flags=[],start=SimpleNamespace(timestamp_utc=origin+timedelta(seconds=i*60)),
        end=SimpleNamespace(timestamp_utc=origin+timedelta(seconds=(i+1)*60))) for i in range(14)]


def test_flat_grade_curve_matches_known_continuous_speed_and_hill_uses_same_transform():
    flat=terrain_speed_curve(intervals(),12)
    hill=terrain_speed_curve(intervals(5),12)
    for a,b in zip(flat,hill):
        assert a['peak_mean_mps']==pytest.approx(2.5)
        assert b['peak_mean_mps']==pytest.approx(2.5*grade_energy_ratio(5))


def test_unknown_or_extreme_grade_breaks_continuity_not_assumed_flat_or_clipped():
    for value in (None,1000.):
        rows=intervals();rows[6].elevation_delta_m=value
        curve=terrain_speed_curve(rows,12)
        assert curve[0]['peak_mean_mps']==pytest.approx(2.5)
        assert curve[-1]['peak_mean_mps'] is None


def candidates():
    return [dict(activity_id=i,run_start=f'2026-10-{i+1:02d}T00:00:00+00:00',start_seconds=600,
        minutes_into_run=11,heat_c=10,wind_log=0,grade_ratio=1,hr_bpm=148,
        terrain_wind_speed_mps=3,adjusted_pace_min_mile=10,alpha1=1) for i in range(4)]


def test_matches_do_not_follow_alpha_or_adjusted_pace_outcomes_and_use_disjoint_runs():
    rows=candidates();first=matched_checks(rows)
    for i,r in enumerate(rows):r.update(alpha1=i*.2,adjusted_pace_min_mile=20-i*2)
    second=matched_checks(list(reversed(rows)))
    for mode in first:
        keys=lambda result:[(p['earlier']['activity_id'],p['later']['activity_id']) for p in result[mode]['pairs']]
        assert keys(first)==keys(second)
        ids=[i for pair in keys(second) for i in pair]
        assert len(ids)==len(set(ids))
        assert not second[mode]['incremental_predictive_value_established']


def test_unknown_heat_cannot_qualify_as_matching_weather():
    rows=candidates()
    for r in rows:r['heat_c']=None
    assert all(v['independent_run_pairs']==0 for v in matched_checks(rows).values())
