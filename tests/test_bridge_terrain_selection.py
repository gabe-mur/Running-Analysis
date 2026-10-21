"""Surface-aware selection uses signals, not a runtime bridge catalogue."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import sqlite3

import pytest

from run_analysis.config import load_config
from run_analysis.elevation_quality import attach_surface_deltas, prepare_terrain, tile_signature
from run_analysis.movement import classify_movement
from run_analysis.pace_comparability import _calibration_rows
from test_bridge_terrain_prototype import points_at


@pytest.fixture
def config():
    return load_config(Path(__file__).parents[1]/'config.example.yaml')


def deck():
    xs=list(range(0,1001,20))
    return points_at(xs,heights=[10+20*max(0,1-abs(d-500)/200) for d in xs])


def select(points,config,tmp_path,monkeypatch,creator='Smashrun'):
    monkeypatch.setattr('run_analysis.elevation_quality.map_elevation',lambda *args:10.)
    intervals=classify_movement(points,config['moving_time']).intervals
    quality=prepare_terrain(points,intervals,config,creator,tmp_path)
    return quality,intervals


@pytest.mark.parametrize('creator',['fr245','Smashrun'])
def test_recorded_surface_preserved_without_geometry(config,tmp_path,monkeypatch,creator):
    points=deck();before=deepcopy(points)
    quality,intervals=select(points,config,tmp_path,monkeypatch,creator)
    assert quality['source']=='mixed_recorded_surface'
    assert quality['status']=='provisional' and quality['usable']
    assert quality['surface_profile_uncertain']
    assert quality['bridge_deck_supported'] is False
    assert quality['calibration_eligible'] is False
    assert any(i.elevation_delta_m>0 for i in intervals if i.elevation_delta_m is not None)
    assert any(i.elevation_delta_m<0 for i in intervals if i.elevation_delta_m is not None)
    assert points==before


def test_constant_offset_cannot_create_a_source_boundary_hill(config,tmp_path,monkeypatch):
    points=deck()
    _,first=select(points,config,tmp_path,monkeypatch)
    _,second=select([replace(p,altitude_m=p.altitude_m+1000.) for p in points],config,tmp_path,monkeypatch)
    assert [i.elevation_delta_m for i in first]==pytest.approx([i.elevation_delta_m for i in second],nan_ok=True)
    assert max(abs(i.elevation_delta_m) for i in second if i.elevation_delta_m is not None)<3


def test_map_default_without_shape_disagreement(config,tmp_path,monkeypatch):
    points=[replace(p,altitude_m=50.) for p in deck()]
    quality,intervals=select(points,config,tmp_path,monkeypatch)
    assert quality['source']=='local_dem'
    assert quality['surface_recorded_points']==0
    assert all(i.elevation_delta_m==0 for i in intervals)


def test_missing_map_does_not_invent_a_surface_candidate(config):
    points=deck()
    intervals=classify_movement(points,config['moving_time']).intervals
    quality=prepare_terrain(points,intervals,config,'Smashrun')
    assert not quality['usable']
    assert quality['surface_recorded_points']==0
    assert all(i.elevation_delta_m is None for i in intervals)


def test_conflicting_visits_remove_only_their_neighbourhood(config,tmp_path,monkeypatch):
    # A clean barometric climb first, then contradictory visits much later.
    points=deck()
    origin=points[0].timestamp_utc
    from datetime import timedelta
    later=points_at([1200,1220,1240,1260,1280,1260,1240,1220,1200],
                    heights=[10,11,12,13,14,27,26,25,24])
    later=[replace(p,timestamp_utc=origin+timedelta(seconds=(len(points)+i)*60),
                   point_index=len(points)+i,distance_m=1200+i*20) for i,p in enumerate(later)]
    points+=later
    quality,intervals=select(points,config,tmp_path,monkeypatch,'fr265 small')
    assert quality['status']=='partial' and quality['usable']
    assert quality['local_conflict_withheld_points']>0
    assert intervals[20].elevation_delta_m is not None
    assert any(i.elevation_delta_m is None for i in intervals[-8:])


def test_surface_comparison_keeps_missing_start_missing(config,tmp_path,monkeypatch):
    xs=list(range(0,1001,20))
    points=points_at(xs,heights=[25.+d*.005 for d in xs])
    points=[replace(p,gps_valid=False,altitude_m=None) if i<10 else p for i,p in enumerate(points)]
    by_coordinate={round(p.longitude,9):10-20*max(0,1-abs(x-600)/100) for p,x in zip(points,xs)}
    monkeypatch.setattr('run_analysis.elevation_quality.map_elevation',
                        lambda lat,lon,directory:by_coordinate[round(lon,9)])
    intervals=classify_movement(points,config['moving_time']).intervals
    quality=prepare_terrain(points,intervals,config,'fr245',tmp_path)
    assert quality['map_coverage_fraction']<.9
    assert quality['surface_recorded_points']>0 and quality['usable']
    assert all(i.elevation_delta_m is None for i in intervals[:8])
    assert any(i.elevation_delta_m is not None for i in intervals[25:35])


def test_source_switch_and_missing_interval_cannot_leak_through_smoothing(config):
    points=points_at([0,10,20,30,40,50,60])
    intervals=[i for i in classify_movement(points,config['moving_time']).intervals if i.index!=2]
    attach_surface_deltas(points,intervals,[0.,0.,0.,100.,100.,100.,100.],['a']*3+['b']*4,200.)
    assert all(i.elevation_delta_m==0 for i in intervals)


def test_bridge_catalogue_does_not_affect_runtime_cache(config,tmp_path):
    c=sqlite3.connect(tmp_path/'athlete.sqlite')
    before=tile_signature(c,config)
    directory=tmp_path/'bridge_geometry';directory.mkdir()
    (directory/'irrelevant.json').write_text('{bad')
    assert tile_signature(c,config)==before
    assert c.total_changes==0
    c.close()


def test_provisional_surface_cannot_teach_shared_comparison_parameters():
    assert _calibration_rows([{'terrain_quality':{'status':'provisional','calibration_eligible':False}}])==[]


def test_both_comparison_readers_use_signal_only_selection(config,monkeypatch):
    from run_analysis.db import initialize
    from run_analysis.model_windows import load_model_window_sets,load_overlapping_model_windows
    from test_progress import _insert_run
    points=deck()
    from datetime import timedelta
    origin=points[0].timestamp_utc
    points=[replace(p,timestamp_utc=origin+timedelta(seconds=8*i)) for i,p in enumerate(points)]
    c=sqlite3.connect(':memory:');c.row_factory=sqlite3.Row;initialize(c)
    _insert_run(c,1,points[0].timestamp_utc,2,24,100)
    c.execute("UPDATE activities SET creator='Smashrun'")
    c.execute('UPDATE activity_metrics SET model_eligible=1')
    monkeypatch.setattr('run_analysis.model_windows._load_points',lambda *args:points)
    monkeypatch.setattr('run_analysis.model_windows._hourly_for_activity',lambda *args:{})
    monkeypatch.setattr('run_analysis.elevation_quality.map_elevation',lambda *args:10.)
    # If runtime tries to load bridge geometry, fail this integration test.
    monkeypatch.setattr('run_analysis.bridge_geometry.load_bridge_paths',lambda *args:pytest.fail('bridge dependency'))
    before=c.total_changes
    sets,_=load_model_window_sets(c,config,(120,),check_elevation=True)
    overlapping,_=load_overlapping_model_windows(c,config,window_seconds=120,stride_seconds=60,check_elevation=True)
    for rows in (sets[120],overlapping):
        assert rows
        assert all(r['terrain_quality']['surface_profile_uncertain'] for r in rows)
        assert any(r['grade_energy_ratio'] is not None and r['grade_energy_ratio']!=1 for r in rows)
    assert c.total_changes==before
    c.close()
