from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from run_analysis.config import load_config
from run_analysis.elevation_quality import (
    assess_recorded_elevation, map_elevation, prepare_terrain,
    recorded_elevation_source, tile_signature,
)
from run_analysis.models import Trackpoint
from run_analysis.movement import classify_movement


@pytest.fixture
def config():
    return load_config(Path(__file__).parents[1]/'config.example.yaml')


def route(drift=0., hill=.05):
    positions=[0,20,40,60,80,60,40,20,0]
    origin=datetime(2026,1,1,tzinfo=timezone.utc)
    return [Trackpoint(lap_index=0,track_index=0,point_index=i,
        timestamp_utc=origin+timedelta(seconds=i*60),latitude=40.5,
        longitude=-73.5+x/84550.,gps_valid=True,altitude_m=10+x*hill+(drift if i>4 else 0),
        distance_m=i*150.,heart_rate_bpm=145,cadence=80,run_cadence=80,cadence_source='test',speed_mps=2.5)
        for i,x in enumerate(positions)]


@pytest.mark.parametrize('creator,source',[
    ('fr245','recorded_gps_only'),('Forerunner 245','recorded_gps_only'),
    ('fr265 small','recorded_barometer_capable'),('Forerunner 265S','recorded_barometer_capable'),
    ('Smashrun','recorded_unknown'),('Some new watch','recorded_unknown'),(None,'recorded_unknown')])
def test_source_is_explicit_not_inferred_from_enhanced_altitude(creator,source):
    assert recorded_elevation_source(creator)==source


@pytest.mark.parametrize('creator',['fr245','fr265 small'])
def test_conflicting_revisits_reject_terrain_not_the_run(config,creator):
    points=route(drift=14);before=deepcopy(points)
    movement=classify_movement(points,config['moving_time'])
    quality=prepare_terrain(points,movement.intervals,config,creator)
    assert quality['status']=='rejected'
    assert quality['inconsistent_pairs']>=3
    assert quality['maximum_repeat_difference_m']==pytest.approx(14)
    assert all(i.elevation_delta_m is None for i in movement.intervals)
    assert sum(i.moving_time_s for i in movement.intervals)>0
    assert points==before


def test_real_hill_with_consistent_return_visits_is_retained(config):
    points=route(hill=.1);movement=classify_movement(points,config['moving_time'])
    quality=prepare_terrain(points,movement.intervals,config,'fr265 small')
    assert quality['status']=='supported'
    assert quality['source']=='recorded_barometer_capable'
    assert quality['inconsistent_pairs']==0
    assert any(i.elevation_delta_m>0 for i in movement.intervals)
    assert any(i.elevation_delta_m<0 for i in movement.intervals)


def test_consistent_loop_is_not_an_independent_terrain_reference(config):
    points=route();movement=classify_movement(points,config['moving_time'])
    quality=prepare_terrain(points,movement.intervals,config,'fr245')
    assert quality['repeat_pairs']>=3 and quality['inconsistent_pairs']==0
    assert quality['status']=='unverified' and not quality['usable']


def test_unknown_or_gps_only_without_support_has_no_invented_flatness(config):
    points=route()[:4]
    for creator in ('fr245','unknown'):
        movement=classify_movement(points,config['moving_time'])
        quality=prepare_terrain(points,movement.intervals,config,creator)
        assert quality['status']=='unverified'
        assert not quality['usable']
        assert all(i.elevation_delta_m is None for i in movement.intervals)
    movement=classify_movement(points,config['moving_time'])
    assert prepare_terrain(points,movement.intervals,config,'fr265 small')['usable']


def test_barometer_does_not_bypass_contradictory_evidence():
    assert assess_recorded_elevation(route(drift=14),'fr265 small')['status']=='rejected'


def test_isolated_outlier_is_not_multiple_independent_bad_visits():
    points=route()
    points[-1]=replace(points[-1],altitude_m=100)
    assert assess_recorded_elevation(points,'fr265 small')['status']!='rejected'


def write_tile(path,values):
    np.asarray(values,dtype='>i2').tofile(path)


def test_local_dem_interpolation_orientation_missing_and_void(tmp_path):
    tile=np.zeros((1201,1201),dtype=np.int16)
    tile[0,0]=10;tile[0,1]=20;tile[1,0]=30;tile[1,1]=40
    path=tmp_path/'N40W074.hgt';write_tile(path,tile)
    assert map_elevation(41-.5/1200,-74+.5/1200,tmp_path)==pytest.approx(25)
    assert map_elevation(40.5,-72.5,tmp_path) is None
    tile[0,0]=-32768;write_tile(path,tile)
    assert map_elevation(41-.5/1200,-74+.5/1200,tmp_path) is None
    path.write_bytes(b'bad tile')
    assert map_elevation(40.5,-73.5,tmp_path) is None


def test_southern_eastern_tile_naming(tmp_path):
    write_tile(tmp_path/'S01E030.hgt',np.full((1201,1201),123,dtype=np.int16))
    assert map_elevation(-.5,30.5,tmp_path)==123


def test_map_replaces_bad_recorded_profile_without_mutating_source(config,tmp_path):
    points=route(drift=14);before=deepcopy(points)
    write_tile(tmp_path/'N40W074.hgt',np.full((1201,1201),10,dtype=np.int16))
    movement=classify_movement(points,config['moving_time'])
    quality=prepare_terrain(points,movement.intervals,config,'fr245',tmp_path)
    assert quality['source']=='local_dem' and quality['usable']
    assert quality['recorded_status']=='rejected'
    assert all(i.elevation_delta_m==pytest.approx(0) for i in movement.intervals)
    assert points==before


def test_map_voids_never_splice_to_raw_altitude(config,tmp_path):
    points=route(drift=14)
    write_tile(tmp_path/'N40W074.hgt',np.full((1201,1201),-32768,dtype=np.int16))
    movement=classify_movement(points,config['moving_time'])
    quality=prepare_terrain(points,movement.intervals,config,'fr245',tmp_path)
    assert quality['source']=='recorded_gps_only' and not quality['usable']
    assert all(i.elevation_delta_m is None for i in movement.intervals)


def test_good_barometric_profile_is_not_replaced_by_coarser_map(config,tmp_path):
    points=route(hill=.1)
    write_tile(tmp_path/'N40W074.hgt',np.full((1201,1201),100,dtype=np.int16))
    movement=classify_movement(points,config['moving_time'])
    quality=prepare_terrain(points,movement.intervals,config,'fr265 small',tmp_path)
    assert quality['source']=='recorded_barometer_capable'
    assert any(i.elevation_delta_m>0 for i in movement.intervals)


def test_bridge_deck_from_supported_barometer_is_not_replaced_by_riverbed(config,tmp_path,monkeypatch):
    # Synthetic deck rises above a river; ground DEM describes the ground below,
    # not the surface being run on. Both passes use the same deck elevation.
    points=route(hill=0)
    for point in points:
        position=(point.longitude+73.5)*84550.
        point.altitude_m=20+max(0,1-abs(position-40)/40)*12
    monkeypatch.setattr('run_analysis.elevation_quality.map_elevation',lambda *args:-5.)
    movement=classify_movement(points,config['moving_time'])
    quality=prepare_terrain(points,movement.intervals,config,'fr265 small',tmp_path)
    assert quality['source']=='recorded_barometer_capable'
    assert any(i.elevation_delta_m>1 for i in movement.intervals)
    assert any(i.elevation_delta_m< -1 for i in movement.intervals)
    assert points[0].altitude_m>=20  # No replacement of source observations.


def test_bare_earth_map_does_not_claim_bridge_deck_support(config,tmp_path,monkeypatch):
    points=route(drift=14)
    monkeypatch.setattr('run_analysis.elevation_quality.map_elevation',lambda *args:0.)
    movement=classify_movement(points,config['moving_time'])
    quality=prepare_terrain(points,movement.intervals,config,'fr245',tmp_path)
    assert quality['source']=='local_dem'
    assert quality['surface']=='bare_earth'
    assert quality['bridge_deck_supported'] is False
    assert 'Bridge-deck height is not guaranteed' in quality['reason']


def test_stacked_passages_cannot_be_resolved_by_snapping_bad_barometer_to_ground(config,tmp_path,monkeypatch):
    points=route(drift=14)
    monkeypatch.setattr('run_analysis.elevation_quality.map_elevation',lambda *args:0.)
    movement=classify_movement(points,config['moving_time'])
    quality=prepare_terrain(points,movement.intervals,config,'fr265 small',tmp_path)
    assert quality['map_coverage_fraction']==1
    assert not quality['usable']
    assert quality['source']=='recorded_barometer_capable'
    assert all(i.elevation_delta_m is None for i in movement.intervals)
    assert 'bridge or road levels' in quality['reason']


def test_local_map_version_changes_without_database_write(config,tmp_path):
    import sqlite3
    c=sqlite3.connect(tmp_path/'test.sqlite')
    cfg=deepcopy(config);cfg['elevation']['dem_directory']=str(tmp_path)
    before=tile_signature(c,cfg)
    path=tmp_path/'N40W074.hgt';write_tile(path,np.zeros((1201,1201),dtype=np.int16))
    after=tile_signature(c,cfg)
    assert before!=after
    path.write_bytes(b'invalid')
    assert tile_signature(c,cfg)!=after
    c.close()


def test_both_comparison_window_readers_apply_the_same_guard_without_dropping_windows(config,monkeypatch):
    import sqlite3
    from run_analysis.db import initialize
    from run_analysis.model_windows import load_model_window_sets,load_overlapping_model_windows
    from test_progress import _insert_run
    points=route(drift=14)
    c=sqlite3.connect(':memory:');c.row_factory=sqlite3.Row;initialize(c)
    _insert_run(c,1,points[0].timestamp_utc,2,24,100)
    c.execute("UPDATE activities SET creator='fr245'")
    c.execute('UPDATE activity_metrics SET model_eligible=1')
    monkeypatch.setattr('run_analysis.model_windows._load_points',lambda *args:points)
    monkeypatch.setattr('run_analysis.model_windows._hourly_for_activity',lambda *args:{})
    before=c.total_changes
    sets,_=load_model_window_sets(c,config,(120,),check_elevation=True)
    overlapping,_=load_overlapping_model_windows(c,config,window_seconds=120,stride_seconds=60,check_elevation=True)
    for rows in (sets[120],overlapping):
        assert rows
        assert all(r['terrain_quality']['status']=='rejected' for r in rows)
        assert all(r['grade_energy_ratio'] is None and r['grade_complete_fraction']==0 for r in rows)
        assert all(r['average_hr_bpm']==145 and r['moving_pace_min_mile']>0 for r in rows)
    assert c.total_changes==before
    c.close()


def test_comparison_cache_invalidates_when_local_maps_are_added(config,tmp_path,monkeypatch):
    import sqlite3
    from run_analysis.pace_comparability import cached_comparability
    from run_analysis.read_cache import invariant_snapshot
    c=sqlite3.connect(tmp_path/'test.sqlite');c.row_factory=sqlite3.Row
    c.execute('CREATE TABLE evidence(id INTEGER)');c.commit()
    cfg=deepcopy(config);cfg['elevation']['dem_directory']=str(tmp_path)
    calls=[]
    def build(*args):
        calls.append(1)
        return {'build':len(calls)}
    monkeypatch.setattr('run_analysis.pace_comparability.build_pace_comparability',build)
    with invariant_snapshot(c):
        assert cached_comparability(c,cfg)['build']==1
        assert cached_comparability(c,cfg)['build']==1
    write_tile(tmp_path/'N40W074.hgt',np.zeros((1201,1201),dtype=np.int16))
    with invariant_snapshot(c):
        assert cached_comparability(c,cfg)['build']==2
    assert len(calls)==2
    c.close()
