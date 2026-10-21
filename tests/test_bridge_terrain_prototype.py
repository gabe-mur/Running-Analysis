from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from math import cos, radians
from pathlib import Path

import pytest

from run_analysis.bridge_terrain_prototype import (
    BridgePath, attach_prototype_deltas, bridge_paths_from_osm, prototype_bridge_profile,
)
from run_analysis.config import load_config
from run_analysis.models import Trackpoint
from run_analysis.movement import attach_elevation_deltas, classify_movement


def coordinate(x, y=0):
    return 45+y/111195., x/(111195.*cos(radians(45)))


def points_at(xs, *, ys=None, heights=None):
    origin = datetime(2026, 1, 1, tzinfo=timezone.utc)
    result = []
    distance = 0.
    for i, x in enumerate(xs):
        if i:
            distance += abs(x-xs[i-1])
        lat, lon = coordinate(x, ys[i] if ys else 0)
        result.append(Trackpoint(0, 0, i, origin+timedelta(seconds=60*i), lat, lon, True,
            heights[i] if heights else 10., distance, 145, 80, 80, 'test', 2.5))
    return result


def bridge():
    return BridgePath((1,), 'Test bridge', (coordinate(100), coordinate(300)))


def test_endpoint_estimate_removes_a_ground_valley_without_calling_it_measured():
    points = points_at([0, 50, 100, 150, 200, 250, 300, 350, 400])
    mapped = [10., 10., 10., -10., -20., -10., 10., 10., 10.]
    before = deepcopy(points), list(mapped)
    result = prototype_bridge_profile(points, mapped, [bridge()], lambda *p: 10., creator='fr245')
    assert result['endpoint_estimate'] == pytest.approx([10.]*9)
    assert result['bridges'][0]['follows_geometry']
    assert not result['bridges'][0]['measured_deck_height']
    assert result['guarded_elevations'][4] is None
    assert result['point_sources'][4] == 'bridge_level_uncertain'
    assert (points, mapped) == before


def test_partial_out_and_back_uses_structure_endpoints_not_turnaround():
    points = points_at([90, 110, 150, 200, 180, 120, 90])
    def lookup(lat, lon):
        return 10. if lon < coordinate(200)[1] else 30.
    result = prototype_bridge_profile(points, [0.]*7, [bridge()], lookup)
    assert result['endpoint_estimate'][3] == pytest.approx(20., abs=.01)
    assert result['endpoint_estimate'][2] == pytest.approx(15., abs=.01)
    assert result['endpoint_estimate'][4] > result['endpoint_estimate'][5]


def test_supported_barometer_keeps_real_bridge_hump():
    heights = [10, 10, 10, 20, 30, 20, 10, 10, 10]
    points = points_at(list(range(0, 450, 50)), heights=heights)
    result = prototype_bridge_profile(points, [0.]*9, [bridge()], lambda *p: 0., creator='fr265 small')
    assert result['guarded_elevations'] == heights
    assert max(result['endpoint_estimate']) == 0  # Approximation cannot know the hump.
    assert set(result['point_sources']) == {'recorded_barometer'}


def test_unverified_hump_is_not_silently_accepted_as_flat_measured_terrain():
    points = points_at([0, 100, 150, 200, 250, 300, 400], heights=[10, 10, 20, 30, 20, 10, 10])
    result = prototype_bridge_profile(points, [10.]*7, [bridge()], lambda *p: 10., creator='fr245')
    assert result['endpoint_estimate'][3] == 10
    assert result['guarded_elevations'][3] is None
    assert result['guarded_elevations'][0] == 10


def test_underpass_is_ambiguous_not_declared_a_bridge_traversal():
    points = points_at([200]*5, ys=[-60, -30, 0, 30, 60])
    result = prototype_bridge_profile(points, [5.]*5, [bridge()], lambda *p: 15.)
    assert not result['bridges'][0]['follows_geometry']
    assert result['bridges'][0]['status'] == 'proximity_only'
    assert result['endpoint_estimate'] == [5.]*5
    assert not any(result['estimated_mask'])
    assert result['guarded_elevations'][2] is None
    assert result['guarded_elevations'][0] == 5


def test_missing_anchor_and_missing_map_samples_remain_missing():
    points = points_at([100, 150, 200, 250, 300])
    result = prototype_bridge_profile(points, [5.]*5, [bridge()], lambda *p: None)
    assert result['endpoint_estimate'] == [None]*5
    assert result['bridges'][0]['status'] == 'missing_anchors'
    result = prototype_bridge_profile(points, [5., 5., None, 5., 5.], [bridge()], lambda *p: 10.)
    assert result['endpoint_estimate'][2] is None


def test_missing_gps_never_matches_bridge():
    points = [replace(p, gps_valid=False) for p in points_at([100, 200, 300])]
    result = prototype_bridge_profile(points, [None]*3, [bridge()], lambda *p: 10.)
    assert result['bridge_mask'] == [False]*3
    assert result['guarded_elevations'] == [None]*3


def test_contradictory_barometer_is_not_used_to_keep_a_hump():
    points = points_at([0, 20, 40, 60, 80, 60, 40, 20, 0],
        heights=[10, 11, 12, 13, 14, 27, 26, 25, 24])
    path = BridgePath((1,), 'Test', (coordinate(0), coordinate(80)))
    result = prototype_bridge_profile(points, [10.]*9, [path], lambda *p: 10., creator='fr265 small')
    assert result['recorded_quality']['status'] == 'rejected'
    assert all(v is None for v in result['guarded_elevations'])


def test_no_bridge_preserves_map_profile_and_existing_smoothing():
    points = points_at([0, 20, 40, 60, 80], heights=[10, 11, 12, 13, 14])
    mapped = [p.altitude_m for p in points]
    result = prototype_bridge_profile(points, mapped, [], lambda *p: 10.)
    assert result['guarded_elevations'] == result['endpoint_estimate'] == mapped
    config = load_config(Path(__file__).parents[1]/'config.example.yaml')
    first = classify_movement(points, config['moving_time']).intervals
    second = classify_movement(points, config['moving_time']).intervals
    attach_elevation_deltas(points, first, 60.)
    attach_prototype_deltas(points, second, mapped, 60.)
    assert [i.elevation_delta_m for i in first] == [i.elevation_delta_m for i in second]


def test_smoothing_cannot_fill_bridge_gap_or_leak_a_step_into_other_grades():
    points = points_at([0, 10, 20, 30, 40, 50, 60])
    config = load_config(Path(__file__).parents[1]/'config.example.yaml')
    intervals = classify_movement(points, config['moving_time']).intervals
    attach_prototype_deltas(points, intervals, [0., 0., 0., None, 100., 100., 100.], 200.)
    assert [i.elevation_delta_m for i in intervals] == [0., 0., None, None, 0., 0.]


def osm_payload():
    return {'elements': [
        *[{'type': 'node', 'id': i, 'lat': coordinate(100*i)[0], 'lon': coordinate(100*i)[1]} for i in range(1, 5)],
        {'type': 'way', 'id': 10, 'nodes': [1, 2], 'tags': {'highway': 'footway', 'bridge': 'yes', 'name': 'Bridge'}},
        {'type': 'way', 'id': 11, 'nodes': [3, 2], 'tags': {'highway': 'footway', 'bridge': 'yes', 'name': 'Bridge'}},
    ]}


def test_connected_reversed_way_pieces_join_before_selecting_endpoints():
    paths, info = bridge_paths_from_osm([osm_payload()])
    assert len(paths) == 1
    assert paths[0].coordinates == (coordinate(100), coordinate(200), coordinate(300))
    assert paths[0].way_ids == (10, 11)
    assert not info['skipped_way_ids']


def test_branches_and_different_levels_are_not_joined_into_one_deck():
    payload = osm_payload()
    payload['elements'].append({'type': 'way', 'id': 12, 'nodes': [2, 4],
        'tags': {'highway': 'footway', 'bridge': 'yes'}})
    paths, _ = bridge_paths_from_osm([payload])
    assert len(paths) == 3
    payload = osm_payload()
    payload['elements'][-1]['tags']['layer'] = '2'
    assert len(bridge_paths_from_osm([payload])[0]) == 2


def test_outlines_incomplete_ways_and_closed_loops_are_not_interpolation_paths():
    payload = osm_payload()
    payload['elements'][-1]['nodes'] = [2, 999]
    payload['elements'].extend([
        {'type': 'way', 'id': 12, 'nodes': [1, 2, 1], 'tags': {'man_made': 'bridge'}},
        {'type': 'way', 'id': 13, 'nodes': [1, 2, 1], 'tags': {'highway': 'footway', 'bridge': 'yes'}},
    ])
    paths, info = bridge_paths_from_osm([payload])
    assert [p.way_ids for p in paths] == [(10,)]
    assert info['skipped_way_ids'] == [11, 13]


def test_matching_works_across_dateline_without_a_world_spanning_segment():
    path = BridgePath((1,), 'Dateline', ((0., 179.999), (0., -179.999)))
    points = [replace(p, latitude=0., longitude=lon) for p, lon in zip(points_at([0, 100, 200]), [179.9995, -180., -179.9995])]
    result = prototype_bridge_profile(points, [0.]*3, [path], lambda *p: 10.)
    assert result['bridge_mask'] == [True]*3
    assert result['bridges'][0]['length_m'] == pytest.approx(222.39, abs=.1)


def test_invalid_lengths_fail_explicitly():
    with pytest.raises(ValueError):
        prototype_bridge_profile(points_at([0, 1]), [0.], [], lambda *p: 0.)
