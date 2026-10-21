from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from run_analysis.config import load_config
from run_analysis.movement import classify_movement
from run_analysis.surface_mismatch import surface_candidates
from test_bridge_terrain_prototype import points_at


@pytest.fixture
def config():
    return load_config(Path(__file__).parents[1]/'config.example.yaml')


def detect(points, mapped, config):
    return surface_candidates(points, classify_movement(points, config['moving_time']).intervals, mapped, config)


def test_bounded_climb_needs_no_bridge_catalogue(config):
    x = list(range(0,1001,20))
    heights = [10+20*max(0,1-abs(d-500)/200) for d in x]
    points = points_at(x, heights=heights)
    result = detect(points, [10.]*len(x), config)
    assert result['mask'][25]
    assert not result['mask'][0] and not result['mask'][-1]
    assert any(c['kind']=='bounded_raised_profile' for r in result['regions'] for c in r['candidates'])


def test_constant_recorded_datum_offset_does_not_change_selection(config):
    x = list(range(0,1001,20)); heights = [10+20*max(0,1-abs(d-500)/200) for d in x]
    points = points_at(x, heights=heights)
    shifted = [replace(p,altitude_m=p.altitude_m+500) for p in points]
    assert detect(points,[10.]*len(x),config)['mask'] == detect(shifted,[10.]*len(x),config)['mask']


def test_real_terrain_agreement_keeps_maps(config):
    x=list(range(0,1001,20));ground=[10+20*np.sin(d/300) for d in x]
    points=points_at(x,heights=[v+40 for v in ground])
    assert not any(detect(points,ground,config)['mask'])


def test_constant_offset_or_unbounded_drift_is_not_an_omitted_bridge(config):
    x=list(range(0,1001,20))
    for heights in ([50.]*len(x),[10+d*.04 for d in x]):
        assert not any(detect(points_at(x,heights=heights),[10.]*len(x),config)['mask'])


def test_noisy_recording_is_not_promoted_over_quiet_ground(config):
    x=list(range(0,3001,20));heights=[10+14*np.sin(d/100) for d in x]
    assert not any(detect(points_at(x,heights=heights),[10.]*len(x),config)['mask'])


def test_map_valley_candidate_works_after_missing_start(config):
    x=list(range(0,1001,20));points=points_at(x,heights=[25.]*len(x))
    points=[replace(p,gps_valid=False,altitude_m=None) if i<10 else p for i,p in enumerate(points)]
    ground=[10-20*max(0,1-abs(d-600)/100) for d in x]
    ground=[None if i<10 else v for i,v in enumerate(ground)]
    result=detect(points,ground,config)
    assert not any(result['mask'][:10])
    assert result['mask'][30]
    assert any(c['kind']=='ground_dip_below_smoother_recording' for r in result['regions'] for c in r['candidates'])


def test_missing_map_or_coordinates_never_becomes_a_height_anchor(config):
    points=points_at(list(range(0,1001,20)),heights=[10.]*51)
    assert not any(detect(points,[None]*51,config)['mask'])
    points=[replace(p,gps_valid=False) for p in points]
    assert not any(detect(points,[0.]*51,config)['mask'])


def test_spike_is_not_a_coherent_raised_surface(config):
    x=list(range(0,1001,20));heights=[10.]*len(x);heights[25]=10000.
    assert not any(detect(points_at(x,heights=heights),[10.]*len(x),config)['mask'])


def test_partial_recording_without_a_comparable_map_is_not_invented(config):
    points=points_at([0,10,20],heights=[20,25,30])
    assert not any(detect(points,[0,0,0],config)['mask'])
