import gzip
import json
from pathlib import Path
import sqlite3

import numpy as np
import pytest

from run_analysis.elevation_download import download_tile,needed_tiles,tile_name,tile_url,ATTRIBUTION_URL


@pytest.mark.parametrize('lat,lon,name',[(40.1,-73.9,'N40W074'),(-.1,30.1,'S01E030'),(0,0,'N00E000')])
def test_tile_selection_uses_southwestern_corner(lat,lon,name):
    assert tile_name(lat,lon)==name
    assert tile_url(name).endswith(f'/{name[:3]}/{name}.hgt.gz')


@pytest.mark.parametrize('name',['../outside','N99W074','N40E999','https://elsewhere','N40W074.hgt'])
def test_download_target_is_a_bounded_tile_not_a_url_or_path(name,tmp_path):
    with pytest.raises(ValueError):download_tile(name,tmp_path)


def tile_bytes():
    return np.full((1201,1201),12,dtype='>i2').tobytes()


def test_explicit_download_preserves_source_attribution_and_is_reusable_offline(tmp_path):
    calls=[];data=tile_bytes()
    def fetch(url,maximum):
        calls.append(url)
        return b'# Upstream terrain attribution\n' if url==ATTRIBUTION_URL else gzip.compress(data)
    result=download_tile('N40W074',tmp_path,fetch=fetch)
    assert result['status']=='downloaded'
    assert (tmp_path/'N40W074.hgt').read_bytes()==data
    metadata=json.loads((tmp_path/'N40W074.metadata.json').read_text())
    assert metadata['surface']=='bare_earth' and not metadata['bridge_deck_supported']
    assert metadata['sha256']==result['sha256']
    assert (tmp_path/'ATTRIBUTION.md').exists()
    result=download_tile('N40W074',tmp_path,fetch=lambda *a:pytest.fail('Cached tile made a network request'))
    assert result['status']=='cached'
    assert len(calls)==2
    assert not list(tmp_path.glob('*.part'))


@pytest.mark.parametrize('body',[b'not gzip',gzip.compress(b'too short')])
def test_failed_download_never_publishes_a_partial_map(tmp_path,body):
    with pytest.raises(ValueError):
        download_tile('N40W074',tmp_path,fetch=lambda url,maximum:b'credits' if url==ATTRIBUTION_URL else body)
    assert not (tmp_path/'N40W074.hgt').exists()
    assert not list(tmp_path.glob('*.part'))


def test_existing_invalid_map_is_not_overwritten(tmp_path):
    path=tmp_path/'N40W074.hgt';path.write_bytes(b'preserve me')
    with pytest.raises(ValueError,match='Existing terrain tile'):
        download_tile('N40W074',tmp_path,fetch=lambda *a:pytest.fail('Should not fetch'))
    assert path.read_bytes()==b'preserve me'


def test_missing_tile_can_recover_with_its_existing_matching_provenance(tmp_path):
    data=tile_bytes()
    fetch=lambda url,maximum:b'credits' if url==ATTRIBUTION_URL else gzip.compress(data)
    download_tile('N40W074',tmp_path,fetch=fetch)
    (tmp_path/'N40W074.hgt').unlink()
    assert download_tile('N40W074',tmp_path,fetch=fetch)['status']=='downloaded'


def test_download_planning_is_read_only_and_only_requests_tile_names():
    c=sqlite3.connect(':memory:')
    c.executescript('CREATE TABLE trackpoints(activity_id INTEGER,gps_valid INTEGER,latitude REAL,longitude REAL);'
        'CREATE TABLE activity_metrics(activity_id INTEGER,model_eligible INTEGER);'
        'INSERT INTO activity_metrics VALUES(1,1),(2,0);'
        'INSERT INTO trackpoints VALUES(1,1,40.1,-73.9),(1,1,40.2,-73.8),(1,0,42,-72),(2,1,43,-71);')
    before=c.total_changes
    assert needed_tiles(c)==['N40W074']
    assert needed_tiles(c,activity_id=2)==[]
    assert c.total_changes==before
    c.close()
