"""Explicit, bounded download of public terrain tiles for offline comparison.

Normal analysis never imports this downloader or makes elevation requests.
Only one-degree tile names are sent, never points or athlete/session details.
"""
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from math import floor, isfinite
import os
from pathlib import Path
import re
import tempfile
from urllib.request import Request, HTTPRedirectHandler, build_opener

from .privacy import private_directory, private_file

PROVIDER = 'Tilezen / Mapzen Terrain Tiles'
BASE_URL = 'https://elevation-tiles-prod.s3.amazonaws.com/skadi'
ATTRIBUTION_URL = 'https://raw.githubusercontent.com/tilezen/joerd/master/docs/attribution.md'
MAX_DOWNLOAD_BYTES = 32 * 1024 * 1024
MAX_TILE_BYTES = 3601 * 3601 * 2
VALID_TILE_BYTES = {1201 * 1201 * 2, MAX_TILE_BYTES}


def tile_name(latitude, longitude):
    if not (isfinite(latitude) and isfinite(longitude) and -90 < latitude < 90 and -180 <= longitude < 180):
        raise ValueError('Invalid terrain coordinate')
    south, west = floor(latitude), floor(longitude)
    return f'{"N" if south>=0 else "S"}{abs(south):02d}{"E" if west>=0 else "W"}{abs(west):03d}'


def needed_tiles(connection, *, activity_id=None):
    query = '''SELECT t.latitude,t.longitude FROM trackpoints t
        JOIN activity_metrics m ON m.activity_id=t.activity_id
        WHERE m.model_eligible=1 AND t.gps_valid=1
        AND t.latitude IS NOT NULL AND t.longitude IS NOT NULL'''
    args = ()
    if activity_id is not None:
        query += ' AND t.activity_id=?'
        args = (activity_id,)
    return sorted({tile_name(float(lat), float(lon)) for lat, lon in connection.execute(query,args)})


def tile_url(name):
    if not re.fullmatch(r'[NS]\d{2}[EW]\d{3}', name):
        raise ValueError('Invalid terrain tile name')
    latitude, longitude = int(name[1:3]), int(name[4:7])
    if latitude > (89 if name[0]=='N' else 90) or longitude > (179 if name[3]=='E' else 180):
        raise ValueError('Terrain tile is outside the supported world grid')
    return f'{BASE_URL}/{name[:3]}/{name}.hgt.gz'


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def _fetch(url, maximum_bytes):
    request = Request(url, headers={'User-Agent':'RunningAnalysis-offline-terrain/1.0'})
    with build_opener(_NoRedirect).open(request, timeout=45) as response:
        # Refuse redirects to an unexpected recipient or plaintext transport.
        if response.geturl() != url:
            raise ValueError('Unexpected terrain download redirect')
        data = response.read(maximum_bytes+1)
    if len(data)>maximum_bytes:
        raise ValueError('Terrain download exceeded the size limit')
    return data


def _atomic_new(path, data):
    """A failed request cannot publish a partial tile or overwrite local maps."""
    if path.exists():
        raise FileExistsError(f'Local terrain file already exists: {path.name}')
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.terrain-',suffix='.part',delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        private_file(temporary)
        # Hard-link publication is atomic and fails if another process won.
        os.link(temporary,path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def download_tile(name, directory, *, fetch=_fetch):
    url = tile_url(name)
    directory = private_directory(directory)
    path = directory / f'{name}.hgt'
    if path.exists():
        if path.stat().st_size not in VALID_TILE_BYTES:
            raise ValueError(f'Existing terrain tile is invalid; preserve and inspect it: {path.name}')
        return dict(tile=name,status='cached',bytes=path.stat().st_size)
    # Keep the upstream source/licensing information beside the data. This is
    # ordinary public documentation; no tile names are included in this request.
    attribution = directory / 'ATTRIBUTION.md'
    if not attribution.exists():
        credits = fetch(ATTRIBUTION_URL,256*1024)
        credits.decode('utf-8')
        _atomic_new(attribution,credits)
    packed = fetch(url,MAX_DOWNLOAD_BYTES)
    if len(packed)>MAX_DOWNLOAD_BYTES:
        raise ValueError('Terrain download exceeded the size limit')
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(packed)) as handle:
            data = handle.read(MAX_TILE_BYTES+1)
    except (OSError, EOFError) as error:
        raise ValueError('Terrain download is not a valid gzip tile') from error
    if len(data) not in VALID_TILE_BYTES:
        raise ValueError('Terrain tile has invalid raster dimensions or excessive size')
    metadata = dict(provider=PROVIDER,url=url,downloaded_at_utc=datetime.now(timezone.utc).isoformat(),
        bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),
        surface='bare_earth',bridge_deck_supported=False,
        attribution='ATTRIBUTION.md',source_information='https://registry.opendata.aws/terrain-tiles/')
    metadata_path=path.with_suffix('.metadata.json')
    if metadata_path.exists():
        previous=json.loads(metadata_path.read_text())
        if previous.get('sha256')!=metadata['sha256'] or previous.get('url')!=url:
            raise ValueError(f'Existing terrain provenance conflicts with download: {metadata_path.name}')
    else:
        _atomic_new(metadata_path,(json.dumps(metadata,indent=2)+'\n').encode())
    _atomic_new(path,data)
    return dict(tile=name,status='downloaded',bytes=len(data),sha256=metadata['sha256'])
