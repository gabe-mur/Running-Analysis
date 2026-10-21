"""Plan or explicitly download terrain tiles; never runs during page loads.

This shares coarse tile areas with the named public provider. Use --download
only after the athlete authorizes that download. No points/files are uploaded.
"""
import argparse
from contextlib import closing
from pathlib import Path
import sqlite3

from run_analysis.config import load_config,resolve_project_path
from run_analysis.elevation_download import download_tile,needed_tiles
from run_analysis.elevation_quality import tile_directory


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('config.yaml'))
    parser.add_argument('--activity-id',type=int)
    parser.add_argument('--download',action='store_true',help='Explicitly authorize public tile requests')
    parser.add_argument('--max-tiles',type=int,default=16)
    args=parser.parse_args()
    if args.max_tiles<1:parser.error('--max-tiles must be positive')
    config=load_config(args.config)
    database=resolve_project_path(args.config.resolve().parent,config['paths']['database'])
    with closing(sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True)) as connection:
        names=needed_tiles(connection,activity_id=args.activity_id)
        directory=tile_directory(connection,config)
    if len(names)>args.max_tiles:
        parser.error(f'{len(names)} tiles exceeds the explicit limit of {args.max_tiles}')
    print(f'{len(names)} terrain tiles needed; directory: {directory}')
    print('Provider: Tilezen Terrain Tiles on public AWS S3. Only coarse tile areas are requested.')
    print('Bridge-deck elevation is not guaranteed by this bare-earth dataset.')
    for name in names:
        if args.download:
            result=download_tile(name,directory)
            print(f"{name}: {result['status']} ({result['bytes']/1024**2:.1f} MiB)",flush=True)
        else:
            print(name)
    if not args.download:print('Dry run only. Use --download after authorizing requests to the named provider.')


if __name__=='__main__':main()
