"""Read-only local terrain-selection audit; no bridge catalogue or network."""
import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3
from time import perf_counter

from run_analysis.config import load_config, resolve_project_path
from run_analysis.elevation_quality import prepare_terrain, tile_directory
from run_analysis.fit import parse_fit
from run_analysis.movement import classify_movement
from run_analysis.privacy import private_file
from run_analysis.processing import _load_points
from run_analysis.read_cache import invariant_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--activity-id',type=int,action='append')
    parser.add_argument('--source-fit',nargs=2,action='append',default=[],metavar=('ACTIVITY_ID','PATH'))
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    cfg = load_config('config.yaml')
    database = resolve_project_path(Path.cwd(),cfg['paths']['database'])
    overrides = {int(key):Path(path) for key,path in args.source_fit}
    result = dict(scope='Terrain source selection only; no physiology validation or canonical writes',runs=[])
    start = perf_counter()
    with closing(sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True)) as c:
        c.row_factory = sqlite3.Row
        with invariant_snapshot(c):
            directory = tile_directory(c,cfg)
            activities = c.execute('SELECT id,start_time_local,creator FROM activities ORDER BY id').fetchall()
            if args.activity_id and set(args.activity_id)-{a['id'] for a in activities}:
                parser.error('Requested activity is unavailable')
            for a in activities:
                if args.activity_id and a['id'] not in args.activity_id:
                    continue
                stored = _load_points(c,a['id'])
                variants = [('stored',stored,a['creator'])]
                if a['id'] in overrides:
                    parsed = parse_fit(overrides[a['id']],cfg['timezone_default']).activities
                    if len(parsed)!=1 or not stored or abs((parsed[0].start_time_utc-stored[0].timestamp_utc).total_seconds())>60:
                        parser.error('Original FIT must match the selected activity')
                    variants.append(('original_fit',parsed[0].trackpoints,parsed[0].creator))
                for variant,points,creator in variants:
                    movement = classify_movement(points,cfg['moving_time'])
                    quality = prepare_terrain(points,movement.intervals,cfg,creator,directory)
                    known = [i for i in movement.intervals if i.elevation_delta_m is not None]
                    result['runs'].append(dict(activity_id=a['id'],date=a['start_time_local'],variant=variant,
                        creator=creator,points=len(points),gps_points=sum(p.gps_valid for p in points),
                        quality=quality,known_grade_m=sum(i.distance_m for i in known),
                        ascent_m=sum(max(0.,i.elevation_delta_m) for i in known) if known else None,
                        descent_m=sum(max(0.,-i.elevation_delta_m) for i in known) if known else None))
            result['database_writes'] = c.total_changes
            assert c.total_changes==0
    result['elapsed_seconds'] = perf_counter()-start
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n'); private_file(args.output)
    chosen = [r for r in result['runs'] if r['quality']['surface_recorded_points']]
    print(json.dumps(dict(runs=len(result['runs']),selected_runs=[dict(activity_id=r['activity_id'],variant=r['variant'],
        selected_points=r['quality']['surface_recorded_points']) for r in chosen],
        database_writes=result['database_writes'],elapsed_seconds=result['elapsed_seconds']),indent=2))


if __name__=='__main__':
    main()
