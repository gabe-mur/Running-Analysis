"""Read-only bridge experiment using already downloaded public OSM geometry.

Run IDs and any original FIT overrides are explicit CLI inputs, not geographic
rules. Nothing is imported, fitted, published, downloaded or scheduled here.
"""
import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3

from run_analysis.bridge_terrain_prototype import (
    VERSION, MATCH_RADIUS_M, MINIMUM_MATCHED_POINTS, MINIMUM_MATCHED_SPAN_M,
    attach_prototype_deltas, bridge_paths_from_osm, prototype_bridge_profile,
)
from run_analysis.config import load_config, resolve_project_path
from run_analysis.elevation_quality import map_elevation, tile_directory
from run_analysis.fit import parse_fit
from run_analysis.movement import classify_movement
from run_analysis.privacy import private_file
from run_analysis.processing import _load_points
from run_analysis.read_cache import invariant_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--activity-id', type=int, action='append', required=True)
    parser.add_argument('--elevation-only', type=int, action='append', default=[])
    parser.add_argument('--source-fit', nargs=2, action='append', default=[], metavar=('ACTIVITY_ID', 'PATH'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = load_config('config.yaml')
    originals = {int(identifier): Path(path) for identifier, path in args.source_fit}
    payloads = [json.loads(path.read_text()) for path in sorted(args.input_dir.glob('osm-*.json'))]
    paths, geometry = bridge_paths_from_osm(payloads)
    if not paths:
        parser.error('No complete bridge road/path geometries in the input directory')
    database = resolve_project_path(Path.cwd(), config['paths']['database'])
    report = dict(version=VERSION, scope='Offline prototype, not live terrain selection or physiological validation',
        physiological_predictions_evaluated=False,
        geometry=geometry, assumptions=dict(match_radius_m=MATCH_RADIUS_M,
            minimum_matched_points=MINIMUM_MATCHED_POINTS, minimum_following_span_m=MINIMUM_MATCHED_SPAN_M,
            endpoint_heights='Ground-map samples at connected bridge-path endpoints, not measured deck heights',
            interpolation='Linear in distance along bridge geometry, not time or traversal endpoints',
            trusted_recording='Existing supported barometer assessment, no new athlete-tuned thresholds',
            guarded_fallback='Bridge-level ambiguity is missing grade evidence; never observed flatness',
            smoothing='Existing distance window applied independently to contiguous valid chunks'), runs=[])
    with closing(sqlite3.connect(database.resolve().as_uri()+'?mode=ro', uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        with invariant_snapshot(connection):
            directory = tile_directory(connection, config)
            lookup = lambda lat, lon: map_elevation(lat, lon, directory)
            for identifier in args.activity_id:
                activity = connection.execute('SELECT * FROM activities WHERE id=?', (identifier,)).fetchone()
                if activity is None:
                    parser.error(f'Activity {identifier} is unavailable')
                points = _load_points(connection, identifier)
                creator = activity['creator']
                if identifier in originals:
                    parsed = parse_fit(originals[identifier], config['timezone_default']).activities
                    if len(parsed) != 1:
                        parser.error('Original FIT must contain exactly one activity')
                    # Stored historical exports can round start timestamps to
                    # minutes. Reject a wrong-date override rather than splice.
                    if abs((parsed[0].start_time_utc-points[0].timestamp_utc).total_seconds()) > 60:
                        parser.error('Original FIT does not match the selected activity time')
                    points, creator = parsed[0].trackpoints, parsed[0].creator
                mapped = [lookup(p.latitude, p.longitude) if p.gps_valid else None for p in points]
                prototype = prototype_bridge_profile(points, mapped, paths, lookup, creator=creator)
                movement = classify_movement(points, config['moving_time'])
                mask = prototype['bridge_mask']
                metrics = {}
                profiles = dict(recorded=[p.altitude_m for p in points], ground_map=mapped,
                    endpoint_estimate=prototype['endpoint_estimate'], guarded=prototype['guarded_elevations'])
                for name, heights in profiles.items():
                    attach_prototype_deltas(points, movement.intervals, heights,
                        float(config['elevation']['smoothing_window_meters']))
                    bridge_intervals = [i for i in movement.intervals if mask[i.index] and mask[i.index+1]]
                    known_bridge = [i for i in bridge_intervals if i.elevation_delta_m is not None]
                    values = [v for v, matched in zip(heights, mask) if matched and v is not None]
                    metrics[name] = dict(
                        matched_height_range_m=[min(values), max(values)] if values else None,
                        bridge_ascent_m=sum(max(0., i.elevation_delta_m) for i in known_bridge) if known_bridge else None,
                        bridge_descent_m=sum(max(0., -i.elevation_delta_m) for i in known_bridge) if known_bridge else None,
                        bridge_known_grade_m=sum(i.distance_m for i in known_bridge),
                        total_known_grade_m=sum(i.distance_m for i in movement.intervals if i.elevation_delta_m is not None))
                report['runs'].append(dict(activity_id=identifier, date=activity['start_time_local'],
                    creator=creator, native_fit_override=identifier in originals,
                    elevation_only=identifier in args.elevation_only,
                    points=len(points), gps_points=sum(p.gps_valid for p in points),
                    bridge_points=sum(mask), estimated_points=sum(prototype['estimated_mask']),
                    recorded_quality=prototype['recorded_quality'],
                    bridges=prototype['bridges'], metrics=metrics,
                    limitations=prototype['limitations']))
            report['database_writes'] = connection.total_changes
            assert report['database_writes'] == 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    private_file(args.output)
    for run in report['runs']:
        print(json.dumps({key: run[key] for key in ('activity_id', 'date', 'bridge_points', 'recorded_quality', 'metrics')}, indent=2))
    print('Report:', args.output)


if __name__ == '__main__':
    main()
