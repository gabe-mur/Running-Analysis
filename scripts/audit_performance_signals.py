"""Read-only performance curves and additional FIT measurement inventory.

Reuse the exact continuous-duration search and empirical frontier prototype.
Neither raw nor grade-adjusted training bests prove maximal effort or VO2max.
"""
from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3
from time import perf_counter

from run_analysis.config import load_config
from run_analysis.elevation_quality import prepare_terrain, tile_directory
from run_analysis.fit import parse_fit
from run_analysis.movement import classify_movement
from run_analysis.physiology import grade_energy_ratio
from run_analysis.processing import _load_points
from run_analysis.read_cache import invariant_snapshot
from run_analysis.signal_structure import SignalSpan, speed_duration_curve, describe_signals
from scripts.speed_duration_reference import ObservedSpeedCurve, WINDOWS, freeze_frontier


def terrain_speed_curve(intervals, maximum_grade):
    """Unknown grade breaks continuity; no inferred flat ground or clipping."""
    spans=[]
    origin=intervals[0].start.timestamp_utc if intervals else None
    for i in intervals:
        if (i.elevation_delta_m is None or i.distance_m<=0 or i.distance_source=='unknown'
                or i.moving_time_s<=0 or abs(i.elapsed_s-i.moving_time_s)>1e-8
                or {'large_recording_gap','implausible_distance_speed','distance_reset'} & set(i.flags)):
            continue
        grade=100*i.elevation_delta_m/i.distance_m
        if abs(grade)>maximum_grade:
            continue
        start=(i.start.timestamp_utc-origin).total_seconds()
        end=(i.end.timestamp_utc-origin).total_seconds()
        speed=i.distance_m/i.elapsed_s*grade_energy_ratio(grade,maximum_grade)
        spans.append(SignalSpan(start,end,speed,speed))
    return speed_duration_curve(spans,WINDOWS)


def audit(database,config,as_of):
    started=perf_counter(); raw=[]; adjusted=[]; curves=[]; inventory=[]
    with sqlite3.connect(f'{database.resolve().as_uri()}?mode=ro',uri=True) as c:
        c.row_factory=sqlite3.Row
        with invariant_snapshot(c):
            activities=c.execute('''SELECT a.id,a.start_time_utc,a.start_time_local,a.created_at_utc,a.creator,
                COALESCE(o.workout_type,m.detected_workout_type,'unknown') role
                FROM activities a LEFT JOIN activity_metrics m ON m.activity_id=a.id
                LEFT JOIN run_overrides o ON o.activity_id=a.activity_id
                WHERE a.sport='Running' AND a.start_time_utc_epoch>=? AND a.start_time_utc_epoch<=?
                ORDER BY a.start_time_utc_epoch''',((as_of-timedelta(days=365)).timestamp(),as_of.timestamp())).fetchall()
            directory=tile_directory(c,config)
            for a in activities:
                points=_load_points(c,a['id'])
                if len(points)<2 or points[0].timestamp_utc is None or points[-1].timestamp_utc is None:
                    continue
                intervals=classify_movement(points,config['moving_time']).intervals
                if not intervals:
                    continue
                measured=describe_signals(intervals,hr_thresholds_bpm=(),windows_seconds=WINDOWS)
                terrain=prepare_terrain(points,intervals,config,a['creator'],directory)
                grade=terrain_speed_curve(intervals,float(config['elevation']['maximum_plausible_grade_percent']))
                start=datetime.fromisoformat(a['start_time_utc']);end=points[-1].timestamp_utc
                available=max(end,datetime.fromisoformat(a['created_at_utc']))
                for series,rows in ((raw,measured['speed_duration_curve']),(adjusted,grade)):
                    series.append(ObservedSpeedCurve(str(a['id']),start,end,available,
                        tuple((p['window_seconds'],p['peak_mean_mps']) for p in rows
                              if p['peak_mean_mps'] is not None and p['peak_mean_mps']>0)))
                curves.append(dict(activity_id=a['id'],date=a['start_time_local'][:10],role=a['role'],
                    available_at=available.isoformat(),raw=measured['speed_duration_curve'],grade_adjusted=grade,
                    terrain_source=terrain['source'],terrain_status=terrain['status'],
                    maximal_effort_established=False))
            sources=c.execute('''SELECT a.id,a.start_time_local,s.path,s.sha256 FROM activity_sources l
                JOIN activities a ON a.id=l.activity_id JOIN source_files s ON s.id=l.source_file_id
                WHERE lower(s.path) LIKE '%.fit' OR lower(s.path) LIKE '%.fit.gz'
                ORDER BY a.start_time_utc,s.id DESC''').fetchall()
            seen=set()
            for row in sources:
                if row['id'] in seen:
                    continue
                seen.add(row['id'])
                activity=parse_fit(Path(row['path'])).activities[0];stream=activity.sensor_stream
                counts=Counter(k for r in stream.performance_records for k in r['fields'])
                units={k:f['units'] for r in stream.performance_records for k,f in r['fields'].items()}
                inventory.append(dict(activity_id=row['id'],date=row['start_time_local'][:10],source_sha256=row['sha256'],
                    record_counts=dict(counts),units=units,session_fields=stream.performance_session_fields,
                    records=stream.performance_records,source_measurements_not_fitness_predictions=True))
            assert c.total_changes==0
    snapshots=[]
    for days in (42.,84.,180.):
        snapshots.append(dict(lookback_days=days,raw=freeze_frontier(raw,as_of,lookback_days=days),
            grade_adjusted=freeze_frontier(adjusted,as_of,lookback_days=days)))
    return dict(version='additional-performance-signals-v1',as_of=as_of.isoformat(),
        curves=curves,snapshots=snapshots,fit_measurements=inventory,
        source_database_read_only=True,planner_influence=False,fitness_influence=False,
        elapsed_seconds=perf_counter()-started,
        assumptions=['Existing 3/7/12 minute continuous-duration search, not concatenated repetitions.',
            'Grade correction reuses selected local terrain and existing Minetti cost model; wind and heat remain uncorrected.',
            'Available dates use first physical import, not a reconstructed source-revision history; no prospective accuracy claim.',
            'Training bests need not be maximal; leave-one-run sensitivity is not a confidence interval.',
            'Power/dynamics are preserved with decoder units, not averaged into independent fitness votes.'])


def main():
    parser=ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Choose a new output path')
    result=audit(Path('data/run_analysis.sqlite'),load_config('config.yaml'),datetime.now(timezone.utc))
    from run_analysis.privacy import private_file
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as out:
        json.dump(result,out,default=lambda x:x.isoformat(),indent=2,allow_nan=False)
    private_file(args.output)
    print(json.dumps(dict(curves=len(result['curves']),snapshots=[dict(days=s['lookback_days'],
        **{k:dict(slope_mps=s[k]['fit']['slope_mps'],anchors=s[k]['fit']['anchors'],
            deletion_sensitivity_mps=s[k]['deletion_sensitivity_mps']) for k in ('raw','grade_adjusted')}) for s in result['snapshots']],
        additional_fit_runs=[dict(id=r['activity_id'],date=r['date'],fields=r['record_counts'],units=r['units'])
            for r in result['fit_measurements'] if r['record_counts']],seconds=result['elapsed_seconds']),indent=2))


if __name__=='__main__':main()
