"""Read-only corpus audit of recorded workout detection, never a live replan."""
from argparse import ArgumentParser
from datetime import datetime,timezone
from dataclasses import asdict
import json
from pathlib import Path
import sqlite3
from time import perf_counter

from run_analysis.config import load_config
from run_analysis.movement import classify_movement
from run_analysis.processing import _load_points
from run_analysis.workout_detection import detect_structured_workout,recorded_laps
from scripts.research_manifest import freeze_sources


def audit(database,output):
    output.mkdir(parents=True,exist_ok=False)
    config=load_config();stamp=perf_counter()
    manifest=freeze_sources(output/'sources.zip',config=config,entrypoint=__name__)
    rows=[];changes=[]
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True) as db:
        db.row_factory=sqlite3.Row
        for activity in db.execute('''SELECT a.id,a.start_time_local,m.detected_workout_type,m.workout_detection_source,
            o.workout_type AS manual_type FROM activities a LEFT JOIN activity_metrics m ON a.id=m.activity_id
            LEFT JOIN run_overrides o ON o.activity_id=a.activity_id WHERE a.sport='Running'
            ORDER BY a.start_time_utc_epoch''').fetchall():
            intervals=classify_movement(_load_points(db,activity['id']),config['moving_time']).intervals
            result=detect_structured_workout(db,activity['id'],intervals,z3_floor=config['zones']['z3'][0])
            detected=result.workout_type.value if result else None
            row=dict(activity_id=activity['id'],date=activity['start_time_local'],
                old_type=activity['detected_workout_type'],new_type=detected,
                old_source=activity['workout_detection_source'],new_source=result.source if result else None,
                manual_type=activity['manual_type'])
            rows.append(row)
            if detected!=activity['detected_workout_type']:
                changes.append(dict(row,laps=[dict(r) for r in recorded_laps(db,activity['id'])]))
        # Logical identity for this read transaction, not a hash of a live WAL.
        digest_query='SELECT id,activity_uid,start_time_utc,total_distance_m,updated_at_utc FROM activities ORDER BY id'
        logical_identity=[tuple(r) for r in db.execute(digest_query)]
    from hashlib import sha256
    report=dict(at=datetime.now(timezone.utc).isoformat(),database=str(database),
        activity_identity_sha256=sha256(json.dumps(logical_identity,sort_keys=True).encode()).hexdigest(),
        manifest=manifest,activities=len(rows),changes=changes,
        before_detected=sum(r['old_type'] is not None for r in rows),
        after_detected=sum(r['new_type'] is not None for r in rows),
        seconds=perf_counter()-stamp,rows=rows,read_only=True,live_replan=False)
    with (output/'report.json').open('x') as stream:
        json.dump(report,stream,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in report.items() if k not in {'rows','changes','manifest'}},indent=2))
    print(json.dumps([dict(r,laps=[{k:lap[k] for k in ('lap_index','total_time_s','distance_m','average_hr_bpm','trigger_method','intensity')} for lap in r['laps']]) for r in changes],indent=2))


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('--database',type=Path,default=Path('data/run_analysis.sqlite'))
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    audit(args.database,args.output)
