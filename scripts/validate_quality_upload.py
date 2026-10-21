"""Original FIT round trip and metrics reprocessing on a backed-up sandbox DB."""
from argparse import ArgumentParser
from dataclasses import asdict
from datetime import datetime,timezone
from hashlib import sha256
import json
from pathlib import Path
import sqlite3

from run_analysis.config import load_config
from run_analysis.adaptive.execution_upload_sandbox import ExecutionUploadSandbox
from run_analysis.web.upload_service import UploadPayload
from run_analysis.processing import process_activities
from run_analysis.run_feedback import get_run_feedback
from scripts.research_manifest import freeze_sources


def validate(database,activity_id,output):
    output.mkdir(parents=True,exist_ok=False)
    config=load_config()
    manifest=freeze_sources(output/'sources.zip',config=config,entrypoint=__name__)
    source=sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True)
    source.row_factory=sqlite3.Row
    row=source.execute('''SELECT s.path,s.sha256 FROM source_files s
        JOIN activity_sources a ON a.source_file_id=s.id
        WHERE a.activity_id=? AND a.is_primary=1''',(activity_id,)).fetchone()
    path=Path(row['path']);payload=path.read_bytes()
    assert sha256(payload).hexdigest()==row['sha256']
    clone=sqlite3.connect(output/'app-copy.sqlite');clone.row_factory=sqlite3.Row
    source.backup(clone)
    before=[dict(r) for r in source.execute('SELECT activity_id,detected_workout_type FROM activity_metrics ORDER BY activity_id')]
    external=source.execute('SELECT activity_id FROM activities WHERE id=?',(activity_id,)).fetchone()[0]
    # Only the isolated copy loses its explicit label, proving auto detection.
    clone.execute('UPDATE run_overrides SET workout_type=NULL WHERE activity_id=?',(external,));clone.commit()
    stamp=datetime.now(timezone.utc)
    sandbox=ExecutionUploadSandbox(output/'upload',config,create=True)
    try:
        result=sandbox.upload(UploadPayload(path.name,payload),fingerprint=None,
            available_at=stamp,resolve_evidence=True,long_threshold_miles=7.)
        physical=json.loads(sandbox.db.execute('SELECT accounted FROM sandbox_completions').fetchone()[0])
        observed=result['comparison']['observed_evidence']
        assert observed['role_source']=='recorded_progression_laps'
        assert observed['workout_type']=='tempo_threshold'
        assert observed['aligned_work'] and not observed['prescription_used_as_observation']
        assert physical['role']=='quality' and len(physical['bouts'])==1
        processing=process_activities(clone,config,activity_ids=(activity_id,))
        feedback=get_run_feedback(clone,config,activity_id)
        assert feedback.metadata.workout_type.value=='tempo_threshold'
        after=[dict(r) for r in clone.execute('SELECT activity_id,detected_workout_type FROM activity_metrics ORDER BY activity_id')]
        changed=[b['activity_id'] for b,a in zip(before,after) if b!=a]
        assert changed==[activity_id]
        for table in ('activity_metrics','segments'):
            query=f'SELECT * FROM {table} WHERE activity_id!=? ORDER BY activity_id'
            assert [tuple(r) for r in source.execute(query,(activity_id,))]==[
                tuple(r) for r in clone.execute(query,(activity_id,))]
        report=dict(at=stamp.isoformat(),activity_id=activity_id,source_path=str(path),
            source_sha256=row['sha256'],manifest=manifest,upload=result,physical=physical,
            processing=asdict(processing),changed_automatic_roles=changed,
            feedback=feedback.model_dump(mode='json'),live_database_unchanged=True)
        with (output/'report.json').open('x') as f:
            json.dump(report,f,indent=2,allow_nan=False)
        print(json.dumps(dict(automatic_evidence=observed,physical_bouts=physical['bouts'],
            changed_roles=changed,processing=asdict(processing),
            workout_analysis=feedback.workout_analysis.model_dump(mode='json')),indent=2))
    finally:
        sandbox.close();clone.close();source.close()


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('--database',type=Path,default=Path('data/run_analysis.sqlite'))
    p.add_argument('--activity-id',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();validate(args.database,args.activity_id,args.output)
