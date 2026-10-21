"""Back up and attach a verified real-data runtime; activation flag is separate.

Never restore a whole application DB to roll back: that would erase new uploads.
Disabling the internal engine selector is sufficient to return to the legacy
route while preserving the adaptive journal and all original activity data.
"""
from argparse import ArgumentParser
from datetime import datetime,timezone
from pathlib import Path
import json
import shutil
import sqlite3

from run_analysis.config import load_config,resolve_project_path
from run_analysis.db import connect
from run_analysis.adaptive.live_bridge import attach_runtime
from scripts.replay_adaptive_search import load_decision
from scripts.research_manifest import freeze_sources


def attach(staged,runtime,backup):
    config=load_config()
    database=resolve_project_path(Path.cwd(),config['paths']['database'])
    # Validate before any app mutation; source set is checked again by attach.
    checked=load_decision(staged,1)
    backup.mkdir(parents=True,exist_ok=False)
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True) as source,sqlite3.connect(backup/'app-before.sqlite') as dest:
        source.backup(dest)
    shutil.copy2('config.local.yaml',backup/'config.local.before.yaml')
    manifest=freeze_sources(backup/'release.sources.zip',config=config,
        entrypoint=__name__,inputs=(staged/'live-adoption-summary.json',staged/'decision-0001.json',
            Path('output/frequency-v3-perfect-daily-21-20261003/daily-audit.json'),
            Path('output/live-frequency-v3-route-smoke-b-20261003/report.json')))
    shutil.copytree(staged,runtime)
    with connect(database) as db:
        before=db.execute('SELECT COUNT(*) FROM activities').fetchone()[0]
        attach_runtime(db,runtime,config,datetime.now(timezone.utc))
        assert db.execute('SELECT COUNT(*) FROM activities').fetchone()[0]==before
    result=dict(runtime=str(runtime.resolve()),backup=str(backup.resolve()),
        archived_cost=checked['archived_cost'],original_activity_count=before,
        replay_completions_imported=False,engine_flag_changed=False,manifest=manifest,
        rollback='Remove adaptive_planner.engine from config.local.yaml. Preserve all runtime and activity data; do not restore app-before.sqlite over new uploads.')
    with (backup/'attachment.json').open('x') as stream:
        json.dump(result,stream,indent=2)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    parser=ArgumentParser()
    parser.add_argument('--staged',type=Path,required=True)
    parser.add_argument('--runtime',type=Path,required=True)
    parser.add_argument('--backup',type=Path,required=True)
    args=parser.parse_args()
    attach(args.staged,args.runtime,args.backup)
