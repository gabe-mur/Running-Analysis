"""Stage live evidence through the exact adaptive service without live activation."""
from argparse import ArgumentParser
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3

from run_analysis.config import load_config, resolve_project_path
from run_analysis.adaptive.live_bridge import cold_adopt
from run_analysis.adaptive.web_materialization import render
from run_analysis.run_feedback import list_runs
from scripts.research_manifest import freeze_sources


def stage(root):
    config=load_config()
    database=resolve_project_path(Path.cwd(),config['paths']['database'])
    at=datetime.now(timezone.utc)
    manifest=freeze_sources(root.with_suffix('.sources.zip'),config=config,
        entrypoint=__name__,inputs=(database,))
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True) as db:
        db.row_factory=sqlite3.Row
        service, report=cold_adopt(root,db,config,database,at)
        try:
            result=dict(report,manifest=manifest,as_of=at,live_server_changed=False)
            with (root/'live-adoption-summary.json').open('x') as stream:
                json.dump(result,stream,indent=2,allow_nan=False,
                    default=lambda value:asdict(value) if is_dataclass(value) else value.isoformat())
            schedule=render(service.harness,service.latest,config,list_runs(db,limit=5000))
            with (root/'web-schedule.json').open('x') as stream:
                stream.write(schedule.model_dump_json(indent=2))
            print(json.dumps({k:report[k] for k in ('engine','planner_version','comparator_seconds',
                'adoption_seconds','reference_fatigue','reference_days','proposal_status')}) ,flush=True)
        finally:
            service.harness.sandbox.close()


if __name__=='__main__':
    parser=ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    stage(parser.parse_args().output)
