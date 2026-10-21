"""Enable/check the future live reviewer without resetting or searching a phase.

The real bridge owns activation. Preserve backups and verify all journal rows,
the accepted stage and current publication before reporting a no-search release.
"""
from argparse import ArgumentParser
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sqlite3

from run_analysis.config import load_config, resolve_project_path
from run_analysis.db import connect
from run_analysis.adaptive.live_bridge import schedule
from run_analysis.adaptive.factory import resume_service
from scripts.replay_adaptive_search import load_decision
from scripts.research_manifest import freeze_sources


def journal_rows(root):
    with sqlite3.connect(root.resolve().as_uri()+'/sandbox.sqlite?mode=ro', uri=True) as db:
        return {table:db.execute(f'SELECT * FROM {table} ORDER BY rowid').fetchall()
            for table in ('sandbox_decisions','sandbox_issues','sandbox_completions','stages','current_stage')}


def run(output):
    config=load_config()
    database=resolve_project_path(Path.cwd(),config['paths']['database'])
    output.mkdir(exist_ok=False)
    at=datetime.now(timezone.utc)
    with connect(database) as db:
        pointer=json.loads(db.execute("SELECT value_json FROM app_state WHERE key='adaptive_runtime'").fetchone()[0])
        root=Path(pointer['root'])
        before=json.loads(db.execute("SELECT value_json FROM app_state WHERE key='weekly_schedule'").fetchone()[0])
        rows=journal_rows(root)
        shutil.copy2(root/'scenario.json',output/'scenario-before.json')
        with sqlite3.connect(output/'app-before.sqlite') as backup:
            db.backup(backup)
        manifest=freeze_sources(output/'sources.zip',config=config,entrypoint=__name__,inputs=(root/'scenario.json',))
        result=schedule(db,config,Path.cwd(),as_of=at)
        if result.planner_status!='ready' or result.planner_version!=204:
            raise ValueError('General progression release is not ready')
        assert journal_rows(root)==rows, 'Activation unexpectedly appended/changed a decision or physical record'
        assert result.generated_at.isoformat()==datetime.fromisoformat(before['generated_at'].replace('Z','+00:00')).isoformat()
        for index in range(1,len(rows['sandbox_decisions'])+1):
            load_decision(root,index)
        service=resume_service(root,config,at)
        try:
            h=service.harness
            assert h.policy.general_progression=='easy_first_coordinated_v2'
            assert h.stage.policy.review_days==14 and h.stage.policy.growth_fraction==.05
            report=dict(status='live_future_progression_enabled_without_new_search',manifest=manifest,
                runtime=str(root),planner_version=result.planner_version,generated_at=result.generated_at.isoformat(),
                verified_archived_decisions=len(rows['sandbox_decisions']),journal_rows_unchanged=True,
                current_phase_unchanged=True,search_started=False,
                phase_volume_start=h.stage.weekly_equivalent_miles,phase_volume_goal=h.stage.policy.volume_goal_miles,
                long_target=h.stage.long_miles,long_destination=h.stage.goal_long_miles,
                review_days=h.stage.policy.review_days,growth_fraction=h.stage.policy.growth_fraction,
                acceptance_scope='Phase seam and live wiring; not multi-phase/90-day stability or validated growth rate')
        finally:
            service.harness.sandbox.close()
    with (output/'report.json').open('x') as stream:
        json.dump(report,stream,indent=2)
    print(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
