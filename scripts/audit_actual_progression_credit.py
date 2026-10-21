"""Read-only comparison of archived nominal credit and measured-mile credit.

No search, import, stage acceptance, publication or application database writes.
"""
from argparse import ArgumentParser
from datetime import timedelta
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from scripts.replay_adaptive_search import load_decision
from run_analysis.adaptive.adaptive_stage_lifecycle import review_entries
from run_analysis.adaptive.execution_upload_sandbox import ExecutionUploadSandbox
from run_analysis.adaptive.stage_acceptance import read_stage
from run_analysis.adaptive.review_seed import read_entry
from run_analysis.stage_target_history import target_miles_between
from run_analysis.staged_progression import review_recent_delivery


def audit(root, index):
    root=Path(root).resolve()
    case=load_decision(root,index)  # Exact archived winner must still reproduce.
    state=case['state']
    scenario=json.loads((root/'scenario.json').read_text())
    # Bind only read methods; never initialize/migrate the actual ledger.
    with sqlite3.connect((root/'sandbox.sqlite').as_uri()+'?mode=ro',uri=True) as db:
        db.row_factory=sqlite3.Row
        sandbox=ExecutionUploadSandbox.__new__(ExecutionUploadSandbox)
        sandbox.db=db
        event=case['event']
        stage=read_stage(json.dumps(event['stage_runtime']['stage']))
        harness=SimpleNamespace(sandbox=sandbox,zone=ZoneInfo(state.timezone),
            events=[json.loads(p.read_text()) for p in sorted(root.glob('decision-*.json'))
                if int(p.stem.split('-')[-1])<=index],catalog=case['catalog'],
            seed_review_entries=tuple(read_entry(e) for e in scenario.get('seed_review_entries',())))
        entries=review_entries(harness,state.as_of)
        since=max(stage.issued_at,state.as_of-timedelta(days=stage.policy.review_days))
        required=target_miles_between(state.context.volume_targets,since,state.as_of,harness.zone)
        rows=[]
        for basis in ('issued_nominal_v1','actual_distance_v2'):
            result=review_recent_delivery(stage,entries,state.as_of,ledger_complete=True,
                longest_prior_30d_miles=max((r.miles for r in state.history
                    if state.as_of-timedelta(days=30)<=r.start<state.as_of),default=None),
                required_miles=required,distance_credit=basis)
            rows.append(dict(basis=basis,status=result['status'],
                credited_miles=result.get('credited_miles'),required_miles=result.get('required_miles'),
                volume_accounting=result.get('volume_accounting')))
    return dict(decision_file=index,as_of=state.as_of.isoformat(),
        archived_cost_reproduced=float(case['archived_cost']),reviews=rows,
        searches=0,publication=False,raw_history_unchanged=True)


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('index',type=int,help='Archived filename number, not event index')
    args=parser.parse_args()
    print(json.dumps(audit(args.root,args.index),indent=2,allow_nan=False))


if __name__=='__main__':
    main()
