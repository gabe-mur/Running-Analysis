"""Bounded same-day/next-day refresh probe on an existing offline build."""
from argparse import ArgumentParser
from datetime import datetime,timedelta
import json
from pathlib import Path

from run_analysis.config import load_config
from scripts.new_model_daily_harness import DailyHarness
from scripts.research_manifest import freeze_sources


if __name__=='__main__':
    p=ArgumentParser()
    p.add_argument('root',type=Path)
    a=p.parse_args()
    config=load_config()
    manifest=freeze_sources(a.root/'rollover.sources.zip',config=config,entrypoint='scripts.audit_candidate_rollover')
    first=json.loads((a.root/'decision-0001.json').read_text())
    origin=datetime.fromisoformat(first['as_of'])
    harness=DailyHarness.resume(a.root,config,as_of=origin+timedelta(hours=1))
    cases=[]
    try:
        for elapsed,reason in ((timedelta(hours=1),'same_day_no_new_evidence'),(timedelta(days=1),'rollover_no_new_evidence')):
            event=harness.replan(origin+elapsed,reason=reason)
            case={k:event[k] for k in ('as_of','reason','seconds','cost','prior_cost','churn','program','volume','evaluations','budget_exhausted')}
            cases.append(case)
            print(json.dumps(case),flush=True)
        with (a.root/'rollover-summary.json').open('x') as stream:
            json.dump(dict(cases=cases,manifest=manifest,not_full_daily_adherence_test=True),stream,indent=2)
    finally:
        harness.sandbox.close()
