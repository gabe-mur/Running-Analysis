"""Short original-HR-backed upload/rollover probe, not long acceptance clearance."""
from argparse import ArgumentParser
from datetime import datetime,timedelta,time
import json
from pathlib import Path
from zoneinfo import ZoneInfo

from run_analysis.config import load_config
from scripts.new_model_daily_harness import DailyHarness
from scripts.recorded_trace_execution import donor,transformed_payload
from scripts.research_manifest import freeze_sources


if __name__=='__main__':
    p=ArgumentParser()
    p.add_argument('root',type=Path)
    p.add_argument('--database',type=Path,default=Path('data/run_analysis.sqlite'))
    a=p.parse_args()
    config=load_config()
    manifest=freeze_sources(a.root/'execution-probe.sources.zip',config=config,
        entrypoint='scripts.audit_candidate_execution',inputs=(a.database,))
    first=json.loads((a.root/'decision-0001.json').read_text())
    origin=datetime.fromisoformat(first['as_of'])
    harness=DailyHarness.resume(a.root,config,as_of=origin+timedelta(hours=1))
    cases=[]
    try:
        same=harness.replan(origin+timedelta(hours=1),reason='same_day_no_new_evidence')
        cases.append({k:same[k] for k in ('as_of','reason','cost','prior_cost','churn','seconds')})
        day=origin.astimezone(harness.zone).date()
        chosen=harness.active.get(day.isoformat())
        upload=None
        if chosen:
            dose=harness.catalog[chosen['dose']]
            source=max((r for r in harness.history if r.role==dose.role and r.available_at<=origin),key=lambda r:(r.start,r.key))
            original=donor(a.database,source,config,harness.ref)
            payload,phases,end,provenance=transformed_payload(original,dose,datetime.combine(day,time(16),harness.zone))
            outcome=harness.upload(payload,day=day.isoformat(),available_at=end,observed_role=dose.role,observed_phases=phases)
            upload=dict(outcome=outcome,provenance=provenance)
            after=harness.replan(end+timedelta(seconds=1),reason='post_upload_prescribed_dose_prior_recorded_HR')
            cases.append({k:after[k] for k in ('as_of','reason','cost','prior_cost','churn','seconds')})
        rolled=harness.replan(origin+timedelta(days=1),reason='daily_rollover_after_compliance')
        cases.append({k:rolled[k] for k in ('as_of','reason','cost','prior_cost','churn','seconds')})
        report=dict(cases=cases,upload=upload,manifest=manifest,
            caveat='Prescribed distance/phase adherence with transformed prior measured HR, not exact predicted physiology or automatic workout classification')
        with (a.root/'execution-probe-summary.json').open('x') as stream:
            json.dump(report,stream,default=lambda v:v.isoformat(),indent=2)
        print(json.dumps(report,default=lambda v:v.isoformat()))
    finally:
        harness.sandbox.close()
