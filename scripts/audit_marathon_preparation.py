"""Read-only marathon scenarios from one independently checked issued snapshot.

No service resume, search, settings save, stage acceptance or live deployment.
An omitted race date uses a disclosed comparison horizon, NOT an athlete goal.
"""
from argparse import ArgumentParser
from dataclasses import asdict
from datetime import date, datetime, timedelta
import json
from pathlib import Path
import sqlite3
from zoneinfo import ZoneInfo

from run_analysis.adaptive.decision_snapshot import restore
from run_analysis.race_preparation import marathon_preparation_options
from run_analysis.staged_progression import DevelopmentStage, ProgressionPolicy


def observed_reference(history, *, as_of, timezone):
    zone=ZoneInfo(timezone)
    today=as_of.astimezone(zone).date()
    # Same local-date30-day reference and prior-day boundary as program guards.
    prior=[r for r in history if r.source=='observed' and r.available_at<=as_of and r.end<=as_of
           and today-timedelta(days=30)<=r.start.astimezone(zone).date()<today]
    longest=max(prior,key=lambda r:r.miles) if prior else None
    return dict(longest_prior_30d_miles=longest.miles if longest else None,
                reference_key=longest.key if longest else None,
                reference_start=longest.start if longest else None,
                prior_observed_count=len(prior),timezone=timezone)


def audit(runtime_root, *, race_date=None):
    db_path=runtime_root/'sandbox.sqlite'
    with sqlite3.connect(db_path.resolve().as_uri()+'?mode=ro',uri=True) as db:
        db.execute('PRAGMA query_only=ON')
        db.execute('BEGIN')
        row=db.execute('SELECT event,snapshot FROM sandbox_decisions ORDER BY sequence DESC LIMIT 1').fetchone()
        if row is None:
            raise ValueError('No issued decision available')
        event=json.loads(row[0])
        snapshot=restore(row[1],expected_fingerprint=event['snapshot_fingerprint'])
        if db.total_changes:
            raise AssertionError('Read-only audit attempted a write')
    value=dict(event['stage_runtime']['stage'])
    value['issued_at']=datetime.fromisoformat(value['issued_at'])
    value['policy']=ProgressionPolicy(**value['policy'])
    stage=DevelopmentStage(**value)
    if stage.key!=snapshot.stage_key or stage.issued_at!=snapshot.stage_issued_at:
        raise ValueError('Runtime stage and independently verified snapshot differ')
    as_of=snapshot.as_of.astimezone(ZoneInfo(snapshot.timezone))
    reference=observed_reference(snapshot.history,as_of=as_of,timezone=snapshot.timezone)
    # Date is only needed to compare windows. Never infer a chosen race from an
    # inactive legacy setting or general-fitness17-mile destination.
    comparison=race_date or as_of.date()+timedelta(days=366)
    result=marathon_preparation_options(stage,race_date=comparison,as_of=as_of,
        longest_prior_30d_miles=reference['longest_prior_30d_miles'],response_state='unknown')
    return dict(runtime=str(runtime_root.resolve()),snapshot_fingerprint=snapshot.fingerprint,
        stage=stage,as_of=as_of,reference=reference,race_date_is_illustrative=race_date is None,
        scenarios=result,acceptance_scope='Numerical preparation scenarios from frozen real dose/distance evidence only',
        limitations=['Response is unresolved for this audit; no inferred recovery clearance',
            'No activity/calendar simulation, complete-program feasibility, dose optimality or missed-run resilience proved',
            'No race goal or progression policy saved; current issued calendar and general-fitness settings unchanged'])


def serialize(value):
    if hasattr(value,'__dataclass_fields__'):
        return asdict(value)
    if isinstance(value,(datetime,date)):
        return value.isoformat()
    raise TypeError(type(value).__name__)


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('runtime_root',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--race-date',type=date.fromisoformat)
    args=parser.parse_args()
    report=audit(args.runtime_root,race_date=args.race_date)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report,stream,default=serialize,allow_nan=False,indent=2)
    print(json.dumps(dict(as_of=report['as_of'],reference=report['reference'],
        race_date_is_illustrative=report['race_date_is_illustrative'],
        options=[{k:o[k] for k in ('mode','target','earliest_conditional_date',
            'recommended_conditional_date','volume_growth_per_review','review_days')}
            for o in report['scenarios']['options']]),default=serialize,indent=2))


if __name__=='__main__':
    main()
