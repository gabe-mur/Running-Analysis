"""Original-FIT equivalence and duplicate import test, never a live import.

Downloads are private local audit artifacts. The existing importer is exercised
against an in-memory copy of the live database; no processing, weather lookup,
model fit, planner evaluation or automatic sync is invoked.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import tempfile

from run_analysis.config import load_config, resolve_project_path
from run_analysis.fit import parse_fit
from run_analysis.importer import import_files
from run_analysis.intervals_connection import ConnectionProblem, download_original_fit
from run_analysis.privacy import private_directory, private_file


def _fingerprints(connection):
    """Canonical observations, derived results and issued plans must not move."""
    tables=('activities','laps','trackpoints','activity_metrics','segments','model_runs',
        'activity_weather','run_overrides','activity_plan_matches','planned_workout_history','app_state')
    result={}
    for table in tables:
        digest=sha256()
        for row in connection.execute(f'SELECT * FROM {table} ORDER BY rowid'):
            digest.update(json.dumps(tuple(row),separators=(',',':'),ensure_ascii=True).encode())
            digest.update(b'\n')
        result[table]=digest.hexdigest()
    return result


def run(root: Path, comparisons):
    config=load_config(root/'config.yaml')
    database=resolve_project_path(root,config['paths']['database']).resolve()
    byte_hash_before=sha256(database.read_bytes()).hexdigest()
    audit_root=Path(tempfile.mkdtemp(prefix='intervals-fit-sync-',dir=private_directory(root/'output')))
    private_directory(audit_root)
    report=dict(scope='original-file retrieval, parsing and sandbox importer only',
        automatic_sync_enabled=False,live_import_performed=False,results=[])
    with sqlite3.connect(f'{database.as_uri()}?mode=ro',uri=True) as live, sqlite3.connect(':memory:') as sandbox:
        live.row_factory=sandbox.row_factory=sqlite3.Row
        live.execute('BEGIN')
        live.backup(sandbox)
        canonical_before=_fingerprints(sandbox)
        count_before=sandbox.execute('SELECT COUNT(*) FROM activities').fetchone()[0]
        for activity_id,original_path in comparisons:
            original_path=Path(original_path).resolve()
            original=original_path.read_bytes()
            fit=download_original_fit(root,activity_id)
            # Safe because download_original_fit validates the identifier.
            target=audit_root/f'{activity_id}.fit'
            target.write_bytes(fit);private_file(target)
            entry=dict(intervals_activity=activity_id,original_filename=original_path.name,
                fit_bytes=len(fit),original_bytes=len(original),byte_identical=fit==original,
                sha256_match=sha256(fit).digest()==sha256(original).digest())
            report['results'].append(entry)
            if not entry['byte_identical']:
                raise ValueError(f'Original FIT comparison failed for {activity_id}; no live import attempted.')
            decoded=parse_fit(target,config['timezone_default'])
            original_decoded=parse_fit(original_path,config['timezone_default'])
            entry['parsed_identical']=decoded==original_decoded
            if not entry['parsed_identical']:
                raise ValueError(f'Parsed FIT comparison failed for {activity_id}.')
            activity=decoded.activities[0];sensor=activity.sensor_stream
            entry.update(trackpoints=len(activity.trackpoints),
                heart_rate_points=sum(p.heart_rate_bpm is not None for p in activity.trackpoints),
                laps=len(activity.laps),timer_events=len(sensor.timer_events),
                rr_batches=len(sensor.rr_batches),
                rr_intervals=sum(len(b.intervals_seconds) for b in sensor.rr_batches),
                session_fields=sensor.session_fields)
            first=import_files(sandbox,audit_root,config['timezone_default'],paths=[target])
            entry['first_import']=asdict(first)
            if first.failed_files or first.activities_added or first.duplicate_activities!=1:
                raise ValueError(f'Sandbox did not recognize {activity_id} as an existing run.')
            second=import_files(sandbox,audit_root,config['timezone_default'],paths=[target])
            entry['repeat_import']=asdict(second)
            if second.failed_files or second.activities_added or second.unchanged_files!=1:
                raise ValueError(f'Repeat import was not idempotent for {activity_id}.')
        report['observed_activity_count_before']=count_before
        report['observed_activity_count_after']=sandbox.execute('SELECT COUNT(*) FROM activities').fetchone()[0]
        report['canonical_and_plan_fingerprints_unchanged']=_fingerprints(sandbox)==canonical_before
        if not report['canonical_and_plan_fingerprints_unchanged']:
            raise ValueError('Sandbox canonical observations or issued plan data changed.')
        report['live_connection_writes']=live.total_changes
    report['live_database_bytes_unchanged']=sha256(database.read_bytes()).hexdigest()==byte_hash_before
    report['full_upload_noop']='Not evaluated by this importer-only audit. Use scripts.audit_duplicate_uploads with the cached downloads to test full orchestration on an isolated disk backup.'
    report_path=audit_root/'summary.json'
    report_path.write_text(json.dumps(report,indent=2,default=str)+'\n');private_file(report_path)
    print(json.dumps(report,indent=2,default=str))
    print(f'Report: {report_path}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compare',nargs=2,action='append',required=True,metavar=('INTERVALS_ID','ORIGINAL_FIT'))
    args=parser.parse_args()
    try:
        run(Path.cwd(),args.compare)
    except (ConnectionProblem,ValueError) as error:
        parser.exit(1,f'{error}\n')


if __name__=='__main__':
    main()
