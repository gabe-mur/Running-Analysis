"""Replay saved numeric assignments through actual sandbox file uploads, not live."""
from argparse import ArgumentParser
from datetime import datetime, timedelta
from hashlib import sha256
import json
from math import isclose
from pathlib import Path
import sqlite3
from time import perf_counter

from run_analysis.config import load_config
from run_analysis.web.upload_service import UploadPayload
from scripts.audit_issued_quality_load import audit
from scripts.direct_quality_load import checked_layout
from scripts.execution_forecast_contract import IntendedExecution, IntendedPhase, TargetBand
from scripts.execution_upload_sandbox import ExecutionUploadSandbox, encode, measurement_context


def run(database, config, output_dir):
    output_dir.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    digest = sha256(database.read_bytes()).hexdigest()
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro', uri=True) as db:
        sources = {str(key): Path(path) for key,path in db.execute('''
            SELECT a.activity_id,s.path FROM activity_sources a JOIN source_files s
            ON s.id=a.source_file_id WHERE a.is_primary=1''')}
    cases = []

    def replay(row, observed_run, observed_layout, trace, history, reference):
        path = sources[row['key']]
        source_digest = sha256(path.read_bytes()).hexdigest()
        root = output_dir/f"case-{row['key']}"
        sandbox = ExecutionUploadSandbox(root, config, create=True)
        try:
            intent = IntendedExecution(row['key'], row['issued_at'], datetime.fromisoformat(row['planned_for']),
                tuple(IntendedPhase(k,TargetBand(b-a,None,None,'saved numeric target; margins not declared'))
                      for k,a,b in row['contract']['layout']),
                None, reference.accounting_key, measurement_context(config), None)
            issued = sandbox.publish(intent,history)
            if issued.payload['target_forecast']['predicted_load'] != row['forecast']['predicted_load']:
                raise AssertionError('Publication changed prior forecast')
        finally:
            sandbox.close()
        # Real persistence boundary, then parser/importer/accounting on source bytes.
        sandbox = ExecutionUploadSandbox(root,config)
        try:
            at = row['actual']['available_at']
            before = sandbox.state(at-timedelta(microseconds=1))
            phases = tuple((k,b-a) for k,a,b in checked_layout(observed_layout)) if observed_layout else None
            payload = UploadPayload(path.name,path.read_bytes())
            result = sandbox.upload(payload,fingerprint=issued.fingerprint,available_at=at,
                                    observed_phases=phases,phase_evidence='inferred')
            after = sandbox.state(at)
            retry = sandbox.upload(payload,fingerprint=issued.fingerprint,available_at=at+timedelta(days=1),
                                   observed_phases=phases,phase_evidence='inferred')
            if len(before['pending']) != 1 or before['actual'] or after['pending'] or len(after['actual']) != 1:
                raise AssertionError('Forecast not replaced exactly once at evidence availability')
            if retry['status'] != 'duplicate' or sandbox.state(at+timedelta(days=1)) != after:
                raise AssertionError('Duplicate upload changed measured state')
            bounds = trace['load_bounds_within_reference']
            if any(not isclose(after['observed_cardio_bounds'][i],bounds[k],abs_tol=1e-9)
                   for i,k in enumerate(('low','high'))):
                raise AssertionError('File upload changed original HR load')
            if not isclose(after['actual'][0]['distance_miles'],observed_run.miles,abs_tol=1e-9):
                raise AssertionError('File upload changed original distance')
            if source_digest != sha256(path.read_bytes()).hexdigest():
                raise AssertionError('Original file changed')
            cases.append(dict(key=row['key'],saved_plan_id=row['plan_id'],source_format=path.suffix,
                fingerprint=issued.fingerprint,comparison=result['comparison'],
                duplicate_status=retry['status'],pending_before=len(before['pending']),
                pending_after=len(after['pending']),actual_count=len(after['actual']),
                observed_cardio_bounds=after['observed_cardio_bounds'],source_unchanged=True,
                legacy_distance_range=row['payload'].get('distance_range_miles')))
        finally:
            sandbox.close()

    inventory = audit(database,config,execution_observer=replay)
    unchanged = digest == sha256(database.read_bytes()).hexdigest()
    if not unchanged:
        raise AssertionError('Original database changed')
    report = dict(version='execution_upload_sandbox_audit_v1',cases=cases,
        inventory=[dict(key=c['key'],status=c.get('forecast_status'),contract=c['contract']['status'])
                   for c in inventory['cases']],database_unchanged=unchanged,
        elapsed_seconds=perf_counter()-started,live_scheduler_influence=False,
        limitations=['Saved assignments are not proof of the viewed version',
            'Historical duration margins, load allowances and new canonical distance targets not invented',
            'Observed phase alignment supplied by existing original-file detector, not copied from prescription',
            'Upload/state integration only; no daily calendar search or stability claim'])
    with (output_dir/'report.json').open('x') as stream:
        stream.write(encode(report))
    return report


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    args = parser.parse_args()
    print(encode(run(Path('data/run_analysis.sqlite'),load_config('config.yaml'),args.output_dir)))
