"""Bounded original-file signal audit; read-only source DB, no planner search.

Uses the app's existing session resolution, feedback precedence and interval
analysis. Current metadata is not a reconstruction of what was known historically.
"""
from argparse import ArgumentParser
from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from time import perf_counter

from run_analysis.config import load_config
from run_analysis.db import initialize
from run_analysis.fit import parse_fit
from run_analysis.importer import parser_for
from run_analysis.response_model import Policy
from run_analysis.run_feedback import get_run_feedback
from run_analysis.training_signals import observed_training_signals, training_signals_from_run_feedback
from run_analysis.web.upload_archive import fit_members


# UTC source dates: July 12 evening local is July 13 UTC.
DEFAULT_DATES = ('2026-06-29', '2026-07-13', '2026-09-18',
                 '2026-09-20', '2026-09-22', '2026-09-24')


def audit(database, config, dates=DEFAULT_DATES, fit_zip=None):
    started = perf_counter()
    result = []
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro', uri=True) as source:
        db = sqlite3.connect(':memory:')
        try:
            source.backup(db)
            db.row_factory = sqlite3.Row
            initialize(db)  # Optional migrations only on the disposable copy.
            rows = db.execute('''SELECT a.id,a.start_time_utc,s.path FROM activities a
                JOIN activity_sources l ON l.activity_id=a.id
                JOIN source_files s ON s.id=l.source_file_id
                WHERE l.is_primary=1 AND a.sport='Running'
                ORDER BY a.start_time_utc,a.id''').fetchall()
            keys = set()
            for row in rows:
                if row['start_time_utc'][:10] not in dates:
                    continue
                if row['id'] in keys:
                    raise ValueError('Duplicate primary source would count the run twice')
                keys.add(row['id'])
                path = Path(row['path'])
                at = datetime.fromisoformat(row['start_time_utc'])
                parsed = parser_for(path)(path, default_timezone=config['timezone_default'])
                matches = [a for a in parsed.activities if a.start_time_utc == at]
                if len(matches) != 1:
                    raise ValueError('Original source must match exactly one activity')
                activity = matches[0]
                app = get_run_feedback(db, config, row['id'])
                report = training_signals_from_run_feedback(activity, config, app)
                alternative = training_signals_from_run_feedback(activity, config, app,
                    policy=replace(Policy(), intensity_exponent=1.67))['cardiovascular']
                result.append(dict(activity_id=row['id'], source_name=path.name, signals=report,
                    exponent_1_67_comparator=alternative))
        finally:
            db.close()
    missing = set(dates)-{r['signals']['activity_start'][:10] for r in result}
    if missing:
        raise ValueError(f'Requested UTC dates missing: {sorted(missing)}')
    fit_reports = []
    if fit_zip is not None:
        # Use the same bounded ZIP handling as upload, never archive paths.
        with TemporaryDirectory(prefix='training-signals-') as tmp:
            for index, (name, content) in enumerate(fit_members(fit_zip.read_bytes())):
                path = Path(tmp)/f'activity-{index}.fit'
                path.write_bytes(content)
                for activity in parse_fit(path, config['timezone_default']).activities:
                    peers = [r for r in result if r['signals']['activity_start'] == activity.start_time_utc.isoformat()]
                    if len(peers) != 1:
                        raise ValueError('FIT must match one audited session; no guessed role')
                    report = observed_training_signals(activity, config,
                        workout_type=peers[0]['signals']['workout_type'])
                    fit_reports.append(dict(source_name=name, signals=report,
                        primary_activity_id=peers[0]['activity_id']))
    return dict(version='training_signal_connection_audit_v1', sessions=result,
        additional_fit_sources=fit_reports, elapsed_seconds=perf_counter()-started,
        planner_influence=False, historical_availability_reconstructed=False,
        note='Demand/response inspection, not fatigue calibration or compliance/stability validation')


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('config.yaml'))
    parser.add_argument('--fit-zip', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output path')
    report = audit(args.database, load_config(args.config), fit_zip=args.fit_zip)
    with args.output.open('x') as out:
        json.dump(report, out, indent=2, allow_nan=False)
        out.write('\n')
    print('seconds', round(report['elapsed_seconds'], 3))
    for row in [*report['sessions'], *report['additional_fit_sources']]:
        signal = row['signals']
        c = signal['cardiovascular']
        print(signal['activity_start'], signal['workout_type'],
              'HRR load', c['observed_load'], 'unknown seconds', c['unknown_seconds'],
              'density candidate', c['response_load_candidate'],
              'drift', signal['response']['cardiac_drift'],
              'RPE', signal['response']['feedback'].get('perceived_exertion'))


if __name__ == '__main__':
    main()
