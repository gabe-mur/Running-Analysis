"""Freeze a chronological development comparison before any predictor upgrade.

Reuses the complete original-file inventory and existing predictors. The
conditional comparison supplies realized timing; it does not invent historical
prescriptions or claim a prospective validation. No search or live publication.
"""
from argparse import ArgumentParser
from datetime import datetime
import json
from pathlib import Path
import sqlite3

from run_analysis.config import load_config, resolve_project_path
from scripts.audit_direct_quality_load import DirectAudit, summary
from scripts.audit_work_stimulus import encode
from scripts.research_manifest import freeze_sources
from scripts.validate_quality_load import audit


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--as-of', required=True, type=datetime.fromisoformat)
    parser.add_argument('--database', type=Path, help='Optional previously frozen evidence database')
    args = parser.parse_args()
    if args.as_of.tzinfo is None:
        parser.error('Timezone-aware cutoff required')
    config = load_config()
    source = args.database or resolve_project_path(Path.cwd(), config['paths']['database'])
    args.output.mkdir(parents=True, exist_ok=False)
    database = args.output/'evidence.sqlite'
    with sqlite3.connect(source.resolve().as_uri()+'?mode=ro', uri=True) as db:
        with sqlite3.connect(database) as copy:
            db.backup(copy)
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro', uri=True) as db:
        originals = tuple(Path(row[0]) for row in db.execute('''SELECT DISTINCT s.path
            FROM source_files s JOIN activity_sources a ON a.source_file_id=s.id
            WHERE a.is_primary=1''') if Path(row[0]).is_file())
    protocol = dict(version='quality_prediction_comparison_v2', cutoff=args.as_of.isoformat(),
        history_days=30, origin='Each held-out run start; target added only after forecasting/scoring',
        candidates=('current direct phase load', 'simple recent-session load rate',
            'single shared intent/sequence upgrade, to be specified before fitting'),
        controls=('unconditioned direct phase load', 'previous HR-curve predictor'),
        required_outputs=('inventory and unresolved cases', 'coverage per predictor',
            'whole/work/recovery/cooldown load error and signed bias',
            'known-HR coverage', 'source availability and intent status',
            'pre-run forecast separately from realized-timing diagnostic'),
        acceptance=('Do not lose baseline supported cases without explicit attribution',
            'Added complexity must improve paired total-load MAE over current direct and simple rate baselines',
            'Report structure/category failures and phase bias rather than accepting aggregate gain alone',
            'Do not tune windows/weights/bands per family or remove cold-start/different-session cases',
            'Limited intent data cannot establish all-family pre-run accuracy'),
        limits=('All existing runs are previously inspected development evidence, not pristine holdouts',
            'Historical conditional audit assumes availability at completion and current labels/HR settings',
            'Prior saved numeric intent requires its own actual-issuance/import-availability check',
            'Observed HR is not retroactive proof of requested intensity',
            'Future numerical-prescription uploads provide prospective evaluation',
            'Source extrema are scenarios, not calibrated confidence intervals'))
    # Persist the protocol before the comparison, not after viewing results.
    (args.output/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    manifest = freeze_sources(args.output/'sources.zip', config=config, entrypoint=__name__,
        inputs=(database, args.output/'protocol.json', *originals))
    result = audit(database, config, args.as_of, case_observer=DirectAudit())
    result['direct_load_summary'] = {label: summary(rows) for label, rows in (
        ('all', result['cases']), ('intervals', [c for c in result['cases'] if c['label']=='intervals']),
        ('tempo_threshold', [c for c in result['cases'] if c['label']=='tempo_threshold']))}
    result.update(protocol=protocol, manifest=manifest, scope='Frozen baseline development comparison; no upgraded predictor fitted')
    with (args.output/'baseline.json').open('x') as stream:
        json.dump(result, stream, default=encode, indent=2, allow_nan=False)
    print(json.dumps(dict(summary=result['direct_load_summary'], seconds=result['elapsed_seconds'],
        planner_influence=False), indent=2), flush=True)


if __name__ == '__main__':
    main()
