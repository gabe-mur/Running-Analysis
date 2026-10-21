"""Causal copied-prefix handoff screen; source/live journals remain untouched.

Restore the last real-HR replay decision BEFORE its first phase handoff, then
execute its issued prescription and refresh with the unchanged full search.
No later decisions/completions are retained. This is a seam screen, not a new
21-day pass or evidence of physiological adaptation.
"""
from argparse import ArgumentParser
from dataclasses import asdict, replace
from datetime import datetime, time, timedelta
import json
from pathlib import Path
import shutil
import sqlite3

from run_analysis.config import load_config
from run_analysis.general_progression import VERSION, RATE_VERSION
from run_analysis.adaptive.factory import resume_service
from scripts.replay_adaptive_search import load_decision
from scripts.recorded_trace_execution import compatible_donor, transformed_payload
from scripts.research_manifest import freeze_sources
from scripts.run_adaptive_lifecycle import fast_fail_edits


def run(source, output, *, index=18, version=VERSION):
    case = load_decision(source, index)  # Mandatory original winner repricing.
    event = case['event']
    stamp = datetime.fromisoformat(event['as_of'])
    output.mkdir(exist_ok=False)
    root = output/'sandbox'
    root.mkdir()
    config = load_config()
    manifest = freeze_sources(output/'sources.zip', config=config, entrypoint=__name__,
        inputs=(source/'scenario.json', source/'sandbox.sqlite', source/f'decision-{index:04d}.json'))
    for name in ('scenario.json', 'execution-sandbox.json'):
        shutil.copy2(source/name, root/name)
    for number in range(1, index+1):
        for prefix in ('decision', 'snapshot'):
            name = f'{prefix}-{number:04d}.json'
            shutil.copy2(source/name, root/name)
    with sqlite3.connect(source.resolve().as_uri()+'/sandbox.sqlite?mode=ro', uri=True) as origin:
        with sqlite3.connect(root/'sandbox.sqlite') as target:
            origin.backup(target)
    # Pruning is confined to this freshly created copy. Compare aware timestamps
    # in Python; mixed UTC/local ISO strings are not lexicographically ordered.
    with sqlite3.connect(root/'sandbox.sqlite') as target:
        target.execute('DELETE FROM sandbox_decisions WHERE sequence>=?', (index,))
        for fingerprint, issued in target.execute('SELECT fingerprint,issued_at FROM sandbox_issues').fetchall():
            if datetime.fromisoformat(issued) > stamp:
                target.execute('DELETE FROM sandbox_issues WHERE fingerprint=?', (fingerprint,))
        for key, available in target.execute('SELECT activity_id,available_at FROM sandbox_completions').fetchall():
            if datetime.fromisoformat(available) > stamp:
                target.execute('DELETE FROM sandbox_completions WHERE activity_id=?', (key,))
        # The source screen has exactly one future accepted phase. Preserve the
        # prefix stage/ancestors, reject more complicated correction histories.
        for key, stage in target.execute('SELECT key,stage FROM stages').fetchall():
            if datetime.fromisoformat(json.loads(stage)['issued_at']) > stamp:
                target.execute('DELETE FROM stages WHERE key=?', (key,))
        prefix_stage = event['stage_runtime']['stage']['key']
        if not target.execute('SELECT 1 FROM stages WHERE key=?', (prefix_stage,)).fetchone():
            raise ValueError('Prefix stage must exist in the copied original ledger')
        target.execute('UPDATE current_stage SET key=? WHERE singleton=1', (prefix_stage,))
        if target.execute('SELECT count(*) FROM sandbox_completion_versions').fetchone()[0]:
            raise ValueError('Correction histories need an explicit causal-prefix adapter')
    scenario_path = root/'scenario.json'
    scenario = json.loads(scenario_path.read_text())
    scenario['search_policy']['general_progression'] = version
    scenario_path.write_text(json.dumps(scenario, indent=2)+'\n')
    service = resume_service(root, config, as_of=stamp)
    h = service.harness
    report = dict(manifest=manifest, source=str(source), source_index=index,
        initial_stage=asdict(h.stage), live_changed=False,
        scope='Issued-prefix upload/next-day seam, not daily multi-phase acceptance', decisions=[])
    try:
        day = stamp.astimezone(h.zone).date()
        issued = h.active.get(day.isoformat())
        if issued is None:
            raise ValueError('Choose a pre-handoff decision with an issued run today')
        dose = h.catalog[issued['dose']]
        seed_donors = [r for r in h.history if r.key in h.seed_keys and r.role == dose.role]
        _, original = compatible_donor(Path(config['paths']['database']), seed_donors, dose,
            config, h.ref, {})
        payload, phases, end, provenance = transformed_payload(original, dose,
            datetime.combine(day, time(19), h.zone))
        if end < h.stage.review_at:
            raise ValueError('Chosen upload does not cross the existing review boundary')
        h.upload(payload, day=day.isoformat(), available_at=end,
            observed_role=dose.role, observed_phases=phases)
        report['upload_provenance'] = provenance
        for label, at in (('handoff_upload', end+timedelta(seconds=1)),
                          ('next_rollover', datetime.combine(day+timedelta(days=1), time(4), h.zone))):
            decision = service.refresh(at, reason=label)
            h = service.harness
            event = decision['event']
            row = dict(label=label, at=at.isoformat(), stage=asdict(h.stage),
                review=decision['stage_review'], churn=event['churn'],
                seconds=event['seconds'], evaluations=event['evaluations'],
                four_day_failures=fast_fail_edits(event),
                seven_day_failures=fast_fail_edits(event, 7), miles21=event['volume'])
            report['decisions'].append(row)
            with (output/(label+'.json')).open('x') as stream:
                json.dump(row, stream, indent=2, default=lambda v:asdict(v) if hasattr(v,'__dataclass_fields__') else v.isoformat())
            print(json.dumps({k:row[k] for k in ('label','seconds','four_day_failures','seven_day_failures')}), flush=True)
            if row['four_day_failures'] or row['seven_day_failures']:
                report['status'] = 'stopped_for_attribution'
                break
        else:
            report['status'] = 'two_decision_seam_clean'
    except Exception as error:
        report.update(status='execution_error', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        service.harness.sandbox.close()
        with (output/'report.json').open('x') as stream:
            json.dump(report, stream, indent=2, default=lambda v:asdict(v) if hasattr(v,'__dataclass_fields__') else v.isoformat())
    return report


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--index', type=int, default=18)
    parser.add_argument('--version', choices=(VERSION, RATE_VERSION), default=VERSION)
    args = parser.parse_args()
    run(args.source, args.output, index=args.index, version=args.version)
