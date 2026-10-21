"""Frozen endurance-command construction; sensitivity, not a selected policy."""
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from time import perf_counter
import hashlib
import json

from scripts.audit_staged_progression import original_history
from scripts.audit_recency_demand import json_ready
from run_analysis.coaching_contracts import ReductionPolicy, compile_endurance_contract
from run_analysis.config import load_config
from run_analysis.staged_progression import ProgressionPolicy, bootstrap_stage


def audit(database, config, as_of):
    started = perf_counter()
    before = hashlib.sha256(database.read_bytes()).hexdigest()
    history, unresolved = original_history(database, config, as_of)
    if unresolved:
        return dict(status='unresolved_original_history', unresolved=unresolved, planner_influence=False)
    seed = bootstrap_stage(history, as_of, goal_long_miles=17., policy=ProgressionPolicy(14, .05))
    cases = {}
    if seed['stage']:
        for fraction in (.70, .80, .90):
            cases[f'easy_fraction_{fraction:.2f}'] = compile_endurance_contract(history, seed['stage'],
                purpose='easy', intensity_key='controlled_easy', reduction=ReductionPolicy(
                    fraction, 20., 'predeclared_sensitivity_only_not_selected_policy'))
        cases['long_no_declared_reduction'] = compile_endurance_contract(history, seed['stage'],
            purpose='long', intensity_key='controlled_easy')
    return dict(as_of=as_of, original_sessions=len(history), seed_status=seed['status'],
        stage=seed['stage'], cases=cases, elapsed_seconds=perf_counter()-started,
        source_database_bytes_unchanged=before == hashlib.sha256(database.read_bytes()).hexdigest(),
        planner_influence=False, physiological_validation=False,
        limitations=['Current roles/health/settings held fixed; upload availability assumed at completion',
            'Phase suitability and original program feasibility remain unresolved',
            '14-day/5% stage review and reduction fractions/floor remain declared experimental inputs',
            'No ideal reduction or progression rate selected from this arithmetic audit'])


if __name__ == '__main__':
    p = ArgumentParser(description=__doc__)
    p.add_argument('--database', type=Path, required=True)
    p.add_argument('--config', type=Path, default=Path('config.yaml'))
    p.add_argument('--as-of', type=datetime.fromisoformat, required=True)
    args = p.parse_args()
    print(json.dumps(audit(args.database, load_config(args.config), args.as_of),
                     default=json_ready, allow_nan=False, indent=2))
