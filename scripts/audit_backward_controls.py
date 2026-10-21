"""BA1: reconstruct and verify original decision frames, without counterfactuals.

Saved synthetic daily/upload decisions only. No search, policy change, future
observations injected into earlier frames, database access or deployment.
"""
from argparse import ArgumentParser
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timedelta
from enum import Enum
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path

from run_analysis import weekly_schedule as planner
from run_analysis.selection_objective import PRIOR_DISTANCE_PRESCRIPTIONS
from scripts.audit_continuation_upload import reconstruct, unpack_soft, evaluate_physical
from scripts.audit_distance_continuity import components
from scripts.audit_regrowth_gates import check_candidate
from scripts.continuation_replan import fingerprint, signature
from scripts.expected_completion import expected_completion_policy
from scripts.typed_continuation_prototype import runs


def plain(value):
    if hasattr(value, 'model_dump'):
        return plain(value.model_dump(mode='json'))
    if is_dataclass(value):
        return plain(asdict(value))
    if isinstance(value, Enum):
        return plain(value.value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(plain(k)): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted(plain(v) for v in value)
    if isinstance(value, float) and not isfinite(value):
        # Existing diagnostics use infinity for rejected candidates. Preserve
        # that meaning explicitly; never confuse it with a finite scored plan.
        return {'non_finite': str(value)}
    return value


def digest(value):
    return sha256(json.dumps(plain(value), sort_keys=True, allow_nan=False).encode()).hexdigest()


def prior_frame(priors):
    # Reconstruction creates response wrappers at the current wall clock.
    # Match the existing physical-fingerprint contract: exclude generated_at,
    # but retain planned_for, fitness_state_as_of and all prescription content.
    return {d.isoformat(): r.model_dump(mode='json', exclude={'generated_at'})
            for d, r in priors.items()}


def validate_control(snapshot, inputs, days, full, history, actual_cost):
    saved = snapshot['planner_diagnostics']['continuation_selection']
    if fingerprint(days) != saved['selected_hard_fingerprint'] or fingerprint(full) != saved['selected_program_fingerprint']:
        raise ValueError('Physical prescription identity mismatch')
    if not isfinite(actual_cost) or abs(actual_cost - saved['search_cost']) > 1e-8:
        raise ValueError('Own-input score mismatch')
    for name in ('winning_candidate', 'published_candidate'):
        score = snapshot['planner_diagnostics'].get(name)
        if score is not None and abs(actual_cost-score['total_cost']) > 1e-8:
            raise ValueError('Saved selection/publication score mismatch')
    origin = inputs['daily_states'][0].as_of
    generated = datetime.fromisoformat(snapshot['generated_at'])
    if any(r.start_time >= min(origin, generated) for r in history):
        raise ValueError('Completed ledger contains unavailable future work')
    if len(days) != len(inputs['daily_states']) or len(full) != len(days) + 28:
        raise ValueError('Unexpected hard/soft accounting lengths')
    expected = [origin.date() + timedelta(days=i) for i in range(len(full))]
    if [d.date for d in full] != expected:
        raise ValueError('Program is not a contiguous absolute-date sequence')


def capture(report, *, indices=(0, 1, 2)):
    if not indices or any(type(i) is not int or i < 0 or i >= len(report['replans']) for i in indices):
        raise ValueError('Require valid decision indices')
    rows = []
    def observe(index, inputs, days, evidence):
        if index not in indices:
            return
        snapshot = report['replans'][index]
        saved = snapshot['planner_diagnostics']['continuation_selection']
        full = days + unpack_soft(saved['soft_sessions'])
        with expected_completion_policy(inputs['config']):
            scored = evaluate_physical(inputs, days, evidence, full=full)
            target_without_future = inputs['expected_target_projector']([], len(full))
            target_with_hard = inputs['expected_target_projector'](runs(days), len(full))
        parts = components(scored)
        validate_control(snapshot, inputs, days, full, evidence['runs'], sum(parts.values()))
        gates = check_candidate(snapshot, inputs, days)
        if not scored['continuation']['fixed_sequence_checks']['checks_pass'] or any(
                gates[k] for k in ('failures', 'whole_hard_peak_failures', 'strict_cadence_findings')):
            raise ValueError('Original program failed unchanged guards')
        fields = ('config', 'request', 'daily_states', 'daily_state_options', 'target_distance_range',
            'target_evidence', 'completed_miles_by_offset', 'prior_distance_midpoints_by_date',
            'prior_run_offsets', 'prior_roles_by_offset', 'forced_rest_offsets', 'completed_run_offsets')
        frame = plain({k: inputs[k] for k in fields})
        frame.update(completed_history=plain(evidence['runs']), history_source=evidence['source'],
            template=plain(evidence['template']),
            prior_prescriptions=prior_frame(PRIOR_DISTANCE_PRESCRIPTIONS.get() or {}),
            target_without_future=plain(target_without_future), target_with_hard=plain(target_with_hard))
        origin = inputs['daily_states'][0].as_of
        row = dict(index=index, generated_at=snapshot['generated_at'], trigger=snapshot['trigger'],
            origin=origin.isoformat(), hard_end_exclusive=(days[-1].date+timedelta(days=1)).isoformat(),
            full_end_exclusive=(full[-1].date+timedelta(days=1)).isoformat(),
            snapshot=snapshot, frame=frame, frame_hash=digest(frame),
            field_hashes={k:digest(v) for k,v in frame.items()},
            hard_program=signature(days), full_program=signature(full),
            hard_fingerprint=fingerprint(days), full_fingerprint=fingerprint(full),
            saved_cost=saved['search_cost'], reproduced_cost=sum(parts.values()), components=parts,
            recovery_components=scored['continuation']['trajectory_recovery_components'],
            guards=scored['continuation']['fixed_sequence_checks'], gates=gates)
        rows.append(row)
        print(f"Verified BA1 decision {index}: {snapshot['generated_at']}, score {row['reproduced_cost']:.9f}", flush=True)
    # Stop once the last requested saved decision is verified, not at the end
    # of a longer report. Earlier unselected decisions still reconstruct normally.
    reconstruct(dict(report, replans=report['replans'][:max(indices)+1]), observe)
    pairs = []
    for before, after in zip(rows, rows[1:]):
        pairs.append(dict(before=before['index'], after=after['index'],
            same_origin=before['origin']==after['origin'],
            same_hard_end=before['hard_end_exclusive']==after['hard_end_exclusive'],
            same_full_end=before['full_end_exclusive']==after['full_end_exclusive'],
            changed_frame_fields=[k for k in before['field_hashes'] if before['field_hashes'][k]!=after['field_hashes'][k]]))
    return dict(decisions=rows, transitions=pairs)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--reports', nargs='+', type=Path, required=True)
    parser.add_argument('--output-json', type=Path, required=True)
    parser.add_argument('--indices', type=int, nargs='+', default=[0, 1, 2])
    args = parser.parse_args()
    if args.output_json.exists():
        parser.error('Require a new output path')
    result = dict(scope=__doc__, branches={})
    for path in args.reports:
        content = path.read_bytes()
        report = json.loads(content)
        result['branches'][path.stem] = dict(source=str(path), source_sha256=sha256(content).hexdigest(),
            policies={k:v for k,v in report['synthetic_sandbox'].items() if k in (
                'expected_completion', 'full_score_refinement', 'volume_path', 'funding_accounting',
                'canonical_distance_targets', 'shared_selection_objective', 'finalized_scoring_trial',
                'opportunity_search_repair', 'continuation_selection')}, **capture(report, indices=args.indices))
    result['numeric_planner_constants'] = {k:v for k,v in vars(planner).items()
        if k.isupper() and isinstance(v, (int, float))}
    encoded = json.dumps(plain(result), indent=2, allow_nan=False)
    with args.output_json.open('x') as stream:
        stream.write(encoded)


if __name__ == '__main__':
    main()
