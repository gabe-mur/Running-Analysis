"""BA2 same-origin backward comparison: saved programs, original input frames.

No horizon translation, tail regeneration, new calendar search or policy change.
Score only after checking the existing hard and downstream feasibility rules.
"""
from argparse import ArgumentParser
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

from run_analysis import weekly_schedule as planner
from run_analysis.aerobic_acceptance import assess_aerobic_exposure
from scripts.audit_backward_controls import plain, validate_control
from scripts.audit_continuation_upload import reconstruct, unpack_soft, evaluate_physical, raw_state_lookup, hard_score
from scripts.audit_distance_continuity import components
from scripts.audit_regrowth_gates import check_candidate
from scripts.continuation_checks import check_sequence
from scripts.continuation_replan import fingerprint, stable_prefix
from scripts.expected_completion import expected_completion_policy


def aligned(before, after):
    for key in ('origin', 'hard_end_exclusive', 'full_end_exclusive'):
        if before[key] != after[key]:
            raise ValueError('Same-origin audit requires identical accounting boundaries')
    for key in ('completed_history', 'config', 'daily_states', 'daily_state_options'):
        if before['field_hashes'][key] != after['field_hashes'][key]:
            raise ValueError('Same-origin audit requires identical history, state and config')


def coverage(report, index, hard, full):
    offsets = [i for i,d in enumerate(hard) if d.recommendation
               and d.recommendation.workout_type != planner.WorkoutType.REST]
    saved = report['replans'][index]['planner_diagnostics']['continuation_selection']
    record = report['synthetic_sandbox']['calendar_neighbor_prototype'][index]
    proposals = record['evaluated'] + saved.get('suffix_regrowth', {}).get('rows', [])
    hard_hash, tail_hash = fingerprint(hard), fingerprint(full[len(hard):])
    same_calendar = [r for r in saved['finalists'] if r['offsets'] == offsets]
    same_hard = [r for r in saved['finalists'] if r.get('hard_fingerprint') == hard_hash]
    exact = [r for r in same_hard if r.get('tail_fingerprint') == tail_hash]
    return dict(offsets=offsets, proposal_calendar_rows=[r for r in proposals if r['offsets'] == offsets],
        finalist_calendar_rows=same_calendar, exact_hard_rows=same_hard, exact_hard_tail_rows=exact,
        exact_pair_scored=any('combined_cost' in r for r in exact),
        limitation='Absent fingerprint means absent from saved finalized trace; not proof the beam never generated an equivalent dose assignment.')


def investigate(report, controls):
    frames = {r['index']: r for r in controls['decisions'] if r['index'] in (1,2)}
    aligned(frames[1], frames[2])
    programs = {}
    for index, row in frames.items():
        hard, full = unpack_soft(row['hard_program']), unpack_soft(row['full_program'])
        if fingerprint(hard) != row['hard_fingerprint'] or fingerprint(full) != row['full_fingerprint']:
            raise ValueError('Saved control program identity mismatch')
        programs[index] = hard, full
    rows = []
    def observe(index, inputs, days, evidence):
        if index not in frames:
            return
        snapshot = report['replans'][index]
        probes = []
        for source, (hard, full) in programs.items():
            with expected_completion_policy(inputs['config']):
                sequence = check_sequence(full, inputs, raw_state_lookup(inputs,evidence), ())
                gates = check_candidate(snapshot, inputs, hard)
                offsets = [i for i,d in enumerate(hard) if d.recommendation
                    and d.recommendation.workout_type != planner.WorkoutType.REST]
                _, _, details = hard_score(inputs, offsets, days=hard)
                acceptance = assess_aerobic_exposure(**details['aerobic_acceptance_facts'],
                    planning_mode=inputs['target_evidence'].planning_mode.value)
                feasible = (sequence['checks_pass'] and not acceptance.rejected
                    and acceptance.reason != 'missing_context'
                    and not any(gates[k] for k in ('failures','whole_hard_peak_failures','strict_cadence_findings'))
                    and not (set(offsets) & (set(inputs['forced_rest_offsets']) | set(inputs['completed_run_offsets']))))
                probe = dict(program_decision=source, feasible=feasible, sequence=sequence,
                    gates=gates, acceptance=asdict(acceptance), stable_prefix=stable_prefix(hard, inputs),
                    coverage=coverage(report,index,hard,full))
                if feasible:
                    scored = evaluate_physical(inputs, hard, evidence, full=full)
                    parts = components(scored)
                    probe.update(cost=sum(parts.values()), components=parts,
                        recovery_components=scored['continuation']['trajectory_recovery_components'])
                    if source == index:
                        validate_control(snapshot, inputs, hard, full, evidence['runs'], probe['cost'])
                elif source == index:
                    raise ValueError('Original program no longer passes feasibility')
                probes.append(probe)
        rows.append(dict(frame=index, generated_at=snapshot['generated_at'], probes=probes))
        print(f"BA2 frame {index}: " + ', '.join(f"program {p['program_decision']} = {p.get('cost', 'infeasible')}" for p in probes), flush=True)
    reconstruct(dict(report,replans=report['replans'][:3]), observe)
    return dict(rows=rows, limitation='Same-origin pair only; initial upload/horizon translation remains untested.')


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--controls', type=Path, required=True)
    parser.add_argument('--output-json', type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        parser.error('Require a new output path')
    controls = json.loads(args.controls.read_text())
    result = dict(scope=__doc__, branches={})
    for name, branch in controls['branches'].items():
        content = Path(branch['source']).read_bytes()
        if sha256(content).hexdigest() != branch['source_sha256']:
            raise ValueError('Source report changed since BA1')
        result['branches'][name] = investigate(json.loads(content),branch)
    encoded = json.dumps(plain(result), indent=2, allow_nan=False)
    with args.output_json.open('x') as stream:
        stream.write(encoded)


if __name__ == '__main__':
    main()
