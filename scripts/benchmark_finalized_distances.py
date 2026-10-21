"""Fixed-input distance-search timing and guard audit, without calendar search."""
from argparse import ArgumentParser
from datetime import timedelta
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

from scripts.audit_continuation_upload import reconstruct, unpack_soft, raw_state_lookup
from scripts.audit_horizon_rollover import probe
from scripts.audit_backward_controls import plain
from scripts.continuation_replan import fingerprint, signature
from scripts.finalized_distance_search import refine_distances


def benchmark(report, indices, *, visible_days=7, budget=64):
    rows=[]
    def observe(index,inputs,days,evidence):
        if index not in indices:
            return
        snapshot=report['replans'][index]
        saved=snapshot['planner_diagnostics']['continuation_selection']
        full=days+unpack_soft(saved['soft_sessions'])
        start=days[0].date
        hard_end=start+timedelta(days=len(days)); full_end=start+timedelta(days=len(full))
        def inspect(days,full):
            return probe(snapshot,inputs,evidence,full,hard_end,full_end)
        at=perf_counter(); base=inspect(days,full); baseline_seconds=perf_counter()-at
        if not base['feasible'] or abs(base['cost']-saved['search_cost'])>1e-8:
            raise ValueError('Baseline score/guards mismatch')
        def wrap(days,full,result):
            offsets=[i for i,d in enumerate(days) if d.recommendation]
            return dict(days=days,program=(full,[],[]),row=dict(hard_fingerprint=fingerprint(days)),
                tail_fingerprint=fingerprint(full[len(days):]),stable=result['stable_prefix'],
                score=SimpleNamespace(total_cost=result['cost'],offsets=offsets),audit=result)
        baseline=wrap(days,full,base)
        candidates=[baseline]; probes=[]
        def evaluate(seed):
            candidate_days=seed['days']; candidate_full=candidate_days+full[len(days):]
            result=inspect(candidate_days,candidate_full)
            probes.append(result)
            if not result['feasible']:
                return None
            candidate=wrap(candidate_days,candidate_full,result)
            candidates.append(candidate)
            return candidate
        refined=refine_distances([baseline],evaluate,inputs,raw_state_lookup(inputs,evidence),
            visible_days=visible_days,budget=budget)
        winner=min(candidates,key=lambda c:c['score'].total_cost)
        rows.append(dict(index=index,baseline_seconds=baseline_seconds,baseline_cost=base['cost'],
            selected_cost=winner['score'].total_cost,refinement=refined,
            baseline_program=signature(full),selected_program=signature(winner['program'][0]),
            selected_audit=winner['audit'],probes=probes))
        print(f"Distance benchmark {index}: {base['cost']:.6f} -> {winner['score'].total_cost:.6f}; "
            f"{refined['evaluations']} probes, {refined['seconds']:.3f}s, {[l['status'] for l in refined['lanes']]}",flush=True)
    reconstruct(dict(report,replans=report['replans'][:max(indices)+1]),observe)
    return dict(scope=__doc__,rows=rows)


def main():
    p=ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--indices',type=int,nargs='+',default=[2,3])
    p.add_argument('--visible-days',type=int,default=7)
    p.add_argument('--budget',type=int,default=64)
    p.add_argument('--output-json',type=Path,required=True)
    a=p.parse_args()
    if a.output_json.exists() or a.visible_days<1 or a.budget<1:
        p.error('Require new output path and positive window/budget')
    result=benchmark(json.loads(a.report.read_text()),a.indices,visible_days=a.visible_days,budget=a.budget)
    with a.output_json.open('x') as out:
        json.dump(plain(result),out,indent=2,allow_nan=False)


if __name__=='__main__':
    main()
