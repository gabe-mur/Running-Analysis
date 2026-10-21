"""Exact evaluator parity across every saved daily/upload decision and neighbors.

Saved physiology, targets, readiness and forecast uncertainty stay intact. All
candidate outputs (including full component traces/fingerprints/admission), not
just final costs, must serialize to identical bytes. No search/publication here.
"""
from argparse import ArgumentParser
from dataclasses import asdict,is_dataclass
from datetime import date,datetime,time
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from zoneinfo import ZoneInfo

from run_analysis.config import load_config
from run_analysis.continuous_training_dose import ContinuousDoseReference
from run_analysis.adaptive.decision_snapshot import evaluate_snapshot
from run_analysis.adaptive.new_model_daily_harness import planned_accounted
from scripts.replay_adaptive_search import load_decision
from scripts.benchmark_lossless_identities import fast_identities
from scripts.research_manifest import freeze_sources


def encode(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value,(date,datetime)):
        return value.isoformat()
    raise TypeError(type(value).__name__)


def audit(root,output,compare_to=None):
    output.mkdir(parents=True,exist_ok=False)
    paths=sorted(root.glob('decision-*.json'))
    manifest=freeze_sources(output/'sources.zip',config=load_config(),entrypoint=__name__,
        inputs=(root/'scenario.json',*paths,*sorted(root.glob('snapshot-*.json')),
                *((compare_to,) if compare_to else ())))
    reference=ContinuousDoseReference('sandbox upload accounting')
    report=[];total=0;stamp=perf_counter()
    baseline={r['decision']:r for r in json.loads(compare_to.read_text())['results']} if compare_to else None
    for path in paths:
        case=load_decision(root,int(path.stem.split('-')[1]))
        state=case['state'];zone=ZoneInfo(state.timezone)
        physical={}
        def detail(program):
            for day,key in program:
                if (day,key) not in physical:
                    physical[(day,key)]=planned_accounted(f'candidate:{day}:{key}',
                        datetime.combine(date.fromisoformat(day),time(18),zone),
                        case['catalog'][key],case['handoffs'][key],reference,available=state.as_of)
            return evaluate_snapshot(state,tuple(physical[p] for p in program),tuple(k for _,k in program),reference=reference)
        original=tuple(map(tuple,case['event']['program']))
        choices={original,(),original[:1],original[1:]}
        index={day:i for i,day in enumerate(case['dates'])}
        for n,(day,key) in enumerate(original):
            rest=original[:n]+original[n+1:]
            choices.add(rest)
            for step in (-1,1):
                at=index[day]+step
                if 0<=at<len(case['dates']) and case['dates'][at] not in dict(rest):
                    choices.add(tuple(sorted((*rest,(case['dates'][at],key)))))
            dose=case['catalog'][key]
            for alternative in sorted(case['handoffs']):
                other=case['catalog'][alternative]
                if other.role==dose.role and round(abs(other.miles-dose.miles)*10)==1:
                    choices.add(tuple(sorted((*rest,(day,alternative)))))
        digest=sha256();accepted=0
        for program in sorted(choices):
            before=detail(program)
            left=json.dumps(before,sort_keys=True,separators=(',',':'),allow_nan=False,default=encode)
            if baseline is None:
                with fast_identities():
                    after=detail(program)
                right=json.dumps(after,sort_keys=True,separators=(',',':'),allow_nan=False,default=encode)
                assert left==right,(path,program)
            digest.update(left.encode());digest.update(b'\n')
            accepted+=bool(before['decision_eligible']);total+=1
        row=dict(decision=path.name,candidates=len(choices),accepted=accepted,
            exact_all_outputs=True,digest=digest.hexdigest())
        if baseline is not None:
            assert row==baseline[path.name],('Current production differs from pre-implementation output',path)
        report.append(row)
        print(json.dumps(row),flush=True)
    result=dict(manifest=manifest,decisions=len(paths),candidates=total,
        seconds=perf_counter()-stamp,exact_all_outputs=True,results=report,live_changed=False)
    if compare_to:
        assert len(report)==len(baseline),'Decision coverage differs'
        result['pre_implementation_comparator']=dict(path=str(compare_to),sha256=sha256(compare_to.read_bytes()).hexdigest())
    with (output/'report.json').open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k not in ('manifest','results')}),flush=True)


if __name__=='__main__':
    parser=ArgumentParser();parser.add_argument('root',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--compare-to',type=Path,help='Require exact pre-implementation per-decision digests')
    args=parser.parse_args();audit(args.root,args.output,args.compare_to)
