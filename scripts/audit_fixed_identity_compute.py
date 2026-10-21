"""Exact frozen-frame output audit; no search, uploads or app writes.

Capture baseline digests BEFORE changing hot functions. Recheck the same
physical candidates, including denied/partial alternatives, after optimization.
Quality proposals bind through the saved coaching sequence before evaluation.
"""
from argparse import ArgumentParser
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

from run_analysis.config import load_config
from scripts.replay_adaptive_search import load_decision
from scripts.research_manifest import freeze_sources


def encode(value):
    if is_dataclass(value):return asdict(value)
    if isinstance(value, (date, datetime)):return value.isoformat()
    raise TypeError(type(value).__name__)


def proposals(case):
    original=tuple(map(tuple,case['event']['program']))
    choices={original,(),original[:1],original[1:]}
    dates=case['dates'];index={d:i for i,d in enumerate(dates)}
    for n,(day,key) in enumerate(original):
        rest=original[:n]+original[n+1:]
        choices.add(rest)
        for step in (-1,1):
            at=index[day]+step
            if 0<=at<len(dates) and dates[at] not in dict(rest):
                choices.add(tuple(sorted((*rest,(dates[at],key)))))
        dose=case['catalog'][key]
        for alternative in sorted(case['handoffs']):
            other=case['catalog'][alternative]
            if other.role==dose.role and round(abs(other.miles-dose.miles)*10)==1:
                choices.add(tuple(sorted((*rest,(day,alternative)))))
    sequence=case['state'].quality_coaching
    if sequence is not None:
        choices={sequence.bind(p,case['catalog']) for p in choices}
    return sorted(choices)


def audit(root,output,compare_to=None):
    output.mkdir(parents=True,exist_ok=False)
    paths=sorted(root.glob('decision-*.json'))
    manifest=freeze_sources(output/'sources.zip',config=load_config(),entrypoint=__name__,
        inputs=(root/'scenario.json',*paths,*sorted(root.glob('snapshot-*.json')),
                *((compare_to,) if compare_to else ())))
    reference=json.loads(compare_to.read_text()) if compare_to else None
    baseline={r['decision']:r for r in reference['results']} if reference else None
    rows=[];start=perf_counter()
    for path in paths:
        case=load_decision(root,int(path.stem.split('-')[1]))
        digest=sha256();candidate_digest=sha256();accepted=0;programs=proposals(case)
        for program in programs:
            candidate_digest.update(json.dumps(program,separators=(',',':')).encode())
            candidate_digest.update(b'\n')
            result=case['evaluate_details'](program)
            raw=json.dumps(result,sort_keys=True,separators=(',',':'),allow_nan=False,default=encode)
            digest.update(raw.encode());digest.update(b'\n')
            accepted+=bool(result['decision_eligible'])
        row=dict(decision=path.name,candidates=len(programs),accepted=accepted,
            candidate_digest=candidate_digest.hexdigest(),digest=digest.hexdigest())
        if baseline is not None:
            if row!=baseline[path.name]:raise ValueError(('Full output parity failed',path.name,row,baseline[path.name]))
        rows.append(row)
        print(json.dumps(row),flush=True)
    if baseline is not None and len(rows)!=len(baseline):raise ValueError('Frame coverage differs')
    report=dict(manifest=manifest,results=rows,decisions=len(rows),
        candidates=sum(r['candidates'] for r in rows),seconds=perf_counter()-start,
        compared_to=str(compare_to) if compare_to else None,
        exact_all_outputs=bool(baseline),live_changed=False,search_performed=False)
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps({k:v for k,v in report.items() if k not in ('manifest','results')}),flush=True)


if __name__=='__main__':
    p=ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--compare-to',type=Path)
    args=p.parse_args();audit(args.root,args.output,args.compare_to)
