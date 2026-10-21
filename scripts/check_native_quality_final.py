"""Frozen terminal-search seam control, not an independent opening or replay.

Retain an already-issued original winner and expose the public quality/easy
operator's catalogue. No hindsight winner is supplied. Discovery is deliberately
limited to isolate the terminal allowance; this cannot prove global optimality.
"""
from argparse import ArgumentParser
import json
from pathlib import Path

from scripts.replay_adaptive_search import load_decision
from scripts.research_manifest import freeze_sources
from run_analysis.adaptive.continuous_search_adapter import search,canonical


def run(root,output,*,index,allowance=1000,terminal_only=False):
    output.mkdir(exist_ok=False)
    case=load_decision(root,index)
    original=canonical(case['event']['program'])
    supported={k:case['catalog'][k] for k in case['handoffs']}
    if terminal_only:
        from run_analysis.adaptive.native_quality_final_refinement import refine
        result=refine(original,case['evaluate'](original),evaluate=case['evaluate'],
            dates=case['dates'],roles={k:d.role for k,d in supported.items()},
            miles={k:d.miles for k,d in supported.items()},budget=allowance)
    else:
        result=search(dates=case['dates'],keys=tuple(supported),previous=original,
            roles={k:d.role for k,d in supported.items()},dose_miles={k:d.miles for k,d in supported.items()},
            hard_end=case['state'].context.hard_end.isoformat(),evaluate=case['evaluate'],
            budget=2,final_winner_budget=allowance,fair_coordinates=True)
    assert result['selected']['eligible'] and result['cost']<=case['archived_cost']
    assert result['evaluations']<=allowance+2
    report=dict(original_cost=case['archived_cost'],result=result,scope=__doc__,
        terminal_only=terminal_only,live_changed=False)
    report['manifest']=freeze_sources(output/'sources.zip',config={},entrypoint=__name__,
        inputs=(root/'scenario.json',root/f'decision-{index:04d}.json',root/case['event']['snapshot_file']))
    with (output/'report.json').open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(dict(original_cost=report['original_cost'],cost=result['cost'],
        seconds=result['seconds'],evaluations=result['evaluations'],
        changes=[dict(day=d,before=dict(original).get(d),after=k) for d,k in result['program']
            if dict(original).get(d)!=k],final_check=(
                {k:result[k] for k in ('passes','certified_fingerprint','scope')} if terminal_only
                else result['refinement']['final_winner_doses'])),indent=2))


if __name__=='__main__':
    p=ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    p.add_argument('--index',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--terminal-only',action='store_true')
    a=p.parse_args();run(a.root,a.output,index=a.index,terminal_only=a.terminal_only)
