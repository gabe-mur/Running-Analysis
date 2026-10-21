"""Reprice local repairs of a pre-edit plan against one immutable decision.

Never search, publish, import or mutate either ledger. The archived winner is
reproduced before comparing repairs under exactly the same objective/constraints.
"""
from argparse import ArgumentParser
import json
from pathlib import Path

from scripts.replay_adaptive_search import load_decision
from run_analysis.adaptive.calendar_constraint_repair import (
    role_date_domains,constraint_repair_neighbors,
)
from run_analysis.adaptive.planner_preferences import long_day_allowed


def audit(root,index,prior_index):
    root=Path(root)
    case=load_decision(root,index)
    old=json.loads((root/f'decision-{prior_index:04d}.json').read_text())
    prior=tuple(tuple(p) for p in old['program'] if p[0] in case['dates'])
    state=case['state'];catalog=case['catalog']
    roles={k:d.role for k,d in catalog.items() if k in case['handoffs']}
    prefs=state.planner_preferences
    from datetime import date
    long_dates=(frozenset(d for d in case['dates'] if long_day_allowed(prefs,date.fromisoformat(d)))
        if prefs is not None and prefs.long_run_weekday is not None else None)
    domains=role_date_domains(case['dates'],roles,restrictions=state.restrictions,
        as_of=state.as_of,long_dates=long_dates)
    candidates=constraint_repair_neighbors(prior,case['dates'],roles,domains)
    rows=[]
    for program in candidates:
        r=case['evaluate_details'](program)
        rows.append(dict(eligible=r['decision_eligible'],cost=float(r['evaluation'].total_cost),
            versus_saved_percent=100*(float(r['evaluation'].total_cost)/case['archived_cost']-1),
            schedule=[dict(day=d,role=catalog[k].role,miles=catalog[k].miles) for d,k in program
                if d<state.context.hard_end.isoformat()]))
    return dict(decision_file=index,prior_file=prior_index,archived_cost_reproduced=float(case['archived_cost']),
        repairs=rows,weights_unchanged=True,searches=0,publication=False)


def main():
    p=ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path);p.add_argument('index',type=int);p.add_argument('prior_index',type=int)
    args=p.parse_args()
    print(json.dumps(audit(args.root,args.index,args.prior_index),indent=2,allow_nan=False))


if __name__=='__main__':
    main()
