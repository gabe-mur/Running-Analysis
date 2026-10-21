"""Reprice archived rollover alternatives; no new search or physical evidence."""
from argparse import ArgumentParser
import json
from pathlib import Path

from scripts.replay_adaptive_search import load_decision


def run(root, *, before=1, after=2):
    a,b=(load_decision(root,i) for i in (before,after))
    old,new=(tuple(tuple(p) for p in case['event']['program']) for case in (a,b))
    remaining=tuple(p for p in old if p[0] in b['dates'])
    # The expired prefix belongs to the original decision, even if it is rest.
    # Drop only the newly exposed suffix that the original evaluator cannot see.
    restored=tuple(sorted((*[p for p in old if p[0]<b['dates'][0]],
        *[p for p in new if p[0] in a['dates']])))
    edit_cases=[]
    for edit in b['event']['churn']['7']:
        if edit['date_changed'] or edit['role_changed'] or edit['distance_change'] is None:
            continue
        day=edit['day'];catalog=a['catalog']
        first,second=edit['before'],edit['after']
        if first not in catalog or second not in catalog or catalog[first].role!='easy':
            continue
        alone=tuple((d,second if d==day else k) for d,k in old)
        units=round((catalog[first].miles-catalog[second].miles)*10)
        controls=[]
        for partner,key in old:
            if partner==day or catalog[key].role!='easy':
                continue
            target=round(catalog[key].miles*10)+units
            choices=[k for k in a['handoffs'] if catalog[k].role=='easy' and round(catalog[k].miles*10)==target]
            if not choices:
                continue
            key2=min(choices)
            proposed=tuple((d,second if d==day else key2 if d==partner else k) for d,k in old)
            priced=a['evaluate'](proposed)
            controls.append(dict(partner=partner,old_key=key,new_key=key2,
                separation_days=abs(a['dates'].index(partner)-a['dates'].index(day)),
                exact_volume=True,**priced))
        edit_cases.append(dict(edit=edit,alone_on_original=a['evaluate'](alone),equal_volume_controls=controls))
    result=dict(scope='Fixed-candidate repricing only; no search, new upload or global certificate',
        before_fingerprint=a['state'].fingerprint,after_fingerprint=b['state'].fingerprint,
        forecasts_identical=a['handoffs']==b['handoffs'],
        physical_history_identical=a['state'].history==b['state'].history,
        phase_history_identical=a['state'].phase_history==b['state'].phase_history,
        old_before=a['evaluate'](old),later_on_original=a['evaluate'](restored),
        old_remaining_after=b['evaluate'](remaining),later_after=b['evaluate'](new),
        newly_exposed_sessions=[p for p in new if p[0] not in a['dates']],dose_edits=edit_cases)
    with (root/'rollover-search-audit.json').open('x') as stream:
        json.dump(result,stream,indent=2)
    print(json.dumps(result,indent=2),flush=True)
    return result


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('--before',type=int,default=1)
    parser.add_argument('--after',type=int,default=2)
    args=parser.parse_args();run(args.root,before=args.before,after=args.after)
