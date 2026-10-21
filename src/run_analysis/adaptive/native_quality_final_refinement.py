"""Check native quality/easy exchanges at the final published mileage state.

Representation-only descent on the authoritative whole score. No physiology,
family weight, continuity allowance, new calendar or unscored adoption.
"""
from time import perf_counter

from .continuous_search_adapter import canonical,fingerprint,interleave_proposals,rebase_edit
from .decimal_dose_search import quality_easy_neighbors,decimal_pair_neighbors


def refine(program,selected,*,evaluate,dates,roles,miles,budget,visible_days=7,program_transform=None):
    if type(budget) is not int or budget<0:
        raise ValueError('Explicit nonnegative terminal evaluation allowance required')
    if not selected['eligible']:
        raise ValueError('Terminal native refinement needs an admitted starting programme')
    start_time=perf_counter();program=original=canonical(program)
    def bound(p):
        p=canonical(p)
        if program_transform is None:return p
        b=canonical(program_transform(p))
        if (tuple(d for d,_ in b)!=tuple(d for d,_ in p) or
                any(k not in roles or roles[k]!=roles[old] for (_,k),(_,old) in zip(b,p)) or
                canonical(program_transform(b))!=b):
            raise ValueError('Terminal binding must preserve dates/roles and be idempotent')
        return b
    if bound(program)!=program:
        raise ValueError('Terminal starting programme must already be bound')
    cache={program:selected};calls=0;passes=[];reuse=[];certified=None
    index={d:i for i,d in enumerate(dates)}
    boundary=dates[visible_days] if visible_days<len(dates) else None
    original_cost=selected['cost']
    while calls<budget:
        parent=best=program;before_calls=calls;examined=0;complete=True;promising={}
        reuse_allowance=min(32,max(0,(budget-calls)//10))
        scan_limit=budget-reuse_allowance
        streams=(
            ((p,'quality_easy') for p in quality_easy_neighbors(parent,roles,miles,
                index=index,priority_before=boundary)),
            ((p,'easy_pair') for p in decimal_pair_neighbors(parent,roles,miles,index=index)))
        if program_transform is not None:
            def date_proposals():
                occupied=dict(parent)
                for day,key in parent:
                    for delta in (-1,1,-2,2,-3,3):
                        pos=index[day]+delta
                        if 0<=pos<len(dates) and dates[pos] not in occupied:
                            yield canonical((dates[pos] if d==day else d,k) for d,k in parent),'calendar'
            streams=(date_proposals(),*streams)
        for p,source in interleave_proposals(streams):
            p=bound(p)
            if p not in cache:
                if calls>=(scan_limit if best!=parent else budget):
                    complete=False;break
                cache[p]=evaluate(p);calls+=1
            examined+=1
            result=cache[p]
            if result['eligible'] and result['cost']<cache[parent]['cost']:
                promising[p]=result
                if len(promising)>8:
                    worst=max(promising,key=lambda q:(promising[q]['cost'],q))
                    del promising[worst]
            if result['eligible'] and result['cost']<cache[best]['cost']:
                best=p
        # Reuse the stable scheduler's scored-edit lesson: a better tail edit
        # must not discard an independently useful visible edit already found.
        # Rebase only on matching touched-date preconditions and fully reprice
        # the combined candidate. Never add deltas or overwrite conflicting work.
        reused=0;reuse_calls=calls;before_reuse=cache[best]['cost']
        while best!=parent and promising and calls<budget:
            next_best=best
            for alternative in sorted(promising,key=lambda q:(promising[q]['cost'],q)):
                p=rebase_edit(parent,alternative,best)
                if p is None:continue
                p=bound(p)
                if p not in cache:
                    if calls>=budget:break
                    cache[p]=evaluate(p);calls+=1
                result=cache[p]
                if result['eligible'] and result['cost']<cache[next_best]['cost']:
                    next_best=p
            if next_best==best:break
            best=next_best;reused+=1
        reuse.append(dict(origin=fingerprint(parent),pool=len(promising),adoptions=reused,
            evaluations=calls-reuse_calls,before_cost=before_reuse,after_cost=cache[best]['cost']))
        passes.append(dict(fingerprint=fingerprint(parent),best_fingerprint=fingerprint(best),
            before_cost=cache[parent]['cost'],after_cost=cache[best]['cost'],
            evaluations=calls-before_calls,examined=examined,scan_complete=complete))
        program=best
        if best==parent:
            if complete:certified=fingerprint(program)
            break
        if not complete:break
    return dict(program=program,selected=cache[program],cost=cache[program]['cost'],
        original_cost=original_cost,changed=program!=original,
        seconds=perf_counter()-start_time,evaluations=calls,maximum_evaluations=budget,
        passes=passes,certified_fingerprint=certified,
        promising_edit_reuse=reuse,
        budget_exhausted=bool(passes and not passes[-1]['scan_complete']),
        scope=('Terminal bound nearby-date and native exact-volume pairs after all calendar/mileage refinements; finite neighborhood only'
            if program_transform is not None else
            'Terminal same-date native quality/easy and easy/easy exact-volume pairs after all calendar/mileage refinements; not a global or calendar certificate'))
