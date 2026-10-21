"""Bounded calendar-and-dose alternatives, priced by the same full evaluator.

Explicit v205 shared-search release; generic archived defaults remain unchanged.
No scoring coefficient, calendar lock, preference for rest, or future-upload
witness enters candidate generation.
"""
from time import perf_counter
from .decimal_dose_search import decimal_merge_neighbors
from .compiled_calendar_doses import compile_calendar, refine


def refine_merges(state,program,*,physical,evaluate_details,max_full_evaluations=32):
    if type(max_full_evaluations) is not int or max_full_evaluations<4:
        raise ValueError('Explicit bounded calendar/dose allowance required')
    started=perf_counter(); original=tuple(sorted(program))
    dates=tuple(sorted({d for d,k in physical})); index={d:i for i,d in enumerate(dates)}
    roles={k:r.role for (d,k),r in physical.items()}
    miles={k:r.miles for (d,k),r in physical.items()}
    cache={}; details={}; calls=0
    def checked(p):
        nonlocal calls
        p=tuple(sorted(p))
        if p not in cache:
            if calls>=max_full_evaluations:
                raise ValueError('Calendar/dose diagnostic exceeded its full-evaluation allowance')
            raw=evaluate_details(p);details[p]=raw;calls+=1
            cache[p]=dict(cost=raw['evaluation'].total_cost,eligible=raw['decision_eligible'])
        return cache[p]
    baseline=checked(original)
    if not baseline['eligible']:
        raise ValueError('Admitted initial calendar required')
    candidates=[]; complete=True
    # Reserve half for allocating retained calendars. Poor INITIAL doses do not
    # get to exhaust the entire allowance before any joint alternative is tried.
    for p in decimal_merge_neighbors(original,roles,miles,index=index):
        if calls>=max_full_evaluations//2:
            complete=False;break
        scored=checked(p)
        if scored['eligible']:
            candidates.append(p)
    traces=[]; winner=original
    ordered=sorted(candidates,key=lambda p:(cache[p]['cost'],p))
    for p in ordered:
        remaining=max_full_evaluations-calls
        if remaining<2:
            complete=False;break
        compiled=compile_calendar(state,p,physical=physical,result=details[p])
        # Each retained calendar gets a bounded lane; this is not a certificate
        # for the other calendars or for the newly adopted winner.
        allowance=min(8,remaining+1)
        polished=refine(compiled,checked,max_full_evaluations=allowance)
        proposed=polished['program']
        if checked(proposed)['eligible'] and cache[proposed]['cost']<cache[winner]['cost']:
            winner=proposed
        traces.append(dict(initial_cost=cache[p]['cost'],**polished))
    return dict(program=winner,cost=cache[winner]['cost'],original_cost=baseline['cost'],
        changed=winner!=original,full_evaluations=calls,seconds=perf_counter()-started,
        retained_calendars=len(candidates),calendar_enumeration_complete=complete,
        max_full_evaluations=max_full_evaluations,trace=traces,
        scope='Bounded exact-volume easy merges with full-scored dose refinement; no global or calendar-local certificate')
