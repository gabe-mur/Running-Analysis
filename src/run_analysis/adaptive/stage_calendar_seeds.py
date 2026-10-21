"""Complete stage-funded starting calendars, never locked training templates.

Role phases and easy-run counts vary independently. The full evaluator prices
them; existing add/delete/move/role/dose refinements remove periodicity afterward.
"""
from itertools import combinations,zip_longest
from math import floor
from datetime import date


def complete_stage_seeds(dates,catalog,policy,*,supporting_miles,limit=168,long_dates=None):
    if limit < 1:
        raise ValueError('Positive seed limit required')
    if not dates:
        return ()
    groups={role:[d for d in catalog.values() if d.role==role] for role in ('easy','long','quality')}
    if not all(groups.values()):
        return ()
    long=max(groups['long'],key=lambda d:(d.miles,d.key))
    quality=max(groups['quality'],key=lambda d:(sum(s for k,s in d.phases if k=='work'),d.key))
    # Initial coarse phase lattice only, not a due-date or a future run-count cap.
    period=7 if long_dates is not None else max(2,round(7/policy.long_per_7))
    quality_period=max(2,round(7/policy.quality_per_7))
    minimum=min(d.miles for d in groups['easy'])
    max_count=min(period-2,max(1,floor(supporting_miles*period/7/minimum)))
    # A fixed date coordinate orders candidates, not workouts. All role phases
    # are still enumerated in Automatic mode. An explicit user weekday filters
    # long phases without constraining the independent quality phase.
    # Advancing the opening date must not rotate geometry ties into/out of a
    # capped library simply because their relative offsets changed.
    origin=date.fromisoformat(dates[0]).toordinal()
    families=[]
    for absolute_long in range(period):
        long_phase=(absolute_long-origin)%period
        if long_dates is not None and not any(i%period==long_phase and day in long_dates for i,day in enumerate(dates)):
            continue
        for absolute_quality in range(quality_period):
            quality_phase=(absolute_quality-origin)%quality_period
            fixed={i:long.key for i in range(long_phase,len(dates),period)
                if long_dates is None or dates[i] in long_dates}
            quality_days=range(quality_phase,len(dates),quality_period)
            if any(i in fixed for i in quality_days):
                continue
            fixed.update((i,quality.key) for i in quality_days)
            free=[i for i in range(period) if i not in fixed]
            family=[]
            for count in range(1,min(max_count,len(free))+1):
                target=supporting_miles*period/7/count
                # These are FUNDING seeds. Choose the smallest representable
                # dose that covers the share, then let full-score refinement
                # compare smaller doses. Nearest rounding can seed every family
                # just under target and make a funded growth proposal invisible.
                sufficient=[d for d in groups['easy'] if d.miles+1e-9>=target]
                dose=min(sufficient,key=lambda d:(d.miles,d.key)) if sufficient else max(groups['easy'],key=lambda d:(d.miles,d.key))
                choices=list(combinations(free,count))
                # Spread initial representatives geometrically, not by an old
                # recovery score. Every selected complete plan is fully priced.
                def geometry(offsets):
                    whole=sorted((long_phase,*offsets,quality_phase))
                    gaps=[b-a for a,b in zip(whole,(*whole[1:],whole[0]+period))]
                    return (max(gaps),sum(g*g for g in gaps),
                            tuple(sorted((origin+i)%period for i in offsets)))
                choices.sort(key=geometry)
                options=[]
                for offsets in choices:
                    plan=dict(fixed)
                    for i in range(len(dates)):
                        if i not in plan and i%period in offsets:
                            plan[i]=dose.key
                    options.append(tuple((dates[i],k) for i,k in sorted(plan.items())))
                family.append(options)
            # Interleave counts so limited compute cannot silently favor the
            # first frequency family, then interleave all long/quality phases.
            families.append([p for row in zip_longest(*family) for p in row if p is not None])
    results=[]
    seen=set()
    for row in zip_longest(*families):
        for p in row:
            if p is not None and p not in seen:
                results.append(p);seen.add(p)
                if len(results)>=limit:
                    return tuple(results)
    return tuple(results)
