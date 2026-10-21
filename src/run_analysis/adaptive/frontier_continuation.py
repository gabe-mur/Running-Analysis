"""Bounded full-score continuation rebuilding for competing prefix calendars.

Adapt the old selector's frontier-tail refresh and multi-day suffix repair, not
its recovery model or allocator. A fixed prefix is a conditional search lane,
never a publication constraint. Every proposal is revalidated as a full physical
programme and the incoming feasible programme remains a contender.
"""


def rebuild_frontiers(frontiers,donors,*,dates,keys,hard_end,prefix_days,budget,assess,
                      compound_neighbors=None,roles=None):
    dates,keys=tuple(dates),tuple(sorted(keys))
    boundary=dates[min(prefix_days,len(dates)-1)]
    index={d:i for i,d in enumerate(dates)}
    def prefix_identity(program):
        return tuple((d,roles[k] if roles is not None else k) for d,k in program if d<boundary)
    unique=[]
    for item in frontiers:
        prefix=prefix_identity(item['physical'])
        if not any(prefix==p for p,_ in unique):
            unique.append((prefix,item))
    if not unique or budget<1:
        return dict(lanes=[],proposal_budget=budget,scope='not run')
    donor_programs=tuple(dict.fromkeys(tuple(sorted(p)) for p in donors))
    lanes=[]
    for lane_index,(prefix,seed) in enumerate(unique):
        quota=budget//len(unique)+(lane_index<budget%len(unique))
        visited={seed['physical']}
        examined=eligible=0
        best=seed
        exhausted=False
        def price(program,source):
            nonlocal examined,eligible,exhausted,best
            p=tuple(sorted(program))
            if prefix_identity(p)!=prefix:
                raise AssertionError('Conditional continuation changed its fixed prefix')
            if p in visited:
                return
            if examined>=quota:
                exhausted=True
                return
            visited.add(p)
            examined+=1
            candidate=assess(p,source)
            if candidate is not None:
                eligible+=1
                if candidate['score'].total_cost<best['score'].total_cost:
                    best=candidate
        # Rebuild the whole uncommitted suffix, then the soft projection alone.
        # Preserve all old feasible tails in the global candidate cache.
        for cut in dict.fromkeys((boundary,hard_end)):
            if cut<boundary:
                continue
            for donor in donor_programs:
                price((*[p for p in seed['physical'] if p[0]<cut],
                       *[p for p in donor if p[0]>=cut]),'frontier_tail_rebuild')
                if exhausted:
                    break
            if exhausted:
                break
        # The established selector repaired up to eight terminal days. Compare
        # each such seam against the strongest physical donors, not just one
        # newly exposed date. Eight is a compute/operator bound, not a cadence.
        for width in range(1,min(8,len(dates)-prefix_days)+1):
            cut=dates[-width]
            for donor in donor_programs[:6]:
                price((*[p for p in seed['physical'] if p[0]<cut],
                       *[p for p in donor if p[0]>=cut]),'frontier_terminal_repair')
            if exhausted:
                break
        rebuild_cost=best['score'].total_cost
        sweeps=[]
        while not exhausted:
            start=best
            before=examined
            program=start['physical']
            occupied=dict(program)
            def proposals():
                for day,key in program:
                    if day<boundary:
                        # Compare calendars, not accidentally frozen mileage.
                        # The old selector alternated fixed-calendar allocation
                        # and tail rebuilding for BOTH frontier contenders.
                        if roles is not None:
                            for other in keys:
                                if other!=key and roles[other]==roles[key]:
                                    yield tuple((d,other if d==day else k) for d,k in program)
                        continue
                    for other in keys:
                        if other!=key:
                            yield tuple((d,other if d==day else k) for d,k in program)
                    yield tuple(p for p in program if p[0]!=day)
                    for step in (-1,1):
                        i=index[day]+step
                        if 0<=i<len(dates) and dates[i]>=boundary and dates[i] not in occupied:
                            yield tuple((dates[i] if d==day else d,k) for d,k in program)
                for day in dates:
                    if day>=boundary and day not in occupied:
                        for key in keys:
                            yield (*program,(day,key))
            for p in proposals():
                price(p,'frontier_tail_coordinate')
                if exhausted:
                    break
            phase='coordinate'
            if not exhausted and best is start and compound_neighbors is not None:
                phase='joint'
                for p in compound_neighbors(start):
                    if prefix_identity(p)!=prefix:
                        continue
                    price(p,'frontier_tail_joint')
                    if exhausted:
                        break
            sweeps.append(dict(cost=start['score'].total_cost,best_cost=best['score'].total_cost,
                proposals=examined-before,complete=not exhausted,phase=phase))
            if best is start:
                break
        lanes.append(dict(prefix=prefix,boundary=boundary,seed_cost=seed['score'].total_cost,
            prefix_identity='dates_and_roles_doses_free' if roles is not None else 'exact_physical',
            rebuild_cost=rebuild_cost,best_cost=best['score'].total_cost,program=best['physical'],
            proposal_budget=quota,proposals=examined,eligible=eligible,sweeps=sweeps,
            status='budget_exhausted' if exhausted else
                'conditional_joint_complete' if compound_neighbors is not None else 'conditional_coordinate_complete'))
    return dict(lanes=lanes,proposal_budget=budget,
        scope='recombined full/soft/terminal suffixes then conditional coordinate/compound search; not global optimality')
