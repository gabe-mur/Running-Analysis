"""Reuse established calendar/dose refiners with the continuous evaluator.

This adapts representations, not legacy recovery models or score units. Shared
diversity retention, full-score calendar/dose refiners and joint neighborhoods
are preserved. Complete starts and bounded prefix construction use the current
objective; full prior programmes survive as revalidated candidates.
"""
from hashlib import sha256
import json
from math import isfinite
from time import perf_counter
from types import SimpleNamespace
from collections import Counter
from collections import deque
from datetime import date, timedelta
from run_analysis.calendar_diversity import retain_diverse
from run_analysis.adaptive.decimal_dose_search import decimal_pair_neighbors, decimal_merge_neighbors,quality_easy_neighbors
from run_analysis.adaptive.program_recombination import recombinations

from run_analysis.adaptive import full_score_refinement, finalized_distance_search
from run_analysis.adaptive.frontier_continuation import rebuild_frontiers
from run_analysis.adaptive.calendar_operators import (
    calendar_neighbors, opening_opportunity_neighbors, paired_move_neighbors,
    nearby_prescription_neighbors, equal_volume_neighbors,
)


class NoEligibleProgram(ValueError):
    """The bounded search found no admitted programme; not proof none exists."""


def canonical(program):
    result = tuple(sorted((d,k) for d,k in program))
    if len(dict(result))!=len(result):
        raise ValueError('A physical programme cannot contain two runs on one date')
    return result


def fingerprint(program):
    return sha256(json.dumps(canonical(program),separators=(',',':')).encode()).hexdigest()


def rebase_edit(origin, alternative, current):
    """Apply an earlier edit only if every touched date still matches origin.

    Empty dates are preconditions too. Never overwrite a newly placed run or
    undo a changed dose. Compatibility is not eligibility or a reusable score.
    """
    source,target,now=(dict(canonical(p)) for p in (origin,alternative,current))
    touched={d for d in source.keys()|target.keys() if source.get(d)!=target.get(d)}
    if not touched or any(now.get(d)!=source.get(d) for d in touched):
        return None
    for day in touched:
        now.pop(day,None)
        if day in target:
            now[day]=target[day]
    return canonical(now.items())


def interleave_proposals(streams):
    """Deterministic round-robin physical proposals with source attribution.

    Fairness applies to supplied streams, not suboperators nested inside them.
    Deduplication is physical, not source-specific.
    """
    pending=deque(iter(stream) for stream in streams)
    seen=set()
    while pending:
        stream=pending.popleft()
        item=next(stream,None)
        if item is None:
            continue
        pending.append(stream)
        program,source=item
        if program not in seen:
            seen.add(program)
            yield program,source


def search(*,dates,keys,previous,evaluate,budget,hard_end,roles,seed_programs=(),
           stability_tolerance=None,stable_days=7,starts=3,seed_keys=None,seed_width=6,
           previous_end=None,frontier_repair=False,dose_miles=None,exploration_keys=None,
           recombination_dates=(),complete_first=False,final_winner_budget=0,fair_coordinates=False,
           program_transform=None,allowed_role_dates=None):
    """Shared multi-start calendar refinement + pre/post easy-dose refinement.

    All costs/eligibility come from `evaluate`, with one decision-local physical
    cache including tail dates and dose IDs. None means no calibrated continuity
    allowance; exact ties may still prefer the prior/stable programme. Never
    transfer the previous model's numerical tolerance into a new objective.
    Frontier rebuilding is an explicit research experiment until matched replay
    demonstrates benefit; it is not a product setting or a stability claim.
    `evaluate` may explicitly return `construction_eligible` for physically valid
    unfinished prefixes lacking final funding. It never authorizes publication;
    complete-program admission and every refinement still require `eligible`.
    `complete_first` is a shared search-allocation experiment, not a model or
    race rule: bound independent starts and skip prefix construction only after
    a supplied complete candidate passes the SAME final admission check. The
    full beam remains the fallback. No valid calendar is forbidden by this cap.
    `allowed_role_dates` filters explicit calendar constraints before spending
    model evaluations; the full evaluator must also enforce them at publication.
    `final_winner_budget` is an explicit ADDITIONAL full-score pair allowance
    after original discovery/recombination. The maximum total is budget plus
    that allowance; never describe it as being inside the original budget.
    `fair_coordinates` interleaves occupied-date swaps and adds exact-volume
    quality/easy exchanges within the SAME budget. It does not change a score,
    workout's validity, or certify a partially examined neighborhood.
    Optional `program_transform` binds coaching doses BEFORE cache identity,
    evaluation and selection. It must be idempotent and preserve dates/roles.
    Absent preserves the existing physical search unchanged.
    """
    if budget<2 or starts<1 or not dates or len(set(dates))!=len(dates):
        raise ValueError('Positive budgets and unique ordered dates required')
    if tuple(dates)!=tuple(sorted(dates)) or set(keys)!=set(roles):
        raise ValueError('Ordered dates and a role for every dose required')
    if stability_tolerance is not None and (not isfinite(stability_tolerance) or stability_tolerance<0):
        raise ValueError('Explicit finite nonnegative new-score tolerance required')
    if (type(frontier_repair) is not bool or type(complete_first) is not bool
            or type(stable_days) is not int or stable_days<1):
        raise ValueError('Explicit frontier-repair switch and positive prefix search span required')
    if type(final_winner_budget) is not int or final_winner_budget<0:
        raise ValueError('Explicit nonnegative final-winner evaluation allowance required')
    if type(fair_coordinates) is not bool:
        raise ValueError('Explicit fair-coordinate search switch required')
    if program_transform is not None and not callable(program_transform):
        raise ValueError('Explicit callable physical-program transform required')
    allowed_role_dates = {} if allowed_role_dates is None else {
        role:frozenset(days) for role,days in allowed_role_dates.items()}
    if any(role not in set(roles.values()) or not days <= set(dates)
           for role,days in allowed_role_dates.items()):
        raise ValueError('Role date restrictions must belong to the search domain')
    def transformed(program):
        original=canonical(program)
        if program_transform is None:
            return original
        bound=canonical(program_transform(original))
        if any(k not in roles for _,k in (*original,*bound)) or (
                tuple((d,roles[k]) for d,k in bound)!=tuple((d,roles[k]) for d,k in original)):
            raise ValueError('Program transform must preserve dates and roles within the supplied domain')
        if canonical(program_transform(bound))!=bound:
            raise ValueError('Program transform must be idempotent')
        return bound
    total_budget=budget+final_winner_budget
    from run_analysis.planner_observation import budget_progress
    budget_progress(0,total_budget)
    stamp = perf_counter()
    exploration_budget=min(3000,budget-budget//2)
    dates,keys = tuple(dates),tuple(sorted(keys))
    # Callers supply scoring boundaries explicitly; representation-only/legacy
    # controls retain their existing search rather than inventing a new seam.
    cuts={hard_end,*recombination_dates} if recombination_dates else set()
    if cuts and stable_days<len(dates):
        cuts.add(dates[stable_days])
    if previous_end is not None:
        shift=date.fromisoformat(dates[-1])+timedelta(days=1)-date.fromisoformat(previous_end)
        # Compare both current and previous scoring seams. This is coverage,
        # not a change to how the horizon is priced or which prefix may move.
        cuts.update((date.fromisoformat(c)-shift).isoformat() for c in recombination_dates)
    cuts=tuple(sorted(c for c in cuts if dates[0]<c<=dates[-1]))
    coarse = keys if exploration_keys is None else tuple(sorted(set(exploration_keys)))
    if not coarse or not set(coarse) <= set(keys):
        raise ValueError('Calendar representatives must be supported dose keys')
    seed_programs=tuple(transformed(p) for p in seed_programs)
    index = {d:i for i,d in enumerate(dates)}
    prior = transformed((d,k) for d,k in previous if d in index)
    if any(k not in roles for _,k in prior):
        raise ValueError('Prior dose missing from catalogue; explicit migration required')
    stable_dates = set(dates[:stable_days])
    wanted = {d:roles[k] for d,k in prior if d in stable_dates}
    cache,admitted,constructible = {},{},{}
    denied = 0
    new_by_source, hits_by_source = Counter(), Counter()
    # Validate metadata before spending any evaluator calls. No guessing miles
    # from key names; representation-only callers may omit this neighborhood.
    if dose_miles is not None:
        list(equal_volume_neighbors({}, (), roles, dose_miles))

    def assess(program,source,*,construction=False):
        nonlocal denied
        program = transformed(program)
        if any(d not in index or k not in roles for d,k in program):
            raise ValueError('Proposal outside the current decision domain')
        if allowed_role_dates and any(roles[k] in allowed_role_dates and
                d not in allowed_role_dates[roles[k]] for d,k in program):
            return None  # Impossible user calendars need no expensive load evaluation.
        if program in cache:
            hits_by_source[source] += 1
            return (constructible if construction else admitted).get(program)
        if len(cache)>=total_budget:
            denied += 1
            return None
        result = evaluate(program)
        new_by_source[source] += 1
        cache[program] = result
        budget_progress(len(cache),total_budget)  # Display-only, identical budget/cache.
        # Final funding is not a physical constraint on an unfinished prefix.
        # Callers must explicitly distinguish that case; never infer that an
        # arbitrary rejected program (health, availability, etc.) can be built on.
        can_construct=result.get('construction_eligible',result['eligible'])
        if not (result['eligible'] or can_construct) or not isfinite(result['cost']):
            return None
        hard = canonical((d,k) for d,k in program if d<hard_end)
        tail = canonical((d,k) for d,k in program if d>=hard_end)
        item = dict(days=hard,program=(program,),physical=program,result=result,
            score=SimpleNamespace(total_cost=result['cost'],offsets=tuple(index[d] for d,_ in hard)),
            stable=bool(prior) and {d:roles[k] for d,k in program if d in stable_dates}==wanted,
            tail_fingerprint=fingerprint(tail),
            row=dict(source=source,hard_fingerprint=fingerprint(hard)))
        if result['eligible']:
            admitted[program] = item
        if can_construct:
            constructible[program] = item
        return (constructible if construction else admitted).get(program)

    # Always revalidate the incumbent before any independent exploration.
    incumbent = assess(prior,'publication_seed' if prior else 'empty_seed')
    assess((),'empty_seed')
    # A rest-day/weekday edit invalidates a SLOT, not the usefulness of the
    # entire prior programme. Offer bounded local repairs inside the same budget.
    # No forecast-earned credit, protected winner or altered score is involved.
    from .calendar_constraint_repair import constraint_repair_neighbors
    constraint_start=len(cache)
    constraint_limit=min(budget,len(cache)+min(32,max(1,budget//20)))
    repaired=[]
    if prior and allowed_role_dates:
        for candidate in constraint_repair_neighbors(prior,dates,roles,allowed_role_dates):
            if candidate not in cache and len(cache)>=constraint_limit:
                break
            item=assess(candidate,'prior_constraint_repair')
            if item is not None:
                repaired.append(item)
    constraint_evaluations=len(cache)-constraint_start
    # A rolled horizon must compare extending the issued programme with replacing
    # it. Last scheduled run is NOT the old horizon: scheduled rest is meaningful.
    # Price all single insertions on newly exposed dates before frontier pruning.
    # This is candidate coverage, not a score preference or calendar lock. Multiple
    # new dates still receive joint refinement; these seeds are not an exhaustive
    # suffix optimizer.
    extension_start=len(cache)
    extension_dates=tuple(d for d in dates if previous_end is not None and d>=previous_end)
    if prior and previous_end is not None:
        occupied=dict(prior)
        for day in extension_dates:
            if day not in occupied:
                for key in coarse:
                    assess((*prior,(day,key)),'prior_horizon_extension')
    extension_evaluations=len(cache)-extension_start
    # A newly rejected prior must not disappear before cheap, scored tail-only
    # repairs are compared. Exact-volume merges also make room for legitimate
    # new endpoint work. These are candidates, not locks or admission bypasses.
    repair_start=len(cache)
    repair_limit=min(budget,len(cache)+min(256,budget//20))
    if prior and dose_miles is not None:
        repairs=tuple(decimal_merge_neighbors(prior,roles,dose_miles,index=index,minimum_date=hard_end))
        def repair_stream():
            for p in repairs:
                yield p,'prior_tail_merge'
        def extended_repair_stream():
            for p in repairs:
                for day in extension_dates:
                    if day not in dict(p):
                        for key in coarse:
                            if roles[key] in {'long','quality'}:
                                yield canonical((*p,(day,key))),'prior_tail_merge_extension'
        for p,source in interleave_proposals((repair_stream(),extended_repair_stream())):
            if p not in cache and len(cache)>=repair_limit:
                break
            assess(p,source)
    repair_evaluations=len(cache)-repair_start
    seed_cap=min(512,max(1,budget//20)) if complete_first else len(seed_programs)
    examined_seeds=0
    supplied_admitted=False
    for program in seed_programs:
        # Bound the supplied library by its deterministic ordering, not cache
        # misses: a matching incumbent must not reveal an extra independent
        # start and change coverage solely by consuming one fewer call.
        if complete_first and examined_seeds>=seed_cap:
            break
        supplied_admitted=assess(program,'supplied_independent_seed') is not None or supplied_admitted
        examined_seeds+=1
    # Build whole-date-domain calendars with the established half-score/half-
    # diversity beam rule. Scores are the same full physical objective; absent
    # suffix work is explicit rest, not an old recovery or allocation proxy.
    # Reserve the majority of compute for existing calendar/dose refinements.
    beam_keys=tuple(seed_keys) if seed_keys is not None else coarse
    if not set(beam_keys)<=set(keys) or not beam_keys or seed_width<1:
        raise ValueError('Explicit supported beam representatives and positive width required')
    beam_limit=len(cache)+max(0,exploration_budget//3)
    # Chronological construction owns only a processed prefix. Complete seeds
    # remain in `admitted` for the full-calendar refiners below, never here.
    empty=assess((),'empty_construction_prefix',construction=True)
    beam=[empty] if empty is not None else []
    completed_depth=0
    def diversity(item):
        p=item['physical']
        offsets=tuple(index[d] for d,_ in p)
        gaps=tuple(b-a for a,b in zip(offsets,offsets[1:]))
        counts=tuple(sorted(Counter(roles[k] for _,k in p).items()))
        return (len(p),counts,offsets[0] if offsets else None,offsets[-1] if offsets else None,
                sum(g==1 for g in gaps),max(gaps,default=0),gaps[-2:])
    skip_beam=complete_first and supplied_admitted
    for day in (() if skip_beam else dates):
        expanded={c['physical']:c for c in beam}  # Explicit rest branch.
        complete=True
        for parent in beam:
            for key in beam_keys:
                p=canonical((*parent['physical'],(day,key)))
                if len(cache)>=beam_limit and p not in cache:
                    complete=False
                    break
                item=assess(p,'diverse_calendar_beam',construction=True)
                if item is not None:
                    expanded[p]=item
            if not complete:
                break
        if not expanded:
            break
        beam=retain_diverse(list(expanded.values()),seed_width,
            rank=lambda c:(c['score'].total_cost,c['physical']),identity=lambda c:c['physical'],signature=diversity)
        if not complete:
            break
        completed_depth+=1
    beam_diagnostics=dict(width=seed_width,representatives=beam_keys,completed_dates=completed_depth,
        total_dates=len(dates),complete=completed_depth==len(dates),
        retained=[c['physical'] for c in beam],budget_limit=beam_limit,
        construction_only_evaluations=sum(p not in admitted for p in constructible),
        complete_seed_fallback=False)
    beam_diagnostics.update(skipped=skip_beam,
        skip_reason='admitted_complete_start_available' if skip_beam else None)
    # Transparent cold-start anchors, not a replacement for the old beam's seed
    # diversity. Downstream refinement is the actual existing implementation.
    for key in coarse:
        assess(((dates[0],key),),'cold_start_anchor')
    if complete_first and not admitted:
        # Admission may depend on information unavailable to a cheap prefix.
        # A capped opening library must not cause a false no-plan result when
        # later supplied complete starts are viable. Retain the original budget.
        for program in seed_programs[examined_seeds:]:
            if len(cache)>=budget:
                break
            assess(program,'complete_seed_fallback')
        beam_diagnostics['complete_seed_fallback']=True
    if not admitted:
        raise NoEligibleProgram('No revalidated feasible seed; no unscored fallback')
    if not prior:
        min(admitted.values(),key=lambda c:(c['score'].total_cost,c['physical']))['row']['source']='publication_seed'

    def dose_neighbors(current):
        if dose_miles is not None and exploration_keys is not None:
            yield from decimal_pair_neighbors(current['days'],roles,dose_miles,index=index)
        for day,key in current['days']:
            if roles[key]!='easy':
                continue
            for other in keys:
                if other!=key and roles[other]=='easy':
                    yield canonical((d,other if d==day else k) for d,k in current['days'])

    def price(seed):
        if 'physical' in seed:
            return assess(seed['physical'],seed['source'])
        parent = seed['tail_parent']
        tail = tuple((d,k) for d,k in parent['physical'] if d>=hard_end)
        return assess((*seed['days'],*tail),seed['source'])

    def physical_neighbors(current,allowed):
        program = current['physical']
        occupants = {index[d]:k for d,k in program}
        offsets = tuple(sorted(occupants))
        pool = set(calendar_neighbors(offsets,allowed))
        pool.update(opening_opportunity_neighbors(offsets,allowed))
        pool.update(paired_move_neighbors(offsets,allowed))
        seen = set()
        def proposal(p):
            p = canonical(p)
            if p not in seen and p!=program:
                seen.add(p)
                # Includes exact hard allocation AND soft suffix. Same dates
                # with different doses/tails must never share a cached score.
                return (fingerprint(p),dict(physical=p,source='full_score_neighbor'))
            return None
        for proposed in sorted(pool):
            removed = sorted(set(offsets)-set(proposed))
            added = sorted(set(proposed)-set(offsets))
            retained = {i:occupants[i] for i in proposed if i in occupants}
            # Pair the existing moved doses in temporal order; insertions try
            # all catalogue roles rather than letting old recovery heuristics
            # choose a different physical workout after it was scored.
            moved = min(len(removed),len(added))
            for old,new in zip(removed[:moved],added[:moved]):
                retained[new]=occupants[old]
            unassigned = added[moved:]
            if len(unassigned)>1:
                raise AssertionError('Existing calendar operators add at most one unpaired run')
            variants = ({**retained,unassigned[0]:key} for key in coarse) if unassigned else (retained,)
            for variant in variants:
                row = proposal((dates[i],k) for i,k in variant.items())
                if row:
                    yield row
        for proposed in nearby_prescription_neighbors(occupants,allowed):
            row=proposal((dates[i],k) for i,k in proposed)
            if row:
                yield row
        # Role/dose menu belongs to the new representation; calendar movement
        # and the lane/descent implementation above are reused, not rewritten.
        for day,key in program:
            for other in coarse:
                if other!=key:
                    row = proposal((d,other if d==day else k) for d,k in program)
                    if row:
                        yield row

    def distance_pass(limit,*,all_seeds=False):
        if limit<1:
            return None
        candidates = list(admitted.values())
        selected=candidates
        if all_seeds:
            # One proposal on each of a thousand starts is also starvation.
            # Spend meaningful effort on a bounded full-score/shape shortlist,
            # using the same diversity rule as the established calendar beam.
            selected=retain_diverse(candidates,starts*2,
                rank=lambda c:(c['score'].total_cost,c['physical']),
                identity=lambda c:c['physical'],
                signature=lambda c:tuple((d,roles[k]) for d,k in c['physical']))
            if incumbent is not None and all(c['physical']!=incumbent['physical'] for c in selected):
                selected.append(incumbent)
        trace=finalized_distance_search.refine_distances(candidates,price,{},None,budget=limit,
            visible_days=sum(d<hard_end for d in dates),
            seed_candidates=selected if all_seeds else None,
            max_sweeps=1 if all_seeds else None,
            neighbor_adapter=dose_neighbors,identity_adapter=fingerprint)
        trace.update(candidate_pool_count=len(candidates),
                     lane_selection='score_and_calendar_diversity' if all_seeds else 'selection_frontier')
        return trace

    # Keep most of the evaluator budget for final physical-neighborhood work.
    # Separate calendar and distance passes cannot certify one another after
    # either changes the programme. This is compute allocation, not a score cost.
    exploration_limit=max(len(cache),exploration_budget)
    # Preserve the established order: price easy allocations before calendar
    # pruning; refine independent/stable frontier allocations again afterward.
    opening_trace=dict(strategy='complete_first' if complete_first else 'prefix_beam',
        seed_limit=seed_cap,supplied_seed_count=len(seed_programs),examined_seeds=examined_seeds,
        evaluations_before_refinement=len(cache),seconds_before_refinement=perf_counter()-stamp,
        scope='Search allocation only; no score preference, validity restriction or global certificate')
    before = distance_pass(min(1000,max(0,exploration_limit-len(cache))//3),all_seeds=True)
    remaining = max(0,exploration_limit-len(cache))
    # Split the actual remaining exploration allowance. Subtracting another
    # fraction of the TOTAL cap used to starve calendar comparison at 10k.
    calendar_budget = (2*remaining + 2)//3
    calendar = None
    if calendar_budget:
        calendar = full_score_refinement.refine(list(admitted.values()),price,set(index.values()),
            budget=calendar_budget,starts=starts,proposal_adapter=physical_neighbors,
            seed_identity_adapter=lambda c:fingerprint(c['physical']))
    post_calendar_candidate=None
    if frontier_repair:
        calendar_candidates=[c for c in admitted.values() if c['row']['source']=='full_score_neighbor']
        allocation_budget=min(budget//20,max(0,exploration_limit-len(cache)))
        if calendar_candidates and allocation_budget:
            neighbor=min(calendar_candidates,key=lambda c:(c['score'].total_cost,c['physical']))
            post_calendar_candidate=finalized_distance_search.refine_distances([neighbor],price,{},None,
                budget=allocation_budget,visible_days=sum(d<hard_end for d in dates),seed_candidates=[neighbor],
                neighbor_adapter=dose_neighbors,identity_adapter=fingerprint)
    after = distance_pass(max(0,exploration_limit-len(cache)))
    recombination_trace=[]
    def recombination_pool():
        ordered=sorted(admitted.values(),key=lambda c:(c['score'].total_cost,c['physical']))
        # Reuse the established half-score/half-shape retention. The prior and
        # stable contender are comparators only; neither receives a score bonus.
        pool=retain_diverse(ordered,starts,rank=lambda c:(c['score'].total_cost,c['physical']),
            identity=lambda c:c['physical'],
            signature=lambda c:tuple((d,roles[k]) for d,k in c['physical']))
        stable_front=next((c for c in ordered if c['stable']),None)
        pool=[*pool,*([incumbent] if incumbent is not None else []),
              *([stable_front] if stable_front is not None else [])]
        parents=tuple(dict.fromkeys(c['physical'] for c in pool))
        return ordered,parents
    def recombine(limit):
        ordered,parents=recombination_pool()
        before_calls=len(cache); before_cost=ordered[0]['score'].total_cost
        examined=0; complete=True
        for p,origin in recombinations(parents,cuts):
            if p not in cache and len(cache)>=limit:
                complete=False
                break
            assess(p,'prefix_suffix_recombination')
            examined+=1
        winner=min(admitted.values(),key=lambda c:(c['score'].total_cost,c['physical']))
        recombination_trace.append(dict(cuts=cuts,parents=[fingerprint(p) for p in parents],
            examined=examined,evaluations=len(cache)-before_calls,complete=complete,
            before_cost=before_cost,after_cost=winner['score'].total_cost))
        return winner
    # Bound work by actual pair/cut count as well as the unchanged global cap.
    # Reserving a final batch prevents ordinary descent from starving this check.
    recombination_cap=(starts+2)*(starts+1)*len(cuts)
    recombination_reserve=min(recombination_cap,budget//20,max(0,budget-len(cache)))
    main_budget=budget-recombination_reserve
    frontier=None
    def refresh_continuations():
        ordered=sorted(admitted.values(),key=lambda c:(c['score'].total_cost,c['physical']))
        stable_front=next((c for c in ordered if c['stable']),None)
        frontiers=[ordered[0],*([stable_front] if stable_front is not None else [])]
        return rebuild_frontiers(frontiers,
            [*[c['physical'] for c in ordered[:6]],*[c['physical'] for c in beam],*seed_programs],
            dates=dates,keys=keys,hard_end=hard_end,prefix_days=stable_days,
            budget=min(4000,budget//2,max(0,main_budget-len(cache))),assess=assess,
            roles=roles,
            compound_neighbors=lambda c:(seed['physical'] for _,seed in physical_neighbors(c,set(index.values()))))
    independent = min(admitted.values(),key=lambda c:(c['score'].total_cost,c['physical']))
    sweeps=[]
    certified=None
    post_joint_calendar=None
    refresh_pending=frontier_repair and len(dates)>stable_days
    # Settle the contenders before refreshing their tails. Do not interrupt
    # baseline descent/certification to reserve speculative continuation work:
    # that can spend the same budget without finishing either search. Use only
    # the remaining budget, then re-certify the final winner. Doing
    # this before joint search can compare a single prefix and miss the changed
    # contender discovered later in the same decision.
    final_check_reserve=min(4000,budget//4,max(0,(main_budget-len(cache))//2)) if dose_miles is not None else 0
    final_check_pending=final_check_reserve>0
    compound_reserve=min(3000,budget//3,max(0,(main_budget-len(cache)-final_check_reserve)//2)) if dose_miles is not None else 0
    compound_pending=compound_reserve>0
    joint_limit=main_budget-compound_reserve-final_check_reserve
    compound_trace=[]
    reuse_trace=[]

    def reuse_edits(start,winner,pool,limit):
        before_calls=len(cache)
        before_cost=winner['score'].total_cost
        examined=conflicts=accepted=0
        # Each adoption can change every remaining edit's score. Rebase from
        # the original preconditions, then assess the entire new physical plan.
        while len(cache)<limit:
            best=winner
            for alternative in pool:
                p=rebase_edit(start['physical'],alternative['physical'],winner['physical'])
                if p is None:
                    conflicts+=1
                    continue
                if p not in cache and len(cache)>=limit:
                    break
                candidate=assess(p,'rebased_promising_edit')
                examined+=1
                if candidate is not None and candidate['score'].total_cost<best['score'].total_cost:
                    best=candidate
            if best is winner:
                break
            winner=best
            accepted+=1
        reuse_trace.append(dict(origin=fingerprint(start['physical']),
            candidates=len(pool),examined=examined,conflicts=conflicts,accepted=accepted,
            evaluations=len(cache)-before_calls,before_cost=before_cost,
            after_cost=winner['score'].total_cost,program=winner['physical']))
        return winner

    def redistribution_neighbors(current):
        if dose_miles is not None:
            # Fine paired allocation and coarse split/merge are separate streams;
            # a dense catalogue must not create a cubic triple-dose enumeration.
            selected=set(coarse)|{k for _,k in current['physical']}
            coarse_stream=(canonical((dates[i],k) for i,k in proposed)
                for proposed in equal_volume_neighbors(
                    {index[d]:k for d,k in current['physical']},index.values(),
                    {k:roles[k] for k in selected},{k:dose_miles[k] for k in selected}))
            fine_stream=decimal_pair_neighbors(current['physical'],roles,dose_miles,index=index)
            merge_stream=decimal_merge_neighbors(current['physical'],roles,dose_miles,index=index)
            streams=(merge_stream,coarse_stream) if exploration_keys is None else (merge_stream,fine_stream,coarse_stream)
            if fair_coordinates:
                streams=(*streams,quality_easy_neighbors(current['physical'],roles,dose_miles,index=index))
            for p,_ in interleave_proposals(tuple(((p,'redistribution') for p in stream) for stream in streams)):
                yield p

    def protected_redistribution(current):
        # Bounded best-improvement batches, then re-open ordinary search. A
        # partial batch is NEVER certified. Cache hits spend no global budget.
        limit=min(main_budget-final_check_reserve,len(cache)+compound_reserve)
        while len(cache)<limit:
            start=best=current
            calls=len(cache)
            batch_limit=min(limit,calls+500)
            complete=True
            examined=0
            for p in redistribution_neighbors(start):
                if p not in cache and len(cache)>=batch_limit:
                    complete=False
                    break
                candidate=assess(p,'protected_redistribution')
                examined+=1
                if candidate is not None and candidate['score'].total_cost<best['score'].total_cost:
                    best=candidate
            compound_trace.append(dict(fingerprint=fingerprint(start['physical']),
                cost=start['score'].total_cost,best_cost=best['score'].total_cost,
                complete=complete,examined=examined,evaluations=len(cache)-calls))
            current=best
            if best is start and complete:
                break
        return current
    def cheap_neighbors(current):
        # Compare a full bounded coordinate neighborhood before accepting its
        # best improvement. A tiny early gain must not restart the entire scan.
        program=current['physical']
        occupied=dict(program)
        if fair_coordinates:
            # Same coordinate universe, fair bounded coverage: check cheap
            # movement/deletion first, then interleave same-role native swaps
            # across ALL occupied dates instead of exhausting one early date.
            def shape():
                for day,key in program:
                    for step in (-1,1):
                        i=index[day]+step
                        if 0<=i<len(dates) and dates[i] not in occupied:
                            yield canonical((dates[i] if d==day else d,k) for d,k in program)
                    yield canonical((d,k) for d,k in program if d!=day)
            def swaps(day,key,same):
                for other in keys:
                    if other!=key and (roles[other]==roles[key])==same:
                        yield canonical((d,other if d==day else k) for d,k in program)
            same=((p,'same_role') for p,_ in interleave_proposals(tuple(
                ((p,'swap') for p in swaps(day,key,True)) for day,key in program)))
            different=((p,'different_role') for p,_ in interleave_proposals(tuple(
                ((p,'swap') for p in swaps(day,key,False)) for day,key in program)))
            for p,_ in interleave_proposals((((p,'shape') for p in shape()),same,different)):
                yield p
        else:
            yield from sequential_coordinates(program,occupied)
        for day in dates:
            if day not in occupied:
                for key in coarse:
                    yield canonical((*program,(day,key)))
        for proposed in nearby_prescription_neighbors({index[d]:k for d,k in program},index.values()):
            yield canonical((dates[i],k) for i,k in proposed)

    def sequential_coordinates(program,occupied):
        for day,key in program:
            for step in (-1,1):
                i=index[day]+step
                if 0<=i<len(dates) and dates[i] not in occupied:
                    yield canonical((dates[i] if d==day else d,k) for d,k in program)
            for other in keys:
                if other!=key:
                    yield canonical((d,other if d==day else k) for d,k in program)
            yield canonical((d,k) for d,k in program if d!=day)
    while True:
        checking=final_check_pending and not compound_pending and (len(cache)>=joint_limit)
        start=independent
        best=start
        before_calls=len(cache)
        complete=True
        examined=0
        phase='coordinate'
        if checking:
            phase='final_check'
            joint_limit=main_budget
        reuse_allowance=min(32,max(0,(joint_limit-len(cache))//10))
        scan_limit=joint_limit-reuse_allowance
        promising={}
        def remember(candidate):
            if candidate is not None and candidate['score'].total_cost<start['score'].total_cost:
                promising[candidate['physical']]=candidate
                if len(promising)>8:
                    worst=max(promising.values(),key=lambda c:(c['score'].total_cost,c['physical']))
                    promising.pop(worst['physical'])
        def scan_exhausted(p):
            # Reserve reuse calls only after an actual improvement is found;
            # an unchanged contender can spend the whole allowance certifying.
            return p not in cache and len(cache)>=(scan_limit if best is not start else joint_limit)
        def round_robin_check():
            streams=(
                ((p,'reserved_coordinate') for p in cheap_neighbors(start)),
                ((seed['physical'],'reserved_calendar') for _,seed in physical_neighbors(start,set(index.values()))),
                ((p,'reserved_redistribution') for p in redistribution_neighbors(start)))
            yield from interleave_proposals(streams)
        for p,source in round_robin_check() if checking else ((p,'final_coordinate') for p in cheap_neighbors(start)):
            if scan_exhausted(p):
                complete=False
                break
            candidate=assess(p,source)
            remember(candidate)
            examined+=1
            if candidate is not None and candidate['score'].total_cost<best['score'].total_cost:
                best=candidate
        if complete and best is start and not checking:
            phase='joint'
        for _,seed in physical_neighbors(start,set(index.values())) if phase=='joint' else ():
            p=seed['physical']
            if scan_exhausted(p):
                complete=False
                break
            candidate=price(seed)
            remember(candidate)
            examined+=1
            if candidate is not None and candidate['score'].total_cost<best['score'].total_cost:
                best=candidate
        for p in redistribution_neighbors(start) if phase=='joint' and complete else ():
            if scan_exhausted(p):
                complete=False
                break
            candidate=assess(p,'final_redistribution')
            remember(candidate)
            examined+=1
            if candidate is not None and candidate['score'].total_cost<best['score'].total_cost:
                best=candidate
        if best is not start and promising and reuse_allowance:
            pool=sorted(promising.values(),key=lambda c:(c['score'].total_cost,c['physical']))
            best=reuse_edits(start,best,pool,min(joint_limit,len(cache)+reuse_allowance))
        sweeps.append(dict(fingerprint=fingerprint(start['physical']),
            cost=start['score'].total_cost,best_cost=best['score'].total_cost,
            phase=phase,complete=complete and best is start,scan_complete=complete,
            best_program=best['physical'],best_fingerprint=fingerprint(best['physical']),
            examined=examined,evaluations=len(cache)-before_calls))
        independent=best
        if compound_pending and (not complete or best is start):
            independent=protected_redistribution(independent)
            compound_pending=False
            joint_limit=main_budget-final_check_reserve
            continue  # Re-check every operator against the new winner.
        if refresh_pending and (not complete or best is start):
            # A truncated coarse calendar phase may miss an alternative that
            # final search discovers. Give the
            # best DIFFERENT hard calendar its own allocation lane here too,
            # before its poor starting doses can eliminate it from the frontier.
            def hard_calendar(c):
                return tuple((d,roles[k]) for d,k in c['physical'] if d<hard_end)
            alternatives=[c for c in admitted.values() if hard_calendar(c)!=hard_calendar(independent)]
            allocation_budget=min(500,budget//20,max(0,main_budget-len(cache)))
            if alternatives and allocation_budget:
                neighbor=min(alternatives,key=lambda c:(c['score'].total_cost,c['physical']))
                post_joint_calendar=finalized_distance_search.refine_distances([neighbor],price,{},None,
                    budget=allocation_budget,visible_days=sum(d<hard_end for d in dates),seed_candidates=[neighbor],
                    neighbor_adapter=dose_neighbors,identity_adapter=fingerprint)
            frontier=refresh_continuations()
            refresh_pending=False
            joint_limit=main_budget-final_check_reserve
            independent=min(admitted.values(),key=lambda c:(c['score'].total_cost,c['physical']))
            continue
        if not complete:
            if final_check_pending and not checking:
                # Spend the reserved calls across every operator family, not
                # another coordinate-only descent. No guarantee of completion.
                joint_limit=len(cache)
                continue
            break
        if best is start:
            certified=fingerprint(start['physical'])
            break
        if checking:
            # The check found a better physical program. Its certificate cannot
            # transfer to that new winner; recheck with whatever budget remains.
            joint_limit=len(cache)
    if cuts:
        independent=recombine(budget)
    final_pair_trace=[]
    pair_certified=None
    if final_winner_budget and dose_miles is not None:
        # Preserve discovery intact. This is a separately reported bounded
        # allowance, not capacity stolen from the original discovery trajectory.
        # Scan the WHOLE existing pair neighborhood before adopting its best
        # admitted edit: tiny early gains must not repeatedly restart a line.
        while True:
            start=best=independent
            calls=len(cache); examined=0; pair_complete=True
            streams=(((p,'final_winner_pair') for p in
                decimal_pair_neighbors(start['physical'],roles,dose_miles,index=index)),)
            if fair_coordinates:
                boundary=dates[min(stable_days,len(dates)-1)] if stable_days<len(dates) else None
                streams=(((p,'final_winner_quality_easy') for p in quality_easy_neighbors(
                    start['physical'],roles,dose_miles,index=index,priority_before=boundary)),*streams)
            if program_transform is not None:
                # Later dose refinement can reopen a previously priced calendar.
                # Recheck nearby dates against the FINAL bound prescription,
                # interleaved inside the unchanged allowance, not after scoring.
                def terminal_dates():
                    occupied=dict(start['physical'])
                    for day,key in start['physical']:
                        for delta in (-1,1,-2,2,-3,3):
                            pos=index[day]+delta
                            if 0<=pos<len(dates) and dates[pos] not in occupied:
                                yield canonical((dates[pos] if d==day else d,k)
                                    for d,k in start['physical']),'final_winner_calendar'
                streams=(terminal_dates(),*streams)
            for p,source in interleave_proposals(streams):
                # Bound aliases may already have a physical score. Do not end
                # the scan solely because an unbound alias is absent in cache.
                p=transformed(p)
                if p not in cache and len(cache)>=total_budget:
                    pair_complete=False
                    break
                candidate=assess(p,source)
                examined+=1
                if candidate is not None and candidate['score'].total_cost<best['score'].total_cost:
                    best=candidate
            final_pair_trace.append(dict(fingerprint=fingerprint(start['physical']),
                before_cost=start['score'].total_cost,after_cost=best['score'].total_cost,
                evaluations=len(cache)-calls,examined=examined,scan_complete=pair_complete,
                best_fingerprint=fingerprint(best['physical'])))
            independent=best
            if best is start:
                if pair_complete:
                    pair_certified=fingerprint(start['physical'])
                break
            if not pair_complete:
                break
    stable = min((c for c in admitted.values() if c['stable']),
                 key=lambda c:(c['score'].total_cost,c['physical']),default=None)
    allowance = stability_tolerance if stability_tolerance is not None else 0.
    winner = stable if stable is not None and stable['score'].total_cost<=independent['score'].total_cost+allowance else independent
    if incumbent is not None and incumbent['score'].total_cost==winner['score'].total_cost:
        winner=incumbent
    traces = [t for t in (before,calendar,post_calendar_candidate,after,post_joint_calendar,frontier) if t]
    exhausted = (not complete or bool(denied)
        or any(not row['scan_complete'] for row in final_pair_trace)
        or any(l['status']=='budget_exhausted' for t in traces for l in t['lanes']))
    # Do NOT turn a completed earlier sweep into a certificate for the final
    # programme: post-calendar dose/tail changes can open calendar improvements.
    return dict(program=winner['physical'],cost=winner['score'].total_cost,selected=winner['result'],
        evaluations=len(cache),iterations=sum(len(l.get('neighborhoods',())) for l in (calendar or {}).get('lanes',())),
        budget_exhausted=exhausted,local_neighborhood_complete=program_transform is None and certified==fingerprint(winner['physical']),
        convergence_scope='Supplied calendar representatives, all single catalogue doses, nearby exact-volume pairs and coarse split/merge; not global optimality',
        **({'fair_coordinates':True,'quality_easy_exchange':True} if fair_coordinates else {}),
        **({'coaching_program_transform':True,'transform_scope':'Every proposal rebound and fully repriced before selection; no transformed-neighborhood certificate'}
            if program_transform is not None else {}),
        dose_resolution=dict(catalogue=len(keys),calendar_representatives=len(coarse),
            decimal_pair_refinement=exploration_keys is not None and dose_miles is not None),
        prior_cost=incumbent['score'].total_cost if incumbent else None,seconds=perf_counter()-stamp,
        search='reused_full_score_and_finalized_distance_refiners_v8_rebased_frontier' if frontier_repair else
               'reused_full_score_and_finalized_distance_refiners_v9_decimal' if exploration_keys is not None else
               'reused_full_score_and_finalized_distance_refiners_v8_rebased',
        prior_revalidation=dict(available=bool(prior),eligible=incumbent is not None,
            physical_fingerprint=fingerprint(prior),retained_as_candidate=prior in admitted,
            previous_end=previous_end,extension_dates=extension_dates,
            extension_evaluations=extension_evaluations),
        prior_constraint_repair=dict(evaluations=constraint_evaluations,
            maximum_new_evaluations=min(32,max(1,budget//20)),
            admitted_candidates=len(repaired),
            fingerprints=tuple(fingerprint(c['physical']) for c in repaired),
            scope='Explicit date conflicts only; candidate repair, no score or continuity bonus'),
        prior_tail_repair=dict(evaluations=repair_evaluations,maximum_new_evaluations=min(256,budget//20),
            scope='Full-score exact-volume tail merges and newly exposed key-session extensions; no prefix lock'),
        continuity=dict(tolerance=stability_tolerance,status='not_calibrated' if stability_tolerance is None else 'explicit_scenario',
            independent_cost=independent['score'].total_cost,stable_cost=stable['score'].total_cost if stable else None),
        refinement=dict(opening=opening_trace,pre_calendar_distance=before,calendar=calendar,post_calendar_distance=after,
            prefix_suffix_recombination=dict(reserved=recombination_reserve,passes=recombination_trace),
            final_winner_doses=dict(additional_allowance=final_winner_budget,
                maximum_total_evaluations=total_budget,passes=final_pair_trace,
                certified_fingerprint=pair_certified,
                scope=('Nearby single-run date moves (up to three days), easy pairs and quality/easy exchanges after coaching binding; bounded partial coverage, not a global certificate'
                    if program_transform is not None else
                    'Six-day exact-volume easy pairs and native quality/easy exchanges; visible quality coverage first; not a calendar or global certificate'
                    if fair_coordinates else 'Existing six-day exact-volume easy pairs; not a calendar or global certificate')),
            post_calendar_candidate_distance=post_calendar_candidate,frontier_continuation=frontier,
            post_joint_calendar_distance=post_joint_calendar,
            final_joint=sweeps,global_budget_denials=denied,
            protected_redistribution=dict(reserved=compound_reserve,batch_size=500,passes=compound_trace),
            final_check_reserved=final_check_reserve,
            promising_edit_reuse=dict(pool_size=8,max_new_calls_per_sweep=32,passes=reuse_trace),
            global_new_by_source=dict(new_by_source),global_cache_hits_by_source=dict(hits_by_source)),
        calendar_beam=beam_diagnostics,
        migration_limits=['Calendar beam uses shared diversity retention with full-objective prefixes; no global certificate',
            'Finite representative beam menu precedes full catalogue refinement',
            'Old numerical continuity tolerance and recovery model not transferred'])
