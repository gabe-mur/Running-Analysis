"""Bounded offline easy-dose descent, conditional on fixed dates and soft tail."""
from contextlib import contextmanager
from contextvars import ContextVar
from time import perf_counter

from run_analysis import weekly_schedule as planner
from run_analysis.prescription_distance import rounded_distance_target, UnrepresentableDistanceTarget
from run_analysis.adaptive.shared_lane_scan import scan_lanes


DISTANCE_REFINEMENT = ContextVar('finalized_distance_refinement', default=None)


@contextmanager
def distance_refinement_policy(
    *, enabled=False, budget=128, visible_days=21, candidate_budget=None,
    post_calendar_budget=0,
):
    candidate_budget = budget * 3 if candidate_budget is None else candidate_budget
    if budget < 1 or candidate_budget < 1 or visible_days < 1 or post_calendar_budget < 0:
        raise ValueError('Require positive distance budget and visible days')
    token = DISTANCE_REFINEMENT.set(
        dict(
            budget=budget,
            visible_days=visible_days,
            candidate_budget=candidate_budget,
            post_calendar_budget=post_calendar_budget,
        )
        if enabled
        else None
    )
    try:
        yield
    finally:
        DISTANCE_REFINEMENT.reset(token)


def neighbors(days, inputs, state_at, *, visible_days):
    """Existing policy menus, one changed easy prescription, no proxy pruning.

    Raw states are projected through the exact prefix. The normal history-led
    bounds/options adapter must remain active. Full-program evaluation rechecks
    all later loads: menu membership alone never grants feasibility.
    """
    primary_long=next((d.recommendation for d in days if d.recommendation
        and d.recommendation.workout_type==planner.WorkoutType.LONG),None)
    ceiling=float('inf')
    if primary_long is not None:
        ordinary=sum(planner.typical_easy_distance(inputs['daily_states'][0]))/2
        margin=max(.1,round(ordinary*.15,1))
        long_miles=primary_long.planned_distance_miles
        ceiling=min(long_miles-margin,long_miles/1.15)
    planned=[]
    previous=None
    for i,day in enumerate(days[:visible_days]):
        rec=day.recommendation
        if rec is None or rec.workout_type==planner.WorkoutType.REST:
            continue
        state=planner._project_state(state_at(rec.planned_for), planned, inputs['config'])
        if rec.workout_type==planner.WorkoutType.EASY and rec.distance_range_miles:
            easy=planner.typical_easy_distance(state)
            aerobic_maximum=max(easy[1],min(easy[1]*1.75,planner.long_run_reference_miles(state)*1.05))
            low,high,preferred,_=planner._easy_allocation_bounds(state,rec,aerobic_maximum,
                i-previous if previous is not None else 3,preserve_projected_recovery_caps=True)
            choices,ranges=planner._easy_allocation_options(state,rec,low,max(low,high),
                aerobic_maximum,plan_date=day.date)
            low=min(low,min(choices)); high=max(high,max(choices))
            for choice in sorted(set(choices)):
                if choice>ceiling+1e-9:
                    continue
                try:
                    target=rounded_distance_target(choice,minimum=low,maximum=high)
                except UnrepresentableDistanceTarget:
                    continue
                if target>ceiling+1e-9:
                    continue
                bounds=ranges.get(choice)
                if bounds is None:
                    raw_low=max(.1,choice-.25); raw_high=max(raw_low,2*choice-raw_low)
                    bounds=(round(raw_low,1),round(raw_high,1))
                    if raw_high-raw_low>=.5-1e-9 and bounds[1]-bounds[0]<.5-1e-9:
                        bounds=(round(max(.1,bounds[1]-.5),1),bounds[1])
                if not bounds[0]<=target<=bounds[1]:
                    continue
                # Keep an identical prescription truly identical, including its
                # existing role/title. Do not introduce metadata-only churn.
                if target==rec.planned_distance_miles and tuple(bounds)==tuple(rec.distance_range_miles):
                    continue
                medium=target>planner._ordinary_easy_expansion_reference(state)+1e-9
                revised=rec.model_copy(update=dict(distance_target_miles=target,
                    distance_range_miles=tuple(bounds),planning_role=(
                        'medium_long' if medium else 'support_easy' if target<preferred-1e-9 else 'ordinary_easy'),
                    title='Medium-long aerobic run' if medium else 'Easy aerobic run'))
                candidate=list(days)
                candidate[i]=day.model_copy(update=dict(recommendation=revised,
                    day_role='medium_long_run' if medium else 'easy_run'))
                yield candidate
        planned.append(rec)
        previous=i


def refine_distances(
    candidates,
    evaluate,
    inputs,
    state_at,
    *,
    budget=128,
    visible_days=21,
    seed_candidates=None,
    max_sweeps=None,
    neighbor_adapter=None,
    identity_adapter=None,
):
    """Best improvement per complete sweep; bounded and deterministic.

    Optimize best independent and stable lanes without changing the selection
    policy. Strict improvements only. A completed final sweep certifies only
    single-easy-dose local optimality for this fixed calendar/tail and menu.
    """
    if identity_adapter is None:
        from run_analysis.adaptive.legacy_program_identity import fingerprint
    else:
        fingerprint = identity_adapter
    start=perf_counter()
    ordered=sorted(candidates,key=lambda c:(c['score'].total_cost,c['row']['hard_fingerprint']))
    if seed_candidates is None:
        seeds=[ordered[0]]
        stable=next((c for c in ordered if c['stable']),None)
        if stable is not None and fingerprint(stable['program'][0])!=fingerprint(seeds[0]['program'][0]):
            seeds.append(stable)
    else:
        # Before full-objective calendar pruning, every retained physical
        # finalist gets a dose lane. A calendar with a poor proxy allocation
        # must not be discarded before its existing dose menu is evaluated.
        seeds=[]; seen_seeds=set()
        for item in sorted(seed_candidates,key=lambda c:(
            c['score'].total_cost,c['row']['hard_fingerprint'],c['tail_fingerprint'])):
            key=(item['row']['hard_fingerprint'],item['tail_fingerprint'])
            if key not in seen_seeds:
                seeds.append(item); seen_seeds.add(key)
    if max_sweeps is not None and max_sweeps < 1:
        raise ValueError('Distance sweep limit must be positive')
    lanes=[dict(current=s,status='pending',path=[s['score'].total_cost],sweeps=[]) for s in seeds]
    cache={}; hits=0
    while any(l['status']=='pending' for l in lanes):
        active=[lane for lane in lanes if lane['status']=='pending']
        def stream(current):
            proposals = (neighbors(current['days'],inputs,state_at,visible_days=visible_days)
                         if neighbor_adapter is None else neighbor_adapter(current))
            for days in proposals:
                key=(fingerprint(days),current['tail_fingerprint'])
                yield key,dict(days=days,offsets=list(current['score'].offsets),
                    source='distance_neighbor',tail_parent=current)
        scans=scan_lanes([stream(lane['current']) for lane in active],evaluate,
                         cache=cache,budget=budget)
        for lane,scan in zip(active,scans):
            current=lane['current']; best=current; complete=scan['complete']
            hits+=scan['cache_hits']
            for result in scan['results']:
                if result is not None and result['score'].total_cost<best['score'].total_cost-1e-8:
                    best=result
            lane['sweeps'].append(dict(complete=complete,neighbors=scan['examined'],
                evaluations=scan['evaluations'],cache_hits=scan['cache_hits'],
                cost=current['score'].total_cost,best_cost=best['score'].total_cost,
                hard_fingerprint=fingerprint(current['days']),tail_fingerprint=current['tail_fingerprint']))
            if best is not current:
                lane['current']=best
                lane['path'].append(best['score'].total_cost)
            if not complete:
                lane['status']='budget_exhausted'
            elif best is current:
                lane['status']='locally_converged'
            elif max_sweeps is not None and len(lane['sweeps']) >= max_sweeps:
                lane['status']='sweep_limit_reached'
    return dict(version='fixed_tail_easy_coordinate_v1',budget=budget,visible_days=visible_days,
        seed_mode='all_retained' if seed_candidates is not None else 'selection_frontier',
        seed_count=len(seeds),
        max_sweeps=max_sweeps,
        evaluations=len(cache),cache_hits=hits,seconds=perf_counter()-start,
        lanes=[dict(status=l['status'],path=l['path'],sweeps=l['sweeps'],
                    hard_fingerprint=fingerprint(l['current']['days']),
                    tail_fingerprint=l['current']['tail_fingerprint']) for l in lanes])
