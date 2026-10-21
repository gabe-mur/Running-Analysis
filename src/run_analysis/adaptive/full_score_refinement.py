"""Offline, decision-local multi-start calendar descent on the full objective.

Convergence is only with respect to add/remove/one-day-move calendars rebuilt
by the current allocator. It is not a global optimum or a role/dose search.
"""
from contextlib import contextmanager
from contextvars import ContextVar

from run_analysis.adaptive.calendar_operators import calendar_neighbors
from run_analysis.adaptive.shared_lane_scan import scan_lanes


REFINEMENT_BUDGET = ContextVar('full_score_refinement_budget', default=None)


@contextmanager
def full_score_refinement_policy(*, budget=240, starts=3, fixed_tail=False):
    if budget < 1 or starts < 1:
        raise ValueError('Refinement requires positive evaluation and start budgets')
    token = REFINEMENT_BUDGET.set(dict(budget=budget, starts=starts, fixed_tail=fixed_tail))
    try:
        yield
    finally:
        REFINEMENT_BUDGET.reset(token)


def refine(candidates, evaluate, allowed, *, budget, starts, fixed_tail=False,
           proposal_adapter=None, seed_identity_adapter=None):
    """Steepest descent, round-robin across distinct physical starting plans.

    `evaluate` returns a feasible candidate or None; its objective and guards
    are exactly those of initial selection. Cached/rejected calendars count
    once against this additional materialization budget. No proxy pruning.
    In fixed-tail mode, each cache key also includes the physical suffix and
    convergence means calendar-local conditional on that particular suffix.
    A representation adapter may supply (physical_cache_key, proposal) pairs.
    Lane selection, descent, budgets and diagnostics remain shared with the
    original scheduler. Default callers retain their original calendar behavior.
    """
    def rank(item):
        return (item['score'].total_cost, tuple(item['score'].offsets),
                item['row']['hard_fingerprint'])

    ordered = sorted(candidates, key=rank)
    seed_identity = seed_identity_adapter or (lambda c:c['row']['hard_fingerprint'])
    seeds = []
    for item in [ordered[0], next((x for x in ordered if x['stable']), ordered[0]),
                 next((x for x in candidates if x['row']['source'] == 'publication_seed'), ordered[0]),
                 *ordered]:
        if all(seed_identity(item) != seed_identity(x) for x in seeds):
            seeds.append(item)
        if len(seeds) == starts:
            break
    lanes = [dict(current=s, path=[s['score'].total_cost], status='pending',
                  seed_fingerprint=seed_identity(s), neighborhoods=[]) for s in seeds]
    cache = {}
    cache_hits = 0
    while any(lane['status'] == 'pending' for lane in lanes):
        active=[lane for lane in lanes if lane['status']=='pending']
        streams=[]
        for lane in active:
            current = lane['current']
            neighbors = sorted(calendar_neighbors(tuple(current['score'].offsets), allowed)) if proposal_adapter is None else None
            def iter_proposals(current=current, neighbors=neighbors):
                if proposal_adapter is not None:
                    yield from proposal_adapter(current, allowed)
                    return
                for offsets in neighbors:
                    tail_key = current['tail_fingerprint'] if fixed_tail else None
                    key = (tuple(offsets), tail_key) if fixed_tail else tuple(offsets)
                    seed = dict(offsets=list(offsets), source='full_score_neighbor')
                    if fixed_tail:
                        seed['tail_parent'] = current
                    yield key,seed
            streams.append(list(iter_proposals()))
        scans=scan_lanes(streams,evaluate,cache=cache,budget=budget)
        for lane,proposals,scan in zip(active,streams,scans):
            current=lane['current']
            evaluated=scan['results']; complete=scan['complete']
            cache_hits+=scan['cache_hits']
            best = min([current, *evaluated], key=rank)
            improved = best['score'].total_cost < current['score'].total_cost - 1e-8
            lane['neighborhoods'].append(dict(offsets=list(current['score'].offsets),
                hard_fingerprint=current['row']['hard_fingerprint'],
                tail_fingerprint=current.get('tail_fingerprint'),
                cost=current['score'].total_cost, neighbors=len(proposals),
                examined=scan['examined'], evaluations=scan['evaluations'],
                complete=complete, eligible=len(evaluated), best_cost=best['score'].total_cost))
            if improved:
                lane['current'] = best
                lane['path'].append(best['score'].total_cost)
            if not complete:
                lane['status'] = 'budget_exhausted'
            elif not improved:
                lane['status'] = 'locally_converged'
    return dict(version='fixed_tail_calendar_descent_v1' if fixed_tail else 'full_score_calendar_descent_v1', budget=budget,
        evaluated_calendars=len(cache), calendar_cache_hits=cache_hits,
        fixed_tail=fixed_tail,
        scope=__doc__, lanes=[dict({k:v for k,v in lane.items() if k != 'current'},
            final_fingerprint=lane['current']['row']['hard_fingerprint'],
            final_offsets=list(lane['current']['score'].offsets)) for lane in lanes])
