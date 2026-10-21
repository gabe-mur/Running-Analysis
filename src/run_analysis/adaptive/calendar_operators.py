"""Shared deterministic calendar neighborhoods; no training score or guards."""
from itertools import combinations
from collections import deque
from functools import lru_cache
from math import isfinite


def equal_volume_neighbors(occupants, allowed, roles, dose_miles, *, radius=6):
    """Bounded, exact-tenths easy merge/split/redistribution proposals.

    Replace 1--3 consecutive easy sessions with 1--3 catalogue sessions, at
    most seven days apart. Destinations extend the selected span by one day;
    other sessions stay untouched. No 3-to-3 exhaustive allocation. These are
    search limits, not training limits or weekly volume quotas. Round-robin
    across groups prevents a large early group from hiding all later groups.
    The caller must still evaluate full-program cost and eligibility.
    """
    if set(dose_miles) != set(roles) or any(
            not isfinite(v) or v <= 0 or abs(v * 10 - round(v * 10)) > 1e-8
            for v in dose_miles.values()):
        raise ValueError('Explicit positive one-decimal miles for every dose required')
    original = tuple(sorted(dict(occupants).items()))
    occupied, allowed = dict(original), set(allowed)
    if not set(occupied) <= allowed:
        return
    units = {k: round(v * 10) for k, v in dose_miles.items()}
    easy_keys = tuple(sorted(k for k in roles if roles[k] == 'easy'))
    easy = tuple(d for d, k in original if roles[k] == 'easy')

    @lru_cache(None)
    def allocations(total, count):
        if count == 0:
            return ((),) if total == 0 else ()
        return tuple((k, *tail) for k in easy_keys if units[k] <= total
                     for tail in allocations(total - units[k], count - 1))

    def group_proposals(group):
        unchanged = tuple(p for p in original if p[0] not in group)
        blocked = dict(unchanged)
        destinations = sorted(d for d in allowed if d not in blocked
                              and group[0] - 1 <= d <= group[-1] + 1)
        total = sum(units[occupied[d]] for d in group)
        for count in (2, 1, 3):
            if count == len(group) and count != 2:
                continue
            for days in combinations(destinations, count):
                if days[-1] - days[0] > radius:
                    continue
                for doses in allocations(total, count):
                    yield tuple(sorted((*unchanged, *zip(days, doses))))

    pending = deque(group_proposals(easy[i:i + count])
                    for count in (3, 2, 1) for i in range(len(easy) - count + 1)
                    if easy[i + count - 1] - easy[i] <= radius)
    seen = {original}
    while pending:
        iterator = pending.popleft()
        candidate = next(iterator, None)
        if candidate is None:
            continue
        pending.append(iterator)
        if candidate not in seen:
            seen.add(candidate)
            yield candidate


def nearby_prescription_neighbors(occupants, allowed):
    """Exchange occupied prescriptions within two days; cross two-day gaps.

    Date-set moves alone cannot express exchanges on occupied dates. Carry the
    entire dose key, not just its role, so the full evaluator sees the actual
    proposed sessions. Ordinary one-day moves into empty dates are already in
    calendar_neighbors. Two days is an operator radius, never a recovery rule.
    At most four proposals per occupied date before deduplication.
    """
    occupied=dict(occupants)
    allowed=set(allowed)
    if not occupied or not set(occupied)<=allowed:
        return []
    original=tuple(sorted(occupied.items()))
    results=set()
    for day,key in sorted(occupied.items()):
        for step in (-2,-1,1,2):
            other=day+step
            if other not in allowed or (other not in occupied and abs(step)==1):
                continue
            changed=dict(occupied)
            changed.pop(day)
            if other in occupied:
                changed[day]=occupied[other]
            changed[other]=key
            candidate=tuple(sorted(changed.items()))
            if candidate!=original:
                results.add(candidate)
    return sorted(results)

def calendar_neighbors(seed, allowed):
    """At most 3H one-edit neighbors, deterministic and without empty plans."""
    seed = tuple(sorted(set(seed)))
    allowed = set(allowed)
    if not seed or not set(seed) <= allowed:
        return []
    occupied = set(seed)
    neighbors = set()
    for day in occupied:
        if len(seed) > 1:
            neighbors.add(tuple(x for x in seed if x != day))
        for destination in (day - 1, day + 1):
            if destination in allowed - occupied:
                neighbors.add(tuple(sorted((occupied - {day}) | {destination})))
    for day in allowed - occupied:
        neighbors.add(tuple(sorted((*seed, day))))
    return sorted(neighbors)


def opening_opportunity_neighbors(seed, allowed):
    """Opening move plus one tail add/remove: <= 2H, no beam multiplication.

    A useful run today can require a different later exposure count. Neither
    intermediate one-edit program has to win to compare the combined program.
    Keep the existing one-edit pool separately; these have no score preference.
    """
    seed = tuple(sorted(set(seed)))
    allowed = set(allowed)
    if not seed or not set(seed) <= allowed:
        return []
    neighbors = set()
    tail = set(seed[1:])
    for opening in (seed[0] - 1, seed[0] + 1):
        if opening not in allowed or (tail and opening >= min(tail)):
            continue
        shifted = tail | {opening}
        for added in allowed - shifted:
            if added > opening:
                neighbors.add(tuple(sorted(shifted | {added})))
        for removed in tail:
            neighbors.add(tuple(sorted(shifted - {removed})))
    return sorted(neighbors)


def paired_move_neighbors(seed, allowed):
    """Two independent one-day moves, checked once when local search stalls.

    Each move crosses an occupied/empty day boundary. On a 21-day horizon
    there are at most 20 such boundaries and 190 pairs, not a larger beam at
    every search depth. Conflicts, blocked dates and empty programs are excluded.
    """
    occupied = set(seed)
    allowed = set(allowed)
    if not occupied or not occupied <= allowed:
        return []
    moves = [(day, destination) for day in sorted(occupied)
             for destination in (day - 1, day + 1) if destination in allowed - occupied]
    return sorted({tuple(sorted((occupied - {a, b}) | {x, y}))
                   for (a, x), (b, y) in combinations(moves, 2) if a != b and x != y})
