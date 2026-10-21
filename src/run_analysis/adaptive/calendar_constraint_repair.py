"""Bounded prior-calendar candidates for explicit changed date constraints.

This is search coverage, not a continuity bonus, volume debt or recovery policy.
Every candidate still needs the ordinary full-program evaluation.
"""
from datetime import date


def role_date_domains(dates, roles, *, restrictions, as_of, long_dates=None):
    """Compile only known absolute date exclusions and the explicit long weekday."""
    known=tuple(r for r in restrictions if r.available_at<=as_of)
    blocked={day for day in dates if any(r.start<=date.fromisoformat(day)<r.end for r in known)}
    result={role:frozenset(set(dates)-blocked) for role in set(roles.values())} if blocked else {}
    if long_dates is not None:
        result['long']=frozenset(set(long_dates)-blocked)
    return result


def constraint_repair_neighbors(program, dates, roles, allowed, *, limit=32, radius_days=7):
    """Keep legal slots; delete or locally relocate conflicts, without dose edits.

    Always offer deletion of ALL conflicting slots, so canceling a run doesn't
    require replacing its mileage. A bounded nearest-date beam also offers
    combined repairs for multiple conflicts. Legal role swaps cover an occupied
    preferred weekday. Ranking allocates candidate coverage only, never score.
    """
    if limit<1 or radius_days<0:
        raise ValueError('Positive repair allowance and nonnegative local radius required')
    dates=tuple(sorted(set(dates)))
    prior=tuple(sorted(program));domain=set(dates)
    def permitted(day,key):
        return day in domain and (roles[key] not in allowed or day in allowed[roles[key]])
    invalid=tuple((day,key) for day,key in prior if not permitted(day,key))
    if not invalid:
        return ()
    clean=tuple((day,key) for day,key in prior if permitted(day,key))
    # (dropped original slots, total date displacement, program)
    beam=[(0,0,clean)]
    for origin,key in invalid:
        expanded={}
        destinations=sorted((day for day in dates if permitted(day,key)
            and abs((date.fromisoformat(day)-date.fromisoformat(origin)).days)<=radius_days),
            key=lambda day:(abs((date.fromisoformat(day)-date.fromisoformat(origin)).days),day))
        for drops,shift,partial in beam:
            rows=dict(partial)
            options=[(drops+1,shift,partial)]
            for day in destinations:
                other=rows.get(day)
                if other is not None and (origin in rows or not permitted(origin,other)):
                    continue
                moved=dict(rows)
                moved[day]=key
                distance=abs((date.fromisoformat(day)-date.fromisoformat(origin)).days)
                if other is not None:
                    moved[origin]=other
                    distance*=2
                options.append((drops,shift+distance,tuple(sorted(moved.items()))))
            for row in options:
                old=expanded.get(row[2])
                if old is None or row<old:
                    expanded[row[2]]=row
        beam=sorted(expanded.values())[:limit]
    # The baseline is protected from beam truncation, NOT from model comparison.
    candidates=[clean]+[row[2] for row in beam]
    return tuple(dict.fromkeys(candidates))[:limit]
