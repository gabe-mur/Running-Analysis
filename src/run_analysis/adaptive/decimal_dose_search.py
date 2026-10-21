"""Finite decimal-dose neighborhoods, independent of scoring and physiology."""
from itertools import combinations


def calendar_representatives(catalog, *, ordinary, support):
    """Keep the former coarse calendar breadth; all tenths remain refinable.

    Anchors depend on the frozen stage, never on the winning plan or today's
    upload. This is a search approximation, not a minimum/maximum prescription.
    """
    from run_analysis.staged_progression import tenth
    from decimal import Decimal
    easy = [d for d in catalog.values() if d.role == 'easy']
    if not easy:
        return tuple(sorted(catalog))
    low = min(round(d.miles * 10) for d in easy)
    anchors = {round(tenth(ordinary * .8) * 10), round(ordinary * 10)}
    anchors.update(round(tenth(Decimal(str(support)) / n) * 10) for n in range(1, 7))
    return tuple(sorted(k for k, d in catalog.items() if d.role != 'easy'
        or (round(d.miles * 10) - low) % 5 == 0 or round(d.miles * 10) in anchors))


def decimal_pair_neighbors(program, roles, miles, *, index, radius=6):
    """Scan each nearby easy pair's full fixed-volume tenth-mile line.

    Evaluate both changes together: a volume floor can reject either single
    edit. No convexity assumption and no enumeration of calendar combinations.
    Nearest edits first, interleaved across pairs to avoid fixed-date starvation.
    """
    program = tuple(sorted(program))
    easy = [(d, k) for d, k in program if roles[k] == 'easy']
    by_tenth = {}
    for k in sorted(roles):
        if roles[k] == 'easy':
            by_tenth.setdefault(round(miles[k] * 10), k)
    streams = []
    for (a, ka), (b, kb) in combinations(easy, 2):
        if index[b] - index[a] > radius:
            continue
        old = round(miles[ka] * 10)
        total = old + round(miles[kb] * 10)
        choices = sorted((n for n in by_tenth if n != old and total-n in by_tenth),
                         key=lambda n: (abs(n-old), n))
        streams.append(iter([tuple((d, by_tenth[n] if d == a else
            by_tenth[total-n] if d == b else k) for d, k in program) for n in choices]))
    while streams:
        remaining = []
        for stream in streams:
            item = next(stream, None)
            if item is not None:
                yield item
                remaining.append(stream)
        streams = remaining


def decimal_merge_neighbors(program, roles, miles, *, index, radius=6, minimum_date=None):
    """Exact-volume two-to-one easy merges on the FULL supported tenth catalog.

    Keep either existing date; leave all other sessions unchanged. Radius is a
    search neighborhood, not a coaching/recovery rule. No zero-dose session or
    unscored bounds inference. Consecutive easy pairs keep enumeration linear.
    """
    program = tuple(sorted(program))
    easy = [(d, k) for d, k in program if roles[k] == 'easy'
        and (minimum_date is None or d >= minimum_date)]
    by_tenth = {}
    for k in sorted(roles):
        if roles[k] == 'easy':
            by_tenth.setdefault(round(miles[k]*10), k)
    for first, second in zip(easy, easy[1:]):
        if index[second[0]]-index[first[0]] > radius:
            continue
        total = sum(round(miles[k]*10) for _, k in (first, second))
        if total not in by_tenth:
            continue
        unchanged = tuple(p for p in program if p not in (first, second))
        for day in (first[0], second[0]):
            yield tuple(sorted((*unchanged, (day, by_tenth[total]))))


def quality_easy_neighbors(program, roles, miles, *, index, radius=6, priority_before=None):
    """Exchange a native quality dose and nearby easy mileage together.

    Preserve dates, roles and exact tenth-mile total. A funded stage can reject
    either single change even when the whole exchange is better. No assumption
    of equal quality stimulus or fatigue: the caller scores every whole plan.
    Round-robin pairs keep a large native menu from starving later dates.
    An optional publication boundary orders visible-quality pairs first, but
    preserves the entire proposal universe. This is coverage, not a score bonus.
    """
    from collections import deque
    program=tuple(sorted(program))
    easy=[(d,k) for d,k in program if roles[k]=='easy']
    quality=[(d,k) for d,k in program if roles[k]=='quality']
    easy_levels={}
    for key in sorted(roles):
        if roles[key]=='easy':easy_levels.setdefault(round(miles[key]*10),key)
    def pair_stream(a,ka,b,kb):
        total=round(miles[ka]*10)+round(miles[kb]*10)
        alternatives=sorted((k for k in roles if roles[k]=='quality' and k!=ka
            and total-round(miles[k]*10) in easy_levels),
            key=lambda k:(abs(miles[k]-miles[ka]),k))
        for key in alternatives:
            other=easy_levels[total-round(miles[key]*10)]
            yield tuple((d,key if d==a else other if d==b else k) for d,k in program)
    seen=set()
    pairs=[(a,ka,b,kb) for a,ka in quality for b,kb in easy
        if abs(index[b]-index[a])<=radius]
    groups=(pairs,) if priority_before is None else (
        [p for p in pairs if p[0]<priority_before],
        [p for p in pairs if p[0]>=priority_before])
    for group in groups:
        streams=deque(iter(pair_stream(*p)) for p in group)
        while streams:
            stream=streams.popleft();p=next(stream,None)
            if p is not None:
                streams.append(stream)
                if p not in seen:
                    seen.add(p);yield p
