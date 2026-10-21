"""Expired issued dose is a lost opportunity, not compulsory make-up mileage.

This is score-only relief, never physical delivery or progression credit. The
frozen past opportunities cannot be created by a candidate's future omissions.
Their influence expires through the SAME rolling/smoothing kernel as the dose.
"""
from dataclasses import dataclass
from datetime import date
from math import isfinite

LEGACY_VERSION = 'forgone_issued_v1'
VERSION = 'forgone_issued_v2'


@dataclass(frozen=True)
class ForegoneVolume:
    day: date
    source: str
    role: str
    issued_miles: float
    observed_miles: float

    def __post_init__(self):
        if (type(self.day) is not date or not self.source or self.role not in {'easy','long','quality'}
                or not all(isfinite(x) for x in (self.issued_miles,self.observed_miles))
                or not 0 <= self.observed_miles < self.issued_miles):
            raise ValueError('Positive unfulfilled issued volume with original provenance required')

    @property
    def miles(self):
        return self.issued_miles-self.observed_miles


def from_reviews(entries, *, since, today):
    """Only expired local dates and causally available original contracts.

    The caller syncs the physical ledger first. Ambiguous distance bounds use
    the upper bound: uncertainty must not manufacture a shortfall. A corrected
    or late completion removes/reduces relief on the next frozen decision.
    """
    rows=[]
    for entry in entries:
        if not entry.key.startswith('slot:'):
            continue
        day=date.fromisoformat(entry.key.removeprefix('slot:'))
        if not since <= day < today:
            continue
        distance=next((t.quantity for t in entry.contract.targets if t.metric=='distance_meters'),None)
        if distance is None or distance.estimate is None:
            continue
        issued=distance.estimate/1609.344
        if entry.completion and entry.completion.distance_meters is None:
            continue
        actual=entry.completion.distance_meters.high/1609.344 if entry.completion else 0.
        if actual+1e-8 >= issued:
            continue
        purpose=entry.contract.purpose
        role='easy' if purpose=='easy_volume' else 'long' if purpose=='long_endurance' else 'quality'
        rows.append(ForegoneVolume(day,entry.contract.source,role,issued,actual))
    if len({r.day for r in rows})!=len(rows):
        raise ValueError('One unfulfilled opportunity per reconciled date required')
    return tuple(sorted(rows,key=lambda r:r.day))


def amounts(rows, start, count):
    """Separate nonphysical channels; never insert these into a load ledger."""
    import numpy as np
    result=np.zeros((2,count))
    for row in rows:
        i=(row.day-start).days
        if 0 <= i < count:
            result[0,i]+=row.miles
            if row.role=='easy': result[1,i]+=row.miles
    return result


def relief_windows(context, dates, kernel, targets, ledger):
    """One fixed adoption boundary; never forgive a future candidate deficit.

    Average the SIGNED pre-adoption target-minus-actual balance before clipping.
    Per-rest-day clipping would invent relief even for fully funded history.
    Future targets and the physical continuation mean are deliberately absent.
    """
    import numpy as np
    start=dates[0]
    relief=np.array([np.convolve(v,kernel,mode='valid')
        for v in amounts(context.foregone_volume,start,len(dates))])
    if context.volume_shortfall_policy!=VERSION:
        return relief
    before=context.volume_shortfall_since
    signed=np.array(targets)*np.array([day<before for day in dates])
    for row in ledger:
        i=(row.day-start).days
        if row.source=='observed' and row.day<before and 0<=i<len(dates):
            if row.role!='race':signed[0,i]-=row.miles
            if row.role=='easy':signed[1,i]-=row.miles
    return relief+np.maximum(0.,np.array([np.convolve(v,kernel,mode='valid') for v in signed]))
