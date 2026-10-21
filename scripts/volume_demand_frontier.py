"""Offline volume-conditioned calendar/dose search. Best found, not global optimum."""
from itertools import combinations, product
from math import ceil, exp
from time import perf_counter

import numpy as np

LONG, QUALITY = -84, -45  # Fixed physical comparison doses, not category weights.


class PatternCost:
    """Exact quadratic compression of the existing repeated-calendar demand score.

    Zero/positive tenths encode rest/easy; fixed negative tokens identify key work.
    Condensing repeated dates is algebra only; all history and continuation costs
    stay in the matrix. No delivery credit, volume target or continuity weight.
    """
    def __init__(self, period, cycles, role_vectors, history=(), *, memory_days=2.,
                 timing_compression_days=0., axis_weights=(1.,1.,1.)):
        if period < 1 or cycles < 1 or memory_days <= 0 or not 0 <= timing_compression_days < 1:
            raise ValueError('Positive scope and explicit memory required')
        self.period, self.cycles = period, cycles
        self.vectors = np.asarray(role_vectors,dtype=float)  # Easy per mile, whole long, whole quality.
        if self.vectors.shape != (3,3) or np.any(~np.isfinite(self.vectors)) or np.any(self.vectors < 0):
            raise ValueError('Three nonnegative finite demand vectors required')
        weights = np.asarray(axis_weights,dtype=float)
        if weights.shape != (3,) or np.any(~np.isfinite(weights)) or np.any(weights<0) or weights.sum()<=0:
            raise ValueError('Nonnegative axis weights required')
        self.weights = weights/weights.sum()*7/(period*cycles)
        self.matrix = np.zeros((period,period))
        self.linear = np.zeros((period,3))
        history = tuple(history)
        for age,q in history:
            if age <= 0 or np.shape(q) != (3,) or np.any(np.asarray(q)<0) or np.any(~np.isfinite(q)):
                raise ValueError('History must precede origin with known scenario demand')
        for t in range(period*cycles):
            i=t%period
            self.matrix[i,i]+=.5
            for u in range(t):
                self.matrix[i,u%period]+=exp(-max(0,t-u-timing_compression_days)/memory_days)
            for age,q in history:
                self.linear[i]+=np.asarray(q)*exp(-max(0,t+age-timing_compression_days)/memory_days)
        self.matrix=(self.matrix+self.matrix.T)/2

    def batch(self, states):
        x=np.asarray(states)
        if x.ndim != 2 or x.shape[1] != self.period or np.any((x<0)&(x!=LONG)&(x!=QUALITY)):
            raise ValueError('Explicit rest/easy/long/quality pattern required')
        q=(np.maximum(x,0)[:,:,None]/10*self.vectors[0]
           +(x==LONG)[:,:,None]*self.vectors[1]+(x==QUALITY)[:,:,None]*self.vectors[2])
        out=np.zeros(len(x))
        for c in range(3):
            v=q[:,:,c]
            out+=self.weights[c]*(np.sum((v@self.matrix)*v,axis=1)+v@self.linear[:,c])
        return out


def seeds(period, easy_tenths, *, floor=14, ceiling=60, step=1, phase_count=7, fixed_keys=None):
    """Different counts and phases, without consulting the objective or oracle."""
    if period%7 or easy_tenths%step or floor%step or ceiling%step or floor<=0 or ceiling<floor:
        raise ValueError('Declared seven-day-equivalent key delivery and dose grid required')
    key_count=period//7
    minimum=ceil(easy_tenths/ceiling)
    maximum=min(easy_tenths//floor,period-2*key_count)
    if minimum>maximum or minimum==0:
        raise ValueError('Easy budget cannot be represented by the declared grid')
    counts=sorted({minimum,(minimum+maximum)//2,maximum})
    phases=sorted({int(i*period/min(phase_count,period)) for i in range(min(phase_count,period))})
    result=[]
    if fixed_keys is not None:
        fixed_keys=np.asarray(fixed_keys,dtype=np.int16)
        if (fixed_keys.shape!=(period,) or np.any((fixed_keys!=0)&(fixed_keys!=LONG)&(fixed_keys!=QUALITY))
                or sum(fixed_keys==LONG)!=key_count or sum(fixed_keys==QUALITY)!=key_count):
            raise ValueError('Fixed keys must preserve the declared purpose count')
    for phase,count in product(phases,counts):
        x=np.zeros(period,dtype=np.int16)
        if fixed_keys is None:
            for block in range(key_count):
                x[(phase+block*7)%period]=LONG
                x[(phase+block*7+3)%period]=QUALITY
        else:
            x[:]=fixed_keys
        available=np.flatnonzero(x==0)
        if fixed_keys is not None:
            available=np.roll(available,phase)
        chosen=available[np.array([int(i*len(available)/count) for i in range(count)])]
        units,remainder=divmod(easy_tenths//step,count)
        doses=np.array([units+(i<remainder) for i in range(count)])*step
        if np.any(doses<floor) or np.any(doses>ceiling):
            continue
        x[chosen]=doses
        result.append(x)
    return result


def neighbors(state, *, floor=14, ceiling=60, step=1, move_keys=True):
    """Calendar swaps/relocations and exact-volume split/merge/dose transfers."""
    x=np.asarray(state,dtype=np.int16)
    variants=[]
    for i,j in combinations(range(len(x)),2):
        if not move_keys and (x[i]<0 or x[j]<0):
            continue
        if x[i]!=x[j]:
            y=x.copy(); y[i],y[j]=x[j],x[i]; variants.append(y)
    recipients=np.flatnonzero(x>=0)
    for i in np.flatnonzero(x>0):
        for j in recipients:
            if i==j:
                continue
            for amount in range(step,min(int(x[i]),ceiling-int(x[j]))+1,step):
                a,b=int(x[i])-amount,int(x[j])+amount
                if (a==0 or a>=floor) and b>=floor:
                    y=x.copy(); y[i],y[j]=a,b; variants.append(y)
    if not variants:
        return np.empty((0,len(x)),dtype=np.int16)
    return np.unique(np.asarray(variants),axis=0)


def search(cost, easy_tenths, *, floor=14, ceiling=60, step=1, phase_count=7, max_steps=60, fixed_keys=None):
    started=perf_counter()
    starts=seeds(cost.period,easy_tenths,floor=floor,ceiling=ceiling,step=step,phase_count=phase_count,fixed_keys=fixed_keys)
    if not starts:
        raise ValueError('No feasible initial candidates')
    evaluated=0; cache={}; endpoints=[]
    for initial in starts:
        x=initial.copy(); value=float(cost.batch([x])[0]); exhausted=True
        for iteration in range(max_steps):
            key=tuple(int(v) for v in x)
            if key in cache:
                x,value,exhausted=cache[key]; x=x.copy(); break
            nearby=neighbors(x,floor=floor,ceiling=ceiling,step=step,move_keys=fixed_keys is None)
            values=cost.batch(nearby); evaluated+=len(nearby)
            if len(values)==0 or float(values.min())>=value-1e-10:
                exhausted=False; break
            best=int(np.argmin(values)); x=nearby[best]; value=float(values[best])
        cache[tuple(int(v) for v in initial)]=(x.copy(),value,exhausted)
        endpoints.append(dict(state=x.copy(),cost=value,budget_exhausted=exhausted))
    best=min(endpoints,key=lambda r:(r['cost'],tuple(r['state'])))
    distinct={tuple(e['state']) for e in endpoints}
    return dict(state=best['state'].tolist(),cost=best['cost'],starts=len(starts),
        distinct_endpoints=len(distinct),endpoint_cost_range=[min(e['cost'] for e in endpoints),max(e['cost'] for e in endpoints)],
        budget_exhausted_starts=sum(e['budget_exhausted'] for e in endpoints),
        evaluated_neighbors=evaluated,elapsed_seconds=perf_counter()-started,global_optimum_proven=False)


def exhaustive_small(cost, *, easy_choices=(0,20,30)):
    """All seven-date, one-long/one-quality calendars on a declared tiny grid."""
    if cost.period!=7:
        raise ValueError('Small oracle scope is exactly seven days')
    states=[]
    for long_day in range(7):
        for quality_day in range(7):
            if long_day==quality_day:
                continue
            free=[i for i in range(7) if i not in (long_day,quality_day)]
            for doses in product(easy_choices,repeat=5):
                x=np.zeros(7,dtype=np.int16); x[long_day]=LONG; x[quality_day]=QUALITY; x[free]=doses
                states.append(x)
    states=np.asarray(states); values=cost.batch(states)
    totals=np.maximum(states,0).sum(axis=1)
    result={}
    for total in sorted(set(totals.tolist())):
        indices=np.flatnonzero(totals==total); best=indices[np.argmin(values[indices])]
        result[total]=dict(cost=float(values[best]),state=states[best].tolist(),candidates=len(indices))
    return dict(total_candidates=len(states),by_easy_tenths=result)
