"""Measure delivered mileage mass, not ratios to alternative optimized doses.

Long-to-long calendar spans provide known exposure durations. They are
observations, NOT progression review windows: every closed observation stays
in the exponential fit, and the controller can still advance between receipts.
No within-span arrangement of the same mileage can alter its measured mass.
The remaining open/leading/trailing span is explicitly unmeasured, not zero.
"""
from dataclasses import dataclass
from datetime import datetime
from math import exp, expm1, fsum, isfinite, log, log1p, sinh

from run_analysis.athlete_projection import aware
from run_analysis.general_progression import VOLUME_GROWTH_PER_7
from .smoothed_progression import days_between

VERSION = 'cadence_mass_growth_prototype_v2'


@dataclass(frozen=True)
class CalendarDose:
    opportunity: str
    scheduled_at: datetime
    available_at: datetime
    miles: float  # original prescription, attribution only; NEVER a divisor
    baseline_key: str
    source: str
    span_start: datetime | None
    span_end: datetime | None
    span_slots: tuple[str, ...]
    basis: str = 'issued_calendar_mass_v2'

    def __post_init__(self):
        aware(self.scheduled_at); aware(self.available_at)
        if (not self.opportunity or not self.baseline_key or not self.source
                or not isfinite(self.miles) or self.miles <= 0
                or self.available_at > self.scheduled_at or self.basis != 'issued_calendar_mass_v2'):
            raise ValueError('Causal original calendar evidence required')
        if (self.span_start is None) != (self.span_end is None):
            raise ValueError('Both calendar exposure boundaries required')
        if self.span_start is not None:
            aware(self.span_start); aware(self.span_end)
            if not self.span_start <= self.scheduled_at < self.span_end or not self.span_slots:
                raise ValueError('Prescription must belong to its declared calendar span')
            if self.opportunity not in self.span_slots or len(set(self.span_slots)) != len(self.span_slots):
                raise ValueError('Exact unique calendar membership required')


def estimate(entries, references, *, as_of, zone, baseline_volume, memory_days=14, supplemental=(),
             maximum_growth_per_7=VOLUME_GROWTH_PER_7):
    aware(as_of)
    if not isfinite(baseline_volume) or baseline_volume <= 0 or type(memory_days) is not int or memory_days <= 0:
        raise ValueError('Positive normalization and declared memory required')
    if not isfinite(maximum_growth_per_7) or not 0 < maximum_growth_per_7 < 1:
        raise ValueError('Positive fractional coaching growth rate required')
    entries, references, supplemental = tuple(entries), tuple(references), tuple(supplemental)
    if len({e.key for e in entries}) != len(entries) or len({r.opportunity for r in references}) != len(references):
        raise ValueError('Duplicate calendar opportunity')
    physical = [e.completion.key for e in entries if e.completion] + [r.key for _, r in supplemental]
    if len(set(physical)) != len(physical):
        raise ValueError('Physical completion counted twice')
    refs = {r.opportunity:r for r in references if r.available_at <= as_of}
    extra = {}
    for key, done in supplemental:
        if key not in {e.key for e in entries}:
            raise ValueError('Supplement requires an explicit known opportunity link')
        extra.setdefault(key, []).append(done)
    included, unresolved, partial, baselines = [], [], [], set()
    for e in entries:
        if e.contract.issued_at > as_of or e.scheduled_at > as_of:
            continue
        done = e.completion
        known = bool(done and done.available_at <= as_of and done.end <= as_of)
        if not known and e.scheduled_at.astimezone(zone).date() >= as_of.astimezone(zone).date():
            continue
        ref = refs.get(e.key)
        if ref is None:
            unresolved.append(e.key+':missing_calendar_reference'); continue
        if ref.scheduled_at != e.scheduled_at or ref.available_at > e.contract.issued_at:
            raise ValueError('Calendar reference must retain original issuance and schedule')
        baselines.add(ref.baseline_key)
        actual, keys = 0., []
        for observed in ([done] if known else []) + extra.get(e.key, []):
            if observed.available_at > as_of or observed.end > as_of:
                continue
            q = observed.distance_meters
            if q is None or q.low != q.high:
                unresolved.append(e.key+':distance_uncertain'); continue
            actual += q.low/1609.344; keys.append(observed.key)
        row = dict(key=e.key,scheduled_at=e.scheduled_at.isoformat(),issued_miles=ref.miles,
            actual_miles=actual,physical_keys=keys,
            span_start=ref.span_start.isoformat() if ref.span_start else None,
            span_end=ref.span_end.isoformat() if ref.span_end else None)
        included.append(row)
        if ref.span_start is None:
            partial.append(e.key)
    if len(baselines) > 1:
        raise ValueError('Cannot mix calendar adoption identities')
    result = dict(version=VERSION,as_of=as_of.isoformat(),memory_days=memory_days,
        baseline_key=next(iter(baselines),None),included=included,unresolved=unresolved,
        partial_span_opportunities=partial,reference_quality='declared_calendar_mass',
        growth_per_7=0.,level_ratio=None,status='insufficient_calendar_exposure',
        planner_influence=False,physical_credit_added=0.,calendar_cutoff=False,
        measurement_scope='Closed variable-duration calendar spans; open spans are not zero or fitted growth')
    if unresolved:
        return dict(result,status='evidence_unresolved')
    by_key = {r['key']:r for r in included}
    spans = {}
    for r in references:
        if r.available_at > as_of or r.span_start is None or r.span_end > as_of:
            continue
        identity = (r.span_start,r.span_end)
        members = tuple(sorted(r.span_slots))
        if identity in spans and spans[identity] != members:
            return dict(result,status='calendar_membership_reconciliation_required')
        spans[identity] = members
    closed = []
    for (start,end), members in sorted(spans.items()):
        if any(k not in by_key for k in members):
            # A leading adoption fragment or changed calendar has incomplete
            # coverage. Do not invent a miss, fill from future intent, or fit it.
            continue
        if any(by_key[k]['span_start'] != start.isoformat() or by_key[k]['span_end'] != end.isoformat()
               for k in members):
            return dict(result,status='calendar_membership_reconciliation_required')
        closed.append(dict(start=start.isoformat(),end=end.isoformat(),days=days_between(end,start),
            actual_miles=fsum(by_key[k]['actual_miles'] for k in members),opportunities=list(members)))
    if any(datetime.fromisoformat(b['start']) < datetime.fromisoformat(a['end']) for a,b in zip(closed,closed[1:])):
        return dict(result,status='calendar_overlap_reconciliation_required')
    result['closed_spans'] = closed
    if len(closed) < 2:
        return result
    origin = max(datetime.fromisoformat(r['end']) for r in closed)
    fitted = fit_mass(closed,origin,memory_days)
    if fitted is None:
        return dict(result,status='calendar_trend_unresolved')
    slope, level = fitted
    maximum = log1p(maximum_growth_per_7)/7
    rate = expm1(7*min(maximum,max(0.,slope)))
    status = 'flat_or_falling' if rate == 0 else 'growing_at_cap' if slope >= maximum else 'growing_below_cap'
    # Do not keep advancing on stale closed-span evidence through a KNOWN miss.
    pending_miss = any(r['actual_miles']==0 and (
        datetime.fromisoformat(r['scheduled_at'])>=origin if r['span_end'] is None
        else datetime.fromisoformat(r['span_end'])>origin)
        for r in included)
    if pending_miss:
        rate=0.; status='pending_missed_opportunity'
    return dict(result,status=status,growth_per_7=rate,level_ratio=level/baseline_volume,
        measured_volume_per_7=level,level_at=origin.isoformat(),last_opportunity_at=origin.isoformat(),
        fitted_log_slope_per_day=slope,pending_missed_opportunity=pending_miss)


def fit_mass(rows, origin, memory_days):
    """Fit total distance = integral of a continuous exponential volume rate.

    Within-span timestamps/roles/dose proportions have no statistical weight.
    Actual and expected mass have exactly the same exposure and age weights.
    The interval integral (not a midpoint approximation) permits 6/8/10-day or
    irregular cadence without defining a preferred seven-day calendar.
    """
    times = [days_between(datetime.fromisoformat(r['start'])+(datetime.fromisoformat(r['end'])-
        datetime.fromisoformat(r['start']))/2,origin) for r in rows]
    if len(set(times)) < 2 or any(r['days']<=0 or not isfinite(r['actual_miles']) or r['actual_miles']<0 for r in rows):
        raise ValueError('Distinct positive exposure intervals and finite mileage required')
    weights = [exp(t/memory_days) for t in times]
    actual = fsum(w*r['actual_miles'] for w,r in zip(weights,rows))
    if actual <= 0:
        return 0.,0.
    target_mean = fsum(w*r['actual_miles']*t for w,r,t in zip(weights,rows,times))/actual
    if not min(times)+1e-12 < target_mean < max(times)-1e-12:
        return None
    def moments(slope):
        logs=[]
        for r,t in zip(rows,times):
            x=abs(slope*r['days']/2)
            correction=(x*x/6 if x<1e-5 else log(sinh(x)/x) if x<300 else x-log(2*x))
            logs.append(t/memory_days+log(r['days']/7)+slope*t+correction)
        largest=max(logs); masses=[exp(v-largest) for v in logs]; total=fsum(masses)
        return fsum(m*t for m,t in zip(masses,times))/total,largest+log(total)
    if abs(moments(0.)[0]-target_mean)<1e-12:
        slope=0.
    else:
        lo,hi=-log1p(.04)/7,log1p(.04)/7
        for _ in range(64):
            if moments(lo)[0]<=target_mean<=moments(hi)[0]: break
            lo*=2;hi*=2
        else:return None
        for _ in range(64):
            mid=(lo+hi)/2
            if moments(mid)[0]<target_mean:lo=mid
            else:hi=mid
        slope=(lo+hi)/2
    return slope,exp(log(actual)-moments(slope)[1])
