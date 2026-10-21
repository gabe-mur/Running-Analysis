"""Separate execution fulfillment from planner funding; no physical credit.

The normal growth margin is spent only once: (1 + nominal_growth) *
fulfilled_fraction - 1. A historical actual/continuous-target ratio is not
an adherence score. Published plan volume is intent backing, never fitness.
"""
from datetime import date,datetime,time
from math import exp,isfinite

from run_analysis.athlete_projection import aware
from run_analysis.general_progression import VOLUME_GROWTH_PER_7
from run_analysis.stage_target_history import StageTarget,target_segments,target_miles_between
from .smoothed_progression import days_between,matched_growth,nominal_growth

VERSION='prescription_fulfillment_v1'


def fulfillment(entries,*,as_of,epoch,zone,review_days=14,records=(),distance_basis='issued_nominal_v1'):
    """Paired causal quantities, same clock/weight, no same-day missed run.

    Mean age remains review_days; unlike physical-exposure gamma smoothing,
    execution evidence uses an exponential with nonzero weight at receipt.
    This is an explicit responsive-adherence assumption, not physiology.
    The issued target owns fulfillment. Internal comparison bounds are not
    a prescribed range. Historical physical exposure is never rewritten.
    """
    aware(as_of);aware(epoch)
    if epoch>as_of or type(review_days) is not int or review_days<=0:
        raise ValueError('Causal fixed adoption epoch and positive memory required')
    if distance_basis!='issued_nominal_v1':
        raise ValueError('Issued target distance required; experimental minimum-basis state needs explicit reconciliation')
    entries=tuple(entries);records=tuple(records)
    if len({e.key for e in entries})!=len(entries):raise ValueError('Duplicate opportunity')
    done_keys=[e.completion.key for e in entries if e.completion]
    if len(done_keys)!=len(set(done_keys)):raise ValueError('Completion counted twice')
    if len({r.key for r in records})!=len(records):raise ValueError('Duplicate physical history')
    actual=required=nominal=0.;included=[];unresolved=[];used=set()
    for e in entries:
        if e.contract.issued_at>as_of or e.scheduled_at<epoch:continue
        done=e.completion
        known=bool(done and done.available_at<=as_of and done.end<=as_of)
        expired=e.scheduled_at.astimezone(zone).date()<as_of.astimezone(zone).date()
        if not known and not expired:continue
        q=next((t.quantity for t in e.contract.targets if t.metric=='distance_meters'),None)
        if q is None or q.estimate is None or q.estimate<=0:
            unresolved.append(e.key);continue
        if known and done.distance_meters is None:
            unresolved.append(e.key);used.add(done.key);continue
        weight=exp(-max(0.,days_between(as_of,e.scheduled_at))/review_days)
        requirement=q.estimate
        if known and done.distance_meters.low<requirement<=done.distance_meters.high:
            unresolved.append(e.key)
        required+=requirement/1609.344*weight;nominal+=q.estimate/1609.344*weight
        if known:
            actual+=done.distance_meters.low/1609.344*weight;used.add(done.key)
        included.append(dict(key=e.key,completed=known,scheduled=e.scheduled_at.isoformat(),
            weight=weight,required_miles=requirement/1609.344,issued_low_miles=q.low/1609.344,
            nominal_miles=q.estimate/1609.344,
            actual_miles=done.distance_meters.low/1609.344 if known else 0.))
    # Independently measured extra runs count once, without invented intent.
    extras=[]
    for r in records:
        if r.key not in used and epoch<=r.start<as_of and r.end<=as_of and r.available_at<=as_of:
            weight=exp(-days_between(as_of,r.start)/review_days)
            actual+=r.miles*weight;extras.append(r.key)
    return dict(version=VERSION,epoch=epoch.isoformat(),as_of=as_of.isoformat(),
        memory_days=review_days,distance_basis=distance_basis,weighted_actual=actual,weighted_required=required,
        weighted_nominal=nominal,fulfillment_ratio=actual/required if required else None,
        nominal_execution_ratio=actual/nominal if nominal else None,
        included=included,extra_physical_keys=extras,unresolved=unresolved,
        future_credit=False,physical_credit_added=0.)


def published_funding(event,*,as_of,zone):
    """Frozen whole issued horizon, not an arbitrary past-week slice.

    Equivalent start level = issued nominal miles / original dated path's
    volume-per-unit-start. Target increases cannot inflate this when the
    issued programme stays flat. Future doses remain intent, not actuals.
    """
    if event is None:return dict(status='unavailable',funded_level=None)
    dt=lambda value:datetime.fromisoformat(value) if isinstance(value,str) else value
    day=lambda value:date.fromisoformat(value) if isinstance(value,str) else value
    issued=dt(event['as_of'])
    if issued>as_of:raise ValueError('Future plan cannot back current growth')
    raw=event['stage_runtime']['target_history']
    paths=tuple(StageTarget(**dict(r,available_at=dt(r['available_at']),
        effective_on=day(r['effective_on']),
        review_anchor=dt(r['review_anchor']) if r.get('review_anchor') else None)) for r in raw)
    start=day(event['decision']);end=day(event['hard_end'])
    if end<=start:raise ValueError('Positive issued horizon required')
    segments,_=target_segments(paths,as_of=issued,start=start,end=end)
    expected=target_miles_between(segments,datetime.combine(start,time.min,zone),
        datetime.combine(end,time.min,zone),zone)
    level=event['progression_target']['planning_volume_per_7'] if event.get('progression_target') else paths[-1].rate(start,'volume_per_7')
    issued_miles=event['volume']
    if not all(isfinite(x) and x>0 for x in (expected,level,issued_miles)):
        raise ValueError('Positive frozen issued and dated volume required')
    return dict(status='known',event=event['index'],published_at=event['as_of'],
        start=start.isoformat(),end=end.isoformat(),issued_miles=issued_miles,
        dated_target_miles=expected,funding_ratio=issued_miles/expected,
        funded_level=level*issued_miles/expected,physical_credit_added=0.)


def growth(volume,physical,execution,funding,*,permitted=True):
    """Full fulfillment permits nominal growth; real shortfall spends margin.

    Missing 1% of required dose supports approximately 3%, not 4%.
    Unknown measurement does not become successful adherence. No post-adoption
    due slots is provisional, not a fabricated perfect-compliance observation.
    The independent planner ceiling also bounds fully followed FLAT plans.
    """
    legacy=matched_growth(volume,physical,permitted=permitted)
    ratio=execution['fulfillment_ratio']
    rate=(VOLUME_GROWTH_PER_7 if ratio is None else
          min(VOLUME_GROWTH_PER_7,max(0.,(1+VOLUME_GROWTH_PER_7)*ratio-1)))
    ceiling=(funding['funded_level']*(1+legacy['support_envelope'])
             if funding['funded_level'] is not None else volume)
    status='provisional_no_due_prescription' if ratio is None else 'fulfilled' if ratio>=1 else 'execution_shortfall'
    if not permitted:rate=0.;status='response_veto'
    elif physical['actual_per_7']<=0:rate=0.;status='actual_exposure_unavailable'
    elif execution['unresolved']:rate=0.;status='execution_evidence_unresolved'
    elif ceiling<=volume+1e-10*volume:rate=0.;status='planner_funding_exhausted'
    if ratio is None:
        # Bootstrap still has an observed backstop until actual paired evidence.
        ceiling=min(ceiling,nominal_growth(volume,physical,permitted=permitted)['supported_volume_ceiling'])
        if ceiling<=volume+1e-10*volume:rate=0.;status='bootstrap_observed_support_exhausted'
    return dict(current_volume=volume,growth_per_7=rate,
        support_fraction=rate/VOLUME_GROWTH_PER_7,supported_volume_ceiling=ceiling,
        long_support_fraction=legacy['support_fraction'],maximum_growth_per_7=VOLUME_GROWTH_PER_7,
        support_envelope=legacy['support_envelope'],rate_status=status,
        rate_contract=VERSION,ceiling_basis='frozen_issued_programme_v1',
        execution=execution,planner_funding=funding,
        future_growth_is_conditional=True,physical_credit_added=0.)
