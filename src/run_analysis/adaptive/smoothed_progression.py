"""Shadow progression controller; no candidate mileage, search or publication.

Three cascaded exponential smoothers give a continuous age kernel: a completion
enters gradually and old evidence fades without a rectangular-window cliff.
Mean evidence age is the existing review duration (tau = review_days / 3).
This memory and the one-growth-step support envelope are coaching assumptions,
not fitted physiology. Physical fatigue accounting is deliberately untouched.
"""
from datetime import datetime, time, timedelta, timezone
from math import exp, expm1, isfinite, log1p

from run_analysis.athlete_projection import aware
from run_analysis.general_progression import VOLUME_GROWTH_PER_7
from run_analysis.stage_target_history import target_segments
from run_analysis.staged_progression import MILE, progression_distance_credit

VERSION = 'smoothed_delivery_v1'


def days_between(later, earlier):
    aware(later); aware(earlier)
    return (later.astimezone(timezone.utc)-earlier.astimezone(timezone.utc)).total_seconds()/86400


def kernel(age, tau):
    """Unit-mass gamma(3, tau) kernel; continuous value/slope at age zero."""
    x=max(0.,age)/tau
    return x*x*exp(-x)/(2*tau)


def mass(age, tau):
    x=max(0., age)/tau
    # Cancellation-safe near zero; asymptotic values are intentionally bounded.
    if x<1e-4:
        return x*x*x*(1/6-x/8+x*x/20-x*x*x/72)
    return -expm1(-x)-(x+x*x/2)*exp(-x)


def support(entries, targets, *, as_of, anchor, zone, review_days=14):
    """Compare actual and dated target through the SAME causal kernel.

    No prescribed/future dose is physical credit. Issued volume is a separate
    attribution channel only. Lower measurement bounds own actual credit.
    The fixed anchor is not moved on refresh; startup normalization uses only
    elapsed coverage, never invented prehistory or zero-padded full windows.
    """
    aware(as_of); aware(anchor)
    if type(review_days) is not int or review_days<=0 or as_of<=anchor:
        raise ValueError('Positive memory and elapsed anchored evidence required')
    entries=tuple(entries)
    if len({e.key for e in entries})!=len(entries):
        raise ValueError('Duplicate opportunity')
    keys=[e.completion.key for e in entries if e.completion]
    if len(keys)!=len(set(keys)):
        raise ValueError('Completion counted twice')
    tau=review_days/3
    normalization=mass(days_between(as_of,anchor),tau)
    segments,_=target_segments(targets,as_of=as_of,
        start=anchor.astimezone(zone).date(),end=as_of.astimezone(zone).date()+timedelta(days=1))
    weighted_target=0.
    for row in segments:
        left=max(anchor,datetime.combine(row.start,time.min,zone))
        right=min(as_of,datetime.combine(row.end,time.min,zone))
        if left<right:
            weighted_target+=row.miles_per_7_days*(
                mass(days_between(as_of,left),tau)-mass(days_between(as_of,right),tau))
    actual=issued=0.; known_issued=True; unresolved=[]; included=[]
    for entry in entries:
        if entry.contract.issued_at>as_of:
            continue
        done=entry.completion
        usable=bool(done and done.available_at<=as_of and done.end<=as_of
            and anchor<=done.start<as_of)
        distance=next((t.quantity.estimate for t in entry.contract.targets
            if t.metric=='distance_meters'),None)
        # Intention keeps its scheduled timestamp even when the run is missed;
        # actual exposure uses completion time. Intention is attribution ONLY.
        if anchor<=entry.scheduled_at<=as_of:
            if distance is None:
                known_issued=False
            else:
                issued+=distance/MILE*7*kernel(days_between(as_of,entry.scheduled_at),tau)
        if not usable:
            continue
        weight=7*kernel(days_between(as_of,done.end),tau)
        credit=progression_distance_credit(entry.contract,done.distance_meters,basis='actual_distance_v2')
        if credit['status'] in {'distance_evidence_unresolved','issued_distance_target_unresolved'}:
            unresolved.append(entry.key)
        actual+=credit['credited_miles']*weight
        included.append(entry.key)
    target=weighted_target/normalization
    actual/=normalization
    issued=issued/normalization if known_issued else None
    if not isfinite(target) or target<=0:
        raise ValueError('Positive causally available target evidence required')
    return dict(version=VERSION,as_of=as_of.isoformat(),anchor=anchor.isoformat(),
        mean_age_days=review_days,kernel_order=3,time_constant_days=tau,elapsed_mass=normalization,
        actual_per_7=actual,target_per_7=target,issued_per_7=issued,
        delivery_ratio=actual/target,
        planner_ratio=issued/target if issued is not None else None,
        execution_ratio=actual/issued if issued else None,
        completed_keys=included,unresolved=unresolved,
        actual_credit_basis='actual_distance_v2',future_credit=False,
        interpretation='Smoothed exposure rates, not measured fitness or fatigue; issued/actual ratio includes timing, not distance adherence alone')


def growth(volume, evidence, *, permitted=True, maximum_growth_per_7=VOLUME_GROWTH_PER_7):
    """Smooth support inside one memory-duration growth envelope.

    Matching the dated exposure path can support the existing maximum rate,
    subject to actual-exposure headroom (never an ever-rising target alone).
    Underfunding tapers that rate to zero, rather than switching at 100%.
    The lower edge uses the existing growth rate over the evidence memory,
    not a newly tuned percentage or a seven-day step on fourteen-day evidence.
    Repeated shortfalls cannot buy indefinite unsupported growth:
    as dated targets rise without actuals, the support ratio falls to zero rate.
    Neither candidate schedules nor issued-but-uncompleted miles enter here.
    """
    ratio=evidence['delivery_ratio']
    if (not isfinite(volume) or volume<=0 or not isfinite(ratio) or ratio<0
            or not isfinite(evidence['actual_per_7']) or evidence['actual_per_7']<0
            or not isfinite(evidence['mean_age_days']) or evidence['mean_age_days']<=0):
        raise ValueError('Positive current target and finite nonnegative actual support required')
    maximum=maximum_growth_per_7
    if not isfinite(maximum) or not 0 < maximum < 1:
        raise ValueError('Positive fractional coaching growth rate required')
    envelope=expm1(log1p(maximum)*evidence['mean_age_days']/7)
    dated_ceiling=volume*(1+envelope)*ratio
    observed_ceiling=evidence['actual_per_7']*(1+envelope)
    ceiling=min(dated_ceiling,observed_ceiling)
    ratio_support=min(1.,max(0.,(dated_ceiling-volume)/(volume*envelope)))
    daily_step=volume*expm1(log1p(maximum)/7)
    headroom=min(1.,max(0.,(ceiling-volume)/daily_step))
    fraction=min(ratio_support,headroom) if permitted else 0.
    return dict(support_fraction=fraction,growth_per_7=expm1(log1p(maximum)*fraction),
        supported_volume_ceiling=ceiling,maximum_growth_per_7=maximum,
        support_envelope=envelope,
        observed_volume_ceiling=observed_ceiling,
        current_volume=volume,permitted=permitted,
        future_growth_is_conditional=True,physical_credit_added=0.)


def advance(volume, proposal, *, elapsed_days, destination):
    """Bounded numerical step, not an approval or target/calendar publication.

    Intraday refreshes earn zero elapsed days. Long unattended gaps cannot
    compound beyond CURRENT observed headroom. No downward rebase or catch-up
    bank is introduced. The full lifecycle must separately own response,
    endurance, session guards and atomic publication before any live adoption.
    """
    if (type(elapsed_days) is not int or elapsed_days<0 or not isfinite(destination)
            or destination<=0 or proposal['current_volume']!=volume):
        raise ValueError('Whole elapsed days and matching frozen support proposal required')
    cap=max(volume,min(destination,proposal['supported_volume_ceiling']))
    if not proposal['growth_per_7'] or not elapsed_days or cap==volume:
        return volume
    from math import log
    days=min(elapsed_days,7*log(cap/volume)/log1p(proposal['growth_per_7']))
    return min(cap,volume*exp(log1p(proposal['growth_per_7'])*days/7))


def matched_growth(volume, evidence, *, permitted=True, maximum_growth_per_7=VOLUME_GROWTH_PER_7):
    """V4 proposal: one lag-matched support comparison, not two time bases.

    Actual exposure and the original dated path use the same causal memory.
    Their ratio adjusts a CONDITIONAL planning level; it is not an estimate of
    current physical volume/capacity. Rising dated references still reduce that
    ratio when actual training stays flat. No raw smoothed level is compared
    with today's ahead-of-memory forecast endpoint a second time.

    The existing memory, 4% cap and 8.16% response envelope remain explicit
    coaching assumptions. This does not establish fitness or future delivery.
    """
    # Reuse validation and the unchanged matched response curve, not its
    # separate unlagged-endpoint/lagged-actual ceiling.
    checked=growth(volume,evidence,permitted=permitted,maximum_growth_per_7=maximum_growth_per_7)
    envelope=checked['support_envelope']
    ceiling=volume*(1+envelope)*evidence['delivery_ratio']
    fraction=min(1.,max(0.,(ceiling-volume)/(volume*envelope))) if permitted else 0.
    return dict(support_fraction=fraction,growth_per_7=expm1(log1p(maximum_growth_per_7)*fraction),
        supported_volume_ceiling=ceiling,maximum_growth_per_7=maximum_growth_per_7,
        support_envelope=envelope,current_volume=volume,permitted=permitted,
        ceiling_basis='matched_dated_delivery_v1',observed_volume=evidence['actual_per_7'],
        future_growth_is_conditional=True,physical_credit_added=0.)


def nominal_growth(volume, evidence, *, permitted=True, maximum_growth_per_7=VOLUME_GROWTH_PER_7):
    """V5: nominal rate inside observed headroom; no shortfall rate multiplier.

    Keep V4's matched actual/datetime support ceiling and validation. The
    bounded advance still clips at that ceiling and a health veto still holds.
    Small historical mismatches no longer repeatedly scale the growth rate.
    This is NOT unconditional growth: exhausted headroom stops it. Removing
    the taper changes transient flat-training headroom; test it explicitly.
    """
    matched=matched_growth(volume,evidence,permitted=permitted,maximum_growth_per_7=maximum_growth_per_7)
    allowed=bool(permitted and evidence['actual_per_7']>0
        and matched['supported_volume_ceiling']>volume+1e-10*volume)
    return dict(matched,support_fraction=float(allowed),
        growth_per_7=maximum_growth_per_7 if allowed else 0.,
        long_support_fraction=matched['support_fraction'],
        rate_contract='nominal_4pct_matched_headroom_v1',
        rate_status='nominal' if allowed else 'response_veto' if not permitted else 'observed_headroom_exhausted')
