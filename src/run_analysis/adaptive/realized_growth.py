"""Offline schedule-matched growth signal; NOT a live progression controller.

Compare actual distance with an independently frozen NO-GROWTH dose at each
original opportunity. Fit an exponential level/trend to those paired events,
not to arbitrary trailing calendar totals. Missing opportunities contribute
zero; rest days contribute neither a zero nor an invented completion.

Reference doses must come from a fixed-level allocation, not issued distance
divided by the current target: the latter can credit growth to unchanged key
runs. This module deliberately refuses to manufacture that reference.
"""
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from math import exp, expm1, isfinite, log, log1p, fsum

from run_analysis.athlete_projection import aware
from run_analysis.general_progression import VOLUME_GROWTH_PER_7
from .smoothed_progression import days_between, matched_growth, advance

VERSION = 'schedule_matched_component_growth_prototype_v1'
REFERENCE_VERSION = 'demonstrated_reference_adoption_prototype_v3'


@dataclass(frozen=True)
class ReferenceDose:
    opportunity: str
    scheduled_at: datetime
    available_at: datetime
    miles: float
    baseline_key: str
    source: str
    basis: str = 'independent_no_growth_allocation_v1'
    component: str = 'total'
    component_share: float = 1.

    def __post_init__(self):
        aware(self.scheduled_at); aware(self.available_at)
        if (not self.opportunity or not self.baseline_key or not self.source
                or not isfinite(self.miles) or self.miles <= 0
                or self.available_at > self.scheduled_at
                or self.basis not in {'independent_no_growth_allocation_v1','diagnostic_easy_ramp_removal_v1'}
                or not self.component or not isfinite(self.component_share)
                or not 0 < self.component_share <= 1):
            raise ValueError('Positive independent no-growth allocation and causal provenance required')


def estimate(entries, references, *, as_of, zone, memory_days=14, supplemental=(), allow_diagnostic_reference=False):
    """Causal event fit with the existing 14-day exponential mean memory.

    Solve the distance-weighted exponential moment equations for level and
    slope. Unlike a log(actual) regression this admits genuinely missed (zero)
    doses. Level and slope are separate: a stable 5% level shortfall is not a
    falling trend. Supplied supplements are explicit (opportunity, completion)
    links; unlinked extra runs must not be guessed into a reference slot.

    There is no calendar quota, finite lookback cutoff or per-refresh growth
    credit. This returns a signal only; health, actual-level headroom, target
    advancement, rounding and atomic publication remain outside this prototype.
    """
    aware(as_of)
    if type(memory_days) is not int or memory_days <= 0:
        raise ValueError('Positive declared memory required')
    entries = tuple(entries); references = tuple(references); supplemental = tuple(supplemental)
    diagnostic = any(r.basis != 'independent_no_growth_allocation_v1' for r in references)
    if diagnostic and not allow_diagnostic_reference:
        raise ValueError('Approximate reference requires explicit diagnostic-only mode')
    if len({e.key for e in entries}) != len(entries):
        raise ValueError('Duplicate opportunity')
    if len({r.opportunity for r in references}) != len(references):
        raise ValueError('Duplicate reference allocation')
    known_refs = {r.opportunity: r for r in references if r.available_at <= as_of}
    done_keys = [e.completion.key for e in entries if e.completion]
    done_keys.extend(done.key for _, done in supplemental)
    if len(done_keys) != len(set(done_keys)):
        raise ValueError('Physical completion counted twice')
    extras = {}
    for key, done in supplemental:
        if key not in {e.key for e in entries}:
            raise ValueError('Supplement requires an explicit known opportunity link')
        extras.setdefault(key, []).append(done)
    included = []; unresolved = []; baselines = set()
    for e in entries:
        if e.contract.issued_at > as_of or e.scheduled_at > as_of:
            continue
        done = e.completion
        known = bool(done and done.available_at <= as_of and done.end <= as_of)
        expired = e.scheduled_at.astimezone(zone).date() < as_of.astimezone(zone).date()
        if not known and not expired:
            continue
        ref = known_refs.get(e.key)
        if ref is None:
            unresolved.append(e.key + ':missing_independent_reference'); continue
        if ref.scheduled_at != e.scheduled_at or ref.available_at > e.contract.issued_at:
            raise ValueError('Reference must precede original issuance and retain its schedule')
        baselines.add(ref.baseline_key)
        actual = 0.; physical_keys = []
        for observed in ([done] if known else []) + extras.get(e.key, []):
            if observed.available_at > as_of or observed.end > as_of:
                continue
            q = observed.distance_meters
            if q is None or q.low != q.high:
                unresolved.append(e.key + ':distance_uncertain'); continue
            actual += q.low / 1609.344; physical_keys.append(observed.key)
        included.append(dict(key=e.key, scheduled_at=e.scheduled_at.isoformat(),
            baseline_miles=ref.miles, actual_miles=actual, physical_keys=physical_keys,
            component=ref.component, component_share=ref.component_share))
    if len(baselines) > 1:
        raise ValueError('Cannot silently mix or reset fixed baseline references')
    result = dict(version=VERSION, as_of=as_of.isoformat(), memory_days=memory_days,
        baseline_key=next(iter(baselines), None), included=included, unresolved=unresolved,
        growth_per_7=0., level_ratio=None, status='insufficient_opportunities',
        planner_influence=False, physical_credit_added=0., calendar_cutoff=False,
        reference_quality='diagnostic_approximation' if diagnostic else 'declared_independent_allocation')
    if unresolved:
        return dict(result, status='evidence_unresolved')
    if len({r['scheduled_at'] for r in included}) < 2:
        return result
    # Use the last due opportunity as origin. Merely refreshing on a rest day
    # cannot extrapolate the fitted level or create new temporal observations.
    origin = max(datetime.fromisoformat(r['scheduled_at']) for r in included)
    result['last_opportunity_at'] = origin.isoformat()
    # A held long run and growing easy doses are different component curves.
    # Shares are fixed-level VOLUME allocations, not subjective score weights.
    groups = {}
    for row in included:
        groups.setdefault(row['component'], []).append(row)
    shares = {}
    for name, rows in groups.items():
        values = {r['component_share'] for r in rows}
        if len(values) != 1:
            raise ValueError('Component volume shares cannot change with the target')
        shares[name] = values.pop()
    if abs(fsum(shares.values())-1.) > 1e-10:
        return dict(result, status='component_coverage_incomplete')
    components = []
    for name, rows in groups.items():
        fitted = _fit_component(rows, origin, memory_days)
        if fitted is None:
            return dict(result, status='component_trend_unresolved')
        slope, level = fitted
        components.append(dict(component=name, share=shares[name], level_ratio=level,
            log_slope_per_day=slope, observations=len(rows)))
    level = fsum(c['share']*c['level_ratio'] for c in components)
    if level <= 0:
        return dict(result, status='no_delivered_distance', level_ratio=0., components=components)
    slope = fsum(c['share']*c['level_ratio']*c['log_slope_per_day'] for c in components)/level
    maximum = log1p(VOLUME_GROWTH_PER_7)/7
    rate = expm1(7*min(maximum,max(0.,slope)))
    status = 'flat_or_falling' if rate == 0 else 'growing_at_cap' if slope >= maximum else 'growing_below_cap'
    return dict(result, status=status, growth_per_7=rate, components=components,
        level_ratio=level, level_at=origin.isoformat(), fitted_log_slope_per_day=slope)


def _fit_component(rows, origin, memory_days):
    """Exponential dose trend, not an assumption of identical key/easy growth."""
    ts = [days_between(datetime.fromisoformat(r['scheduled_at']), origin) for r in rows]
    if len(set(ts)) < 2:
        return None
    weights = [exp(t/memory_days) for t in ts]
    actual = fsum(w*r['actual_miles'] for w, r in zip(weights, rows))
    if actual <= 0:
        return 0., 0.
    actual_mean = fsum(w*r['actual_miles']*t for w, r, t in zip(weights, rows, ts))/actual
    if not min(ts)+1e-12 < actual_mean < max(ts)-1e-12:
        return None  # All measured distance at one extreme cannot identify a finite trend.

    def moments(slope):
        logs = [t/memory_days+log(r['baseline_miles'])+slope*t for r,t in zip(rows,ts)]
        largest = max(logs)
        masses = [exp(v-largest) for v in logs]
        total = fsum(masses)
        return fsum(m*t for m,t in zip(masses,ts))/total, largest+log(total)

    flat, _ = moments(0.)
    if abs(actual_mean-flat) <= 1e-12:
        slope = 0.
    else:
        # Expanding numerical bracket, not a separate component growth policy.
        # Easy volume can grow faster than 4% when key doses are held. Only the
        # final combined TOTAL-volume rate is capped at the existing 4%.
        lo,hi = -log1p(VOLUME_GROWTH_PER_7)/7,log1p(VOLUME_GROWTH_PER_7)/7
        for _ in range(64):
            if moments(lo)[0] <= actual_mean <= moments(hi)[0]: break
            lo *= 2; hi *= 2
        else: return None
        for _ in range(64):
            mid=(lo+hi)/2
            if moments(mid)[0] < actual_mean: lo=mid
            else: hi=mid
        slope=(lo+hi)/2
    _,log_mass=moments(slope)
    return slope,exp(log(actual)-log_mass)


def shadow_proposal(volume, physical, signal, funding, *, baseline_volume, permitted=True,
                    maximum_growth_per_7=VOLUME_GROWTH_PER_7):
    """Unpublished advancement proposal; no live policy version/adoption.

    Demonstrated level can exceed the old target AND its issued-plan funding.
    The growth cap applies to additional planner-initiated progression, not
    recognition of already delivered volume. Funding remains attribution, not
    a ceiling on verified actuals. Unknown/approximate evidence cannot rebase.
    The fitted level is schedule-normalized distance, not physical fitness or
    a replacement for raw fatigue/endurance evidence.
    """
    if not isfinite(baseline_volume) or baseline_volume <= 0:
        raise ValueError('Positive fixed baseline volume required')
    from .cadence_mass_growth import VERSION as MASS_VERSION
    if signal['version'] not in {VERSION,MASS_VERSION}:
        raise ValueError('Explicit growth-signal reconciliation required')
    legacy = matched_growth(volume, physical, permitted=permitted,
        maximum_growth_per_7=maximum_growth_per_7)
    level = signal['level_ratio']
    observed_level = baseline_volume*level if level is not None else None
    funded = funding['funded_level']
    if (observed_level is not None and (not isfinite(observed_level) or observed_level < 0)
            or funded is not None and (not isfinite(funded) or funded <= 0)):
        raise ValueError('Finite nonnegative observed and positive known funded levels required')
    ceiling = observed_level*(1+legacy['support_envelope']) if observed_level is not None else volume
    rate = signal['growth_per_7']
    if not isfinite(rate) or not 0 <= rate <= maximum_growth_per_7+1e-12:
        raise ValueError('Declared capped planner growth rate required')
    status = signal['status']
    recognition_allowed = True
    if signal['reference_quality'] not in {'declared_independent_allocation','declared_calendar_mass'}:
        rate=0.; status='unverified_no_growth_reference'; recognition_allowed=False
    elif not permitted:
        rate=0.; status='response_veto'; recognition_allowed=False
    elif physical['actual_per_7'] <= 0:
        rate=0.; status='actual_exposure_unavailable'; recognition_allowed=False
    elif observed_level is None:
        rate=0.; status='reference_unavailable'; recognition_allowed=False
    elif ceiling <= volume+1e-10*volume:
        rate=0.; status='observed_headroom_exhausted'
    recognized = max(volume,observed_level) if recognition_allowed else volume
    return dict(current_volume=volume, growth_per_7=rate,
        supported_volume_ceiling=ceiling, support_fraction=rate/maximum_growth_per_7,
        long_support_fraction=legacy['support_fraction'], support_envelope=legacy['support_envelope'],
        rate_contract=REFERENCE_VERSION, rate_status=status, signal=signal,
        schedule_normalized_observed_level=observed_level,
        recognized_planning_floor=recognized, recognition_allowed=recognition_allowed,
        recognition_increase=recognized-volume, planner_funding=funding,
        observed_above_funded_level=bool(observed_level is not None and funded is not None and observed_level>funded),
        physical_credit_added=0., planner_influence=False)


def review_reference(previous, *, volume, physical, signal, funding, baseline_volume,
                     as_of, zone, destination, permitted=True, decision_day=None,
                     maximum_growth_per_7=VOLUME_GROWTH_PER_7):
    """Stateful OFFLINE review: recognize actuals once, then grow prospectively.

    First advance the PRIOR planning base over elapsed days at the previously
    authorized rate. Then recognize a higher demonstrated current level. Do
    not compound past elapsed time on the newly recognized level: it describes
    today's evidence, not a baseline that existed at the prior review.

    The fixed no-growth normalization reference never changes when actuals are
    recognized. Repeating a review, resuming, or resting cannot multiply an
    overshoot again. Health/evidence holds consume elapsed time, not bank it.
    This is not an atomic calendar publisher or a live lifecycle migration.
    """
    aware(as_of)
    if datetime.fromisoformat(signal['as_of']) != as_of:
        raise ValueError('Signal must describe this causal review timestamp')
    today = as_of.astimezone(zone).date()
    day = today if decision_day is None else decision_day
    if type(day) is not date or day not in (today,today+timedelta(days=1)):
        raise ValueError('Decision day must be today or the next unrun day')
    if previous:
        if previous['version'] != REFERENCE_VERSION:
            raise ValueError('Explicit reference-policy reconciliation required')
        if as_of < datetime.fromisoformat(previous['available_at']):
            raise ValueError('Reference review cannot precede committed evidence')
        if volume != previous['volume']:
            raise ValueError('Current planning volume must match committed state')
        if (previous['baseline_volume'] != baseline_volume
                or previous['baseline_key'] is not None and signal['baseline_key'] is not None
                and previous['baseline_key'] != signal['baseline_key']):
            raise ValueError('Recognition cannot reset the fixed normalization baseline')
        if as_of == datetime.fromisoformat(previous['available_at']) and signal != previous['signal']:
            raise ValueError('Same-time evidence change needs explicit reconciliation')
    proposal = shadow_proposal(volume,physical,signal,funding,
        baseline_volume=baseline_volume,permitted=permitted,maximum_growth_per_7=maximum_growth_per_7)
    elapsed = (day-date.fromisoformat(previous['day'])).days if previous else 0
    if elapsed < 0:
        raise ValueError('Decision clock cannot move backwards')
    accrued = volume
    if previous and elapsed:
        step = dict(proposal,growth_per_7=min(previous['growth_per_7'],proposal['growth_per_7']),
            supported_volume_ceiling=min(previous['supported_volume_ceiling'],proposal['supported_volume_ceiling']))
        accrued = advance(volume,step,elapsed_days=elapsed,destination=destination)
    elif not isfinite(destination) or destination <= 0:
        raise ValueError('Positive destination required')
    floor = max(volume,min(destination,proposal['recognized_planning_floor']))
    updated = max(accrued,floor)
    # Reprice future slope/headroom from the recognized current level, NOT the
    # old lower prescription. The historical normalization remains untouched.
    future = shadow_proposal(updated,physical,signal,funding,
        baseline_volume=baseline_volume,permitted=permitted,maximum_growth_per_7=maximum_growth_per_7)
    baseline_key = previous['baseline_key'] if previous and previous['baseline_key'] is not None else signal['baseline_key']
    state = dict(version=REFERENCE_VERSION,available_at=as_of.isoformat(),day=day.isoformat(),
        volume=updated,baseline_volume=baseline_volume,baseline_key=baseline_key,
        observed_reference_volume=future['schedule_normalized_observed_level'],
        growth_per_7=future['growth_per_7'],supported_volume_ceiling=future['supported_volume_ceiling'],
        rate_status=future['rate_status'],signal=signal)
    return dict(state=state,proposal=future,prior_volume=volume,
        elapsed_days=elapsed,prior_base_accrual=accrued-volume,
        newly_recognized_increase=updated-accrued,
        physical_credit_added=0.,planner_influence=False)
