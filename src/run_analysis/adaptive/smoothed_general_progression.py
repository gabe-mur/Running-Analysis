"""Explicit V2 lifecycle proposal; existing atomic publisher owns acceptance.

Shared health/history/session safeguards are kept. The rectangular volume test
is diagnostic only; causal smoothed delivered/dated-target support owns growth.
Endurance evidence and actual supporting exposure still gate LONG growth.
"""
from dataclasses import replace
from datetime import date,datetime,timedelta,timezone
from math import isfinite

from run_analysis.general_progression import destination,LONG_SHARE,LONG_GROWTH_PER_REVIEW
from run_analysis.stage_target_history import StageTarget,target_segments,target_miles_between
from run_analysis.staged_progression import review_recent_delivery,tenth
from .smoothed_progression import support,growth,advance,matched_growth,nominal_growth

VERSION='continuous_general_v2'
HISTORICAL_VERSION='continuous_general_v3'
MATCHED_VERSION='continuous_general_v4'
NOMINAL_VERSION='continuous_general_v5'
PRESCRIPTION_VERSION='continuous_general_v6'
REALIZED_VERSION='continuous_general_v7'


def propose(h,as_of,entries,*,responses=(),longest=None):
    day=h.decision(as_of); entries=tuple(entries)
    previous=getattr(h,'general_progression_state',None)
    if previous and as_of<datetime.fromisoformat(previous['available_at']):
        raise ValueError('Continuous review cannot precede its committed evidence')
    realized=h.policy.general_progression==REALIZED_VERSION
    if realized:
        from .progression_policy import resolve
        coaching=resolve(h,as_of)
    paired=h.policy.general_progression==PRESCRIPTION_VERSION
    nominal=h.policy.general_progression in {NOMINAL_VERSION,PRESCRIPTION_VERSION,REALIZED_VERSION}
    matched=h.policy.general_progression in {MATCHED_VERSION,NOMINAL_VERSION,PRESCRIPTION_VERSION,REALIZED_VERSION}
    historical=h.policy.general_progression in {HISTORICAL_VERSION,MATCHED_VERSION,NOMINAL_VERSION,PRESCRIPTION_VERSION,REALIZED_VERSION}
    growth_proposal=nominal_growth if nominal else matched_growth if matched else growth
    paths=[t for t in h.target_history if t.available_at<=as_of] if matched else h.target_history
    if not paths:
        raise ValueError('Known dated progression reference required')
    path=paths[-1]
    anchor=datetime.fromisoformat(previous['observation_anchor']) if previous else h.stage.issued_at
    initialized=previous.get('historical_initialization') if previous else None
    if initialized and initialized.get('version')!='historical_observed_v1':
        raise ValueError('Unknown historical initialization needs explicit reconciliation')
    records=()
    if historical:
        from .historical_progression import observations
        from .race_event import physical_keys
        excluded=set(physical_keys(h))|{k for k,v in getattr(h,'capacity_eligibility',{}).items() if not v}
        records=observations(getattr(h,'history',()),entries,as_of,excluded_keys=excluded)
        if records:
            # Import/adoption is not the start of an experienced runner's
            # observation period. Only real already-available history moves
            # this anchor earlier, never elapsed time or future prescriptions.
            observed_start=min(r.start for r in records)
            anchor=min(datetime.fromisoformat(initialized['history_start']),observed_start) if initialized else observed_start
    accounting_start=date.fromisoformat(previous['accounting_start']) if previous else path.effective_on
    volume=previous['volume'] if previous else path.rate(day,'volume_per_7')
    easy=previous['support'] if previous else path.rate(day,'supporting_easy_per_7')
    elapsed=max(0,(day-date.fromisoformat(previous['day'])).days) if previous else 0
    known_targets=[t for t in h.target_history if t.available_at<=as_of]
    reference_start=datetime.combine(min(t.effective_on for t in known_targets),datetime.min.time(),h.zone)
    since=max(anchor,as_of-timedelta(days=h.stage.policy.review_days),reference_start)
    required=None
    if since<as_of:
        segments,_=target_segments(h.target_history,as_of=as_of,
            start=since.astimezone(h.zone).date(),end=day+timedelta(days=1))
        required=target_miles_between(segments,since,as_of,h.zone)
    # Reuse the existing response/minimum-history/duplicate/credit interpretation.
    # A rolling-volume failure is no longer a veto on smoothed volume support.
    review=review_recent_delivery(h.stage,entries,as_of,ledger_complete=True,
        longest_prior_30d_miles=longest,responses=responses,
        distance_credit=h.policy.progression_credit,observation_anchor=anchor,required_miles=required)
    veto=review['status'] not in {'observed_delivery_supported','recent_delivery_hold_targets',
        'endurance_reference_not_represented','session_guard_reference_unavailable',
        'held_long_requires_session_guard_review'}
    if realized and not coaching.growth_allowed:
        veto=True
    if longest is None or not isfinite(longest) or longest<=0:
        veto=True
        if not review.get('response_state') or review['status'] in {
                'observed_delivery_supported','recent_delivery_hold_targets',
                'endurance_reference_not_represented'}:
            review=dict(review,status='session_guard_reference_unavailable')
    elif h.stage.long_miles>tenth(longest*1.1,down=True):
        veto=True
        if review['status'] in {'observed_delivery_supported','recent_delivery_hold_targets',
                'endurance_reference_not_represented'}:
            review=dict(review,status='held_long_requires_session_guard_review')
    if historical and records:
        from .historical_progression import support as historical_support
        evidence=historical_support(records,entries,h.target_history,as_of=as_of,zone=h.zone,
            review_days=h.stage.policy.review_days)
    else:
        evidence=support(entries,h.target_history,as_of=as_of,anchor=anchor,zone=h.zone,
            review_days=h.stage.policy.review_days) if as_of>anchor else None
    if paired:
        from .prescription_progression import fulfillment,published_funding,growth as paired_growth
        # Explicit policy adoption is fixed, not a sliding amnesty. The
        # accepted level already carries its pre-adoption interpretation;
        # all old physical history still owns exposure/long/response guards.
        epoch=datetime.fromisoformat(previous['execution_epoch']) if previous and previous.get('execution_epoch') else as_of
        distance_basis=previous.get('execution_distance_basis','issued_nominal_v1') if previous else 'issued_nominal_v1'
        execution=fulfillment(entries,as_of=as_of,epoch=epoch,zone=h.zone,
            review_days=h.stage.policy.review_days,records=records,distance_basis=distance_basis)
        published=[e for e in getattr(h,'events',()) if datetime.fromisoformat(e['as_of'])<=as_of]
        funding=published_funding(published[-1] if published else None,as_of=as_of,zone=h.zone)
        growth_proposal=lambda level,physical,permitted=True:paired_growth(
            level,physical,execution,funding,permitted=permitted)
    if realized:
        from .realized_reference_runtime import initialize,signal as reference_signal
        from .realized_growth import shadow_proposal,review_reference,REFERENCE_VERSION
        from .prescription_progression import published_funding
        reference_config=initialize(h,as_of)
        rate_args=({'maximum_growth_per_7':coaching.volume_growth_per_7}
            if coaching.volume_growth_per_7 != .04 else {})
        signal=reference_signal(h,entries,reference_config,as_of,**rate_args)
        published=[e for e in getattr(h,'events',()) if datetime.fromisoformat(e['as_of'])<=as_of]
        funding=published_funding(published[-1] if published else None,as_of=as_of,zone=h.zone)
        controller=previous.get('reference_controller') if previous and previous['version']==REALIZED_VERSION else None
        provisional=controller is None and signal['status'] in {
            'insufficient_opportunities','component_coverage_incomplete','component_trend_unresolved',
            'insufficient_calendar_exposure','calendar_trend_unresolved'}
        def growth_proposal(level,physical,permitted=True):
            if provisional:
                missing=any(r['actual_miles']==0 for r in signal['included'])
                p=nominal_growth(level,physical,permitted=permitted and not missing,
                    maximum_growth_per_7=coaching.volume_growth_per_7)
                return dict(p,rate_contract=REFERENCE_VERSION,
                    rate_status='provisional_missing_opportunity_hold' if missing else 'provisional_reference_observations',
                    signal=signal)
            return shadow_proposal(level,physical,signal,funding,
                baseline_volume=reference_config['volume'],permitted=permitted,
                maximum_growth_per_7=coaching.volume_growth_per_7)
    proposal=growth_proposal(volume,evidence,permitted=not veto) if evidence else dict(
        current_volume=volume,growth_per_7=0.,support_fraction=0.,supported_volume_ceiling=volume)
    goal=destination(h.stage if not h.planner_preferences else replace(h.stage,
        goal_long_miles=h.planner_preferences.goal_long_miles))
    if realized:
        goal=coaching.target
    long_share=coaching.long_share if realized else LONG_SHARE
    historical_longs=tuple(r for r in records if r.role=='long'
        and as_of-timedelta(days=30)<=r.start<as_of)
    initialize_now=bool(historical and (not initialized or not initialized['ready'])
        and evidence and not veto and evidence.get('observed_history_days',0)>=h.stage.policy.review_days)
    if initialize_now:
        before=volume
        # Recognize an already-demonstrated level, NOT an unperformed growth
        # envelope or retroactive 4%-per-week compounding. Never erase a prior
        # accepted level just because today's smoothing estimate is lower.
        recognized=volume if realized else max(volume,min(goal.volume_per_7,evidence['actual_per_7']))
        deferred=recognized>volume and any(t.effective_on==day for t in known_targets)
        volume=volume if deferred else recognized
        easy+=volume-before
        proposal=growth_proposal(volume,evidence,permitted=True)
        initialized=dict(version='historical_observed_v1',ready=True,
            adopted_at=as_of.isoformat(),history_start=anchor.isoformat(),
            recorded_runs=len(records),prior_volume=before,recognized_volume=recognized,
            recognized_increase=recognized-before,observed_volume=evidence['actual_per_7'],
            baseline_applied=not deferred,effective_on=(day+timedelta(days=1) if deferred else day).isoformat(),
            hypothetical_growth_credit=0.)
    elif historical and not initialized:
        initialized=dict(version='historical_observed_v1',ready=False,
            adopted_at=as_of.isoformat(),history_start=anchor.isoformat(),recorded_runs=len(records))
    lifecycle_version=REALIZED_VERSION if realized else PRESCRIPTION_VERSION if paired else NOMINAL_VERSION if nominal else MATCHED_VERSION if matched else HISTORICAL_VERSION if historical else VERSION
    apply_pending=bool(historical and initialized['ready'] and not initialized['baseline_applied']
        and day>=date.fromisoformat(initialized['effective_on']) and not veto)
    if apply_pending:
        before=volume
        volume=max(volume,initialized['recognized_volume'])
        easy+=volume-before
        proposal=growth_proposal(volume,evidence,permitted=True)
        initialized=dict(initialized,baseline_applied=True)
    # Yesterday's fractional authorization may advance today's target. A new
    # veto/unsupported gap cannot bank days. Migration starts at the same target.
    if previous and previous['version']==lifecycle_version and elapsed and not veto and not initialize_now and not apply_pending and (not realized or provisional):
        authorized=min(previous['growth_per_7'],proposal['growth_per_7'])
        step=dict(proposal,growth_per_7=authorized,
            supported_volume_ceiling=min(previous['supported_volume_ceiling'],proposal['supported_volume_ceiling']))
        new=advance(volume,step,elapsed_days=elapsed,destination=goal.volume_per_7)
        easy+=new-volume; volume=new
        proposal=growth_proposal(volume,evidence,permitted=True)
    reference_step=None;deferred_reference=None
    if realized and not provisional and evidence:
        reference_step=review_reference(controller,volume=volume,physical=evidence,signal=signal,
            funding=funding,baseline_volume=reference_config['volume'],as_of=as_of,zone=h.zone,
            destination=goal.volume_per_7,permitted=not veto,decision_day=day,
            maximum_growth_per_7=coaching.volume_growth_per_7)
        new=reference_step['state']['volume']
        # Preserve immutable same-day targets. Recognition is retained in the
        # signal and may apply on the next unpriced decision day, not erased.
        known_day=[p for p in known_targets if p.effective_on==day]
        if known_day and abs(known_day[-1].rate(day,'volume_per_7')-new)>1e-9:
            deferred_reference=new
            new=known_day[-1].rate(day,'volume_per_7')
            reference_step['state']['volume']=new
        easy+=new-volume; volume=new
        proposal=growth_proposal(volume,evidence,permitted=not veto)
    actual=evidence['actual_per_7'] if evidence else 0.
    # The existing endurance/session guards stay separate from volume support.
    support_ok=bool(not veto and (review.get('endurance_keys') or historical_longs)
        and min(volume,actual)+1e-8>=h.stage.long_miles/long_share)
    budget=previous['long_budget'] if previous else h.stage.long_miles
    if (initialize_now or apply_pending) and support_ok and historical_longs:
        budget=max(budget,min(goal.long_miles,min(volume,actual)*long_share,
            max(r.miles for r in historical_longs)))
    if support_ok and previous and previous['version']==lifecycle_version and previous['long_growth_allowed'] and elapsed and not initialize_now and not apply_pending:
        long_fraction=coaching.long_growth_fraction if realized else LONG_GROWTH_PER_REVIEW
        long_days=coaching.long_growth_days if realized else 14
        prior=previous.get('coaching_progression')
        if realized and prior is None and coaching.phase!='general_fitness':
            prior=dict(long_growth_fraction=LONG_GROWTH_PER_REVIEW,long_growth_days=14)
        if realized and prior:
            # A faster newly selected policy cannot backdate its higher rate.
            long_fraction=min(long_fraction,(1+prior['long_growth_fraction'])**(long_days/prior['long_growth_days'])-1)
        budget*=(1+long_fraction)**(min(elapsed,h.stage.policy.review_days)*
            proposal.get('long_support_fraction',proposal['support_fraction'])/long_days)
    elif not support_ok:
        budget=h.stage.long_miles
    guard=tenth(longest*1.1,down=True) if longest and isfinite(longest) and longest>0 else h.stage.long_miles
    budget=max(h.stage.long_miles,min(budget,goal.long_miles,guard,min(volume,actual)*long_share))
    long=max(h.stage.long_miles,min(goal.long_miles,guard,
        tenth(min(volume,actual)*long_share,down=True),tenth(budget,down=True))) if support_ok else h.stage.long_miles
    allowed=proposal['growth_per_7']>1e-12
    state=dict(version=lifecycle_version,available_at=as_of.isoformat(),day=day.isoformat(),
        observation_anchor=anchor.isoformat(),accounting_start=accounting_start.isoformat(),
        volume=volume,support=easy,growth_allowed=allowed,growth_per_7=proposal['growth_per_7'],
        support_fraction=proposal['support_fraction'],supported_volume_ceiling=proposal['supported_volume_ceiling'],
        long_budget=budget,long_growth_allowed=support_ok,long_miles=long,
        evidence_policy=evidence['version'] if evidence else None)
    if historical:
        state['historical_initialization']=initialized
    if nominal:
        state.update(rate_contract=proposal.get('rate_contract','nominal_4pct_matched_headroom_v1'),
            rate_status=proposal.get('rate_status','evidence_unavailable'),
            long_support_fraction=proposal.get('long_support_fraction',0.))
    if paired:
        state.update(execution_epoch=epoch.isoformat(),execution_distance_basis=distance_basis,execution_fulfillment=execution,
            planner_funding=funding)
    if realized:
        state.update(realized_reference=reference_config,reference_signal=signal,
            reference_controller=reference_step['state'] if reference_step else controller,
            reference_provisional=provisional,reference_step=reference_step,
            deferred_reference_volume=deferred_reference,planner_funding=funding)
        if coaching.phase!='general_fitness' or previous and previous.get('coaching_progression'):
            state['coaching_progression']=dict(phase=coaching.phase,
                maximum_growth_per_7=coaching.volume_growth_per_7,
                long_growth_fraction=coaching.long_growth_fraction,long_growth_days=coaching.long_growth_days,
                target_volume_per_7=goal.volume_per_7,target_long_miles=goal.long_miles,
                growth_allowed=coaching.growth_allowed)
    if matched:
        reference=previous.get('progression_reference') if previous else None
        if reference and reference.get('version')!='distinct_reference_v1':
            raise ValueError('Unknown progression reference needs explicit reconciliation')
        if reference is None:
            reference=dict(version='distinct_reference_v1',adopted_at=as_of.isoformat(),
                stage_key=h.stage.key,baseline_volume=h.stage.weekly_equivalent_miles,
                source='Retained accepted stage, not the conditional dated endpoint or measured fitness')
        state.update(progression_reference=reference,planning_volume=volume,
            observed_volume_per_7=actual,
            historical_recognized_increase=initialized.get('recognized_increase',0.),
            planning_volume_is_conditional=True)
    key='continuous-target:'+as_of.astimezone(timezone.utc).isoformat()
    growth_limit=max(volume,goal.volume_per_7)
    if realized and coaching.phase=='build':
        # The already-dated taper stops the base ramp even in a projection
        # made before taper begins; the existing race overlay reduces doses.
        days=max(0,(h.race_goal.taper_start-day).days)
        growth_limit=min(growth_limit,volume*(1+proposal['growth_per_7'])**(days/7))
    target=StageTarget(key,as_of,day,volume,easy,
        'Actual mileage mass over declared calendar exposure; no dose-ratio extrapolation' if realized else
        'Paired prescription fulfillment and separate issued-plan funding; no future physical credit' if paired else
        'Smoothed actual delivery supports fractional continuous growth; no future delivery credit',
        growth_per_7=proposal['growth_per_7'],growth_limit=growth_limit)
    changed_long=long>h.stage.long_miles
    candidate=replace(h.stage,key=key+':long-dose',issued_at=as_of,
        weekly_equivalent_miles=tenth(volume),long_miles=long,
        policy=replace(h.stage.policy,volume_goal_miles=max(tenth(volume),h.stage.policy.volume_goal_miles)),
        source=lifecycle_version) if changed_long else None
    return dict(review,status='continuous_growth' if allowed else 'continuous_growth_hold',
        gate_status=review['status'] if veto else 'smoothed_delivery_supported' if allowed else 'smoothed_delivery_hold',
        legacy_window_status=review['status'],smoothed_support=evidence,growth_proposal=proposal,
        candidate=candidate,target=target,continuous_state=state,
        long_progression=changed_long,phase_handoff=False,quality_progression=False)
