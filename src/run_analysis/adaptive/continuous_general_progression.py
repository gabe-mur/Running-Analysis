"""Daily observed general-fitness progression, without phase completion.

The accepted seed remains an accounting/physical-dose reference. Dated targets
own volume; reviews never confirm a forecast endpoint, reset the observation
clock, rebuild normalization, or bank unused growth. Race preparation is separate.
"""
from copy import copy
from dataclasses import replace
from datetime import date,datetime,timedelta,timezone
from math import log,log1p

from run_analysis.general_progression import destination,VOLUME_GROWTH_PER_7,LONG_SHARE,LONG_GROWTH_PER_REVIEW
from run_analysis.stage_target_history import StageTarget,target_segments,target_miles_between
from run_analysis.staged_progression import review_recent_delivery,tenth,ResponseReport,DELIVERY_ROUNDING_MILES

VERSION='continuous_general_v1'


def propose(h,as_of,entries,*,responses=(),longest=None):
    """One day's target/rate proposal; no calendar search or durable mutation."""
    if getattr(h.policy,'general_progression',None) in {'continuous_general_v2','continuous_general_v3','continuous_general_v4','continuous_general_v5','continuous_general_v6','continuous_general_v7'}:
        from .smoothed_general_progression import propose as smoothed_proposal
        return smoothed_proposal(h,as_of,entries,responses=responses,longest=longest)
    day=h.decision(as_of)
    previous=getattr(h,'general_progression_state',None)
    path=h.target_history[-1]
    if previous and as_of < datetime.fromisoformat(previous['available_at']):
        raise ValueError('Continuous review cannot precede its committed evidence')
    anchor=datetime.fromisoformat(previous['observation_anchor']) if previous else h.stage.issued_at
    accounting_start=date.fromisoformat(previous['accounting_start']) if previous else path.effective_on
    last_day=date.fromisoformat(previous['day']) if previous else day
    # Calendar time, not refresh count, controls the rate. Longer gaps need
    # observed support below; they cannot bank unused growth automatically.
    elapsed=max(0,(day-last_day).days)
    volume=previous['volume'] if previous else path.rate(day,'volume_per_7')
    support=previous['support'] if previous else path.rate(day,'supporting_easy_per_7')
    goal=destination(h.stage)
    if h.planner_preferences:
        goal=destination(replace(h.stage,goal_long_miles=h.planner_preferences.goal_long_miles))
    # Keep the original minimum observation period, not a new waiting period
    # after every target review. Use exactly the same actual-distance rule,
    # response precedence, deduplication and endurance evidence as the old gate.
    since=max(anchor,as_of-timedelta(days=h.stage.policy.review_days))
    required=None
    if since < as_of:
        segments,_=target_segments(h.target_history,as_of=as_of,
            start=since.astimezone(h.zone).date(),end=day+timedelta(days=1))
        required=target_miles_between(segments,since,as_of,h.zone)
    review=review_recent_delivery(h.stage,entries,as_of,ledger_complete=True,
        longest_prior_30d_miles=longest,responses=responses,required_miles=required,
        distance_credit=h.policy.progression_credit,observation_anchor=anchor,
        rounding_tolerance_miles=DELIVERY_ROUNDING_MILES)
    allowed=review['status']=='observed_delivery_supported'
    delivered=review.get('credited_miles',0.)*7/max(1.,(as_of-since).total_seconds()/86400)
    if previous and previous['growth_allowed'] and elapsed:
        # Yesterday's authorization covers today. An unattended longer gap
        # requires current delivery support, and cannot leap above that actual
        # exposure. Checking weekly must not itself make growth 7x slower.
        days=elapsed if allowed else min(1,elapsed)
        cap=max(volume,goal.volume_per_7)
        days=min(days,log(cap/volume)/log1p(VOLUME_GROWTH_PER_7)*7)
        new=min(cap,volume*(1+VOLUME_GROWTH_PER_7)**(days/7))
        if elapsed>1 and allowed:new=min(new,max(volume,delivered))
        support+=new-volume
        volume=new
    # Long-run growth uses DELIVERED support, never the conditional forecast.
    # Fractional budget survives the one-decimal command grid. It accrues only
    # on supported reviewed days, at the existing 5%-per-14-day-equivalent rate.
    budget=previous['long_budget'] if previous else h.stage.long_miles
    support_ok=allowed and min(volume,delivered)+1e-8>=h.stage.long_miles/LONG_SHARE
    if support_ok and elapsed and previous and previous['long_growth_allowed']:
        days=min(elapsed,log(max(budget,goal.long_miles)/budget)/log1p(LONG_GROWTH_PER_REVIEW)*14)
        budget*= (1+LONG_GROWTH_PER_REVIEW)**(days/14)
    elif not support_ok:
        budget=h.stage.long_miles  # No deferred growth credit while unsupported.
    guard=tenth(longest*1.1,down=True) if longest else h.stage.long_miles
    budget=max(h.stage.long_miles,min(budget,goal.long_miles,guard,min(volume,delivered)*LONG_SHARE))
    long=max(h.stage.long_miles,min(goal.long_miles,guard,
        tenth(min(volume,delivered)*LONG_SHARE,down=True),tenth(budget,down=True)))
    if not allowed:long=h.stage.long_miles
    state=dict(version=VERSION,available_at=as_of.isoformat(),day=day.isoformat(),
        observation_anchor=anchor.isoformat(),accounting_start=accounting_start.isoformat(),
        volume=volume,support=support,growth_allowed=allowed,long_budget=budget,
        long_growth_allowed=support_ok,long_miles=long)
    key='continuous-target:'+as_of.astimezone(timezone.utc).isoformat()
    target=StageTarget(key,as_of,day,volume,support,
        'Daily observed general progression; future slope conditional on continued delivery',
        growth_per_7=VOLUME_GROWTH_PER_7 if allowed else 0.,growth_limit=max(volume,goal.volume_per_7))
    changed_long=long>h.stage.long_miles
    candidate=replace(h.stage,key=key+':long-dose',issued_at=as_of,
        weekly_equivalent_miles=tenth(volume),long_miles=long,
        policy=replace(h.stage.policy,volume_goal_miles=max(tenth(volume),h.stage.policy.volume_goal_miles)),
        source=VERSION) if changed_long else None
    return dict(review,status='continuous_growth' if allowed else 'continuous_growth_hold',
        gate_status=review['status'],candidate=candidate,target=target,continuous_state=state,
        long_progression=changed_long,phase_handoff=False,quality_progression=False)


def advance(h,inputs,as_of,*,responses=(),longest=None):
    """One normal full-pipeline selection; journal owns target and calendar together."""
    from .adaptive_stage_lifecycle import review_entries,accept_stage_program
    from run_analysis.recovery_response import assess_response
    reports=list(responses)
    for report in h.responses:
        if not as_of-timedelta(days=h.stage.policy.review_days)<=report.available_at<=as_of:
            continue
        result=assess_response(report.expectation,effort=report.effort,
            feeling=report.feeling,feedback_conflict=report.conflict)
        if result['review_needed']:
            reports.append(ResponseReport(report.available_at,'adverse',report.source))
    review=propose(h,as_of,review_entries(h,as_of,
        original_schedule=h.policy.general_progression in {'continuous_general_v6','continuous_general_v7'}),responses=reports,longest=longest)
    proposed=copy(h)
    for name in ('history','events'):
        setattr(proposed,name,list(getattr(h,name)))
    for name in ('active','executed','issued_by_day'):
        setattr(proposed,name,dict(getattr(h,name)))
    proposed.general_progression_state=review['continuous_state']
    if review['candidate'] is not None:
        # Only a genuinely changed physical long dose uses the guarded atomic
        # dose publisher. Volume reviews do not rebuild the catalogue or stage.
        physical_review=dict(review,status='conditional_development_candidate')
        updated,result,event=accept_stage_program(proposed,inputs,as_of,physical_review,
            amended_target=review['target'])
        return (updated,result,event) if event else (h,result,None)
    proposed.target_history=(h.target_history if h.target_history[-1]==review['target']
        else (*h.target_history,review['target']))
    if proposed.frequency_contract:
        proposed.frequency_contract=proposed.frequency_contract.accepted_stage(
            review['target'].stage_key,h.decision(as_of),tenth(review['continuous_state']['volume']))
    event=proposed.replan(as_of,reason='continuous_general_review')
    return proposed,review,event
