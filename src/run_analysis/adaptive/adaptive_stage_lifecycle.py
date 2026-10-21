"""Conditional progression review and atomic target/calendar publication."""
from copy import copy
from dataclasses import replace
from datetime import datetime,time,timedelta
import json

from run_analysis.delivered_training import DoseContract,DoseTarget,DeliveredDose
from run_analysis.stage_target_history import StageTarget,target_segments,target_miles_between
from run_analysis.staged_progression import ReviewEntry,ResponseReport,review_recent_delivery,MILE
from run_analysis.workload_accounting import Quantity
from run_analysis.adaptive.adaptive_candidate_pipeline import compile_athlete,supporting_easy_target
from run_analysis.adaptive.stage_acceptance import StageLedger,accept_candidate
from run_analysis.adaptive.continuous_search_adapter import NoEligibleProgram


def review_entries(harness,as_of,*,original_schedule=False):
    completed={session.removeprefix('slot:'):r for session,r in harness.sandbox.completed_accounted(as_of)
               if session.startswith('slot:')}
    days={day for event in harness.events for day in event['active']}
    entries=[]
    for day in sorted(days|set(completed)):
        if datetime.fromisoformat(day).date()>as_of.astimezone(harness.zone).date():
            continue
        actual=completed.get(day)
        if actual:
            fingerprint=harness.sandbox.db.execute('SELECT fingerprint FROM sandbox_completions WHERE session_key=?',('slot:'+day,)).fetchone()[0]
        else:
            decisions=[e for e in harness.events if datetime.fromisoformat(e['as_of'])<=as_of
                and datetime.fromisoformat(e['as_of']).astimezone(harness.zone).date()<=datetime.fromisoformat(day).date()]
            selected=decisions[-1]['active'].get(day) if decisions else None
            if not selected:
                continue
            fingerprint=selected['fingerprint']
        intent=harness.sandbox.record(fingerprint).payload['intent']
        band=intent['distance_miles']
        work=[p['seconds'] for p in intent['phases'] if p['kind']=='work']
        # Long purpose comes from the archived selected physical dose, never
        # inferred from actual HR or the current changed catalogue.
        old=next(item for event in harness.events for item in event['active'].values() if item['fingerprint']==fingerprint)
        role=harness.catalog[old['dose']].role
        purpose='high_aerobic' if work else 'long_endurance' if role=='long' else 'easy_volume'
        targets=[DoseTarget('distance_meters',Quantity(band['low']*MILE,band['high']*MILE,band['target']*MILE,'issued distance band'))]
        if work:
            targets.append(DoseTarget('work_seconds',Quantity(sum(p['low'] for p in work),sum(p['high'] for p in work),
                sum(p['target'] for p in work),'issued work band')))
        contract=DoseContract(datetime.fromisoformat(intent['issued_at']),purpose,tuple(targets),fingerprint,
            bout_seconds=Quantity(min(p['low'] for p in work),max(p['high'] for p in work)) if work else None)
        done=(DeliveredDose(actual.key,actual.start,actual.end,actual.available_at,
            Quantity.point(actual.miles*MILE,'observed'),actual.demand[0].scale(60),actual.bouts,
            actual.work_evidence_basis if actual.bouts else 'unknown',actual.role) if actual else None)
        scheduled=(actual.start if actual and not original_schedule else
            max(contract.issued_at,datetime.fromisoformat(intent['suggested_start'])))
        entries.append(ReviewEntry('slot:'+day,scheduled,contract,done))
    # A fresh diagnostic can have physical seed history but no local issuance
    # journal for those runs. Preserve its actual prior contracts/completions;
    # local observations supersede the same identity, never double-count it.
    combined={row.key:row for row in getattr(harness,'seed_review_entries',())
        if row.completion.available_at<=as_of}
    combined.update((row.key,row) for row in entries)
    return tuple(combined[key] for key in sorted(combined))


def advance_stage(harness,original_inputs,as_of):
    """Price a whole programme under its declared delivery contract.

    Preview cannot publish or advance the accepted stage. Acceptance and all
    issued slots commit in the SAME isolated database transaction.
    """
    harness.sync_completions(as_of)
    ledger=StageLedger.for_sandbox(harness.sandbox)
    if ledger.current!=harness.stage:
        raise ValueError('Accepted stage and current calendar disagree; reconcile before proceeding')
    from .race_event import physical_keys
    race_keys=physical_keys(harness)
    recent=[r.miles for r in harness.history if as_of-timedelta(days=30)<=r.start<as_of and r.available_at<=as_of
        and harness.capacity_eligibility.get(r.key,True) and r.key not in race_keys]
    responses=[]
    if harness.current_context and harness.current_context.available_at<=as_of:
        state=harness.current_context.state
        if state in {'health_concern','conflicting','little_tired'}:
            responses.append(ResponseReport(harness.current_context.available_at,
                'adverse' if state=='little_tired' else 'health_concern',harness.current_context.source))
    gradual=harness.stage.policy.volume_path=='gradual'
    required=None
    if gradual:
        since=max(harness.stage.issued_at,as_of-timedelta(days=harness.stage.policy.review_days))
        if since<as_of:
            volume,_=target_segments(harness.target_history,as_of=as_of,
                start=since.astimezone(harness.zone).date(),end=as_of.astimezone(harness.zone).date()+timedelta(days=1))
            required=target_miles_between(volume,since,as_of,harness.zone)
    from .planner_preferences import progression_stage
    race = getattr(harness, 'race_goal', None)
    from .race_event import preparation_active
    race_build = preparation_active(harness,harness.decision(as_of))
    if harness.policy.general_progression in {'continuous_general_v1','continuous_general_v2','continuous_general_v3','continuous_general_v4','continuous_general_v5','continuous_general_v6'} and race_build:
        raise ValueError('Continuous general progression needs an explicit race-target migration before race preparation; no legacy fallback')
    general_build = harness.policy.general_progression is not None and not race_build
    if (general_build and harness.policy.general_progression in {'continuous_general_v1','continuous_general_v2','continuous_general_v3','continuous_general_v4','continuous_general_v5','continuous_general_v6'}
            or harness.policy.general_progression=='continuous_general_v7'):
        from .continuous_general_progression import advance
        return advance(harness,original_inputs,as_of,responses=responses,longest=max(recent,default=None))
    if general_build and not gradual:
        raise ValueError('General phase continuation requires an existing bounded gradual phase')
    preferences=getattr(harness,'planner_preferences',None)
    if race is not None:
        general_goal=preferences.goal_long_miles if preferences else race.general_long_goal_miles
        review_stage=(harness.stage if race_build else replace(harness.stage,
            goal_long_miles=max(harness.stage.long_miles,general_goal or harness.stage.goal_long_miles)))
    else:
        review_stage=progression_stage(harness.stage,preferences)
    if race_build and race.phase(harness.decision(as_of)) != 'build':
        return harness, dict(status='dated_race_growth_hold', phase=race.phase(harness.decision(as_of))), None
    if harness.policy.prospective_review_hour is not None:
        from .missed_checkpoint import propose
        interruption=propose(harness.stage,harness.target_history,as_of,
            harness.config['timezone_default'],harness.policy.prospective_review_hour,0.,
            race_goal=race,race_event=getattr(harness,'race_event',None),
            race_goal_history=getattr(harness,'race_goal_history',()),
            race_event_history=getattr(harness,'race_event_history',()))
        if interruption['candidate'] is not None:
            return accept_stage_program(harness,original_inputs,as_of,interruption,
                amended_target=interruption['amended_target'])
        # A missing prior-date opportunity must not turn OPTIONAL growth into
        # compulsory catch-up before the checkpoint. Same-day timing is never
        # a missed run; canceled/superseded slots are absent from review_entries.
        # Credit interpretation alone cannot create a missed opportunity. The
        # durable physical journal must also have no known upload for the slot.
        completed_days={session.removeprefix('slot:')
            for session,_ in harness.sandbox.completed_accounted(as_of)
            if session.startswith('slot:')}
        past=tuple((e.contract.source,e.scheduled_at.astimezone(harness.zone).date().isoformat())
            for e in review_entries(harness,as_of) if e.completion is None
            and e.key.removeprefix('slot:') not in completed_days
            and harness.target_history[-1].effective_on<=e.scheduled_at.astimezone(harness.zone).date()
            <as_of.astimezone(harness.zone).date())
        omission=propose(harness.stage,harness.target_history,as_of,
            harness.config['timezone_default'],harness.policy.prospective_review_hour,0.,
            race_goal=race,race_event=getattr(harness,'race_event',None),
            race_goal_history=getattr(harness,'race_goal_history',()),
            race_event_history=getattr(harness,'race_event_history',()),
            unconfirmed_opportunities=past) if past and as_of<interruption.get('original_checkpoint',as_of) else dict(candidate=None)
        if omission['candidate'] is not None:
            return accept_stage_program(harness,original_inputs,as_of,omission,
                amended_target=omission['amended_target'])
    if race_build or general_build:
        from run_analysis.recovery_response import assess_response
        for report in harness.responses:
            since=max(harness.stage.issued_at,as_of-timedelta(days=harness.stage.policy.review_days))
            if not since <= report.available_at <= as_of:
                continue
            assessment=assess_response(report.expectation,effort=report.effort,
                feeling=report.feeling,feedback_conflict=report.conflict)
            if assessment['review_needed']:
                responses.append(ResponseReport(report.available_at,'adverse',report.source))
    if race_build:
        from run_analysis.race_preparation import destination
        target = destination(harness.stage, race.goal)
        review_stage = replace(harness.stage, goal_long_miles=target.long_miles)
    if race_build or general_build:
        # Endpoint completion must still pass delivered-volume, endurance and
        # response review; elapsed time alone cannot start another growth phase.
        if (review_stage.policy.growth_focus == 'easy_volume' and
                review_stage.weekly_equivalent_miles >= review_stage.policy.volume_goal_miles):
            from run_analysis.staged_progression import tenth
            review_stage = replace(review_stage, policy=replace(review_stage.policy,
                volume_goal_miles=tenth(review_stage.weekly_equivalent_miles*(1+review_stage.policy.growth_fraction))))
    review=review_recent_delivery(review_stage,review_entries(harness,as_of),as_of,
        ledger_complete=True,longest_prior_30d_miles=max(recent,default=None),responses=responses,
        required_miles=required,distance_credit=harness.policy.progression_credit)
    if review['status']!='conditional_development_candidate':
        # Missed/unuploaded work must not permanently brick the issued growth
        # checkpoint or become debt that the next calendar must catch up.
        # V2 simply retries observed delivery on the next refresh. A partial
        # completion is not a deadline failure; genuine omitted opportunities
        # and resolved race interruptions are still handled above.
        if (harness.policy.progression_admission == 'checkpoint_v1'
                and review['status']=='recent_delivery_hold_targets'
                and review['volume_accounting']['execution_gap_miles']>1e-8):
            from .missed_checkpoint import propose
            maintenance=propose(harness.stage,harness.target_history,as_of,
                harness.config['timezone_default'],harness.policy.prospective_review_hour,
                review['volume_accounting']['execution_gap_miles'],race_goal=race,
                race_event=getattr(harness,'race_event',None),
                race_goal_history=getattr(harness,'race_goal_history',()),
                race_event_history=getattr(harness,'race_event_history',())) if harness.policy.prospective_review_hour is not None else dict(candidate=None)
            if maintenance['candidate'] is not None:
                return accept_stage_program(harness,original_inputs,as_of,maintenance,
                    amended_target=maintenance['amended_target'])
        return harness,review,None
    next_phase_started = False
    if race_build or general_build:
        from run_analysis.race_preparation import next_phase, MARATHON_POLICIES
        reviewed = review['candidate']
        # An existing bounded ramp is confirmed first. Maintenance review checks
        # do not fabricate the extra dose used solely to bypass the early return.
        confirmed_volume = (min(reviewed.weekly_equivalent_miles, harness.stage.policy.volume_goal_miles)
            if gradual else harness.stage.weekly_equivalent_miles)
        confirmed = replace(reviewed, weekly_equivalent_miles=confirmed_volume,
            long_miles=harness.stage.long_miles)
        policy = harness.stage.policy
        if race_build and race.distance == 'marathon':
            scenario = MARATHON_POLICIES[race.preparation_mode]
            policy = replace(policy, growth_fraction=scenario.volume_growth_per_review, review_days=scenario.review_days)
        if race_build:
            proposal = next_phase(confirmed, target, as_of=as_of,
                longest_prior_30d_miles=max(recent,default=None), phase_policy=policy)
        else:
            from run_analysis.general_progression import next_general_phase
            # A pending ramp is confirmed, not interpreted as an additional
            # growth increment. The next path starts from this observed credit.
            confirmed = replace(confirmed, policy=harness.stage.policy,
                goal_long_miles=review_stage.goal_long_miles)
            proposal = next_general_phase(confirmed, as_of=as_of,
                longest_prior_30d_miles=max(recent,default=None),
                version=harness.policy.general_progression)
        if race_build and proposal['candidate'] is not None and (
                proposal['candidate'].review_at.astimezone(harness.zone).date() >= race.taper_start):
            proposal=dict(proposal,status='insufficient_time_for_next_bounded_phase',candidate=None)
        if proposal['candidate'] is None:
            # Confirm a delivered pending ramp even when the destination is now
            # reached. Do not invent further growth or repeatedly accept a hold.
            if confirmed_volume <= harness.stage.weekly_equivalent_miles:
                return harness, dict(review, status=proposal['status']), None
            review = dict(review, candidate=replace(confirmed, policy=harness.stage.policy))
        else:
            review = dict(review, candidate=proposal['candidate'],
                **{('race_phase' if race_build else 'general_phase'):proposal})
            next_phase_started = True
    return accept_stage_program(harness,original_inputs,as_of,review,next_phase_started=next_phase_started)


def accept_stage_program(harness,original_inputs,as_of,review,*,next_phase_started=False,
                         amended_target=None,preview_check=None):
    """One preview/admission/atomic publisher for growth, rate revisions and maintenance."""
    ledger=StageLedger.for_sandbox(harness.sandbox)
    gradual=harness.stage.policy.volume_path=='gradual'
    race=getattr(harness,'race_goal',None)
    from .race_event import preparation_active
    race_build=preparation_active(harness,harness.decision(as_of))
    general_build=harness.policy.general_progression is not None and not race_build
    sessions=harness.session_evidence()
    # Confirming an already priced ramp is not a new dose/normalization epoch.
    maintenance=review['status']=='conditional_maintenance_candidate'
    candidate=(dict(stage=review['candidate'],catalog=(),coaching_contracts={},
        easy_duration_policy=harness.easy_duration_policy,scales=harness.scales) if gradual and not next_phase_started and amended_target is None else
        compile_athlete(dict(original_inputs,sessions=sessions,history=tuple(harness.history)),as_of,stage=review['candidate']))
    if maintenance:
        candidate=dict(stage=review['candidate'],catalog=(),coaching_contracts={},
            easy_duration_policy=harness.easy_duration_policy,scales=harness.scales)
    proposed=copy(harness)
    for name in ('history','events'):
        setattr(proposed,name,list(getattr(harness,name)))
    for name in ('active','executed','issued_by_day'):
        setattr(proposed,name,dict(getattr(harness,name)))
    proposed.stage=candidate['stage']
    if harness.frequency_contract is not None:
        proposed.frequency_contract=harness.frequency_contract.accepted_stage(
            proposed.stage.key, harness.decision(as_of), proposed.stage.weekly_equivalent_miles)
    # Preserve issued physical alternatives and quality purpose. Aerobic stage
    # growth does not simultaneously increase/reformat the quality command.
    proposed.catalog=dict(harness.catalog)
    proposed.coaching_contracts=dict(harness.coaching_contracts)
    for dose in candidate['catalog']:
        if dose.role!='quality':
            proposed.catalog[dose.key]=dose
            proposed.coaching_contracts[dose.key]=candidate['coaching_contracts'][dose.key]
    # A development focus changes dose targets, not cadence or scoring weights.
    # Recompiling default policy must not silently reset those independent choices.
    proposed.training_policy=replace(harness.training_policy,
        weekly_miles=proposed.stage.weekly_equivalent_miles,
        ordinary_miles=proposed.stage.ordinary_miles,long_miles=proposed.stage.long_miles)
    quality=proposed.catalog['quality:ordinary']
    support=supporting_easy_target(proposed.stage,quality.miles,
        single_volume=proposed.training_policy.purpose_weights[1]==0,
        long_per_7=proposed.training_policy.long_per_7,quality_per_7=proposed.training_policy.quality_per_7)
    proposed.easy_duration_policy=replace(candidate['easy_duration_policy'],supporting_easy_miles_per_7=support)
    proposed.scales=harness.scales if race_build or general_build else candidate['scales']
    sustained=harness.policy.delivery_contract=='sustained_v1'
    if amended_target is not None:
        proposed.target_history=(*harness.target_history,amended_target)
    elif next_phase_started:
        from run_analysis.adaptive.factory import initial_target_history
        # Same immutable anchor/ramp contract as the initial accepted phase.
        path_candidate = dict(candidate, training_policy=proposed.training_policy,
            easy_duration_policy=proposed.easy_duration_policy)
        path = initial_target_history(path_candidate,as_of,harness.config['timezone_default'])[-1]
        proposed.target_history=(*harness.target_history,path)
    elif not gradual:
        proposed.target_history=(*harness.target_history,StageTarget(proposed.stage.key,as_of,harness.decision(as_of),
            proposed.stage.weekly_equivalent_miles,support,
            'ordinary-objective development target; not guaranteed delivered mileage' if sustained else
            'conditional review accepted only with a funded programme'))
    from .coached_runtime import rebind_stage
    rebind_stage(proposed,harness,as_of)
    try:
        preview=proposed.replan(as_of,reason='conditional stage preview',publish=False,require_stage_funding=not sustained)
    except NoEligibleProgram:
        # Optional growth was not established by this bounded search. Keep the
        # accepted target, then let the caller evaluate a fresh current-stage
        # calendar. Do not publish an unscored growth fallback or claim global
        # infeasibility; other errors must still surface.
        return harness,dict(review,status='growth_search_no_eligible_candidate' if sustained else 'growth_search_no_funded_candidate',
            candidate_not_accepted=True,stage_target_unchanged=harness.stage),None
    def commit():
        return accept_candidate(ledger,review,preview['state'],preview['planned'],preview['contract_keys'],reference=harness.ref,
            selection=preview if sustained else None)
    if preview_check is not None:
        preview_check(proposed,preview)  # Must pass before any stage/calendar publication.
    if sustained and not preview['planned']:
        return harness,dict(review,status='growth_search_no_training',
            candidate_not_accepted=True,stage_target_unchanged=harness.stage),None
    reason=('accepted_continuous_long_dose' if proposed.policy.general_progression in {'continuous_general_v1','continuous_general_v2','continuous_general_v3','continuous_general_v4','continuous_general_v5','continuous_general_v6','continuous_general_v7'}
        and review.get('long_progression') else
        'accepted_checkpoint_maintenance' if maintenance else
        'accepted_volume_rate_amendment' if amended_target is not None else 'accepted_stage_progression')
    event=proposed.replan(as_of,reason=reason,prepared=preview,
        require_stage_funding=not sustained,commit_stage=commit)
    return proposed,review,event
