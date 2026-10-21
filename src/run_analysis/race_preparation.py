"""Pace-free race preparation proposals. No search, acceptance, or live writes.

Distance/date select a transparent coaching destination. The accepted athlete
stage supplies the start and progression rate, not an arbitrary race-pace goal.
Future stages are conditional on delivery, readiness and whole-program checks.
"""
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
from math import isfinite

from .staged_progression import ProgressionPolicy, tenth
from .program_constraints import _ceiling as session_distance_ceiling


@dataclass(frozen=True)
class PreparationProfile:
    label: str
    distance_miles: float
    long_miles: float
    volume_per_7: float
    taper_days: int
    sources: tuple[str, ...]


# Completion-oriented central coaching scenarios, NOT physiological minimums,
# performance-optimal doses, fitted readiness thresholds or fixed templates.
# Higher established volume/long exposure is not cut to match these scenarios.
# Public provenance contains only downstream inferences. Original source
# identities and material stay in the excluded local research corpus.
POLICY_INFERENCES = 'docs/training_policy_inferences.md'
POLICY_SOURCES = (POLICY_INFERENCES + '#race-preparation',
                  POLICY_INFERENCES + '#progression-and-recovery')


PROFILES = {
    '5k': PreparationProfile('5K', 5000 / 1609.344, 3.1, 10., 7, POLICY_SOURCES),
    '10k': PreparationProfile('10K', 10000 / 1609.344, 5.5, 14., 7, POLICY_SOURCES),
    'half_marathon': PreparationProfile('Half marathon', 21097.5 / 1609.344, 10., 25., 10,
                                        POLICY_SOURCES),
    'marathon': PreparationProfile('Marathon', 42195 / 1609.344, 20., 40., 21,
                                   POLICY_SOURCES),
}


@dataclass(frozen=True)
class RacePreparationGoal:
    distance: str
    race_date: date
    preparation_mode: str = 'balanced'

    def __post_init__(self):
        if self.distance not in PROFILES or type(self.race_date) is not date:
            raise ValueError('Supported race distance and calendar date required')
        if (self.preparation_mode not in {'shorter','balanced','more'}
                or self.distance!='marathon' and self.preparation_mode!='balanced'):
            raise ValueError('Preparation options currently supported only for marathon')

    @property
    def profile(self):
        if self.distance=='marathon' and self.preparation_mode=='shorter':
            return replace(PROFILES['marathon'],long_miles=17.,volume_per_7=35.,taper_days=14)
        return PROFILES[self.distance]


@dataclass(frozen=True)
class PreparationDestination:
    volume_per_7: float
    long_miles: float
    source: str

    def __post_init__(self):
        if (any(not isfinite(v) or v <= 0 or tenth(v) != v
                for v in (self.volume_per_7, self.long_miles))
                or self.long_miles > self.volume_per_7 or not self.source):
            raise ValueError('Positive committed preparation doses and source required')

    def supporting_volume_for(self, long_miles):
        # Explicit preparation allocation, not a causal long/volume law. This
        # coordinates stages rather than adding competing score penalties.
        return long_miles * self.volume_per_7 / self.long_miles


def minimum_guarded_increases(observed_max_miles, target_miles):
    """Optimistic distance-only lower bound, NOT a preparation-week forecast.

    Each exact one-decimal hypothetical exposure becomes the next reference,
    assuming completion at that dose (not above-band or fractional actual miles),
    acceptable response and retention inside the rolling 30-day window. No rest,
    volume support, consolidation, missed running or taper is priced here. Thus
    passing this arithmetic cannot authorize a workout or certify preparation.
    This grid bound is not a universal floor for arbitrary recorded distances.
    Race distance is deliberately NOT used as a training target automatically.
    """
    if (any(not isfinite(v) or v <= 0 for v in (observed_max_miles, target_miles))
            or tenth(target_miles) != target_miles):
        raise ValueError('Positive observed reference and one-decimal training target required')
    current = observed_max_miles
    doses = []
    while current < target_miles:
        candidate = min(target_miles, session_distance_ceiling(current))
        if candidate <= current:
            return dict(status='distance_resolution_hold', minimum_increases=None,
                        hypothetical_doses=tuple(doses), accepted=False)
        doses.append(candidate)
        current = candidate
    return dict(status='distance_only_lower_bound', minimum_increases=len(doses),
                hypothetical_doses=tuple(doses), accepted=False)


def destination(stage, goal):
    """Race preparation supersedes the GENERAL future long-run destination.

    It does not shrink the accepted dose or undo a previously authorized ramp.
    A lower training dose/taper requires its own dated accounting transition.
    """
    profile = goal.profile
    committed_endpoint = (stage.policy.volume_goal_miles
                          if stage.policy.volume_path == 'gradual' else None)
    return PreparationDestination(
        tenth(max(profile.volume_per_7, stage.weekly_equivalent_miles,
                  committed_endpoint or 0.)),
        tenth(max(profile.long_miles, stage.long_miles)),
        'pace_free_race_preparation_v2:' + goal.distance + ':' + goal.preparation_mode,
    )


def next_phase(stage, target, *, as_of, longest_prior_30d_miles, phase_policy=None,
               long_growth_fraction=None):
    """Propose ONE bounded ramp AFTER an independent successful stage review.

    This does not perform that review or authorize its candidate. A forecast
    predecessor is not new observed tolerance. The caller must validate delivery,
    current responses, the selected program and atomic acceptance/publication.
    """
    if as_of.tzinfo is None or as_of < stage.issued_at:
        raise ValueError('Causally ordered aware phase timestamp required')
    if (longest_prior_30d_miles is None or not isfinite(longest_prior_30d_miles)
            or longest_prior_30d_miles <= 0):
        return dict(status='observed_session_reference_required', candidate=None)
    if phase_policy is not None and not isinstance(phase_policy,ProgressionPolicy):
        raise ValueError('Explicit progression policy required')
    development=phase_policy or stage.policy
    long_growth = development.growth_fraction if long_growth_fraction is None else long_growth_fraction
    if not isfinite(long_growth) or not 0 < long_growth < 1:
        raise ValueError('Explicit fractional long development step required')
    guard = session_distance_ceiling(longest_prior_30d_miles)
    if stage.long_miles > guard:
        return dict(status='held_long_requires_session_guard_review', candidate=None)
    current_volume = stage.weekly_equivalent_miles
    current_long = stage.long_miles
    goal_volume = max(current_volume, target.volume_per_7)
    goal_long = max(current_long, target.long_miles)
    if current_volume >= goal_volume and current_long >= goal_long:
        return dict(status='preparation_dose_reached', candidate=None)
    end_volume = min(goal_volume, tenth(current_volume * (1 + development.growth_fraction)))
    needed = target.supporting_volume_for(current_long)
    new_long = current_long
    if current_volume + 1e-8 >= needed:
        # Existing supporting exposure, not anticipated ramp completion, permits
        # long growth. Respect both the accepted step and observed-session guard.
        proposed_long = min(goal_long, tenth(current_long * (1 + long_growth)),
                            guard, tenth(current_volume * target.long_miles / target.volume_per_7, down=True))
        new_long = max(current_long, proposed_long)
    if end_volume <= current_volume and new_long <= current_long:
        return dict(status='dose_resolution_or_observed_guard_hold', candidate=None)
    policy = replace(development, growth_focus='easy_volume', volume_path='gradual',
                     volume_goal_miles=end_volume, balance=None)
    proposed = replace(stage, key=stage.key + ':race-phase:' + as_of.isoformat(),
                       issued_at=as_of, long_miles=new_long, goal_long_miles=goal_long,
                       policy=policy, source=target.source)
    return dict(status='conditional_next_phase', candidate=proposed,
                supporting_volume_required_for_current_long=needed,
                historical_targets_unchanged=True, accepted=False,
                requires_delivery_readiness_program_and_atomic_acceptance=True)


def preparation_outlook(stage, goal, *, as_of, maximum_phases=104,
                        phase_policy=None, longest_prior_30d_miles=None):
    """Conditional DOSE/timing outlook, never predicted adaptation/race time.

    No planning-horizon or score arguments: a rolling horizon cannot move the
    coaching destination. A hypothetical phase assumes complete delivery only
    for this disclosed scenario; it is never inserted into measured history.
    """
    if type(maximum_phases) is not int or not 1 <= maximum_phases <= 104:
        raise ValueError('Bounded whole phase count required')
    if as_of.tzinfo is None or as_of < stage.issued_at:
        raise ValueError('Causally ordered aware outlook timestamp required')
    target = destination(stage, goal)
    current = stage
    phases = []
    start = max(as_of,stage.review_at)
    reference=longest_prior_30d_miles if longest_prior_30d_miles is not None else stage.long_miles
    pending = (stage.policy.volume_path == 'gradual'
               and stage.weekly_equivalent_miles < stage.policy.volume_goal_miles)
    if (not pending and stage.weekly_equivalent_miles>=target.volume_per_7
            and stage.long_miles>=target.long_miles):
        start=as_of  # No new growth/review delay for already established preparation.
    if pending:
        # Reuse the committed endpoint/anchor; do not restart it at page reload.
        endpoint = max(as_of, stage.review_at)
        phases.append(dict(start=as_of, end=endpoint, volume_start=stage.weekly_equivalent_miles,
                           volume_end=stage.policy.volume_goal_miles, long_miles=stage.long_miles,
                           basis='existing committed phase; confirmation still required'))
        current = replace(stage, weekly_equivalent_miles=stage.policy.volume_goal_miles,
                          issued_at=endpoint)
        start = endpoint
    for _ in range(maximum_phases):
        proposal = next_phase(current, target, as_of=start,
                              longest_prior_30d_miles=reference,phase_policy=phase_policy)
        if proposal['candidate'] is None:
            status = proposal['status']
            break
        candidate = proposal['candidate']
        end = start + timedelta(days=candidate.policy.review_days)
        phases.append(dict(start=start, end=end, volume_start=current.weekly_equivalent_miles,
                           volume_end=candidate.policy.volume_goal_miles, long_miles=candidate.long_miles,
                           basis='conditional on complete delivery, acceptable response and feasible program'))
        current = replace(candidate, weekly_equivalent_miles=candidate.policy.volume_goal_miles,
                          issued_at=end)
        reference=current.long_miles  # Disclosed future completion scenario, never measured history.
        start = end
    else:
        status = 'projection_limit'
    reached = status == 'preparation_dose_reached'
    earliest = start.date() + timedelta(days=goal.profile.taper_days) if reached else None
    return dict(status='conditional_preparation_window' if reached else status,
                target=target, phases=tuple(phases), earliest_race_date=earliest,
                requested_date_fits=earliest is not None and goal.race_date >= earliest,
                race_date=goal.race_date, taper_days=goal.profile.taper_days,
                promised_fitness_gain=False, estimated_race_pace=None, accepted=False,
                limitations=(
                    'Coaching dose scenario, not a validated fitness or finish-time forecast',
                    'Every subsequent phase requires actual delivery, response and program feasibility',
                    'No time-pressure acceleration; no injury or recovery clearance',
                    'Race-specific taper, race-day load and post-race recovery still require planner integration',
                ))


@dataclass(frozen=True)
class MarathonPreparationPolicy:
    """Disclosed coaching scenarios, not fitted safety/adaptation thresholds."""
    mode: str
    volume_growth_per_review: float
    review_days: int
    consolidation_days: int
    interruption_allowance_days: int

    @property
    def progression(self):
        return ProgressionPolicy(self.review_days,self.volume_growth_per_review)


MARATHON_POLICIES={
    # Rounded conservative scenarios relative to matched descriptive coaching
    # build rates. These are NOT inferred athlete capacity or a weekly10% rule.
    'shorter':MarathonPreparationPolicy('shorter',.08,7,7,0),
    'balanced':MarathonPreparationPolicy('balanced',.06,7,14,14),
    'more':MarathonPreparationPolicy('more',.04,7,14,28),
}


@dataclass(frozen=True)
class PreparationExposureEvidence:
    """Caller-audited recent maintained dose; elapsed stage age is NOT evidence."""
    available_at: datetime
    observed_through: datetime
    volume_per_7: float
    long_miles: float
    maintained_days: int
    source: str

    def __post_init__(self):
        if (not isinstance(self.available_at,datetime) or self.available_at.tzinfo is None
                or not isinstance(self.observed_through,datetime) or self.observed_through.tzinfo is None
                or self.observed_through>self.available_at
                or any(not isfinite(v) or v<=0 for v in (self.volume_per_7,self.long_miles))
                or type(self.maintained_days) is not int or self.maintained_days<0 or not self.source):
            raise ValueError('Dated observed maintained-dose evidence required')


# Settling time and interruption room are explicit coaching allowances, NOT
# fitted minima, adaptation times or fixed numbers of misses an athlete can take.
# The existing accepted progression rate still owns every short-distance ramp.
SHORT_PREPARATION_ALLOWANCES = {
    '5k': (7, 7),
    '10k': (7, 7),
    'half_marathon': (14, 14),
}


def reference_review():
    """Current qualitative review, distinct from the older numerical scenarios.

    Downstream inferences neither supply a calendar nor vote on optimizer
    weights. Private reference material is not a runtime dependency or exposed
    in public metadata. Ultra guidance remains excluded.
    """
    return dict(reference_database=None,
        reference_review=POLICY_INFERENCES + '#race-preparation',
        reference_use='downstream_inferences_not_templates_or_fitted_weights',
        numerical_scenario_basis='AS133/AS136 disclosed coaching assumptions; numerical doses unchanged',
        exact_doses_corpus_fitted=False, ultra_guidance_allowed=False)


def short_preparation_options(stage, goal, *, as_of, longest_prior_30d_miles,
                              response_state='unknown', maximum_phases=104,
                              consolidation_evidence=None):
    """Completion-oriented short-race guidance, using the accepted build rate.

    A reached dose is not an immediately consolidated dose. Conversely, recently
    maintained observed exposure can earn credit; merely aging a stage cannot.
    This never changes the calendar, imposes a hard quality cadence, accepts a
    stage, treats the race as training or clears uncertain health/readiness.
    """
    if goal.distance not in SHORT_PREPARATION_ALLOWANCES:
        raise ValueError('Shorter-distance goal required')
    if as_of.tzinfo is None or as_of < stage.issued_at or goal.race_date <= as_of.date():
        raise ValueError('Causally ordered timestamp and future race date required')
    if response_state not in {'unknown', 'clear', 'adverse', 'health_concern'}:
        raise ValueError('Explicit interpreted response state required')
    if consolidation_evidence is not None and (
            not isinstance(consolidation_evidence, PreparationExposureEvidence)
            or consolidation_evidence.available_at > as_of):
        raise ValueError('Consolidation evidence must already be available')
    shared = dict(race_date=goal.race_date, assessed_at=as_of, accepted=False,
        may_publish_calendar=False, whole_program_checked=False, guaranteed_completion=False,
        injury_probability=None, event_is_training_prescription=False,
        response_state=response_state, requires_response_review=response_state!='clear',
        sources=goal.profile.sources,
        sources_scope='Downstream coaching assumptions only; original provenance remains private',
        **reference_review(),
        dose_basis='Provisional running-only completion scenario; not a published or physiological minimum')
    if response_state in {'adverse', 'health_concern'}:
        return dict(shared, status=('health_review_required' if response_state=='health_concern'
                                   else 'response_review_required'), options=())
    check = next_phase(stage, destination(stage, goal), as_of=as_of,
                       longest_prior_30d_miles=longest_prior_30d_miles)
    if check['status'] in {'observed_session_reference_required', 'held_long_requires_session_guard_review'}:
        return dict(shared, status=check['status'], options=())
    outlook = preparation_outlook(stage, goal, as_of=as_of, maximum_phases=maximum_phases,
                                  longest_prior_30d_miles=longest_prior_30d_miles)
    consolidation, interruption = SHORT_PREPARATION_ALLOWANCES[goal.distance]
    credit = 0
    if (not outlook['phases'] and consolidation_evidence is not None
            and as_of-consolidation_evidence.observed_through <= timedelta(days=1)
            and consolidation_evidence.volume_per_7 >= outlook['target'].volume_per_7
            and consolidation_evidence.long_miles >= outlook['target'].long_miles):
        credit = min(consolidation, consolidation_evidence.maintained_days)
    remaining = consolidation-credit
    earliest = (outlook['earliest_race_date'] + timedelta(days=remaining)
                if outlook['earliest_race_date'] is not None else None)
    recommended = earliest + timedelta(days=interruption) if earliest else None
    fits = earliest is not None and goal.race_date >= earliest
    allowance_fits = recommended is not None and goal.race_date >= recommended
    status = ('outlook_unresolved' if earliest is None else
              'insufficient_time_under_scenario' if not fits else
              'conditional_limited_preparation' if not allowance_fits else
              'conditional_preparation_with_buffer')
    warnings = ['Target doses, settling time and interruption room are disclosed coaching choices, not fitted minimums.']
    if response_state=='unknown': warnings.append('Response evidence is unresolved; this estimate does not authorize growth.')
    if fits and not allowance_fits: warnings.append('The build, settling time and taper fit, but there is little room for interruptions.')
    if not fits: warnings.append('This scenario does not fit the date; do not compress training or accelerate to catch up.')
    option = dict(mode='balanced', status=status, target=outlook['target'], phases=outlook['phases'],
        outlook_status=outlook['status'], review_days=stage.policy.review_days,
        volume_growth_per_review=stage.policy.growth_fraction,
        consolidation_days=consolidation, observed_consolidation_credit_days=credit,
        consolidation_remaining_days=remaining,
        consolidation_evidence_source=consolidation_evidence.source if credit else None,
        taper_days=outlook['taper_days'], interruption_allowance_days=interruption,
        earliest_conditional_date=earliest, recommended_conditional_date=recommended,
        requested_date_fits=fits, interruption_allowance_fits=allowance_fits,
        warnings=tuple(warnings), accepted=False, whole_program_checked=False)
    return dict(shared, status=('conditional_options_pending_response_review' if response_state=='unknown'
                                else 'conditional_options') if fits else 'no_window_under_tested_scenarios',
        options=(option,), limitations=outlook['limitations']+(
            'Settling and interruption allowances are not causal safety estimates or fixed missed-run quotas',
            'Mixed walking/cross-training evidence does not establish running-only minimums',))


def marathon_preparation_options(stage, *, race_date, as_of, longest_prior_30d_miles,
                                response_state='unknown', maximum_phases=104,
                                consolidation_evidence=None):
    """Conditional build comparisons; never permission to publish a calendar.

    The actual accepted stage and committed phase are immutable. A scenario may
    propose a DIFFERENT future review rate, but cannot accept it or fast-forward
    observed history. A calendar/evaluator/delivery check is still mandatory.
    Consolidation and interruption allowance are separate from growth and taper.
    They are provisional coaching allowances, not probability estimates.
    """
    if type(race_date) is not date or as_of.tzinfo is None or as_of < stage.issued_at:
        raise ValueError('Causally ordered timestamp and race date required')
    if response_state not in {'unknown','clear','adverse','health_concern'}:
        raise ValueError('Explicit interpreted response state required')
    if race_date<=as_of.date():
        raise ValueError('Race date must be after the assessment date')
    if consolidation_evidence is not None and (
            not isinstance(consolidation_evidence,PreparationExposureEvidence)
            or consolidation_evidence.available_at>as_of):
        raise ValueError('Consolidation evidence must already be available')
    shared=dict(race_date=race_date,assessed_at=as_of,accepted=False,
                may_publish_calendar=False,whole_program_checked=False,
                guaranteed_completion=False,injury_probability=None,
                event_is_training_prescription=False,response_state=response_state,
                requires_response_review=response_state!='clear', **reference_review())
    if response_state=='health_concern':
        return dict(shared,status='health_review_required',options=())
    if response_state=='adverse':
        return dict(shared,status='response_review_required',options=())
    # Do not use an anticipated long run as the starting guard reference. Check
    # even an existing pending phase before giving its conditional endpoint.
    check=next_phase(stage,destination(stage,RacePreparationGoal('marathon',race_date)),
                     as_of=as_of,longest_prior_30d_miles=longest_prior_30d_miles)
    if check['status'] in {'observed_session_reference_required','held_long_requires_session_guard_review'}:
        return dict(shared,status=check['status'],options=())
    options=[]
    for mode,policy in MARATHON_POLICIES.items():
        goal=RacePreparationGoal('marathon',race_date,mode)
        outlook=preparation_outlook(stage,goal,as_of=as_of,maximum_phases=maximum_phases,
            phase_policy=policy.progression,longest_prior_30d_miles=longest_prior_30d_miles)
        build_end=outlook['earliest_race_date']
        if build_end is not None:
            build_end-=timedelta(days=outlook['taper_days'])
        credit=0
        if (not outlook['phases'] and consolidation_evidence is not None
                and as_of-consolidation_evidence.observed_through<=timedelta(days=1)
                and consolidation_evidence.volume_per_7>=outlook['target'].volume_per_7
                and consolidation_evidence.long_miles>=outlook['target'].long_miles):
            credit=min(policy.consolidation_days,consolidation_evidence.maintained_days)
        consolidation_remaining=policy.consolidation_days-credit
        earliest=(build_end+timedelta(days=consolidation_remaining+outlook['taper_days'])
                  if build_end else None)
        recommended=(earliest+timedelta(days=policy.interruption_allowance_days)
                     if earliest else None)
        fits=earliest is not None and race_date>=earliest
        allowance_fits=recommended is not None and race_date>=recommended
        status=('outlook_unresolved' if earliest is None else
                'insufficient_time_under_scenario' if not fits else
                'conditional_limited_preparation' if mode=='shorter' or not allowance_fits else
                'conditional_preparation_with_buffer')
        warnings=[]
        if response_state=='unknown':
            warnings.append('Response evidence is unresolved; this comparison does not authorize growth')
        if mode=='shorter':
            warnings.append('Lower preparation exposure and faster proposed growth; little interruption allowance')
        if fits and not allowance_fits:
            warnings.append('Build and taper fit, but the suggested interruption allowance does not')
        if not fits:
            warnings.append('This preparation scenario does not fit; do not compress it or accelerate it to meet the date')
        options.append(dict(mode=mode,status=status,target=outlook['target'],
            phases=outlook['phases'],outlook_status=outlook['status'],
            volume_growth_per_review=policy.volume_growth_per_review,review_days=policy.review_days,
            consolidation_days=policy.consolidation_days,taper_days=outlook['taper_days'],
            observed_consolidation_credit_days=credit,consolidation_remaining_days=consolidation_remaining,
            consolidation_evidence_source=consolidation_evidence.source if credit else None,
            interruption_allowance_days=policy.interruption_allowance_days,
            earliest_conditional_date=earliest,recommended_conditional_date=recommended,
            requested_date_fits=fits,interruption_allowance_fits=allowance_fits,
            warnings=tuple(warnings),accepted=False,whole_program_checked=False))
    return dict(shared,status=(('conditional_options_pending_response_review' if response_state=='unknown' else
                               'conditional_options') if any(o['requested_date_fits'] for o in options)
                               else 'no_window_under_tested_scenarios'),options=tuple(options),
        limitations=(
            'No option is a completed feasibility certificate; same-evaluator calendar and phase acceptance are still required',
            'Published coaching plans inform scenarios, not individual safety, adaptation or completion probabilities',
            'Consolidation and interruption allowances are disclosed coaching assumptions, not a fixed missed-run quota',
            'No deadline-driven rate change, catch-up training or historical dose rewrite',
            'Already prepared athletes can credit maintained-dose evidence through the current or preceding24hours, not elapsed stage age or stale history',
        ))
