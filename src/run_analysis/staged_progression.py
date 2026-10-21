"""Offline development-stage proposals. Not recovery clearance or a scheduler."""
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_FLOOR, ROUND_HALF_UP
from math import isfinite
from statistics import median

from .athlete_projection import aware
from .demonstrated_training import demonstrated_baseline
from .delivered_training import DoseContract, DeliveredDose, PurposeEvidence, assess_delivery
from .long_volume_balance import LongVolumeBalancePolicy

MILE = 1609.344
# One whole-review distance command increment, not a per-run or percent allowance.
DELIVERY_ROUNDING_MILES = .1


def progression_distance_credit(contract, actual, *, basis='issued_nominal_v1'):
    """One distance-credit rule for retrospective and prospective review.

    Archived reviews retain nominal/capped credit. Actual-distance reviews use
    the measured lower bound, including overshoots; an issued target cannot
    manufacture miles. Neither rule changes physical exposure or growth limits.
    """
    if basis not in {'issued_nominal_v1', 'actual_distance_v2'}:
        raise ValueError('Unknown progression distance-credit basis')
    distance=next((t.quantity for t in contract.targets if t.metric=='distance_meters'),None)
    if distance is None or actual is None:
        return dict(status='distance_evidence_unresolved',credited_miles=0.)
    if distance.estimate is None:
        return dict(status='issued_distance_target_unresolved',credited_miles=0.)
    within=distance.low-1e-6<=actual.low<=actual.high<=distance.high+1e-6
    return dict(status='within_band' if within else 'measured_distance_difference',
        credited_miles=(actual.low if basis == 'actual_distance_v2' else
            distance.estimate if within else min(distance.estimate,actual.low))/MILE,
        nominal_miles=distance.estimate/MILE,within_band=within)


def tenth(value, *, down=False):
    return float(Decimal(str(value)).quantize(Decimal('.1'), rounding=ROUND_FLOOR if down else ROUND_HALF_UP))


@dataclass(frozen=True)
class ProgressionPolicy:
    review_days: int
    growth_fraction: float
    growth_focus: str = 'long_and_volume'
    volume_goal_miles: float | None = None
    volume_path: str = 'step'
    balance: LongVolumeBalancePolicy | None = None

    def __post_init__(self):
        if isinstance(self.balance, dict):
            object.__setattr__(self, 'balance', LongVolumeBalancePolicy(**self.balance))
        if self.balance is not None and not isinstance(self.balance, LongVolumeBalancePolicy):
            raise ValueError('Explicit long-volume balance policy required')
        if self.volume_path not in {'step','gradual'} or self.volume_path=='gradual' and self.growth_focus!='easy_volume':
            raise ValueError('Gradual volume path requires an explicit easy-volume phase')
        if type(self.review_days) is not int or self.review_days <= 0:
            raise ValueError('Explicit positive review interval required')
        if not isfinite(self.growth_fraction) or not 0 < self.growth_fraction < 1:
            raise ValueError('Explicit fractional development step required')
        if self.growth_focus not in {'long_and_volume', 'easy_volume'}:
            raise ValueError('Explicit supported progression focus required')
        if self.growth_focus == 'easy_volume':
            if (self.volume_goal_miles is None or not isfinite(self.volume_goal_miles)
                    or self.volume_goal_miles <= 0 or tenth(self.volume_goal_miles) != self.volume_goal_miles):
                raise ValueError('Easy-volume phase requires a committed one-decimal volume goal')
        elif self.volume_goal_miles is not None:
            raise ValueError('Volume goal belongs to an explicit easy-volume phase')


@dataclass(frozen=True)
class DevelopmentStage:
    key: str
    issued_at: datetime
    weekly_equivalent_miles: float
    ordinary_miles: float
    long_miles: float
    goal_long_miles: float
    observed_long_gap_days: float | None
    policy: ProgressionPolicy
    source: str

    def __post_init__(self):
        aware(self.issued_at)
        values = (self.weekly_equivalent_miles, self.ordinary_miles, self.long_miles, self.goal_long_miles)
        if not self.key or not self.source or any(not isfinite(x) or x <= 0 or tenth(x) != x for x in values):
            raise ValueError('Positive committed one-decimal doses and provenance required')
        if self.observed_long_gap_days is not None and (
            not isfinite(self.observed_long_gap_days) or self.observed_long_gap_days <= 0
        ):
            raise ValueError('Positive observed cadence required')

    @property
    def review_at(self):
        return self.issued_at+timedelta(days=self.policy.review_days)


@dataclass(frozen=True)
class ReviewEntry:
    key: str
    scheduled_at: datetime
    contract: DoseContract
    completion: DeliveredDose | None
    purpose_evidence: PurposeEvidence | None = None

    def __post_init__(self):
        aware(self.scheduled_at)
        if not self.key or self.contract.issued_at > self.scheduled_at:
            raise ValueError('Issued opportunity identity and chronology required')


@dataclass(frozen=True)
class ResponseReport:
    available_at: datetime
    state: str
    source: str

    def __post_init__(self):
        aware(self.available_at)
        if self.state not in {'unknown', 'clear', 'adverse', 'health_concern'} or not self.source:
            raise ValueError('Explicit interpreted response and provenance required')


def bootstrap_stage(history, as_of, *, goal_long_miles, policy, intents=()):
    """One-time provisional seed; existing stages must NOT call this on refresh."""
    history = tuple(history)
    baseline = demonstrated_baseline(history, as_of, intents=intents)
    recent = baseline['windows']['28']
    ordinary = recent['ordinary']['meters']['median']
    fallback = ordinary is None
    # Explicit short support runs cannot shrink the ordinary-dose fallback.
    support = set(recent['support']['keys'])
    fallback_meters = [s.meters for s in history if s.key in recent['eligible_easy']['keys']
                       and s.key not in support and s.meters > 0]
    ordinary = ordinary if ordinary is not None else (median(fallback_meters) if fallback_meters else None)
    eligible = set(recent['reference_eligible']['keys'])
    longs = sorted((s for s in history if s.key in eligible and s.role == 'long'), key=lambda s: s.end)
    limitations = ['Recorded exposure is not verified tolerance or sustainable capacity',
                   'Unverified recording completeness; no cold-start mileage inferred']
    if fallback:
        limitations.append('Ordinary intent unresolved: eligible-easy median excluding explicit support is provisional')
    result = dict(stage=None, baseline=baseline, limitations=limitations, planner_influence=False)
    if recent['reference_excluded_keys']:
        return dict(result, status='recent_exclusions_require_starting_dose_review')
    if not ordinary or not longs:
        return dict(result, status='missing_starting_dose_evidence')
    latest = longs[-1]
    prior = [s for s in history if latest.start-timedelta(days=30) < s.end <= latest.start
             and s.available_at <= as_of]
    if prior and latest.meters > max(s.meters for s in prior)*1.1+1e-6:
        return dict(result, status='latest_long_spike_requires_review', latest_long_key=latest.key)
    gaps = [(b.start-a.start).total_seconds()/86400 for a, b in zip(longs, longs[1:])]
    stage = DevelopmentStage('seed:'+as_of.isoformat(), as_of,
        tenth(recent['recorded_meters_per_7_days']/MILE), tenth(ordinary/MILE),
        tenth(latest.meters/MILE), tenth(goal_long_miles), median(gaps) if gaps else None,
        policy, 'provisional_original_exposure:'+','.join(recent['all_sessions']['keys']))
    return dict(result, status='provisional_seed', stage=stage,
                source_long_key=latest.key, ordinary_reference_provisional=fallback)


def review_stage(stage, entries, as_of, *, ledger_complete, longest_prior_30d_miles,
                 responses=(), previous_stage=None):
    """Return proposals only; accepting one and evaluating a calendar are separate.

    `ledger_complete` asserts that ALL issued stage opportunities are represented,
    not that every run was completed or that historical feedback was normal.
    """
    aware(as_of)
    if as_of < stage.issued_at:
        raise ValueError('Review precedes stage')
    entries, responses = tuple(entries), tuple(responses)
    if len({e.key for e in entries}) != len(entries):
        raise ValueError('Duplicate opportunity')
    done = [e.completion.key for e in entries if e.completion is not None]
    if len(set(done)) != len(done):
        raise ValueError('Completion counted twice')
    if any(not stage.issued_at <= e.scheduled_at < stage.review_at for e in entries):
        raise ValueError('Opportunity outside fixed stage')
    base = _response_review(stage,responses,as_of,previous_stage)
    if 'status' in base:
        return base
    if as_of < stage.review_at:
        return dict(base, status='continue_existing_stage')
    if stage.policy.growth_focus == 'easy_volume' and stage.weekly_equivalent_miles >= stage.policy.volume_goal_miles:
        return dict(base, status='volume_phase_complete_reassess_next_stage')
    if (stage.policy.growth_focus == 'long_and_volume' and stage.long_miles >= stage.goal_long_miles
            and (stage.policy.balance is None or stage.weekly_equivalent_miles+1e-8 >=
                 stage.policy.balance.volume_for_long(stage.long_miles))):
        return dict(base, status='destination_reached_reassess_supporting_program')
    if not ledger_complete or not entries:
        return dict(base, status='incomplete_review_ledger')
    return _review_issued_stage(stage,entries,as_of,longest_prior_30d_miles,base)


def _response_review(stage,responses,as_of,previous_stage=None):
    """Shared response precedence, independent of progression timing/phase."""
    known = sorted((r for r in responses if r.available_at <= as_of and r.state != 'unknown'),
                   key=lambda r: r.available_at)
    if len({r.available_at for r in known}) != len(known):
        raise ValueError('Simultaneous response reports need reconciliation')
    response = None
    for report in known:
        # A subsequent fatigue report is not evidence that an earlier health
        # concern resolved. Only explicit resolution can clear that category.
        if response is None or response.state != 'health_concern' or report.state in {'clear', 'health_concern'}:
            response = report
    base = dict(stage_key=stage.key, candidate=None, stage_target_unchanged=stage,
        response_state=response.state if response else 'unknown',
        response_source=response.source if response else None,
        requires_program_feasibility_and_recovery=True, running_authorized=False,
        quality_progression=False, planner_influence=False)
    if response and response.state == 'health_concern':
        return dict(base, status='health_review_required')
    if response and response.state == 'adverse':
        if previous_stage is not None:
            if previous_stage.issued_at >= stage.issued_at or previous_stage.goal_long_miles != stage.goal_long_miles:
                raise ValueError('Previous accepted stage identity invalid')
            if previous_stage.weekly_equivalent_miles > stage.weekly_equivalent_miles or previous_stage.long_miles > stage.long_miles:
                raise ValueError('Reversion cannot increase demand')
        return dict(base, status='rollback_candidate' if previous_stage else 'dose_specific_review_required',
                    candidate=previous_stage)
    return base


def _review_issued_stage(stage,entries,as_of,longest_prior_30d_miles,base):
    comparisons, problems = [], []
    for e in entries:
        if e.completion is None:
            problems.append((e.key, 'uncompleted_or_unuploaded'))
        elif e.completion.available_at > as_of or e.completion.end > as_of:
            problems.append((e.key, 'completion_not_available'))
        elif e.completion.start < stage.issued_at:
            raise ValueError('Completion precedes stage')
        else:
            comparison = assess_delivery(e.contract, e.completion, as_of=as_of, purpose_evidence=e.purpose_evidence)
            comparisons.append(comparison)
            if comparison['numerical_fit'] != 'within_all_issued_bands':
                problems.append((e.key, comparison['numerical_fit']))
            if comparison['purpose_evidence'] == 'different_purpose':
                problems.append((e.key, 'different_purpose'))
            if comparison['structure_basis'] == 'pace_inferred' and e.contract.purpose not in {'easy_volume', 'long_endurance'}:
                problems.append((e.key, 'provisional_work_boundaries'))
    base.update(delivery=comparisons, delivery_problems=problems)
    if problems:
        return dict(base, status='execution_review_hold_targets')
    endurance = [e for e in entries if e.contract.purpose == 'long_endurance' and any(
        t.metric == 'distance_meters' and t.quantity.low <= stage.long_miles*MILE <= t.quantity.high
        for t in e.contract.targets)]
    if not endurance:
        return dict(base, status='endurance_reference_not_represented')
    return _growth_candidate(stage,as_of,longest_prior_30d_miles,base)


def _growth_candidate(stage,as_of,longest_prior_30d_miles,base):
    """Shared bounded arithmetic, independent of the delivery review strategy."""
    if longest_prior_30d_miles is None or not isfinite(longest_prior_30d_miles) or longest_prior_30d_miles <= 0:
        return dict(base, status='session_guard_reference_unavailable')
    volume = tenth(stage.weekly_equivalent_miles*(1+stage.policy.growth_fraction))
    guard = tenth(longest_prior_30d_miles*1.1, down=True)
    if stage.policy.growth_focus == 'easy_volume':
        target = stage.long_miles
        if target > guard:
            return dict(base, status='held_long_requires_session_guard_review')
        volume = min(volume, stage.policy.volume_goal_miles)
    else:
        target = min(stage.goal_long_miles, tenth(stage.long_miles*(1+stage.policy.growth_fraction)),guard)
        balance = stage.policy.balance
        if balance is not None:
            needed = balance.volume_for_long(stage.long_miles)
            if stage.weekly_equivalent_miles+1e-8 < needed:
                # Catch up FIRST; even projected future volume cannot justify
                # simultaneous novel long growth from an imbalanced stage.
                target = stage.long_miles
                volume = min(volume, needed)
            else:
                target = min(target, max(stage.long_miles, balance.committed_long_at_volume(volume)))
            if stage.long_miles > guard:
                return dict(base, status='held_long_requires_session_guard_review')
            base = dict(base, long_volume_balance=dict(
                source=balance.source, required_volume_for_current_long=needed,
                current_long_share=stage.long_miles/stage.weekly_equivalent_miles,
                proposed_long_share=target/volume,
                curve_long_at_proposed_volume=balance.long_at_volume(volume),
                established_long_preserved=True, changes_score=False))
        if target <= stage.long_miles and balance is None:
            return dict(base, status='long_growth_resolution_or_guard_limit')
    if volume <= stage.weekly_equivalent_miles:
        return dict(base, status='volume_growth_resolution_or_phase_limit')
    extra_total = volume-stage.weekly_equivalent_miles
    gap = stage.observed_long_gap_days
    extra_nonlong = extra_total if target == stage.long_miles else (
        extra_total-(target-stage.long_miles)*7/gap if gap else None)
    if extra_nonlong is not None and extra_nonlong < -1e-8:
        return dict(base, status='supporting_volume_conflict', extra_nonlong_miles_per_7_days=extra_nonlong)
    candidate = replace(stage, key=stage.key+':next', issued_at=as_of,
        weekly_equivalent_miles=volume, long_miles=target,
        source='conditional_stage_review:'+stage.key)
    return dict(base, status='conditional_development_candidate', candidate=candidate,
        progression_focus=stage.policy.growth_focus, long_progression=target>stage.long_miles,
        extra_total_miles_per_7_days=extra_total,
        extra_nonlong_miles_per_7_days=extra_nonlong,
        cadence_assumption_days=gap,
        limitations=['Not a verified sufficient supporting-volume target',
            'Cadence used only for marginal accounting; scheduler may change it',
            'Missing feedback does not establish recovery; no automatic acceptance',
            'Numerical prescription completion does not prove whole-program funding',
            'Imperfect execution needs interpretation, not a universal growth veto'])


def review_recent_delivery(stage, entries, as_of, *, ledger_complete,
                           longest_prior_30d_miles, responses=(), previous_stage=None,
                           required_miles=None, distance_credit='issued_nominal_v1',
                           observation_anchor=None, rounding_tolerance_miles=0.):
    """Rolling evidence review against a fixed accepted stage, not rolling targets.

    A missed opportunity is not erased or permanently vetoing. Within-band
    distance retains its archived interpretation unless actual-distance credit
    is selected. Actual exposure stays untouched for fatigue and workload.
    Explicit continuous reviews may tolerate one total command increment of
    planner underprescription, only with complete actual nominal delivery.
    """
    if (not isfinite(rounding_tolerance_miles)
            or not 0 <= rounding_tolerance_miles <= DELIVERY_ROUNDING_MILES
            or rounding_tolerance_miles and (
                observation_anchor is None or distance_credit!='actual_distance_v2')):
        raise ValueError('Rounding tolerance requires continuous actual-distance review and at most one distance increment')
    # Reuse response precedence, review timing and goal cap, without treating
    # every old deviation as an eternal veto.
    if observation_anchor is not None:
        # Continuous progression borrows evidence accounting, NOT phase
        # completion or a hypothetical next-stage target. The observation
        # anchor survives every daily review and physical-dose revision.
        aware(as_of);aware(observation_anchor)
        if as_of < observation_anchor:
            raise ValueError('Review precedes observation history')
        gate=_response_review(stage,responses,as_of,previous_stage)
        if 'status' in gate:
            return gate
        if as_of < observation_anchor+timedelta(days=stage.policy.review_days):
            return dict(gate,status='insufficient_observation_history')
        gate=dict(gate,status='incomplete_review_ledger')
    else:
        gate=review_stage(stage,(),as_of,ledger_complete=False,
            longest_prior_30d_miles=longest_prior_30d_miles,responses=responses,previous_stage=previous_stage)
    if gate['status']!='incomplete_review_ledger':
        return gate
    entries=tuple(entries)
    if len({e.key for e in entries})!=len(entries):
        raise ValueError('Duplicate opportunity')
    completed=[e.completion.key for e in entries if e.completion]
    if len(set(completed))!=len(completed):
        raise ValueError('Completion counted twice')
    since=max(observation_anchor or stage.issued_at,as_of-timedelta(days=stage.policy.review_days))
    selected=[e for e in entries if since<=e.scheduled_at<as_of]
    base=dict(gate,review_window_start=since,review_window_end=as_of,
              stage_target_unchanged=stage,delivery=[],delivery_problems=[])
    if not ledger_complete:
        return dict(base,status='incomplete_review_ledger')
    credited=0.
    endurance=[]
    for e in selected:
        done=e.completion
        if done is None or done.available_at>as_of or done.end>as_of:
            base['delivery_problems'].append((e.key,'uncompleted_or_unuploaded'))
            continue
        if not since<=done.start<as_of:
            base['delivery_problems'].append((e.key,'completion_outside_review_window'))
            continue
        result=assess_delivery(e.contract,done,as_of=as_of,purpose_evidence=e.purpose_evidence)
        base['delivery'].append(result)
        distance=next((t.quantity for t in e.contract.targets if t.metric=='distance_meters'),None)
        actual=done.distance_meters
        credit=progression_distance_credit(e.contract,actual,basis=distance_credit)
        if credit['status'] in {'distance_evidence_unresolved','issued_distance_target_unresolved'}:
            base['delivery_problems'].append((e.key,credit['status']))
            continue
        credited+=credit['credited_miles']
        if not credit['within_band']:
            base['delivery_problems'].append((e.key,'measured_distance_difference'))
        if (e.contract.purpose=='long_endurance' and result['purpose_evidence']!='different_purpose'
                and distance.low-1e-6<=stage.long_miles*MILE<=distance.high+1e-6
                and actual.low>=distance.low-1e-6):
            endurance.append(e.key)
    required=stage.weekly_equivalent_miles*(as_of-since).total_seconds()/86400/7 if required_miles is None else required_miles
    if not isfinite(required) or required<=0:
        raise ValueError('Positive dated delivery requirement required')
    # Keep prescription delivery distinct from adherence to that prescription.
    # This is attribution only: neither planner underdelivery nor compliant
    # execution manufactures missing exposure or changes the advancement gate.
    issued_distances=[next((t.quantity.estimate for t in e.contract.targets
        if t.metric=='distance_meters'),None) for e in selected]
    prescription_known=all(v is not None for v in issued_distances)
    prescribed=sum(issued_distances)/MILE if prescription_known else None
    plan_gap=max(0.,required-prescribed) if prescription_known else None
    execution_gap=max(0.,prescribed-credited) if prescription_known else None
    attribution=('unknown_prescription' if not prescription_known else
        'none' if credited+1e-8>=required else
        'planner_and_execution' if plan_gap>1e-8 and execution_gap>1e-8 else
        'planner_underprescription' if plan_gap>1e-8 else 'execution_shortfall')
    base.update(credited_miles=credited,required_miles=required,endurance_keys=endurance,
                distance_credit_basis=distance_credit,
                excess_does_not_accelerate=distance_credit=='issued_nominal_v1',
                raw_actuals_unchanged=True,
                volume_accounting=dict(prescribed_miles=prescribed,
                    credited_miles=credited,required_miles=required,
                    planner_shortfall_miles=plan_gap,execution_gap_miles=execution_gap,
                    attribution=attribution,changes_progression_gate=False))
    rounding_applied=bool(rounding_tolerance_miles and prescription_known and prescribed>0
        and not base['delivery_problems'] and credited+1e-8>=prescribed
        and 1e-8<required-credited<=rounding_tolerance_miles+1e-8)
    if rounding_tolerance_miles:
        base['volume_accounting'].update(rounding_tolerance_miles=rounding_tolerance_miles,
            rounding_tolerance_applied=rounding_applied,
            changes_progression_gate=rounding_applied)
    if credited+1e-8<required and not rounding_applied:
        return dict(base,status='recent_delivery_hold_targets')
    if not endurance:
        return dict(base,status='endurance_reference_not_represented')
    if observation_anchor is not None:
        if longest_prior_30d_miles is None or not isfinite(longest_prior_30d_miles) or longest_prior_30d_miles <= 0:
            return dict(base,status='session_guard_reference_unavailable')
        if stage.long_miles > tenth(longest_prior_30d_miles*1.1,down=True):
            return dict(base,status='held_long_requires_session_guard_review')
        return dict(base,status='observed_delivery_supported')
    return _growth_candidate(stage,as_of,longest_prior_30d_miles,base)
