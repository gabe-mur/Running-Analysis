"""Explicit sequence-level coaching comparison; no recovery clock or live use."""
from collections import Counter
from datetime import timedelta

import numpy as np

from run_analysis.athlete_projection import aware
from run_analysis.easy_dose import assess_easy_dose
from run_analysis.program_constraints import assess_program_constraints
from run_analysis.recency_demand import compare_recency
from scripts.easy_tolerance_trial import quantity


def concentration(history, option, start, end):
    """All date-window peak doses, not fitted fatigue or safe-load envelopes.

    Long miles are purpose-labelled exposure. Known quality bouts are not a
    complete measurement of fast work in sessions with missing classification.
    Date windows intentionally avoid using precise suggested start times.
    """
    span = (end-start).days
    if span <= 0:
        raise ValueError('Positive explicit comparison interval required')
    ledger = (*history, *option.demands)
    if len({e.key for e in ledger}) != len(ledger) or len({e.day for e in ledger}) != len(ledger):
        raise ValueError('Unique reconciled activities and local dates required')
    if len({e.accounting_key for e in ledger}) > 1:
        raise ValueError('Shared accounting units required')
    origin = start-timedelta(days=span-1)
    values = np.zeros(((end-origin).days, 3))
    for e in ledger:
        if not origin <= e.day < end:
            continue
        miles = quantity(e,'miles')
        if miles is None or miles.low != miles.high:
            raise ValueError('This external-dose comparison requires known distance')
        values[(e.day-origin).days] += (miles.low, miles.low if e.purpose == 'long' else 0.,
            sum(e.structure or ())/60 if e.purpose in {'quality','threshold','high_aerobic','speed_skill','race_specific'} else 0.)
    prefix = np.vstack((np.zeros((1,3)),np.cumsum(values,axis=0)))
    ends = np.arange((start-origin).days+1,len(prefix))
    peaks = np.array([np.max(prefix[ends]-prefix[ends-window],axis=0) for window in range(1,span+1)])
    return dict(window_days=list(range(1,span+1)),
        channels={k:peaks[:,i].tolist() for i,k in enumerate(('total_miles','long_miles','known_quality_work_minutes'))},
        history_included=True, recovery_estimate=False)


def compare_peaks(reference, candidate):
    if reference['window_days'] != candidate['window_days']:
        raise ValueError('Same physical window definitions required')
    report = {}
    for channel, a in reference['channels'].items():
        b = candidate['channels'][channel]
        delta = np.array(b)-a
        higher, lower = np.any(delta > 1e-8), np.any(delta < -1e-8)
        order = ('crossing' if higher and lower else 'candidate_higher' if higher else
                 'candidate_lower' if lower else 'equal')
        report[channel] = dict(order=order,changed_windows=[dict(days=d,reference=x,candidate=y)
            for d,x,y in zip(reference['window_days'],a,b) if abs(x-y) > 1e-8])
    return report


def _physical(e):
    return (e.day,e.miles,e.role,e.easy_miles,e.quality_purpose,e.work_bout_seconds)


def _key_doses(option, start, end):
    # Identity binds a key opportunity to its dose; moved dates do not break it.
    return {e.key:(e.miles,e.role,e.easy_miles,e.quality_purpose,e.work_bout_seconds)
            for e in option.exposures if start <= e.day < end and e.role != 'easy'}


def _opportunity(reference, candidate, a, b, history):
    r = compare_recency((*history,*(e for e in reference.demands if e.day < a.day)),a,
                        (*history,*(e for e in candidate.demands if e.day < b.day)),b)
    return dict(key=b.key,reference_day=a.day,candidate_day=b.day,
                order=r['recency_demand_order'],channels=r['channel_orders'])


def compare(reference, candidate, history, policy, *, as_of, start, main_end, end,
            context=None, unexpected_previous_response=False, restrictions=()):
    """Compare controlled alternatives under a stated phase priority.

    `policy` supplies ONLY the frozen ordinary command and aerobic minimum.
    Its failed local reduction fraction is deliberately unused. This comparison
    never claims the reference or candidate has established recovery suitability.
    """
    aware(as_of)
    if not as_of.date() <= start < main_end <= end or policy.reference.issued_at > as_of:
        raise ValueError('Available frozen reference and future physical interval required')
    for option in (reference,candidate):
        if any(not start <= e.day < end for e in option.exposures):
            raise ValueError('Program must cover the declared comparison scope only')
        if len(option.exposures) != len(option.demands) or {e.key for e in option.exposures} != {e.key for e in option.demands}:
            raise ValueError('Matched exposure and demand identities required')
        for e in option.exposures:
            d = next(d for d in option.demands if d.key == e.key)
            q = quantity(d,'miles')
            if q is None or q.low != q.high or abs(q.low-e.miles)>1e-8 or d.day != e.day:
                raise ValueError('Physical accounting mismatch')
            if d.structure != e.work_bout_seconds or d.purpose != (e.quality_purpose if e.role == 'quality' else e.role):
                raise ValueError('Purpose/structure mismatch')
    if any(e.day >= start for e in history):
        raise ValueError('Actual lead-in cannot overlap the projected program')
    r = dict(reference_key=reference.key,candidate_key=candidate.key,
        policy='protect_key_work_allow_explicit_spacing_trials_v1',
        program_approved=False,recovery_cleared=False,planner_influence=False,
        mandatory_next_date_reduction=False, reference_suitability_established=False)
    def finish(decision):
        return dict(r,decision=decision)
    tail_a = sorted(((e.key,_physical(e)) for e in reference.exposures if e.day >= main_end),key=str)
    tail_b = sorted(((e.key,_physical(e)) for e in candidate.exposures if e.day >= main_end),key=str)
    if tail_a != tail_b:
        return finish('different_continuation_requires_joint_evaluation')
    if _key_doses(reference,start,main_end) != _key_doses(candidate,start,main_end):
        return finish('changed_key_prescription_requires_purpose_review')
    totals = {o.key:dict(miles=sum(e.miles for e in o.exposures if e.day < main_end),
        easy_miles=sum(e.easy_miles or 0. for e in o.exposures if e.day < main_end)) for o in (reference,candidate)}
    if reference.key == candidate.key:
        raise ValueError('Distinct option keys required')
    r['delivery'] = totals
    if any(abs(totals[reference.key][k]-totals[candidate.key][k])>1e-8 for k in ('miles','easy_miles')):
        return finish('changed_volume_requires_stage_review')
    r['constraints'] = assess_program_constraints(candidate.demands,history,as_of=as_of,
                                                   restrictions=restrictions,plan_complete=True)
    if r['constraints']['status'] != 'passes_checked_constraints_conditionally':
        return finish('program_constraints_require_review')
    known = context if context and context.available_at <= as_of else None
    r['current_context'] = dict(state=known.state,source=known.source,applies_on=known.applies_on) if known else None
    if known and known.applies_on > as_of.date():
        raise ValueError('Current report cannot assert future readiness')
    if known and known.state in {'health_concern','conflicting'}:
        return finish('review_current_context')
    if (known and known.state == 'little_tired') or unexpected_previous_response:
        return finish('response_specific_dose_review')
    checks = []
    for e in candidate.exposures:
        if e.role == 'easy':
            demand = next(d for d in candidate.demands if d.key == e.key)
            work = quantity(demand,'work_minutes')
            if work is None or work.high > 1e-8:
                return finish('easy_intensity_requires_review')
            checks.append(dict(key=e.key,day=e.day,**assess_easy_dose(e.miles,policy.reference,policy.minimum)))
    r['easy_dose_checks'] = checks
    if any(not c['passes_planned_minimum'] for c in checks):
        return finish('below_aerobic_minimum')
    if any(c['committed_miles'] > policy.ordinary_miles+1e-8 for c in checks):
        return finish('larger_easy_requires_dose_development_review')
    profiles = [concentration(history,o,start,end) for o in (reference,candidate)]
    r['concentration'] = compare_peaks(*profiles)
    key_orders = [r['concentration'][k]['order'] for k in ('long_miles','known_quality_work_minutes')]
    comparisons = [_opportunity(reference,candidate,next(a for a in reference.demands if a.key==b.key),b,history)
        for b in candidate.demands if b.purpose != 'easy']
    r['key_opportunities'] = comparisons
    r['key_order_counts'] = dict(Counter(c['order'] for c in comparisons))
    easy_a = sorted((e for e in reference.demands if e.purpose=='easy'),key=lambda e:e.day)
    easy_b = sorted((e for e in candidate.demands if e.purpose=='easy'),key=lambda e:e.day)
    r['easy_opportunities'] = ([_opportunity(reference,candidate,a,b,history) for a,b in zip(easy_a,easy_b)]
                               if len(easy_a)==len(easy_b) else None)
    r['changed_easy_dates'] = (sum(a.day != b.day for a,b in zip(easy_a,easy_b))
                              if len(easy_a)==len(easy_b) else None)
    if all(k in {'equal','candidate_higher'} for k in key_orders) and 'candidate_higher' in key_orders:
        return finish('reject_uncompensated_key_concentration')
    if any(k not in {'equal','candidate_lower'} for k in key_orders):
        return finish('key_concentration_tradeoff_requires_review')
    orders = {c['order'] for c in comparisons}
    if orders <= {'equal','right_no_greater'} and 'right_no_greater' in orders:
        r['total_concentration_tradeoff'] = r['concentration']['total_miles']['order'] in {'candidate_higher','crossing'}
        return finish('provisional_spacing_trial_not_recovery_approval')
    if orders == {'equal'}:
        return finish('no_key_preparation_preference')
    return finish('receiving_session_tradeoff_requires_review')
