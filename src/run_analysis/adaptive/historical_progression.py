"""Initialize from already observed training, without invented prescriptions.

This is a present-time interpretation of known original evidence, not backdated
publication or elapsed-time growth credit. Raw physical history and completion
receipts share ONE physical identity; a receipt never adds its run twice.
"""
from dataclasses import dataclass
from datetime import datetime,time,timedelta
from math import isfinite

from run_analysis.athlete_projection import aware
from run_analysis.stage_target_history import target_segments
from run_analysis.staged_progression import MILE
from .smoothed_progression import kernel,mass,days_between

VERSION='historical_observed_v1'


@dataclass(frozen=True)
class Exposure:
    key: str
    start: datetime
    end: datetime
    available_at: datetime
    miles: float
    role: str
    provenance: str


def observations(history,entries,as_of,*,excluded_keys=()):
    """Actual-only identity reconciliation; no old issued target is required."""
    aware(as_of)
    history=tuple(history);entries=tuple(entries);excluded=set(excluded_keys)
    if len({r.key for r in history})!=len(history):
        raise ValueError('Duplicate physical history')
    if len({e.key for e in entries})!=len(entries):
        raise ValueError('Duplicate opportunity')
    done_keys=[e.completion.key for e in entries if e.completion]
    if len(done_keys)!=len(set(done_keys)):
        raise ValueError('Completion counted twice')
    rows={}
    for r in history:
        if (r.source!='observed' or r.key in excluded or r.end>as_of or r.available_at>as_of):
            continue
        rows[r.key]=Exposure(r.key,r.start,r.end,r.available_at,r.miles,r.role,'original_physical_history')
    for e in entries:
        d=e.completion
        if (e.contract.issued_at>as_of or not d or d.key in excluded
                or d.end>as_of or d.available_at>as_of):
            continue
        old=rows.get(d.key)
        if old and (old.start!=d.start or old.end!=d.end):
            raise ValueError('Physical history and completion chronology need reconciliation')
        if d.distance_meters is None:
            # An independent known raw distance can still count, but absence
            # of both sources never becomes nominal prescribed distance.
            continue
        if old and not d.distance_meters.low-1e-5<=old.miles*MILE<=d.distance_meters.high+1e-5:
            raise ValueError('Physical history and completion distance need reconciliation')
        role=old.role if old else d.role or (
            'long' if e.contract.purpose=='long_endurance' else
            'easy' if e.contract.purpose=='easy_volume' else 'quality')
        rows[d.key]=Exposure(d.key,d.start,d.end,d.available_at,d.distance_meters.low/MILE,
            role,'observed_completion_lower_bound')
    result=tuple(sorted(rows.values(),key=lambda r:(r.start,r.key)))
    if any(b.start<a.end for a,b in zip(result,result[1:])):
        raise ValueError('Overlapping physical evidence needs reconciliation')
    return result


def support(records,entries,targets,*,as_of,zone,review_days=14):
    """All eligible actuals own exposure; only existing dated targets compare.

    Older unprescribed runs count toward observed volume and observation age.
    No target is fabricated before the first known dated reference. Exposure
    and comparison therefore disclose separate anchors when coverage differs.
    """
    if type(review_days) is not int or review_days<=0 or not records:
        raise ValueError('Positive memory and actual historical observations required')
    anchor=min(r.start for r in records)
    if as_of<=anchor:
        raise ValueError('Elapsed observed history required')
    available=[t for t in targets if t.available_at<=as_of]
    if not available:
        raise ValueError('Known dated target reference required')
    reference=max(anchor,datetime.combine(min(t.effective_on for t in available),time.min,zone))
    if reference>=as_of:
        raise ValueError('Elapsed dated target reference required')
    tau=review_days/3
    norm=mass(days_between(as_of,anchor),tau)
    refnorm=mass(days_between(as_of,reference),tau)
    actual=7*sum(r.miles*kernel(days_between(as_of,r.end),tau) for r in records)/norm
    comparison=7*sum(r.miles*kernel(days_between(as_of,r.end),tau)
        for r in records if r.start>=reference)/refnorm
    segments,_=target_segments(targets,as_of=as_of,start=reference.astimezone(zone).date(),
        end=as_of.astimezone(zone).date()+timedelta(days=1))
    target=0.
    for t in segments:
        left=max(reference,datetime.combine(t.start,time.min,zone))
        right=min(as_of,datetime.combine(t.end,time.min,zone))
        if left<right:
            target+=t.miles_per_7_days*(mass(days_between(as_of,left),tau)-mass(days_between(as_of,right),tau))
    target/=refnorm
    if not isfinite(target) or target<=0:
        raise ValueError('Positive elapsed target reference required')
    return dict(version=VERSION,as_of=as_of.isoformat(),anchor=anchor.isoformat(),
        reference_anchor=reference.isoformat(),mean_age_days=review_days,kernel_order=3,
        time_constant_days=tau,elapsed_mass=norm,actual_per_7=actual,
        comparison_actual_per_7=comparison,target_per_7=target,delivery_ratio=comparison/target,
        completed_keys=[r.key for r in records],issued_per_7=None,planner_ratio=None,execution_ratio=None,
        observed_history_days=days_between(as_of,anchor),future_credit=False,
        actual_credit_basis='deduplicated_original_history_and_completion_lower_bound',
        interpretation='Historical recorded exposure, not elapsed-time growth or established fitness')
