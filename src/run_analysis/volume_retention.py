"""Offline volume-stage proposals, not a physiological detraining model.

No calendar search, rolling target reset, or automatic stage acceptance. Fixed
review blocks must come from a complete execution/issued-opportunity ledger.
"""
from dataclasses import dataclass, replace
from datetime import datetime, time, timedelta
from math import isfinite
from zoneinfo import ZoneInfo

from .athlete_projection import aware
from .staged_progression import MILE, tenth


def recent_volume_proposal(history, as_of, baseline, *, timezone, evidence_days=14):
    """One-time candidate anchored to the latest completed local training date.

    Retain baseline separately for historical target accounting. Never call this
    to rebase an accepted stage on refresh. This is not an automatic growth rule.
    """
    aware(as_of)
    if as_of < baseline.issued_at:
        raise ValueError('Starting adjustment precedes baseline')
    if type(evidence_days) is not int or evidence_days <= 0:
        raise ValueError('Positive evidence duration required')
    known = sorted((s for s in history if s.end <= as_of and s.available_at <= as_of),
                   key=lambda s: (s.end, s.key))
    result = dict(candidate=None, historical_stage=baseline, accepted=False,
                  physiology_established=False, evidence_days=evidence_days)
    if not known:
        return dict(result, status='missing_history')
    if len({s.key for s in known}) != len(known):
        raise ValueError('Duplicate completed evidence')
    zone = ZoneInfo(timezone)
    end = datetime.combine(known[-1].end.astimezone(zone).date()+timedelta(days=1), time(), zone)
    start = end-timedelta(days=evidence_days)
    recent = [s for s in known if start <= s.end < end]
    result.update(evidence_start=start, evidence_end=end, source_keys=[s.key for s in recent])
    if as_of-known[-1].end >= timedelta(days=evidence_days):
        return dict(result, status='stale_history_requires_return_review')
    if known[0].start > start or any(not s.capacity_eligible for s in recent):
        return dict(result, status='limited_or_excluded_evidence')
    volume = tenth(sum(s.meters for s in recent)/MILE*7/evidence_days)
    result['observed_volume_per_7'] = volume
    if volume <= baseline.weekly_equivalent_miles:
        return dict(result, status='no_higher_recent_volume_candidate')
    candidate = replace(baseline, key=baseline.key+':recent-volume', issued_at=as_of,
        weekly_equivalent_miles=volume,
        source='provisional_recent_volume:'+','.join(s.key for s in recent))
    return dict(result, status='provisional_recent_volume_candidate', candidate=candidate,
        requires_whole_program_evaluation=True,
        limitations=['Recorded exposure is not sustainable capacity',
                    'Fourteen-day bootstrap evidence choice is provisional, not a daily target window',
                    'History completeness is unverified; proposal is not accepted automatically'])


@dataclass(frozen=True)
class VolumeReviewBlock:
    start: datetime
    end: datetime
    issued_miles: float
    credited_miles: float
    actual_miles: float
    recording_complete: bool
    available_at: datetime
    source: str
    intent: str = 'ordinary'  # confirmed_break / planned_deload are explicit.

    def __post_init__(self):
        for stamp in (self.start, self.end, self.available_at):
            aware(stamp)
        if (self.end <= self.start or self.available_at < self.end or not self.source
                or self.intent not in {'ordinary', 'confirmed_break', 'planned_deload'}
                or type(self.recording_complete) is not bool
                or any(not isfinite(v) or v < 0 for v in
                       (self.issued_miles, self.credited_miles, self.actual_miles))):
            raise ValueError('Complete explicit block chronology, volume and provenance required')
        if self.intent == 'confirmed_break' and (self.actual_miles or self.credited_miles):
            raise ValueError('A confirmed complete break cannot contain running')


@dataclass(frozen=True)
class VolumeRetentionPolicy:
    review_days: int = 14
    partial_confirmation_blocks: int = 2
    half_life_days: float = 28.

    def __post_init__(self):
        if (type(self.review_days) is not int or self.review_days <= 0
                or type(self.partial_confirmation_blocks) is not int or self.partial_confirmation_blocks < 1
                or not isfinite(self.half_life_days) or self.half_life_days <= 0):
            raise ValueError('Explicit positive retention policy required')


def review_volume_retention(stage, blocks, as_of, *, policy=VolumeRetentionPolicy()):
    """Replay fixed blocks from a fixed anchor; refresh never compounds decay.

    Within-band completions should earn issued nominal credit, exactly as in the
    existing progression ledger. Raw actual volume stays separate for load. The
    caller must certify recording completeness, not infer rest from no uploads.
    This computes a proposal only: long/quality doses and return constraints must
    be re-evaluated before a lower-volume stage can be accepted.
    """
    aware(as_of)
    if as_of < stage.issued_at:
        raise ValueError('Review precedes stage')
    rows = sorted((b for b in blocks if b.end <= as_of), key=lambda b: b.start)
    target = stage.weekly_equivalent_miles
    retained = target
    previous = stage.issued_at
    pending = []
    reduction_active = False
    audit = []
    weight = 2**(-policy.review_days/policy.half_life_days)
    reason = 'hold_target'
    for block in rows:
        if block.start != previous or block.end != block.start+timedelta(days=policy.review_days):
            raise ValueError('Contiguous fixed review blocks from the unchanged stage anchor required')
        if block.available_at > as_of:
            # A later known block cannot bridge an earlier not-yet-known one.
            break
        previous = block.end
        required = target*policy.review_days/7
        rate = min(target, block.credited_miles*7/policy.review_days)
        if not block.recording_complete:
            pending.clear()
            reduction_active = False
            reason = 'incomplete_recording_hold'
        elif block.intent == 'ordinary' and block.issued_miles+1e-8 < required:
            pending.clear()
            reduction_active = False
            reason = 'planner_under_delivery_review'
        elif rate+1e-8 >= target:
            pending.clear()
            reduction_active = False
            reason = 'delivery_restored_growth_review_separate'
        else:
            pending.append(rate)
            if reduction_active or block.intent == 'confirmed_break' or len(pending) >= policy.partial_confirmation_blocks:
                # Catch up the first confirmed low blocks exactly once. Thereafter
                # only this newly completed block updates the retained exposure.
                for observed in pending:
                    retained = min(retained, observed+(retained-observed)*weight)
                pending.clear()
                reduction_active = True
                reason = 'reduced_volume_proposal'
            else:
                reason = 'planned_deload_hold' if block.intent == 'planned_deload' else 'await_sustained_reduction'
        audit.append(dict(end=block.end, reason=reason, credited_rate=rate,
                          retained_volume=retained, source=block.source))
    # Never manufacture additional completed blocks from passage of time alone.
    due = stage.issued_at+timedelta(days=(len(audit)+1)*policy.review_days)
    if as_of >= due:
        reason = 'missing_review_block_hold'
    proposed = tenth(retained)
    actionable = (reason == 'reduced_volume_proposal' and proposed < target)
    return dict(status=reason, proposed_volume_per_7=proposed if actionable else None,
        retained_exposure_volume_per_7=proposed, accepted_stage_unchanged=stage,
        reviewed_through=previous, next_review_at=due, audit=audit,
        requires_return_dose_and_quality_review=actionable,
        accepted=False, physiology_established=False,
        scope='offline volume proposal; not automatic stage publication or a fitness-loss estimate')
