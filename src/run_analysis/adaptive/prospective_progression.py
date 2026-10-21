"""Offline prospective counterpart to the existing realized-volume gate.

No score weight, future adaptation claim, calendar-week quota or recovery
override. The first review checkpoint is explicitly frozen; postponing a run
beyond it cannot fund it. This trial does not define missed-checkpoint recovery.
"""
from dataclasses import dataclass
from datetime import datetime,timedelta
from math import isfinite
from zoneinfo import ZoneInfo

from run_analysis.athlete_projection import aware
from run_analysis.stage_target_history import target_segments,target_miles_between
from run_analysis.staged_progression import progression_distance_credit


@dataclass(frozen=True)
class ProspectiveVolumeReview:
    stage_key: str
    available_at: datetime
    start: datetime
    end: datetime
    required_miles: float
    completed_credit: tuple[tuple[str,float],...]
    timezone: str
    source: str

    def __post_init__(self):
        for at in (self.available_at,self.start,self.end):aware(at)
        if (not self.stage_key or not self.source or self.start>=self.end
                or self.available_at>=self.end or not isfinite(self.required_miles)
                or self.required_miles<=0):
            raise ValueError('Explicit future progression checkpoint and positive dated requirement required')
        ZoneInfo(self.timezone)
        rows=tuple(tuple(row) for row in self.completed_credit)
        if len({k for k,_ in rows})!=len(rows) or any(not k or not isfinite(v) or v<0 for k,v in rows):
            raise ValueError('Unique nonnegative observed progression credits required')
        object.__setattr__(self,'completed_credit',rows)


def freeze_volume_review(stage,target_history,entries,as_of,*,review_at,timezone):
    """Freeze the SAME exact interval/target integral used at realized review.

    ``review_at`` is a declared decision checkpoint, not a run-time or recovery
    constraint. Actual past credit is computed from durable issued contracts;
    missed past sessions are not assumed completed. A missed checkpoint needs
    separate interpretation, not an automatically sliding funding deadline.
    """
    aware(as_of);aware(review_at)
    if stage.policy.volume_path!='gradual':
        raise ValueError('Prospective trial currently supports the bounded gradual-volume phase only')
    if review_at<stage.review_at or review_at<=as_of:
        raise ValueError('Review checkpoint must be future and at/after the existing minimum review interval')
    zone=ZoneInfo(timezone)
    start=max(stage.issued_at,review_at-timedelta(days=stage.policy.review_days))
    volume,_=target_segments(target_history,as_of=as_of,
        start=start.astimezone(zone).date(),end=review_at.astimezone(zone).date()+timedelta(days=1))
    credits=[];seen=set();done_keys=set()
    for entry in entries:
        if entry.key in seen:raise ValueError('Duplicate opportunity')
        seen.add(entry.key)
        done=entry.completion
        if done is None:continue
        if done.key in done_keys:raise ValueError('Completion counted twice')
        done_keys.add(done.key)
        if (start<=entry.scheduled_at<review_at and start<=done.start<review_at
                and done.end<=as_of and done.available_at<=as_of):
            credits.append((entry.key,progression_distance_credit(entry.contract,done.distance_meters)['credited_miles']))
    return ProspectiveVolumeReview(stage.key,as_of,start,review_at,
        target_miles_between(volume,start,review_at,zone),tuple(sorted(credits)),timezone,
        'Expected prescription completion at the declared first review; realized gate unchanged')


def expected_volume_review(review,planned):
    """Funding only, not clearance, completed training or stage acceptance.

    Candidate dates are freely searchable. Planned mileage is nominal distance,
    consistent with within-band realized credit. Forecast workload is untouched.
    Advisory starts before capture are moved forward, never backdated; sessions
    crossing the checkpoint do not earn completion credit at that checkpoint.
    """
    rows=tuple(planned)
    if len({r.key for r in rows})!=len(rows):raise ValueError('Duplicate future session')
    credit=sum(v for _,v in review.completed_credit)
    included=[];excluded=[]
    for r in rows:
        if r.source!='planned':raise ValueError('Expected review requires planned work only')
        if not isfinite(r.miles) or r.miles<0:raise ValueError('Finite nonnegative nominal distance required')
        if r.key in dict(review.completed_credit):raise ValueError('Past completion cannot also be projected')
        start=max(r.start,review.available_at)
        end=start+(r.end-r.start)
        if review.start<=start<review.end and end<=review.end:
            included.append(r.key);credit+=r.miles
        else:excluded.append(r.key)
    gap=max(0.,review.required_miles-credit)
    return dict(status='expected_volume_funded' if credit+1e-8>=review.required_miles else 'expected_volume_underfunded',
        funded=credit+1e-8>=review.required_miles,required_miles=review.required_miles,
        expected_credited_miles=credit,observed_credited_miles=sum(v for _,v in review.completed_credit),
        shortfall_miles=gap,start=review.start.isoformat(),end=review.end.isoformat(),
        included_keys=tuple(included),excluded_keys=tuple(excluded),
        scope='Prospective volume gate only; other progression, health and recovery checks remain separate',
        future_completion_assumed=True,raw_workload_unchanged=True)
