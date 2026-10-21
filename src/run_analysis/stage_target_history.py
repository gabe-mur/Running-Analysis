"""Dated accepted targets, distinct from measured training and rolling capacity."""
from dataclasses import dataclass
from datetime import date,datetime,timedelta,time
from math import isfinite,log,log1p

from .athlete_projection import aware
from .program_delivery import TargetSegment


@dataclass(frozen=True)
class StageTarget:
    stage_key: str
    available_at: datetime
    effective_on: date
    volume_per_7: float
    supporting_easy_per_7: float
    source: str
    ramp_days: int = 0
    volume_goal: float | None = None
    support_goal: float | None = None
    review_anchor: datetime | None = None  # A rate revision does not restart the phase review clock.
    growth_per_7: float | None = None  # Conditional continuous forecast, not earned future delivery.
    growth_limit: float | None = None

    def __post_init__(self):
        aware(self.available_at)
        if self.review_anchor is not None:
            aware(self.review_anchor)
            if self.review_anchor > self.available_at or not self.ramp_days:
                raise ValueError('Rate revision requires a prior phase review anchor and a bounded ramp')
        if (not self.stage_key or not self.source or type(self.effective_on) is not date or
                any(not isfinite(x) or x<=0 for x in (self.volume_per_7,self.supporting_easy_per_7))):
            raise ValueError('Dated positive accepted targets and provenance required')
        if type(self.ramp_days) is not int or self.ramp_days < 0:
            raise ValueError('Nonnegative whole ramp days required')
        if self.ramp_days:
            if any(v is None or not isfinite(v) or v < initial for v,initial in
                   ((self.volume_goal,self.volume_per_7),(self.support_goal,self.supporting_easy_per_7))):
                raise ValueError('Finite nondecreasing ramp endpoints required')
        elif self.volume_goal is not None or self.support_goal is not None:
            raise ValueError('Ramp endpoints require a duration')
        if self.growth_per_7 is not None:
            if (self.ramp_days or not isfinite(self.growth_per_7) or not 0 <= self.growth_per_7 < 1
                    or self.growth_limit is None or not isfinite(self.growth_limit)
                    or self.growth_limit < self.volume_per_7):
                raise ValueError('Continuous growth requires a nonnegative rate and bounded destination')
        elif self.growth_limit is not None:
            raise ValueError('Continuous destination requires a declared growth rate')

    def rate(self,day,field):
        initial=getattr(self,field)
        if self.growth_per_7 is not None:
            # All additional volume is supporting easy volume. This is a
            # prospective scenario; daily review separately owns earned growth.
            days=max(0,(day-self.effective_on).days)
            cap_days=0. if self.growth_per_7 == 0 else log(
                self.growth_limit/self.volume_per_7)/log1p(self.growth_per_7)*7
            volume=self.volume_per_7*(1+self.growth_per_7)**(min(days,cap_days)/7)
            return min(self.growth_limit,volume) if field=='volume_per_7' else initial+min(self.growth_limit,volume)-self.volume_per_7
        if not self.ramp_days:
            return initial
        goal=self.volume_goal if field=='volume_per_7' else self.support_goal
        fraction=min(1.,max(0.,(day-self.effective_on).days/self.ramp_days))
        return initial+(goal-initial)*fraction


def target_segments(records,*,as_of,start,end):
    """Resolve declared target paths, with complete dated coverage.

    The initial provisional record must explicitly cover the lead-in. A later
    growth target cannot be backfilled over that history. Target records are
    not load observations. An explicitly authorized bounded ramp is prospective
    planning intent, not confirmation of future adaptation or an endless ramp.
    """
    aware(as_of)
    if type(start) is not date or type(end) is not date or start>=end:
        raise ValueError('Ordered target coverage dates required')
    available=sorted((r for r in records if r.available_at<=as_of),key=lambda r:(r.effective_on,r.available_at))
    # An intraday continuous review may change TOMORROW'S slope, never today's
    # value. Keep every amendment in the journal; resolve the last known one.
    by_day={}
    for row in available:
        old=by_day.get(row.effective_on)
        if old is not None and not (row.growth_per_7 is not None
                and old.available_at < row.available_at
                and old.rate(row.effective_on,'volume_per_7') == row.volume_per_7
                and old.rate(row.effective_on,'supporting_easy_per_7') == row.supporting_easy_per_7):
            raise ValueError('Unique dated stage targets must cover the complete lead-in; intraday reviews cannot reprice today')
        by_day[row.effective_on]=row
    rows=list(by_day.values())
    if (not rows or rows[0].effective_on>start
            or len({r.stage_key for r in rows})!=len(rows)):
        raise ValueError('Unique dated stage targets must cover the complete lead-in')
    result=[]
    for field in ('volume_per_7','supporting_easy_per_7'):
        segments=[]
        for i,row in enumerate(rows):
            left=max(start,row.effective_on)
            right=min(end,rows[i+1].effective_on if i+1<len(rows) else end)
            if left<right:
                while left<right:
                    changing=(row.growth_per_7 is not None or row.ramp_days and left<row.effective_on+timedelta(days=row.ramp_days))
                    boundary=min(right,left+timedelta(days=1)) if changing else right
                    segments.append(TargetSegment(left,boundary,row.rate(left,field)))
                    left=boundary
        result.append(tuple(segments))
    return tuple(result)


def target_miles_between(segments,start,end,zone):
    """Integrate dated rates over the exact review interval, including partial days."""
    total=0.
    for segment in segments:
        left=datetime.combine(segment.start,time.min,zone)
        right=datetime.combine(segment.end,time.min,zone)
        seconds=(min(end,right)-max(start,left)).total_seconds()
        total+=max(0.,seconds)/86400*segment.miles_per_7_days/7
    return total
