"""Offline physical-program accounting; no scalar score or recovery clearance."""
from dataclasses import dataclass
from functools import lru_cache
from datetime import date, timedelta
from math import isfinite


@dataclass(frozen=True)
class ProgramExposure:
    key: str
    day: date
    miles: float
    role: str
    source: str
    easy_miles: float | None = None
    quality_purpose: str | None = None
    work_bout_seconds: tuple[float, ...] | None = None

    def __post_init__(self):
        if type(self.day) is not date or not self.key or self.source not in {'observed', 'planned'}:
            raise ValueError('Physical date, reconciled identity and source required')
        if not isfinite(self.miles) or self.miles <= 0 or self.role not in {'easy', 'long', 'quality', 'unknown'}:
            raise ValueError('Positive exposure and explicit role required')
        if self.source == 'planned' and abs(self.miles*10-round(self.miles*10)) > 1e-8:
            raise ValueError('Planned distance must commit to one decimal')
        if self.easy_miles is not None and (not isfinite(self.easy_miles) or not 0 <= self.easy_miles <= self.miles):
            raise ValueError('Easy attribution must be within total exposure')
        if self.work_bout_seconds is not None and any(not isfinite(x) or x <= 0 for x in self.work_bout_seconds):
            raise ValueError('Positive finite work bouts required')
        if self.role != 'quality' and (self.quality_purpose is not None or self.work_bout_seconds is not None):
            raise ValueError('Quality attributes require a quality-role entry')


@dataclass(frozen=True)
class TargetSegment:
    start: date
    end: date
    miles_per_7_days: float

    def __post_init__(self):
        if type(self.start) is not date or type(self.end) is not date or self.end <= self.start:
            raise ValueError('Half-open physical target interval required')
        if not isfinite(self.miles_per_7_days) or self.miles_per_7_days <= 0:
            raise ValueError('Positive explicit target rate required')


def _target_total(segments, start, end):
    segments=tuple(segments)
    # Only immutable validated records enter the value cache. Preserve the
    # existing general reader for custom/unhashable diagnostic inputs.
    if type(start) is date and type(end) is date and all(type(s) is TargetSegment for s in segments):
        return _target_total_cached(segments,start,end)
    return _target_total_uncached(segments,start,end)


@lru_cache(maxsize=8192)
def _target_total_cached(segments,start,end):
    return _target_total_uncached(segments,start,end)


def _target_total_uncached(segments, start, end):
    overlaps = [(max(start, s.start), min(end, s.end), s.miles_per_7_days) for s in segments
                if s.start < end and s.end > start]
    if sum((b-a).days for a, b, _ in overlaps) != (end-start).days:
        raise ValueError('Target coverage missing for evaluated interval')
    return sum((b-a).days*rate/7 for a, b, rate in overlaps)


def _gaps(days, start, end):
    days = sorted(set(days))
    gaps = [dict(start=a, end=b, days=(b-a).days) for a, b in zip(days, days[1:])
            if a < end and b >= start]
    return dict(bracketed=gaps, maximum_bracketed_days=max((g['days'] for g in gaps), default=None),
        left_censored=not any(d < start for d in days),
        right_censored=not any(d >= end for d in days))


def _streaks(days, start, end):
    streaks = []
    for day in sorted(set(days)):
        if streaks and day == streaks[-1][-1]+timedelta(days=1):
            streaks[-1].append(day)
        else:
            streaks.append([day])
    return [dict(start=s[0], end=s[-1], running_days=len(s)) for s in streaks
            if s[0] < end and s[-1] >= start]


def audit_program(exposures, targets, *, coverage_start, coverage_end,
                  evaluation_starts, spans=(21, 28, 42), coverage_complete=False):
    """Measure fixed physical intervals; source changes never change the dose.

    Coverage is a caller assertion about this ledger, not inferred from run
    density. Sparse uploads alone cannot support a verified funding verdict.
    """
    exposures = tuple(sorted(exposures, key=lambda e: (e.day, e.key)))
    targets = tuple(sorted(targets, key=lambda t: t.start))
    starts, spans = tuple(evaluation_starts), tuple(spans)
    if type(coverage_start) is not date or type(coverage_end) is not date or coverage_end <= coverage_start:
        raise ValueError('Explicit ledger coverage required')
    if type(coverage_complete) is not bool:
        raise ValueError('Explicit completeness assertion required')
    if len({e.key for e in exposures}) != len(exposures):
        raise ValueError('Duplicate actual/projected identity')
    if any(not coverage_start <= e.day < coverage_end for e in exposures):
        raise ValueError('Exposure outside declared coverage')
    if any(b.start < a.end for a, b in zip(targets, targets[1:])):
        raise ValueError('Overlapping targets would double count requirements')
    if not starts or len(set(starts)) != len(starts) or any(type(d) is not date for d in starts):
        raise ValueError('Unique physical evaluation starts required')
    if not spans or len(set(spans)) != len(spans) or any(type(n) is not int or n <= 0 for n in spans):
        raise ValueError('Unique positive diagnostic spans required')
    rows = []
    for start in sorted(starts):
        for span in sorted(spans):
            end = start+timedelta(days=span)
            if start < coverage_start or end > coverage_end:
                raise ValueError('Ledger does not cover requested interval')
            selected = [e for e in exposures if start <= e.day < end]
            target = _target_total(targets, start, end)
            total = sum(e.miles for e in selected)
            by_day = {}
            for e in selected:
                by_day[e.day] = by_day.get(e.day, 0.)+e.miles
            easy_low = sum(e.easy_miles or 0. for e in selected)
            easy_unknown = [e.key for e in selected if e.easy_miles is None]
            quality = [e for e in selected if e.role == 'quality']
            work_known = sum(sum(e.work_bout_seconds or ()) for e in quality)
            work_unknown = [e.key for e in quality if e.work_bout_seconds is None]
            nonlong_low = sum(e.miles for e in selected if e.role not in {'long', 'unknown'})
            role_unknown = [e.key for e in selected if e.role == 'unknown']
            gaps = {role: _gaps([e.day for e in exposures if role == 'all' or e.role == role], start, end)
                    for role in ('all', 'easy', 'long', 'quality')}
            streaks = _streaks([e.day for e in exposures], start, end)
            rows.append(dict(start=start, end=end, days=span, miles=total, target_miles=target,
                difference_miles=total-target, target_ratio=total/target,
                miles_per_7_days=total*7/span, running_days=len(by_day),
                running_days_per_7_days=len(by_day)*7/span,
                nonlong_miles=nonlong_low if not role_unknown else None,
                nonlong_miles_lower=nonlong_low,
                nonlong_miles_upper=nonlong_low+sum(e.miles for e in selected if e.role == 'unknown'),
                role_unknown_keys=role_unknown,
                long_miles=sum(e.miles for e in selected if e.role == 'long'),
                easy_miles_lower=easy_low,
                easy_miles_upper=easy_low+sum(e.miles for e in selected if e.easy_miles is None),
                easy_unknown_keys=easy_unknown,
                largest_day_miles=max(by_day.values(), default=None),
                largest_day_fraction=max(by_day.values())/total if total else None,
                gaps=gaps, running_streaks=streaks,
                maximum_running_streak=max((s['running_days'] for s in streaks), default=0),
                role_running_days={role: len({e.day for e in selected if e.role == role}) for role in ('easy', 'long', 'quality', 'unknown')},
                known_quality_work_seconds=work_known,
                total_quality_work_seconds=None if work_unknown else work_known,
                quality_work_unknown_keys=work_unknown,
                quality_sessions=[dict(key=e.key, day=e.day, purpose=e.quality_purpose,
                    work_bout_seconds=e.work_bout_seconds) for e in quality]))
    ratios = [r['target_ratio'] for r in rows]
    status = ('all_checked_intervals_below_target' if max(ratios) < 1-1e-9 else
              'all_checked_intervals_at_or_above_target' if min(ratios) >= 1-1e-9 else
              'mixed_or_boundary_sensitive')
    return dict(intervals=rows, numerical_funding_pattern=status,
        funding_evidence_status=status if coverage_complete else 'unverified_ledger_coverage',
        target_ratio_range=(min(ratios), max(ratios)), coverage_complete=coverage_complete,
        recovery_feasibility='not_assessed', progression_accepted=False, planner_influence=False)


def compare_allocation(reference, candidate):
    """Compare matched intervals without weighting training purposes together."""
    before = {(r['start'], r['end']): r for r in reference['intervals']}
    after = {(r['start'], r['end']): r for r in candidate['intervals']}
    if set(before) != set(after):
        raise ValueError('Allocation comparison requires the same physical intervals')
    rows = []
    for key in sorted(before):
        a, b = before[key], after[key]
        easy_change = (b['easy_miles_lower']-a['easy_miles_upper'], b['easy_miles_upper']-a['easy_miles_lower'])
        quality_known = a['total_quality_work_seconds'] is not None and b['total_quality_work_seconds'] is not None
        def quality_shape(row):
            return sorted((e['purpose'] or '', e['work_bout_seconds'] is None,
                           tuple(e['work_bout_seconds'] or ())) for e in row['quality_sessions'])
        rows.append(dict(start=key[0], end=key[1], miles_change=b['miles']-a['miles'],
            nonlong_miles_change=(b['nonlong_miles']-a['nonlong_miles']
                                 if a['nonlong_miles'] is not None and b['nonlong_miles'] is not None else None),
            nonlong_miles_change_range=(b['nonlong_miles_lower']-a['nonlong_miles_upper'],
                                       b['nonlong_miles_upper']-a['nonlong_miles_lower']),
            long_miles_change=b['long_miles']-a['long_miles'],
            easy_miles_change_range=easy_change,
            quality_work_seconds_change=(b['total_quality_work_seconds']-a['total_quality_work_seconds'] if quality_known else None),
            quality_structure_or_purpose_changed=quality_shape(a) != quality_shape(b),
            quality_dates_changed=([e['day'] for e in a['quality_sessions']] !=
                                   [e['day'] for e in b['quality_sessions']]),
            running_days_change=b['running_days']-a['running_days'],
            maximum_gap_days_before=a['gaps']['all']['maximum_bracketed_days'],
            maximum_gap_days_after=b['gaps']['all']['maximum_bracketed_days'],
            interpretation='Allocation changes, not physiological benefit or automatic acceptance'))
    return rows


def screen_development(reference, candidate):
    """Route to the next unresolved comparison, never approve progression.

    This trial preserves supporting volume and quality dose while developing
    aerobic volume/endurance. It is not a universal policy for every training
    phase. Mixed cutoff evidence is reviewed, not automatically rejected.
    """
    changes = compare_allocation(reference, candidate)
    reasons = []
    if not reference['coverage_complete'] or not candidate['coverage_complete']:
        reasons.append('unverified_ledger_coverage')
    pattern = candidate['funding_evidence_status']
    if pattern == 'all_checked_intervals_below_target':
        reasons.append('underfunded_across_checked_intervals')
    elif pattern == 'mixed_or_boundary_sensitive':
        reasons.append('cutoff_or_delivery_pattern_requires_review')
    if all(r['nonlong_miles_change_range'][1] < -1e-8 for r in changes):
        reasons.append('supporting_distance_reduced_across_checked_intervals')
    if any(r['quality_structure_or_purpose_changed'] for r in changes):
        reasons.append('quality_dose_or_purpose_changed')
    if any(r['quality_work_seconds_change'] is None for r in changes):
        reasons.append('quality_work_comparison_incomplete')
    return dict(next_check='allocation_or_evidence_review' if reasons else 'dose_specific_recovery_and_distribution',
        review_reasons=reasons, progression_accepted=False, running_authorized=False,
        note='Passing numerical allocation checks does not clear gaps, compression, recovery or progression')
