"""Descriptive exposure baseline, not a capacity or recovery prescription."""
from dataclasses import dataclass
from datetime import datetime, timedelta
from math import isfinite
from zoneinfo import ZoneInfo

from .athlete_projection import aware


@dataclass(frozen=True)
class SessionIntent:
    key: str
    purpose: str
    available_at: datetime
    source: str

    def __post_init__(self):
        aware(self.available_at)
        if not self.key or not self.source or self.purpose not in {
            'ordinary', 'medium_long', 'support'
        }:
            raise ValueError('Explicit session intent and provenance required')


@dataclass(frozen=True)
class UntimedSession:
    key: str
    start: datetime
    available_at: datetime
    source: str

    def __post_init__(self):
        aware(self.start)
        aware(self.available_at)
        if not self.key or not self.source or self.available_at < self.start:
            raise ValueError('Untimed record identity and availability required')


@dataclass(frozen=True)
class RecordingCoverage:
    since: datetime
    available_at: datetime
    source: str

    def __post_init__(self):
        aware(self.since)
        aware(self.available_at)
        if not self.source or self.available_at < self.since:
            raise ValueError('Independent coverage assertion required')


def distribution(values):
    """Linear empirical quantiles, never statistical confidence bounds."""
    values = sorted(values)
    if not values:
        return dict(count=0, minimum=None, q25=None, median=None, q75=None, maximum=None)

    def quantile(p):
        position = (len(values)-1)*p
        left = int(position)
        return values[left] + (values[min(left+1, len(values)-1)]-values[left])*(position-left)

    return dict(count=len(values), minimum=values[0], q25=quantile(.25),
                median=quantile(.5), q75=quantile(.75), maximum=values[-1])


def doses(sessions):
    return dict(keys=[s.key for s in sessions],
        starts=[s.start for s in sessions],
        start_gaps_days=[(b.start-a.start).total_seconds()/86400 for a, b in zip(sessions, sessions[1:])],
        meters=distribution(s.meters for s in sessions),
        moving_minutes=distribution(s.moving_seconds/60 for s in sessions))


def demonstrated_baseline(history, as_of, *, timezone='UTC', intents=(),
                          untimed=(), coverage=None, windows=(7, 14, 28, 84)):
    """Use original exposure even when HR units differ or capacity is excluded.

    Role/health flags must already reflect caller's metadata snapshot. Intent and
    recording assertions have separate availability cuts. No dose recommendation.
    """
    aware(as_of)
    zone = ZoneInfo(timezone)
    history, untimed, intents, windows = map(tuple, (history, untimed, intents, windows))
    if not windows or any(not isfinite(d) or d <= 0 for d in windows) or len(set(windows)) != len(windows):
        raise ValueError('Unique positive reporting windows required')
    keys = [s.key for s in (*history, *untimed)]
    if len(keys) != len(set(keys)) or len({i.key for i in intents}) != len(intents):
        raise ValueError('Duplicate evidence/intent identity')
    accepted, excluded = [], []
    for s in sorted(history, key=lambda s: (s.start, s.key)):
        reason = ('not_completed' if s.end > as_of else
                  'not_available' if s.available_at > as_of else None)
        if reason:
            excluded.append(dict(key=s.key, reason=reason))
        else:
            accepted.append(s)
    if any(b.start < a.end for a, b in zip(accepted, accepted[1:])):
        raise ValueError('Overlapping sources require reconciliation')
    known_intents = {i.key: i for i in intents if i.available_at <= as_of}
    reports = {}
    for days in windows:
        cutoff = as_of-timedelta(days=days)
        recent = [s for s in accepted if s.end > cutoff]
        eligible = [s for s in recent if s.capacity_eligible]
        easy = [s for s in eligible if s.role == 'easy']
        ordinary = [s for s in easy if s.key in known_intents and
                    known_intents[s.key].purpose in {'ordinary', 'medium_long'}]
        support = [s for s in easy if s.key in known_intents and
                   known_intents[s.key].purpose == 'support']
        unknown = [s.key for s in easy if s.key not in known_intents]
        quality = [s for s in recent if s.role in {'intervals', 'tempo_threshold', 'race'}]
        # Unknown end time cannot be assigned confidently to a completion window;
        # keep all available unresolved earlier records visible conservatively.
        unresolved = [s.key for s in untimed if s.start <= as_of and s.available_at <= as_of]
        unavailable_untimed = [s.key for s in untimed if s.start <= as_of < s.available_at]
        unavailable = [s.key for s in history if cutoff < s.end <= as_of < s.available_at]
        complete = bool(coverage and coverage.available_at <= as_of and
                        coverage.since <= cutoff and not unresolved and not unavailable and not unavailable_untimed)
        total_meters = sum(s.meters for s in recent)
        total_minutes = sum(s.moving_seconds for s in recent)/60
        cardio = {}
        for signature in sorted({s.accounting_key for s in recent}):
            group = [s for s in recent if s.accounting_key == signature]
            measured = [s.cardio for s in group if s.cardio is not None]
            cardio[signature] = dict(keys=[s.key for s in group],
                missing_keys=[s.key for s in group if s.cardio is None],
                recorded_lower=sum(q.low for q in measured),
                total_upper=sum(q.high for q in measured) if len(measured) == len(group) else None)
        reports[str(days)] = dict(
            days=days, completion_cutoff=cutoff, run_count=len(recent),
            local_running_dates=sorted({s.start.astimezone(zone).date().isoformat() for s in recent}),
            total_meters=total_meters, total_moving_minutes=total_minutes,
            recorded_runs_per_7_days=len(recent)*7/days,
            recorded_meters_per_7_days=total_meters*7/days,
            largest_session_distance_fraction=(max(s.meters for s in recent)/total_meters if total_meters else None),
            start_gaps_days=[(b.start-a.start).total_seconds()/86400 for a, b in zip(recent, recent[1:])],
            all_sessions=doses(recent), reference_eligible=doses(eligible),
            reference_excluded_keys=[s.key for s in recent if not s.capacity_eligible],
            eligible_easy=doses(easy), ordinary=doses(ordinary), support=doses(support),
            easy_intent_unknown_keys=unknown,
            ordinary_reference_status='explicit_intent_observations' if ordinary else 'unresolved_use_easy_distribution_descriptively',
            long=doses([s for s in recent if s.role == 'long']), quality=doses(quality),
            quality_work=[dict(key=s.key, total_sampled_work_seconds=(sum(p.seconds for p in s.pace_samples if p.phase == 'work')
                if any(p.phase == 'work' for p in s.pace_samples) else None),
                longest_sampled_bout_seconds=max((p.seconds for p in s.pace_samples if p.phase == 'work'), default=None),
                sample_count=sum(p.phase == 'work' for p in s.pace_samples),
                sources=sorted({p.basis for p in s.pace_samples if p.phase == 'work'}),
                structure_status='observed_samples_not_guaranteed_complete' if any(p.phase == 'work' for p in s.pace_samples) else 'unknown')
                for s in quality],
            cardio_by_accounting_signature=cardio,
            history_status='asserted_complete' if complete else 'recorded_only',
            unresolved_timing_keys=unresolved, unavailable_completion_keys=unavailable,
            unavailable_untimed_keys=unavailable_untimed)
    return dict(as_of=as_of, timezone=timezone, windows=reports, excluded=excluded,
        last_completion=accepted[-1].end if accepted else None,
        days_since_last_completion=(as_of-accepted[-1].end).total_seconds()/86400 if accepted else None,
        intent_sources={k: dict(purpose=i.purpose, source=i.source, available_at=i.available_at)
                        for k, i in known_intents.items() if k in {s.key for s in accepted}},
        coverage_source=coverage.source if coverage and coverage.available_at <= as_of else None,
        capacity_established=False, planner_influence=False,
        convention='whole_completed_sessions; empirical_dispersion_not_confidence_intervals')
