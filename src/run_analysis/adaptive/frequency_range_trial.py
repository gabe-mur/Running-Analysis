"""Offline stage-volume frequency envelope; no seven-day quotas.

Fixed 28-day exposure blocks count distinct running dates. Dated stage targets
change future bounds only, without rebasing the block or repricing past days.
Incomplete projection blocks are explicitly unassessed. This prototype is not
default admission, missed-run recovery, novice purpose policy or deployment.
The AS119/120 helpers retain their historical fixed-block semantics. Explicit v2
contracts use prospective upper admission, advisory lower rates and a single
weighted history/forecast exposure with no discrete full-block activation.
"""
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta, timezone
from math import ceil, floor, isfinite


@dataclass(frozen=True)
class FrequencyRangePolicy:
    anchors: tuple[tuple[float, float, float], ...] = (
        (5., 2., 3.), (22., 3., 5.), (42., 4., 6.), (60., 5., 7.))
    window_days: int = 28
    source: str = 'AS119 provisional volume-linked frequency ranges; not physiological thresholds'

    def __post_init__(self):
        anchors = tuple(tuple(row) for row in self.anchors)
        if (len(anchors) < 2 or any(len(row) != 3 for row in anchors)
                or any(not all(isfinite(v) for v in row) or
                       not (row[0] > 0 and 0 < row[1] <= row[2] <= 7) for row in anchors)
                or any(b[0] <= a[0] or b[1] < a[1] or b[2] < a[2]
                       for a, b in zip(anchors, anchors[1:]))
                or type(self.window_days) is not int or self.window_days < 21 or not self.source):
            raise ValueError('Ordered monotone frequency anchors, multiweek span and provenance required')
        object.__setattr__(self, 'anchors', anchors)

    def bounds(self, sustained_volume):
        if not isfinite(sustained_volume) or sustained_volume <= 0:
            raise ValueError('Positive sustained accepted volume required; no zero-history prescription')
        if sustained_volume <= self.anchors[0][0]:
            return self.anchors[0][1:]
        for a, b in zip(self.anchors, self.anchors[1:]):
            if sustained_volume <= b[0]:
                f = (sustained_volume-a[0])/(b[0]-a[0])
                return tuple(a[i]+f*(b[i]-a[i]) for i in (1, 2))
        return self.anchors[-1][1:]  # Single-session days only; no doubles extrapolated.


@dataclass(frozen=True)
class FrequencyStageTarget:
    stage_key: str
    effective_on: date
    volume_per_7: float
    lower_per_7: float
    upper_per_7: float
    source: str

    def __post_init__(self):
        if (not self.stage_key or not self.source or type(self.effective_on) is not date
                or any(not isfinite(v) for v in (self.volume_per_7, self.lower_per_7, self.upper_per_7))
                or self.volume_per_7 <= 0 or not 0 < self.lower_per_7 <= self.upper_per_7 <= 7):
            raise ValueError('Dated positive committed frequency target required')

    @classmethod
    def freeze(cls, stage_key, effective_on, volume_per_7, *, policy=None):
        policy = policy or FrequencyRangePolicy()
        lower, upper = policy.bounds(volume_per_7)
        return cls(stage_key, effective_on, volume_per_7, lower, upper, policy.source)


def assess_frequency(targets, observed_dates, planned_dates, *, decision, coverage_end, policy=None):
    """Complete upcoming blocks only, using a common original phase anchor.

    Observed completed dates replace planned dates on rollover. Missing history
    is not inferred: adopting callers assert complete date coverage since the
    initial target. No prior closed block can permanently veto future training.
    Planned rest is explicit through coverage_end; no tail session beyond it is
    counted. Published short prefixes must use their same full priced projection.
    """
    policy = policy or FrequencyRangePolicy()
    rows = tuple(sorted(targets, key=lambda r: r.effective_on))
    if (type(decision) is not date or type(coverage_end) is not date or decision >= coverage_end
            or not rows or rows[0].effective_on > decision
            or len({r.stage_key for r in rows}) != len(rows)
            or len({r.effective_on for r in rows}) != len(rows)
            or any(r.effective_on > decision for r in rows)):
        raise ValueError('Unique accepted target history must cover the current decision')
    observed, planned = set(observed_dates), set(planned_dates)
    if (any(type(d) is not date or d > decision for d in observed)
            or any(type(d) is not date or not decision <= d < coverage_end for d in planned)):
        raise ValueError('Observed past and selectable planned date scopes must be reconciled')
    anchor = rows[0].effective_on
    start = anchor+timedelta(days=((decision-anchor).days//policy.window_days)*policy.window_days)
    dates = observed | planned
    blocks = []
    while start+timedelta(days=policy.window_days) <= coverage_end:
        end = start+timedelta(days=policy.window_days)
        lower = upper = 0.
        for i, row in enumerate(rows):
            left = max(start, row.effective_on)
            right = min(end, rows[i+1].effective_on if i+1 < len(rows) else end)
            days = max(0, (right-left).days)
            lower += days*row.lower_per_7/7
            upper += days*row.upper_per_7/7
        minimum, maximum = ceil(lower-1e-9), floor(upper+1e-9)
        count = sum(start <= d < end for d in dates)
        blocks.append(dict(start=start.isoformat(), end=end.isoformat(), count=count,
            required_count_min=minimum, allowed_count_max=maximum,
            lower_expected_count=lower, upper_expected_count=upper,
            observed_count=sum(start <= d < end for d in observed),
            planned_count=sum(start <= d < end for d in planned-observed),
            within_range=minimum <= count <= maximum,
            deficit=max(0, minimum-count), excess=max(0, count-maximum)))
        start = end
    eligible = bool(blocks) and all(r['within_range'] for r in blocks)
    return dict(status='within_range' if eligible else
        'insufficient_projection_coverage' if not blocks else 'outside_range',
        eligible=eligible,
        # Lower-count incomplete construction must remain searchable. No
        # frequency result can override the caller's physical/safety verdict.
        construction_eligible=all(r['excess'] == 0 for r in blocks),
        blocks=blocks, unassessed_from=start.isoformat(),
        coverage_end=coverage_end.isoformat(), anchor=anchor.isoformat(),
        changes_score=False, assumed_history_coverage_complete=True)


def ranged_evaluator(evaluate, targets, observed_dates, *, decision, coverage_end, policy=None):
    """Wrap the EXISTING full evaluator/search, not a new scheduling engine."""
    targets, observed_dates = tuple(targets), tuple(observed_dates)
    def wrapped(program):
        result = evaluate(program)
        audit = assess_frequency(targets, observed_dates,
            (date.fromisoformat(day) for day, _ in program), decision=decision,
            coverage_end=coverage_end, policy=policy)
        return dict(result, eligible=result['eligible'] and audit['eligible'],
            construction_eligible=result.get('construction_eligible', result['eligible'])
                and audit['construction_eligible'], frequency_range=audit)
    return wrapped


@dataclass(frozen=True)
class FrequencyMidpointPreference:
    """Offline coaching utility, not physiological benefit or continuity.

    At either continuous bound the cost is edge_fraction of a FROZEN reference
    fatigue for that exposure. Reference days/fatigue must persist, not be
    recomputed from each candidate or daily refresh. Only complete assessed
    blocks contribute; their partial-tail limitation remains explicit.
    """
    reference_fatigue: float
    reference_days: int
    edge_fraction: float
    center_fraction: float = .5

    def __post_init__(self):
        if (not isfinite(self.reference_fatigue) or self.reference_fatigue <= 0
                or type(self.reference_days) is not int or self.reference_days <= 0
                or not isfinite(self.edge_fraction) or self.edge_fraction < 0
                or not isfinite(self.center_fraction) or not 0 <= self.center_fraction <= 1):
            raise ValueError('Positive frozen fatigue/span and nonnegative edge fraction required')

    def assess(self, audit):
        blocks = []
        for row in audit['blocks']:
            center = (row['lower_expected_count']+row['upper_expected_count'])/2
            if self.center_fraction != .5:
                center = row['lower_expected_count'] + self.center_fraction * (
                    row['upper_expected_count']-row['lower_expected_count'])
            half_width = (row['upper_expected_count']-row['lower_expected_count'])/2
            # A collapsed range has no interior choice. Out-of-range prefixes
            # remain rejected by admission, not charged unbounded utility.
            departure = (row['count']-center)/half_width if half_width > 0 else 0.
            days = row.get('weighted_days',
                (date.fromisoformat(row['end'])-date.fromisoformat(row['start'])).days)
            cost = (self.reference_fatigue*self.edge_fraction*days/self.reference_days
                    * min(departure**2, 1.))
            blocks.append(dict(start=row['start'], end=row['end'], center_count=center,
                normalized_departure=departure, cost=cost))
        return dict(cost=sum(r['cost'] for r in blocks), blocks=blocks,
            edge_fraction=self.edge_fraction, reference_fatigue=self.reference_fatigue,
            reference_days=self.reference_days,
            **({'center_fraction': self.center_fraction} if self.center_fraction != .5 else {}),
            scope=('Explicit midpoint coaching preference; weighted covered exposure, not recovery evidence'
                if audit.get('contract') in {'prospective_weighted_v2', 'prospective_count_v3'} else
                'Explicit midpoint coaching preference; complete blocks only, not recovery evidence')
                if self.center_fraction == .5 else 'Explicit shifted frequency preference; not recovery evidence')


def midpoint_evaluator(evaluate, preference):
    """Add utility to a range-wrapped evaluator WITHOUT changing admission."""
    def wrapped(program):
        result = evaluate(program)
        audit = preference.assess(result['frequency_range'])
        return dict(result, model_cost=result['cost'], cost=result['cost']+audit['cost'],
                    frequency_preference=audit)
    return wrapped


@dataclass(frozen=True)
class FrequencyContract:
    """One durable sandbox adoption; its utility normalization never rolls."""
    adopted_at: datetime
    policy: FrequencyRangePolicy
    targets: tuple[FrequencyStageTarget, ...]
    preference: FrequencyMidpointPreference
    version: str = 'frequency_midpoint_v3'

    def __post_init__(self):
        if (not isinstance(self.adopted_at, datetime) or self.adopted_at.tzinfo is None
                or self.adopted_at.utcoffset() is None
                or not isinstance(self.policy, FrequencyRangePolicy)
                or not isinstance(self.preference, FrequencyMidpointPreference)
                or self.version not in {'frequency_midpoint_v1', 'frequency_midpoint_v2', 'frequency_midpoint_v3'}):
            raise ValueError('Aware explicit frequency adoption and typed policy/reference required')
        rows = tuple(self.targets)
        if (not rows or any(not isinstance(r, FrequencyStageTarget) for r in rows)
                or len({r.stage_key for r in rows}) != len(rows)
                or any(b.effective_on <= a.effective_on for a, b in zip(rows, rows[1:]))
                or any((r.lower_per_7, r.upper_per_7) != self.policy.bounds(r.volume_per_7)
                       or r.source != self.policy.source for r in rows)):
            raise ValueError('Ordered accepted frequency history must match the adopted policy')
        object.__setattr__(self, 'targets', rows)
        object.__setattr__(self, 'adopted_at', self.adopted_at.astimezone(timezone.utc))

    @classmethod
    def freeze(cls, *, adopted_at, stage_key, effective_on, volume_per_7,
               reference_fatigue, reference_days, edge_fraction=.02, policy=None,
               version='frequency_midpoint_v3'):
        policy = policy or FrequencyRangePolicy()
        return cls(adopted_at, policy,
            (FrequencyStageTarget.freeze(stage_key, effective_on, volume_per_7, policy=policy),),
            FrequencyMidpointPreference(reference_fatigue, reference_days, edge_fraction), version)

    def accepted_stage(self, stage_key, effective_on, volume_per_7):
        last = self.targets[-1]
        if volume_per_7 == last.volume_per_7:
            return self  # A renamed/confirmed same-dose stage is not a new epoch.
        row = FrequencyStageTarget.freeze(stage_key, effective_on, volume_per_7, policy=self.policy)
        return replace(self, targets=(*self.targets, row))


def read_frequency_contract(value):
    if value is None:
        return None
    return FrequencyContract(datetime.fromisoformat(value['adopted_at']),
        FrequencyRangePolicy(**value['policy']),
        tuple(FrequencyStageTarget(**dict(r, effective_on=date.fromisoformat(r['effective_on'])))
              for r in value['targets']), FrequencyMidpointPreference(**value['preference']),
        value['version'])


@dataclass(frozen=True)
class FrequencyExposure:
    """Compiled once per decision; no fixed-block activation or count repayment.

    Past running fades linearly over policy.window_days. Future running is fully
    counted through the existing objective taper start, then linearly fades to
    the priced endpoint. Positive partial-tail weights never invent work/rest
    beyond that endpoint. Upper admission counts only future planned exposure;
    lower remains a guide, not a quota or catch-up debt.
    """
    decision: date
    end: date
    start: date
    weights: tuple[tuple[date, float], ...]
    observed: frozenset[date]
    lower: float
    upper: float
    future_lower: float
    future_upper: float
    anchor: date
    unweighted_future_upper: float = 0.
    count_admission: bool = False

    def assess(self, planned_dates):
        planned = set(planned_dates)
        if any(type(d) is not date or not self.decision <= d < self.end for d in planned):
            raise ValueError('Planned frequency dates must belong to the priced projection')
        weights = dict(self.weights)
        observed = self.observed & weights.keys()
        planned -= observed
        past_count = sum(weights[d] for d in observed)
        future_count = sum(weights[d] for d in planned)
        count = past_count+future_count
        admission_count = len(planned) if self.count_admission else future_count
        admission_upper = self.unweighted_future_upper if self.count_admission else self.future_upper
        upper_ok = admission_count <= admission_upper+1e-9
        within = self.future_lower-1e-9 <= future_count and upper_ok
        row = dict(start=self.start.isoformat(), end=self.end.isoformat(), count=count,
            weighted_days=sum(weights.values()), lower_expected_count=self.lower,
            upper_expected_count=self.upper, observed_count=len(observed), planned_count=len(planned),
            observed_weighted_count=past_count, planned_weighted_count=future_count,
            future_lower_guide=self.future_lower, future_upper_bound=self.future_upper,
            required_count_min=None, allowed_count_max=None, within_range=within,
            deficit=max(0., self.future_lower-future_count), excess=max(0., admission_count-admission_upper))
        if self.count_admission:
            row.update(admission_count=admission_count, admission_upper_bound=admission_upper,
                admission_basis='distinct_prospective_dates', weighted_future_upper_guide=self.future_upper,
                future_upper_bound=admission_upper)
        return dict(contract='prospective_count_v3' if self.count_admission else 'prospective_weighted_v2', status='above_future_upper_bound' if not upper_ok
            else 'within_range' if within else 'below_lower_guide', eligible=upper_ok,
            construction_eligible=upper_ok, blocks=[row], unassessed_from=self.end.isoformat(),
            coverage_end=self.end.isoformat(), anchor=self.anchor.isoformat(), changes_score=False,
            lower_is_advisory=True, assumed_history_coverage_complete=True)


def compile_frequency_exposure(contract, observed_dates, *, decision, coverage_end, taper_start):
    rows = contract.targets
    if (type(decision) is not date or type(coverage_end) is not date or type(taper_start) is not date
            or not decision < taper_start < coverage_end or rows[0].effective_on > decision
            or any(r.effective_on > decision for r in rows)):
        raise ValueError('Covered dated frequency history and ordered projection taper required')
    observed = frozenset(observed_dates)
    if any(type(d) is not date or d > decision for d in observed):
        raise ValueError('Available observed past dates required')
    start = max(rows[0].effective_on, decision-timedelta(days=contract.policy.window_days))
    weights = []
    lower = upper = future_lower = future_upper = unweighted_future_upper = 0.
    index = 0
    for offset in range((coverage_end-start).days):
        day = start+timedelta(days=offset)
        while index+1 < len(rows) and rows[index+1].effective_on <= day:
            index += 1
        weight = ((contract.policy.window_days-(decision-day).days)/contract.policy.window_days
            if day < decision else 1. if day < taper_start else
            (coverage_end-day).days/(coverage_end-taper_start).days)
        weights.append((day, weight))
        lo, hi = weight*rows[index].lower_per_7/7, weight*rows[index].upper_per_7/7
        lower += lo; upper += hi
        if day >= decision:
            future_lower += lo; future_upper += hi
            unweighted_future_upper += rows[index].upper_per_7/7
    return FrequencyExposure(decision, coverage_end, start, tuple(weights), observed,
        lower, upper, future_lower, future_upper, rows[0].effective_on,
        unweighted_future_upper, contract.version == 'frequency_midpoint_v3')
