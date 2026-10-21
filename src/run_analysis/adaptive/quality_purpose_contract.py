"""Unconnected Q3 purpose contract, not another scheduler or fatigue model.

Targets must be frozen before comparing alternatives. Reuse work_stimulus for
the actual exposure integral; no default HR/pace bands or family multipliers.
The two purpose ledgers share ONE quality-opportunity budget. Neither ledger
may compensate for a deficit in the other by producing more of its own work.
"""
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from hashlib import sha256
import json
from math import fsum, isfinite
import re

from run_analysis.athlete_projection import aware
from run_analysis.work_stimulus import CappedStimulusTarget, StimulusTarget, assess_work_stimulus
from run_analysis.workload_accounting import Quantity

VERSION = 'quality_purpose_contract_v1'
PURPOSES = ('sustained', 'faster')


@dataclass(frozen=True)
class PurposeRequirement:
    purpose: str
    target: StimulusTarget | CappedStimulusTarget
    opportunity_share: float
    minimum_bout_seconds: float
    source: str

    def __post_init__(self):
        expected = {'sustained': {'hrr'}, 'faster': {'speed_over_easy', 'speed_mps'}}
        if (self.purpose not in expected or self.target.channel not in expected[self.purpose]
                or not isfinite(self.opportunity_share) or not 0 < self.opportunity_share < 1
                or not isfinite(self.minimum_bout_seconds) or self.minimum_bout_seconds < 0
                or self.purpose == 'sustained' and self.minimum_bout_seconds <= 0
                or not self.source):
            raise ValueError('Explicit purpose, matching exposure channel, share and structure required')


@dataclass(frozen=True)
class QualityPurposeContract:
    issued_at: datetime
    stage_key: str
    cadence_days: float
    requirements: tuple[PurposeRequirement, ...]
    source: str

    def __post_init__(self):
        aware(self.issued_at)
        object.__setattr__(self, 'requirements', tuple(self.requirements))
        if (not self.stage_key or not self.source or not isfinite(self.cadence_days)
                or self.cadence_days <= 0 or len(self.requirements) != 2
                or {r.purpose for r in self.requirements} != set(PURPOSES)
                or abs(fsum(r.opportunity_share for r in self.requirements)-1) > 1e-12):
            raise ValueError('Two distinct purposes must share one existing quality cadence')

    @property
    def key(self):
        payload = json.dumps(asdict(self), sort_keys=True, default=lambda v: v.isoformat(), allow_nan=False)
        return VERSION+':'+sha256(payload.encode()).hexdigest()[:20]


def _planned_windows(prescription, purpose, minimum):
    """Use frozen phase intent, not its family/title, to authorize planned credit.

    This is a coaching-intent check, NOT an inference of a physiological HR
    threshold. Numeric exposure must still meet the SAME independently frozen
    target. An easy-to-Z3 progression cannot be asserted to supply faster work.
    """
    offset, windows, unresolved = 0., [], []
    for phase in prescription.phases:
        seconds = phase.seconds.estimate
        cues = ' '.join((*phase.effort.start_cues, *phase.effort.finish_cues))
        zones = {int(z) for z in re.findall(r'\bZ([1-5])\b', cues)}
        requested = bool(zones & ({3, 4, 5} if purpose == 'sustained' else {4, 5}))
        if phase.kind == 'work' and phase.effort.purpose != 'easy' and seconds >= minimum:
            if requested:
                windows.append((offset, offset+seconds))
            elif not zones:
                unresolved.append((offset, offset+seconds))
        offset += seconds
    return tuple(windows), tuple(unresolved)


def assess_quality_purposes(evidence, *, contract, role, as_of, prescription=None):
    """Same target integrator for observations/forecasts; identity checks first.

    Observed windows must come from existing supported physical work, not from
    an assigned title. Callers retain the detector's provenance in WorkSignals.
    Planned windows must exactly match the supplied immutable prescription.
    Unknown exposure remains bounded; it is neither proof nor an automatic ban.
    """
    aware(as_of)
    if contract.issued_at > as_of or evidence.available_at > as_of:
        raise ValueError('Purpose contract or evidence is not yet available')
    if role not in {'easy', 'long', 'quality'}:
        raise ValueError('Resolved run role required')
    if evidence.source == 'planned':
        if (prescription is None or not contract.issued_at <= prescription.request.issued_at <= evidence.available_at
                or prescription.key != evidence.key):
            raise ValueError('Planned evidence needs its exact previously issued prescription')
        offset, expected = 0., []
        for phase in prescription.phases:
            end = offset+phase.seconds.estimate
            if phase.kind == 'work':
                expected.append((offset, end))
            offset = end
        elapsed = (evidence.end-evidence.start).total_seconds()
        if (abs(elapsed-offset) > 1e-6 or len(expected) != len(evidence.windows)
                or any(abs(a-c) > 1e-6 or abs(b-d) > 1e-6
                       for (a, b), (c, d) in zip(expected, evidence.windows))):
            raise ValueError('Planned signals must align with exact compiled work phases')
    elif prescription is not None:
        raise ValueError('Observed exposure must not inherit assigned intent as physical evidence')
    reports = {}
    for requirement in contract.requirements:
        windows, unresolved = evidence.windows, ()
        if evidence.source == 'planned':
            windows, unresolved = _planned_windows(prescription, requirement.purpose, requirement.minimum_bout_seconds)
        else:
            # Do not join distinct reps through recovery or multiply tiny
            # repetitions until they appear to be one sustained work bout.
            windows = tuple((a, b) for a, b in windows if b-a >= requirement.minimum_bout_seconds)
        if role != 'quality':
            windows, unresolved = (), ()
        report = assess_work_stimulus(replace(evidence, windows=windows), requirement.target)
        if unresolved:
            possible = assess_work_stimulus(replace(evidence, windows=tuple(sorted((*windows, *unresolved)))),
                requirement.target)
            report['coverage'] = Quantity(report['coverage'].low, possible['coverage'].high,
                basis='unresolved_requested_purpose_with_exposure_bounds')
            report['possible_exposure_with_unresolved_intent'] = possible
        if evidence.basis == 'unknown' and role == 'quality':
            elapsed = (evidence.end-evidence.start).total_seconds()
            upper = min(1., elapsed/60/requirement.target.minutes) if elapsed >= requirement.minimum_bout_seconds else 0.
            report['coverage'] = Quantity(0., upper, basis='unresolved_physical_work_alignment')
            report['status'] = 'unresolved_physical_work_alignment'
        report['unresolved_intent_windows'] = unresolved
        reports[requirement.purpose] = report
    credits = {name: r['coverage'] for name, r in reports.items()}
    # Opportunity accounting ONLY, not summed physiological benefit. A session
    # supporting both purposes is still at most one quality opportunity.
    opportunity = Quantity(max(q.low for q in credits.values()), max(q.high for q in credits.values()),
        basis='one_shared_quality_opportunity_not_adaptation')
    payload = dict(contract_key=contract.key, evidence=asdict(evidence), role=role, as_of=as_of,
        prescription_key=prescription.key if prescription else None)
    fingerprint = sha256(json.dumps(payload, sort_keys=True, default=str, allow_nan=False).encode()).hexdigest()
    return dict(version=VERSION, key=evidence.key, contract_key=contract.key, role=role,
        status='unresolved_physical_work_alignment' if evidence.basis == 'unknown' else 'assessed_conditionally',
        reports=reports, credits=credits, opportunity=opportunity, fingerprint=fingerprint,
        physiological_equivalence_established=False, planner_influence=False)


def assess_purpose_balance(assessments, *, contract, quality_opportunities):
    """Audit supplied delivery against one shared budget, not a window scheduler.

    The existing cadence/continuation evaluator owns elapsed-time demand. This
    function consumes that demand; it must not invent a Sunday-week or reset it
    with the horizon. Independent deficits stay independent, with no genre bonus
    and no second copy of the generic quality-minute reward.
    """
    assessments = tuple(assessments)
    if (not isfinite(quality_opportunities) or quality_opportunities < 0
            or len({a['key'] for a in assessments}) != len(assessments)
            or any(a['contract_key'] != contract.key for a in assessments)):
        raise ValueError('Unique evidence under one frozen contract and nonnegative demand required')
    def total(values):
        return Quantity(fsum(q.low for q in values), fsum(q.high for q in values), basis='purpose_delivery_bounds')
    delivery, deficits, targets = {}, {}, {}
    for requirement in contract.requirements:
        name = requirement.purpose
        target = quality_opportunities*requirement.opportunity_share
        delivered = total([a['credits'][name] for a in assessments])
        targets[name], delivery[name] = target, delivered
        deficits[name] = Quantity(max(0., target-delivered.high), max(0., target-delivered.low),
            basis='noncompensating_purpose_deficit')
    delivered_opportunities = total([a['opportunity'] for a in assessments])
    opportunity_deficit = Quantity(max(0., quality_opportunities-delivered_opportunities.high),
        max(0., quality_opportunities-delivered_opportunities.low), basis='shared_quality_opportunity_deficit')
    purposes_supported = all(q.high == 0 for q in deficits.values())
    return dict(contract_key=contract.key, targets=targets, delivery=delivery, deficits=deficits,
        quality_opportunities=quality_opportunities,
        delivered_opportunities=delivered_opportunities, opportunity_deficit=opportunity_deficit,
        all_purposes_supported=purposes_supported,
        all_requirements_supported=purposes_supported and opportunity_deficit.high == 0,
        planner_influence=False)
