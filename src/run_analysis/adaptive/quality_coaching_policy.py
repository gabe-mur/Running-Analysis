"""Offline coaching assignments, independent of the fatigue/exposure forecasts.

Choose a purpose, structure and ordinary commanded dose before calendar search.
The forecast prices those commands; it does not choose their training purpose.
These are explicit general-fitness coaching assumptions, not adaptation fits.
"""
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from hashlib import sha256
from functools import cached_property
import json
from math import fsum, gcd, isfinite, lcm

from .continuous_search_adapter import canonical
from .shared_program_evaluator import evaluate_program
from .quality_dose_policy import QualityDosePolicy
from run_analysis.workload_accounting import Quantity
from run_analysis.athlete_projection import aware

VERSION = 'quality_coaching_sequence_v1'
SCORING_VERSION = 'coached_shared_fulfillment_v2'
FAMILIES = {'sustained': ('threshold', 'progression'),
    'faster': ('short_intervals', 'long_intervals', 'fartlek', 'hill_repeats')}
# Authored representative commands, not distance conversions or fitted optima.
REPRESENTATIVES = {'threshold_continuous': None, 'threshold_repeats': 300.,
    'aerobic_timed': 180., 'aerobic_distance': 600.,
    'pickups_timed': 60., 'pickups_distance': 400., 'fartlek_surges': 60.,
    'hill_repeats': 45., 'progression': None}


def read_sequence(data):
    return QualityCoachingSequence(**dict(data,ordinary=tuple(CoachedRecipe(**r) for r in data['ordinary'])))


@dataclass(frozen=True)
class CoachedRecipe:
    key: str
    family: str
    purpose: str
    recipe: str
    requested_work_minutes: float | None
    accepted_work_minutes: float | None
    commanded_work_minutes: float
    miles: float
    dose_selection_basis: str


@dataclass(frozen=True)
class QualityCoachingSequence:
    stage_key: str
    issued_at: str
    ordinary: tuple[CoachedRecipe, ...]
    first_purpose: str = 'sustained'
    consumed: int = 0
    source: str = 'AS192 general-fitness variety prototype; not calibrated adaptation'
    cadence_contract: str = 'amplitude_v1'  # Absent archived field preserves original scoring.
    focus: str = 'variety'  # Omitted from default identities to preserve issued plans.

    def __post_init__(self):
        aware(datetime.fromisoformat(self.issued_at))
        from .timely_purpose_trial import CADENCE_CONTRACTS
        if (not self.stage_key or not self.issued_at or not self.source
                or self.cadence_contract not in CADENCE_CONTRACTS
                or self.focus not in {'variety', 'vo2max'}
                or self.first_purpose not in FAMILIES or type(self.consumed) is not int
                or self.consumed < 0 or not isinstance(self.ordinary, tuple)
                or len({r.key for r in self.ordinary}) != len(self.ordinary)
                or any(r.purpose not in FAMILIES or r.family not in FAMILIES[r.purpose]
                    or not isfinite(r.commanded_work_minutes) or r.commanded_work_minutes <= 0
                    or not isfinite(r.miles) or r.miles <= 0 for r in self.ordinary)
                or {r.purpose for r in self.ordinary} != set(FAMILIES)):
            raise ValueError('Frozen identified ordinary recipes for both coaching purposes required')

    @property
    def key(self):
        payload=asdict(self)
        if self.cadence_contract=='amplitude_v1':payload.pop('cadence_contract')
        if self.focus == 'variety': payload.pop('focus')
        return VERSION + ':' + sha256(json.dumps(payload, sort_keys=True,
            allow_nan=False).encode()).hexdigest()[:20]

    @cached_property
    def purpose_pattern(self):
        pattern = ('sustained', 'faster', 'faster') if self.focus == 'vo2max' else ('sustained', 'faster')
        offset = pattern.index(self.first_purpose)
        return pattern[offset:] + pattern[:offset]

    def family_slots(self, purpose):
        return self._family_slots[purpose]

    @cached_property
    def _family_slots(self):
        return {purpose:self._build_family_slots(purpose) for purpose in FAMILIES}

    def _build_family_slots(self, purpose):
        families = tuple(f for f in FAMILIES[purpose] if any(r.family == f for r in self.ordinary))
        if self.focus == 'vo2max' and purpose == 'faster' and 'long_intervals' in families:
            # Coaching emphasis, not a physiological benefit coefficient. Keep
            # enabled alternatives and never substitute a disabled family.
            return ('long_intervals', 'long_intervals', *(f for f in families if f != 'long_intervals'))
        return families

    @cached_property
    def _family_recipes(self):
        return {family:tuple(sorted((r for r in self.ordinary if r.family==family),key=lambda r:r.recipe))
            for families in FAMILIES.values() for family in families}

    def assignment(self, future_rank):
        if type(future_rank) is not int or future_rank < 0:
            raise ValueError('Nonnegative quality-opportunity rank required')
        rank = self.consumed + future_rank
        pattern = self.purpose_pattern
        cycles, position = divmod(rank, len(pattern))
        purpose = pattern[position]
        occurrence = cycles * pattern.count(purpose) + pattern[:position].count(purpose)
        families = self.family_slots(purpose)
        rounds, slot = divmod(occurrence, len(families))
        family = families[slot]
        recipes = self._family_recipes[family]
        recipe_rank = rounds * families.count(family) + families[:slot].count(family)
        return recipes[recipe_rank % len(recipes)]

    @property
    def cycle_length(self):
        periods = []
        for purpose in FAMILIES:
            slots = self.family_slots(purpose)
            groups = {f: [r for r in self.ordinary if r.family == f]
                for f in FAMILIES[purpose] if any(r.family == f for r in self.ordinary)}
            period = len(slots) * lcm(*(len(rs) // gcd(len(rs), slots.count(f)) for f, rs in groups.items()))
            count = self.purpose_pattern.count(purpose)
            periods.append(period // gcd(period, count))
        return len(self.purpose_pattern) * lcm(*periods)

    def bind(self, program, catalog):
        """Assign by quality order, never by weekday, fatigue score or horizon.

        Calendar edits may change the later assignments. The entire normalized
        physical program must be scored before selection; this is not post-hoc
        substitution. Advancing the clock does NOT consume an assignment.
        """
        result = []; rank = 0
        for day, key in canonical(program):
            if catalog[key].role == 'quality':
                key = self.assignment(rank).key
                rank += 1
            result.append((day, key))
        return tuple(result)

    def after_completion(self, completed_key):
        """Only an externally reconciled completion may advance the sequence.

        This checks assignment identity, not adherence or observed physiology.
        The upload adapter must establish those separately before calling it.
        """
        if completed_key != self.assignment(0).key:
            raise ValueError('Completion must match the next reconciled coaching assignment')
        return replace(self, consumed=self.consumed + 1)


def freeze_sequence(catalog, *, stage_key, issued_at, ordinary_miles,
                    quality_work_minutes, first_purpose='sustained', policy=QualityDosePolicy(),
                    cadence_contract='amplitude_v1', focus='variety'):
    """Command-based ordinary doses only; no forecast or score is an input.

    Keep representative bout length/unit, then choose the nearest admitted rep
    count. The physical stage ceiling is not equivalent HR exposure. When the
    authored goal cannot fit that ceiling, disclose the smaller offered dose.
    Reduced recipes are not the default calendar's escape hatch.
    """
    if any(not isfinite(v) or v <= 0 for v in (ordinary_miles, quality_work_minutes)):
        raise ValueError('Positive accepted physical stage doses required')
    scale = min(1., quality_work_minutes / policy.reference_work_minutes)
    groups = {}
    for key, dose in catalog.items():
        definition = dose.workout_definition
        if dose.role != 'quality' or not definition or not definition.get('recipe_key'):
            continue
        recipe, family = definition['recipe_key'], definition['family']
        if recipe not in REPRESENTATIVES:
            raise ValueError('Unreviewed coaching recipe: ' + recipe)
        groups.setdefault((family, recipe), []).append(dose)
    rows = []
    for (family, recipe), options in sorted(groups.items()):
        purpose = next(p for p, fs in FAMILIES.items() if family in fs)
        work = lambda d: fsum(s for kind, s in d.phases if kind == 'work') / 60
        if family == 'progression':
            requested = accepted = None
            chosen = min(options, key=lambda d: (abs(d.miles - ordinary_miles), d.key))
        else:
            requested = (18. if family == 'threshold' else policy.reference_work_minutes
                if family == 'long_intervals' else policy.faster_reference_minutes) * scale
            accepted = min(requested, quality_work_minutes)
            representative = ({'aerobic_timed': 240., 'aerobic_distance': 800.}.get(recipe,
                REPRESENTATIVES[recipe]) if focus == 'vo2max' else REPRESENTATIVES[recipe])
            def rank(d):
                phases = d.workout_definition['numeric_contract']['phases']
                primary = next(p['primary']['amount']['estimate'] for p in phases if p['kind'] == 'work')
                return (abs(primary - representative) if representative is not None else 0.,
                    abs(work(d) - accepted), work(d) > accepted, d.key)
            chosen = min(options, key=rank)
        rows.append(CoachedRecipe(chosen.key, family, purpose, recipe, requested, accepted,
            work(chosen), chosen.miles,
            'Frozen representative primary unit/bout, then nearest admitted ordinary dose; no fatigue ranking'))
    extra = dict(source='VO2max focus v1: aerobic-power emphasis; provisional coaching mix, not calibrated adaptation') if focus == 'vo2max' else {}
    return QualityCoachingSequence(stage_key, issued_at, tuple(rows), first_purpose,
        cadence_contract=cadence_contract, focus=focus, **extra)


def continuation_reference(sequence, physical, day):
    """Mean of a complete coaching cycle, not the cheapest quality forecast."""
    runs = [physical[day, sequence.assignment(i).key] for i in range(sequence.cycle_length)]
    def mean(values):
        return Quantity(fsum(v.low for v in values) / len(values),
            fsum(v.high for v in values) / len(values),
            fsum(v.estimate for v in values) / len(values) if all(v.estimate is not None for v in values) else None,
            'Conditional complete coaching-cycle mean; not confidence bounds')
    return ('quality', fsum(r.miles for r in runs) / len(runs),
        tuple(mean([r.demand[i] for r in runs]) for i in range(3)))


def evaluate_coached_program(context, candidate, *, sequence, quality_reference,
                            demand_interaction='independent_v1'):
    """Dose fulfillment and cadence are distinct from physiological exposure.

    Every ordinary assignment supplies one COACHING opportunity, not identical
    adaptation. Physical demands/volume remain exactly those of its prescription.
    Existing historical credit is retained, never manufactured from a role title.
    """
    rank = 0; assigned = []
    def scoped(rows):
        nonlocal rank
        result = []
        for row in rows:
            if row.role == 'quality':
                dose = row.key.split(':', 2)[2]
                expected = sequence.assignment(rank)
                if dose != expected.key:
                    raise ValueError('Calendar quality dose does not match its pre-scored coaching assignment')
                assigned.append(dict(rank=sequence.consumed + rank, day=row.day.isoformat(),
                    key=dose, purpose=expected.purpose, credit_basis='ordinary coaching-dose fulfillment'))
                rank += 1
                if row.quality_credit!=1.:
                    raise ValueError('Ordinary planned fulfillment must be supplied by the shared evidence adapter')
            result.append(row)
        return tuple(result)
    coached = replace(candidate, hard=scoped(candidate.hard), continuation=scoped(candidate.continuation))
    refs = tuple(quality_reference if r[0] == 'quality' else r
        for r in context.continuation_boundary_references)
    view = replace(context, quality_programme=None, continuation_boundary_references=refs)
    result = evaluate_program(view, coached,demand_interaction=demand_interaction,
        quality_cadence_contract=sequence.cadence_contract)
    identity = sha256(json.dumps(dict(evaluation=result.fingerprint, coaching=sequence.key),
        sort_keys=True).encode()).hexdigest()
    version='coached_proportional_cadence_v3' if sequence.cadence_contract!='amplitude_v1' else SCORING_VERSION
    return replace(result, fingerprint=identity, model_version=result.model_version + ':' + version,
        parts=dict(result.parts, quality_coaching=dict(contract_key=sequence.key, assignments=assigned,
            adaptation_equivalence_established=False, dose_choice_by_fatigue=False)),
        limitations=(*result.limitations,
            'Ordinary coaching-dose fulfillment is not measured adaptation; reduced-dose policy remains separate.'))
