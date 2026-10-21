"""Offline native quality recipes and exhaustive variation within declared grids.

No new scheduler, progression rate, purpose-credit coefficient or effort-to-pace
fit. Callers supply causal per-purpose pace scenarios and dose windows. These are
generation bounds, not physiological admission or calendar acceptance. Existing
issued v2/v3 commands and the live menu are untouched. Strides are out of scope.
"""
from dataclasses import dataclass, replace
from math import ceil, floor, fsum, isfinite

from run_analysis.staged_progression import MILE, tenth
from run_analysis.workload_accounting import Quantity
from .quality_block_contract import BlockCommand, BlockRequest, compile_blocks
from .quality_prescription_contract import Effort, Measure, PaceReference, compile_numeric_phase

VERSION = 'quality_recipe_catalogue_v1'
SOURCE = 'AS182 provisional native recipe grid; not calibrated physiology'


@dataclass(frozen=True)
class DoseWindow:
    """Caller-declared NOMINAL generation bounds, not uncertainty/safety limits."""
    minimum_work_seconds: float
    maximum_work_seconds: float
    maximum_session_miles: float
    maximum_session_seconds: float
    maximum_repetitions: int
    source: str

    def __post_init__(self):
        if (any(isinstance(v, bool) or not isfinite(v) or v <= 0 for v in
                (self.minimum_work_seconds, self.maximum_work_seconds,
                 self.maximum_session_miles, self.maximum_session_seconds))
                or self.minimum_work_seconds > self.maximum_work_seconds
                or type(self.maximum_repetitions) is not int or self.maximum_repetitions <= 0
                or not isinstance(self.source, str) or not self.source.strip()):
            raise ValueError('Explicit positive ordered generation bounds and provenance required')


@dataclass(frozen=True)
class Recipe:
    key: str
    purpose: str
    base: BlockRequest
    work_targets: tuple[int, ...]
    minimum_repetitions: int
    continuous: bool = False
    final_recovery: bool = False
    bout_seconds: tuple[float, float] | None = None
    work_target_range: tuple[int, int] | None = None

    def __post_init__(self):
        kinds = tuple(b.kind for b in self.base.blocks)
        expected = ('warmup', 'work', 'cooldown') if self.continuous else (
            'warmup', 'work', 'recovery', 'cooldown')
        if (not self.key or self.purpose not in {'sustained', 'faster'} or kinds != expected
                or not isinstance(self.work_targets, tuple) or not self.work_targets
                or any(type(v) is not int or v <= 0 for v in self.work_targets)
                or tuple(sorted(set(self.work_targets))) != self.work_targets
                or type(self.minimum_repetitions) is not int or self.minimum_repetitions <= 0
                or type(self.continuous) is not bool or type(self.final_recovery) is not bool
                or self.continuous and (self.minimum_repetitions != 1 or self.final_recovery)
                or self.bout_seconds is not None and (len(self.bout_seconds) != 2
                    or not all(isfinite(s) for s in self.bout_seconds)
                    or not 0 < self.bout_seconds[0] <= self.bout_seconds[1])
                or self.work_target_range is not None and (len(self.work_target_range) != 2
                    or any(type(v) is not int for v in self.work_target_range)
                    or not 0 < self.work_target_range[0] <= self.work_targets[0]
                        <= self.work_targets[-1] <= self.work_target_range[1]
                    or len(self.work_targets) != 1)
                or self.base.blocks[1].primary.unit not in {'time', 'distance'}):
            raise ValueError('Explicit native recipe grid and ordered prototype blocks required')


@dataclass(frozen=True)
class Candidate:
    recipe_key: str
    purpose: str
    prescription: object
    work_seconds: Quantity
    repetitions: int
    generation_source: str


def reviewed_recipes(*, issued_at, easy_pace, threshold_pace, aerobic_pace,
                     fast_pace, open_recovery_seconds, hill_pace=None):
    """Provisional coaching structures, not fitted physiological thresholds.

    Threshold repeats: 5–10 min / 1 min jog. Hard aerobic repeats: 2–4 min /
    2 min jog. Published coaching/review ranges motivate these subsets, not an
    assertion that these exact grids/recoveries are optimal for all athletes.
    Distance variants are native alternatives, not conversions of timed recipes.
    HR cues remain authored settings-relative proxies, not identified LT/VO2.
    """
    paces = (easy_pace, threshold_pace, aerobic_pace, fast_pace)
    if any(not isinstance(p, PaceReference) for p in paces):
        raise ValueError('Distinct explicit per-purpose pace scenarios required')
    if not (easy_pace.seconds_per_meter.estimate > threshold_pace.seconds_per_meter.estimate
            > aerobic_pace.seconds_per_meter.estimate
            and fast_pace.seconds_per_meter.estimate < easy_pace.seconds_per_meter.estimate):
        raise ValueError('Nominal harder aerobic pace must differ from threshold and easy')
    if type(open_recovery_seconds) is not int or open_recovery_seconds <= 0:
        raise ValueError('Positive explicit open-recovery timing scenario required')
    if hill_pace is not None and not isinstance(hill_pace, PaceReference):
        raise ValueError('Explicit terrain-specific hill extent scenario required')
    easy = Effort(('Z1', 'Z2'), ('Z1', 'Z2'), 'easy')
    def block(kind, amount, pace, cues, instruction, *, unit='time', opened=False):
        effort = Effort(cues, cues, 'purposeful') if kind == 'work' else easy
        return BlockCommand(kind, Measure(unit, Quantity.point(amount, SOURCE)), effort,
            pace, instruction, open_recovery=opened)
    warm = block('warmup', 720, easy_pace, (), 'Warm up easily.')
    cool = block('cooldown', 600, easy_pace, (), 'Cool down easily.')
    specs = (
        # key, family, purpose, title, unit, grid, pace, cues, instruction, rest, min reps, continuous, final, open
        ('threshold_continuous', 'threshold', 'sustained', 'Threshold run', 'time',
         tuple(range(360, 1801, 30)), threshold_pace, ('upper Z3', 'low Z4'),
         'At comfortably hard threshold effort; start conservatively.', 0, 1, True, False, False),
        ('threshold_repeats', 'threshold', 'sustained', 'Threshold repeats', 'time',
         tuple(range(300, 601, 30)), threshold_pace, ('upper Z3', 'low Z4'),
         'At comfortably hard threshold effort; keep the effort even.', 60, 2, False, False, False),
        ('aerobic_timed', 'long_intervals', 'faster', 'Aerobic intervals', 'time',
         tuple(range(120, 241, 30)), aerobic_pace, ('upper Z4 effort',),
         'At hard aerobic effort, not sprinting; do not chase HR at the start.', 120, 2, False, False, False),
        ('aerobic_distance', 'long_intervals', 'faster', 'Aerobic intervals', 'distance',
         (600, 800, 1000), aerobic_pace, ('upper Z4 effort',),
         'At hard aerobic effort, not sprinting; keep repetitions even.', 120, 2, False, False, False),
        ('pickups_timed', 'short_intervals', 'faster', 'Pickups', 'time',
         (30, 45, 60, 90), fast_pace, ('Z4 effort',),
         'At smooth fast effort, not sprinting; do not chase HR lag.', 90, 2, False, True, False),
        ('pickups_distance', 'short_intervals', 'faster', 'Intervals', 'distance',
         (200, 400, 500), fast_pace, ('Z4 effort',),
         'At smooth fast effort, not sprinting; finish in control.', 90, 2, False, True, False),
        ('fartlek_surges', 'fartlek', 'faster', 'Fartlek', 'time',
         (60,), fast_pace, ('Z3', 'Z4 effort'),
         'Use landmarks for a strong surge; do not force identical repetitions.', open_recovery_seconds, 2, False, True, True),
        ('hill_repeats', 'hill_repeats', 'faster', 'Hill repeats', 'time',
         (30, 45, 60, 90), hill_pace, ('Z4 effort',),
         'Run uphill at strong effort; stop if form deteriorates.', open_recovery_seconds, 2, False, False, True),
    )
    result = []
    for key, family, purpose, title, unit, grid, pace, cues, text, rest, reps, continuous, final, opened in specs:
        if pace is None:
            continue  # No flat-speed conversion masquerading as an uphill extent.
        w = block('work', grid[0], pace, cues, text, unit=unit)
        r = block('recovery', rest, easy_pace, (),
            'Jog gently downhill until breathing and form are composed.' if family == 'hill_repeats'
            else 'Jog easily until breathing is composed.' if opened else 'Jog easily',
            opened=opened) if rest else None
        blocks = (warm, w, cool) if continuous else (warm, w, r, cool)
        if family == 'hill_repeats':
            blocks = (replace(warm, instruction='Warm up easily on flat terrain.'), *blocks[1:])
        base = BlockRequest(family, key, title, 'tempo_threshold' if family == 'threshold' else 'intervals',
            issued_at, SOURCE, blocks)
        duration_bounds = (120., 240.) if key == 'aerobic_distance' else (
            (30., 180.) if key == 'pickups_distance' else None)
        result.append(Recipe(key, purpose, base, grid, reps, continuous, final, duration_bounds,
            (45, 90) if family == 'fartlek' else None))
    return tuple(result)


def compile_recipe_options(recipe, window):
    """All valid count/target combinations in the DECLARED grid, no score pruning.

    Algebraic count/session bounds avoid compiling impossible combinations.
    Bounds use nominal supplied extent scenarios; all uncertainty stays in the
    compiled command for later forecast/admission. Absence of options is explicit.
    """
    if not isinstance(recipe, Recipe) or not isinstance(window, DoseWindow):
        raise ValueError('Typed recipe and caller-owned dose window required')
    base = recipe.base
    warm, work, cool = base.blocks[0], base.blocks[1], base.blocks[-1]
    recovery = None if recipe.continuous else base.blocks[2]
    source = SOURCE+';'+window.source
    rows = []
    for target in recipe.work_targets:
        seconds = target if work.primary.unit == 'time' else target*work.pace.seconds_per_meter.estimate
        if recipe.bout_seconds is not None and not recipe.bout_seconds[0] <= seconds <= recipe.bout_seconds[1]:
            continue
        first = max(recipe.minimum_repetitions, ceil(window.minimum_work_seconds/seconds-1e-12))
        last = min(window.maximum_repetitions, floor(window.maximum_work_seconds/seconds+1e-12))
        if recipe.continuous:
            first, last = max(first, 1), min(last, 1)
        amount = Quantity(*recipe.work_target_range, target, source) if recipe.work_target_range else Quantity.point(target, source)
        w = replace(work, primary=Measure(work.primary.unit, amount))
        for count in range(first, last+1):
            rests = count if recipe.final_recovery else count-1
            blocks = [warm]
            for i in range(count):
                blocks.append(w)
                if recovery is not None and i < rests:
                    blocks.append(recovery)
            blocks.append(cool)
            # Shared compiler owns ALL physical extents; never use a work-only
            # distance or omit the final jog from the generation window.
            phases = tuple(compile_numeric_phase(b.kind, b.primary, b.pace, b.effort,
                open_recovery=b.open_recovery) for b in blocks)
            if (fsum(p.seconds.estimate for p in phases) > window.maximum_session_seconds+1e-9
                    or fsum(p.meters.estimate for p in phases)/MILE > window.maximum_session_miles+1e-9):
                continue
            variant = f'{recipe.key}_{count}x{target}'+('s' if work.primary.unit == 'time' else 'm')
            p = compile_blocks(replace(base, variant=variant, source=source, blocks=tuple(blocks)))
            amounts = [p.seconds for p in p.phases if p.kind == 'work']
            extent = Quantity(fsum(q.low for q in amounts), fsum(q.high for q in amounts),
                fsum(q.estimate for q in amounts), source+';native work extent scenario')
            rows.append(Candidate(recipe.key, recipe.purpose, p, extent, count, window.source))
    return tuple(rows)


def compile_catalogue(recipes, windows):
    recipes = tuple(recipes)
    if len({r.key for r in recipes}) != len(recipes):
        raise ValueError('Duplicate recipe key')
    if set(windows) != {r.purpose for r in recipes}:
        raise ValueError('Supply a generation window for every requested purpose, without unused windows')
    options = tuple((r, compile_recipe_options(r, windows[r.purpose])) for r in recipes)
    return dict(version=VERSION, candidates=tuple(c for _, rows in options for c in rows), planner_influence=False,
        empty_recipes=tuple(r.key for r, rows in options if not rows),
        purpose_credit_assigned=False, exhaustive_scope='Declared target/count grids only; not global optimality')


def progression_options(base, totals, *, window):
    """Vary an explicit portion/profile prescription, without guessing its ramp."""
    if (base.request.family != 'progression' or base.request.total_miles is None
            or any(b.primary.unit != 'portion' for b in base.request.blocks)
            or not isinstance(window, DoseWindow)):
        raise ValueError('Frozen portion/profile progression and declared window required')
    totals = tuple(totals)
    if any(not isfinite(m) or m <= 0 or tenth(m) != m for m in totals):
        raise ValueError('Commit positive tenth-mile progression targets')
    result = []
    for miles in sorted(set(totals)):
        p = compile_blocks(replace(base.request, total_miles=miles,
            variant=f'progression_{miles:.1f}miles', source=SOURCE+';'+window.source))
        work = [p.seconds for p in p.phases if p.kind == 'work']
        seconds = fsum(q.estimate for q in work)
        if (window.minimum_work_seconds <= seconds <= window.maximum_work_seconds
                and miles <= window.maximum_session_miles
                and fsum(q.seconds.estimate for q in p.phases) <= window.maximum_session_seconds):
            result.append(Candidate('progression', 'gradual', p,
                Quantity(fsum(q.low for q in work), fsum(q.high for q in work), seconds,
                    window.source+';explicit portion/profile work extent'), 1, window.source))
    return tuple(result)
