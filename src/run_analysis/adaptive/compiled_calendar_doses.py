"""Exact catalogue score compilation for an already fixed physical calendar.

No new physiology, weights, affine HR assumption, or feasibility shortcut.
Demand uses each catalogue dose's actual frozen quantities. Candidate costs
are verified against the full scorer; every adopted edit still needs admission.
The finite neighbourhood is wider than the six-day pair search, not global.
"""
from dataclasses import dataclass
from datetime import timedelta
from itertools import combinations, permutations
from math import exp, expm1, isfinite
from time import perf_counter

import numpy as np

from run_analysis.program_delivery import _target_total
from .program_objective_trial import demand_value
from .soft_horizon_trial import day_weight


def _continuation_arrays(context, references, *, include_daily_targets=False):
    from .race_event import training_fraction,dated_purpose_fraction
    """Same dated mean continuation and convolution as the authoritative scorer."""
    policy = context.policy
    window, smoothing = policy.volume_window_days, context.volume_smoothing_days
    start = context.decision - timedelta(days=window + smoothing - 2)
    stop = context.end + timedelta(days=window + smoothing - 2)
    race = context.race_goal
    if race and race.goal:
        stop = max(stop, race.race_date + timedelta(days=race.return_days + window + smoothing))
    dates = tuple(start + timedelta(days=i) for i in range((stop-start).days))
    support = context.easy_duration_policy.supporting_easy_miles_per_7
    targets = []
    for segments, rate in ((context.volume_targets, policy.weekly_miles), (context.support_targets, support)):
        if not segments:
            values = np.full(len(dates), rate/7)
        else:
            final = segments[-1].miles_per_7_days
            base = final
            if race and race.goal:
                supported = next((s for s in reversed(segments)
                    if training_fraction(race,context.race_event,s.end-timedelta(days=1)) > 0), None)
                if supported is not None:
                    base = supported.miles_per_7_days/training_fraction(race,context.race_event,supported.end-timedelta(days=1))
            values = np.array([_target_total(segments, d, d+timedelta(days=1))
                if d < segments[-1].end else max(.1, base*training_fraction(race,context.race_event,d))/7
                if race and race.goal else final/7 for d in dates])
        targets.append(values)
    nominal = lambda q: np.array([demand_value(v, context.scenario.bound_fraction) for v in q])/context.scales
    em, eq = references['easy']; lm, lq = references['long']; qm, qq = references['quality']
    purpose = np.array([dated_purpose_fraction(context,d,'long') for d in dates]) if race or context.race_goal_history else np.ones(len(dates))
    key_miles = (lm*policy.long_per_7 + qm*policy.quality_per_7)/7*purpose
    easy = np.maximum(0., targets[0]-key_miles) if policy.purpose_weights[1] == 0 else np.maximum(targets[1], targets[0]-key_miles)
    mean = easy[:, None]*nominal(eq)/em + purpose[:, None]*(nominal(lq)*policy.long_per_7 + nominal(qq)*policy.quality_per_7)/7
    if race and race.goal:
        mean[np.array([d == race.race_date for d in dates])] = 0.
    kernel = np.convolve(np.ones(window), np.ones(smoothing)/smoothing)
    target_windows = np.array([np.convolve(v, kernel, mode='valid') for v in targets])
    result=(start, dates, kernel, target_windows, np.array([easy+key_miles, easy]), mean)
    return (*result,targets) if include_daily_targets else result


@dataclass
class CompiledCalendar:
    context: object
    program: tuple
    days: tuple
    keys: tuple
    miles: np.ndarray
    running: np.ndarray
    quantities: np.ndarray
    quadratic: np.ndarray
    linear: np.ndarray
    deficit_base: np.ndarray
    deficit_matrix: np.ndarray
    squared_weights: np.ndarray
    linear_weights: np.ndarray
    short_weights: np.ndarray
    original_levels: np.ndarray
    constant: float
    credit: np.ndarray
    required_credit: float
    scored_rows: int = 0

    def cost(self, levels):
        """One committed catalogue level per date, or a batch of such vectors."""
        choices = np.asarray(levels, dtype=int)
        single = choices.ndim == 1
        if single:
            choices = choices[None, :]
        self.scored_rows += len(choices)
        q = self.quantities[choices]
        running = self.running[choices]
        miles = self.miles[choices]
        result = np.full(len(choices), self.constant)
        result += np.einsum('bia,ij,bja->b', q, self.quadratic, q, optimize=True)
        result += np.einsum('bia,ia->b', q, self.linear)
        deficits = np.maximum(0., self.deficit_base[None, :] - miles @ self.deficit_matrix.T)
        result += (deficits*deficits) @ self.squared_weights + deficits @ self.linear_weights
        p = self.context.easy_duration_policy
        fractions = np.clip((p.penalty_free_minutes-running)/(p.penalty_free_minutes-p.minimum_minutes), 0., 1.)
        result += (p.maximum_penalty*fractions*fractions) @ self.short_weights
        return float(result[0]) if single else result

    def physical(self, levels):
        replacements = dict(zip(self.days, (self.keys[int(i)] for i in levels)))
        return tuple((d, replacements.get(d, k)) for d, k in self.program)

    def funded(self, levels):
        choices = np.asarray(levels, dtype=int)
        return np.sum(self.credit[np.arange(len(self.days)), choices], axis=-1) + 1e-8 >= self.required_credit


def compile_calendar(state, program, *, physical, result,day_weight_function=None,handoff_probabilities=None):
    """Compile the current sustained distributed-handoff scoring contract only.

    Unsupported contracts fail closed. Constant-purpose/frequency contributions
    are anchored once to the full score, never recalibrated for alternatives.
    This is a catalogue lookup model, not a smooth fitted fatigue approximation.
    """
    context = state.context
    if (day_weight_function is None and handoff_probabilities is None
            and context.continuation_distribution!='linear_v1'):
        day_weight_function=lambda day:day_weight(context,day)
        handoff_probabilities=context.handoff_probabilities
    weight = (lambda day: day_weight(context,day)) if day_weight_function is None else day_weight_function
    if (day_weight_function is None)!=(handoff_probabilities is None):
        raise ValueError('Alternate pricing requires BOTH explicit survival and handoff probabilities')
    if (not context.continuation_boundary_references or not context.soft_horizon_taper
            or context.delivery_contract != 'sustained_v1' or context.downstream_end is not None
            or context.steady_continuation_references):
        raise ValueError('Compiler requires the sustained distributed-continuation contract')
    program = tuple(sorted(program))
    evaluation = result['evaluation']
    ledger = evaluation.conditional_ledger
    easy = tuple((d, k) for d, k in program if physical[(d, k)].role == 'easy')
    if not easy:
        raise ValueError('Fixed calendar has no easy-dose coordinates')
    days = tuple(d for d, _ in easy)
    keys = tuple(sorted((k for d, k in physical if d == days[0] and physical[(d,k)].role == 'easy'),
        key=lambda k: (physical[(days[0],k)].miles, k)))
    if len({physical[(days[0],k)].miles for k in keys}) != len(keys):
        raise ValueError('Ambiguous same-distance easy catalogue requires separate handling')
    nominal = lambda q: np.array([demand_value(v, context.scenario.bound_fraction) for v in q])/context.scales
    quantities = np.array([nominal(physical[(days[0],k)].demand) for k in keys])
    running = np.array([demand_value(physical[(days[0],k)].demand[0], context.scenario.bound_fraction) for k in keys])
    miles = np.array([physical[(days[0],k)].miles for k in keys])
    for day in days:
        for i,k in enumerate(keys):
            row = physical[(day,k)]
            if row.miles != miles[i] or not np.allclose(nominal(row.demand), quantities[i], atol=1e-13, rtol=0):
                raise ValueError('Date-conditioned easy forecasts need their own lookup table')
    positions = {day: i for i, day in enumerate(days)}
    selected = {d: keys.index(k) for d, k in easy}
    original = np.array([selected[d] for d in days], dtype=int)
    n = len(days); span = (context.end-context.stage_start).days
    axis = np.array(context.scenario.axis_weights)/sum(context.scenario.axis_weights)
    # Store sqrt(axis weights) in catalogue quantities so the shared quadratic
    # kernel can serve all axes without assuming equal load scales.
    weighted_q = quantities*np.sqrt(axis)
    quadratic = np.zeros((n,n)); linear = np.zeros((n,3))
    factor = context.exchange*7/span
    memory = context.scenario.memory_days
    compression = context.scenario.timing_compression_days
    variable = {row.day: positions[row.day.isoformat()] for row in ledger if row.source == 'planned' and row.day.isoformat() in positions}
    for row in ledger:
        if row.day not in variable:
            continue
        i = variable[row.day]; w = weight(row.day)
        quadratic[i,i] = .5*w*factor
        for other in ledger:
            if other.day == row.day:
                continue
            age = abs((row.day-other.day).days)-compression
            later_weight = w if other.source == 'observed' or other.day < row.day else weight(other.day)
            coefficient = factor*later_weight*exp(-age/memory)
            if other.day in variable:
                quadratic[i,variable[other.day]] = .5*coefficient
            else:
                linear[i] += coefficient*nominal(other.demand)*np.sqrt(axis)
    references = {r: (m,q) for r,m,q in context.continuation_boundary_references}
    start, dates, kernel, targets, means, mean_demand, daily_targets = _continuation_arrays(context,references,include_daily_targets=True)
    from .foregone_volume import relief_windows
    relief=relief_windows(context,dates,kernel,daily_targets,ledger)
    handoffs = tuple(context.taper_start+timedelta(days=i) for i in range(1,(context.end-context.taper_start).days+1))
    probabilities = None if handoff_probabilities is None else tuple(handoff_probabilities)
    if probabilities is not None and (len(probabilities)!=len(handoffs)
            or any(not isfinite(p) or p<=0 for p in probabilities) or abs(sum(probabilities)-1.)>1e-12):
        raise ValueError('Positive normalized probabilities for every handoff required')
    bases, matrices, squared, linears = [], [], [], []
    for number,h in enumerate(handoffs):
        probability = None if probabilities is None else probabilities[number]
        prefix = tuple(row for row in ledger if row.day < h)
        amount = np.zeros((2,len(dates)))
        influence = np.zeros((len(dates),n))
        for row in prefix:
            index = (row.day-start).days
            if not 0 <= index < len(dates):
                continue
            if row.day in variable:
                influence[index,variable[row.day]] = 1.
            else:
                if row.role != 'race':
                    amount[0,index] += row.miles
                if row.role == 'easy': amount[1,index] += row.miles
        boundary = (h-start).days
        amount[:,boundary:] = means[:,boundary:]
        count = (h+timedelta(days=len(kernel)-1)-context.decision).days
        balances = np.array([np.convolve(v,kernel,mode='valid') for v in amount])[:, :count]
        design = np.array([np.convolve(influence[:,i],kernel,mode='valid')[:count] for i in range(n)]).T
        for channel in range(2):
            bases.append(1-(balances[channel]+relief[channel,:count])/targets[channel,:count])
            matrices.append(design/targets[channel,:count,None])
            coefficient = context.policy.purpose_weights[channel]/span
            coefficient = coefficient/len(handoffs) if probability is None else coefficient*probability
            squared.extend([coefficient]*count)
            linears.extend([coefficient*context.volume_linear_coefficient if channel == 0 else 0.]*count)
        count_future = len(dates)-boundary
        future = np.exp(-np.arange(count_future)/memory) @ mean_demand[boundary:]
        future += mean_demand[-1]*exp(-count_future/memory)/(-expm1(-1/memory))
        for row in prefix:
            if row.day in variable:
                coefficient = factor/len(handoffs) if probability is None else factor*probability
                linear[variable[row.day]] += coefficient*future*np.sqrt(axis)*exp(-((h-row.day).days-compression)/memory)
    if context.race_demand and context.race_goal.race_date >= context.decision:
        event = nominal(context.race_demand)*np.sqrt(axis)
        for row in ledger:
            if row.day in variable:
                linear[variable[row.day]] += factor*weight(row.day)*event*exp(-(abs((context.race_goal.race_date-row.day).days)-compression)/memory)
    short_weights = np.array([weight(row.day)*7/span
        for d in days for row in ledger if row.source == 'planned' and row.day.isoformat() == d])
    credit = np.zeros((n,len(keys))); required = 0.
    review = state.prospective_volume_review
    if review is not None:
        required = review.required_miles-sum(v for _,v in review.completed_credit)
        for d,k in program:
            row = physical[(d,k)]
            if d not in positions:
                begin = max(row.start,review.available_at); end = begin+(row.end-row.start)
                if review.start <= begin < review.end and end <= review.end: required -= row.miles
        for i,d in enumerate(days):
            for j,k in enumerate(keys):
                row = physical[(d,k)]
                begin = max(row.start,review.available_at); end = begin+(row.end-row.start)
                if review.start <= begin < review.end and end <= review.end: credit[i,j] = row.miles
    compiled = CompiledCalendar(context,program,days,keys,miles,running,weighted_q,
        quadratic,linear,np.concatenate(bases),np.concatenate(matrices),np.array(squared),
        np.array(linears),short_weights,original,0.,credit,required)
    compiled.constant = float(evaluation.total_cost)-compiled.cost(original)
    return compiled


def neighbors(compiled, levels, *, triples=True):
    """All catalogue singles, all-span exact-volume pairs, unit three-way moves."""
    levels = np.asarray(levels, dtype=int)
    units = np.rint(compiled.miles*10).astype(int)
    lookup = {v: i for i,v in enumerate(units)}
    for i in range(len(levels)):
        for j in range(len(units)):
            if j != levels[i]:
                p = levels.copy(); p[i] = j; yield p
    for a,b in combinations(range(len(levels)),2):
        total = units[levels[a]]+units[levels[b]]
        for u,j in lookup.items():
            if j != levels[a] and total-u in lookup:
                p = levels.copy(); p[a] = j; p[b] = lookup[total-u]; yield p
    if triples:
        changes = tuple(sorted(set((*permutations((1,1,-2)), *permutations((-1,-1,2))))))
        for indices in combinations(range(len(levels)),3):
            for change in changes:
                values = [units[levels[i]]+v for i,v in zip(indices,change)]
                if all(v in lookup for v in values):
                    p = levels.copy()
                    for i,v in zip(indices,values): p[i] = lookup[v]
                    yield p


def refine(compiled, evaluate, *, max_sweeps=100, batch_size=128, max_full_evaluations=None):
    """Exact compiled best improvement, full scoring/admission on every adoption."""
    if (type(max_sweeps) is not int or max_sweeps < 1 or type(batch_size) is not int or batch_size < 1
            or max_full_evaluations is not None and
            (type(max_full_evaluations) is not int or max_full_evaluations < 1)):
        raise ValueError('Positive explicit refinement allowances required')
    stamp = perf_counter(); start_count=compiled.scored_rows; current = compiled.original_levels.copy()
    official = evaluate(compiled.physical(current))
    if (not official['eligible'] or not isfinite(official['cost'])
            or abs(official['cost']-compiled.cost(current)) > 1e-10):
        raise ValueError('Initial compiled calendar must reproduce an admitted full score')
    original_cost = float(official['cost'])
    trace = []; full_calls = 1; complete = False; exhausted = False
    for sweep in range(max_sweeps):
        contenders = []; batch = []
        def consider():
            if not batch: return
            values = np.array(batch)
            costs = compiled.cost(values)
            admitted = compiled.funded(values)
            for p,c in zip(values[admitted],costs[admitted]):
                if c < official['cost']-1e-12:
                    contenders.append((float(c),tuple(int(v) for v in p)))
            batch.clear()
        for proposed in neighbors(compiled,current):
            batch.append(proposed)
            if len(batch) >= batch_size: consider()
        consider()
        best = None; rejected = 0
        for cost,levels in sorted(set(contenders)):
            if max_full_evaluations is not None and full_calls >= max_full_evaluations:
                exhausted = True; break
            checked = evaluate(compiled.physical(levels)); full_calls += 1
            if not isfinite(checked['cost']) or abs(checked['cost']-cost) > 1e-10:
                raise ValueError('Compiled alternative does not reproduce the authoritative score')
            if checked['eligible'] and checked['cost'] < official['cost']-1e-12:
                best = (np.array(levels),checked); break
            rejected += 1
        trace.append(dict(sweep=sweep,before_cost=official['cost'],after_cost=best[1]['cost'] if best else official['cost'],
            better_proposals=len(contenders),rejected=rejected))
        if best is None:
            complete = not exhausted; break
        current,official = best
    return dict(program=compiled.physical(current),cost=float(official['cost']),eligible=official['eligible'],
        original_cost=original_cost,trace=trace,
        full_evaluations=full_calls,compiled_evaluations=compiled.scored_rows-start_count,seconds=perf_counter()-stamp,
        neighborhood_complete=complete,budget_exhausted=exhausted or not complete,
        max_full_evaluations=max_full_evaluations,
        scope='Fixed calendar: full-catalogue singles, all-span fixed-volume pairs and unit triples; not global optimality')
