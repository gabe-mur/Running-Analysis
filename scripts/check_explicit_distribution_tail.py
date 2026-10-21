"""Reprice fixed distribution finalists without fluid continuation. No deployment.

Finite explicit windows are averages of an ongoing declared recurring programme,
not truncated plans with rest after the endpoint. An exact infinite-periodic
demand control independently removes all start/end dependence. Neither is a
replacement online scheduler or evidence of physiological optimality.
"""
from argparse import ArgumentParser
from dataclasses import replace
from datetime import datetime, time, timedelta
from itertools import combinations, permutations
from functools import lru_cache
from math import ceil, exp, expm1, isfinite, lcm
import json
from pathlib import Path
from time import perf_counter
from zoneinfo import ZoneInfo
import numpy as np

from scripts.check_easy_distribution_profiles import Profile, allocations, patterns
from scripts.replay_adaptive_search import load_decision
from scripts.research_manifest import freeze_sources
from run_analysis.config import load_config
from run_analysis.adaptive.program_objective_trial import AXES, TrialSession, demand_value
from run_analysis.adaptive.shared_program_evaluator import ProgramCandidate, evaluate_program
from run_analysis.adaptive.volume_benefit_trial import volume_benefit_cost
from run_analysis.workload_accounting import Quantity
from run_analysis.adaptive.decision_snapshot import evaluate_snapshot
from run_analysis.adaptive.new_model_daily_harness import planned_accounted
from run_analysis.adaptive.program_evidence_adapter import stage_quality_credit, _choose
from run_analysis.continuous_training_dose import ContinuousDoseReference
from run_analysis.program_delivery import TargetSegment
from run_analysis.adaptive.timely_purpose_trial import interval_loss


COMPRESSION_WEIGHTS = (0., .001, .002, .005, .01, .02, .05, .1)


def incoming_demand_trace(evaluation, context):
    """Inspect the actual evaluated ledger, never substitute periodic history.

    Carryover indices are normalized modeled demand, not readiness percentages.
    Verify each raw session charge before taper/boundary reweighting.
    """
    ledger = sorted(evaluation.conditional_ledger,key=lambda r:r.day)
    priced = {r['key']:r for r in evaluation.parts['session_costs']}
    dates = {r.day for r in ledger}
    prior = [0.,0.,0.]; previous = None; result = []
    compression = exp(context.scenario.timing_compression_days/context.scenario.memory_days)
    weights = context.scenario.axis_weights
    def weighted(values):
        return context.exchange*sum(w*v for w,v in zip(weights,values))/sum(weights)
    for r in ledger:
        if previous is not None:
            prior = [v*exp(-(r.day-previous).days/context.scenario.memory_days) for v in prior]
        q = [demand_value(v,context.scenario.bound_fraction)/s for v,s in zip(r.demand,context.scales)]
        from run_analysis.adaptive.demand_interaction import interaction
        mode=evaluation.parts.get('demand_interaction','independent_v1')
        own = (.5*interaction(q,q,mode)).tolist()
        overlap = (compression*interaction(q,prior,mode)).tolist()
        if r.source == 'planned':
            assert all(abs(priced[r.key]['axis_cost'][axis]-a-b)<1e-12 for axis,a,b in zip(AXES,own,overlap))
            if r.day<context.hard_end:
                result.append(dict(day=r.day.isoformat(),role=r.role,miles=r.miles,
                    preceding_by_axis=dict(zip(AXES,prior)),normalized_session_demand=dict(zip(AXES,q)),
                    raw_weighted_own=weighted(own),raw_weighted_overlap=weighted(overlap),
                    no_run_before=r.day-timedelta(days=1) not in dates,
                    no_run_after=r.day+timedelta(days=1) not in dates,
                    purpose_credit=r.quality_credit if r.role=='quality' else
                        min(1.,r.miles/context.policy.long_miles) if r.role=='long' else None))
        prior = [a+b for a,b in zip(prior,q)]; previous = r.day
    return result


def run_actual_plan_audit(root, output, *, index=1):
    """Local counterfactuals on the saved 21-day plan and its unchanged full tail.

    No invented recurring template/history, target freeze or new score. Reprice
    the whole original evaluator with its dated ramp and original eligibility.
    """
    case = load_decision(root,index)
    state,catalog,handoffs = case['state'],case['catalog'],case['handoffs']
    output.mkdir(exist_ok=False)
    manifest = freeze_sources(output/'sources.zip',config=load_config(),entrypoint=__name__,
        inputs=(root/f'decision-{index:04d}.json',root/case['event']['snapshot_file'],root/'scenario.json'))
    stamp = perf_counter(); reference = ContinuousDoseReference('sandbox upload accounting')
    original = tuple(sorted(tuple(p) for p in case['event']['program']))
    hard = tuple(p for p in original if datetime.fromisoformat(p[0]).date()<state.context.hard_end)
    tail = tuple(p for p in original if p not in hard)
    physical = {}; cache = {}
    def evaluate(program):
        program = tuple(sorted(program))
        if program in cache:
            return cache[program]
        assert len(dict(program))==len(program)
        assert tuple(p for p in program if datetime.fromisoformat(p[0]).date()>=state.context.hard_end)==tail
        for day,key in program:
            if (day,key) not in physical:
                physical[day,key] = planned_accounted(f'candidate:{day}:{key}',
                    datetime.combine(datetime.fromisoformat(day).date(),time(18),ZoneInfo(state.timezone)),
                    catalog[key],handoffs[key],reference,available=state.as_of)
        result = evaluate_snapshot(state,tuple(physical[p] for p in program),tuple(k for _,k in program),reference=reference)
        e = result['evaluation']
        trace = incoming_demand_trace(e,state.context)
        active = {r['day'] for r in trace}; streak = longest = 0
        # Include the recorded consecutive lead-in, not just the selectable dates.
        actual_dates = {r.start.astimezone(ZoneInfo(state.timezone)).date() for r in state.history}
        for n in range(-30,(state.context.hard_end-state.context.decision).days):
            d = state.context.decision+timedelta(days=n)
            streak = streak+1 if d in actual_dates or d.isoformat() in active else 0
            if n>=0:
                longest = max(longest,streak)
        row = dict(program=program,cost=e.total_cost,fatigue=e.weighted_demand_cost,
            delivery=e.delivery_cost,short=e.short_run_cost,losses=e.parts['shortfall_by_purpose'],
            eligible=result['decision_eligible'],constraints=result['constraints']['status'],
            suitability=result['suitability']['status'],hard_miles=sum(r['miles'] for r in trace),
            hard_runs=len(trace),longest_run_streak=longest,
            key_sessions=tuple(r for r in trace if r['role'] in ('long','quality')),
            trace=trace)
        assert abs(row['cost']-row['fatigue']-row['delivery']-row['short'])<1e-12
        cache[program] = row
        return row
    baseline = evaluate(original)
    assert abs(baseline['cost']-case['archived_cost'])<1e-12 and baseline['eligible']
    proposals = []
    empty = tuple(d for d in case['dates'] if datetime.fromisoformat(d).date()<state.context.hard_end and d not in dict(hard))
    for source in hard:
        for destination in empty:
            revised = tuple(sorted((*[p for p in original if p!=source],(destination,source[1]))))
            r = evaluate(revised)
            assert abs(r['hard_miles']-baseline['hard_miles'])<1e-10
            proposals.append(dict(kind='one_session_move',source=source,destination=destination,**r))
    easies = tuple(p for p in hard if catalog[p[1]].role=='easy')
    easy_units = {round(catalog[k].miles*10):k for k in handoffs if catalog[k].role=='easy'}
    for first,second in combinations(easies,2):
        if (datetime.fromisoformat(second[0])-datetime.fromisoformat(first[0])).days>3:
            continue
        combined = round(catalog[first[1]].miles*10)+round(catalog[second[1]].miles*10)
        if combined not in easy_units:
            continue
        for destination in (first[0],second[0]):
            revised = tuple(sorted((*[p for p in original if p not in (first,second)],(destination,easy_units[combined]))))
            r = evaluate(revised)
            assert abs(r['hard_miles']-baseline['hard_miles'])<1e-10
            proposals.append(dict(kind='merge_nearby_easy',source=(first,second),destination=destination,**r))
    key_calendar = tuple(p for p in hard if catalog[p[1]].role in ('long','quality'))
    def preserves_key_rest(row):
        if tuple(p for p in row['program'] if p in key_calendar)!=key_calendar:
            return False
        # Preserve all baseline rest immediately before keys and after longs;
        # additionally create rest after each hard-prefix quality. Diagnostic
        # grouping, not a new selection/eligibility rule.
        return all(k['no_run_before'] and k['no_run_after'] for k in row['key_sessions'])
    eligible = tuple(r for r in proposals if r['eligible'])
    rest_options = tuple(r for r in eligible if preserves_key_rest(r))
    quality = tuple(r for r in baseline['key_sessions'] if r['role']=='quality')
    report = dict(manifest=manifest,source_snapshot=state.fingerprint,as_of=state.as_of.isoformat(),
        decision=state.context.decision.isoformat(),hard_end=state.context.hard_end.isoformat(),
        end=state.context.end.isoformat(),seconds=perf_counter()-stamp,unique_full_evaluations=len(cache),
        baseline=baseline,proposals=proposals,eligible_proposals=len(eligible),
        best_local=min(eligible,key=lambda r:(r['cost'],r['program'])) if eligible else None,
        all_keys_surrounded_by_rest=sorted(rest_options,key=lambda r:(r['cost'],r['program'])),
        quality_entry_range={axis:(min(r['preceding_by_axis'][axis] for r in quality),
            max(r['preceding_by_axis'][axis] for r in quality)) for axis in AXES},
        assumptions=['Actual saved 21-day prescriptions, full unchanged 54-day continuation, actual observed history and dated gradual targets',
            'Every single hard-prefix run move to every empty hard-prefix date; supported nearby easy merges preserve exact hard mileage',
            'All candidates use current whole-program score and suitability; no seven-day block quotas or fixed endpoint target',
            'Rest-surrounded keys are a labelled diagnostic subset, not an imposed coaching rule',
            'Per-session recurrence reproduces charged axis costs before taper; normalized carryover is not physical readiness',
            'Bounded local neighborhood, not exhaustive search, daily churn test or physiological validation'],
        online_status='Offline diagnosis only; no model, live plan, database or score change')
    (output/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(seconds=report['seconds'],evaluations=len(cache),baseline={k:baseline[k] for k in
        ('cost','fatigue','delivery','short','hard_miles','hard_runs','key_sessions')},
        best_local={k:report['best_local'][k] for k in ('kind','source','destination','cost','fatigue','delivery','short','longest_run_streak')},
        rest_options=[{k:r[k] for k in ('source','destination','cost','fatigue','delivery','short','longest_run_streak','key_sessions')} for r in rest_options])),flush=True)


def four_dose_cycles(keys):
    """All 120 relative placements of four distinct fixed doses in seven days.

    Anchor the first dose at zero; all seven absolute phases are priced separately.
    This is a bounded existence test, not a new calendar-week planner or search.
    """
    if len(keys) != 4 or len(set(keys)) != 4:
        raise ValueError('Four distinct fixed dose keys required')
    return tuple(tuple(sorted(((0, keys[0]), *zip(days, keys[1:]))))
                 for days in permutations(range(1, 7), 3))


def shift_cycle(pattern, phase):
    return tuple(sorted(((d+phase)%7, key) for d, key in pattern))


def cycle_program(pattern, origin, days):
    return tuple(((origin+timedelta(days=n)).isoformat(), key)
                 for n in range(days) for d, key in pattern if n%7 == d)


def decimal_compositions(total, count, supported):
    """Every ordered allocation in integer tenths, without inventing forecasts."""
    supported = tuple(sorted(set(supported)))
    if type(total) is not int or type(count) is not int or count < 1 or not supported or any(
            type(v) is not int or v <= 0 for v in supported):
        raise ValueError('Positive supported integer tenths and count required')
    def walk(remaining, slots):
        if slots == 1:
            if remaining in supported:
                yield (remaining,)
            return
        for value in supported:
            if (slots-1)*supported[0] <= remaining-value <= (slots-1)*supported[-1]:
                for suffix in walk(remaining-value, slots-1):
                    yield (value, *suffix)
    return tuple(walk(total, count))


def stationary_batch(days, vectors, shorts, context, *, period=7):
    """Vectorized exact periodic score, independently checked against the ledger.

    vectors: candidate x session x axis, already normalized with frozen scales.
    This evaluates all supplied candidates, not an approximate search budget.
    """
    days = tuple(days)
    if type(period) is not int or period < 2 or len(set(days)) != len(days) or any(not 0 <= d < period for d in days):
        raise ValueError('Unique dates within a declared cycle required')
    q = np.asarray(vectors, dtype=float)
    s = np.asarray(shorts, dtype=float)
    if q.ndim != 3 or q.shape[1:] != (len(days), 3) or s.shape != q.shape[:2]:
        raise ValueError('Candidate/session/axis shapes must agree')
    memory = context.scenario.memory_days
    kernel = np.asarray([[exp(-(((a-b)%period or period)-context.scenario.timing_compression_days)/memory)
                          /(-expm1(-period/memory)) for b in days] for a in days])
    raw = .5*np.sum(q*q, axis=1)+np.einsum('nia,ij,nja->na', q, kernel, q, optimize=True)
    weights = np.asarray(context.scenario.axis_weights)
    fatigue = context.exchange*(raw @ weights)/weights.sum()*7/period
    short = s.sum(axis=1)*7/period
    return fatigue, short


@lru_cache(maxsize=8192)
def periodic_volume_weights(days, period, window, smoothing):
    return np.asarray([[sum((day-back-smooth)%period == d
        for back in range(window) for smooth in range(smoothing))/smoothing
        for d in days] for day in range(period)])


def stationary_volume_batch(days, miles, context, *, period, target):
    """Exact trailing-window/smoothing loss of an infinite periodic ledger.

    Unlike seven-day recurrence, a longer cycle can have nonconstant rolling
    delivery. Preserve the existing signed-average-before-clipping contract.
    """
    weights = periodic_volume_weights(tuple(days),period,context.policy.volume_window_days,context.volume_smoothing_days)
    delivered = np.asarray(miles) @ weights.T
    return np.mean(np.maximum(0., 1-delivered/(target*context.policy.volume_window_days/7))**2, axis=1)


def stationary_timing_cost(key_days, context, *, period):
    """Exact full-credit cadence loss, including gaps across the boundary."""
    total = 0.
    for role in ('long', 'quality'):
        days = sorted(d for d,r in key_days if r == role)
        if not days:
            raise ValueError('Both key purposes required for this diagnostic')
        memory = 7/getattr(context.policy, role+'_per_7')*context.timing.memory_scale
        grace = 7/getattr(context.policy, role+'_per_7')*context.timing.grace_fraction
        weight = context.policy.purpose_weights[2 if role == 'long' else 3]
        total += weight*sum(interval_loss(1., max(0., (days[(i+1)%len(days)]-d)%period or period)-grace,
                             memory) if ((days[(i+1)%len(days)]-d)%period or period)>grace else 0.
                            for i,d in enumerate(days))/period
    return total


def periodic_fatigue_lower_bound(total_vector, context, *, period):
    """Jensen bound for integrated squared residue; nominal compression only.

    A componentwise lower bound on total normalized positive demand is valid
    regardless of its placement. No short/volume/cadence cost is assumed here.
    """
    if context.scenario.timing_compression_days != 0:
        raise ValueError('Squared-residue bound requires zero timing compression')
    weights = context.scenario.axis_weights
    return context.exchange*7*context.scenario.memory_days*sum(w*q*q/period**2
        for w,q in zip(weights,total_vector))/sum(weights)


def mixed_key_layouts(keys, *, period=14):
    """Two indistinguishable quality/long doses; canonical quality anchor only."""
    result = []
    for second_q in range(1,period):
        free = tuple(d for d in range(1,period) if d!=second_q)
        for longs in combinations(free,2):
            layout = tuple(sorted(((0,keys[0]),(second_q,keys[0]),*((d,keys[1]) for d in longs))))
            rotated = tuple(sorted(((d-second_q)%period,k) for d,k in layout))
            if layout<=rotated:
                result.append(layout)
    return tuple(result)


def cyclic_gaps(days, period):
    days = tuple(sorted(days))
    if not days or len(set(days))!=len(days) or any(not 0<=d<period for d in days):
        raise ValueError('Distinct nonempty cycle offsets required')
    return tuple((days[(i+1)%len(days)]-d)%period or period for i,d in enumerate(days))


def run_variable_cycle_probe(root, output, *, index=1):
    """Bounded longer-cycle study; retain original cadence costs, not due dates.

    Exhaust dates/counts in declared 6..10-day and mixed 14-day cycles. Dose seeds reuse the
    existing allocation family, then exact decimal transfers refine finalists.
    This is neither exhaustive across dose grids nor a new online scheduler.
    """
    case = load_decision(root, index)
    state, catalog, handoffs = case['state'], case['catalog'], case['handoffs']
    ctx = state.context
    if ctx.scenario.bound_fraction != .5 or ctx.policy.purpose_weights != (1.,0.,1.,1.):
        raise ValueError('Nominal single-volume snapshot required')
    output.mkdir(exist_ok=False)
    manifest = freeze_sources(output/'sources.zip', config=load_config(), entrypoint=__name__,
        inputs=(root/f'decision-{index:04d}.json', root/case['event']['snapshot_file'], root/'scenario.json'))
    stamp = perf_counter()
    origin = ctx.decision
    reference = ContinuousDoseReference('sandbox upload accounting')
    key_names = ('quality:ordinary', 'long:8.9')
    easy_keys = tuple(k for k in catalog if k.startswith('easy:') and k in handoffs)
    templates = {key:planned_accounted(f'variable-cycle:{key}',
        datetime.combine(origin, time(18), ZoneInfo(state.timezone)), catalog[key], handoffs[key],
        reference, available=state.as_of) for key in (*key_names, *easy_keys)}
    units = {round(templates[k].miles*10):k for k in easy_keys}
    key_units = sum(round(templates[k].miles*10) for k in key_names)
    assert all(_choose(stage_quality_credit(templates[k],ctx.policy),.5) == 1.
               for k in key_names if templates[k].role == 'quality')
    assert templates[key_names[1]].miles == ctx.policy.long_miles
    vectors = {k:tuple(q.estimate/s for q,s in zip(r.demand,ctx.scales)) for k,r in templates.items()}
    short = {k:ctx.easy_duration_policy.penalty(r.demand[0].estimate) if r.role == 'easy' else 0.
             for k,r in templates.items()}
    pace = templates['easy:4.4'].demand[0].estimate/4.4
    target = ctx.volume_targets[-1].miles_per_7_days
    @lru_cache(maxsize=200000)
    def price(pattern, period):
        days = tuple(d for d,k in pattern)
        f,s = stationary_batch(days, [[vectors[k] for _,k in pattern]], [[short[k] for _,k in pattern]],ctx,period=period)
        volume = stationary_volume_batch(days, [[templates[k].miles for _,k in pattern]],ctx,period=period,target=target)[0]
        timing = stationary_timing_cost(tuple((d,templates[k].role) for d,k in pattern
            if templates[k].role != 'easy'),ctx,period=period)
        return dict(pattern=pattern,period=period,cost=float(f[0]+s[0]+volume+timing),
            fatigue=float(f[0]),short=float(s[0]),volume_loss=float(volume),timing=float(timing))
    def verify(result):
        pattern,period = result['pattern'],result['period']
        days = lcm(period,7); days *= ceil(56/days)
        past, future = [], []
        for n in range(-max(84,period*4), days):
            for offset,key in pattern:
                if n%period == offset:
                    r = templates[key]
                    item = TrialSession(f'periodic:{n}:{key}',origin+timedelta(days=n),r.miles,r.role,r.demand,
                        _choose(stage_quality_credit(r,ctx.policy),.5),'observed' if n < 0 else 'planned')
                    (past if n < 0 else future).append(item)
        c = explicit_context(ctx,days)
        c = replace(c,history=tuple(past),stage_start=origin,
            volume_targets=(TargetSegment(origin-timedelta(days=84),c.end,target),),
            support_targets=(TargetSegment(origin-timedelta(days=84),c.end,7.5),))
        e = evaluate_program(c,ProgramCandidate(tuple(r for r in future if r.day<c.hard_end),
            tuple(r for r in future if r.day>=c.hard_end),'explicit continuing variable cycle with matched hypothetical history'))
        assert abs(e.total_cost-result['cost']) < 1e-12, (result,e.total_cost)
        active = {d for d,k in pattern}; streak=longest=0
        for d in range(period*2):
            streak = streak+1 if d%period in active else 0
            longest = max(longest,streak)
        return dict(**result,full_cost=e.total_cost,verification_days=days,
            losses=e.parts['shortfall_by_purpose'],miles_per_7=sum(templates[k].miles for _,k in pattern)*7/period,
            runs_per_7=len(pattern)*7/period,
            no_run_dates=period-len(pattern),
            longest_run_streak=longest if len(active)<period else 'continuous',
            key_gaps={role:cyclic_gaps(tuple(d for d,k in pattern if templates[k].role==role),period)
                      for role in ('long','quality')},
            easy_miles=tuple(templates[k].miles for _,k in pattern if templates[k].role=='easy'),
            minimum_duration_pass=all(templates[k].demand[0].high>=ctx.easy_duration_policy.minimum_minutes
                for _,k in pattern if templates[k].role=='easy'))
    evaluations = 0; panels = []
    for period in (*range(6,11),14):
        total_units = round(target*period/7*10)
        if abs(total_units/10*7/period-target)>1e-8:
            raise ValueError('Equal average volume cannot be represented at this period')
        pruning = None
        if period == 14:
            layouts = mixed_key_layouts(key_names,period=period)
            support = total_units-2*key_units
            total_lower = tuple(2*sum(vectors[k][a] for k in key_names)+support/10*
                min(vectors[k][a]/templates[k].miles for k in easy_keys) for a in range(3))
            bound = periodic_fatigue_lower_bound(total_lower,ctx,period=period)
            incumbent = min(p['best']['cost'] for p in panels)
            retained = tuple(layout for layout in layouts if bound+stationary_timing_cost(
                tuple((d,templates[k].role) for d,k in layout),ctx,period=period)<incumbent)
            pruning = dict(relative_key_layouts=len(layouts),retained=len(retained),
                fatigue_lower_bound=bound,incumbent_cost=incumbent,
                basis='Only layouts whose rigorous fatigue lower bound plus exact timing already loses to incumbent are skipped')
            layouts = retained
        else:
            support = total_units-key_units
            layouts = tuple(((0,key_names[0]),(long_day,key_names[1])) for long_day in range(1,period))
        by_count = []; uneven_candidates = []
        key_count = len(layouts[0])
        for count in range(1,period-key_count+1):
            seeds = tuple(row for row in allocations(support,count,pace) if all(u in units for u in row))
            if not seeds:
                continue
            eq = np.asarray([[vectors[units[u]] for u in row] for row in seeds])
            es = np.asarray([[short[units[u]] for u in row] for row in seeds])
            em = np.asarray(seeds)/10
            finalists = []
            for key_layout in layouts:
                timing = stationary_timing_cost(tuple((d,templates[k].role) for d,k in key_layout),ctx,period=period)
                key_days = tuple(d for d,k in key_layout)
                free = tuple(d for d in range(period) if d not in key_days)
                for positions in combinations(free,count):
                    dates = (*key_days,*positions)
                    q = np.empty((len(seeds),count+key_count,3))
                    q[:,:key_count,:],q[:,key_count:,:] = [vectors[k] for _,k in key_layout],eq
                    penalties = np.zeros((len(seeds),count+key_count)); penalties[:,key_count:] = es
                    miles = np.empty((len(seeds),count+key_count))
                    miles[:,:key_count],miles[:,key_count:] = [templates[k].miles for _,k in key_layout],em
                    f,s = stationary_batch(dates,q,penalties,ctx,period=period)
                    # An exact funded cycle dividing 28 has constant trailing
                    # delivery, regardless of date placement. Not a waiver.
                    v = 0. if ctx.policy.volume_window_days%period==0 else stationary_volume_batch(
                        dates,miles,ctx,period=period,target=target)
                    costs = f+s+v+timing
                    ix = int(np.argmin(costs))
                    pattern = tuple(sorted((*key_layout,
                        *zip(positions,(units[u] for u in seeds[ix])))))
                    finalists.append((float(costs[ix]),pattern))
                    evaluations += len(seeds)
            # Improve top distinct calendars with exact one-decimal, equal-volume
            # transfers. Local minimum only; never claim a dose-grid certificate.
            refined = []
            shortlist = sorted(finalists)[:8]
            if period == 14:
                uneven = sorted(row for row in finalists if any(cyclic_gaps(
                    tuple(d for d,k in row[1] if templates[k].role==role),period)!=(7,7)
                    for role in ('long','quality')))
                shortlist = sorted(set((*shortlist,*uneven[:8])))
            for _,pattern in shortlist:
                current = price(pattern,period)
                while True:
                    neighbors = []
                    easy = tuple((i,round(templates[k].miles*10)) for i,(_,k) in enumerate(current['pattern'])
                                 if templates[k].role=='easy')
                    for i,a in easy:
                        for j,b in easy:
                            if i == j or a-1 not in units or b+1 not in units:
                                continue
                            changed = list(current['pattern'])
                            changed[i] = (changed[i][0],units[a-1]); changed[j] = (changed[j][0],units[b+1])
                            neighbors.append(price(tuple(changed),period)); evaluations += 1
                    if not neighbors:
                        break
                    winner = min(neighbors,key=lambda r:(r['cost'],r['pattern']))
                    if winner['cost']>=current['cost']-1e-14:
                        break
                    current = winner
                refined.append(current)
                if period == 14 and any(cyclic_gaps(tuple(d for d,k in current['pattern']
                    if templates[k].role==role),period)!=(7,7) for role in ('long','quality')):
                    uneven_candidates.append(current)
            best = min(refined,key=lambda r:(r['cost'],r['pattern']))
            by_count.append(dict(easy_count=count,seeds=len(seeds),calendar_placements=len(finalists),**verify(best)))
        best = min(by_count,key=lambda r:(r['cost'],r['pattern']))
        panel = dict(period=period,total_cycle_miles=total_units/10,by_easy_count=by_count,best=best,
            key_layout_pruning=pruning,best_uneven=verify(min(uneven_candidates,key=lambda r:(r['cost'],r['pattern'])))
                if uneven_candidates else None)
        panels.append(panel)
        print(json.dumps(dict(period=period,best=best,evaluations=evaluations)),flush=True)
    best = min((p['best'] for p in panels),key=lambda r:(r['cost'],r['period'],r['pattern']))
    report = dict(manifest=manifest,seconds=perf_counter()-stamp,panels=panels,best=best,evaluations=evaluations,
        assumptions=['No history timing or transition costs; hypothetical matched history only for evaluator parity',
            'Compare 6..10-day recurrences with one unchanged long/quality per cycle and 14-day cycles with two of each at freely varied dates',
            'Same exact average volume, original cadence weights/plateau, rolling 28-day delivery and 7-day smoothing retained',
            'All date placements/counts for existing balanced/anchor dose family, plus decimal transfers on top eight calendars per count',
            'Not exhaustive across all decimal allocations or all possible recurring schedules',
            'Complete periodic boundary costs; no monthly reset, mandatory rest or new fatigue weights',
            'Mathematical ranking is not physiological feasibility or daily stability'],online_status='Diagnostic only; no live changes')
    (output/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(seconds=report['seconds'],evaluations=evaluations,best=best)),flush=True)


def run_stationary_allocation_probe(root, output, *, index=1):
    """Ongoing equal-volume comparison only; deliberately no transition pricing."""
    case = load_decision(root, index)  # Confirm original snapshot's exact price.
    state, catalog, handoffs = case['state'], case['catalog'], case['handoffs']
    if state.context.scenario.bound_fraction != .5:
        raise ValueError('This probe requires the explicit nominal forecast scenario')
    output.mkdir(exist_ok=False)
    manifest = freeze_sources(output/'sources.zip', config=load_config(), entrypoint=__name__,
        inputs=(root/f'decision-{index:04d}.json', root/case['event']['snapshot_file'], root/'scenario.json'))
    stamp = perf_counter()
    origin = state.context.decision
    reference = ContinuousDoseReference('sandbox upload accounting')
    keys = ('quality:ordinary', 'long:8.9')
    easy_keys = tuple(k for k in catalog if k.startswith('easy:') and k in handoffs)
    templates = {key:planned_accounted(f'stationary:{key}',
        datetime.combine(origin, time(18), ZoneInfo(state.timezone)), catalog[key], handoffs[key],
        reference, available=state.as_of) for key in (*keys, *easy_keys)}
    units = {round(templates[k].miles*10):k for k in easy_keys}
    if len(units) != len(easy_keys) or any(abs(templates[k].miles*10-u) > 1e-8 for u,k in units.items()):
        raise ValueError('Unique committed one-decimal easy forecasts required')
    def vector(key):
        return tuple(q.estimate/scale for q,scale in zip(templates[key].demand, state.context.scales))
    def penalty(key):
        return state.context.easy_duration_policy.penalty(templates[key].demand[0].estimate)
    def analytic(pattern):
        entries = tuple((d, templates[k].role, templates[k].miles) for d,k in pattern)
        def demand(role, miles):
            return next(r.demand for r in templates.values() if r.role == role and r.miles == miles)
        result = periodic_demand(entries, demand, state.context)
        return dict(pattern=pattern, cost=result['fatigue']+result['short'], **result)
    def verify(pattern, total):
        past, future = [], []
        for n in range(-84, 56):
            for day,key in pattern:
                if n%7 == day:
                    r = templates[key]
                    credit = _choose(stage_quality_credit(r, state.context.policy), .5)
                    item = TrialSession(f'stationary:{n}:{key}', origin+timedelta(days=n), r.miles,
                        r.role, r.demand, credit, 'observed' if n < 0 else 'planned')
                    (past if n < 0 else future).append(item)
        c = explicit_context(state.context, 56)
        c = replace(c, history=tuple(past), stage_start=origin,
            volume_targets=(TargetSegment(origin-timedelta(days=84), c.end, 20.3),),
            support_targets=(TargetSegment(origin-timedelta(days=84), c.end, 7.5),))
        e = evaluate_program(c, ProgramCandidate(tuple(r for r in future if r.day < c.hard_end),
            tuple(r for r in future if r.day >= c.hard_end), 'matched recurring hypothetical history, not actual history'))
        r = analytic(pattern)
        assert abs(e.weighted_demand_cost-r['fatigue']) < 1e-12
        assert abs(e.short_run_cost-r['short']) < 1e-12
        assert abs(sum(templates[k].miles for _,k in pattern)-total) < 1e-10
        return dict(**r, full_cost=e.total_cost, delivery=e.delivery_cost,
            losses=e.parts['shortfall_by_purpose'],
            minimum_duration_pass=all(templates[k].demand[0].high >= c.easy_duration_policy.minimum_minutes
                for _,k in pattern if templates[k].role == 'easy'),
            physiological_feasibility='not established by this diagnostic')
    panels = []
    for support, baseline_easy in ((56, (33,23)), (65, (33,32)), (75, (37,38))):
        baseline = tuple(sorted(((0,keys[0]), (5,keys[1]), (2,units[baseline_easy[0]]), (3,units[baseline_easy[1]]))))
        by_count = []
        total_candidates = 0
        for count in range(1,6):
            choices = decimal_compositions(support, count, tuple(units))
            if not choices:
                continue
            # Forecast lookup, never scaling a measured run or a quality template.
            eq = np.asarray([[vector(units[u]) for u in row] for row in choices])
            es = np.asarray([[penalty(units[u]) for u in row] for row in choices])
            best = None; evaluated = 0
            for long_day in range(1,7):
                free = tuple(d for d in range(1,7) if d != long_day)
                for positions in combinations(free, count):
                    q = np.empty((len(choices), count+2, 3))
                    q[:,0,:], q[:,1,:], q[:,2:,:] = vector(keys[0]), vector(keys[1]), eq
                    shorts = np.zeros((len(choices), count+2)); shorts[:,2:] = es
                    fatigue, short = stationary_batch((0,long_day,*positions), q, shorts, state.context)
                    costs = fatigue+short
                    winner = int(np.argmin(costs))
                    pattern = tuple(sorted(((0,keys[0]), (long_day,keys[1]),
                        *zip(positions, (units[u] for u in choices[winner])))))
                    candidate = (float(costs[winner]), pattern)
                    if best is None or candidate < best:
                        best = candidate
                    evaluated += len(choices)
            checked = verify(best[1], 12.8+support/10)
            assert abs(checked['cost']-best[0]) < 1e-12
            by_count.append(dict(easy_count=count, run_count=count+2, no_run_days=5-count,
                allocations=len(choices), calendars=evaluated, **checked))
            total_candidates += evaluated
        original = verify(baseline, 12.8+support/10)
        best = min(by_count, key=lambda r:(r['cost'],r['pattern']))
        assert all(abs(r['delivery']-original['delivery']) < 1e-12 for r in by_count)
        panel = dict(support_miles=support/10, total_miles=12.8+support/10,
            calendars=total_candidates, baseline=original, best=best, by_easy_count=by_count,
            gain=original['cost']-best['cost'], fatigue_short_gain_percent=100*(original['cost']-best['cost'])/original['cost'])
        panels.append(panel)
        print(json.dumps(panel), flush=True)
    report = dict(seconds=perf_counter()-stamp, manifest=manifest, source_snapshot=state.fingerprint,
        supported_easy_miles=sorted(u/10 for u in units), panels=panels,
        calendars=sum(p['calendars'] for p in panels),
        assumptions=['No transition cost, actual-history timing or continuity penalty in ongoing ranking',
            'One quality 3.9 and one long 8.9 per repeating seven days; cadence itself is not optimized',
            'Every supported decimal easy allocation and free-date placement is enumerated at equal volume',
            'Current nominal demands, normalization, memory, short-duration curve and weights unchanged',
            'Matched hypothetical history used only to verify stationary scoring, not to alter app history',
            'Existing endpoint target 20.3 retained to show underfunding, constant within each equal-volume panel',
            'No coaching compression reward or mandatory rest; no daily-churn or physiological-optimality claim'],
        online_status='Diagnostic only; live server and database untouched')
    (output/'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(seconds=report['seconds'], calendars=report['calendars'])), flush=True)


def run_calendar_probe(root, output, *, index=1):
    """Reuse the actual frozen forecast/evaluator to test for a better pattern.

    Fixed four-dose panels only. No history deletion, new continuity penalty,
    fatigue setting, volume utility, clinical clearance or live publication.
    Long scopes freeze the stage solely to separate transient from recurring cost.
    """
    case = load_decision(root, index)  # Mandatory exact original repricing.
    state, catalog, handoffs = case['state'], case['catalog'], case['handoffs']
    output.mkdir(exist_ok=False)
    manifest = freeze_sources(output/'sources.zip', config=load_config(), entrypoint=__name__,
        inputs=(root/f'decision-{index:04d}.json', root/case['event']['snapshot_file'], root/'scenario.json'))
    stamp = perf_counter(); calls = 0
    reference = ContinuousDoseReference('sandbox upload accounting')
    origin = state.context.decision
    physical = {}; results = {}
    def accounted(day, key):
        if (day, key) not in physical:
            physical[day, key] = planned_accounted(f'candidate:{day}:{key}',
                datetime.combine(datetime.fromisoformat(day).date(), time(18), ZoneInfo(state.timezone)),
                catalog[key], handoffs[key], reference, available=state.as_of)
        return physical[day, key]
    def evaluate(pattern, days=54):
        nonlocal calls
        key = (pattern, days)
        if key in results:
            return results[key]
        c = state.context if days == 54 else explicit_context(state.context, days)
        if days != 54:
            # Preserve the dated ramp and lead-in; extend ONLY its last endpoint.
            c = replace(c, volume_targets=(*c.volume_targets[:-1], replace(c.volume_targets[-1], end=c.end)),
                support_targets=(*c.support_targets[:-1], replace(c.support_targets[-1], end=c.end)))
        program = cycle_program(pattern, origin, days)
        runs = tuple(accounted(d, k) for d, k in program)
        r = evaluate_snapshot(replace(state, context=c), runs, tuple(k for _, k in program), reference=reference)
        e = r['evaluation']; calls += 1
        assert abs(e.total_cost-e.delivery_cost-e.weighted_demand_cost-e.short_run_cost) < 1e-10
        results[key] = dict(pattern=pattern, cost=e.total_cost, fatigue=e.weighted_demand_cost,
            delivery=e.delivery_cost, short=e.short_run_cost, losses=e.parts['shortfall_by_purpose'],
            eligible=r['decision_eligible'], constraints=r['constraints']['status'],
            scope='Guarded/tapered 54 days' if days == 54 else 'Explicit recurring future with unchanged actual history and dated ramp')
        return results[key]
    reports = []
    panels = (('selected_initial_doses', ('quality:ordinary', 'long:8.9', 'easy:3.3', 'easy:2.3')),
              ('endpoint_volume_doses', ('quality:ordinary', 'long:8.9', 'easy:3.7', 'easy:3.8')))
    for name, keys in panels:
        if any(key not in handoffs for key in keys):
            raise ValueError(f'Unsupported fixed panel {name}; do not invent a forecast')
        templates = {key:accounted(origin.isoformat(), key) for key in keys}
        doses = tuple(sorted((key, r.role, r.miles) for key, r in templates.items()))
        cycles = four_dose_cycles(keys)
        baseline = tuple(sorted(((0, keys[0]), (2, keys[2]), (3, keys[3]), (5, keys[1]))))
        def demand(role, miles):
            return next(r.demand for r in templates.values() if r.role == role and r.miles == miles)
        def stationary(pattern):
            entries = tuple((d, templates[k].role, templates[k].miles) for d, k in pattern)
            r = periodic_demand(entries, demand, state.context)
            return dict(pattern=pattern, cost=r['fatigue']+r['short'], **r)
        ranked = sorted((stationary(p) for p in cycles), key=lambda r:(r['cost'], r['pattern']))
        best = ranked[0]
        tied = [r for r in ranked if abs(r['cost']-best['cost']) < 1e-12]
        original = stationary(baseline)
        # Independently verify stationary fatigue/short and timing parity through
        # the existing full evaluator, using declared matched recurring history.
        stationary_checks = []
        for pattern in (baseline, *(r['pattern'] for r in tied)):
            future = [] ; past = []
            for n in range(-84, 56):
                for d, key in pattern:
                    if n%7 != d:
                        continue
                    r = templates[key]
                    q = _choose(stage_quality_credit(r, state.context.policy), .5)
                    item = TrialSession(f'cycle:{n}:{key}', origin+timedelta(days=n), r.miles,
                        r.role, r.demand, q, 'observed' if n < 0 else 'planned')
                    (past if n < 0 else future).append(item)
            c = explicit_context(state.context, 56)
            # Constant endpoint target for stationary comparison; frozen physical
            # doses/quality credit and timing unchanged. Not backdated app history.
            c = replace(c, history=tuple(past), volume_targets=(TargetSegment(origin-timedelta(days=84), c.end, 20.3),),
                support_targets=(TargetSegment(origin-timedelta(days=84), c.end, 7.5),))
            e = evaluate_program(c, ProgramCandidate(tuple(r for r in future if r.day < c.hard_end),
                tuple(r for r in future if r.day >= c.hard_end), 'declared stationary comparison cycle'))
            analytic = stationary(pattern)
            assert abs(e.weighted_demand_cost-analytic['fatigue']) < 1e-12
            assert abs(e.short_run_cost-analytic['short']) < 1e-12
            stationary_checks.append(dict(pattern=pattern, cost=e.total_cost, fatigue=e.weighted_demand_cost,
                delivery=e.delivery_cost, short=e.short_run_cost, losses=e.parts['shortfall_by_purpose']))
        assert all(abs(r['delivery']-stationary_checks[0]['delivery']) < 1e-12 for r in stationary_checks)
        priced = []
        for i, p in enumerate(cycles):
            priced.extend(evaluate(shift_cycle(p, phase)) for phase in range(7))
            if (i+1)%30 == 0:
                print(json.dumps(dict(panel=name, relative_calendars=i+1, full_evaluations=calls)), flush=True)
        eligible = [r for r in priced if r['eligible']]
        if not eligible:
            raise ValueError('No eligible recurring comparison calendar')
        full_best = min(eligible, key=lambda r:(r['cost'], r['pattern']))
        baselines = [evaluate(shift_cycle(baseline, phase)) for phase in range(7)]
        baseline_best = min((r for r in baselines if r['eligible']), key=lambda r:(r['cost'], r['pattern']))
        ongoing_candidates = [evaluate(shift_cycle(r['pattern'], phase)) for r in tied for phase in range(7)]
        ongoing_best = min((r for r in ongoing_candidates if r['eligible']), key=lambda r:(r['cost'], r['pattern']))
        named = dict(selected_phase=shift_cycle(baseline, 1), best_baseline_phase=baseline_best['pattern'],
            best_stationary_transition=ongoing_best['pattern'], best_guarded_calendar=full_best['pattern'])
        horizons = {label:{str(days):evaluate(p, days) for days in (54, 56, 112, 224, 448)}
                    for label, p in named.items()}
        report = dict(panel=name, doses=doses, miles_per_7=sum(r.miles for r in templates.values()),
            relative_calendars=len(cycles), absolute_calendars=len(priced), eligible_calendars=len(eligible),
            baseline_stationary=original, best_stationary=best, tied_stationary=tied,
            stationary_full_checks=stationary_checks, best_guarded=full_best,
            best_baseline_phase=baseline_best, best_stationary_transition=ongoing_best,
            named_horizons=horizons, all_guarded_prices=priced)
        reports.append(report)
        print(json.dumps(dict(panel=name, stationary_gain=original['cost']-best['cost'],
            best_guarded=full_best, selected_phase=evaluate(shift_cycle(baseline, 1)))), flush=True)
    report = dict(manifest=manifest, source_snapshot=state.fingerprint, seconds=perf_counter()-stamp,
        full_transition_evaluations=calls, panels=reports,
        scope='Fixed four-dose recurring calendar existence probe, not a complete scheduler or daily simulation',
        assumptions=['Seven-day recurrence fixes the existing once-per-seven long/quality goal for this probe only',
            'Compare all relative placements and absolute phases; no weekday preference or continuity allowance',
            'Forecasts, rates, quantities, targets, eligibility and actual physical history retained in transition checks',
            'Stationary matched history is a separately labelled hypothetical control, never substituted into the actual transition',
            'Endpoint targets 20.3/7.5 are frozen existing stage goals, not new progression',
            'Long scopes intentionally freeze capacity/stage for pattern comparison, not prescribe 448 days without adaptation',
            'No AS80 volume utility or AS82 compression reward; mathematical costs are not tendon safety or physiological optimality'])
    (output/'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(seconds=report['seconds'], full_transition_evaluations=calls)))


def easy_concentration(pattern):
    """Offline policy utility, NOT training benefit. Empty support gets no reward.

    Scale invariant, hence unsafe as a standalone volume-preserving objective.
    No dependence on labels, calendar origin or count of rest days here.
    """
    miles = tuple(m for _, m in pattern)
    if any(not isfinite(m) or m <= 0 for m in miles):
        raise ValueError('Concentration requires finite positive easy doses')
    total = sum(miles)
    return sum(m*m for m in miles)/total**2 if total else 0.


def compression_winner(sampled, weight):
    """Existing fatigue + short cost minus explicit concentration preference."""
    if not isfinite(weight) or weight < 0:
        raise ValueError('Compression weight must be finite and nonnegative')
    return min(sampled, key=lambda item:(item[0]-weight*easy_concentration(item[1]), item))


def compression_switches(sampled):
    """Exact positive-weight lower envelope within the original sampled family.

    Each candidate is a line cost-w*H. Increasing weight can only increase H.
    This is not an exhaustive calendar or dose-space optimality claim.
    """
    current = compression_winner(sampled, 0.)
    result = [(0., current)]
    while True:
        h = easy_concentration(current[1])
        alternatives = [(max(0., (cost-current[0])/(other-h)), -other, (cost, pattern))
                        for cost, pattern in sampled
                        if (other := easy_concentration(pattern)) > h+1e-12]
        if not alternatives:
            return result
        weight, _, current = min(alternatives)
        assert weight >= result[-1][0]-1e-10
        result.append((weight, current))


def periodic_demand(entries, demand, context, *, period=7):
    """Exact weekly-equivalent cost with the same exponential pair/self formula."""
    vectors = [(d, tuple(q.estimate/s for q, s in zip(demand(role, miles), context.scales)))
               for d, role, miles in entries]
    memory = context.scenario.memory_days
    axes = []
    for axis in range(3):
        value = 0.
        for day, q in vectors:
            prior = sum(other[axis]*exp(-(((day-d)%period or period)-context.scenario.timing_compression_days)/memory)
                        /(-expm1(-period/memory)) for d, other in vectors)
            value += .5*q[axis]**2+q[axis]*prior
        axes.append(value*7/period)
    weights = context.scenario.axis_weights
    fatigue = context.exchange*sum(w*v for w, v in zip(weights, axes))/sum(weights)
    short = sum(context.easy_duration_policy.penalty(demand(role, m)[0].estimate)
                for _, role, m in entries if role == 'easy')*7/period
    return dict(fatigue=fatigue, short=short, demand_by_axis=dict(zip(AXES, axes)))


def periodic_recovery_audit(entries, demand, context):
    """Decompose the existing score; residue is an index, NOT tissue recovery.

    All scalar values use the existing exchange/axis weights. No added recovery
    penalty, threshold, benefit or different decay. Continuous residue excludes
    the timing-compression multiplier; its integral identity holds at zero
    compression (the nominal contract), not for arbitrary compressed timing.
    """
    entries = tuple(sorted(entries))
    if not entries or any(not 0 <= d < 7 for d, _, _ in entries) or len({d for d, _, _ in entries}) != len(entries):
        raise ValueError('Unique dates in one nonempty seven-day cycle required')
    vectors = [(d, role, tuple(v.estimate/s for v, s in zip(demand(role, miles), context.scales)))
               for d, role, miles in entries]
    memory = context.scenario.memory_days
    denominator = -expm1(-7/memory)
    compression = exp(context.scenario.timing_compression_days/memory)
    weights = context.scenario.axis_weights
    def weighted(values):
        return context.exchange*sum(w*v for w, v in zip(weights, values))/sum(weights)
    def residue(day, *, before):
        # Before a session excludes today's impulse, but includes previous cycles.
        return tuple(sum(q[a]*exp(-(((day-d)%7 or 7) if before else (day-d)%7)/memory)
                         /denominator for d, _, q in vectors) for a in range(3))
    terms = {k:[0., 0., 0.] for k in ('self', 'easy_to_easy', 'easy_to_key', 'key_to_easy', 'key_to_key')}
    keys = []
    for d, role, q in vectors:
        for a in range(3):
            terms['self'][a] += .5*q[a]**2
        for prior_day, prior_role, prior_q in vectors:
            group = ('easy' if prior_role == 'easy' else 'key')+'_to_'+('easy' if role == 'easy' else 'key')
            kernel = exp(-((d-prior_day)%7 or 7)/memory)/denominator*compression
            for a in range(3):
                terms[group][a] += q[a]*prior_q[a]*kernel
        if role != 'easy':
            before = residue(d, before=True)
            keys.append(dict(day=d, role=role, preceding_by_axis=dict(zip(AXES, before)),
                overlap_cost=weighted(tuple(x*y*compression for x, y in zip(q, before))),
                easy_miles_next_24h=sum(m for t, r, m in entries if r == 'easy' and (t-d)%7 == 1),
                easy_miles_next_48h=sum(m for t, r, m in entries if r == 'easy' and 0 < (t-d)%7 <= 2)))
    integral = [0., 0., 0.]
    for i, (d, _, _) in enumerate(vectors):
        gap = (vectors[(i+1)%len(vectors)][0]-d)%7 or 7
        after = residue(d, before=False)
        for a in range(3):
            integral[a] += after[a]**2*memory/2*(-expm1(-2*gap/memory))
    mean = tuple(memory*sum(q[a] for _, _, q in vectors)/7 for a in range(3))
    mean_square = tuple(v/7 for v in integral)
    score = periodic_demand(entries, demand, context)
    costs = {k:weighted(v) for k, v in terms.items()}
    assert abs(sum(costs.values())-score['fatigue']) < 1e-12
    integral_cost = weighted(tuple(v/memory for v in integral))
    if context.scenario.timing_compression_days == 0:
        assert abs(integral_cost-score['fatigue']) < 1e-12
    return dict(score=score, fatigue_components=costs,
        raw_fatigue_components_by_axis={k:dict(zip(AXES, v)) for k, v in terms.items()}, key_sessions=keys,
        mean_residue_by_axis=dict(zip(AXES, mean)),
        variance_residue_by_axis=dict(zip(AXES, (max(0., s-m*m) for m, s in zip(mean, mean_square)))),
        minimum_pre_session_residue_by_axis={a:min(residue(d, before=True)[i] for d, _, _ in vectors) for i, a in enumerate(AXES)},
        maximum_post_session_residue_by_axis={a:max(residue(d, before=False)[i] for d, _, _ in vectors) for i, a in enumerate(AXES)},
        squared_residue_integral_cost=integral_cost,
        integral_matches_score=context.scenario.timing_compression_days == 0,
        scope='Stationary demand indices; no physiological clearance, adaptation prediction or rest benefit')


def audit_profile_recovery(case, panel, *, compression_sweep=False):
    """Reuse the original sampled distributions, including the easy-only control.

    Reoptimization is only within the existing dose families and fixed key dates.
    Common-history transition scores remain in the main report, not replaced by
    these independently labelled stationary component controls.
    """
    base, hs, catalog = case['state'].context, case['handoffs'], case['catalog']
    key = min((k for k in hs if catalog[k].role == 'easy'), key=lambda k:abs(catalog[k].miles-4.4))
    rate = hs[key]['cardio']['estimate']/hs[key]['running_minutes']['estimate']
    reports = []; evaluations = 0
    for row in panel['rows']:
        p = Profile(**row['profile'])
        if p.long:
            c, demand, cycle, _, _ = fixture(case, p, row['scales'],
                single_volume=panel.get('assumptions', {}).get('single_volume', False))
        else:
            # No fake key sessions or attempt to promote this to the full contract.
            c = replace(base, scales=tuple(row['scales']))
            def demand(role, miles):
                return tuple(Quantity.point(v, 'original synthetic easy-only point rate')
                             for v in (miles*p.pace, miles*p.pace*rate, 0.))
            def cycle(pattern):
                return tuple((d, 'easy', m) for d, m in pattern)
        sampled = []
        for pattern in patterns(p):
            score = periodic_demand(cycle(pattern), demand, c)
            evaluations += 1
            sampled.append((score['fatigue']+score['short'], pattern))
        def summarize(item):
            cost, pattern = item
            audit = periodic_recovery_audit(cycle(pattern), demand, c)
            assert abs(cost-audit['score']['fatigue']-audit['score']['short']) < 1e-12
            concentration = easy_concentration(pattern)
            return dict(pattern=pattern, total_runs=len(pattern)+(2 if p.long else 0),
                cost=cost, easy_concentration=concentration,
                effective_easy_sessions=1/concentration,
                easy_exceeds_long=bool(p.long and max(m for _, m in pattern)>p.long+1e-9), **audit)
        best = min(sampled)
        counts = sorted({len(pattern) for _, pattern in sampled})
        finalists = [summarize(min(item for item in sampled if len(item[1]) == n)) for n in counts]
        winner = summarize(best)
        # Counterfactual subsets, not added constraints or physiological thresholds.
        # Use the original generator's rounded ~25-minute anchor. A 2.3-mile
        # command at 11 min/mi is 25.3 min, not a reason to require omission.
        light_controls = {}
        bounded_finalists = []
        if p.long:
            bounded = [item for item in sampled if max(m for _, m in item[1]) <= p.long+1e-9]
            bounded_finalists = [summarize(min(item for item in bounded if len(item[1]) == n))
                                for n in counts if any(len(item[1]) == n for item in bounded)]
            light_distance = round(c.easy_duration_policy.penalty_free_minutes/p.pace*10)/10
            for name, days in (('lighter_after_quality', (1,)), ('lighter_after_long', (5,)),
                               ('lighter_after_both', (1, 5))):
                eligible = [item for item in sampled if all(
                    dict(item[1]).get(day, 0.) <= light_distance+1e-9 for day in days)]
                light_controls[name] = summarize(min(eligible)) if eligible else None
                bounded_light = [item for item in eligible if max(m for _, m in item[1]) <= p.long+1e-9]
                light_controls[name+'_bounded_by_long_reference'] = summarize(min(bounded_light)) if bounded_light else None
        for alternative in (*finalists, *bounded_finalists, *(v for v in light_controls.values() if v)):
            alternative['delta_cost'] = alternative['cost']-winner['cost']
            alternative['fatigue_component_deltas'] = {k:v-winner['fatigue_components'][k]
                                                       for k, v in alternative['fatigue_components'].items()}
            gain = alternative['easy_concentration']-winner['easy_concentration']
            alternative['concentration_gain'] = gain
            alternative['marginal_fatigue_per_concentration_gain'] = (
                (alternative['score']['fatigue']-winner['score']['fatigue'])/gain if gain>1e-12 else None)
        result = dict(profile=p.name, candidates=len(sampled), winner=winner,
                            count_finalists=finalists, bounded_count_finalists=bounded_finalists,
                            light_controls=light_controls)
        if compression_sweep:
            def contrast(item, weight):
                r = summarize(item)
                r.update(weight=weight, rewarded_cost=r['cost']-weight*r['easy_concentration'],
                    fatigue_delta=r['score']['fatigue']-winner['score']['fatigue'],
                    short_delta=r['score']['short']-winner['score']['short'],
                    baseline_cost_delta=r['cost']-winner['cost'],
                    largest_easy=max(m for _, m in item[1]))
                return r
            pools = {'unrestricted':sampled}
            if p.long:
                pools['easy_not_above_long_reference'] = bounded
            result['compression_sweep'] = {name:dict(
                winners=[contrast(compression_winner(pool, w), w) for w in COMPRESSION_WEIGHTS],
                best_by_count=[dict(weight=w, finalists=[contrast(compression_winner(
                    [item for item in pool if len(item[1]) == n], w), w)
                    for n in sorted({len(pattern) for _, pattern in pool})]) for w in COMPRESSION_WEIGHTS],
                switches=[contrast(item, w) for w, item in compression_switches(pool)])
                for name, pool in pools.items()}
        reports.append(result)
    return dict(profiles=reports, analytic_evaluations=evaluations,
        assumptions=['Original profiles, point rates, normalization, memory, weights and sampled dose families unchanged',
            'Easy concentration is sum(miles squared)/(supporting miles squared); descriptive, not a new reward',
            'Lighter-day subsets use existing rounded approximately 25-minute generator anchor; omission allowed',
            'Bounded comparators keep nominal easy no larger than fixed long reference; diagnostic only, not new production eligibility',
            'All long/quality doses and dates fixed; equal volume, stationary zero deficit; no daily simulation',
            'Low-volume remains easy-only component control, not full admission/progression',
            'No additional score term or training-effectiveness inference'])


def explicit_context(context, days):
    if days <= 21 or days % 7:
        raise ValueError('Use full recurring cycles beyond the hard window')
    return replace(context, end=context.decision+timedelta(days=days),
                   continuation_guard_days=0, continuation_boundary_references=(), soft_horizon_taper=False)


def fixture(case, profile, scales, *, single_volume=False):
    ctx, hs, catalog = case['state'].context, case['handoffs'], case['catalog']
    key = min((k for k in hs if catalog[k].role == 'easy'), key=lambda k:abs(catalog[k].miles-4.4))
    rate = hs[key]['cardio']['estimate']/hs[key]['running_minutes']['estimate']
    quality_rate = hs['quality:ordinary']['cardio']['estimate']/hs['quality:ordinary']['running_minutes']['estimate']
    p = profile
    def demand(role, miles):
        minutes = miles*p.pace
        return tuple(Quantity.point(v, 'synthetic point rate; not observed athlete data') for v in (
            minutes, minutes*(quality_rate if role == 'quality' else rate), p.work if role == 'quality' else 0.))
    def cycle(pattern):
        return tuple(sorted((*((d, 'easy', m) for d, m in pattern),
                             (0, 'quality', p.quality), (4, 'long', p.long))))
    def repeat(entries, start, end, source, phase=0):
        return tuple(TrialSession(f'{source}:{n}', ctx.decision+timedelta(days=n), m, role,
                                  demand(role, m), float(role == 'quality'), source)
                     for n in range(start, end) for d, role, m in entries if (n-phase)%7 == d)
    history_days = {2:(2, 6), 3:(1, 3, 6), 4:(1, 2, 3, 6), 5:(1, 2, 3, 5, 6)}[p.history_easy_count]
    balanced = min(allocations(round(p.support*10), p.history_easy_count, p.pace), key=lambda d:max(d)-min(d))
    historical = cycle(tuple(zip(history_days, (n/10 for n in balanced))))
    policy = replace(ctx.policy, weekly_miles=p.volume, ordinary_miles=p.support/p.history_easy_count,
                     long_miles=p.long, quality_work_minutes=p.work,
                     purpose_weights=(1., 0., 1., 1.) if single_volume else ctx.policy.purpose_weights)
    ctx = replace(ctx, history=repeat(historical, -56, 0, 'observed'), stage_start=ctx.decision,
        scales=tuple(scales), policy=policy, volume_targets=(), support_targets=(), opening_coverage=(),
        easy_duration_policy=replace(ctx.easy_duration_policy, supporting_easy_miles_per_7=p.support),
        continuation_boundary_references=tuple((role, m, demand(role, m)) for role, m in (
            ('easy', policy.ordinary_miles), ('long', p.long), ('quality', p.quality))))
    return ctx, demand, cycle, repeat, historical


def run(root, panel_path, output, *, recovery_audit=False, compression_sweep=False):
    case = load_decision(root, 6)
    panel = json.loads(panel_path.read_text())
    single_volume = panel.get('assumptions', {}).get('single_volume', False)
    volume_linear = panel.get('assumptions', {}).get('volume_linear', 0.)
    output.mkdir(exist_ok=False)
    manifest = freeze_sources(output/'sources.zip', config=load_config(), entrypoint=__name__,
        inputs=(panel_path, root/'decision-0006.json', root/case['event']['snapshot_file'], root/'scenario.json'))
    stamp = perf_counter(); calls = 0; reports = []
    recovery = audit_profile_recovery(case, panel, compression_sweep=compression_sweep) if recovery_audit or compression_sweep else None
    recovery_by_name = {r['profile']:r for r in recovery['profiles']} if recovery else {}
    for row in panel['rows']:
        if row['contract'] != 'guarded_shared_score':
            continue  # Do not silently promote the separate low-volume control.
        p = Profile(**row['profile'])
        ctx, demand, cycle, repeat, historical = fixture(case, p, row['scales'], single_volume=single_volume)
        def evaluate(pattern, days=54, phase=0, matched=False, deficit=0.):
            nonlocal calls
            calls += 1
            c = ctx if days == 54 else explicit_context(ctx, days)
            c = replace(c, history=repeat(cycle(pattern) if matched else historical, -140 if matched else -56,
                                          0, 'observed', phase))
            physical = repeat(cycle(pattern), 0, days, 'planned', phase)
            if deficit:
                # Negative control: delete volume, NOT merely redistribute it.
                physical = tuple(replace(r, miles=round(r.miles-deficit, 1),
                    demand=demand('easy', round(r.miles-deficit, 1))) if r.role == 'easy' else r for r in physical)
            e = evaluate_program(c, ProgramCandidate(tuple(r for r in physical if r.day < c.hard_end),
                tuple(r for r in physical if r.day >= c.hard_end), 'explicit declared recurring programme'))
            assert abs(e.total_cost-e.delivery_cost-e.weighted_demand_cost-e.short_run_cost) < 1e-10
            curve = volume_benefit_cost(c, e, linear_coefficient=volume_linear) if single_volume else None
            return dict(cost=curve['cost'] if curve else e.total_cost,
                        fatigue=e.weighted_demand_cost, delivery=curve['delivery_cost'] if curve else e.delivery_cost,
                        short=e.short_run_cost, losses=e.parts['shortfall_by_purpose'])
        finalists = []
        for saved in row['finalists']:
            pattern = saved['pattern']
            old = evaluate(pattern)
            for key in ('cost', 'fatigue', 'delivery', 'short'):
                assert abs(old[key]-saved[key]) < 1e-12, (p.name, key)
            explicit = {str(days):evaluate(pattern, days) for days in (56, 112, 224)}
            periodic = periodic_demand(cycle(pattern), demand, ctx)
            witness = evaluate(pattern, 56, matched=True)
            assert abs(periodic['fatigue']-witness['fatigue']) < 1e-12
            assert abs(periodic['short']-witness['short']) < 1e-12
            # Exact-volume floating arithmetic can leave a ~1e-17 linear loss.
            assert witness['delivery'] < (1e-12 if single_volume else 1e-24)
            periodic = dict(periodic, delivery=witness['delivery'], cost=periodic['fatigue']+periodic['short'])
            phases = [evaluate(pattern, 112, phase) for phase in range(7)]
            finalists.append(dict(pattern=pattern, easy_count=len(pattern), original=old, explicit=explicit,
                                  periodic=periodic, explicit_phase_costs=[r['cost'] for r in phases]))
        winner = min(finalists, key=lambda r:r['explicit']['112']['cost'])
        negative = evaluate(winner['pattern'], 112, deficit=.5)
        assert negative['delivery'] > winner['explicit']['112']['delivery']
        extras = []
        if p.volume == 60:
            # Same total volume, one ~25-minute easy day, no recovery credit.
            anchor = round(25/p.pace, 1)
            for pattern in patterns(p):
                if len(pattern) == 5 and any(abs(m-anchor) < 1e-9 for _, m in pattern):
                    extras.append(dict(pattern=pattern, score=evaluate(pattern, 112),
                                       minutes=[round(m*p.pace, 1) for _, m in pattern]))
        result = dict(profile=p.name, finalists=finalists,
            original_winner=min(finalists, key=lambda r:r['original']['cost'])['easy_count'],
            explicit_winners={str(days):min(finalists, key=lambda r:r['explicit'][str(days)]['cost'])['easy_count']
                              for days in (56,112,224)},
            periodic_winner=min(finalists, key=lambda r:r['periodic']['cost'])['easy_count'],
            explicit_phase_winners=[min(finalists, key=lambda r:r['explicit_phase_costs'][i])['easy_count'] for i in range(7)],
            underfunding_control=dict(removed_miles_per_7=.5*len(winner['pattern']), score=negative,
                                      funded=winner['explicit']['112']),
            light_day_best=min(extras, key=lambda r:r['score']['cost']) if extras else None)
        if compression_sweep:
            # Reprice the union of stationary sweep winners, NOT all finite
            # candidates. Distinguish diagnostic subset winners from full search.
            selected = {tuple(tuple(x) for x in r['pattern'])
                        for pool in recovery_by_name[p.name]['compression_sweep'].values()
                        for r in pool['winners']}
            prices = {pattern:{str(days):evaluate(pattern, days) for days in (54, 112)}
                      for pattern in sorted(selected)}
            controls = {}
            for pattern in sorted(selected):
                removed = min(pattern, key=lambda x:(x[1], x[0]))
                missing = tuple(x for x in pattern if x != removed)
                cut = tuple((d, round(m-.1, 1) if d == removed[0] else m) for d, m in pattern)
                for days in (54, 112):
                    controls[pattern, days] = (removed, missing, evaluate(missing, days), cut, evaluate(cut, days))
            checks = []
            for w in COMPRESSION_WEIGHTS:
                windows = {}
                for days in (54, 112):
                    def cost(pattern):
                        return prices[pattern][str(days)]['cost']-w*easy_concentration(pattern)
                    chosen = min(selected, key=lambda pattern:(cost(pattern), pattern))
                    removed, missing, missing_price, cut, cut_price = controls[chosen, days]
                    windows[str(days)] = dict(pattern=chosen, total_runs=len(chosen)+2,
                        score=prices[chosen][str(days)], rewarded_cost=cost(chosen),
                        omission_control=dict(removed=removed, remaining_pattern=missing, score=missing_price,
                            concentration_gain=easy_concentration(missing)-easy_concentration(chosen),
                            rewarded_cost_delta=missing_price['cost']-w*easy_concentration(missing)-cost(chosen)),
                        small_cut_control=dict(removed_miles_per_7=.1, remaining_pattern=cut, score=cut_price,
                            concentration_gain=easy_concentration(cut)-easy_concentration(chosen),
                            baseline_cost_delta=cut_price['cost']-prices[chosen][str(days)]['cost'],
                            rewarded_cost_delta=cut_price['cost']-w*easy_concentration(cut)-cost(chosen)))
                checks.append(dict(weight=w, windows=windows))
            result['compression_transition_checks'] = dict(selected_candidates=len(selected), weights=checks,
                scope='Common-history guarded 54 and explicit 112 days; repriced stationary winners only, not finite global search. Omission is an adverse control, not an admitted prescription.')
        reports.append(result)
        print(json.dumps({k:v for k,v in result.items() if k in (
            'profile', 'original_winner', 'explicit_winners', 'periodic_winner', 'explicit_phase_winners')}), flush=True)
    report = dict(manifest=manifest, evaluations=calls, seconds=perf_counter()-stamp, profiles=reports,
        assumptions=dict(single_volume=single_volume, volume_linear=volume_linear),
        scope='Fixed synthetic finalists; explicit recurring windows and exact periodic demand. Not online search or daily stability.',
        limitations=['Different evaluation contracts: compare rankings/components, not absolute old/new scores',
            'Finite explicit windows average a declared ongoing programme, not future rest or a new terminal value',
            'No physiological feasibility, frequency adaptation, doubles, or reoptimization of finalist doses',
            'Synthetic normalization unchanged; low-volume unsupported full contract remains open'])
    if recovery_audit or compression_sweep:
        report['recovery_audit'] = recovery
    if compression_sweep:
        report['compression_policy_trial'] = dict(weights=COMPRESSION_WEIGHTS,
            formula='baseline cost - weight * sum(easy miles squared)/(total easy miles squared)',
            scope='Fixed equal-volume sampled cycles; transparent utility assumption, not physiological recovery benefit or rest requirement',
            bounded_subset='Easy <= long reference is diagnostic only; no new production limit',
            scale_invariance_warning='Uniformly cutting easy miles leaves the reward unchanged; deleting a small easy can increase it',
            online_status='Not wired into production, search, persisted policy, or daily harness')
    (output/'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(evaluations=calls, seconds=report['seconds'])))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('output/memory-continuation-daily-2-20260929'))
    parser.add_argument('--panel', type=Path, default=Path('output/easy-profile-distribution-20260929-c/report.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--recovery-audit', action='store_true')
    parser.add_argument('--compression-sweep', action='store_true')
    parser.add_argument('--calendar-root', type=Path)
    parser.add_argument('--calendar-index', type=int, default=1)
    parser.add_argument('--stationary-allocations', action='store_true')
    parser.add_argument('--variable-cycles', action='store_true')
    parser.add_argument('--actual-plan-audit', action='store_true')
    args = parser.parse_args()
    if args.calendar_root:
        if args.actual_plan_audit:
            run_actual_plan_audit(args.calendar_root,args.output,index=args.calendar_index)
        elif args.variable_cycles:
            run_variable_cycle_probe(args.calendar_root,args.output,index=args.calendar_index)
        elif args.stationary_allocations:
            run_stationary_allocation_probe(args.calendar_root, args.output, index=args.calendar_index)
        else:
            run_calendar_probe(args.calendar_root, args.output, index=args.calendar_index)
    else:
        run(args.root, args.panel, args.output, recovery_audit=args.recovery_audit, compression_sweep=args.compression_sweep)
