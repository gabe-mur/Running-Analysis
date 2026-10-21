"""HR-backed whole-program sensitivity audit; no live weights or calendar search.

Keep the saved 21-day calendar, 28-day continuation, and all legacy diagnostics.
Compare fixed-dose adjacent-date alternatives using recorded-HR transformations.
Report recovery/non-recovery tradeoffs without choosing an exchange-rate weight.
"""
from argparse import ArgumentParser
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import perf_counter
import json

import yaml

from run_analysis.config import load_config
from run_analysis.recovery_assessment import assess_next_session
from run_analysis.workload_accounting import Quantity
from run_analysis.workload_profile import prescribed_profile
from run_analysis.web.schemas import RecommendationResponse, WorkoutStep
from scripts.audit_backward_controls import digest, plain
from scripts.audit_recovery_full_score import panel
from scripts.audit_recovery_placement import audit
from scripts.audit_sandbox_recovery_adapter import policies
from scripts.hr_backed_sandbox import choose_donor, load_bank, transformed_entry
from scripts.typed_continuation_prototype import runs
from scripts.expected_completion import ExpectedCompletion
from scripts.volume_path_prototype import volume_path
from run_analysis.web.schemas import WeeklyScheduleDay


TIMINGS = {'nominal': (0,), 'alternating_early': (-6, 6), 'alternating_late': (6, -6)}
# Historical incomplete boundary retained as an ablation, not the new ownership
# rule. ownership_split removes the entire recovery bundle and mixed ratio;
# the separate full-path target objective retains weeks-scale volume ownership.
REPLACED = ('immediate', 'opening_compression', 'three_session_compression',
            'short_term_bridge', 'soft_continuation')
RECOVERY_COMPONENTS = (*REPLACED, 'continuous_short', 'continuous_slow')
COACHING_COMPONENTS = ('rest_selection', 'recovery_substitution', 'readiness',
    'recovery_interaction', 'weather', 'load_ratio', 'mileage_path', 'cadence_idle', 'key_session_cadence')


def ownership_split(legacy):
    """Explicit diagnostic boundary; unknown/mixed causes cannot silently pass.

    New acute demand owns all old recovery costs. Existing full-path target
    accounting owns sustained volume; the generic max(short,slow) ratio charge
    is not a second volume/recovery owner. Health/weather readiness provenance
    still needs separate integration when its generic label carries a cost.
    """
    recovery, coaching = legacy['recovery_components'], legacy['coaching_components']
    if set(recovery) != set(RECOVERY_COMPONENTS) or set(coaching) != set(COACHING_COMPONENTS):
        raise ValueError('Unmapped legacy component; update explicit ownership before comparison')
    for name, parts in (('recovery',recovery),('coaching',coaching)):
        if abs(sum(parts.values())-legacy['components'][name])>1e-8:
            raise ValueError(f'{name} decomposition does not reproduce the original score')
    initial=sum(recovery[k] for k in REPLACED)
    removed=sum(recovery.values())+coaching['load_ratio']+coaching['recovery_interaction']
    unresolved={k:coaching[k] for k in ('rest_selection','recovery_substitution','readiness') if coaching[k]}
    # Finalized full-path scoring must not contain a second provisional mileage
    # path or calendar-rest preference. Refuse rather than quietly count twice.
    if coaching['mileage_path'] or coaching['cadence_idle']:
        raise ValueError('Provisional volume/calendar cost leaked into finalized ownership')
    return dict(initial_mapping_retained=legacy['cost']-initial,
        without_short_density=legacy['cost']-initial-recovery['continuous_short'],
        single_owner_retained=legacy['cost']-removed,
        removed_legacy_cost=removed, unresolved_readiness=unresolved,
        preserved_components={k:v for k,v in legacy['components'].items() if k not in ('recovery','coaching')},
        preserved_coaching={k:v for k,v in coaching.items() if k not in ('load_ratio','recovery_interaction')},
        eligible_for_score_comparison=not unresolved,
        production_eligibility='not_assessed; old guard results are comparison-only')


def recovery_controls(history, events, families, *, volume_assessor=None):
    """Predeclared counterfactual doses; no fitting policies to their outcomes."""
    origin=events[0][2].start.replace(hour=12,minute=0,second=0,microsecond=0)
    easy=next(e for _,role,e in events if role=='easy')
    long=next(e for _,role,e in events if role=='long')
    quality=next(e for _,role,e in events if role=='intervals' and e.work_minutes.low>=15)
    long_day=next(day for day,role,_ in events if role=='long')
    prefix=[row for row in events if row[0]<long_day]
    def repeat(entry, days, role, prefix):
        rows=[]
        for day in days:
            at=origin+timedelta(days=day)
            rows.append((day,role,replace(entry,key=f'{prefix}/{day}',start=at,end=at+(entry.end-entry.start))))
        return rows
    tail=[row for row in events if row[0]>=21]
    scenarios={
        # Preserve the lead-in to the actual first planned long. Starting a
        # second long on day 0 would accidentally change preceding long cadence
        # as well as the easy-after-long placement we intend to inspect.
        'ordinary_easy_after_long':prefix+repeat(long,[long_day],'long','ordinary')+
            repeat(easy,[long_day+1],'easy','ordinary'),
        'three_daily_easy':prefix+repeat(easy,range(long_day,long_day+3),'easy','easy'),
        'four_daily_long':prefix+repeat(long,range(long_day,long_day+4),'long','long'),
        'four_daily_quality':prefix+repeat(quality,range(long_day,long_day+4),'intervals','quality'),
        'same_work_compressed':[(i,role,replace(e,start=origin+timedelta(days=i),
            end=origin+timedelta(days=i)+(e.end-e.start)))
            for i,(_,role,e) in enumerate(row for row in events if row[0]<21)]+tail,
        'daily_easy_volume':repeat(easy,range(21),'easy','volume')+tail,
    }
    output={}
    families=dict(families)
    if 'two_channels' in families and 'three_channels' in families:
        def core(p):
            return (p.running_minutes_envelope,p.cardio_points_envelope,
                    p.running_half_life_hours,p.cardio_half_life_hours)
        cores={core(p) for p in families['two_channels']}
        families['three_channels_matched_core']=tuple(p for p in families['three_channels'] if core(p) in cores)
    cache={}
    for name,program in scenarios.items():
        by_family={}
        for family,group in families.items():
            statuses=Counter(); last_statuses=Counter(); counts=[]
            for p in group:
                paths=[]; final_sessions=[]
                for timing in TIMINGS:
                    key=(name,timing,p.running_half_life_hours,p.cardio_half_life_hours,p.work_half_life_hours)
                    if key not in cache: cache[key]=trace(history,program,p,timing=timing)
                    paths.append(score(cache[key],p))
                    final_sessions.append(score(cache[key][-1:],p))
                # A whole program is supported only if every timing path fits;
                # contradicted here means at least one unavoidable modeled
                # excess on EVERY tested path. Other cases remain uncertain.
                status=('model_supported' if all(x['full'][1]<=1e-8 for x in paths) else
                        'model_contradicted' if all(x['full'][0]>1e-8 for x in paths) else 'uncertain')
                statuses[status]+=1
                last_statuses['model_supported' if all(x['full'][1]<=1e-8 for x in final_sessions) else
                    'model_contradicted' if all(x['full'][0]>1e-8 for x in final_sessions) else 'uncertain']+=1
                counts.append(max(x['unsupported_sessions'] for x in paths))
            by_family[family]=dict(status_counts=dict(statuses),final_session_status_counts=dict(last_statuses),
                                  unsupported_session_range=(min(counts),max(counts)))
        output[name]=dict(families=by_family,
            hard_miles=sum(e.workload.distance_meters.low/1609.344 for day,_,e in program if day<21),
            hard_runs=sum(day<21 for day,_,_ in program),
            physical_program=[dict(day=day,role=role,minutes=e.workload.running_minutes.low,
                cardio=(e.workload.cardio_points.low,e.workload.cardio_points.high),
                work_minutes=e.work_minutes.low,donor=e.workload.structure['donor_key']) for day,role,e in program])
        if volume_assessor is not None and name in {'same_work_compressed','daily_easy_volume'}:
            output[name]['independent_volume']=volume_assessor(program)
    return output


def numeric_fixture(rec):
    """Declare missing numeric test intent before HR donor choice, not from HR."""
    if 'work_duration_missing' not in prescribed_profile(rec)['limitations']:
        return rec
    minutes = round(rec.planned_distance_miles * 11 * .25, 1)
    return rec.model_copy(deep=True, update={'structure': [
        s.model_copy(update={'duration_minutes': minutes})
        if s.phase == 'work' and s.duration_minutes is None and s.work_duration_minutes is None
        and s.work_duration_range_minutes is None else s for s in rec.structure]})


def scaled_entry(entry, factor):
    """Explicit workload stress test, NOT a physiological progression forecast."""
    def scale(q):
        return Quantity(q.low * factor, q.high * factor, basis='synthetic dose-stage sensitivity')
    w = entry.workload
    return replace(entry, end=entry.start+(entry.end-entry.start)*factor,
        workload=replace(w, distance_meters=scale(w.distance_meters),
            running_minutes=scale(w.running_minutes), cardio_points=scale(w.cardio_points)),
        work_minutes=scale(entry.work_minutes))


def trace(history, events, policy, *, timing='nominal', factor=1.):
    """Cacheable physical demand: budgets do not affect channel totals."""
    ledger = [scaled_entry(e, factor) for e in history]
    offsets = TIMINGS[timing]
    result = []
    for i, (day, role, original) in enumerate(events):
        event = scaled_entry(original, factor)
        offset = timedelta(hours=offsets[i % len(offsets)])
        event = replace(event, start=event.start+offset, end=event.end+offset)
        a = assess_next_session(ledger, event.workload, event.start, policy,
                                proposed_work_minutes=event.work_minutes)
        result.append(dict(day=day, role=role, totals={k: (v['total_low'], v['total_high'])
                                                    for k, v in a['channels'].items()}))
        ledger.append(event)
    return result


def score(trace_rows, policy, *, capacity_factor=1.):
    budgets = dict(running_minutes=policy.running_minutes_envelope,
                   cardio_points=policy.cardio_points_envelope)
    if policy.work_minutes_envelope is not None:
        budgets['work_minutes'] = policy.work_minutes_envelope
    costs = []
    for row in trace_rows:
        if any(row['totals'][k][1] is None for k in budgets):
            raise ValueError('Unbounded demand cannot become a finite ranked cost')
        bounds = tuple(max(max(0., row['totals'][k][i]/(b*capacity_factor)-1)
                           for k,b in budgets.items()) for i in (0,1))
        costs.append((row['day'], bounds))
    return {part: tuple(sum(q[i] for day,q in costs if predicate(day)) for i in (0,1))
            for part,predicate in [('hard',lambda d:d<21), ('tail',lambda d:d>=21),
                                   ('full',lambda d:True)]} | {
        'contradicted_sessions': sum(q[0]>1e-8 for _,q in costs),
        'unsupported_sessions': sum(q[1]>1e-8 for _,q in costs)}


def demand_delta(candidate, baseline):
    # Interval subtraction: do not assume missing HR errors cancel across plans.
    return candidate[0]-baseline[1], candidate[1]-baseline[0]


def tradeoff(other_delta, recovery_delta):
    lo,hi = recovery_delta
    if other_delta < -1e-8 and hi <= 1e-8:
        return 'candidate_dominates'
    if other_delta > 1e-8 and lo >= -1e-8:
        return 'baseline_dominates'
    if abs(other_delta)<=1e-8:
        return 'candidate_dominates' if hi < -1e-8 else 'baseline_dominates' if lo > 1e-8 else 'tie_or_uncertain'
    return 'weight_or_evidence_dependent'


def paired_panel(*args, bank, families):
    snapshot, hard, full, template, history, config, *_ = args
    origin = datetime.fromisoformat(snapshot['decision_start_date']).replace(tzinfo=timezone.utc)
    future_origin = datetime(2027,1,1,tzinfo=timezone.utc)
    shift = future_origin-origin
    # This first-frame audit has a known 84-day synthetic seed: 6x1-minute
    # quality. Do not guess missing work for later uploaded ProjectionRuns.
    if snapshot['trigger'] == 'post_upload' or len(history) != 36:
        raise ValueError('This audit requires the initial declared 84-day seed')
    seed = []
    for run in history:
        at = run.start_time+shift
        rec = RecommendationResponse(generated_at=at, fitness_state_as_of=at, planned_for=at,
            workout_type=run.workout_type, title='Original synthetic seed with recorded-HR donor',
            distance_range_miles=(run.distance_miles,run.distance_miles),
            confidence='low', readiness='ready', structure=[WorkoutStep(instruction='6x1 test seed',
                phase='work', repetitions=6, work_duration_minutes=1)]
            if run.workout_type.value=='intervals' else [])
        seed.append(transformed_entry(choose_donor(bank,rec),rec))
    evaluations = []
    trace_cache = {}
    def observe(program, legacy):
        events = []
        overrides = []
        for rec in runs(program):
            prepared = numeric_fixture(rec)
            if prepared != rec:
                overrides.append(str(rec.planned_for))
            at = rec.planned_for+shift
            prepared = prepared.model_copy(update={'planned_for':at})
            entry = transformed_entry(choose_donor(bank,prepared),prepared,source='projected')
            events.append(((rec.planned_for.date()-origin.date()).days,rec.workout_type.value,entry))
        costs = {}
        signature = digest(events)
        for family, group in families.items():
            rows = []
            for p in group:
                scenarios = []
                for timing in TIMINGS:
                    key = (signature, timing,p.running_half_life_hours,
                           p.cardio_half_life_hours,p.work_half_life_hours)
                    if key not in trace_cache:
                        trace_cache[key] = trace(seed,events,p,timing=timing)
                    scenarios.append(score(trace_cache[key],p))
                # Whole-path robust bound, not a sum of independently worst gaps.
                rows.append(dict(policy=p.name,**{part:tuple(max(r[part][i] for r in scenarios)
                    for i in (0,1)) for part in ('hard','tail','full')},
                    unsupported_sessions=max(r['unsupported_sessions'] for r in scenarios)))
            costs[family] = rows
        removed = sum(legacy['recovery_components'][k] for k in REPLACED)
        result = dict(policies=costs, removed_legacy_recovery=removed,
            ownership=ownership_split(legacy),
            retained_cost=legacy['cost']-removed, numeric_fixture_overrides=overrides,
            physical_program=[dict(day=d,role=r,miles=e.workload.distance_meters.low/1609.344,
                minutes=e.workload.running_minutes.low,cardio=asdict(e.workload.cardio_points),
                work_minutes=e.work_minutes.low,donor=e.workload.structure['donor_key']) for d,r,e in events])
        def physical(rows):
            return sorted(digest({k:v for k,v in row.items() if k!='day'}) for row in rows)
        if evaluations and physical(result['physical_program']) != physical(evaluations[0][1]['physical_program']):
            raise AssertionError('A fixed-dose date probe changed physical workload or HR donor')
        evaluations.append((events,result))
        return result
    result = panel(*args,candidate_observer=observe)
    baseline = result['baseline']['replacement_diagnostic']
    for probe in result['probes']:
        candidate = probe['replacement_diagnostic']
        delta = candidate['retained_cost']-baseline['retained_cost']
        candidate['retained_cost_delta'] = delta
        candidate['tradeoffs'] = {}
        candidate['single_owner_tradeoffs'] = {}
        for family, rows in candidate['policies'].items():
            paired = list(zip(rows,baseline['policies'][family]))
            bounds = [demand_delta(a['full'],b['full']) for a,b in paired]
            candidate['tradeoffs'][family] = dict(
                counts=dict(Counter(tradeoff(delta,d) for d in bounds)),
                full_recovery_delta_range=(min(d[0] for d in bounds),max(d[1] for d in bounds)),
                tail_changes_direction=sum(tradeoff(0,demand_delta(a['hard'],b['hard'])) !=
                    tradeoff(0,demand_delta(a['full'],b['full'])) for a,b in paired))
            clean_delta=candidate['ownership']['single_owner_retained']-baseline['ownership']['single_owner_retained']
            candidate['single_owner_tradeoffs'][family]=dict(retained_delta=clean_delta,
                counts=dict(Counter(tradeoff(clean_delta,d) for d in bounds)))
    events = evaluations[0][0]
    state_model=ExpectedCompletion(template,history,config)
    volume_states=[state_model.at(origin+timedelta(days=i),[]) for i in range(len(full))]
    volume_targets=args[6](runs(hard),len(full))
    def volume_control(program):
        by_day={day:(role,entry) for day,role,entry in program}
        days=[]
        for i,state in enumerate(volume_states):
            rec=None
            if i in by_day:
                role,entry=by_day[i]
                at=entry.start-shift
                miles=round(entry.workload.distance_meters.low/1609.344,1)
                rec=RecommendationResponse(generated_at=origin,fitness_state_as_of=at,planned_for=at,
                    workout_type=role,title='Fixed workload volume control',distance_target_miles=miles,
                    distance_range_miles=(miles,miles),confidence='low',readiness='ready')
            days.append(WeeklyScheduleDay(date=state.as_of.date(),recommendation=rec,
                planned_at=rec.planned_for if rec else None,day_role='easy_run' if rec else 'rest_day',rationale='control'))
        result=volume_path(days,volume_states,volume_targets,config)
        return {k:result[k] for k in ('cost','underload_squared_violation','overload_squared_violation')}
    stages = []
    for factor in (.75,1.,1.25,1.5,2.):
        for capacity_factor in sorted({1.,factor}):
            counts = {}
            for family, group in families.items():
                values = []
                for p in group:
                    paths = []
                    for timing in TIMINGS:
                        key=(factor,timing,p.running_half_life_hours,p.cardio_half_life_hours,p.work_half_life_hours)
                        if key not in trace_cache:
                            trace_cache[key]=trace(seed,events,p,timing=timing,factor=factor)
                        paths.append(score(trace_cache[key],p,capacity_factor=capacity_factor))
                    values.append(max(x['unsupported_sessions'] for x in paths))
                counts[family]=dict(fully_supported_policies=sum(x==0 for x in values),
                                    unsupported_session_range=(min(values),max(values)))
            stages.append(dict(dose_factor=factor,capacity_factor=capacity_factor,families=counts))
    result.update(stage_sensitivity=stages,trace_cache_entries=len(trace_cache),
        ownership_controls=recovery_controls(seed,events,families,volume_assessor=volume_control),
        baseline_independent_volume=volume_control(events),
        replacement_scope=list(REPLACED), history_sessions=len(seed),
        scope='Paired full-program diagnostic. Legacy full score reproduced; legacy gates reported, '
              'not used to filter single-owner comparisons or replacement demand controls. '
              'Synthetic HR is added to the same seed dates/distances/quality work; donor pace determines duration. '
              'No exchange-rate weight selected, no new recovery eligibility imposed, no search or stability claim. '
              'Stage scaling is a stress test, not a progression or capacity-learning algorithm. '
              '28-day saved tail included; replacement terminal closure after day 49 remains unspecified.')
    return result


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--journal',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): parser.error('Choose a new output path')
    started=perf_counter()
    config=load_config(Path('config.yaml'))
    bank=load_bank(Path(config['paths']['database']),config)
    before=digest(bank)
    families=policies()
    result=audit(args.journal,yaml.safe_load(Path('config.example.yaml').read_text()),{0},
        observer=lambda *values:paired_panel(*values,bank=bank,families=families))
    if digest(bank)!=before: raise AssertionError('Donor bank changed')
    result.update(source_bank_hash=before,source_bank_count=len(bank),
                  policy_counts={k:len(v) for k,v in families.items()},total_seconds=perf_counter()-started)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f: json.dump(plain(result),f,indent=2,allow_nan=False)
    print(f'Whole-program comparison completed in {result["total_seconds"]:.2f}s',flush=True)


if __name__=='__main__': main()
