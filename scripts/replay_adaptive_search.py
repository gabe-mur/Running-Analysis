"""Re-run one archived decision with current search, without upload or publication.

Exact old-candidate repricing is mandatory; this is a frozen diagnostic, not a
daily adherence test. The input artifact and application database are untouched.
"""
from argparse import ArgumentParser
from dataclasses import replace
from datetime import date, datetime, time, timedelta
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from zoneinfo import ZoneInfo
from zipfile import ZipFile

from run_analysis.config import load_config
from run_analysis.continuous_training_dose import ContinuousDoseReference
from run_analysis.adaptive.decision_snapshot import migrate_snapshot_defaults, evaluate_snapshot
from run_analysis.adaptive.forecast_handoff import read_scope
from run_analysis.adaptive.new_model_daily_harness import Dose, planned_accounted, churn
from run_analysis.adaptive.continuous_search_adapter import search
from run_analysis.adaptive.stage_calendar_seeds import complete_stage_seeds
from run_analysis.adaptive.decimal_dose_search import calendar_representatives
from run_analysis.adaptive.continuous_program_adapter import evaluate_continuous
from run_analysis.adaptive.shared_program_evaluator import evaluate_program
from run_analysis.adaptive.program_objective_trial import TrialSession
from run_analysis.workload_accounting import Quantity
from scripts.research_manifest import freeze_sources


def load_decision(root, index):
    """Reconstruct one immutable decision and verify its archived winner."""
    event=json.loads((root/f'decision-{index:04d}.json').read_text())
    state=migrate_snapshot_defaults((root/event['snapshot_file']).read_text(),
        expected_fingerprint=event['snapshot_fingerprint'])['state']
    scenario=json.loads((root/'scenario.json').read_text())
    catalog={d['key']:Dose(**dict(d,phases=tuple(tuple(p) for p in d['phases']),
        phase_scenarios=tuple(tuple(p) for p in d['phase_scenarios']),
        forecast_scope=read_scope(d['forecast_scope']))) for d in
        event.get('stage_runtime',{}).get('catalog',scenario['catalog'])}
    dates=tuple((state.context.decision+timedelta(days=i)).isoformat()
        for i in range((state.context.end-state.context.decision).days))
    handoffs=event['forecast_handoff_by_dose']
    reference=ContinuousDoseReference('sandbox upload accounting')
    physical={(day,key):planned_accounted(f'candidate:{day}:{key}',
        datetime.combine(date.fromisoformat(day),time(18),ZoneInfo(state.timezone)),
        catalog[key],handoffs[key],reference,available=state.as_of)
        for day in dates for key in handoffs}
    def evaluate_details(program, *, state_override=None):
        return evaluate_snapshot(state_override or state,tuple(physical[p] for p in program),
            tuple(k for _,k in program),reference=reference)
    def evaluate(program, *, state_override=None):
        r=evaluate_details(program,state_override=state_override)
        return dict(cost=r['evaluation'].total_cost,eligible=r['decision_eligible'])
    old=evaluate(tuple(tuple(p) for p in event['program']))
    if not old['eligible'] or abs(old['cost']-event['cost'])>1e-12:
        raise ValueError('Archived winner does not reproduce; resolve accounting before search')
    return dict(event=event,state=state,catalog=catalog,dates=dates,
        handoffs=handoffs,evaluate=evaluate,evaluate_details=evaluate_details,
        physical=physical,archived_cost=old['cost'])


def replayed_prior(result_path, expected_event):
    """Only hand off a replay from the declared earlier frozen decision."""
    saved=json.loads(result_path.read_text())
    archive=Path(saved['manifest']['path'])
    if sha256(archive.read_bytes()).hexdigest()!=saved['manifest']['sha256']:
        raise ValueError('Prior replay source archive changed')
    with ZipFile(archive) as source:
        inputs=json.loads(source.read('manifest.json'))['input_sha256']
    decisions={Path(p).resolve():h for p,h in inputs.items() if Path(p).name.startswith('decision-')}
    expected=expected_event.resolve()
    if (not decisions or max(decisions,key=lambda p:p.name)!=expected or
            decisions.get(expected)!=sha256(expected_event.read_bytes()).hexdigest()):
        raise ValueError('Prior replay must originate at the specified earlier decision')
    return saved['program']


def tapered_case(case):
    """Explicit experiment after verifying the unmodified archived score."""
    state=replace(case['state'],context=replace(case['state'].context,soft_horizon_taper=True))
    def evaluate(program):
        return case['evaluate'](program,state_override=state)
    return dict(case,state=state,evaluate=evaluate)


def neutral_cadence_evaluator(context, candidate):
    """Remove observed key-session phase ONLY; not a repeating-plan constraint.

    Both purposes have full cadence credit one day before the decision. These
    zero-demand timing markers never enter the physical ledger or eligibility.
    This is an explicit shared-opening assumption, not measured completion.
    """
    zero = tuple(Quantity.point(0., 'diagnostic timing marker; not physical work') for _ in range(3))
    markers = tuple(TrialSession('neutral-opening:'+role,context.decision-timedelta(days=1),
        context.policy.long_miles if role=='long' else context.policy.ordinary_miles,role,zero,
        float(role=='quality'),'observed') for role in context.timing.purposes)
    return evaluate_program(context,candidate,timing_history=markers,timing_opening_coverage=())


def recurrence_summary(program, catalog, dates):
    """Measure recurrence after search, including rest, without enforcing it."""
    program=tuple((d,k) for d,k in program if d in set(dates))
    occupied=dict(program)
    def signature(day, doses):
        key=occupied.get(day)
        return (catalog[key].role,catalog[key].miles) if doses and key else catalog[key].role if key else 'rest'
    periods={}
    for lag in range(2,15):
        pairs=tuple(zip(dates,dates[lag:]))
        periods[str(lag)]=dict(comparisons=len(pairs),
            role_matches=sum(signature(a,False)==signature(b,False) for a,b in pairs),
            dose_matches=sum(signature(a,True)==signature(b,True) for a,b in pairs))
    gaps={role:[(date.fromisoformat(b)-date.fromisoformat(a)).days for a,b in zip(
        [d for d,k in program if catalog[k].role==role],
        [d for d,k in program if catalog[k].role==role][1:])] for role in ('long','quality')}
    return dict(periods=periods,key_gaps=gaps,
        exact_role_periods=[int(k) for k,v in periods.items() if v['role_matches']==v['comparisons']],
        exact_dose_periods=[int(k) for k,v in periods.items() if v['dose_matches']==v['comparisons']])


def aperiodic_starting_calendars(dates, programs):
    """Deterministic irregular starting candidates; never a scheduling rule.

    Low-discrepancy offsets vary individual session dates, not one repeated
    template's phase. Preserve every seed's exact dose inventory/total mileage.
    These are candidate-coverage probes; the original whole-program evaluator
    and unrestricted date/dose refiners still select the result. No RNG.
    """
    indices={d:i for i,d in enumerate(dates)}
    def radical_inverse(n,base):
        fraction=0.; scale=1.
        while n:
            n,digit=divmod(n,base); scale/=base; fraction+=digit*scale
        return fraction
    result=[]; seen=set()
    for variant,program in enumerate(programs):
        radius=1+variant%3; base=(2,3,5)[variant%3]
        occupied={}
        for rank,(day,key) in enumerate(sorted(program)):
            phase=(radical_inverse(rank+1,base)+(variant%17)/17)%1
            offset=int(phase*(2*radius+1))-radius
            desired=max(0,min(len(dates)-1,indices[day]+offset))
            free=(i for i in range(len(dates)) if i not in occupied)
            destination=min(free,key=lambda i:(abs(i-desired),i if variant%2 else -i))
            occupied[destination]=key
        alternative=tuple((dates[i],k) for i,k in sorted(occupied.items()))
        if alternative!=tuple(sorted(program)) and alternative not in seen:
            result.append(alternative); seen.add(alternative)
    return tuple(result)


def run_neutral_cadence(root, output, *, index=1, evaluations=10000, aperiodic_starts=False):
    """Same dated projector/search, independent dates/doses and selectable tail.

    No issued incumbent, fixed key dates, required periodicity or settled fake
    physical history. Retain actual readiness, volume history and dated ramp.
    """
    case=load_decision(root,index)  # Mandatory unchanged archive parity.
    output.mkdir(exist_ok=False)
    state,event,catalog,dates,handoffs=(case[k] for k in ('state','event','catalog','dates','handoffs'))
    manifest=freeze_sources(output/'sources.zip',config=load_config(),entrypoint=__name__,
        inputs=(root/f'decision-{index:04d}.json',root/event['snapshot_file'],root/'scenario.json'))
    reference=ContinuousDoseReference('sandbox upload accounting')
    physical={}; cache={}
    def connected(context, history, planned, **kwargs):
        return evaluate_continuous(context,history,planned,program_evaluator=neutral_cadence_evaluator,**kwargs)
    def detail(program, *, neutral=None):
        neutral=not aperiodic_starts if neutral is None else neutral
        program=tuple(sorted(program)); identity=(neutral,program)
        if identity not in cache:
            for day,key in program:
                if (day,key) not in physical:
                    physical[day,key]=planned_accounted(f'candidate:{day}:{key}',
                        datetime.combine(date.fromisoformat(day),time(18),ZoneInfo(state.timezone)),
                        catalog[key],handoffs[key],reference,available=state.as_of)
            cache[identity]=evaluate_snapshot(state,tuple(physical[p] for p in program),
                tuple(k for _,k in program),reference=reference,
                evaluator=connected if neutral else evaluate_continuous)
        return cache[identity]
    def evaluate(program):
        r=detail(program)
        return dict(cost=r['evaluation'].total_cost,eligible=r['decision_eligible'],
            construction_eligible=r['decision_eligible'])
    archived=tuple(tuple(p) for p in event['program'])
    unchanged=detail(archived,neutral=False)
    search_old=detail(archived)
    assert abs(unchanged['evaluation'].total_cost-case['archived_cost'])<1e-12
    assert search_old['decision_eligible']==unchanged['decision_eligible']
    assert search_old['evaluation'].weighted_demand_cost==unchanged['evaluation'].weighted_demand_cost
    assert search_old['evaluation'].short_run_cost==unchanged['evaluation'].short_run_cost
    supported={k:catalog[k] for k in handoffs}
    representatives=calendar_representatives(supported,ordinary=state.context.policy.ordinary_miles,
        support=state.context.easy_duration_policy.supporting_easy_miles_per_7)
    seed_keys=set()
    for target in (state.context.policy.ordinary_miles,state.context.policy.ordinary_miles/2):
        seed_keys.add(min((k for k in handoffs if catalog[k].role=='easy'),
            key=lambda k:(abs(catalog[k].miles-target),k)))
    for role in ('long','quality'):
        seed_keys.add(max((k for k in handoffs if catalog[k].role==role),key=lambda k:(
            sum(s for p,s in catalog[k].phases if p=='work') if role=='quality' else catalog[k].miles,k)))
    support=max(s.miles_per_7_days for s in state.context.support_targets)
    # Match the app caller's endpoint-aware funding seeds. They are starting
    # candidates only; every date/role/dose remains independently editable.
    seed_policy=replace(state.context.policy,weekly_miles=max(s.miles_per_7_days for s in state.context.volume_targets))
    starting_calendars=complete_stage_seeds(dates,supported,seed_policy,supporting_miles=support)
    if aperiodic_starts:
        starting_calendars=aperiodic_starting_calendars(dates,starting_calendars)
        # Known eligible benchmark is one nonprivileged candidate, never an
        # incumbent or a date lock. Do not report an inferior cold start as best.
        starting_calendars=(*starting_calendars,archived)
    stamp=perf_counter()
    result=search(dates=dates,keys=tuple(sorted(handoffs)),previous=(),evaluate=evaluate,
        budget=evaluations,hard_end=state.context.hard_end.isoformat(),
        roles={k:catalog[k].role for k in handoffs},seed_keys=tuple(sorted(seed_keys)),
        seed_programs=starting_calendars,
        dose_miles={k:catalog[k].miles for k in handoffs},exploration_keys=representatives)
    program=result['program']; selected=detail(program); actual_selected=detail(program,neutral=False)
    from scripts.check_explicit_distribution_tail import incoming_demand_trace
    from run_analysis.adaptive.new_model_daily_harness import training_metrics
    def describe(r):
        e=r['evaluation']
        return dict(cost=e.total_cost,fatigue=e.weighted_demand_cost,delivery=e.delivery_cost,
            short=e.short_run_cost,eligible=r['decision_eligible'],
            losses=e.parts['shortfall_by_purpose'])
    report=dict(manifest=manifest,source_snapshot=state.fingerprint,search=result,
        seconds=perf_counter()-stamp,as_of=state.as_of.isoformat(),decision=state.context.decision.isoformat(),
        hard_end=state.context.hard_end.isoformat(),end=state.context.end.isoformat(),
        archived_under_actual=describe(unchanged),archived_under_search=describe(search_old),
        selected_under_search=describe(selected),selected_under_actual=describe(actual_selected),
        opening_mode='real_observed_history' if aperiodic_starts else 'equal_diagnostic_cadence_markers',
        seed_mode='aperiodic_plus_one_unprivileged_benchmark' if aperiodic_starts else 'original_funded_seeds',
        starting_calendar_count=len(starting_calendars),
        hard_schedule=[dict(day=d,role=catalog[k].role,miles=catalog[k].miles,key=k) for d,k in program
            if d<state.context.hard_end.isoformat()],
        hard_recurrence=recurrence_summary(program,catalog,dates[:(state.context.hard_end-state.context.decision).days]),
        full_recurrence=recurrence_summary(program,catalog,dates),
        key_entry=[r for r in incoming_demand_trace(selected['evaluation'],state.context) if r['role']!='easy'],
        training_metrics=training_metrics([physical[p] for p in program],state.history,state.context.decision,
            (state.context.hard_end-state.context.decision).days,ZoneInfo(state.timezone)),
        assumptions=[('No artificial opening credit; all actual historical cadence and load retained' if aperiodic_starts else
            'Historical long/quality cadence credit replaced with equal full credit one day before decision; timing markers are not physical history'),
            'Actual physiological history, volume lead-in, readiness, gradual targets, forecasts and original whole-horizon boundary/taper retained',
            ('No issued-plan incumbent or continuity allowance; original winner is one unprivileged benchmark seed' if aperiodic_starts else
            'Cold search with no issued-plan incumbent or continuity allowance; original winner repriced for comparison only, not supplied as seed'),
            'All dates in hard 21 days and original scored continuation independently editable; no required cycle, run count, key date or weekly mileage quota',
            ('Deterministic irregular starts preserve seed dose inventories and do not impose recurring dates; no RNG' if aperiodic_starts else
            'Existing funded periodic seeds remain initial candidates only; search coverage is bounded, so a repeated winner is not global proof'),
            'Recurrence measured after selection, not encoded as an admission condition'],
        status='Offline diagnostic; no live DB, publication, default numerical policy or scheduler behavior change')
    with (output/('aperiodic-search-result.json' if aperiodic_starts else 'neutral-cadence-result.json')).open('x') as stream:
        json.dump(report,stream,indent=2,default=str)
    print(json.dumps({k:report[k] for k in ('seconds','archived_under_actual','archived_under_search',
        'selected_under_search','selected_under_actual','hard_schedule','hard_recurrence','training_metrics')},default=str),flush=True)
    return report


def run(root, output, *, earlier, later, evaluations=10000,frontier_repair=True,prefix_days=7,previous_result=None,soft_horizon_taper=False):
    output.mkdir(exist_ok=False)
    paths=[root/f'decision-{n:04d}.json' for n in (earlier,later)]
    prior=json.loads(paths[0].read_text())
    previous=replayed_prior(previous_result,paths[0]) if previous_result is not None else prior['program']
    case=load_decision(root,later)
    if soft_horizon_taper:
        case=tapered_case(case)
        if previous_result is not None and not json.loads(previous_result.read_text()).get('objective',{}).get('soft_horizon_taper'):
            raise ValueError('Causal taper experiment requires an earlier tapered replay')
    manifest=freeze_sources(output/'sources.zip',config=load_config(),
        entrypoint='scripts.replay_adaptive_search',inputs=(*paths,root/'scenario.json',
            root/prior['snapshot_file'],root/case['event']['snapshot_file'],
            *((previous_result,) if previous_result is not None else ())))
    event,state,catalog,dates,handoffs=(case[k] for k in ('event','state','catalog','dates','handoffs'))
    evaluate=case['evaluate']
    archived_candidate=evaluate(tuple(map(tuple,event['program'])))
    keys=set()
    for target in (state.context.policy.ordinary_miles,state.context.policy.ordinary_miles/2):
        keys.add(min((k for k in handoffs if catalog[k].role=='easy'),
            key=lambda k:(abs(catalog[k].miles-target),k)))
    for role in ('long','quality'):
        keys.add(max((k for k in handoffs if catalog[k].role==role),key=lambda k:(
            sum(s for p,s in catalog[k].phases if p=='work') if role=='quality' else catalog[k].miles,k)))
    result=search(dates=dates,keys=tuple(sorted(handoffs)),previous=previous,
        previous_end=prior['end'],evaluate=evaluate,budget=evaluations,
        frontier_repair=frontier_repair,
        stable_days=prefix_days,
        hard_end=state.context.hard_end.isoformat(),roles={k:catalog[k].role for k in handoffs},
        seed_keys=tuple(sorted(keys)),seed_programs=complete_stage_seeds(dates,catalog,state.context.policy,
            supporting_miles=state.context.easy_duration_policy.supporting_easy_miles_per_7))
    result.update(manifest=manifest,archived_cost=case['archived_cost'],frontier_repair=frontier_repair,prefix_days=prefix_days,
        archived_candidate_cost_under_search_objective=archived_candidate['cost'],
        objective=dict(soft_horizon_taper=state.context.soft_horizon_taper,
            explicit_taper_override=soft_horizon_taper),
        previous_plan_source=str(previous_result) if previous_result is not None else str(paths[0]),
        source_decision=str(paths[1]),snapshot_fingerprint=event['snapshot_fingerprint'],
        churn=churn(previous,result['program'],state.context.decision,catalog),
        scope='frozen decision current-search diagnostic; not daily lifecycle acceptance')
    with (output/'result.json').open('x') as stream:
        json.dump(result,stream,indent=2)
    print(json.dumps({k:result[k] for k in ('cost','archived_cost','seconds','evaluations',
        'local_neighborhood_complete','churn','prior_revalidation')}),flush=True)
    return result


if __name__=='__main__':
    p=ArgumentParser()
    p.add_argument('root',type=Path)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--events',type=int,nargs=2)
    p.add_argument('--neutral-cadence',action='store_true')
    p.add_argument('--aperiodic-starts',action='store_true')
    p.add_argument('--index',type=int,default=1)
    p.add_argument('--evaluations',type=int,default=10000)
    p.add_argument('--without-frontier-repair',action='store_true')
    p.add_argument('--prefix-days',type=int,default=7)
    p.add_argument('--previous-result',type=Path)
    p.add_argument('--soft-horizon-taper',action='store_true')
    a=p.parse_args()
    if a.neutral_cadence or a.aperiodic_starts:
        if a.neutral_cadence and a.aperiodic_starts:
            p.error('Aperiodic coverage check retains real history, not artificial neutral credits')
        if a.events or a.previous_result or a.soft_horizon_taper:
            p.error('Neutral cadence uses one original snapshot; do not mix replay overrides')
        run_neutral_cadence(a.root,a.output_dir,index=a.index,evaluations=a.evaluations,
            aperiodic_starts=a.aperiodic_starts)
    else:
        if not a.events:
            p.error('--events required for causal replay')
        run(a.root,a.output_dir,earlier=a.events[0],later=a.events[1],evaluations=a.evaluations,
            frontier_repair=not a.without_frontier_repair,prefix_days=a.prefix_days,previous_result=a.previous_result,
            soft_horizon_taper=a.soft_horizon_taper)
