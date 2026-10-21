"""Matched-volume offline controls using frozen original-file audit evidence."""
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path
from time import perf_counter
import json

from scripts.audit_recency_demand import json_ready
from run_analysis.development_report import ProgramOption
from run_analysis.easy_dose import AerobicMinimum, EasyDurationReference, assess_easy_dose
from run_analysis.program_constraints import assess_program_constraints
from run_analysis.program_delivery import ProgramExposure, TargetSegment, audit_program
from run_analysis.recency_demand import CHANNELS, DemandEvent, DemandTerm, compare_recency
from run_analysis.workload_accounting import Quantity


def exposures(count, origin):
    offsets = {2: [0], 4: [0], 8: [0,5], 12: [0,2,5], 16: [0,1,2,5], 20: [0,1,2,4,5]}
    if count not in offsets:
        raise ValueError('Undeclared panel count')
    days = [cycle*7+offset for cycle in range(4) for offset in offsets[count]
            if count != 2 or cycle % 2 == 0]
    base, remainder = divmod(176, count)
    rows = [ProgramExposure(f'easy:{day}', origin+timedelta(days=day),
            (base+(i < remainder))/10, 'easy', 'planned', (base+(i < remainder))/10)
            for i, day in enumerate(days)]
    for cycle in range(6):  # Four evaluated motifs plus two common tail motifs.
        for offset, miles, role in [(3,8.4,'long'), (6,4.5,'quality')]:
            day = cycle*7+offset
            rows.append(ProgramExposure(f'{role}:{day}', origin+timedelta(days=day), miles, role,
                'planned', 8.4 if role == 'long' else 2., 'threshold' if role == 'quality' else None,
                (300.,)*3 if role == 'quality' else None))
        if cycle >= 4:
            day = cycle*7
            rows.append(ProgramExposure(f'easy:{day}', origin+timedelta(days=day), 4.4, 'easy', 'planned', 4.4))
    return tuple(sorted(rows, key=lambda e: (e.day,e.key)))


def program_option(key, records, history, rates):
    """Shared physical exposure/demand adapter for the bounded offline panels."""
    unit = Quantity.point(1., 'prescription_unit')
    events = []
    for e in records:
        terms = [DemandTerm('miles','prescription_mile',e.miles,unit),
                 DemandTerm('work_minutes','prescription_work_minute',sum(e.work_bout_seconds or ())/60,unit)]
        terms += [DemandTerm(c,f'scenario:{e.role}:{c}',e.miles,
                  Quantity(**rates[e.role][c]['bounds']) if rates[e.role][c]['bounds'] else None)
                  for c in ('running_minutes','cardio')]
        events.append(DemandEvent(e.key,e.day,tuple(terms),history[0].accounting_key,
            e.role if e.role != 'quality' else 'threshold', e.work_bout_seconds,'planned'))
    return ProgramOption(str(key),tuple(records),tuple(events))


def panel(directory=Path('output')):
    started = perf_counter()
    names = ['aerobic-entry-panel-20260925.json', 'coaching-contract-panel-20260925.json',
             'recency-demand-panel-20260925.json']
    entry, contract, recency = [json.loads((directory/name).read_text()) for name in names]
    if len({r['as_of'] for r in (entry, contract, recency)}) != 1:
        raise ValueError('Frozen evidence snapshots must match')
    as_of = datetime.fromisoformat(entry['as_of'])
    origin = date(2026,9,25)
    paired = contract['cases']['easy_fraction_0.80']  # Pace only; no 80% reduction policy imported.
    pace = paired['source_minutes']/paired['source_miles']
    timing = entry['summaries']['84']['easy']['variants']['5min_80pct']['uncensored_sustained_entry_minutes']
    refs = {name: EasyDurationReference(as_of, timing[field], pace,
                f'frozen_original_entry_84d_{field};paired_pace:{paired["source_key"]}')
            for name,field in [('central','median'), ('upper_quartile','q75')]}
    policy = AerobicMinimum(5., 'user_selected_provisional_minimum_20260925_not_optimum')
    q = lambda value: Quantity(**value) if value is not None else None
    history = tuple(DemandEvent('actual:'+r['key'], date.fromisoformat(r['day']), tuple(
        DemandTerm(c, f'actual:{r["key"]}:{c}', 1., q(r['quantities'][c])) for c in CHANNELS),
        r['accounting_key'], r['role'], tuple(r['structure']) if r['structure'] else None, 'observed')
        for r in recency['sessions'])
    def option(count):
        return program_option(count, exposures(count, origin), history, recency['rates'])
    reference = option(4)
    reports = {}
    for count in (2,4,8,12,16,20):
        candidate = option(count)
        end = origin+timedelta(days=28)
        easy = [e for e in candidate.exposures if e.role == 'easy' and e.day < end]
        delivery = audit_program(candidate.exposures, [TargetSegment(origin,origin+timedelta(days=42),17.3)],
            coverage_start=origin,coverage_end=origin+timedelta(days=42),evaluation_starts=[origin],
            spans=(28,),coverage_complete=True)['intervals'][0]
        comparisons = []
        for target in candidate.demands:
            if target.purpose not in {'long','threshold'}:
                continue
            before = next(e for e in reference.demands if e.key == target.key)
            a = (*history, *(e for e in reference.demands if e.day < before.day))
            b = (*history, *(e for e in candidate.demands if e.day < target.day))
            comparison = compare_recency(a,before,b,target)
            comparisons.append(dict(key=target.key,day=target.day,tail=target.day >= end,
                order=comparison['recency_demand_order'],channels=comparison['channel_orders']))
        floors = {}
        for name,ref in refs.items():
            checks = [dict(key=e.key,**assess_easy_dose(e.miles,ref,policy)) for e in easy]
            floors[name] = dict(minimum_command_miles=checks[0]['minimum_command_miles'],
                passing_sessions=sum(c['passes_planned_minimum'] for c in checks),
                failing_keys=[c['key'] for c in checks if not c['passes_planned_minimum']],
                intended_block_minutes_range=[min(c['intended_aerobic_budget_minutes'] for c in checks),
                                              max(c['intended_aerobic_budget_minutes'] for c in checks)])
        constraints = assess_program_constraints(candidate.demands,history,as_of=as_of,plan_complete=True)
        reports[str(count)] = dict(easy_sessions=count,easy_miles=sum(e.miles for e in easy),
            easy_distance_range=[min(e.miles for e in easy),max(e.miles for e in easy)],
            easy_total_minutes_at_reference_pace=sum(e.miles*pace for e in easy),
            longest_easy_minutes_at_reference_pace=max(e.miles*pace for e in easy),
            delivery=delivery, floors=floors, checked_constraints=constraints['status'],
            receiving_session_comparisons=comparisons,
            main_order_counts=dict(Counter(c['order'] for c in comparisons if not c['tail'])),
            tail_order_counts=dict(Counter(c['order'] for c in comparisons if c['tail'])),
            selection=None, calendar=[dict(key=e.key,day=e.day,miles=e.miles,role=e.role) for e in candidate.exposures])
    return dict(as_of=as_of, source_artifacts=names, original_history_count=len(history),
        references=refs, policy=policy, cases=reports, elapsed_seconds=perf_counter()-started,
        planner_influence=False, recovery_cleared=False, optimal_calendar_established=False,
        limitations=['Fixed-count/date controls, not searched or recommended calendars',
            'Pace/startup phase-budget arithmetic does not predict actual zone-2 occupancy',
            'Passing the five-minute floor supplies no workout-value or frequency reward',
            'Recency uses frozen role-specific demand scenarios, not calibrated fatigue',
            'Total-volume equality is over the declared 28-day interval, not every rolling cutoff',
            'Common 14-day tail is a boundary check, not an infinite-horizon guarantee',
            'No current athlete readiness, progression approval or daily-replanning validation'])


if __name__ == '__main__':
    print(json.dumps(panel(),default=json_ready,allow_nan=False,indent=2))
