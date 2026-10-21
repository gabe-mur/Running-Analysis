"""Bounded offline suffix proposals, using current inputs only.

Reuse conditional soft rollout to propose dates, then rematerialize them through
the real hard allocator/timing options. Rollout distances/times are not issued.
No scoring weights, hard constraints or live hooks are changed.
"""
from dataclasses import asdict
from contextlib import contextmanager
from math import isfinite
from time import perf_counter

from run_analysis import weekly_schedule as planner
from run_analysis.aerobic_acceptance import assess_aerobic_exposure
from run_analysis.prescription_distance import UnrepresentableDistanceTarget
from scripts.audit_continuation_upload import hard_score, raw_state_lookup
from scripts.typed_continuation_prototype import rollout, runs


def diverse_seeds(record, *, prefix_days=4, limit=12):
    if prefix_days < 1 or limit < 1:
        raise ValueError('Positive prefix and seed budgets required')
    rows = sorted((r for r in record['evaluated'] if isfinite(r['total_cost'])
        and not r.get('acceptance', {}).get('rejected')
        and not r.get('unrepresentable_distance_target')),
        key=lambda r:(r['total_cost'],r['offsets']))
    selected, seen = [], set()
    for row in rows:
        physical_roles=tuple(tuple(item.split(':')[:3]) for item in row.get('allocation_trace',[])
            if ':' in item and int(item.split(':')[0]) < prefix_days)
        key = (tuple(i for i in row['offsets'] if i < prefix_days),physical_roles,
               tuple(sorted((int(k),v) for k,v in row.get('roles',{}).items()
                            if int(k) < prefix_days)))
        if key not in seen:
            selected.append(row)
            seen.add(key)
        if len(selected) == limit:
            break
    return selected


@contextmanager
def regrowth_finalist_policy():
    """Reserve up to two existing slots; mandatory search lanes still survive."""
    from scripts import continuation_replan as module
    original=module.shortlist
    def selected(record,*,limit):
        base=original(record,limit=limit)
        regrown=diverse_seeds(dict(evaluated=[r for r in record['evaluated']
            if r.get('proposal_source')=='suffix_regrowth']),limit=2)
        if not regrown:
            return base
        chosen,seen=[],set()
        def add(row):
            key=(tuple(row['offsets']),tuple(sorted((int(k),v) for k,v in row.get('roles',{}).items())))
            if key not in seen and len(chosen)<limit-1:
                chosen.append(row)
                seen.add(key)
        for lane in ('unconstrained','stable'):
            target=record.get(lane)
            if target is not None:
                add(next(r for r in base if r['offsets']==target['offsets']
                    and abs(r['total_cost']-target['total_cost'])<1e-8
                    and r['allocation_trace']==target['allocation_trace']))
        for row in regrown:
            add(row)
        for row in base:
            add(row)
        return chosen
    try:
        module.shortlist=selected
        yield
    finally:
        module.shortlist=original


@contextmanager
def regrowth_replan_policy():
    """Offline daily-runner hook; all proposed rows retained for exact replay."""
    from scripts import continuation_replan as module
    from run_analysis.selection_objective import COMPLETED_SELECTION_HISTORY
    original=module.select_program
    records=[]
    def select(planning_days,diagnostics,inputs,record,*,limit,continuation_days):
        evidence=COMPLETED_SELECTION_HISTORY.get()
        if evidence is None:
            raise ValueError('Regrowth requires current completed-history evidence')
        rows,_,metrics=propose(inputs,record,evidence)
        audit=dict(version='bounded_suffix_v1',rows=rows,metrics=metrics,
                   reserved_slots=2,finalist_budget=limit)
        records.append(audit)
        with regrowth_finalist_policy():
            result=original(planning_days,diagnostics,inputs,
                dict(record,evaluated=record['evaluated']+rows),limit=limit,
                continuation_days=continuation_days)
        result['evidence']['suffix_regrowth']=audit
        return result
    try:
        module.select_program=select
        yield records
    finally:
        module.select_program=original


def propose(inputs, record, evidence, *, cuts=(4,7), seed_limit=12):
    """Up to seed_limit * len(cuts) suffixes, not a recursive search."""
    started=perf_counter()
    horizon=len(inputs['daily_states'])
    if not cuts or any(not isinstance(c,int) or not 0 < c < horizon for c in cuts):
        raise ValueError('Cuts must lie strictly within the hard horizon')
    if len(set(cuts)) != len(cuts):
        raise ValueError('Duplicate cuts waste the proposal budget')
    if planner.configured_race_goal(inputs['config']) is not None:
        raise ValueError('Race-date regrowth is not supported by this prototype')
    blocked=set(inputs['forced_rest_offsets'] or ()) | set(inputs['completed_run_offsets'] or ())
    state_at=raw_state_lookup(inputs,evidence)
    origin=inputs['daily_states'][0].as_of.date()
    seeds=diverse_seeds(record,prefix_days=min(cuts),limit=seed_limit)
    rows, programs, attempts, seen = [], {}, [], set()
    for seed in seeds:
        seed_roles={int(k):v for k,v in seed.get('roles',{}).items()}
        days, price, _=hard_score(inputs,seed['offsets'],roles=seed_roles)
        # Never silently seed a different program than the saved current search.
        assert abs(price.total_cost-seed['total_cost']) < 1e-8
        assert price.model_dump(mode='json')['allocation_trace'] == seed['allocation_trace']
        for cut in cuts:
            entry=dict(seed_offsets=seed['offsets'],cut=cut)
            attempts.append(entry)
            prefix=days[:cut]
            prefix_roles={i:('long' if d.recommendation.workout_type==planner.WorkoutType.LONG
                else 'quality' if d.recommendation.workout_type in planner.QUALITY_WORKOUT_TYPES
                else 'easy') for i,d in enumerate(prefix) if d.recommendation
                and d.recommendation.workout_type != planner.WorkoutType.REST}
            if set(prefix_roles)&blocked:
                raise ValueError('Seed uses blocked date')
            try:
                targets=inputs['expected_target_projector'](runs(prefix),horizon)
                generated,_,_=rollout(prefix,inputs['daily_states'][:cut],inputs['request'],
                    inputs['config'],targets,state_at=state_at,continuation_days=horizon-cut,
                    distance_policy='finalized',role_overrides={
                        inputs['daily_states'][i].as_of.date():'rest' for i in blocked if cut<=i<horizon})
                offsets=tuple((r.planned_for.date()-origin).days for r in runs(generated))
                assert {i for i in offsets if i<cut} == set(prefix_roles)
                if set(offsets)&blocked:
                    raise ValueError('Regrowth generated a blocked date')
                key=(offsets,tuple(sorted(prefix_roles.items())))
                entry['offsets']=list(offsets)
                if key in seen:
                    entry['status']='duplicate'
                    continue
                seen.add(key)
                candidate, hard, details=hard_score(inputs,offsets,roles=prefix_roles)
                if details.get('unrepresentable_distance_target'):
                    entry.update(status='unrepresentable',reason=details['unrepresentable_distance_target'])
                    continue
                acceptance=assess_aerobic_exposure(**details['aerobic_acceptance_facts'],
                    planning_mode=inputs['target_evidence'].planning_mode.value)
                row=dict(hard.model_dump(mode='json'),roles=prefix_roles,acceptance=asdict(acceptance),
                         proposal_source='suffix_regrowth',regrowth_cut=cut)
                rows.append(row)
                programs[key]=candidate
                entry.update(status='rejected' if acceptance.rejected else 'accepted',hard_cost=hard.total_cost)
            except UnrepresentableDistanceTarget as error:
                entry.update(status='unrepresentable',reason=str(error))
    return rows,programs,dict(seconds=perf_counter()-started,seeds=len(seeds),
        proposal_budget=seed_limit*len(cuts),attempts=attempts,materialized=len(rows),
        accepted=sum(not r['acceptance']['rejected'] for r in rows))
