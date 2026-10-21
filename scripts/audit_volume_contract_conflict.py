"""Frozen score contrasts and duplicate-date diagnosis; no planner repair/publication."""
from argparse import ArgumentParser
from dataclasses import asdict
from datetime import date, datetime, time, timedelta
from itertools import product
import json
from pathlib import Path
import shutil
import sqlite3
from zoneinfo import ZoneInfo

from run_analysis.config import load_config
from run_analysis.continuous_training_dose import ContinuousDoseReference
from run_analysis.adaptive.decision_snapshot import migrate_snapshot_defaults, evaluate_snapshot
from run_analysis.adaptive.forecast_handoff import read_scope
from run_analysis.adaptive.new_model_daily_harness import Dose, planned_accounted
from run_analysis.adaptive.continuous_search_adapter import search, NoEligibleProgram
from scripts.research_manifest import freeze_sources


def frozen_contrasts(root, index):
    event = json.loads((root / f'decision-{index:04d}.json').read_text())
    state = migrate_snapshot_defaults((root / event['snapshot_file']).read_text(),
        expected_fingerprint=event['snapshot_fingerprint'])['state']
    catalog = {d['key']: Dose(**dict(d, phases=tuple(map(tuple, d['phases'])),
        phase_scenarios=tuple(map(tuple, d['phase_scenarios'])),
        forecast_scope=read_scope(d['forecast_scope']))) for d in event['stage_runtime']['catalog']}
    reference = ContinuousDoseReference('sandbox upload accounting')
    handoffs = event['forecast_handoff_by_dose']
    baseline = tuple(map(tuple, event['program']))
    physical = {}

    def evaluate(program):
        for day, key in program:
            if (day, key) not in physical:
                physical[day, key] = planned_accounted(f'candidate:{day}:{key}',
                    datetime.combine(date.fromisoformat(day), time(18), ZoneInfo(state.timezone)),
                    catalog[key], handoffs[key], reference, available=state.as_of)
        result = evaluate_snapshot(state, tuple(physical[p] for p in program),
            tuple(k for _, k in program), reference=reference)
        scored = result['evaluation']
        parts = scored.parts
        return dict(cost=scored.total_cost, delivery=parts['delivery_loss'],
            demand=state.context.exchange * parts['demand_cost'],
            short=parts['easy_duration']['cost'], losses=parts['shortfall_by_purpose'],
            eligible=result['decision_eligible'],
            first14_miles=sum(catalog[k].miles for d,k in program
                if date.fromisoformat(d) < state.context.decision + timedelta(days=14)),
            program=program)

    original = evaluate(baseline)
    if not original['eligible'] or abs(original['cost'] - event['cost']) > 1e-12:
        raise ValueError('Archived score did not reproduce')
    changed = []
    for day, key in baseline:
        if catalog[key].role != 'easy' or date.fromisoformat(day) >= state.context.decision + timedelta(days=14):
            continue
        larger = [k for k in handoffs if catalog[k].role == 'easy'
            and catalog[k].miles >= catalog[key].miles + .5 - 1e-8]
        if larger:
            changed.append((day, key, min(larger, key=lambda k:(catalog[k].miles, k))))
    candidates = []
    for switches in product((False, True), repeat=len(changed)):
        choices = {d:(b if switch else a) for (d,a,b),switch in zip(changed,switches)}
        candidates.append(evaluate(tuple((d, choices.get(d,k)) for d,k in baseline)))
    target = state.context.policy.weekly_miles * 2
    funded = [r for r in candidates if r['eligible'] and r['first14_miles'] >= target - 1e-8]
    best = min(funded, key=lambda r:r['cost']) if funded else None
    return dict(index=index, as_of=event['as_of'], target14=target, original=original,
        alternative_count=len(candidates), best_full_14_volume=best,
        delta={k:best[k]-original[k] for k in ('cost','delivery','demand','short')} if best else None,
        all_alternatives=candidates,
        scope='Fixed calendar and tail, enumerated nearest >=0.5-mile easy increases in first 14 days; not global optimization')


def small_search_cases():
    dates=('2026-10-14','2026-10-15','2026-10-16')
    seed=((dates[0],'easy'),(dates[2],'easy'))
    rows=[]
    for funding, supplied in ((False, True),(True, True),(True, False)):
        calls=[]
        def evaluate(p):
            calls.append(p)
            return dict(cost=float(len(p)),eligible=not funding or len(p)>=2)
        try:
            result=search(dates=dates,keys=('easy',),roles={'easy':'easy'},
                previous=(),hard_end=dates[-1],budget=100,evaluate=evaluate,
                seed_programs=(seed,) if supplied else ())
            status=dict(result='returned',program=result['program'])
        except (ValueError,NoEligibleProgram) as exc:
            status=dict(result=type(exc).__name__,error=str(exc))
        rows.append(dict(funding=funding,complete_seed=supplied,calls=calls,**status))
    return rows


def actual_crash(root, output, config):
    """Reproduce only the failed preview in a disposable copy; forbid publication."""
    from run_analysis.adaptive.factory import resume_service
    from run_analysis.adaptive.adaptive_stage_lifecycle import advance_stage, review_entries
    from run_analysis.staged_progression import review_recent_delivery
    import run_analysis.adaptive.new_model_daily_harness as module
    clone=output/'sandbox-copy'
    clone.mkdir()
    for name in ('scenario.json','execution-sandbox.json'):
        shutil.copy2(root/name,clone/name)
    with sqlite3.connect(f'file:{root.resolve()}/sandbox.sqlite?mode=ro',uri=True) as source:
        with sqlite3.connect(clone/'sandbox.sqlite') as dest:
            source.backup(dest)
        at=datetime.fromisoformat(source.execute('SELECT max(available_at) FROM sandbox_completions').fetchone()[0])+timedelta(seconds=1)
    service=resume_service(clone,config,at)
    h=service.harness
    prior_stage=h.stage
    result={}
    original=module.migrated_search
    def traced_search(**kw):
        evaluate=kw['evaluate']
        evaluations=[]
        def traced(program):
            r=evaluate(program)
            evaluations.append(dict(program=program,eligible=r['eligible'],cost=r['cost']))
            return r
        try:
            original(**dict(kw,evaluate=traced))
        except ValueError as exc:
            frames=[]
            tb=exc.__traceback__
            while tb:
                if tb.tb_frame.f_code.co_name=='search':
                    local=tb.tb_frame.f_locals
                    frames.append(dict(day=local.get('day'),key=local.get('key'),
                        parent=local.get('parent',{}).get('physical'),
                        empty_admitted=() in local.get('admitted',{}),
                        admitted_count=len(local.get('admitted',{}))))
                tb=tb.tb_next
            result.update(error=str(exc),frames=frames,evaluations=evaluations)
            raise
        raise AssertionError('Diagnostic expected a crash; publication forbidden')
    module.migrated_search=traced_search
    try:
        review=review_recent_delivery(h.stage,review_entries(h,at),at,ledger_complete=True,
            longest_prior_30d_miles=max(r.miles for r in h.history if at-timedelta(days=30)<=r.start<at))
        result['review']={k:review.get(k) for k in ('status','credited_miles','required_miles','candidate')}
        advance_stage(h,service.initial_inputs,at)
    except ValueError as exc:
        if str(exc)!='A physical programme cannot contain two runs on one date':
            raise
    finally:
        module.migrated_search=original
        result['stage_unchanged']=h.stage==prior_stage
        h.sandbox.close()
    return result


def main():
    p=ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    a.output_dir.mkdir(exist_ok=False)
    config=load_config()
    manifest=freeze_sources(a.output_dir/'sources.zip',config=config,
        entrypoint='scripts.audit_volume_contract_conflict',inputs=(a.root/'sandbox.sqlite',
            a.root/'scenario.json',a.root/'decision-0001.json',a.root/'snapshot-0001.json'))
    result=dict(manifest=manifest,small_search_cases=small_search_cases(),
        frozen_contrasts=[frozen_contrasts(a.root,i) for i in (1,23)],
        actual_crash=actual_crash(a.root,a.output_dir,config))
    with (a.output_dir/'audit.json').open('x') as f:
        json.dump(result,f,indent=2,default=lambda v:asdict(v) if hasattr(v,'__dataclass_fields__') else v.isoformat())
    print(json.dumps(dict(small_search_cases=result['small_search_cases'],
        contrasts=[{k:v for k,v in r.items() if k not in ('all_alternatives','original','best_full_14_volume')} for r in result['frozen_contrasts']],
        crash={k:v for k,v in result['actual_crash'].items() if k!='evaluations'}),default=str),flush=True)


if __name__=='__main__':
    main()
