"""Daily + post-upload real-trace sandbox screen; never a weekly rerun shortcut."""
from argparse import ArgumentParser
from datetime import date,datetime,time,timedelta,timezone
import json
from pathlib import Path
import shutil
import sqlite3

from run_analysis.config import load_config
from scripts.adaptive_candidate_pipeline import load_athlete,compile_athlete,compile_easy_volume_phase
from scripts.run_adaptive_candidate_plan import build
from scripts.recorded_trace_execution import donor,transformed_payload,compatible_donor
from scripts.research_manifest import freeze_sources
from scripts.stage_acceptance import StageLedger
from scripts.adaptive_stage_lifecycle import advance_stage
from scripts.new_model_daily_harness import DailyHarness
from run_analysis.adaptive.service import AdaptivePlannerService


def fast_fail_edits(event, window=4):
    return [e for e in event['churn'][str(window)] if e['date_changed'] or e['role_changed'] or
            e['distance_change'] is not None and abs(e['distance_change'])>.5+1e-8]


def verify_prescribed_quality(harness,actual_key,*,quality=False):
    """Whole miles/phase time cannot certify native distance/portion adherence.

    This is a TEST-protocol gate only. Real partial runs keep their measured
    receipt, fatigue and adaptive planning path; they are never rejected here.
    """
    if getattr(harness,'coaching_base',None) is None:return None
    matches=[r for r in harness.coaching_receipts if r.activity_key==actual_key]
    if not matches:
        if quality:raise ValueError('Prescribed quality execution lacks its issued primary receipt')
        return None  # Nonquality execution has no quality receipt.
    if len(matches)!=1:raise ValueError('One issued quality receipt per execution required')
    receipt=matches[0]
    if receipt.purpose_matched is not True or receipt.credit.low<1.-1e-7:
        raise ValueError('Prescribed-scenario fixture failed verified primary quality fulfillment; '
            'not a perfect-adherence planner result: '+str(receipt))
    return dict(primary_unit=receipt.assignment.primary_unit,credit_low=receipt.credit.low,
        purpose_matched=True,verified_receipt=True)


def fast_fail_quality_edits(event, catalog, window=4):
    """Native recipe/family changes must not hide inside unchanged quality role.

    Same-recipe repetition/extent edits remain dose changes, separately audited;
    they are not automatically equivalent stimuli or automatically failures.
    """
    from scripts.audit_adaptive_daily_replay import quality_edit_counts
    result=quality_edit_counts([event],{event['index']:catalog},window)
    if result['unknown_family_comparisons']:
        return [dict(reason='native_quality_identity_unresolved',
            comparisons=result['unknown_family_comparisons'])]
    return [r for r in result['edits'] if r['family_changed'] or r['recipe_changed']]


def restored_uploads(root, sandbox):
    """Retain original transformation/HR provenance instead of inventing it on resume."""
    archived=[json.loads(p.read_text()) for p in sorted(root.glob('execution-[0-9][0-9][0-9][0-9].json'))]
    actual=[json.loads(r['comparison']) for r in sandbox.db.execute('SELECT comparison FROM sandbox_completions')]
    inherited_file=root/'replay-inherited-completions.json'
    inherited=json.loads(inherited_file.read_text()) if inherited_file.exists() else []
    inherited_by_key={r['actual_key']:r for r in inherited}
    if len(inherited_by_key)!=len(inherited):
        raise ValueError('Inherited completion identities must be unique')
    actual_by_key={r['actual_key']:r for r in actual}
    if any(actual_by_key.get(k)!=r for k,r in inherited_by_key.items()):
        raise ValueError('Inherited completion differs from its frozen seed evidence')
    by_key={r['comparison']['actual_key']:r for r in archived}
    if (len(by_key)!=len(archived) or set(by_key)&set(inherited_by_key) or
            set(actual_by_key)!=set(by_key)|set(inherited_by_key)):
        raise ValueError('Resume requires one archived execution provenance record per completion')
    if any(actual_by_key[k]!=r['comparison'] for k,r in by_key.items()):
        raise ValueError('Archived execution evidence differs from durable completion')
    return archived


def screen_boundary(harness,requested,previous_report=None):
    """Replay clock is distinct from the immutable training-stage origin."""
    saved=(previous_report or {}).get('screen_start')
    if saved and requested and saved!=requested:
        raise ValueError('Resume must retain its explicit replay start')
    selected=saved or requested
    if selected is None:return harness.origin,None,None
    start=date.fromisoformat(selected)
    clock=datetime.combine(start,time(4),harness.zone)
    if not previous_report and clock<=harness.last_as_of:
        raise ValueError('Explicit replay start must follow the latest seeded decision')
    baseline=(previous_report or {}).get('screen_baseline_event',harness.events[-1]['index'])
    return clock,start,baseline


def screen_summary(harness,days,uploads,start,baseline):
    if start is None:return harness.summary(days,uploads)
    report=harness.summary(days,[r for r in uploads if date.fromisoformat(r['day'])>=start],
        delivery_start=start)
    events=[e for e in harness.events if e['index']>=baseline]
    # Opening promise is context, not an edit relative to the older model.
    report['events']=[dict(events[0],churn={k:[] for k in events[0]['churn']}),*events[1:]]
    report.update(screen_start=start.isoformat(),screen_baseline_event=baseline,
        initial_search_reused=True,refresh_clock='04:00 local each subsequent day')
    return report


def report_path(root, resume_number=None):
    return root/(f'resume-{resume_number:04d}.summary.json' if resume_number is not None else 'lifecycle-summary.json')


def replay_config(config, root):
    """Pin changed HR only through a verified adopted reference.

    Generic research sandboxes with already matching HR keep their original
    strict resume path; they do not need live-bridge normalization records.
    """
    marker=json.loads((root/'execution-sandbox.json').read_text())
    expected=dict(max_hr=config['max_hr'],resting_hr=config['resting_hr'])
    if json.loads(marker['measurement_context'])==expected:
        return config
    from run_analysis.adaptive.hr_reference import frozen_live_config
    return frozen_live_config(config,root=root)


def resume_offset(harness,origin,previous_report):
    observed=(harness.last_as_of.astimezone(harness.zone).date()-origin.astimezone(harness.zone).date()).days
    if previous_report.get('status')=='bounded_screen_complete':
        completed=previous_report['completed_days']
        if type(completed) is not int or completed<observed or completed>observed+1:
            raise ValueError('Completed driver-day boundary differs from saved decisions')
        # Rest and missed days are completed driver days too; no upload is needed
        # to establish their boundary. Never repeat an already archived omission.
        return completed
    return observed


def fork_seed_plan(seed,root):
    """Copy an issued sandbox, not a newly optimized or fabricated opening plan."""
    from scripts.replay_adaptive_search import load_decision
    events=sorted(seed.glob('decision-*.json'))
    if not events:
        raise ValueError('Seed requires an archived decision')
    latest=json.loads(events[-1].read_text())
    load_decision(seed,latest['index']+1)  # Mandatory original score/eligibility parity.
    root.mkdir(exist_ok=False)
    for path in (seed/'scenario.json',seed/'execution-sandbox.json',*events,*sorted(seed.glob('snapshot-*.json')),
                 *sorted(seed.glob('execution-*.json'))):
        shutil.copy2(path,root/path.name)
    source=sqlite3.connect(f'file:{(seed/"sandbox.sqlite").resolve()}?mode=ro',uri=True)
    destination=sqlite3.connect(root/'sandbox.sqlite')
    try:
        source.backup(destination)
    finally:
        source.close(); destination.close()
    return datetime.fromisoformat(latest['as_of'])


def run(root,database,*,days=7,evaluations=10000,scenario='prescribed',resume=False,tail_days=14,steady_continuation=False,soft_horizon_taper=False,easy_volume_phase=False,checkpoint_days=None,continuation_boundary=False,single_volume=False,seed_plan=None,recover_stopped=False,volume_linear_coefficient=None,opening_run_hour=None,purpose_continuation=False,purpose_continuation_phase=None,stability_window=4,continuation_distribution=None,missed_run_offsets=(),screen_start=None):
    if screen_start is not None and not (seed_plan is not None or resume):
        raise ValueError('Explicit replay start requires a seed or resume')
    missed_run_offsets=tuple(sorted(set(missed_run_offsets)))
    if any(type(n) is not int or n<0 or n>=days for n in missed_run_offsets):
        raise ValueError('Missed runs require explicit offsets inside this replay')
    if continuation_distribution is not None and continuation_distribution not in {'linear_v1','c1_v1'}:
        raise ValueError('Named continuation distribution required')
    if stability_window not in (4,7):
        raise ValueError('Explicit four- or seven-day stability window required')
    if purpose_continuation_phase is not None and purpose_continuation_phase not in {'uniform_v1','known_full_phase_v1'}:
        raise ValueError('Named purpose continuation phase required')
    if opening_run_hour is not None and (type(opening_run_hour) is not int or not 0<=opening_run_hour<=23):
        raise ValueError('Opening execution hour must be an explicit local hour')
    if opening_run_hour is not None and resume:
        raise ValueError('Resume cannot change the original opening execution clock')
    if checkpoint_days is not None and not 0<checkpoint_days<days:
        raise ValueError('Checkpoint must precede final simulated day count')
    if resume and easy_volume_phase:
        raise ValueError('Resume retains its accepted stage; cannot rebootstrap a new focus')
    if seed_plan is not None and (resume or easy_volume_phase):
        raise ValueError('Seed retains its original stage; cannot resume or rebootstrap simultaneously')
    if recover_stopped and not resume:
        raise ValueError('Stopped-driver recovery requires explicit resume')
    config=load_config()
    origin=datetime(2026,9,27,8,tzinfo=timezone.utc)
    manifests=list(root.glob('resume-*.sources.zip')) if resume else []
    resume_number=len(manifests)+1 if resume else None
    previous_report=None
    if resume:
        reports=sorted(root.glob('resume-*.summary.json'))
        original=root/'lifecycle-summary.json'
        if recover_stopped and not reports and not original.exists():
            stopped=json.loads(sorted(root.glob('stopped-*.json'))[-1].read_text())
            if stopped['error']!='Donor phase must be covered by recorded source timestamps':
                raise ValueError('Only attributed donor-boundary failure may resume through this recovery')
            if scenario!='prescribed' or any(json.loads(p.read_text())['provenance']['distance_factor']!=1.
                    for p in root.glob('execution-[0-9][0-9][0-9][0-9].json')):
                raise ValueError('Stopped replay must retain prescribed execution')
            previous_report=dict(status='attributed_driver_stop',scenario=scenario,stage_reviews=[],
                initial_search_reused=True,refresh_clock='04:00 local each subsequent day',
                recovery=dict(original_stop=stopped,missing_stage_review_transcript=True,
                    explanation='Original exception predates summary export; saved decisions/executions retained'))
        else:
            previous_report=json.loads((reports[-1] if reports else original).read_text())
        if recover_stopped and previous_report['status']=='bounded_screen_complete':
            stopped=json.loads(sorted(root.glob('stopped-*.json'))[-1].read_text())
            last=json.loads(sorted(root.glob('decision-*.json'))[-1].read_text())
            if (stopped['last_event']!=last['as_of'] or stopped['error_type']!='ValueError'
                    or not stopped['error'].startswith("invalid literal for int() with base 10: 'sandbox:")):
                raise ValueError('Only the attributed original-donor alias stop may extend a completed checkpoint')
            # Verify every nonnumeric pre-origin seed has an exact original
            # database activity BEFORE permitting recovery; don't skip the run.
            from scripts.recorded_trace_execution import original_activity_id
            from run_analysis.adaptive.program_evidence_adapter import read_accounted
            import sqlite3
            scenario_data=json.loads((root/'scenario.json').read_text())
            with sqlite3.connect(Path(database).resolve().as_uri()+'?mode=ro',uri=True) as db:
                for raw in scenario_data['history']:
                    if not raw['key'].isdecimal():
                        physical=read_accounted(raw)
                        if physical.available_at>datetime.fromisoformat(scenario_data['origin']):
                            raise ValueError('Recovery donor must predate the original simulation origin')
                        original_activity_id(db,physical)
            previous_report=dict(previous_report,status='attributed_driver_stop',
                recovery=dict(original_stop=stopped,missing_stage_review_transcript=True,
                    explanation='Original donor alias failed before upload; exact original identities verified; intervening decisions retained'))
        if previous_report['scenario']!=scenario:
            raise ValueError('Resume must retain its original execution scenario')
        if tuple(previous_report.get('missed_run_offsets',()))!=missed_run_offsets:
            raise ValueError('Resume must retain its original missed-run scenario')
        if previous_report['status']!='bounded_screen_complete' and not (recover_stopped and
                previous_report['status']=='attributed_driver_stop'):
            raise ValueError('Attribute the stopped screen before extending it')
    if seed_plan is not None or resume:
        # Validate resume/scenario/recovery first, then pin the copied ledger's
        # load unit. Never reinterpret sources using today's watch settings.
        config=replay_config(config,seed_plan if seed_plan is not None else root)
    seed_inputs=(tuple(seed_plan.glob('decision-*.json'))+tuple(seed_plan.glob('snapshot-*.json'))+
        (seed_plan/'scenario.json',seed_plan/'sandbox.sqlite')) if seed_plan is not None else ()
    manifest=freeze_sources(root/f'resume-{len(manifests)+1:04d}.sources.zip' if resume else root.with_suffix('.sources.zip'),config=config,
        entrypoint='scripts.run_adaptive_lifecycle',inputs=(database,*seed_inputs))
    if seed_plan is not None:
        seed_as_of=fork_seed_plan(seed_plan,root)
        harness=DailyHarness.resume(root,config,as_of=seed_as_of)
        origin=harness.origin
        if harness.policy.evaluations!=evaluations:
            harness.sandbox.close()
            raise ValueError('Seed must retain its original search budget')
        inputs=dict(history=tuple(r for r in harness.history if r.key in harness.seed_keys),
            sessions=tuple(r for r in harness.session_evidence() if r.key in harness.seed_keys),
            phase_history=harness.phase_history,reference=harness.ref)
        ledger=StageLedger.for_sandbox(harness.sandbox,seed=harness.stage)
    elif resume:
        last=json.loads(sorted(root.glob('decision-*.json'))[-1].read_text())
        harness=DailyHarness.resume(root,config,as_of=datetime.fromisoformat(last['as_of']))
        origin=harness.origin
        inputs=dict(history=tuple(r for r in harness.history if r.key in harness.seed_keys),
            sessions=tuple(r for r in harness.session_evidence() if r.key in harness.seed_keys),
            phase_history=harness.phase_history,reference=harness.ref)
        if recover_stopped and any(datetime.fromisoformat(json.loads(r['actual'])['available_at'])>harness.last_as_of
                for r in harness.sandbox.db.execute('SELECT actual FROM sandbox_completions')):
            harness.sandbox.close()
            raise ValueError('Pending post-upload recovery requires separate accounting review')
        if continuation_boundary and not harness.policy.continuation_boundary:
            harness.sandbox.close()
            raise ValueError('Resume cannot silently change the objective contract')
        if single_volume and harness.training_policy.purpose_weights[1]!=0:
            harness.sandbox.close()
            raise ValueError('Resume cannot migrate a dual-volume objective')
        # Explicit recovery of an initial fixed-stage artifact, not a new seed
        # from today's history. Accepted progression must already have a ledger.
        ledger=StageLedger.for_sandbox(harness.sandbox,seed=harness.stage)
    else:
        inputs=load_athlete(database,config,origin)
        baseline=None
        if easy_volume_phase:
            baseline,candidate,_=compile_easy_volume_phase(inputs,origin,timezone=config['timezone_default'])
        else:
            candidate=compile_athlete(inputs,origin)
        harness=build(root,config,inputs,candidate,origin,evaluations=evaluations,tail_days=tail_days,
            steady_continuation=steady_continuation,soft_horizon_taper=soft_horizon_taper,
            baseline_candidate=baseline,continuation_boundary=continuation_boundary,single_volume=single_volume,
            volume_linear_coefficient=volume_linear_coefficient,purpose_continuation=purpose_continuation,
            purpose_continuation_phase=purpose_continuation_phase,
            continuation_distribution=continuation_distribution or 'linear_v1')
        ledger=StageLedger.for_sandbox(harness.sandbox)
    if continuation_distribution is not None and harness.policy.continuation_distribution!=continuation_distribution:
        harness.sandbox.close(); ledger.close()
        raise ValueError('Resume/seed must retain its continuation distribution; prepare a freshly scored seed')
    if volume_linear_coefficient is not None and (resume or seed_plan is not None) and (
            harness.policy.volume_linear_coefficient!=volume_linear_coefficient):
        harness.sandbox.close()
        raise ValueError('Resume/seed must retain its archived volume curve; build a new experiment to change it')
    if purpose_continuation and not harness.policy.purpose_continuation:
        harness.sandbox.close()
        raise ValueError('Resume/seed cannot migrate the purpose contract; prepare a freshly scored seed')
    if purpose_continuation_phase is not None and harness.policy.purpose_continuation_phase!=purpose_continuation_phase:
        harness.sandbox.close()
        raise ValueError('Resume/seed must retain its archived purpose phase; prepare a freshly scored seed')
    service=AdaptivePlannerService(harness,inputs)
    originals={}
    screen_date=baseline=None
    if screen_start is not None or previous_report and previous_report.get('screen_start'):
        origin,screen_date,baseline=screen_boundary(harness,screen_start,previous_report)
    if seed_plan is not None and screen_date is not None:
        inherited=[]
        for row in harness.sandbox.completion_rows(harness.last_as_of):
            actual=json.loads(row['accounted'])
            if datetime.fromisoformat(actual['start']).astimezone(harness.zone).date()>=screen_date:
                raise ValueError('Inherited completion lies inside the new replay delivery window')
            inherited.append(json.loads(row['comparison']))
        with (root/'replay-inherited-completions.json').open('x') as stream:
            json.dump(inherited,stream,indent=2)
        # Existing per-upload archives remain evidence but are excluded from
        # this forward screen; never pretend to have synthesized their HR.
        for path in root.glob('execution-[0-9][0-9][0-9][0-9].json'):
            path.rename(path.with_name('inherited-'+path.name))
    uploads=restored_uploads(root,harness.sandbox) if resume or seed_plan is not None else []
    failures=[]
    reviews=list(previous_report.get('stage_reviews',())) if previous_report else []
    start_offset=resume_offset(harness,origin,previous_report) if resume else 0
    finished=start_offset
    missed_runs=list(previous_report.get('missed_runs',())) if previous_report else []
    stability_contract=('previously_visible_UI_dates_v1; raw entering-window edits retained; '
        'date/role/family/recipe or adjacent distance >0.5 miles fast-fail; cumulative dose movement audited'
        if harness.quality_runtime is not None else 'raw_current_window_edits_v1')
    def check(event):
        gate_event=event
        if harness.quality_runtime is not None:
            from scripts.audit_adaptive_daily_replay import visible_churn_event
            scoped=[e for e in harness.events if baseline is None or e['index']>=baseline]
            gate_event=visible_churn_event(scoped,
                {e['index']:harness.catalog for e in scoped},stability_window)
        bad=fast_fail_edits(gate_event,stability_window)
        quality_bad=(fast_fail_quality_edits(gate_event,harness.catalog,stability_window)
            if harness.quality_runtime is not None else [])
        print(json.dumps(dict(as_of=event['as_of'],reason=event['reason'],seconds=event['seconds'],
            local_complete=event['local_neighborhood_complete'],churn=event['churn'],miles21=event['volume'],
            stage_volume=harness.stage.weekly_equivalent_miles,stage_long=harness.stage.long_miles,
            progression_target=event.get('progression_target'))),flush=True)
        if bad:
            failures.append(dict(event=event['index'],reason=('four' if stability_window==4 else 'seven')+'_day_edit_requires_attribution',edits=bad))
        if quality_bad:
            failures.append(dict(event=event['index'],reason='native_quality_family_or_recipe_edit_requires_attribution',edits=quality_bad))
        return not bad and not quality_bad
    try:
        for offset in range(start_offset,days):
            morning=(datetime.combine(origin.astimezone(harness.zone).date()+timedelta(days=offset),
                time(4),harness.zone) if seed_plan is not None or previous_report and
                previous_report.get('refresh_clock') else origin+timedelta(days=offset))
            day=morning.astimezone(harness.zone).date()
            if harness.last_as_of is None or morning>harness.last_as_of:
                decision=service.refresh(morning,reason='initial' if offset==0 else 'daily_rollover')
                harness=service.harness
                review,event=decision['stage_review'],decision['event']
                reviews.append(dict(at=morning,**review))
                if not check(event):
                    break
            if offset==checkpoint_days:
                # Fourteen completed execution days PLUS their boundary refresh:
                # the review is due now, before executing any day-15 workout.
                from dataclasses import asdict,is_dataclass
                checkpoint=screen_summary(harness,finished,uploads,screen_date,baseline)
                checkpoint.update(status='checkpoint_no_fast_fail',completed_days=finished,
                    stage_reviews=reviews,final_stage=harness.stage,failures=failures,manifest=manifest,
                    continuation_days=days,includes_boundary_refresh=True,
                    stability_window=stability_window,stability_contract=stability_contract)
                with (root/f'checkpoint-{checkpoint_days:04d}.json').open('x') as stream:
                    json.dump(checkpoint,stream,default=lambda v:asdict(v) if is_dataclass(v) else v.isoformat(),indent=2)
                print(json.dumps(dict(checkpoint_days=finished,status='checkpoint_no_fast_fail',
                    stage_volume=harness.stage.weekly_equivalent_miles,stage_long=harness.stage.long_miles,
                    last_review=reviews[-1]['status'],continuing_to=days)),flush=True)
            chosen=harness.active.get(day.isoformat())
            if offset in missed_run_offsets:
                if not chosen:
                    raise ValueError('Specified missed-run day has no issued run')
                missed=dict(day=day.isoformat(),offset=offset,issued=chosen,
                    status='not_completed',physical_activity_created=False,
                    scope='Known driver omission; next refresh sees missing completion, not a zero-load uploaded run')
                missed_runs.append(missed)
                with (root/f'missed-run-{offset:04d}.json').open('x') as stream:
                    json.dump(missed,stream,indent=2)
                finished=offset+1
                continue
            if chosen:
                dose=harness.catalog[chosen['dose']]
                eligible=[r for r in inputs['history'] if r.role==dose.role and r.available_at<=origin]
                source,original=compatible_donor(database,eligible,dose,config,harness.ref,originals)
                distance_factor=1. if scenario=='prescribed' else (1.08 if offset%3==0 else .94)
                duration_factor=distance_factor
                hour=opening_run_hour if offset==0 and opening_run_hour is not None else 16 if offset%2==0 else 19
                start=datetime.combine(day,time(hour),harness.zone)
                if start<harness.last_as_of:
                    raise ValueError('Cannot execute an opening-day run before the seeded decision')
                payload,phases,end,provenance=transformed_payload(original,dose,start,
                    distance_factor=distance_factor,duration_factor=duration_factor)
                outcome=harness.upload(payload,day=day.isoformat(),available_at=end,
                    observed_role=dose.role,observed_phases=phases)
                protocol_error=None
                if scenario=='prescribed':
                    try:
                        verified=verify_prescribed_quality(harness,outcome['comparison']['actual_key'],
                            quality=dose.role=='quality')
                        if verified is not None:provenance['primary_fulfillment_check']=verified
                    except ValueError as error:
                        provenance['primary_fulfillment_check']=dict(passed=False,error=str(error))
                        protocol_error=error
                uploads.append(dict(day=day.isoformat(),**outcome,provenance=provenance))
                with (root/f'execution-{len(uploads):04d}.json').open('x') as stream:
                    json.dump(uploads[-1],stream,default=lambda v:v.isoformat(),indent=2)
                if protocol_error is not None:raise protocol_error
                decision=service.refresh(end+timedelta(seconds=1),reason='post_upload')
                harness=service.harness
                reviews.append(dict(at=end+timedelta(seconds=1),**decision['stage_review']))
                after=decision['event']
                if not check(after):
                    break
            finished=offset+1
        report=screen_summary(harness,finished,uploads,screen_date,baseline)
        report.update(requested_days=days,completed_days=finished,failures=failures,manifest=manifest,
            stability_window=stability_window,
            stability_contract=stability_contract,
            stage_reviews=reviews,final_stage=harness.stage,
            status='stopped_for_attribution' if failures else 'bounded_screen_complete',scenario=scenario,
            acceptance_scope='original-HR transformed execution with conditional stage review; full app-route acceptance still separate')
        if missed_run_offsets:
            report.update(missed_run_offsets=missed_run_offsets,missed_runs=missed_runs,
                missed_run_scope='Not perfect adherence; changes after omissions require adaptation/churn attribution, not automatic exemption')
        if seed_plan is not None:
            report.update(seed_plan=str(seed_plan),seed_as_of=seed_as_of.isoformat(),
                initial_search_reused=True,refresh_clock='04:00 local each subsequent day')
        if opening_run_hour is not None:
            report['opening_run_hour']=opening_run_hour
        elif previous_report:
            report.update({k:previous_report[k] for k in ('seed_plan','seed_as_of','initial_search_reused',
                'refresh_clock','recovery','opening_run_hour') if k in previous_report})
        with report_path(root,resume_number).open('x') as stream:
            from dataclasses import asdict,is_dataclass
            json.dump(report,stream,default=lambda v:asdict(v) if is_dataclass(v) else v.isoformat(),indent=2)
        return report
    except Exception as exc:
        with (root/f'stopped-{len(harness.events):04d}.json').open('x') as stream:
            json.dump(dict(status='incomplete',error_type=type(exc).__name__,error=str(exc),
                completed_days=finished,last_event=harness.events[-1]['as_of'] if harness.events else None,
                screen_start=screen_date.isoformat() if screen_date else None,screen_baseline_event=baseline,
                manifest=manifest),stream,indent=2)
        raise
    finally:
        harness.sandbox.close()
        ledger.close()


if __name__=='__main__':
    p=ArgumentParser()
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--database',type=Path,default=Path('data/run_analysis.sqlite'))
    p.add_argument('--days',type=int,default=7)
    p.add_argument('--stability-window',type=int,choices=(4,7),default=4)
    p.add_argument('--evaluations',type=int,default=10000)
    p.add_argument('--tail-days',type=int,default=14)
    p.add_argument('--steady-continuation',action='store_true')
    p.add_argument('--soft-horizon-taper',action='store_true')
    p.add_argument('--continuation-boundary',action='store_true')
    p.add_argument('--single-volume',action='store_true')
    p.add_argument('--purpose-continuation',action='store_true')
    p.add_argument('--purpose-phase',choices=('uniform_v1','known_full_phase_v1'),default=None,
        help='Explicit scoring contract assertion; cannot silently migrate a seed/resume')
    p.add_argument('--continuation-distribution',choices=('linear_v1','c1_v1'),default=None,
        help='Explicit scoring contract assertion; cannot silently migrate a seed/resume')
    p.add_argument('--volume-linear-coefficient',type=float,default=None)
    p.add_argument('--opening-run-hour',type=int,default=None,
        help='Explicit opening-day local execution hour; use an hour after a late issued seed')
    p.add_argument('--scenario',choices=('prescribed','imperfect'),default='prescribed')
    p.add_argument('--missed-run-offset',action='append',type=int,default=[],
        help='Zero-based issued day omitted entirely; no synthetic activity or post-upload refresh')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--easy-volume-phase',action='store_true')
    p.add_argument('--checkpoint-days',type=int)
    p.add_argument('--seed-plan',type=Path)
    p.add_argument('--screen-start',help='Explicit local YYYY-MM-DD after the seeded decision; stage origin stays unchanged')
    p.add_argument('--recover-stopped',action='store_true')
    a=p.parse_args()
    r=run(a.output_dir,a.database,days=a.days,evaluations=a.evaluations,scenario=a.scenario,resume=a.resume,
        stability_window=a.stability_window,
        tail_days=a.tail_days,steady_continuation=a.steady_continuation,soft_horizon_taper=a.soft_horizon_taper,
        easy_volume_phase=a.easy_volume_phase,checkpoint_days=a.checkpoint_days,continuation_boundary=a.continuation_boundary,
        single_volume=a.single_volume,seed_plan=a.seed_plan,recover_stopped=a.recover_stopped,
        purpose_continuation=a.purpose_continuation,purpose_continuation_phase=a.purpose_phase,
        continuation_distribution=a.continuation_distribution,
        volume_linear_coefficient=a.volume_linear_coefficient,opening_run_hour=a.opening_run_hour,
        missed_run_offsets=a.missed_run_offset,screen_start=a.screen_start)
    print(json.dumps({k:r[k] for k in ('status','completed_days','requested_days','failures')}))
