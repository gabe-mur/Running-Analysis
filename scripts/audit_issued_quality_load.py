"""Recent saved-prescription audit, not manufactured intention for old runs."""
from argparse import ArgumentParser
from datetime import datetime,timedelta,timezone
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import sqlite3

from run_analysis.config import load_config
from run_analysis.athlete_projection import aware
from run_analysis.continuous_training_dose import ContinuousDoseReference
from run_analysis.movement import classify_movement
from run_analysis.training_signals import HeartRateReference, observed_cardio_trace
from run_analysis.work_stimulus_evidence import observed_work_signals
from run_analysis.workload_accounting import Quantity
from scripts.audit_program_evidence_adapter import load_history
from scripts.audit_workload_expectation import app_session_types
from scripts.audit_work_stimulus import encode
from scripts.continuous_program_adapter import with_continuous_trace
from scripts.direct_quality_load import PhaseHistory, phase_observations, forecast_direct
from scripts.program_evidence_adapter import QualityReference
from scripts.validate_quality_load import phase_layout


def parse_issued_structure(payload):
    """Narrow typed adapter: no target-derived durations or invented text values.

    Legacy schema lacks recovery placement. Recognize its two explicit English
    phrases, retain provenance; otherwise report ambiguity. Variable fartlek and
    distance-only structures remain unresolved rather than secretly midpointed.
    """
    steps = payload.get('structure',[])
    # Modern adapter exports ordered numeric phases. Preserve their declared
    # times/units without converting estimated distance or using completion.
    if (steps and all(s.get('phase') in {'warmup','work','recovery','cooldown'}
            and s.get('duration_minutes') is not None and s.get('repetitions') in (None,1)
            and s.get('work_duration_range_minutes') is None for s in steps)
            and any(s['phase']=='work' for s in steps)):
        at=0.; layout=[]
        for step in steps:
            minutes=step['duration_minutes']
            if isinstance(minutes,bool) or not isfinite(minutes) or minutes<=0:
                raise ValueError('Finite positive explicit phase durations required')
            layout.append((step['phase'],at,at+minutes*60));at+=minutes*60
        work=[b-a for kind,a,b in layout if kind=='work']
        return dict(status='resolved_saved_numeric_phase_scenario',layout=tuple(layout),
            repetitions=len(work),work_seconds=work[0] if len(set(work))==1 else None,
            work_seconds_total=sum(work),recovery_count=sum(k=='recovery' for k,_,_ in layout),
            recovery_placement_basis='explicit saved phase sequence',total_seconds=at,
            extent_basis='saved nominal phase timing; primary units retained, not inferred from completion',
            primary_units=[s.get('prescription_basis') for s in steps],
            intended_effort_cues=[s.get('target_zones',[]) for s in steps])
    if len(steps) != 3 or steps[1].get('phase') != 'work':
        return dict(status='unsupported_structure_shape',layout=None)
    warm,work,cool = steps
    # Continuous threshold work has no repetitions/recovery requirement.
    if (work.get('repetitions') in (None,1) and work.get('recovery_duration_minutes') is None
            and work.get('work_duration_range_minutes') is None
            and all(s.get('duration_minutes') is not None for s in steps)):
        return parse_issued_structure(dict(payload,structure=[dict(warm,phase='warmup'),
            dict(work,phase='work'),dict(cool,phase='cooldown')]))
    required = [warm.get('duration_minutes'),work.get('work_duration_minutes'),
                work.get('recovery_duration_minutes'),cool.get('duration_minutes')]
    if any(v is None for v in required) or work.get('work_duration_range_minutes') is not None:
        return dict(status='unresolved_prescribed_durations',layout=None,
            missing=['warmup','work','recovery','cooldown'] if all(v is None for v in required) else
                [k for k,v in zip(('warmup','work','recovery','cooldown'),required) if v is None])
    if any(isinstance(v,bool) or not isfinite(v) or v <= 0 for v in required):
        raise ValueError('Finite positive explicit phase durations required')
    reps = work.get('repetitions')
    if isinstance(reps,bool) or not isinstance(reps,int) or reps <= 0:
        return dict(status='unresolved_repetition_count',layout=None)
    text = work.get('instruction','').lower()
    if 'after each' in text and 'between efforts' not in text:
        recovery_count = reps
    elif 'between efforts' in text and 'after each' not in text:
        recovery_count = reps-1
    else:
        return dict(status='unresolved_recovery_placement',layout=None)
    phases,at = [],0.
    def add(kind,minutes):
        nonlocal at
        end = at+minutes*60
        phases.append((kind,at,end))
        at = end
    add('warmup',required[0])
    for i in range(reps):
        add('work',required[1])
        if i < recovery_count:
            add('recovery',required[2])
    add('cooldown',required[3])
    return dict(status='resolved_explicit_durations',layout=tuple(phases),repetitions=reps,
        work_seconds=required[1]*60,work_seconds_total=required[1]*60*reps,recovery_count=recovery_count,
        recovery_placement_basis='saved instruction: after each' if recovery_count==reps else 'saved instruction: between efforts',
        total_seconds=at,extent_basis='saved explicit time structure; mileage range not converted using actual pace')


def issue_boundary(saved_at,payload,run_start):
    generated = datetime.fromisoformat(payload['generated_at'])
    persisted = datetime.fromisoformat(saved_at)
    if any(at.tzinfo is None or at.utcoffset() is None for at in (generated,persisted,run_start)):
        raise ValueError('Timezone-aware provenance required')
    issue = max(generated,persisted)
    if issue >= run_start:
        raise ValueError('Saved prescription must predate the run')
    return issue


def whole_load_error(predicted,low,high):
    """Whole prescribed load vs whole observed bounds; never trim plan to actual.

    Difference includes execution deviation and is not pure response-model error.
    """
    if predicted is None or low is None or high is None:
        return None
    return dict(predicted_load=predicted,observed_low=low,observed_high=high,
        relative_error_bounds=(predicted/high-1,predicted/low-1) if low > 0 else None)


def matched_assignments(db,as_of):
    """Include all quality assignments, not only interval-labelled workouts."""
    return db.execute('''SELECT m.activity_id,p.id,p.schedule_generated_at,p.recommendation_json,a.start_time_utc,
        p.plan_date,p.planned_for,m.match_confidence FROM activity_plan_matches m
        JOIN planned_workout_history p ON p.id=m.planned_workout_id JOIN activities a ON a.id=m.activity_id
        WHERE p.workout_type IN ('intervals','tempo_threshold') AND a.start_time_utc < ?
        ORDER BY a.start_time_utc''',(as_of.isoformat(),)).fetchall()


def audit(database,config,*,execution_observer=None,as_of=datetime(2026,9,26,4,tzinfo=timezone.utc)):
    aware(as_of)
    digest = sha256(database.read_bytes()).hexdigest()
    roles = app_session_types(database)
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True) as db:
        created = {str(k):max(datetime.fromisoformat(at),datetime.fromisoformat(imported))
            for k,at,imported in db.execute('''SELECT a.id,a.created_at_utc,s.imported_at_utc FROM activities a
                JOIN activity_sources l ON l.activity_id=a.id AND l.is_primary=1
                JOIN source_files s ON s.id=l.source_file_id''')}
        matches = matched_assignments(db,as_of)
        completed_inventory = {str(k) for k,start in db.execute(
            "SELECT id,start_time_utc FROM activities WHERE sport='Running' AND start_time_utc IS NOT NULL")
            if datetime.fromisoformat(start)<as_of}
        quality_ids = {k for k,v in roles.items() if k in completed_inventory
                       and v in {'intervals','tempo_threshold'}}
        plan_min = db.execute('SELECT MIN(schedule_generated_at) FROM planned_workout_history').fetchone()[0]
    targets = {}
    for key,pid,saved,raw,start,plan_date,planned_for,confidence in matches:
        payload = json.loads(raw)
        row = dict(key=str(key),plan_id=pid,saved_at=saved,payload=payload,run_start=datetime.fromisoformat(start),
            plan_date=plan_date,planned_for=planned_for,retrospective_match_confidence=confidence,
            saved_payload_sha256=sha256(raw.encode()).hexdigest(),current_actual_role=roles[str(key)])
        try:
            row['issued_at'] = issue_boundary(saved,payload,row['run_start'])
            row['provenance_status'] = 'saved_before_run_not_proof_of_viewed_version'
            row['contract'] = parse_issued_structure(payload)
        except ValueError as e:
            row.update(provenance_status=str(e),issued_at=None,contract=dict(status='invalid_provenance',layout=None))
        targets[str(key)] = row
    valid = [r['issued_at'] for r in targets.values() if r['issued_at'] is not None]
    if not valid:
        return dict(status='no_pre_run_saved_quality_assignments',cases=list(targets.values()))
    reference = ContinuousDoseReference('unchanged direct-load issued-prescription check')
    hr = HeartRateReference(config['resting_hr'],config['max_hr'],'current settings; not reconstructed historically')
    history,actuals = [],{}
    def collect(row,activity):
        if row.role != 'quality' and row.key not in targets:
            return row
        intervals = classify_movement(activity.trackpoints,config['moving_time']).intervals
        trace = observed_cardio_trace(activity,intervals,hr,reference.policy,include_spans=True,compute_density=False)
        run,_ = with_continuous_trace(row,trace['normalized_spans'],reference=reference,trace_key=row.key,
            trace_source='observed',elapsed_seconds=trace['elapsed_seconds'],
            outside_reference_seconds=trace['exclusions']['outside_reference_seconds'])
        signals = observed_work_signals(run,activity,config,intervals,trace['normalized_spans']) if run.bouts else None
        layout = phase_layout(signals.windows,trace['elapsed_seconds']) if signals else ()
        actuals[row.key] = dict(layout=layout,trace=trace,run=run)
        if row.role == 'quality':
            history.append(PhaseHistory(row.key,row.start,row.end,max(row.end,created[row.key]),reference.accounting_key,
                phase_observations(layout,trace['normalized_spans'])))
        return row
    old = QualityReference('threshold','work_seconds',Quantity.point(900,'discarded comparator'),Quantity(290,310),
        'physical original-file validation only')
    _,missing = load_history(database,config,as_of,old,
        since=min(valid)-timedelta(days=30),run_transform=collect)
    for key,row in targets.items():
        if key not in actuals:
            row['actual_status'] = 'missing_original_evidence'
            continue
        actual = actuals[key]
        trace,run = actual['trace'],actual['run']
        layout = actual['layout']
        row['actual'] = dict(elapsed_seconds=trace['elapsed_seconds'],hr_coverage=trace['coverage'],
            available_at=max(run.end,created[key]),
            load_bounds=trace['load_bounds_within_reference'],layout=layout,
            work_bouts_seconds=[b-a for k,a,b in layout if k=='work'],
            work_seconds=sum(b-a for k,a,b in layout if k=='work') if layout else None,
            distance_miles=run.miles)
        observations=phase_observations(layout,trace['normalized_spans']) if layout else ()
        row['actual']['phase_loads']=[dict(kind=p.kind,seconds=p.seconds,known_seconds=p.known_seconds,load=p.load)
            for p in observations]
        if row['issued_at'] is None:
            continue
        contract = row['contract']
        if contract['layout'] is None:
            row['forecast_status'] = 'unknown_prescription_timing_not_filled_from_completion'
            continue
        # Freeze at actual issuance, not the activity start or a future fitness-state field.
        forecast = forecast_direct(contract['layout'],history,as_of=row['issued_at'],accounting_key=reference.accounting_key)
        simple = forecast_direct(contract['layout'],history,as_of=row['issued_at'],accounting_key=reference.accounting_key,conditioned=False)
        row.update(forecast_status=forecast['status'],forecast=forecast,phase_only_control=simple)
        bounds = trace['load_bounds_within_reference']
        row['whole_plan_vs_execution'] = whole_load_error(forecast['predicted_load'],bounds['low'],bounds['high'])
        row['whole_plan_phase_only_vs_execution'] = whole_load_error(simple['predicted_load'],bounds['low'],bounds['high'])
        row['execution_differences'] = dict(elapsed_seconds_delta=trace['elapsed_seconds']-contract['total_seconds'],
            prescribed_reps=contract['repetitions'],detected_reps=len(row['actual']['work_bouts_seconds']),
            work_seconds_delta=row['actual']['work_seconds']-contract['work_seconds_total'] if layout else None,
            full_adherence_established=False)
        # Separate diagnostic: realized timing, same issue-frozen evidence. Not a pre-run forecast.
        conditional = forecast_direct(layout,history,as_of=row['issued_at'],accounting_key=reference.accounting_key)
        row['actual_timing_diagnostic'] = dict(forecast=conditional,whole_load_error=whole_load_error(
            conditional['predicted_load'],bounds['low'],bounds['high']))
        if conditional['status']=='predicted':
            by_kind={}
            for kind in ('warmup','work','recovery','cooldown'):
                pairs=[(p,o) for p,o in zip(conditional['phases'],observations) if p['kind']==kind]
                if not pairs:continue
                known=sum(o.known_seconds for _,o in pairs)
                observed=sum(o.load or 0 for _,o in pairs)
                predicted=sum(p['rate']*o.known_seconds for p,o in pairs)
                by_kind[kind]=dict(known_seconds=known,observed_load=observed if known else None,
                    predicted_load=predicted if known else None,
                    fraction_error=predicted/observed-1 if observed>0 else None)
            row['actual_timing_diagnostic']['phase_errors']=by_kind
        if execution_observer is not None:
            execution_observer(row,run,layout,trace,history,reference)
    if sha256(database.read_bytes()).hexdigest()!=digest:
        raise AssertionError('Source database changed')
    return dict(version='issued_quality_load_validation_v2',as_of=as_of,cases=list(targets.values()),
        quality_inventory=len(quality_ids),quality_without_saved_matched_assignment=sorted(quality_ids-set(targets)),
        first_saved_plan_at=plan_min,missing_sources=missing,database_unchanged=True,planner_influence=False,
        limitations=['Retrospective matches identify saved candidates, not proof of the precise version viewed or intended',
            'Activity creation and current primary-source import bound availability; historical settings/label edits and primary-source replacement not reconstructed',
            'Exact saved timing assessed separately from realized timing; deviations are not pure load-prediction errors',
            'No fabricated prescriptions for old runs, variable fartlek or different completed workouts',
            'No execution uncertainty interval inferred from a few mismatched runs; rate dispersion is not confidence',
            'No tuning, live writes, rollout or scheduler stability claim'])


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--database',type=Path,default=Path('data/run_analysis.sqlite'))
    parser.add_argument('--as-of',type=datetime.fromisoformat,default=datetime(2026,9,26,4,tzinfo=timezone.utc))
    parser.add_argument('--manifest',type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output path')
    if args.as_of.tzinfo is None:parser.error('Timezone-aware cutoff required')
    config=load_config(Path('config.yaml'))
    manifest=None
    if args.manifest:
        from scripts.research_manifest import freeze_sources
        manifest=freeze_sources(args.manifest,config=config,entrypoint=__name__,inputs=(args.database,))
    result = audit(args.database,config,as_of=args.as_of)
    if manifest:result['manifest']=manifest
    with args.output.open('x') as stream:
        json.dump(result,stream,default=encode,indent=2,allow_nan=False)
    print(json.dumps([dict(key=r['key'],role=r['current_actual_role'],contract=r['contract']['status'],
        forecast=r.get('forecast_status'),comparison=r.get('whole_plan_vs_execution'),differences=r.get('execution_differences'))
        for r in result['cases']],indent=2))


if __name__=='__main__':
    main()
