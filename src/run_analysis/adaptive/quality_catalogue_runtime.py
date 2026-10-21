"""Explicit offline catalogue integration with the existing daily harness.

The caller freezes stage ownership, raw evidence and pace inputs. No live
activation, family multipliers, implicit terrain pace or new search engine.
"""
from dataclasses import asdict, replace
from datetime import datetime, timedelta
import json
from math import fsum
from pathlib import Path
from functools import lru_cache

from run_analysis.phase_calibration import clip_trace
from run_analysis.session_decision import CoachingContract, DoseLimits
from run_analysis.signal_structure import SignalSpan
from run_analysis.work_stimulus import CappedStimulusTarget, WorkSignals
from run_analysis.workload_accounting import Quantity
from .quality_block_contract import read_quality_contract
from .quality_catalogue_admission import compile_admitted_catalogue
from .quality_forecast_runtime import VERSION, read_profiles
from .quality_prescription_contract import PaceReference
from .quality_program_contract import PurposeDelivery, QualityProgramme
from .quality_purpose_contract import PurposeRequirement, QualityPurposeContract, assess_quality_purposes
from .quality_purpose_forecast import forecast_purposes
from .work_exposure_matching import band_fraction


def json_value(value):
    return json.loads(json.dumps(value,default=lambda x:asdict(x) if hasattr(x,'__dataclass_fields__')
        else x.isoformat(),allow_nan=False))


def exposure_pace(history,phase_history,profiles,target,*,as_of,days=30):
    """One vote per causal session, weighted by measured band overlap.

    Whole-bout pace is an extent proxy, not proof of commanded effort or an
    identified physiological threshold. Freeze once; never refit to winners.
    """
    by_key={h.key:h for h in phase_history};rows=[]
    for r in history:
        if not (r.role=='quality' and as_of-timedelta(days=days)<=r.start<r.end<as_of
                and r.available_at<=as_of and r.key in profiles and r.key in by_key):
            continue
        matched=[(o,t) for o,t in zip(by_key[r.key].phases,profiles[r.key]) if o.kind=='work']
        if len(matched)!=len(r.bouts):
            raise ValueError('Pace and raw exposure phases must share physical bout ownership')
        values=[]
        for (o,t),b in zip(matched,r.bouts):
            if o.load is None or not b.distance_meters:
                continue
            fraction=band_fraction(t.spans,t.start,t.end,target.start_band,target.finish_band)
            if fraction:
                values.append((fraction,b.work_seconds/b.distance_meters))
        if values:
            weight=fsum(w for w,_ in values)
            rows.append((r.key,weight/len(values),fsum(w*v for w,v in values)/weight))
    if not rows:
        return None
    nominal=fsum(w*v for _,w,v in rows)/fsum(w for _,w,_ in rows)
    source='Frozen whole-bout pace weighted by observed HR-band overlap; not calibrated cue-response or confidence'
    return PaceReference(Quantity(min(v for _,_,v in rows),max(v for _,_,v in rows),nominal,source),
        as_of,tuple(k for k,_,_ in rows),source)


def catalogue_doses(admission,*,issued_at,anchor,coaching,**recipe_inputs):
    generated=compile_admitted_catalogue(admission,issued_at=issued_at,**recipe_inputs)
    doses=[];contracts={}
    for c in generated['candidates']:
        dose,binding=dose_from_prescription(c.prescription,stage_key=admission.stage.key,
            recipe=c.recipe_key,anchor=anchor,coaching=coaching)
        contracts[dose.key]=binding
        doses.append(dose)
    return tuple(doses),contracts,generated


def dose_from_prescription(p,*,stage_key,recipe,anchor,coaching):
    from .new_model_daily_harness import Dose
    definition=p.definition()
    definition.update(calendar_integration=VERSION,load_model='shared measured work-band forecast',
        requested_effort_conditioned=True,purpose_accounting='separate sustained/faster exposure; one opportunity',
        accepted_stage_key=stage_key,recipe_key=recipe)
    scope=replace(anchor.forecast_scope,duration_source=VERSION+';frozen numeric commands',
        movement_source=VERSION+';explicit per-purpose extent scenarios')
    dose=Dose(p.key,'quality',p.miles,tuple((b.kind,b.seconds.estimate) for b in p.phases),scope,
        tuple((b.seconds.low,b.seconds.high) for b in p.phases),definition)
    work=[b.seconds.estimate for b in p.phases if b.kind=='work']
    ratios=[p.phases[i+1].seconds.estimate/b.seconds.estimate for i,b in enumerate(p.phases[:-1])
        if b.kind=='work' and p.phases[i+1].kind=='recovery'] or [0.]
    limits=DoseLimits(Quantity.point(p.miles,VERSION),Quantity.point(fsum(work),VERSION),
        Quantity(min(work),max(work)),Quantity(min(ratios),max(ratios)))
    # Outer admission must precede this recipe-shape binding. It is not a
    # stand-alone authority to create larger doses or advance a stage.
    return dose,CoachingContract(p.request.issued_at,coaching.source+';'+VERSION,
        coaching.purpose,coaching.intensity_key,limits,None)


def retained_progression(h,*,issued_at):
    """Refine the existing independently bound progression envelope, not Z4 minutes.

    Preserve its authored gradual profile and first/second halves. Use the old
    ordinary/reduced committed distance limits; do not increase quality growth.
    """
    from .quality_block_contract import adapt_v2
    from .quality_recipe_catalogue import DoseWindow, progression_options
    candidates=[d for d in h.catalog.values() if d.workout_definition
        and d.workout_definition['family']=='progression'
        and d.workout_definition.get('numeric_contract')]
    if not candidates:return (),{}
    anchor=max(candidates,key=lambda d:(d.miles,d.key));binding=h.coaching_contracts[anchor.key]
    original=read_quality_contract(anchor.workout_definition['numeric_contract'])
    p=adapt_v2(original) if original.version!='quality_block_contract_v3' else original
    p=replace(p,request=replace(p.request,issued_at=issued_at))
    minimum=min(d.miles for d in candidates);maximum=binding.ordinary.miles.high
    if abs(maximum-anchor.miles)>1e-8:
        raise ValueError('Existing progression must own its maximum committed distance')
    work=fsum(b.seconds.estimate for b in p.phases if b.kind=='work')
    window=DoseWindow(1.,work,maximum,fsum(b.seconds.estimate for b in p.phases),1,
        'Preserved pre-existing progression ordinary/reduced envelope; not a harder-work minute ceiling')
    options=progression_options(p,tuple(i/10 for i in range(round(minimum*10),round(maximum*10)+1)),window=window)
    pairs=[dose_from_prescription(c.prescription,stage_key=h.stage.key,recipe='progression',
        anchor=anchor,coaching=binding) for c in options]
    return tuple(d for d,_ in pairs),{d.key:c for d,c in pairs}


def read_contract(r):
    def target(q):
        t=CappedStimulusTarget(**{k:v for k,v in q.items() if k!='version'})
        if q.get('version')!=t.version:raise ValueError('Unknown target version')
        return t
    return QualityPurposeContract(datetime.fromisoformat(r['issued_at']),r['stage_key'],r['cadence_days'],
        tuple(PurposeRequirement(**dict(q,target=target(q['target']))) for q in r['requirements']),r['source'])


def read_delivery(r):
    return PurposeDelivery(r['key'],*(Quantity(**r[k]) for k in ('sustained','faster','opportunity')))


def delivery_from(report,key):
    return PurposeDelivery(key,report['credits']['sustained'],report['credits']['faster'],report['opportunity'])


@lru_cache(maxsize=32)
def _original_completion_trace(path, digest, start, timezone, movement_json, resting_hr, maximum_hr, policy):
    """Pure source analysis keyed by verified content and every normalization input."""
    from run_analysis.importer import parser_for, sha256_file
    from run_analysis.movement import classify_movement
    from run_analysis.training_signals import HeartRateReference, observed_cardio_trace
    parsed=parser_for(path)(path,default_timezone=timezone)
    activity=next(a for a in parsed.activities if a.start_time_utc==start)
    intervals=classify_movement(activity.trackpoints,json.loads(movement_json)).intervals
    trace=observed_cardio_trace(activity,intervals,HeartRateReference(resting_hr,maximum_hr,
        'frozen sandbox normalization'),policy,include_spans=True,compute_density=False)
    bad={'large_recording_gap','distance_reset','implausible_distance_speed'}
    speeds=tuple(SignalSpan((i.start.timestamp_utc-start).total_seconds(),
        (i.end.timestamp_utc-start).total_seconds(),i.distance_m/i.elapsed_s,i.distance_m/i.elapsed_s)
        for i in intervals if i.elapsed_s>0 and abs(i.moving_time_s-i.elapsed_s)<1e-8
        and not bad&set(i.flags) and i.distance_source!='unknown')
    if sha256_file(path)!=digest:
        raise ValueError('Original source changed during completion analysis')
    return tuple(trace['normalized_spans']),speeds


def completion_signals(h,as_of):
    """Use the version-owned original source and recorded physical windows.

    No assignment title/effort becomes observed evidence. The replay driver may
    supply explicit synthetic physical phases; that provenance remains separate.
    """
    from run_analysis.importer import sha256_file
    from .program_evidence_adapter import read_accounted
    result={};profiles={}
    for row in h.sandbox.completion_rows(as_of):
        run=read_accounted(json.loads(row['accounted']))
        if run.role!='quality':continue
        paths=h.sandbox.db.execute('''SELECT s.path FROM source_files s JOIN activity_sources a
            ON s.id=a.source_file_id WHERE a.activity_id=? AND s.sha256=?''',
            (row['activity_id'],row['source_digest'])).fetchall()
        if len(paths)!=1 or sha256_file(Path(paths[0][0]))!=row['source_digest']:
            raise ValueError('Current causal completion needs its exact original source')
        path=Path(paths[0][0])
        spans,speeds=_original_completion_trace(path,row['source_digest'],run.start,
            h.config['timezone_default'],json.dumps(h.config['moving_time'],sort_keys=True),
            h.config['resting_hr'],h.config['max_hr'],h.ref.policy)
        phases=json.loads(row['phase_history'])['phases'];offset=0.;windows=[];local=[]
        from .work_exposure_matching import PhaseTrace
        for p in phases:
            end=offset+p['seconds'];local.append(PhaseTrace(offset,end,clip_trace(spans,offset,end)))
            if p['kind']=='work':windows.append((offset,end))
            offset=end
        result[run.key]=WorkSignals(run.key,'observed',run.start,run.end,run.available_at,
            tuple(windows),spans,speeds,run.work_evidence_basis)
        profiles[run.key]=tuple(local)
    return result,profiles


def decision_evidence(h,as_of):
    runtime=h.quality_runtime
    if datetime.fromisoformat(runtime['evidence']['available_at'])>as_of:
        raise ValueError('Future catalogue inputs cannot enter the decision')
    signals,profiles=completion_signals(h,as_of)
    from .direct_quality_load import POLICY
    accepted={r.key for r in (*h.phase_history,*h.sandbox.completed_history(as_of))
        if as_of-timedelta(days=POLICY.current_days)<=r.start<r.end<as_of and r.available_at<=as_of}
    all_profiles={**runtime['evidence']['profiles'],**json_value(profiles)}
    # Exact causal-window reduction: profiles outside the load predictor's
    # accepted source set cannot affect any phase. Keep the full frozen seeds
    # in runtime for provenance, not in every issued command's forecast input.
    evidence=dict(runtime['evidence'],profiles={k:v for k,v in all_profiles.items() if k in accepted})
    # Reprice future prescriptions against CURRENT zone boundaries, while
    # retaining the accepted HRR measurement reference and immutable earlier
    # forecasts. Recorded profiles are still in that same accounting system.
    evidence['config']=dict(evidence['config'], zones=h.config['zones'])
    return evidence,signals


def programme(h,history,forecasts,as_of,signals):
    runtime=h.quality_runtime;contract=read_contract(runtime['purpose_contract'])
    observed={r['key']:read_delivery(r) for r in runtime['observed_purposes']}
    for key,signal in signals.items():
        observed[key]=delivery_from(assess_quality_purposes(signal,contract=contract,role='quality',as_of=as_of),key)
    needed={r.key for r in h.history if r.role=='quality'}
    if needed-set(observed):
        raise ValueError('Unassessed physical quality evidence: '+str(sorted(needed-set(observed))))
    planned=[];profiles=read_profiles(h.quality_decision_evidence['profiles'])
    for key,forecast in forecasts.items():
        dose=h.catalog[key]
        if dose.role!='quality':continue
        p=read_quality_contract(dose.workout_definition['numeric_contract'])
        # The forecast already owns CURRENT causal sources. Keep the original
        # command identity and issuance, as well as pace/profile and units.
        result=forecast_purposes(p,forecast,history,profiles,h.config,contract=contract)
        planned.append(delivery_from(result,key))
    return QualityProgramme(contract,tuple(observed[k] for k in sorted(needed)),tuple(planned))
