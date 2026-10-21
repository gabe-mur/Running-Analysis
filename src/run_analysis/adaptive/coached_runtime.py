"""Durable command/receipt ownership in the existing decision journal.

Decisions own issued assignments atomically with their real issued fingerprints.
Receipts are reproducibly derived from the durable original completions; retries
cannot increment a mutable counter twice and corrections cannot retain old credit.
"""
from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
import sqlite3

from .quality_coaching_policy import freeze_sequence, read_sequence
from .quality_assignment_accounting import read_assignment,reconcile_assignment,advance_sequence
from .quality_catalogue_runtime import completion_signals,read_contract
from .quality_purpose_contract import assess_quality_purposes
from .program_evidence_adapter import read_accounted


def offline_guard(h):
    from run_analysis.config import resolve_project_path
    db=resolve_project_path(Path.cwd(),h.config['paths']['database'])
    if db.exists():
        with sqlite3.connect(db.resolve().as_uri()+'?mode=ro',uri=True) as connection:
            if connection.execute("SELECT 1 FROM sqlite_master WHERE name='app_state'").fetchone():
                row=connection.execute("SELECT value_json FROM app_state WHERE key='adaptive_runtime'").fetchone()
                if row and Path(json.loads(row[0])['root']).resolve()==h.sandbox.root.resolve():
                    certificate=h.sandbox.root/'coached-release.json'
                    if not certificate.exists():
                        raise ValueError('Coached runtime cannot activate or publish on the live server')
                    release=json.loads(certificate.read_text())
                    if (release.get('version')!='coached_live_v206' or
                            Path(release.get('database','')).resolve()!=db.resolve() or
                            release.get('coaching_base_key')!=h.coaching_base.key or
                            not release.get('validated_app_copy')):
                        raise ValueError('Coached live release certificate differs from its verified owner')
    if not (h.sandbox.root/'execution-sandbox.json').exists():
        raise ValueError('Explicit offline execution sandbox required')


def enable(h, *, first_purpose='sustained'):
    offline_guard(h)
    if getattr(h,'quality_runtime',None) is None or h.stage is None:
        raise ValueError('Accepted native catalogue and stage required')
    if getattr(h,'coaching_base',None) is not None:
        raise ValueError('Coached runtime already owns an issued sequence')
    from .quality_forecast_runtime import VERSION
    from .quality_choices import allowed, quality_focus
    catalog={k:d for k,d in h.catalog.items() if d.workout_definition and
        d.workout_definition.get('calendar_integration')==VERSION and
        allowed(d,h.coaching_contracts[k],h.config)}
    h.coaching_base=freeze_sequence(catalog,stage_key=h.stage.key,issued_at=h.last_as_of.isoformat(),
        ordinary_miles=h.training_policy.ordinary_miles,
        quality_work_minutes=h.training_policy.quality_work_minutes,first_purpose=first_purpose,
        cadence_contract='proportional_grace_v1', focus=quality_focus(h.config))
    h.coaching_sequence=h.coaching_base
    h.coaching_receipts=()
    h.coaching_stage_keys=(h.stage.key,)
    h.coaching_epochs=(h.coaching_base,)


def revise_preferences(h, as_of):
    """Future menu revision; old assignments and their receipts remain owned.

    A menu edit starts a new variety epoch, not a physiological reset. Earlier
    completions retain their physical demand and dose credit, but cannot consume
    a command issued by the new epoch. No edit means no sequence change.
    """
    if getattr(h,'coaching_base',None) is None:return
    from .quality_choices import allowed, quality_focus
    from .quality_forecast_runtime import VERSION
    from .quality_coaching_policy import FAMILIES
    catalog={k:d for k,d in h.catalog.items() if d.workout_definition and
        d.workout_definition.get('calendar_integration')==VERSION and
        allowed(d,h.coaching_contracts[k],h.config)}
    families={d.workout_definition['family'] for d in catalog.values()}
    focus=quality_focus(h.config)
    if families=={r.family for r in h.coaching_sequence.ordinary} and focus==h.coaching_sequence.focus:return
    missing=[p for p,fs in FAMILIES.items() if not families.intersection(fs)]
    if missing:
        # The user deferred the all-families-disabled policy. Keep Settings
        # editable and the issued journal intact; never silently override it.
        raise ValueError('Quality preferences need review: no enabled '+
            ' or '.join(missing)+' workout family. This preference combination is not supported yet.')
    current=h.coaching_sequence
    if as_of<=datetime.fromisoformat(current.issued_at):
        raise ValueError('Quality menu revision must follow its previous issue')
    accepted=h.quality_runtime['admission']['stage']
    epoch=freeze_sequence(catalog,stage_key=h.stage.key,issued_at=as_of.isoformat(),
        ordinary_miles=accepted['ordinary_miles'],quality_work_minutes=accepted['quality_work_minutes'],
        first_purpose=current.assignment(0).purpose,cadence_contract=current.cadence_contract,focus=focus)
    h.coaching_epochs=(*getattr(h,'coaching_epochs',(h.coaching_base,)),epoch)
    h.coaching_sequence=epoch


def rebind_stage(proposed, parent, as_of):
    """Volume-stage acceptance keeps the issued quality policy and ordinal.

    This is an in-memory preview only; the existing atomic stage/calendar
    transaction commits the lineage. Changing quality policy is a separate
    revision, not a side effect of confirming or growing aerobic volume.
    """
    if getattr(parent,'coaching_base',None) is None:return
    offline_guard(proposed)
    keys=getattr(parent,'coaching_stage_keys',(parent.coaching_base.stage_key,))
    if (keys[-1]!=parent.stage.key or parent.coaching_sequence.stage_key!=parent.stage.key
            or proposed.stage.key==parent.stage.key or proposed.stage.key in keys
            or proposed.stage.issued_at>as_of):
        raise ValueError('Coaching rebind requires a new available child of the current stage')
    if proposed.quality_runtime!=parent.quality_runtime:
        raise ValueError('Volume progression cannot silently revise quality forecasting policy')
    for command in parent.coaching_base.ordinary:
        key=command.key
        if (proposed.catalog.get(key)!=parent.catalog.get(key) or key not in proposed.catalog
                or proposed.coaching_contracts.get(key)!=parent.coaching_contracts.get(key)):
            raise ValueError('Volume progression must preserve issued quality commands and contracts')
    proposed.coaching_base=parent.coaching_base
    proposed.coaching_sequence=replace(parent.coaching_sequence,stage_key=proposed.stage.key)
    proposed.coaching_receipts=parent.coaching_receipts
    proposed.coaching_stage_keys=(*keys,proposed.stage.key)
    proposed.coaching_epochs=getattr(parent,'coaching_epochs',(parent.coaching_base,))


def restore_runtime(h, runtime, state):
    h.coaching_base=read_sequence(runtime['coaching_base']) if runtime.get('coaching_base') else None
    h.coaching_sequence=state.quality_coaching
    h.coaching_receipts=state.quality_completions
    if bool(h.coaching_base)!=bool(h.coaching_sequence):
        raise ValueError('Coaching journal and frozen scoring contract must agree')
    if h.coaching_base is not None:
        offline_guard(h)
        h.coaching_stage_keys=tuple(runtime.get('coaching_stage_keys',(h.coaching_base.stage_key,)))
        keys=h.coaching_stage_keys
        if (not keys or keys[0]!=h.coaching_base.stage_key or keys[-1]!=h.stage.key
                or h.coaching_sequence.stage_key!=h.stage.key or len(set(keys))!=len(keys)):
            raise ValueError('Coaching lineage must retain its original policy and current accepted stage')
        h.coaching_epochs=tuple(read_sequence(r) for r in runtime.get('coaching_epochs',
            (runtime['coaching_base'],)))
        epochs=h.coaching_epochs
        if (not epochs or epochs[0]!=h.coaching_base or
                any(e.stage_key not in keys or e.consumed!=0 for e in epochs) or
                any(datetime.fromisoformat(a.issued_at)>=datetime.fromisoformat(b.issued_at)
                    for a,b in zip(epochs,epochs[1:])) or
                len(epochs)>1 and datetime.fromisoformat(epochs[-1].issued_at)>state.as_of):
            raise ValueError('Invalid durable quality preference lineage')
        if h.coaching_sequence!=replace(epochs[-1],stage_key=h.stage.key,
                consumed=h.coaching_sequence.consumed):
            raise ValueError('Volume-stage rebind cannot change the frozen quality policy')
        parents={r['key']:r['parent'] for r in h.sandbox.db.execute('SELECT key,parent FROM stages')}
        current=h.sandbox.db.execute('SELECT key FROM current_stage WHERE singleton=1').fetchone()
        if (current is None or current['key']!=keys[-1] or any(k not in parents for k in keys)
                or any(parents[child]!=parent for parent,child in zip(keys,keys[1:]))):
            raise ValueError('Coaching stage lineage differs from the atomic accepted-stage ledger')


def sync(h, as_of):
    base=getattr(h,'coaching_base',None)
    if base is None:return
    keys=getattr(h,'coaching_stage_keys',(base.stage_key,))
    if not keys or keys[0]!=base.stage_key or keys[-1]!=h.stage.key:
        raise ValueError('Coached stage transition needs explicit command rebinding; do not drop the sequence')
    signals,_=completion_signals(h,as_of)
    contract=read_contract(h.quality_runtime['purpose_contract'])
    receipts=[]
    for row in h.sandbox.completion_rows(as_of):
        day=row['session_key'].removeprefix('slot:')
        owner=next((e for e in h.events if day in e.get('coached_assignments',{}) and
            e['active'].get(day,{}).get('fingerprint')==row['fingerprint']),None)
        if owner is None:continue
        assignment=read_assignment(owner['coached_assignments'][day])
        if (assignment.stage_key not in keys or
                owner['stage_runtime']['stage']['key']!=assignment.stage_key):
            raise ValueError('Completion assignment must belong to its original issued stage')
        actual=read_accounted(json.loads(row['accounted']))
        signal=signals.get(actual.key)
        if actual.role!='quality':matched=False
        elif signal is None:matched=None
        else:
            q=assess_quality_purposes(signal,contract=contract,role=actual.role,as_of=as_of)['credits'][assignment.purpose]
            matched=True if q.low>0 else None if q.high>0 else False
        receipt=reconcile_assignment(assignment,actual,purpose_matched=matched)
        receipt.for_run(actual)
        receipts.append(receipt)
    # Replay from the original issued sequence, not the last mutable counter.
    # A missing rank cannot be consumed by a later assignment out of order.
    sequence=getattr(h,'coaching_epochs',(base,))[-1]
    epoch_at=datetime.fromisoformat(sequence.issued_at)
    for receipt in sorted(receipts,key=lambda r:(r.assignment.rank,r.available_at,r.activity_key)):
        if receipt.assignment.issued_at!=epoch_at:continue
        if receipt.assignment.rank==sequence.consumed:
            sequence=replace(sequence,stage_key=receipt.assignment.stage_key)
            sequence=advance_sequence(sequence,receipt)
    h.coaching_sequence=replace(sequence,stage_key=h.stage.key)
    h.coaching_receipts=tuple(sorted(receipts,key=lambda r:r.activity_key))
