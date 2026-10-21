"""Explicit dated corrections to adopted evidence, never rewritten decisions."""
from dataclasses import replace
from datetime import datetime
from hashlib import sha256
import json

from run_analysis.athlete_projection import aware
from .program_evidence_adapter import read_accounted
from .direct_quality_load import PhaseHistory,PhaseDose


def initialize(db):
    db.execute('''CREATE TABLE IF NOT EXISTS sandbox_seed_revisions (
        key TEXT NOT NULL, predecessor TEXT NOT NULL, available_at TEXT NOT NULL,
        serialized TEXT NOT NULL, fingerprint TEXT NOT NULL,
        PRIMARY KEY(key,predecessor))''')


def read_phase(value):
    return PhaseHistory(**dict(value,
        **{k:datetime.fromisoformat(value[k]) for k in ('start','end','available_at')},
        phases=tuple(PhaseDose(**p) for p in value['phases'])))


def revisions(sandbox,as_of):
    from .execution_upload_sandbox import encode
    aware(as_of)
    result=[]
    for row in sandbox.db.execute('SELECT * FROM sandbox_seed_revisions ORDER BY available_at,key'):
        if sha256(row['serialized'].encode()).hexdigest()!=row['fingerprint']:
            raise ValueError('Seed correction fingerprint changed; reconciliation required')
        value=json.loads(row['serialized'])
        if (value['key'],value['predecessor'],value['available_at'])!=(row['key'],row['predecessor'],row['available_at']):
            raise ValueError('Seed correction identity changed')
        if (sha256(encode(value['before']).encode()).hexdigest()!=value['predecessor']
                or value['after']['available_at']!=value['available_at']
                or value['after_phase']['available_at']!=value['available_at']):
            raise ValueError('Seed correction predecessor or availability changed')
        if datetime.fromisoformat(row['available_at'])<=as_of:
            result.append(value)
    return result


def phase_records(sandbox,as_of):
    """Original immutable phase records plus only corrections known at issuance."""
    result={r['key']:r['serialized'] for r in sandbox.db.execute('SELECT * FROM sandbox_seed_history')}
    from .execution_upload_sandbox import encode
    for value in revisions(sandbox,as_of):
        key=value['key']
        if result.get(key)!=encode(value['before_phase']):
            raise ValueError('Seed phase correction predecessor changed')
        result[key]=encode(value['after_phase'])
    return result


def apply(harness,as_of):
    """Overlay a corrected seed under its SAME physical identity exactly once."""
    from .execution_upload_sandbox import encode
    grouped={}
    for value in revisions(harness.sandbox,as_of):
        group=grouped.setdefault(value['key'],[])
        if group and (group[-1]['after']!=value['before'] or group[-1]['after_phase']!=value['before_phase']):
            raise ValueError('Seed correction chain changed')
        group.append(value)
    for key,group in grouped.items():
        old=next((r for r in harness.history if r.key==key),None)
        if key not in harness.seed_keys or old is None:
            raise ValueError('Seed correction does not own this physical history')
        allowed={encode(v[part]) for v in group for part in ('before','after')}
        if encode(old) not in allowed:
            raise ValueError('Seed physical correction predecessor changed')
        old_phase=next((p for p in harness.phase_history if p.key==key),None)
        allowed_phases={encode(v[part]) for v in group for part in ('before_phase','after_phase')}
        if old_phase is None or encode(old_phase) not in allowed_phases:
            raise ValueError('Seed phase correction predecessor changed')
        harness.history[harness.history.index(old)]=read_accounted(group[-1]['after'])
        phase=read_phase(group[-1]['after_phase'])
        harness.phase_history=tuple(p for p in harness.phase_history if p.key!=key)+(phase,)


def reconcile(harness,replacement,phase,*,expected_fingerprint,available_at,source_sha256,reason):
    """Append an explicitly reviewed role/work correction with unchanged raw load.

    No source replacement, synthetic dose, new completion, stage re-adoption,
    historical cutoff rewrite, or implicit response/readiness change is allowed.
    """
    from .execution_upload_sandbox import encode
    aware(available_at)
    if (not reason.strip() or len(source_sha256)!=64
            or any(c not in '0123456789abcdef' for c in source_sha256)):
        raise ValueError('Correction requires a reason and verified original source hash')
    if available_at<=harness.last_as_of or available_at<harness.evidence_as_of:
        raise ValueError('Seed correction cannot backdate current decisions or evidence')
    key=replacement.key
    old=next((r for r in harness.history if r.key==key),None)
    prior=next((p for p in harness.phase_history if p.key==key),None)
    if key not in harness.seed_keys or old is None or prior is None:
        raise ValueError('Existing adopted physical and phase evidence required')
    new=replace(replacement,available_at=available_at)
    after_phase=replace(phase,available_at=available_at)
    immutable=('key','start','end','miles','accounting_key','quality_reference_key','source',
        'command_running_minutes','advisory_timezone')
    if any(getattr(old,k)!=getattr(new,k) for k in immutable) or old.demand[:2]!=new.demand[:2]:
        raise ValueError('Role correction must preserve original identity, movement and cardio load')
    if (new.source!='observed' or new.role!='quality' or not new.bouts
            or new.work_evidence_basis not in {'recorded_continuous_lap','recorded_lap_contrast'}):
        raise ValueError('Independent recorded quality work required, not a manual title alone')
    if (after_phase.key,after_phase.start,after_phase.end,after_phase.accounting_key)!=(
            old.key,old.start,old.end,old.accounting_key):
        raise ValueError('Corrected phase must describe the same session')
    if abs(sum(p.load or 0 for p in prior.phases)-sum(p.load or 0 for p in after_phase.phases))>1e-8:
        raise ValueError('Phase correction must preserve measured cardio integral')
    predecessor=sha256(encode(old).encode()).hexdigest()
    if predecessor!=expected_fingerprint:
        raise ValueError('Expected physical predecessor changed')
    body=dict(key=key,predecessor=predecessor,available_at=available_at.isoformat(),
        before=json.loads(encode(old)),after=json.loads(encode(new)),
        before_phase=json.loads(encode(prior)),after_phase=json.loads(encode(after_phase)),
        source_sha256=source_sha256,reason=reason)
    serialized=encode(body);fingerprint=sha256(serialized.encode()).hexdigest()
    with harness.sandbox.db:
        harness.sandbox.db.execute('BEGIN IMMEDIATE')
        # Check the current persisted phase chain before committing anything.
        records=phase_records(harness.sandbox,available_at)
        if records.get(key)!=encode(prior):
            raise ValueError('Persisted phase predecessor changed')
        harness.sandbox.db.execute('INSERT INTO sandbox_seed_revisions VALUES (?,?,?,?,?)',
            (key,predecessor,available_at.isoformat(),serialized,fingerprint))
    apply(harness,available_at)
    harness.evidence_as_of=available_at
    return dict(key=key,fingerprint=fingerprint,predecessor=predecessor,available_at=available_at.isoformat())


def reconcile_source(harness,replacement,phase,*,activity_id,old_digest,new_digest,
                     available_at,proof):
    """Dated richer-original correction; update bridge identity in the SAME commit.

    The bridge verifies both retained original files and importer deduplication.
    Unlike a role-only correction, measured movement/cardio may legitimately
    improve. Frozen normalization, physical identity and issued history may not.
    """
    from .execution_upload_sandbox import encode
    aware(available_at)
    key=replacement.key
    old=next((r for r in harness.history if r.key==key),None)
    prior=next((p for p in harness.phase_history if p.key==key),None)
    if key not in harness.seed_keys or old is None or prior is None:
        raise ValueError('Richer source needs existing adopted physical and phase evidence')
    if (available_at<=harness.last_as_of or available_at<harness.evidence_as_of
            or available_at<replacement.end):
        raise ValueError('Source correction cannot backdate decisions or physical evidence')
    if (proof.get('old_sha256'),proof.get('new_sha256'),proof.get('activity_id'),
            proof.get('duplicate_reason'))!=(old_digest,new_digest,activity_id,
            'richer_duplicate_replaced_canonical'):
        raise ValueError('Verified richer original source proof required')
    new=replace(replacement,available_at=available_at)
    after_phase=replace(phase,available_at=available_at)
    if proof.get('manual_workout_type') is not None and new.role!=old.role:
        raise ValueError('Manual role change requires separate physical-evidence reconciliation')
    if (new.source!='observed' or any(getattr(old,k)!=getattr(new,k)
            for k in ('key','start','accounting_key','quality_reference_key','advisory_timezone'))):
        raise ValueError('Source correction must preserve physical identity and frozen normalization')
    if (after_phase.key,after_phase.start,after_phase.end,after_phase.accounting_key)!=(
            new.key,new.start,new.end,new.accounting_key):
        raise ValueError('Replacement phase must describe the same measured session')
    predecessor=sha256(encode(old).encode()).hexdigest()
    body=dict(key=key,predecessor=predecessor,available_at=available_at.isoformat(),
        before=json.loads(encode(old)),after=json.loads(encode(new)),
        before_phase=json.loads(encode(prior)),after_phase=json.loads(encode(after_phase)),
        source_sha256=new_digest,source_replacement=proof,
        reason='Verified richer original FIT for the same application activity')
    serialized=encode(body)
    with harness.sandbox.db:
        harness.sandbox.db.execute('BEGIN IMMEDIATE')
        if phase_records(harness.sandbox,available_at).get(key)!=encode(prior):
            raise ValueError('Persisted source predecessor changed')
        updated=harness.sandbox.db.execute('''UPDATE adaptive_app_sources SET source_sha256=?
            WHERE activity_id=? AND physical_key=? AND source_sha256=?''',
            (new_digest,activity_id,key,old_digest))
        if updated.rowcount!=1:
            raise ValueError('Tracked source predecessor changed')
        harness.sandbox.db.execute('INSERT INTO sandbox_seed_revisions VALUES (?,?,?,?,?)',
            (key,predecessor,available_at.isoformat(),serialized,sha256(serialized.encode()).hexdigest()))
    apply(harness,available_at)
    harness.evidence_as_of=available_at
