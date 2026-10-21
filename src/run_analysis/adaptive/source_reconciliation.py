"""Verify original FIT enrichment before admitting a dated physical correction."""
from datetime import datetime,timedelta
from hashlib import sha256
from pathlib import Path

from run_analysis.importer import parser_for,_within,NEAR_DUPLICATE_TOLERANCE
from run_analysis.web.upload_service import UploadPayload


def verified_replacement(connection,activity_id,old_digest,row,config,as_of):
    """Importer recognition AND independently matching raw recording required.

    A changed hash alone, deleted evidence, tampered file, manual role title or
    an arbitrary replacement is not authorization to rewrite physical history.
    """
    links=[dict(r) for r in connection.execute('''SELECT s.path,s.sha256,
        l.is_primary,l.duplicate_reason FROM activity_sources l
        JOIN source_files s ON s.id=l.source_file_id WHERE l.activity_id=?''',(activity_id,))]
    old=next((r for r in links if r['sha256']==old_digest),None)
    new=next((r for r in links if r['sha256']==row['sha256'] and r['is_primary']),None)
    if (old is None or new is None or new['duplicate_reason']!='richer_duplicate_replaced_canonical'
            or not new['path'].lower().endswith(('.fit','.fit.gz'))
            or datetime.fromisoformat(row['imported_at_utc'])>as_of):
        raise ValueError('Retained original and importer-verified richer FIT required')
    parsed=[];content=None
    for source in (old,new):
        path=Path(source['path']);raw=path.read_bytes()
        if sha256(raw).hexdigest()!=source['sha256']:
            raise ValueError('Original source bytes differ from the recorded hash')
        parser=parser_for(path)
        activities=parser(path,default_timezone=config['timezone_default']).activities if parser else ()
        if len(activities)!=1 or activities[0].sport!='Running':
            raise ValueError('Exactly one timestamped running recording required')
        parsed.append(activities[0]);content=raw
    before,after=parsed
    if (before.start_time_utc!=after.start_time_utc
            or after.start_time_utc!=datetime.fromisoformat(row['start_time_utc'])
            or any(not _within(getattr(before,k),getattr(after,k),0.,NEAR_DUPLICATE_TOLERANCE)
                for k in ('total_distance_m','lap_recorded_time_s'))
            or any(getattr(r,k) is None for r in parsed
                for k in ('total_distance_m','lap_recorded_time_s'))):
        raise ValueError('Replacement recording does not match original start, distance and duration')
    proof=dict(activity_id=activity_id,old_sha256=old_digest,new_sha256=row['sha256'],
        duplicate_reason=new['duplicate_reason'],identity_basis='raw start/distance/duration and retained importer links')
    manual=connection.execute('''SELECT o.workout_type FROM activities a
        LEFT JOIN run_overrides o ON o.activity_id=COALESCE(NULLIF(a.activity_id,''),a.activity_uid)
        WHERE a.id=?''',(activity_id,)).fetchone()
    proof['manual_workout_type']=manual[0] if manual else None
    return UploadPayload(Path(new['path']).name,content),proof


def reconcile_sources(service,connection,sources,known,as_of):
    h=service.harness
    pending=[]
    for activity_id,(key,digest) in known.items():
        if activity_id not in sources:
            raise ValueError('An existing activity source was removed; original evidence review required')
        row=sources[activity_id]
        if row['sha256']!=digest:
            payload,proof=verified_replacement(connection,activity_id,digest,row,h.config,as_of)
            pending.append((activity_id,key,digest,row,payload,proof))
    seeds=[r for r in h.history if any(key==r.key for _,key,*_ in pending) and r.key in h.seed_keys]
    evidence=None
    if seeds:
        databases=connection.execute('PRAGMA database_list').fetchall()
        database=next(r[2] for r in databases if r[1]=='main')
        if not database:
            raise ValueError('Original application evidence database required')
        from .adaptive_candidate_pipeline import load_athlete
        evidence=load_athlete(Path(database),h.config,as_of,
            since=min(r.start for r in seeds)-timedelta(microseconds=1))
    for activity_id,key,digest,row,payload,proof in pending:
        if key in h.seed_keys:
            from .seed_reconciliation import reconcile_source
            replacement=next((r for r in evidence['history'] if r.key==key),None)
            phase=next((r for r in evidence['phase_history'] if r.key==key),None)
            if replacement is None or phase is None:
                raise ValueError('Richer source lacks admitted original physical/phase evidence')
            reconcile_source(h,replacement,phase,activity_id=activity_id,old_digest=digest,
                new_digest=row['sha256'],available_at=as_of,proof=proof)
        else:
            import json
            completion=next((r for r in h.sandbox.completion_rows(as_of)
                if json.loads(r['actual'])['key']==key),None)
            if completion is None:
                raise ValueError('Tracked physical completion missing')
            day=(completion['session_key'][5:] if completion['session_key'].startswith('slot:') else None)
            # Recovery after the completion committed but before tracking did:
            # replay identical bytes as an ordinary duplicate, not a new
            # correction with a changed predecessor/reason.
            correction=(None if completion['source_digest']==row['sha256'] else
                dict(expected_physical_fingerprint=completion['physical_fingerprint'],
                    reason='Verified richer original FIT; prior source '+digest))
            result=service.upload(payload,day=day,available_at=as_of,replan=False,correction=correction)
            if result['upload']['comparison']['actual_key']!=key:
                raise ValueError('Source replacement introduced another physical identity')
            with h.sandbox.db:
                h.sandbox.db.execute('UPDATE adaptive_app_sources SET source_sha256=? WHERE activity_id=?',
                    (row['sha256'],activity_id))
    if seeds:
        # Match resume_service's evidence view for a stage review in THIS request,
        # not just after restart. Do not re-bootstrap frozen targets or scales.
        service.initial_inputs=dict(service.initial_inputs,
            history=tuple(r for r in h.history if r.key in h.seed_keys),
            sessions=tuple(r for r in h.session_evidence() if r.key in h.seed_keys),
            phase_history=h.phase_history)
