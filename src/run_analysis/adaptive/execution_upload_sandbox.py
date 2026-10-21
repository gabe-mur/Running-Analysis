"""Persistent offline publication → real file upload → measured-state handoff.

No production route, weather lookup, model fit or calendar search. Explicit issue
identity replaces automatic retrospective matching. This is an accounting seam,
not a new scheduler or a default tolerance policy.
"""
from datetime import date,datetime, timedelta
from dataclasses import replace
from copy import deepcopy
from contextlib import nullcontext
import json
from pathlib import Path

from run_analysis.athlete_projection import aware
from run_analysis.continuous_training_dose import ContinuousDoseReference
from run_analysis.db import connect, initialize
from run_analysis.importer import import_files, parser_for, sha256_file
from run_analysis.movement import classify_movement
from run_analysis.training_signals import HeartRateReference, observed_cardio_trace
from run_analysis.web.upload_archive import fit_members
from run_analysis.web.upload_service import UploadPayload, validate_upload, _persist_upload
from run_analysis.workload_accounting import Quantity
from run_analysis.workload_profile import observed_profile
from run_analysis.adaptive.program_evidence_adapter import accounted_phases, read_accounted
from hashlib import sha256
from run_analysis.adaptive.direct_quality_load import PhaseDose, PhaseHistory, phase_observations
from run_analysis.adaptive.execution_forecast_contract import (
    CompletedExecution, FrozenExecution, issue_execution, compare_execution, layout, _encode,
)


def measurement_context(config):
    return json.dumps({k: config[k] for k in ('resting_hr', 'max_hr')}, sort_keys=True)


def encode(value):
    return json.dumps(value, default=lambda v:v.isoformat() if isinstance(v,date) else _encode(v), sort_keys=True, allow_nan=False)


def measured_work_distances(intervals,start,phases):
    """Integrate original distance only within independently supplied work windows.

    Recording gaps, resets, stopped/unknown movement remain unresolved. Never
    derive work distance from whole-run miles or the assigned prescription.
    """
    offset=0.;values=[]
    bad={'large_recording_gap','distance_reset','implausible_distance_speed'}
    for kind,seconds in phases:
        if kind=='work':
            known=meters=0.
            for i in intervals:
                a=(i.start.timestamp_utc-start).total_seconds()
                b=(i.end.timestamp_utc-start).total_seconds()
                overlap=max(0.,min(b,offset+seconds)-max(a,offset))
                if (overlap and i.elapsed_s>0 and abs(i.moving_time_s-i.elapsed_s)<1e-8
                        and not bad&set(i.flags) and i.distance_source!='unknown'):
                    known+=overlap;meters+=i.distance_m*overlap/i.elapsed_s
            values.append(meters if abs(known-seconds)<1e-3 and meters>0 else None)
        offset+=seconds
    return tuple(values)


class ExecutionUploadSandbox:
    """Owns a new, explicitly marked directory; cannot open the live database.

    One upload event is one activity and one explicitly selected issued record.
    Unmatched/batch/corrected evidence needs a separate association workflow.
    Import may persist before comparison fails; workload consumption is atomic
    and retryable, and is keyed by the real importer's deduplicated activity ID.
    """

    def __init__(self, root, config, *, create=False):
        self.root = Path(root)
        self.config = deepcopy(config)
        self.reference = ContinuousDoseReference('sandbox upload accounting')
        self.context = measurement_context(config)
        marker = self.root / 'execution-sandbox.json'
        identity = dict(version=1, measurement_context=self.context,
                        accounting_key=self.reference.accounting_key,
                        timezone=config['timezone_default'], moving_time=config['moving_time'])
        if create:
            self.root.mkdir(parents=True, exist_ok=False)
            with marker.open('x') as stream:
                stream.write(encode(identity))
        elif json.loads(marker.read_text()) != identity:
            raise ValueError('Sandbox measurement configuration changed; explicit migration required')
        self.db = connect(self.root / 'sandbox.sqlite')
        initialize(self.db)
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS sandbox_issues (
                fingerprint TEXT PRIMARY KEY, session_key TEXT NOT NULL,
                issued_at TEXT NOT NULL, serialized TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS sandbox_seed_history (
                key TEXT PRIMARY KEY, serialized TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS sandbox_completions (
                activity_id INTEGER PRIMARY KEY, session_key TEXT NOT NULL UNIQUE,
                fingerprint TEXT NOT NULL, source_digest TEXT NOT NULL,
                available_at TEXT NOT NULL, actual TEXT NOT NULL,
                comparison TEXT NOT NULL, phase_history TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS sandbox_decisions (
                sequence INTEGER PRIMARY KEY,event TEXT NOT NULL,snapshot TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS sandbox_completion_versions (
                activity_id INTEGER NOT NULL,old_fingerprint TEXT NOT NULL,
                corrected_at TEXT NOT NULL,reason TEXT NOT NULL,payload TEXT NOT NULL,
                PRIMARY KEY(activity_id,old_fingerprint));
        ''')
        columns = {r[1] for r in self.db.execute('PRAGMA table_info(sandbox_completions)')}
        for column in ('accounted', 'physical_fingerprint'):
            if column not in columns:
                self.db.execute(f'ALTER TABLE sandbox_completions ADD COLUMN {column} TEXT')
        from .seed_reconciliation import initialize as initialize_seed_revisions
        initialize_seed_revisions(self.db)
        self.db.commit()

    def close(self):
        self.db.close()

    def record(self, fingerprint):
        row = self.db.execute('SELECT serialized FROM sandbox_issues WHERE fingerprint=?',
                              (fingerprint,)).fetchone()
        if row is None:
            raise ValueError('Unknown issued contract; publish before upload')
        return FrozenExecution(row[0], fingerprint)

    def completed_history(self, as_of):
        aware(as_of)
        history = []
        for row in self.completion_rows(as_of):
            h = json.loads(row['phase_history'])
            if datetime.fromisoformat(h['available_at']) <= as_of:
                history.append(PhaseHistory(h['key'], datetime.fromisoformat(h['start']),
                    datetime.fromisoformat(h['end']), datetime.fromisoformat(h['available_at']),
                    h['accounting_key'], tuple(PhaseDose(**p) for p in h['phases'])))
        return tuple(history)

    def completed_accounted(self, as_of):
        """Durable physical handoff; old incomplete artifacts need explicit migration."""
        aware(as_of)
        result = []
        for row in self.completion_rows(as_of):
            if datetime.fromisoformat(row['available_at']) > as_of:
                continue
            if row['accounted'] is None or row['physical_fingerprint'] != sha256(row['accounted'].encode()).hexdigest():
                raise ValueError('Persisted physical evidence missing or changed; explicit migration required')
            actual = read_accounted(json.loads(row['accounted']))
            if actual.available_at != datetime.fromisoformat(row['available_at']):
                raise ValueError('Persisted completion availability mismatch')
            result.append((row['session_key'], actual))
        return tuple(result)

    def completion_rows(self,as_of):
        """Known-at-time version, not today's corrected source rewritten backward."""
        aware(as_of)
        rows=[dict(r) for r in self.db.execute('SELECT * FROM sandbox_completions')]
        rows.extend(json.loads(r[0]) for r in self.db.execute('SELECT payload FROM sandbox_completion_versions'))
        chosen={}
        for row in rows:
            at=datetime.fromisoformat(row['available_at'])
            old=chosen.get(row['activity_id'])
            if at<=as_of and (old is None or at>datetime.fromisoformat(old['available_at'])):
                chosen[row['activity_id']]=row
        return tuple(sorted(chosen.values(),key=lambda r:(r['available_at'],r['activity_id'])))

    def superseded(self,run):
        return any(read_accounted(json.loads(json.loads(r[0])['accounted']))==run
            for r in self.db.execute('SELECT payload FROM sandbox_completion_versions'))

    def publish(self, intent, history=(), *, parent=None, revision_reason=None,_in_batch=False):
        history = tuple(history)
        if (intent.accounting_key, intent.measurement_context) != (self.reference.accounting_key, self.context):
            raise ValueError('Publication must use sandbox accounting and HR normalization')
        # Never let an observed completion be overwritten by a new forecast.
        with nullcontext() if _in_batch else self.db:
            if not _in_batch:
                self.db.execute('BEGIN IMMEDIATE')
            elif not self.db.in_transaction:
                raise ValueError('Batch issuance requires an owning transaction')
            from .seed_reconciliation import phase_records
            current_seed=phase_records(self,intent.issued_at)
            same_issue = self.db.execute('SELECT fingerprint FROM sandbox_issues WHERE session_key=? AND issued_at=?',
                (intent.session_key,intent.issued_at.isoformat())).fetchone()
            if same_issue:
                saved = self.record(same_issue[0])
                body = saved.payload
                if (body['intent'] != json.loads(encode(intent)) or body['parent_fingerprint'] != parent
                        or body['revision_reason'] != revision_reason):
                    raise ValueError('Publication retry changed immutable issuance')
                for h in history:
                    old = current_seed.get(h.key)
                    if old is None or old != encode(h):
                        raise ValueError('Publication retry changed seed evidence; issue an explicit revision')
                return saved
            if self.db.execute('SELECT 1 FROM sandbox_completions WHERE session_key=?',
                               (intent.session_key,)).fetchone():
                raise ValueError('Completed session cannot be republished')
            previous = [self.record(r[0]) for r in self.db.execute(
                'SELECT fingerprint FROM sandbox_issues WHERE session_key=?', (intent.session_key,))]
            newest = max(previous, key=lambda r: datetime.fromisoformat(r.payload['intent']['issued_at']), default=None)
            if (newest.fingerprint if newest else None) != parent:
                raise ValueError('Explicit latest predecessor required; no silent refresh or revision branch')
            if len({h.key for h in history}) != len(history):
                raise ValueError('Duplicate history identity')
            for h in history:
                saved = current_seed.get(h.key)
                if saved and saved != encode(h):
                    raise ValueError('Conflicting seed history; corrections require explicit reconciliation')
                self.db.execute('INSERT OR IGNORE INTO sandbox_seed_history VALUES (?,?)', (h.key,encode(h)))
            evidence = {}
            for serialized in phase_records(self,intent.issued_at).values():
                h = json.loads(serialized)
                evidence[h['key']] = PhaseHistory(h['key'],datetime.fromisoformat(h['start']),
                    datetime.fromisoformat(h['end']),datetime.fromisoformat(h['available_at']),h['accounting_key'],
                    tuple(PhaseDose(**p) for p in h['phases']))
            for h in self.completed_history(intent.issued_at):
                if h.key in evidence and evidence[h.key] != h:
                    raise ValueError('Conflicting completed evidence')
                if any(old.key != h.key and old.start == h.start and old.end == h.end for old in evidence.values()):
                    raise ValueError('Same session under different history identities; explicit reconciliation required')
                evidence[h.key] = h
            record = issue_execution(intent, evidence.values(), parent=newest, revision_reason=revision_reason)
            self.db.execute('INSERT INTO sandbox_issues VALUES (?,?,?,?)',
                (record.fingerprint, intent.session_key, intent.issued_at.isoformat(), record.serialized))
        return record

    def publish_plan(self,requests,history,event,snapshot,*,commit_stage=None):
        """Atomic current calendar + immutable issues + exact decision snapshot.

        Snapshot/JSON files are audit exports, not transaction owners. A crash
        after commit is recoverable from this journal; a failed slot rolls the
        whole batch back. Stale concurrent decisions must be re-evaluated.
        """
        requests=tuple(requests)
        if sha256(snapshot.encode()).hexdigest()!=event['snapshot_fingerprint']:
            raise ValueError('Exact scored snapshot required for publication')
        with self.db:
            self.db.execute('BEGIN IMMEDIATE')
            latest=self.db.execute('SELECT sequence FROM sandbox_decisions ORDER BY sequence DESC LIMIT 1').fetchone()
            expected=latest[0]+1 if latest else 0
            if event['index']!=expected:
                raise ValueError('Stale decision or legacy archive requires explicit reconciliation')
            active={}
            for day,key,intent,parent in requests:
                record=self.publish(intent,history,parent=parent,
                    revision_reason='sandbox search republished this date using current evidence' if parent else None,
                    _in_batch=True)
                if (abs(record.payload['target_forecast']['predicted_load']-event['forecast_load_by_dose'][key])>1e-9
                        or record.payload['planning_handoff']!=event['forecast_handoff_by_dose'][key]):
                    raise ValueError('Publication differs from the searched physical forecast')
                active[day]=dict(dose=key,fingerprint=record.fingerprint,session_key=intent.session_key)
            event=dict(event,active=active)
            if 'realized_reference_allocation' in event:
                block=event['realized_reference_allocation']
                refs=[]
                for row in block['allocations']:
                    day=row['opportunity'].removeprefix('slot:')
                    if day not in active:
                        raise ValueError('Reference allocation is not an issued hard prescription')
                    if datetime.fromisoformat(row['available_at'])!=datetime.fromisoformat(event['as_of']):
                        raise ValueError('Reference must be frozen at its original issuance')
                    refs.append(dict(row,issued_fingerprint=active[day]['fingerprint']))
                if len(refs)!=len(active):
                    raise ValueError('Every issued hard prescription needs its original reference')
                event['realized_reference_allocation']=dict(block,allocations=refs)
            if commit_stage is not None:
                acceptance=commit_stage()
                if not acceptance.get('accepted'):
                    raise ValueError('Stage proposal did not pass acceptance; no calendar published')
                event['stage_acceptance']=acceptance['acceptance']
            self.db.execute('INSERT INTO sandbox_decisions VALUES (?,?,?)',(event['index'],encode(event),snapshot))
        return event

    def decisions(self):
        return tuple((json.loads(event),snapshot) for event,snapshot in self.serialized_decisions())

    def serialized_decisions(self):
        """Read committed bytes afresh; callers own decoding and validation."""
        return tuple((r[0],r[1]) for r in self.db.execute(
            'SELECT event,snapshot FROM sandbox_decisions ORDER BY sequence'))

    def prepare_upload(self,payload):
        """Validate and parse an owned upload without importing or consuming it."""
        validate_upload(payload)
        if payload.filename.casefold().endswith('.zip'):
            members=fit_members(payload.content)
            if len(members)!=1:
                raise ValueError('Sandbox requires exactly one FIT per associated upload')
            payload=UploadPayload(*members[0])
            validate_upload(payload)
        path=_persist_upload(self.root,payload)
        parsed=parser_for(path)(path,default_timezone=self.config['timezone_default'])
        if len(parsed.activities)!=1:
            raise ValueError('Sandbox requires exactly one activity per association')
        activity=parsed.activities[0]
        if activity.start_time_utc is None or activity.sport.casefold() not in {'running','run'}:
            raise ValueError('Timestamped running activity required')
        return path,activity

    def upload(self, payload, *, fingerprint, available_at, observed_phases=None,
               phase_evidence='inferred', readiness_changes=(), observed_role='unknown',
               resolve_evidence=False,long_threshold_miles=None,correction=None):
        """Use original TCX/FIT/ZIP bytes. Phases are external evidence, not plan copies.

        If the caller has no observed alignment, leave it unknown. Never infer
        work solely from an assigned quality title or force timing to the target.
        Same-content retries return the original event and original availability.
        Different recordings of the same activity require explicit reconciliation.
        """
        aware(available_at)
        readiness_changes = tuple(readiness_changes)
        if correction is not None and (set(correction)!={'expected_physical_fingerprint','reason'}
                or not all(isinstance(v,str) and v.strip() for v in correction.values())):
            raise ValueError('Correction requires exact prior physical identity and a reason')
        record = self.record(fingerprint) if fingerprint is not None else None
        path,activity=self.prepare_upload(payload)
        intervals = classify_movement(activity.trackpoints, self.config['moving_time']).intervals
        hr = HeartRateReference(self.config['resting_hr'], self.config['max_hr'], 'sandbox frozen settings')
        trace = observed_cardio_trace(activity, intervals, hr, self.reference.policy,
                                     include_spans=True, compute_density=False)
        end = activity.start_time_utc + timedelta(seconds=trace['elapsed_seconds'])
        bounds = trace['load_bounds_within_reference']
        if observed_role not in {'easy', 'long', 'quality', 'unknown'}:
            raise ValueError('Resolved observed role or explicit unknown required')
        if bounds is None:
            raise ValueError('Unsupported HR reference; cannot manufacture bounded physical load')
        # Importer owns cross-file/format deduplication, in this isolated DB only.
        summary = import_files(self.db, self.root, default_timezone=self.config['timezone_default'], paths=[path])
        if summary.failed_files:
            raise ValueError('Sandbox activity import failed')
        row = self.db.execute('''SELECT a.activity_id FROM activity_sources a
            JOIN source_files s ON s.id=a.source_file_id WHERE s.path=?''', (str(path.resolve()),)).fetchone()
        if row is None:
            raise ValueError('Imported activity association missing')
        activity_id = row[0]
        resolved=None
        if resolve_evidence:
            if observed_phases is not None or observed_role!='unknown' or long_threshold_miles is None:
                raise ValueError('Automatic original evidence cannot be combined with supplied role/phase labels')
            from run_analysis.uploaded_evidence import resolve_original_upload
            resolved=resolve_original_upload(self.db,activity_id,activity,self.config,available_at=available_at,
                intervals=intervals,trace=trace,long_threshold_miles=long_threshold_miles)
            observed_role,observed_phases,phase_evidence=resolved['role'],resolved['phases'],resolved['phase_evidence']
        session_key=record.payload['intent']['session_key'] if record else f'unplanned:{activity_id}'
        fingerprint=record.fingerprint if record else session_key
        # A copied seed can contain IDs from another sandbox database. Preserve
        # existing completion identities, but never reuse a seeded physical key
        # for a different recording just because SQLite starts its IDs at one.
        previous = self.db.execute('SELECT actual FROM sandbox_completions WHERE activity_id=?',
                                   (activity_id,)).fetchone()
        physical_key = json.loads(previous[0])['key'] if previous else f'sandbox:{activity_id}'
        if previous is None and self.db.execute('SELECT 1 FROM sandbox_seed_history WHERE key=?',
                                               (physical_key,)).fetchone():
            physical_key = 'sandbox:recording:'+sha256_file(path)
        actual = CompletedExecution(physical_key, session_key,
            activity.start_time_utc, end, available_at, observed_phases,
            activity.total_distance_m / 1609.344 if activity.total_distance_m is not None else None,
            Quantity(bounds['low'], bounds['high'], trace['total_load'], 'original continuous HR accounting'),
            self.reference.accounting_key, self.context, phase_evidence)
        if record is not None:
            comparison = compare_execution(record, actual, as_of=available_at, readiness_changes=readiness_changes)
            comparison['association_kind']='issued'
        else:
            if actual.end>available_at:
                raise ValueError('Completed unplanned work must be available after its end')
            comparison=dict(association_kind='unplanned',issued_fingerprint=None,actual_key=actual.key,
                distance_adherence='not_prescribed',structure='not_prescribed',phase_adherence=[],
                actual_cardio=dict(low=actual.cardio.low,high=actual.cardio.high,estimate=actual.cardio.estimate),
                actual_distance_miles=actual.distance_miles,actual_elapsed_seconds=(actual.end-actual.start).total_seconds(),
                target_load=None,execution_conditioned_load=None,response_status='no_preissued_forecast',
                readiness_changes=list(readiness_changes),review_signals=['unplanned_work'],
                routine_variation_without_new_replan_signal=False,calendar_action='none')
        if correction is not None:
            comparison['correction']=correction
        if resolved:
            comparison['observed_evidence']=resolved['provenance']
        history = PhaseHistory(actual.key, actual.start, actual.end, actual.available_at,
            actual.accounting_key, phase_observations(layout(observed_phases), trace['normalized_spans'])
            if observed_phases else ())
        exposure = observed_profile(activity, self.config)['running_exposure']
        lo, hi = exposure['moving_seconds_range']
        physical = accounted_phases(actual.key,actual.start,actual.distance_miles,observed_phases,
            actual.cardio,observed_role,self.reference,source='observed',available=available_at,
            moving=Quantity(lo/60,hi/60,basis='original parsed movement, not phase elapsed'),
            elapsed_seconds=(actual.end-actual.start).total_seconds(),explicit=phase_evidence=='explicit')
        if resolved and resolved['dose'].bouts:
            from run_analysis.continuous_training_dose import general_quality_credit
            bouts=resolved['dose'].bouts
            work=Quantity.point(sum(b.work_seconds for b in bouts)/60,resolved['dose'].structure_basis)
            physical=replace(physical,bouts=bouts,work_evidence_basis=resolved['dose'].structure_basis,
                demand=(*physical.demand[:2],work),quality=general_quality_credit(work,role=observed_role,reference=self.reference))
        elif physical.bouts and observed_phases is not None:
            distances=measured_work_distances(intervals,actual.start,observed_phases)
            if len(distances)!=len(physical.bouts):raise ValueError('Observed work-distance alignment differs')
            physical=replace(physical,bouts=tuple(replace(b,distance_meters=d)
                for b,d in zip(physical.bouts,distances)))
        physical_json = encode(physical)
        digest = sha256_file(path)
        with self.db:
            self.db.execute('BEGIN IMMEDIATE')
            old = self.db.execute('SELECT * FROM sandbox_completions WHERE activity_id=?', (activity_id,)).fetchone()
            if old:
                if correction is not None and (old['accounted'] is None or
                        old['physical_fingerprint']!=sha256(old['accounted'].encode()).hexdigest()):
                    raise ValueError('Corrupt prior evidence requires migration, not correction')
                old_actual = json.loads(old['actual'])
                retry_actual = json.loads(encode(actual))
                retry_actual['available_at'] = old_actual['available_at']
                retry_physical = json.loads(physical_json)
                retry_physical['available_at'] = old_actual['available_at']
                changed=(old['fingerprint'] != fingerprint or old['source_digest'] != digest
                        or retry_actual != old_actual
                        or old['accounted'] is None or retry_physical != json.loads(old['accounted'])
                        or old['physical_fingerprint'] != sha256(old['accounted'].encode()).hexdigest()
                        or json.loads(old['comparison'])['readiness_changes'] != list(readiness_changes))
                if changed:
                    if (correction is None or old['fingerprint']!=fingerprint or old['session_key']!=actual.session_key
                            or old['physical_fingerprint']!=correction['expected_physical_fingerprint']
                            or available_at<=datetime.fromisoformat(old['available_at'])):
                        raise ValueError('Already consumed activity; correction/reassociation must be explicit and current')
                    self.db.execute('INSERT INTO sandbox_completion_versions VALUES (?,?,?,?,?)',
                        (activity_id,old['physical_fingerprint'],available_at.isoformat(),correction['reason'],encode(dict(old))))
                    self.db.execute('''UPDATE sandbox_completions SET source_digest=?,available_at=?,actual=?,comparison=?,
                        phase_history=?,accounted=?,physical_fingerprint=? WHERE activity_id=?''',
                        (digest,available_at.isoformat(),encode(actual),encode(comparison),encode(history),physical_json,
                         sha256(physical_json.encode()).hexdigest(),activity_id))
                    return dict(status='corrected',comparison=comparison)
                if correction is not None and json.loads(old['comparison']).get('correction')!=correction:
                    raise ValueError('Correction retry changed its immutable reason or predecessor')
                if available_at < datetime.fromisoformat(old['available_at']):
                    raise ValueError('Retry cannot backdate evidence availability')
                return dict(status='duplicate', comparison=json.loads(old['comparison']))
            if correction is not None:
                raise ValueError('Correction must resolve to an already consumed canonical activity')
            if self.db.execute('SELECT 1 FROM sandbox_completions WHERE session_key=?',
                               (actual.session_key,)).fetchone():
                raise ValueError('Session already consumed by another activity')
            self.db.execute('INSERT INTO sandbox_completions VALUES (?,?,?,?,?,?,?,?,?,?)',
                (activity_id, actual.session_key, fingerprint, digest, available_at.isoformat(),
                 encode(actual), encode(comparison), encode(history), physical_json,
                 sha256(physical_json.encode()).hexdigest()))
        return dict(status='consumed', comparison=comparison)

    def state(self, as_of):
        """Point-in-time handoff: one actual per completion, only unconsumed forecasts.

        These are distinct channels, never summed as if an expectation were an
        observation. No decay, readiness verdict, calendar edit or fitness gain.
        """
        aware(as_of)
        completed = [(json.loads(r['actual']),json.loads(r['comparison'])) for r in self.completion_rows(as_of)]
        actuals = [a for a,_ in completed]
        consumed = {a['session_key'] for a in actuals}
        latest = {}
        for row in self.db.execute('SELECT fingerprint FROM sandbox_issues'):
            r = self.record(row[0])
            intent = r.payload['intent']
            issue = datetime.fromisoformat(intent['issued_at'])
            key = intent['session_key']
            if issue <= as_of and key not in consumed and (key not in latest or issue > latest[key][0]):
                latest[key] = (issue, r)
        known_plans=[event for event,_ in self.decisions() if datetime.fromisoformat(event['as_of'])<=as_of]
        retired=[]
        if known_plans:
            selected={value['session_key']:value['fingerprint'] for value in known_plans[-1]['active'].values()}
            retired=[key for key in latest if key not in selected]
            latest={key:value for key,value in latest.items() if key in selected}
            if any(value[1].fingerprint!=selected[key] for key,value in latest.items()):
                raise ValueError('Current calendar and latest issuance disagree')
        return dict(actual=sorted(actuals, key=lambda a: (a['start'], a['key'])),
            comparisons=sorted((c for _,c in completed),key=lambda c:c['actual_key']),
            pending=[latest[k][1].payload for k in sorted(latest)],
            pending_scope='committed_current_calendar' if known_plans else 'latest_unfulfilled_issuance_not_current_selected_calendar',
            retired_session_keys=retired,
            observed_cardio_bounds=[sum(a['cardio'][edge] for a in actuals) for edge in ('low', 'high')],
            live_scheduler_influence=False)
