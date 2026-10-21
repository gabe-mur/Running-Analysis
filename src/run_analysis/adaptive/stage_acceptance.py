"""Offline persistent stage acceptance through the shared evaluator, not flags."""
from dataclasses import asdict
from datetime import datetime
from hashlib import sha256
import json
import sqlite3
from math import isfinite
from contextlib import nullcontext

from run_analysis.staged_progression import DevelopmentStage, ProgressionPolicy
from run_analysis.adaptive.decision_snapshot import evaluate_snapshot, serialize


def encode(value):
    return json.dumps(value,default=lambda v:asdict(v) if hasattr(v,'__dataclass_fields__') else v.isoformat(),
                      sort_keys=True,allow_nan=False)


def read_stage(value):
    v=json.loads(value)
    return DevelopmentStage(**dict(v,issued_at=datetime.fromisoformat(v['issued_at']),policy=ProgressionPolicy(**v['policy'])))


class StageLedger:
    """Dedicated offline database; no application database may be opened here."""
    def __init__(self,path,*,seed=None,_sandbox=None):
        self.owns_connection=_sandbox is None
        self.db=sqlite3.connect(path) if self.owns_connection else _sandbox.db
        if self.db.in_transaction:
            raise ValueError('Initialize stage storage outside an active transaction')
        tables={r[0] for r in self.db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        shared=(_sandbox is not None and (_sandbox.root/'execution-sandbox.json').exists()
            and {'sandbox_issues','sandbox_decisions'}<=tables)
        if tables and 'stage_ledger_identity' not in tables and not shared:
            self.close()
            raise ValueError('Dedicated stage ledger required; refusing unrelated database')
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS stage_ledger_identity(version INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS stages(key TEXT PRIMARY KEY, parent TEXT, stage TEXT NOT NULL,
                acceptance TEXT UNIQUE NOT NULL, evidence TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS current_stage(singleton INTEGER PRIMARY KEY CHECK(singleton=1),key TEXT NOT NULL);
        ''')
        if not self.db.execute('SELECT 1 FROM stage_ledger_identity').fetchone():
            self.db.execute('INSERT INTO stage_ledger_identity VALUES (1)')
        self.db.commit()
        if self.current is None:
            if seed is None:
                self.close()
                raise ValueError('Empty ledger requires an explicit provisional seed')
            self._commit(seed,None,'seed:'+seed.key,encode(dict(status='provisional_seed_not_validated_capacity')))
        elif seed is not None and self.current!=seed:
            self.close()
            raise ValueError('Do not rebootstrap an accepted stage on refresh')

    @property
    def current(self):
        row=self.db.execute('SELECT s.stage FROM stages s JOIN current_stage c ON c.key=s.key').fetchone()
        return read_stage(row[0]) if row else None

    def close(self):
        if self.owns_connection:
            self.db.close()

    @classmethod
    def for_sandbox(cls,sandbox,*,seed=None):
        return cls(None,seed=seed,_sandbox=sandbox)

    def _commit(self,stage,parent,acceptance,evidence):
        nested=self.db.in_transaction
        with nullcontext() if nested else self.db:
            if not nested:
                self.db.execute('BEGIN IMMEDIATE')
            old=self.db.execute('SELECT stage,evidence FROM stages WHERE acceptance=?',(acceptance,)).fetchone()
            if old:
                if tuple(old)!=(encode(stage),evidence):
                    raise ValueError('Acceptance identity changed')
                return read_stage(old[0])
            if (self.current.key if self.current else None)!=parent:
                raise ValueError('Stale stage acceptance; review current state again')
            self.db.execute('INSERT INTO stages VALUES (?,?,?,?,?)',(stage.key,parent,encode(stage),acceptance,evidence))
            self.db.execute('INSERT INTO current_stage VALUES (1,?) ON CONFLICT(singleton) DO UPDATE SET key=excluded.key',(stage.key,))
        return stage


def accept_candidate(ledger,review,state,planned,contract_keys,*,reference,selection=None):
    """Re-evaluate the selected physical program under its delivery contract.

    A proposal cannot be accepted just because one long run was completed. This
    gate is for a growth transition, not a hard funding constraint on every replan.
    """
    stage=review.get('candidate')
    amendment=review.get('status')=='conditional_rate_amendment'
    maintenance=review.get('status')=='conditional_maintenance_candidate'
    if review.get('status') not in {'conditional_development_candidate','conditional_rate_amendment','conditional_maintenance_candidate'} or stage is None:
        return dict(accepted=False,reason='no_growth_proposal')
    if amendment:
        from .volume_rate_amendment import propose
        expected=propose(ledger.current,review['original_targets'],state.as_of,state.timezone)
        if expected!=review:
            raise ValueError('Rate revision must reproduce the exact prospective amendment')
    if maintenance:
        from .missed_checkpoint import propose
        expected=propose(ledger.current,review['original_targets'],state.as_of,state.timezone,
            review['review_hour'],review['execution_gap_miles'],race_goal=state.context.race_goal,
            race_event=state.context.race_event,race_goal_history=state.context.race_goal_history,
            race_event_history=state.context.race_event_history,
            unconfirmed_opportunities=review['unconfirmed_opportunities'])
        if expected!=review:
            raise ValueError('Maintenance must reproduce the unconfirmed checkpoint interpretation')
        from zoneinfo import ZoneInfo
        zone=ZoneInfo(state.timezone)
        for fingerprint,day in review['unconfirmed_opportunities']:
            events=[json.loads(r[0]) for r in ledger.db.execute('SELECT event FROM sandbox_decisions')]
            known=[e for e in events if datetime.fromisoformat(e['as_of'])<=state.as_of
                and datetime.fromisoformat(e['as_of']).astimezone(zone).date().isoformat()<=day]
            latest=max(known,key=lambda e:datetime.fromisoformat(e['as_of'])) if known else None
            active=latest['active'].get(day) if latest else None
            completed=any(datetime.fromisoformat(r[0])<=state.as_of for r in ledger.db.execute(
                'SELECT available_at FROM sandbox_completions WHERE session_key=?',('slot:'+day,)))
            if not active or active['fingerprint']!=fingerprint or completed:
                raise ValueError('Unconfirmed opportunity differs from the original issued/completion journal')
    policy=state.context.policy
    if (state.stage_key!=stage.key or state.stage_issued_at!=stage.issued_at or not amendment and state.as_of!=stage.issued_at
            or (policy.weekly_miles,policy.ordinary_miles,policy.long_miles)!=
               (stage.weekly_equivalent_miles,stage.ordinary_miles,stage.long_miles)):
        raise ValueError('Candidate program must be evaluated against its exact proposed stage')
    result=evaluate_snapshot(state,tuple(planned),tuple(contract_keys),reference=reference)
    parts=result['evaluation'].parts
    funding=parts['easy_duration']['supporting_easy_volume']
    sufficient=all(parts['delivery'][p]+1e-8>=parts['targets'][p] for p in ('volume','long','quality'))
    sufficient=sufficient and funding is not None and funding['delivered_miles']+1e-8>=funding['target_miles']
    sustained=state.context.delivery_contract=='sustained_v1'
    if sustained:
        selected=selection['search']['selected'] if selection and 'search' in selection else {}
        if (not selection or selection.get('requires_stage_funding') is not False
                or selection['state'].fingerprint!=state.fingerprint
                or tuple(selection['planned'])!=tuple(planned)
                or tuple(selection['contract_keys'])!=tuple(contract_keys)
                or selected.get('score_fingerprint')!=result['evaluation'].fingerprint
                or selected.get('eligible') is not True
                or not isfinite(selected.get('cost',float('inf')))
                or abs(selected.get('cost',float('inf'))-result['evaluation'].total_cost)>1e-12):
            raise ValueError('Growth requires revalidated ordinary-objective selection')
    if not result['decision_eligible'] or not sustained and not sufficient:
        return dict(accepted=False,reason='candidate_requires_alternative' if not result['decision_eligible'] else 'candidate_stage_underfunded',
                    evaluation=result)
    if sustained and not planned:
        return dict(accepted=False,reason='no_planned_training',evaluation=result)
    evidence=encode(dict(snapshot=json.loads(serialize(state)),score_fingerprint=result['evaluation'].fingerprint,
                         review=review,planned=planned,contracts=contract_keys,
                         delivery_contract=state.context.delivery_contract,
                         target_is_delivery_guarantee=False,
                         aggregate_funding_diagnostic=sufficient))
    identity=sha256(evidence.encode()).hexdigest()
    accepted=ledger._commit(stage,review['stage_key'],identity,evidence)
    return dict(accepted=True,stage=accepted,acceptance=identity,evaluation=result,
                delivery_contract=state.context.delivery_contract,
                aggregate_funding_diagnostic=sufficient,
                target_is_delivery_guarantee=False,
                delivery_confirmed=not amendment and not maintenance,
                physiological_recovery_established=False)
